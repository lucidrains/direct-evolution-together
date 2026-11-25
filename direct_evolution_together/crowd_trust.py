from __future__ import annotations

import base64
from dataclasses import dataclass, field
import json
import math
import statistics
import threading
import time
from typing import Iterable, Sequence

from django.db import OperationalError, transaction
from django.utils import timezone

from direct_evolution_together.config import TrustPolicyConfig, get_config
from direct_evolution_together.models import ContributorTrust, ExternalEvalResult

MAX_WORKER_POSTER_B64 = 350_000
MAX_ATTESTATION_BATCH = 1024
MAX_SHARD_CLAIM_BATCH = 256
SQLITE_SYNC_LOCK = threading.RLock()


def _loads_safe(raw: str, expected: type) -> object:
    try:
        val = json.loads(raw)
        return val if isinstance(val, expected) else expected()
    except Exception:
        return expected()


def sanitize_summary(raw_sum: dict) -> dict[str, object]:
    try:
        pval = float(raw_sum.get("primary_value", 0.0))
    except (TypeError, ValueError):
        pval = 0.0
    if not math.isfinite(pval) or abs(pval) >= 1e12:
        pval = 0.0

    tier = str(raw_sum.get("tier") or "")
    tier = tier if tier in ("", "mid", "solved") else ""
    badge = str(raw_sum.get("badge_text") or f"{pval:.2f}")[:24]
    out: dict[str, object] = {
        "primary_value": round(pval, 4),
        "primary_display": str(raw_sum.get("primary_display") or badge)[:32],
        "badge_text": badge,
        "tier": tier,
        "secondary_text": str(raw_sum.get("secondary_text") or "")[:120],
        "solved": bool(raw_sum.get("solved", tier == "solved")),
    }
    extra_kept = 0
    for key, val in raw_sum.items():
        if key in out or extra_kept >= 16:
            continue
        if not isinstance(key, str) or not (1 <= len(key) <= 24) or not key.replace("_", "").isalnum():
            continue
        if isinstance(val, bool):
            continue
        if isinstance(val, (int, float)) and math.isfinite(float(val)) and abs(float(val)) < 1e12:
            out[key] = round(float(val), 4)
            extra_kept += 1
    return out


def sanitize_traj_sample(raw_traj: object) -> list[list[float]] | None:
    """Validate optional subsampled behavioral trajectory matrix (<= 32 steps x 32 dims) for PopuLoRA AURORA."""
    if not isinstance(raw_traj, list) or not raw_traj or len(raw_traj) > 32:
        return None
    first = raw_traj[0]
    if not isinstance(first, list) or not (2 <= len(first) <= 32):
        return None
    dim = len(first)
    out: list[list[float]] = []
    for row in raw_traj:
        if not isinstance(row, list) or len(row) != dim:
            return None
        clean_row: list[float] = []
        for x in row:
            if not isinstance(x, (int, float)) or not math.isfinite(float(x)) or abs(float(x)) > 1e6:
                return None
            clean_row.append(round(float(x), 4))
        out.append(clean_row)
    return out


MAX_REPLAY_KEYFRAMES = 32


def sanitize_replay(raw_replay: object) -> dict[str, object] | None:
    """Validate optional compact subsampled pose replay for peer 3D card/fullscreen playback."""
    if not isinstance(raw_replay, dict):
        return None
    body_poses = raw_replay.get("body_poses")
    if not isinstance(body_poses, list) or not body_poses or len(body_poses) > 220:
        return None
    row_len = len(body_poses[0]) if isinstance(body_poses[0], list) else 0
    if not (7 <= row_len <= 160):
        return None
    orig_len = len(body_poses)
    if orig_len > MAX_REPLAY_KEYFRAMES:
        indices = [
            round(k * (orig_len - 1) / (MAX_REPLAY_KEYFRAMES - 1))
            for k in range(MAX_REPLAY_KEYFRAMES)
        ]
        sampled_rows = [body_poses[i] for i in indices]
    else:
        sampled_rows = body_poses

    clean_poses: list[list[float]] = []
    for row in sampled_rows:
        if not isinstance(row, list) or len(row) != row_len:
            return None
        clean_row: list[float] = []
        for v in row:
            if not isinstance(v, (int, float)) or not math.isfinite(float(v)) or abs(float(v)) > 1e4:
                return None
            clean_row.append(round(float(v), 3))
        clean_poses.append(clean_row)
    out: dict[str, object] = {"body_poses": clean_poses}
    raw_stride = raw_replay.get("pose_stride")
    base_stride = int(raw_stride) if isinstance(raw_stride, int) and 1 <= raw_stride <= 64 else 1
    if orig_len > len(clean_poses) and len(clean_poses) > 1:
        eff_stride = max(1, min(64, round(base_stride * (orig_len - 1) / (len(clean_poses) - 1))))
    else:
        eff_stride = base_stride
    out["pose_stride"] = eff_stride
    vxt = raw_replay.get("vx_trace")
    if isinstance(vxt, list) and vxt and len(vxt) <= 450:
        clean_vxt = [
            round(float(x), 3)
            for x in vxt
            if isinstance(x, (int, float)) and math.isfinite(float(x))
        ]
        if len(clean_vxt) > 48:
            clean_vxt = [clean_vxt[round(k * (len(clean_vxt) - 1) / 47)] for k in range(48)]
        out["vx_trace"] = clean_vxt
    return out


def _safe_finite_float(raw: object, fallback: float = 0.0) -> float:
    try:
        val = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback
    return val if math.isfinite(val) else fallback


def sanitize_topology(raw_topo: object) -> dict | None:
    if not isinstance(raw_topo, dict) or not isinstance(raw_topo.get("nodes"), list):
        return None
    raw_edges = raw_topo.get("edges") or raw_topo.get("connections") or []
    if len(raw_topo["nodes"]) > 128 or not isinstance(raw_edges, list) or len(raw_edges) > 512:
        return None
    return {
        "nodes": [
            {
                "id": str(n.get("id", i))[:24],
                "kind": str(n.get("kind") or n.get("type") or "hidden")[:16],
                "label": str(n.get("label") or "")[:24] if n.get("label") else None,
                "activation": str(n.get("activation") or "tanh")[:16],
                "bias": _safe_finite_float(n.get("bias", 0.0)),
            }
            for i, n in enumerate(raw_topo["nodes"])
            if isinstance(n, dict)
        ],
        "edges": [
            {
                "from": str(e.get("from", e.get("in", "")))[:24],
                "to": str(e.get("to", e.get("out", "")))[:24],
                "weight": _safe_finite_float(e.get("weight", 0.0)),
                "enabled": bool(e.get("enabled", True)),
            }
            for e in raw_edges
            if isinstance(e, dict)
        ],
    }


def values_agree(a: float, b: float, trust_cfg: TrustPolicyConfig) -> bool:
    return abs(a - b) <= max(trust_cfg.tolerance_abs, trust_cfg.tolerance_rel * max(abs(a), abs(b)))


def _find_best_cluster(attestations: list[dict], trust_cfg: TrustPolicyConfig) -> list[dict]:
    best: list[dict] = []
    best_weight = -1.0
    for anchor in attestations:
        a_val = float(anchor.get("val", 0.0))
        cluster = [att for att in attestations if values_agree(float(att.get("val", 0.0)), a_val, trust_cfg)]
        total_w = sum(float(att.get("weight", 1.0)) for att in cluster)
        if len(cluster) > len(best) or (len(cluster) == len(best) and total_w > best_weight):
            best, best_weight = cluster, total_w
    return best


def _aggregate_cluster(cluster: list[dict], trust_cfg: TrustPolicyConfig) -> dict:
    if trust_cfg.aggregation == "weighted_mean":
        w_sum = sum(max(0.05, float(c.get("weight", 1.0))) for c in cluster)
        target = sum(float(c.get("val", 0.0)) * max(0.05, float(c.get("weight", 1.0))) for c in cluster) / w_sum
    else:
        target = statistics.median(float(c.get("val", 0.0)) for c in cluster)

    rep_sum = dict(min(cluster, key=lambda c: abs(float(c.get("val", 0.0)) - target)).get("summary") or {})
    prev_primary = float(rep_sum.get("primary_value", target))
    rep_sum["primary_value"] = round(target, 4)
    if "return" in rep_sum and abs(float(rep_sum["return"]) - prev_primary) <= 1e-2:
        rep_sum["return"] = round(target, 2)
        if "mean_distance" in rep_sum:
            mdx = float(rep_sum["mean_distance"])
            disp = f"{target:+.0f}"
            rep_sum["primary_display"] = f"{disp} ({mdx:+.2f}m)"
            rep_sum["badge_text"] = f"{disp} \u00b7 {mdx:+.2f}m"
    elif "mean_distance" in rep_sum:
        rep_sum["mean_distance"] = round(target, 2)
        rep_sum["primary_display"] = f"{target:+.1f}m"
        rep_sum["badge_text"] = f"{target:+.1f}m"
    elif "mean_steps" in rep_sum:
        rep_sum["mean_steps"] = round(target, 1)
    return rep_sum


def _queue_reputation_delta(
    rep_deltas: dict[str, list[bool]],
    cids: Iterable[str],
    agreed: bool,
) -> None:
    for cid in cids:
        if not cid or cid.startswith("worker:"):
            continue
        rep_deltas.setdefault(cid, []).append(agreed)


def _flush_reputation_deltas(
    room: str,
    rep_deltas: dict[str, list[bool]],
    trust_cfg: TrustPolicyConfig,
    seed_contrib: ContributorTrust | None = None,
    submission_count: int = 0,
) -> None:
    target_cids = set(rep_deltas.keys())
    if seed_contrib is not None and submission_count > 0:
        target_cids.add(seed_contrib.contributor_id)
    if not target_cids:
        return

    by_cid: dict[str, ContributorTrust] = {}
    if seed_contrib is not None:
        by_cid[seed_contrib.contributor_id] = seed_contrib

    missing_lookup = [c for c in target_cids if c not in by_cid]
    if missing_lookup:
        for obj in ContributorTrust.objects.filter(room=room, contributor_id__in=missing_lookup):
            by_cid[obj.contributor_id] = obj

    to_create = [
        ContributorTrust(room=room, contributor_id=cid, reputation=trust_cfg.initial_reputation)
        for cid in target_cids
        if cid not in by_cid
    ]
    if to_create:
        ContributorTrust.objects.bulk_create(to_create, ignore_conflicts=True)
        for obj in ContributorTrust.objects.filter(room=room, contributor_id__in=[c.contributor_id for c in to_create]):
            by_cid[obj.contributor_id] = obj

    now = timezone.now()
    dirty: list[ContributorTrust] = []
    for cid in target_cids:
        obj = by_cid.get(cid)
        if obj is None:
            continue
        if seed_contrib is not None and cid == seed_contrib.contributor_id and submission_count > 0:
            obj.submissions += submission_count
        for agreed in rep_deltas.get(cid, ()):
            delta = trust_cfg.agree_reward if agreed else -trust_cfg.mismatch_penalty
            new_rep = max(0.0, min(1.0, float(obj.reputation) + delta))
            obj.reputation = round(new_rep, 4)
            obj.agreements += int(agreed)
            obj.mismatches += int(not agreed)
            obj.quarantined = new_rep < trust_cfg.min_trusted_reputation
        obj.updated_at = now
        dirty.append(obj)

    if dirty:
        ContributorTrust.objects.bulk_update(
            dirty,
            ["submissions", "reputation", "agreements", "mismatches", "quarantined", "updated_at"],
        )


def _ingest_authority(
    raw_item: dict,
    gh: str,
    clean_sum: dict[str, object],
    existing_payload: dict,
    contributor_id: str,
) -> tuple[str, list[dict], dict[str, object]]:
    pval = float(clean_sum["primary_value"])  # type: ignore[arg-type]
    atts = [{"cid": contributor_id, "val": pval, "weight": 1.0, "summary": clean_sum, "authority": True}]
    entry: dict[str, object] = {
        "hash": gh,
        "summary": clean_sum,
        "trust": {"status": "verified", "attestations": 1, "quorum": 1, "authority": True},
    }
    topo = sanitize_topology(raw_item.get("topology"))
    if topo is not None:
        entry["topology"] = topo
    elif "topology" in existing_payload:
        entry["topology"] = existing_payload["topology"]

    poster_b64 = raw_item.get("poster_png_b64")
    if isinstance(poster_b64, str) and poster_b64 and len(poster_b64) <= MAX_WORKER_POSTER_B64:
        try:
            if base64.b64decode(poster_b64, validate=True).startswith(b"\x89PNG\r\n\x1a\n"):
                entry["poster_data_url"] = f"data:image/png;base64,{poster_b64}"
        except Exception:
            pass
    elif "poster_data_url" in existing_payload:
        entry["poster_data_url"] = existing_payload["poster_data_url"]
    traj = sanitize_traj_sample(raw_item.get("traj_sample"))
    if traj is not None:
        entry["traj_sample"] = traj
    rep = sanitize_replay(raw_item.get("replay"))
    if rep is not None:
        entry["replay"] = rep
    return "verified", atts, entry


def _ingest_peer(
    raw_item: dict,
    gh: str,
    clean_sum: dict[str, object],
    prev_status: str,
    existing_payload: dict,
    atts: list[dict],
    contrib: ContributorTrust,
    trust_cfg: TrustPolicyConfig,
    eff_quorum: int,
    rep_deltas: dict[str, list[bool]],
) -> tuple[str, list[dict], dict[str, object]]:
    pval = float(clean_sum["primary_value"])  # type: ignore[arg-type]
    already_idx = next((i for i, a in enumerate(atts) if a.get("cid") == contrib.contributor_id), None)
    att_entry = {
        "cid": contrib.contributor_id,
        "val": pval,
        "weight": float(contrib.reputation),
        "summary": clean_sum,
        "scored": False,
    }
    appended = False
    if already_idx is not None:
        if prev_status != "verified" and not atts[already_idx].get("scored"):
            atts[already_idx] = att_entry
    elif len(atts) < trust_cfg.max_attestations:
        atts.append(att_entry)
        appended = True

    cluster = _find_best_cluster(atts, trust_cfg)
    cluster_cids = {c.get("cid") for c in cluster}

    if len(cluster) >= eff_quorum:
        status = "verified"
        eff_summary = _aggregate_cluster(cluster, trust_cfg)
        if eff_quorum >= 2:
            newly_agreed = [c["cid"] for c in atts if c.get("cid") in cluster_cids and not c.get("scored")]
            newly_outlier = [c["cid"] for c in atts if c.get("cid") not in cluster_cids and not c.get("scored")]
            if newly_agreed:
                _queue_reputation_delta(rep_deltas, newly_agreed, True)
            if newly_outlier:
                _queue_reputation_delta(rep_deltas, newly_outlier, False)
            for c in atts:
                c["scored"] = True
            if already_idx is None and not appended and len(atts) >= trust_cfg.max_attestations:
                _queue_reputation_delta(
                    rep_deltas,
                    [contrib.contributor_id],
                    values_agree(pval, float(eff_summary["primary_value"]), trust_cfg),
                )
    elif prev_status == "verified" and existing_payload.get("summary"):
        status = "verified"
        eff_summary = existing_payload["summary"]
        if already_idx is None:
            matches = values_agree(pval, float(eff_summary.get("primary_value", 0.0)), trust_cfg)
            _queue_reputation_delta(rep_deltas, [contrib.contributor_id], matches)
            for a in atts:
                if a.get("cid") == contrib.contributor_id:
                    a["scored"] = True
    else:
        status = "disputed" if len(atts) >= 2 else "pending"
        eff_summary = existing_payload.get("summary") or clean_sum

    entry: dict[str, object] = {
        **existing_payload,
        "hash": gh,
        "summary": eff_summary,
        "trust": {
            "status": status,
            "attestations": len(cluster) if status == "verified" else len(atts),
            "quorum": eff_quorum,
            "authority": bool(existing_payload.get("trust", {}).get("authority")),
        },
    }
    if "traj_sample" not in entry or status == "verified":
        traj = sanitize_traj_sample(raw_item.get("traj_sample"))
        if traj is not None:
            entry["traj_sample"] = traj
    if "replay" not in entry:
        rep = sanitize_replay(raw_item.get("replay"))
        if rep is not None:
            entry["replay"] = rep
    return status, atts, entry


def resolve_effective_quorum(
    trust_cfg: TrustPolicyConfig,
    *,
    active_peers: int | None = None,
    sharded: bool = False,
) -> int:
    """Return effective attestation quorum given peer count and sharded population mode."""
    if sharded:
        return max(1, int(trust_cfg.shard_quorum))
    if active_peers is None:
        return int(trust_cfg.quorum)
    return 1 if active_peers <= 1 else min(int(trust_cfg.quorum), active_peers)


def submit_eval_attestations(
    *,
    room: str,
    goal: str,
    generation: int,
    results: list[dict],
    contributor_id: str,
    authority: bool = False,
    active_peers: int | None = None,
    sharded: bool = False,
    max_batch_size: int = MAX_ATTESTATION_BATCH,
    max_hash_len: int = 32,
    max_db_retries: int = 4,
    retry_backoff_sec: float = 0.02,
) -> dict[str, object]:
    trust_cfg = get_config().trust
    eff_quorum = resolve_effective_quorum(trust_cfg, active_peers=active_peers, sharded=sharded)
    stored_items: list[dict] = []
    changed_items: list[dict] = []
    verified_count = 0
    contrib: ContributorTrust | None = None

    parsed_batch: list[tuple[str, dict, dict[str, object]]] = []
    for raw_item in results[:max_batch_size]:
        if not isinstance(raw_item, dict):
            continue
        gh = str(raw_item.get("hash") or "").strip()[:max_hash_len]
        raw_sum = raw_item.get("summary")
        if not gh or not isinstance(raw_sum, dict):
            continue
        parsed_batch.append((gh, raw_item, sanitize_summary(raw_sum)))

    if not parsed_batch:
        return {
            "items": [],
            "changed_items": [],
            "contributor_reputation": 1.0 if authority else float(trust_cfg.initial_reputation),
            "quarantined": False,
            "verified_count": 0,
        }

    for attempt in range(max_db_retries):
        stored_items.clear()
        changed_items.clear()
        verified_count = 0
        try:
            with SQLITE_SYNC_LOCK, transaction.atomic():
                if not authority:
                    contrib, _ = ContributorTrust.objects.get_or_create(
                        room=room,
                        contributor_id=contributor_id,
                        defaults={"reputation": trust_cfg.initial_reputation},
                    )
                    if contrib.quarantined or contrib.reputation < trust_cfg.min_trusted_reputation:
                        return {
                            "items": [],
                            "changed_items": [],
                            "contributor_reputation": float(contrib.reputation),
                            "quarantined": True,
                            "verified_count": 0,
                        }

                unique_hashes = list({gh for gh, _, _ in parsed_batch})
                rows_by_hash: dict[str, ExternalEvalResult] = {
                    r.genome_hash: r
                    for r in ExternalEvalResult.objects.filter(
                        room=room,
                        goal=goal,
                        generation=generation,
                        genome_hash__in=unique_hashes,
                    )
                }

                rep_deltas: dict[str, list[bool]] = {}
                new_rows: dict[str, ExternalEvalResult] = {}
                dirty_rows: dict[str, ExternalEvalResult] = {}
                now = timezone.now()

                for gh, raw_item, clean_sum in parsed_batch:
                    row = rows_by_hash.get(gh)
                    existing_payload = _loads_safe(row.payload_json, dict) if row else {}
                    attestations = _loads_safe(row.attestations_json, list) if row else []
                    prev_status = row.status if row else ""

                    if authority:
                        status, attestations, entry = _ingest_authority(raw_item, gh, clean_sum, existing_payload, contributor_id)
                        did_change = True
                    else:
                        assert contrib is not None
                        eff_prev = prev_status or "pending"
                        status, attestations, entry = _ingest_peer(
                            raw_item, gh, clean_sum, eff_prev, existing_payload, attestations, contrib, trust_cfg, eff_quorum, rep_deltas
                        )
                        did_change = (
                            row is None
                            or status != prev_status
                            or entry.get("summary") != existing_payload.get("summary")
                            or (status != "verified" and entry.get("trust") != existing_payload.get("trust"))
                        )

                    atts_json = json.dumps(attestations)
                    payload_json = json.dumps(entry)
                    if row is None:
                        created = ExternalEvalResult(
                            room=room,
                            goal=goal,
                            generation=generation,
                            genome_hash=gh,
                            status=status,
                            attestations_json=atts_json,
                            payload_json=payload_json,
                        )
                        rows_by_hash[gh] = created
                        new_rows[gh] = created
                    else:
                        row.status = status
                        row.attestations_json = atts_json
                        row.payload_json = payload_json
                        row.updated_at = now
                        if gh not in new_rows:
                            dirty_rows[gh] = row

                    if status == "verified":
                        verified_count += 1
                    stored_items.append(entry)
                    if did_change:
                        changed_items.append(entry)

                if new_rows:
                    ExternalEvalResult.objects.bulk_create(list(new_rows.values()))
                if dirty_rows:
                    ExternalEvalResult.objects.bulk_update(
                        list(dirty_rows.values()),
                        ["status", "attestations_json", "payload_json", "updated_at"],
                    )
                if not authority and contrib is not None:
                    _flush_reputation_deltas(
                        room,
                        rep_deltas,
                        trust_cfg,
                        seed_contrib=contrib,
                        submission_count=len(parsed_batch),
                    )
            break
        except OperationalError as exc:
            if "locked" not in str(exc).lower() or attempt == max_db_retries - 1:
                raise
            time.sleep(retry_backoff_sec * (attempt + 1))

    rep_val, quarantined = 1.0, False
    if contrib is not None:
        rep_val = float(contrib.reputation)
        quarantined = bool(contrib.quarantined)

    if stored_items:
        _record_shard_eval_updates(room, goal, generation, stored_items)

    return {
        "items": stored_items,
        "changed_items": changed_items,
        "contributor_reputation": rep_val,
        "quarantined": quarantined,
        "verified_count": verified_count,
        "quorum": eff_quorum,
    }


def fetch_room_eval_map(room: str, goal: str, generation: int, *, trusted_only: bool = True) -> dict[str, dict]:
    items = fetch_room_evals(room, goal, generation, trusted_only=trusted_only)
    return {str(it["hash"]): it for it in items if isinstance(it, dict) and it.get("hash")}


_LEGACY_SUMMARY_KEYS = (
    ("primaryValue", "primary_value"),
    ("primaryDisplay", "primary_display"),
    ("badgeText", "badge_text"),
    ("secondaryText", "secondary_text"),
    ("meanDistance", "mean_distance"),
    ("minDistance", "min_distance"),
    ("meanSteps", "mean_steps"),
    ("minSteps", "min_steps"),
    ("meanVx", "mean_vx"),
    ("meanUpright", "mean_upright"),
)
_LEGACY_REPLAY_KEYS = (
    ("bodyPoses", "body_poses"),
    ("poseStride", "pose_stride"),
    ("vxTrace", "vx_trace"),
)


def _normalize_legacy_eval_payload(payload: dict) -> dict:
    """Upgrade any pre-snake_case persisted eval payload dict to canonical snake_case keys."""
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    if "trajSample" in out and "traj_sample" not in out:
        out["traj_sample"] = out.pop("trajSample")
    raw_sum = out.get("summary")
    if isinstance(raw_sum, dict) and any(k in raw_sum for k, _ in _LEGACY_SUMMARY_KEYS):
        clean_sum = dict(raw_sum)
        for old_k, new_k in _LEGACY_SUMMARY_KEYS:
            if old_k in clean_sum:
                v = clean_sum.pop(old_k)
                clean_sum.setdefault(new_k, v)
        out["summary"] = clean_sum
    raw_rep = out.get("replay")
    if isinstance(raw_rep, dict) and any(k in raw_rep for k, _ in _LEGACY_REPLAY_KEYS):
        clean_rep = dict(raw_rep)
        for old_k, new_k in _LEGACY_REPLAY_KEYS:
            if old_k in clean_rep:
                v = clean_rep.pop(old_k)
                clean_rep.setdefault(new_k, v)
        out["replay"] = clean_rep
    return out


def fetch_room_evals(
    room: str,
    goal: str,
    generation: int,
    *,
    trusted_only: bool = False,
    include_traj: bool = True,
    genome_hashes: Iterable[str] | None = None,
) -> list[dict]:
    hash_filter: set[str] | None = None
    if genome_hashes is not None:
        hash_filter = {str(h) for h in genome_hashes if h}
        if not hash_filter:
            return []
    with SQLITE_SYNC_LOCK:
        qs = ExternalEvalResult.objects.filter(room=room, goal=goal, generation=generation)
        if trusted_only:
            qs = qs.filter(status="verified")
        if hash_filter is not None:
            qs = qs.filter(genome_hash__in=hash_filter)
        rows = list(qs.order_by("id"))
    items: list[dict] = []
    for row in rows:
        payload = _loads_safe(row.payload_json, dict)
        if payload:
            payload = _normalize_legacy_eval_payload(payload)
            if not include_traj and "traj_sample" in payload:
                payload = {k: v for k, v in payload.items() if k != "traj_sample"}
            payload.setdefault("trust", {"status": row.status})
            items.append(payload)
    return items


@dataclass
class _GenEvalCache:
    verified: set[str] = field(default_factory=set)
    evals: dict[str, dict] = field(default_factory=dict)


_SHARD_LOCK = threading.Lock()
_SHARD_LEASES: dict[tuple[str, str, int], dict[str, tuple[str, float]]] = {}
_SHARD_CURSORS: dict[tuple[str, str, int], int] = {}
_GEN_EVAL_STATE: dict[tuple[str, str, int], _GenEvalCache] = {}


def reset_goal_shard_state(room: str, goal: str = "") -> None:
    clean_goal = (goal or "").strip()
    match_all = not clean_goal or clean_goal == "*"
    with _SHARD_LOCK:
        for store in (_SHARD_LEASES, _SHARD_CURSORS, _GEN_EVAL_STATE):
            for k in [k for k in store if k[0] == room and (match_all or k[1] == clean_goal)]:
                store.pop(k, None)


def is_contributor_quarantined(room: str, contributor_id: str) -> bool:
    if not contributor_id or contributor_id.startswith("worker:"):
        return False
    min_rep = get_config().trust.min_trusted_reputation
    with SQLITE_SYNC_LOCK:
        row = (
            ContributorTrust.objects.filter(room=room, contributor_id=contributor_id)
            .values("quarantined", "reputation")
            .first()
        )
    return bool(row and (row["quarantined"] or float(row["reputation"]) < min_rep))


def _ensure_gen_eval_state(
    room: str,
    goal: str,
    generation: int,
    *,
    max_cache_entries: int = 128,
) -> _GenEvalCache:
    key = (room, goal, int(generation))
    with _SHARD_LOCK:
        if (st := _GEN_EVAL_STATE.get(key)) is not None:
            return st

    verified: set[str] = set()
    evals: dict[str, dict] = {}
    with SQLITE_SYNC_LOCK:
        rows = list(
            ExternalEvalResult.objects.filter(room=room, goal=goal, generation=generation)
            .values_list("genome_hash", "status", "payload_json")
        )
    for gh, status, payload_json in rows:
        if not gh:
            continue
        if status == "verified":
            verified.add(gh)
        parsed = _loads_safe(payload_json, dict)
        if isinstance(parsed, dict) and parsed.get("summary"):
            parsed = _normalize_legacy_eval_payload(parsed)
            evals[gh] = {
                "hash": gh,
                "summary": parsed["summary"],
                "trust": parsed.get("trust") or {"status": status},
            }

    with _SHARD_LOCK:
        if len(_GEN_EVAL_STATE) > max_cache_entries:
            _GEN_EVAL_STATE.pop(next(iter(_GEN_EVAL_STATE)), None)
        st = _GEN_EVAL_STATE.setdefault(key, _GenEvalCache())
        st.verified.update(verified)
        for gh, ev in evals.items():
            st.evals.setdefault(gh, ev)
        return st


def _record_shard_eval_updates(room: str, goal: str, generation: int, stored_items: Iterable[dict]) -> None:
    key = (room, goal, int(generation))
    with _SHARD_LOCK:
        st = _GEN_EVAL_STATE.get(key)
        leases = _SHARD_LEASES.get(key)
        for it in stored_items:
            if not isinstance(it, dict) or not (gh := str(it.get("hash") or "")):
                continue
            status = str((it.get("trust") or {}).get("status") or "")
            if st is not None:
                if status == "verified":
                    st.verified.add(gh)
                if it.get("summary"):
                    st.evals[gh] = {
                        "hash": gh,
                        "summary": it["summary"],
                        "trust": it.get("trust") or {"status": status},
                    }
            if status == "verified" and leases is not None:
                leases.pop(gh, None)
        if leases is not None and not leases:
            _SHARD_LEASES.pop(key, None)


def get_cached_gen_eval_state(room: str, goal: str, generation: int) -> tuple[set[str], dict[str, dict]]:
    st = _ensure_gen_eval_state(room, goal, generation)
    with _SHARD_LOCK:
        return set(st.verified), dict(st.evals)


def count_verified_population(
    done_hashes: set[str],
    pop_hashes: Sequence[str],
    pop_hash_set: frozenset[str] | set[str] | None = None,
) -> int:
    if pop_hash_set is not None and len(pop_hash_set) == len(pop_hashes):
        return len(done_hashes & pop_hash_set)
    return sum(1 for h in pop_hashes if h and h in done_hashes)


def claim_crowd_eval_shard_indices(
    room: str,
    goal: str,
    generation: int,
    pop_hashes: Sequence[str],
    *,
    peer_id: str,
    batch_size: int = 4,
    lease_ttl_sec: float = 45.0,
    done_hashes: set[str] | None = None,
    pop_hash_set: frozenset[str] | set[str] | None = None,
    max_batch_size: int = MAX_SHARD_CLAIM_BATCH,
    max_lease_buckets: int = 256,
) -> tuple[list[int], int]:
    """O(batch_size) roving-cursor shard allocator returning ``(chosen_indices, completed_count)``."""
    total = len(pop_hashes)
    if total == 0 or batch_size <= 0:
        return [], 0

    done = done_hashes if done_hashes is not None else get_cached_gen_eval_state(room, goal, generation)[0]
    completed_cnt = count_verified_population(done, pop_hashes, pop_hash_set)
    if completed_cnt >= total:
        return [], completed_cnt

    now = time.monotonic()
    key = (room, goal, int(generation))
    eff_batch = max(1, min(max_batch_size, int(batch_size)))

    with _SHARD_LOCK:
        if len(_SHARD_LEASES) > max_lease_buckets:
            for sk in [k for k, lm in _SHARD_LEASES.items() if not lm or all(exp <= now for _, exp in lm.values())]:
                _SHARD_LEASES.pop(sk, None)
                _SHARD_CURSORS.pop(sk, None)

        leases = _SHARD_LEASES.setdefault(key, {})
        for gh in [gh for gh, (_, exp) in leases.items() if gh in done or exp <= now]:
            leases.pop(gh, None)

        chosen: list[int] = []
        start_cur = _SHARD_CURSORS.get(key, 0) % total
        for step in range(total):
            idx = (start_cur + step) % total
            gh = pop_hashes[idx]
            if not gh or gh in done or gh in leases:
                continue
            leases[gh] = (peer_id, now + lease_ttl_sec)
            chosen.append(idx)
            if len(chosen) >= eff_batch:
                break

        if chosen:
            _SHARD_CURSORS[key] = (chosen[-1] + 1) % total
            return chosen, completed_cnt

        # Tail straggler fallback: re-issue oldest leased undone slots
        leased_undone = sorted(
            (1 if leases[gh][0] == peer_id else 0, leases[gh][1], idx)
            for idx, gh in enumerate(pop_hashes)
            if gh and gh not in done and gh in leases
        )
        return [idx for _, _, idx in leased_undone[:eff_batch]], completed_cnt


def compute_goal_eval_rankings(
    genomes: list[dict],
    eval_map: dict[str, dict],
    *,
    value_digits: int = 4,
) -> list[dict[str, object]]:
    """Build global 1..K leaderboard across evaluated individuals in ``genomes``."""
    scored: list[tuple[float, int, dict[str, object]]] = []
    for idx, g in enumerate(genomes):
        if not isinstance(g, dict):
            continue
        gh = str(g.get("hash") or "")
        rec = eval_map.get(gh)
        if not isinstance(rec, dict):
            continue
        summ = rec.get("summary") if isinstance(rec.get("summary"), dict) else {}
        try:
            pval = float(summ.get("primary_value", 0.0))
        except (TypeError, ValueError):
            pval = 0.0
        if not math.isfinite(pval):
            continue
        trust = rec.get("trust") if isinstance(rec.get("trust"), dict) else {}
        scored.append(
            (
                pval,
                idx,
                {
                    "index": idx,
                    "id": g.get("id", 0),
                    "uuid": str(g.get("uuid") or ""),
                    "hash": gh,
                    "primary_value": round(pval, value_digits),
                    "primary_display": str(summ.get("primary_display") or ""),
                    "badge_text": str(summ.get("badge_text") or ""),
                    "tier": str(summ.get("tier") or ""),
                    "status": str(trust.get("status") or "verified"),
                },
            )
        )
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [{"rank": rank_idx, **entry} for rank_idx, (_, _, entry) in enumerate(scored, start=1)]
