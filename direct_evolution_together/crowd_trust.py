from __future__ import annotations

import base64
import json
import math
import statistics
import threading
import time
from typing import Iterable

from django.db import OperationalError, transaction
from django.utils import timezone

from direct_evolution_together.config import TrustPolicyConfig, get_config
from direct_evolution_together.models import ContributorTrust, ExternalEvalResult

MAX_WORKER_POSTER_B64 = 350_000
SQLITE_SYNC_LOCK = threading.RLock()


def _loads_safe(raw: str, expected: type) -> object:
    try:
        val = json.loads(raw)
        return val if isinstance(val, expected) else expected()
    except Exception:
        return expected()


def sanitize_summary(raw_sum: dict) -> dict[str, object]:
    try:
        pval = float(raw_sum.get("primaryValue", 0.0))
    except (TypeError, ValueError):
        pval = 0.0
    if not math.isfinite(pval) or abs(pval) >= 1e12:
        pval = 0.0

    tier = str(raw_sum.get("tier") or "")
    tier = tier if tier in ("", "mid", "solved") else ""
    badge = str(raw_sum.get("badgeText") or f"{pval:.2f}")[:24]
    out: dict[str, object] = {
        "primaryValue": round(pval, 4),
        "primaryDisplay": str(raw_sum.get("primaryDisplay") or badge)[:32],
        "badgeText": badge,
        "tier": tier,
        "secondaryText": str(raw_sum.get("secondaryText") or "")[:120],
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
    body_poses = raw_replay.get("bodyPoses")
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
    out: dict[str, object] = {"bodyPoses": clean_poses}
    raw_stride = raw_replay.get("poseStride")
    base_stride = int(raw_stride) if isinstance(raw_stride, int) and 1 <= raw_stride <= 64 else 1
    if orig_len > len(clean_poses) and len(clean_poses) > 1:
        eff_stride = max(1, min(64, round(base_stride * (orig_len - 1) / (len(clean_poses) - 1))))
    else:
        eff_stride = base_stride
    out["poseStride"] = eff_stride
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
                "bias": float(n.get("bias", 0.0)),
            }
            for i, n in enumerate(raw_topo["nodes"])
            if isinstance(n, dict)
        ],
        "edges": [
            {
                "from": str(e.get("from", e.get("in", "")))[:24],
                "to": str(e.get("to", e.get("out", "")))[:24],
                "weight": float(e.get("weight", 0.0)),
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
    prev_primary = float(rep_sum.get("primaryValue", target))
    rep_sum["primaryValue"] = round(target, 4)
    if "return" in rep_sum and abs(float(rep_sum["return"]) - prev_primary) <= 1e-2:
        rep_sum["return"] = round(target, 2)
        if "meanDistance" in rep_sum:
            mdx = float(rep_sum["meanDistance"])
            disp = f"{target:+.0f}"
            rep_sum["primaryDisplay"] = f"{disp} ({mdx:+.2f}m)"
            rep_sum["badgeText"] = f"{disp} \u00b7 {mdx:+.2f}m"
    elif "meanDistance" in rep_sum:
        rep_sum["meanDistance"] = round(target, 2)
        rep_sum["primaryDisplay"] = f"{target:+.1f}m"
        rep_sum["badgeText"] = f"{target:+.1f}m"
    elif "meanSteps" in rep_sum:
        rep_sum["meanSteps"] = round(target, 1)
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
    pval = float(clean_sum["primaryValue"])  # type: ignore[arg-type]
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

    poster_b64 = raw_item.get("posterPngB64")
    if isinstance(poster_b64, str) and poster_b64 and len(poster_b64) <= MAX_WORKER_POSTER_B64:
        try:
            if base64.b64decode(poster_b64, validate=True).startswith(b"\x89PNG\r\n\x1a\n"):
                entry["posterDataUrl"] = f"data:image/png;base64,{poster_b64}"
        except Exception:
            pass
    elif "posterDataUrl" in existing_payload:
        entry["posterDataUrl"] = existing_payload["posterDataUrl"]
    traj = sanitize_traj_sample(raw_item.get("trajSample"))
    if traj is not None:
        entry["trajSample"] = traj
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
    pval = float(clean_sum["primaryValue"])  # type: ignore[arg-type]
    already_idx = next((i for i, a in enumerate(atts) if a.get("cid") == contrib.contributor_id), None)
    att_entry = {
        "cid": contrib.contributor_id,
        "val": pval,
        "weight": float(contrib.reputation),
        "summary": clean_sum,
        "scored": False,
    }
    if already_idx is not None:
        if prev_status != "verified" and not atts[already_idx].get("scored"):
            atts[already_idx] = att_entry
    elif len(atts) < trust_cfg.max_attestations:
        atts.append(att_entry)

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
            if already_idx is None and len(atts) >= trust_cfg.max_attestations:
                _queue_reputation_delta(
                    rep_deltas,
                    [contrib.contributor_id],
                    values_agree(pval, float(eff_summary["primaryValue"]), trust_cfg),
                )
    elif prev_status == "verified" and existing_payload.get("summary"):
        status = "verified"
        eff_summary = existing_payload["summary"]
        if already_idx is None and eff_quorum >= 2:
            matches = values_agree(pval, float(eff_summary.get("primaryValue", 0.0)), trust_cfg)
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
    if "trajSample" not in entry or status == "verified":
        traj = sanitize_traj_sample(raw_item.get("trajSample"))
        if traj is not None:
            entry["trajSample"] = traj
    if "replay" not in entry:
        rep = sanitize_replay(raw_item.get("replay"))
        if rep is not None:
            entry["replay"] = rep
    return status, atts, entry


def submit_eval_attestations(
    *,
    room: str,
    goal: str,
    generation: int,
    results: list[dict],
    contributor_id: str,
    authority: bool = False,
    active_peers: int | None = None,
) -> dict[str, object]:
    trust_cfg = get_config().trust
    eff_quorum = (
        1
        if (active_peers is not None and active_peers <= 1)
        else (min(trust_cfg.quorum, max(1, active_peers)) if active_peers is not None else trust_cfg.quorum)
    )
    stored_items: list[dict] = []
    changed_items: list[dict] = []
    verified_count = 0
    contrib: ContributorTrust | None = None

    parsed_batch: list[tuple[str, dict, dict[str, object]]] = []
    for raw_item in results[:64]:
        if not isinstance(raw_item, dict):
            continue
        gh = str(raw_item.get("hash") or "").strip()[:32]
        raw_sum = raw_item.get("summary")
        if not gh or not isinstance(raw_sum, dict):
            continue
        parsed_batch.append((gh, raw_item, sanitize_summary(raw_sum)))

    if not parsed_batch:
        return {
            "items": [],
            "changedItems": [],
            "contributorReputation": 1.0 if authority else float(trust_cfg.initial_reputation),
            "quarantined": False,
            "verifiedCount": 0,
        }

    for attempt in range(4):
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
                            "changedItems": [],
                            "contributorReputation": float(contrib.reputation),
                            "quarantined": True,
                            "verifiedCount": 0,
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
                        submission_count=len(results[:64]),
                    )
            break
        except OperationalError as exc:
            if "locked" not in str(exc).lower() or attempt == 3:
                raise
            time.sleep(0.02 * (attempt + 1))

    rep_val, quarantined = 1.0, False
    if contrib is not None:
        rep_val = float(contrib.reputation)
        quarantined = bool(contrib.quarantined)

    return {
        "items": stored_items,
        "changedItems": changed_items,
        "contributorReputation": rep_val,
        "quarantined": quarantined,
        "verifiedCount": verified_count,
        "quorum": eff_quorum,
    }


def count_verified_room_evals(room: str, goal: str, generation: int) -> int:
    return ExternalEvalResult.objects.filter(
        room=room, goal=goal, generation=generation, status="verified"
    ).count()


def fetch_room_eval_map(room: str, goal: str, generation: int, *, trusted_only: bool = True) -> dict[str, dict]:
    items = fetch_room_evals(room, goal, generation, trusted_only=trusted_only)
    return {str(it["hash"]): it for it in items if isinstance(it, dict) and it.get("hash")}


def fetch_room_evals(
    room: str,
    goal: str,
    generation: int,
    *,
    trusted_only: bool = False,
    include_traj: bool = True,
) -> list[dict]:
    with SQLITE_SYNC_LOCK:
        qs = ExternalEvalResult.objects.filter(room=room, goal=goal, generation=generation)
        if trusted_only:
            qs = qs.filter(status="verified")
        rows = list(qs.order_by("id"))
    items: list[dict] = []
    for row in rows:
        payload = _loads_safe(row.payload_json, dict)
        if payload:
            if not include_traj and "trajSample" in payload:
                payload = {k: v for k, v in payload.items() if k != "trajSample"}
            payload.setdefault("trust", {"status": row.status})
            items.append(payload)
    return items
