from __future__ import annotations

import base64
import json
import math
import statistics
from typing import Iterable

from django.db import transaction
from django.db.models import F

from direct_evolution_together.config import TrustPolicyConfig, get_config
from direct_evolution_together.models import ContributorTrust, ExternalEvalResult

MAX_WORKER_POSTER_B64 = 350_000


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
    for key in ("meanDistance", "minDistance", "meanVx", "meanUpright", "meanSteps", "minSteps", "meanAbsX", "return"):
        val = raw_sum.get(key)
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            out[key] = round(float(val), 4)
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
    rep_sum["primaryValue"] = round(target, 4)
    if "meanDistance" in rep_sum:
        rep_sum["meanDistance"] = round(target, 2)
        rep_sum["primaryDisplay"] = f"{target:+.1f}m"
        rep_sum["badgeText"] = f"{target:+.1f}m"
    elif "meanSteps" in rep_sum:
        rep_sum["meanSteps"] = round(target, 1)
    return rep_sum


def _adjust_reputation(room: str, cids: Iterable[str], agreed: bool, trust_cfg: TrustPolicyConfig) -> None:
    delta = trust_cfg.agree_reward if agreed else -trust_cfg.mismatch_penalty
    for cid in cids:
        if not cid or cid.startswith("worker:"):
            continue
        obj, _ = ContributorTrust.objects.get_or_create(
            room=room,
            contributor_id=cid,
            defaults={"reputation": trust_cfg.initial_reputation},
        )
        new_rep = max(0.0, min(1.0, float(obj.reputation) + delta))
        obj.reputation = round(new_rep, 4)
        obj.agreements += int(agreed)
        obj.mismatches += int(not agreed)
        obj.quarantined = new_rep < trust_cfg.min_trusted_reputation
        obj.save(update_fields=["reputation", "agreements", "mismatches", "quarantined", "updated_at"])


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
    return "verified", atts, entry


def _ingest_peer(
    room: str,
    gh: str,
    clean_sum: dict[str, object],
    prev_status: str,
    existing_payload: dict,
    atts: list[dict],
    contrib: ContributorTrust,
    trust_cfg: TrustPolicyConfig,
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

    if len(cluster) >= trust_cfg.quorum:
        status = "verified"
        eff_summary = _aggregate_cluster(cluster, trust_cfg)
        newly_agreed = [c["cid"] for c in atts if c.get("cid") in cluster_cids and not c.get("scored")]
        newly_outlier = [c["cid"] for c in atts if c.get("cid") not in cluster_cids and not c.get("scored")]
        if newly_agreed:
            _adjust_reputation(room, newly_agreed, True, trust_cfg)
        if newly_outlier:
            _adjust_reputation(room, newly_outlier, False, trust_cfg)
        for c in atts:
            c["scored"] = True
        if already_idx is None and len(atts) >= trust_cfg.max_attestations:
            _adjust_reputation(room, [contrib.contributor_id], values_agree(pval, float(eff_summary["primaryValue"]), trust_cfg), trust_cfg)
    elif prev_status == "verified" and existing_payload.get("summary"):
        status = "verified"
        eff_summary = existing_payload["summary"]
        if already_idx is None:
            matches = values_agree(pval, float(eff_summary.get("primaryValue", 0.0)), trust_cfg)
            _adjust_reputation(room, [contrib.contributor_id], matches, trust_cfg)
            for a in atts:
                if a.get("cid") == contrib.contributor_id:
                    a["scored"] = True
    else:
        status = "disputed" if len(atts) >= 2 else "pending"
        eff_summary = existing_payload.get("summary") or clean_sum

    entry = {
        **existing_payload,
        "hash": gh,
        "summary": eff_summary,
        "trust": {
            "status": status,
            "attestations": len(cluster) if status == "verified" else len(atts),
            "quorum": trust_cfg.quorum,
            "authority": bool(existing_payload.get("trust", {}).get("authority")),
        },
    }
    return status, atts, entry


def submit_eval_attestations(
    *,
    room: str,
    goal: str,
    generation: int,
    results: list[dict],
    contributor_id: str,
    authority: bool = False,
) -> dict[str, object]:
    trust_cfg = get_config().trust
    stored_items: list[dict] = []
    verified_count = 0
    contrib: ContributorTrust | None = None

    with transaction.atomic():
        if not authority:
            contrib, _ = ContributorTrust.objects.get_or_create(
                room=room,
                contributor_id=contributor_id,
                defaults={"reputation": trust_cfg.initial_reputation},
            )
            if contrib.quarantined or contrib.reputation < trust_cfg.min_trusted_reputation:
                return {
                    "items": [],
                    "contributorReputation": float(contrib.reputation),
                    "quarantined": True,
                    "verifiedCount": 0,
                }
            # Acquire SQLite write lock before any SELECT to avoid BUSY_SNAPSHOT across uvicorn workers.
            ContributorTrust.objects.filter(pk=contrib.pk).update(submissions=F("submissions") + len(results[:64]))

        for raw_item in results[:64]:
            if not isinstance(raw_item, dict):
                continue
            gh = str(raw_item.get("hash") or "").strip()[:32]
            raw_sum = raw_item.get("summary")
            if not gh or not isinstance(raw_sum, dict):
                continue
            clean_sum = sanitize_summary(raw_sum)

            row = ExternalEvalResult.objects.filter(
                room=room, goal=goal, generation=generation, genome_hash=gh
            ).first()
            existing_payload = _loads_safe(row.payload_json, dict) if row else {}
            attestations = _loads_safe(row.attestations_json, list) if row else []

            if authority:
                status, attestations, entry = _ingest_authority(raw_item, gh, clean_sum, existing_payload, contributor_id)
            else:
                assert contrib is not None
                prev_status = row.status if row else "pending"
                status, attestations, entry = _ingest_peer(
                    room, gh, clean_sum, prev_status, existing_payload, attestations, contrib, trust_cfg
                )

            ExternalEvalResult.objects.update_or_create(
                room=room,
                goal=goal,
                generation=generation,
                genome_hash=gh,
                defaults={
                    "status": status,
                    "attestations_json": json.dumps(attestations),
                    "payload_json": json.dumps(entry),
                },
            )
            if status == "verified":
                verified_count += 1
            stored_items.append(entry)

    rep_val, quarantined = 1.0, False
    if contrib is not None:
        contrib.refresh_from_db()
        rep_val = float(contrib.reputation)
        quarantined = bool(contrib.quarantined)

    return {
        "items": stored_items,
        "contributorReputation": rep_val,
        "quarantined": quarantined,
        "verifiedCount": verified_count,
    }


def fetch_room_evals(room: str, goal: str, generation: int, *, trusted_only: bool = False) -> list[dict]:
    qs = ExternalEvalResult.objects.filter(room=room, goal=goal, generation=generation)
    if trusted_only:
        qs = qs.filter(status="verified")
    items: list[dict] = []
    for row in qs.order_by("id"):
        payload = _loads_safe(row.payload_json, dict)
        if payload:
            payload.setdefault("trust", {"status": row.status})
            items.append(payload)
    return items
