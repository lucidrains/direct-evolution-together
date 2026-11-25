import atexit
import json
import os
import re
import threading
import time

import nimporter_plus  # noqa: F401 - registers Nim import hook
from django.db import close_old_connections

from direct_evolution_together import mummy_ws
from direct_evolution_together.cppn import (
    MAX_BROOD_SIZE,
    MIN_BROOD_SIZE,
    ensure_genome_uuid,
    evolve_population,
    evolve_population_broods,
    normalize_short_uuid,
    sync_genome_counter,
)
from direct_evolution_together.config import get_config
from direct_evolution_together.crowd_trust import SQLITE_SYNC_LOCK, fetch_room_eval_map
from direct_evolution_together.environments import (
    clamp_pop_for_task,
    create_task_population,
    default_elite_for_task,
    default_pop_for_task,
    get_nim_task_configs_payload,
    get_task_spec,
    is_gen0_founder_presolved,
    normalize_task_id,
    sanitize_task_env_params,
)
from direct_evolution_together.models import ChatMessage, RoomEvolutionState
from direct_evolution_together.populora_engine import (
    evolve_populora_broods,
    evolve_populora_population,
)

HISTORY_LIMIT = 100
DEFAULT_GOAL_SPECS: tuple[tuple[str, str, int | None], ...] = get_config().get_lobby_goal_specs()
DEFAULT_GOALS = tuple(name for name, _, _ in DEFAULT_GOAL_SPECS)

_bridge_lock = threading.Lock()
_stop_event = threading.Event()
_worker_thread: threading.Thread | None = None
_started = False
_persist_to_db = False


def _build_lineage_from_genomes(genomes: list[dict], gen: int = 0) -> list[dict]:
    nodes: list[dict] = []
    for g in genomes:
        guuid = ensure_genome_uuid(g)
        if g.get("kind") == "populora":
            nodes.append(
                {
                    "id": int(g["id"]),
                    "uuid": guuid,
                    "gen": gen,
                    "parents": [int(p) for p in g.get("parents", [])],
                    "op": str(g.get("op", "random")),
                    "kind": "populora",
                    "hash": str(g.get("hash", "")),
                    "base_hash": str(g.get("base_hash", "")),
                }
            )
        else:
            nodes.append(
                {
                    "id": int(g["id"]),
                    "uuid": guuid,
                    "gen": gen,
                    "parents": [int(p) for p in g.get("parents", [])],
                    "op": str(g.get("op", "random")),
                    "weights": g["weights"],
                    "biases": g["biases"],
                    "activations": g["activations"],
                }
            )
    return nodes


_LEGACY_GOAL_KEY_MAP = (
    ("envParams", "env_params"),
    ("autoEvolve", "auto_evolve"),
    ("simPhaseEnabled", "sim_phase_enabled"),
    ("simCompleted", "sim_completed"),
    ("lineageTotal", "lineage_total"),
)
_LEGACY_PARAM_KEY_MAP = (
    ("minVotes", "min_votes"),
    ("maxVotesPerUser", "max_votes_per_user"),
    ("mutSigma", "mut_sigma"),
    ("eliteFrac", "elite_frac"),
    ("crossoverRate", "crossover_rate"),
)
_LEGACY_GENOME_KEY_MAP = (
    ("baseHash", "base_hash"),
    ("baseSeed", "base_seed"),
    ("lowRank", "low_rank"),
)


def _rename_legacy_keys(obj: dict, mapping: tuple[tuple[str, str], ...]) -> bool:
    changed = False
    for old_key, new_key in mapping:
        if old_key in obj:
            val = obj.pop(old_key)
            if new_key not in obj:
                obj[new_key] = val
            changed = True
    return changed


def _ensure_state_uuids(state_json: str) -> tuple[str, bool]:
    """Backfill short UUIDs and migrate legacy camelCase keys on persisted DB room states."""
    try:
        parsed = json.loads(state_json)
    except ValueError:
        return state_json, False
    if not isinstance(parsed, dict) or not isinstance(parsed.get("goals"), list):
        return state_json, False

    changed = False
    if isinstance(parsed.get("params"), dict):
        changed |= _rename_legacy_keys(parsed["params"], _LEGACY_PARAM_KEY_MAP)
    for goal in parsed["goals"]:
        if not isinstance(goal, dict):
            continue
        changed |= _rename_legacy_keys(goal, _LEGACY_GOAL_KEY_MAP)
        if isinstance(goal.get("params"), dict):
            changed |= _rename_legacy_keys(goal["params"], _LEGACY_PARAM_KEY_MAP)
        id_to_uuid: dict[int, str] = {}
        for g in goal.get("genomes") or []:
            if isinstance(g, dict):
                changed |= _rename_legacy_keys(g, _LEGACY_GENOME_KEY_MAP)
                had = bool(normalize_short_uuid(g.get("uuid")))
                u = ensure_genome_uuid(g)
                if not had:
                    changed = True
                gid = g.get("id")
                if isinstance(gid, int):
                    id_to_uuid[gid] = u
        for node in goal.get("lineage") or []:
            if isinstance(node, dict):
                changed |= _rename_legacy_keys(node, _LEGACY_GENOME_KEY_MAP)
                cur_u = normalize_short_uuid(node.get("uuid"))
                gid = node.get("id")
                if not cur_u:
                    if isinstance(gid, int) and gid in id_to_uuid:
                        node["uuid"] = id_to_uuid[gid]
                    else:
                        u = ensure_genome_uuid(node)
                        if isinstance(gid, int):
                            id_to_uuid[gid] = u
                    changed = True
    if not changed:
        return state_json, False
    return json.dumps(parsed, separators=(",", ":")), True


def _build_goal_entry(
    name: str,
    task: str,
    pop_size: int,
    env_params: object = None,
) -> dict:
    clean_task = normalize_task_id(task)
    spec = get_task_spec(clean_task)
    eff_pop = clamp_pop_for_task(clean_task, pop_size)
    clean_env = sanitize_task_env_params(clean_task, env_params)
    entry: dict = {
        "name": name,
        "task": clean_task,
        "min_votes": spec.default_min_votes,
        "genomes": create_task_population(clean_task, eff_pop),
    }
    if clean_env:
        entry["env_params"] = clean_env
    if spec.default_elite_frac is not None:
        entry["elite_frac"] = float(spec.default_elite_frac)
    if spec.default_brood_size > 1:
        entry["brood_size"] = int(spec.default_brood_size)
    return entry


def _resolve_lobby_goal_pop(task_pop: int | None, pop_override: int | None, image_pop: int = 30) -> int:
    base = int(image_pop) if pop_override is None else int(pop_override)
    if task_pop is None:
        return base
    return min(base, task_pop) if pop_override is not None else task_pop


def _build_goals_payload(
    goal_specs: tuple[tuple[str, str, int | None], ...] | tuple[str, ...] | None = None,
    pop_size: int | None = None,
    env_params: object = None,
    image_pop: int = 30,
) -> str:
    specs = get_config().get_lobby_goal_specs() if goal_specs is None else goal_specs
    entries: list[dict] = []
    for item in specs:
        if isinstance(item, tuple):
            name, task, cap_pop = item
            target_pop = _resolve_lobby_goal_pop(cap_pop, pop_size, image_pop=image_pop)
        else:
            name, task = str(item), "image"
            target_pop = _resolve_lobby_goal_pop(None, pop_size, image_pop=image_pop)
        entries.append(_build_goal_entry(name, task, target_pop, env_params=env_params))
    return json.dumps(entries, separators=(",", ":"))


def build_room_state_json(
    goal_name: str,
    genomes: list[dict],
    *,
    task: str = "image",
    env_params: object = None,
    min_votes: int | None = None,
    max_votes_per_user: int = 20,
    mut_sigma: float = 0.30,
    elite_frac: float | None = None,
    default_elite_fallback: float = 0.20,
    max_elite_frac: float = 0.50,
    crossover_rate: float = 0.75,
    brood_size: int | None = None,
    auto_evolve: bool | None = None,
) -> str:
    """Serialize initial Generation-0 room state matching Mummy's RoomEvolutionState schema."""
    clean_task = normalize_task_id(task)
    spec = get_task_spec(clean_task)
    clean_env = sanitize_task_env_params(clean_task, env_params)
    eff_min_votes = spec.default_min_votes if min_votes is None else int(min_votes)
    if spec.kind == "control":
        eff_min_votes = max(1, min(eff_min_votes, int(max_votes_per_user)))
    eff_elite_frac = (
        default_elite_for_task(clean_task, default_elite_fallback)
        if elite_frac is None
        else max(0.0, min(max_elite_frac, float(elite_frac)))
    )
    eff_brood_size = max(
        MIN_BROOD_SIZE,
        min(MAX_BROOD_SIZE, int(spec.default_brood_size if brood_size is None else brood_size)),
    )
    params = {
        "max_votes_per_user": int(max_votes_per_user),
        "min_votes": int(eff_min_votes),
        "mut_sigma": float(mut_sigma),
        "elite_frac": float(eff_elite_frac),
        "crossover_rate": float(crossover_rate),
        "brood_size": int(eff_brood_size),
    }
    lineage = _build_lineage_from_genomes(genomes, gen=0)
    eff_auto = spec.default_auto_evolve if auto_evolve is None else bool(auto_evolve)
    goal_doc: dict[str, object] = {
        "name": goal_name,
        "task": clean_task,
        "generation": 0,
        "genomes": genomes,
        "lineage": lineage,
        "params": params,
        "phase": "simulating" if spec.sim_phase_enabled else "voting",
        "sim_completed": 0,
        "auto_evolve": eff_auto,
    }
    if clean_env:
        goal_doc["env_params"] = clean_env
    return json.dumps(
        {
            "default_params": params,
            "goals": [goal_doc],
        },
        separators=(",", ":"),
    )


_FAST_ID_SCAN_RE = re.compile(r'"id":(\d+)')
_FAST_UUID_SCAN_RE = re.compile(r'"uuid":"[0-9a-fA-F]{8}"')


def _fast_ensure_uuids_and_max_id(state_json: str) -> tuple[str, bool, int]:
    """Scan persisted room JSON for max genome ID without parsing 40 MB JSON when UUIDs already exist."""
    if not state_json:
        return state_json, False, 0
    id_matches = _FAST_ID_SCAN_RE.findall(state_json)
    uuid_count = len(_FAST_UUID_SCAN_RE.findall(state_json))
    if id_matches and uuid_count >= len(id_matches):
        return state_json, False, max(int(v) for v in id_matches)
    updated_json, changed = _ensure_state_uuids(state_json)
    return updated_json, changed, _max_genome_id_in_state(updated_json)


def _max_genome_id_in_state(state_json: str) -> int:
    max_id = 0
    try:
        parsed = json.loads(state_json)
    except ValueError:
        return 0
    if not isinstance(parsed, dict):
        return 0
    for goal in parsed.get("goals", []):
        if not isinstance(goal, dict):
            continue
        for key in ("genomes", "lineage"):
            for item in goal.get(key, []):
                if isinstance(item, dict):
                    gid = item.get("id")
                    if isinstance(gid, int) and gid > max_id:
                        max_id = gid
    return max_id


def _ensure_lobby_default_goals(
    state_json: str,
    default_pop_size: int | None = None,
    default_max_votes: int | None = None,
    *,
    fallback_max_votes: int = 20,
    default_mut_sigma: float = 0.30,
    default_elite_fallback: float = 0.20,
    default_crossover_rate: float = 0.75,
    elite_tol: float = 1e-6,
) -> tuple[str, bool]:
    """Ensure newly added default lobby goals are seeded and vote caps match runtime config."""
    try:
        parsed = json.loads(state_json)
    except ValueError:
        return state_json, False
    if not isinstance(parsed, dict) or not isinstance(parsed.get("goals"), list):
        return state_json, False

    existing = {str(g.get("name", "")) for g in parsed["goals"] if isinstance(g, dict)}
    base_params = parsed.get("default_params") if isinstance(parsed.get("default_params"), dict) else {}
    changed = False
    if (
        default_max_votes is not None
        and isinstance(base_params, dict)
        and int(base_params.get("max_votes_per_user", 0)) < int(default_max_votes)
    ):
        base_params["max_votes_per_user"] = int(default_max_votes)
        changed = True
    cfg = get_config()
    for g in parsed["goals"]:
        if not isinstance(g, dict):
            continue
        p = g.get("params")
        if (
            default_max_votes is not None
            and isinstance(p, dict)
            and int(p.get("max_votes_per_user", 0)) < int(default_max_votes)
        ):
            p["max_votes_per_user"] = int(default_max_votes)
            changed = True
        task_id = normalize_task_id(str(g.get("task", "image")))
        if task_id != "image":
            env_cfg = cfg.get_env(task_id)
            canonical_name = env_cfg.lobby_name or task_id.title()
            default_task_env = sanitize_task_env_params(task_id, None)
            cur_name = str(g.get("name", ""))
            if env_cfg.lobby_default and cur_name != canonical_name and canonical_name not in existing:
                existing.discard(cur_name)
                g["name"] = canonical_name
                existing.add(canonical_name)
                if default_task_env:
                    g["env_params"] = default_task_env
                changed = True
            elif default_task_env and not g.get("env_params"):
                g["env_params"] = default_task_env
                changed = True
        task_spec = get_task_spec(task_id)
        if task_spec.kind == "control":
            p = g.get("params")
            if isinstance(p, dict):
                t_max_votes = max(
                    1,
                    int(p.get("max_votes_per_user", base_params.get("max_votes_per_user", fallback_max_votes))),
                )
                t_min_votes = max(1, min(task_spec.default_min_votes, t_max_votes))
                if int(p.get("min_votes", 0)) != t_min_votes:
                    p["min_votes"] = t_min_votes
                    changed = True
                exp_elite = default_elite_for_task(task_id, float(p.get("elite_frac", 0.0)))
                if abs(float(p.get("elite_frac", 0.0)) - exp_elite) > elite_tol:
                    p["elite_frac"] = exp_elite
                    changed = True
            cur_genomes = g.get("genomes")
            populora_incompatible = False
            if (
                task_spec.genome_kind == "populora"
                and task_spec.policy_spec is not None
                and isinstance(cur_genomes, list)
                and cur_genomes
            ):
                try:
                    from direct_evolution_together.populora_engine import validate_populora_genome_dict
                    validate_populora_genome_dict(cur_genomes[0], task_spec.policy_spec)
                except Exception:
                    populora_incompatible = True

            if populora_incompatible:
                target_pop = clamp_pop_for_task(
                    task_id,
                    _resolve_lobby_goal_pop(task_spec.default_pop_size, default_pop_size),
                )
                fresh = create_task_population(task_id, target_pop)
                g["generation"] = 0
                g["genomes"] = fresh
                g["lineage"] = _build_lineage_from_genomes(fresh, gen=0)
                g["sim_phase_enabled"] = bool(task_spec.sim_phase_enabled)
                g["auto_evolve"] = bool(task_spec.default_auto_evolve)
                g["phase"] = "simulating" if task_spec.sim_phase_enabled else "voting"
                g["sim_completed"] = 0
                changed = True
            elif int(g.get("generation", 0)) == 0:
                if isinstance(cur_genomes, list) and cur_genomes:
                    target_pop = clamp_pop_for_task(
                        task_id,
                        _resolve_lobby_goal_pop(task_spec.default_pop_size, default_pop_size),
                    )
                    needs_resize = len(cur_genomes) != target_pop
                    needs_refresh = any(
                        is_gen0_founder_presolved(task_id, cg)
                        for cg in cur_genomes
                    )
                    if needs_resize or needs_refresh:
                        fresh = create_task_population(
                            task_id,
                            target_pop if needs_resize else len(cur_genomes),
                        )
                        g["genomes"] = fresh
                        g["lineage"] = _build_lineage_from_genomes(fresh, gen=0)
                        g["phase"] = "simulating" if task_spec.sim_phase_enabled else "voting"
                        g["sim_completed"] = 0
                        changed = True
    added: list[dict] = []
    default_goal_max_votes = max(
        1,
        int(base_params.get("max_votes_per_user", default_max_votes or fallback_max_votes)),
    )
    for name, task, cap_pop in get_config().get_lobby_goal_specs():
        if name in existing:
            continue
        spec = get_task_spec(task)
        t_pop = _resolve_lobby_goal_pop(cap_pop, default_pop_size)
        genomes = create_task_population(task, t_pop)
        params = {
            "max_votes_per_user": default_goal_max_votes,
            "min_votes": max(0, min(int(spec.default_min_votes), default_goal_max_votes)),
            "mut_sigma": float(base_params.get("mut_sigma", default_mut_sigma)),
            "elite_frac": default_elite_for_task(task, float(base_params.get("elite_frac", default_elite_fallback))),
            "crossover_rate": float(base_params.get("crossover_rate", default_crossover_rate)),
            "brood_size": int(spec.default_brood_size),
        }
        new_goal_doc: dict[str, object] = {
            "name": name,
            "task": task,
            "generation": 0,
            "genomes": genomes,
            "lineage": _build_lineage_from_genomes(genomes, gen=0),
            "params": params,
            "phase": "simulating" if spec.sim_phase_enabled else "voting",
            "sim_completed": 0,
            "auto_evolve": bool(spec.default_auto_evolve),
        }
        default_env = sanitize_task_env_params(task, None)
        if default_env:
            new_goal_doc["env_params"] = default_env
        added.append(new_goal_doc)
    lobby_order_map = {
        name: idx for idx, (name, _, _) in enumerate(get_config().get_lobby_goal_specs())
    }
    combined_goals = list(parsed["goals"]) + added
    ordered_goals = sorted(
        enumerate(combined_goals),
        key=lambda pair: (
            lobby_order_map.get(str(pair[1].get("name", "")), 1000 + pair[0])
            if isinstance(pair[1], dict)
            else 1000 + pair[0]
        ),
    )
    new_goal_list = [g for _, g in ordered_goals]
    order_changed = [g.get("name") for g in new_goal_list if isinstance(g, dict)] != [
        g.get("name") for g in parsed["goals"] if isinstance(g, dict)
    ]
    if not added and not changed and not order_changed:
        return state_json, False

    parsed["goals"] = new_goal_list
    return json.dumps(parsed, separators=(",", ":")), True


def seed_from_db(default_pop_size: int | None = None, default_max_votes: int | None = None) -> None:
    """Seed Mummy's per-room chat history and evolved room states from Django DB."""
    close_old_connections()
    rooms = list(ChatMessage.objects.values_list("room", flat=True).distinct())
    for room in rooms:
        recent = list(
            ChatMessage.objects.filter(room=room).order_by("-created_at", "-id")[:HISTORY_LIMIT]
        )
        recent.reverse()
        payload = [msg.to_payload() for msg in recent]
        mummy_ws.seed_room_history(room, json.dumps(payload))

    max_id = 0
    for row in RoomEvolutionState.objects.all():
        state_json, uuid_changed, row_max_id = _fast_ensure_uuids_and_max_id(row.state_json)
        max_id = max(max_id, row_max_id)
        sync_genome_counter(max_id)
        lobby_changed = False
        if row.room == "lobby":
            updated_json, lobby_changed = _ensure_lobby_default_goals(
                state_json,
                default_pop_size=default_pop_size,
                default_max_votes=default_max_votes,
            )
            if lobby_changed:
                state_json = updated_json
                max_id = max(max_id, _max_genome_id_in_state(state_json))
        if uuid_changed or lobby_changed:
            row.state_json = state_json
            row.save(update_fields=["state_json", "updated_at"])
        mummy_ws.restore_room_state(row.room, state_json)
    sync_genome_counter(max_id)


def flush_room_states() -> int:
    """Drain dirty room evolution snapshots from Mummy and persist them via Django ORM."""
    raw = mummy_ws.drain_dirty_room_states()
    if not raw or raw == "[]":
        return 0
    items = json.loads(raw)
    if not items:
        return 0

    with SQLITE_SYNC_LOCK:
        close_old_connections()
        for item in items:
            RoomEvolutionState.objects.update_or_create(
                room=item["room"],
                defaults={"state_json": item["state_json"]},
            )
    return len(items)


def flush_pending_messages() -> int:
    """Drain queued chat messages from Mummy and persist them via Django ORM."""
    raw = mummy_ws.drain_pending_messages()
    if not raw or raw == "[]":
        return 0
    items = json.loads(raw)
    if not items:
        return 0

    with SQLITE_SYNC_LOCK:
        close_old_connections()
        ChatMessage.objects.bulk_create(
            [
                ChatMessage(
                    room=item["room"],
                    username=item["username"],
                    message=item["message"],
                )
                for item in items
            ]
        )
    return len(items)


def flush_evo_jobs() -> int:
    """Process pending PyTorch CPPN initialization/evolution jobs from Mummy."""
    raw = mummy_ws.drain_evo_jobs()
    if not raw or raw == "[]":
        return 0
    jobs = json.loads(raw)
    if not jobs:
        return 0

    processed = 0
    for job in jobs:
        kind = str(job.get("kind") or "")
        room = str(job.get("room") or "")
        goal_name = str(job.get("goal") or "")
        try:
            task = normalize_task_id(job.get("task", "image"))
            pop_size = clamp_pop_for_task(task, int(job.get("population_size") or default_pop_for_task(task)))
            if kind == "init_room":
                if _persist_to_db:
                    close_old_connections()
                    saved_row = RoomEvolutionState.objects.filter(room=room).first()
                    if saved_row is not None:
                        st_json, st_changed = _ensure_state_uuids(saved_row.state_json)
                        if st_changed:
                            saved_row.state_json = st_json
                            saved_row.save(update_fields=["state_json", "updated_at"])
                        sync_genome_counter(_max_genome_id_in_state(st_json))
                        mummy_ws.restore_room_state(room, st_json)
                        processed += 1
                        continue
                initial_goal = goal_name.strip()
                if initial_goal:
                    goal_specs: tuple[tuple[str, str, int | None], ...] = ((initial_goal, task, pop_size),)
                elif room != "lobby":
                    goal_specs = ()
                else:
                    goal_specs = get_config().get_lobby_goal_specs()
                mummy_ws.apply_room_init(
                    room,
                    _build_goals_payload(
                        goal_specs,
                        pop_size=pop_size,
                        env_params=job.get("env_params") if initial_goal else None,
                    ),
                )
            elif kind == "create":
                genomes = create_task_population(task, pop_size)
                clean_env = sanitize_task_env_params(task, job.get("env_params"))
                mummy_ws.apply_goal_create_with_env(
                    room,
                    goal_name,
                    job["username"],
                    json.dumps(genomes, separators=(",", ":")),
                    json.dumps(clean_env, separators=(",", ":")) if clean_env else "{}",
                )
            elif kind == "evolve":
                spec = get_task_spec(task)
                brood_size = max(MIN_BROOD_SIZE, min(MAX_BROOD_SIZE, int(job.get("brood_size", 1))))
                if spec.genome_kind == "populora" and spec.policy_spec is not None:
                    close_old_connections()
                    gen_idx = int(job.get("generation", 0))
                    eval_map = fetch_room_eval_map(room, goal_name, gen_idx, trusted_only=True)
                    if not eval_map:
                        eval_map = fetch_room_eval_map(room, goal_name, gen_idx, trusted_only=False)
                    if brood_size > 1:
                        brood_doc, top_score = evolve_populora_broods(
                            room=room,
                            goal=goal_name,
                            task_id=spec.task_id,
                            generation=gen_idx,
                            policy_spec=spec.policy_spec,
                            genome_dicts=job["genomes"],
                            scores=job["scores"],
                            up_counts=job["up_counts"],
                            eval_by_hash=eval_map,
                            mut_sigma=float(job.get("mut_sigma", 0.30)),
                            pop_size=pop_size,
                            elite_frac=float(job.get("elite_frac", 0.25)),
                            brood_size=brood_size,
                            max_horizon=max(1, spec.max_steps),
                        )
                        mummy_ws.apply_goal_brood(
                            room, goal_name, json.dumps(brood_doc, separators=(",", ":")), int(top_score)
                        )
                    else:
                        next_genomes, top_score = evolve_populora_population(
                            room=room,
                            goal=goal_name,
                            task_id=spec.task_id,
                            generation=gen_idx,
                            policy_spec=spec.policy_spec,
                            genome_dicts=job["genomes"],
                            scores=job["scores"],
                            up_counts=job["up_counts"],
                            eval_by_hash=eval_map,
                            mut_sigma=float(job.get("mut_sigma", 0.30)),
                            pop_size=pop_size,
                            elite_frac=float(job.get("elite_frac", 0.25)),
                            max_horizon=max(1, spec.max_steps),
                        )
                        mummy_ws.apply_goal_evolve(
                            room, goal_name, json.dumps(next_genomes, separators=(",", ":")), int(top_score)
                        )
                elif brood_size > 1:
                    brood_doc, top_score = evolve_population_broods(
                        job["genomes"],
                        job["scores"],
                        job["up_counts"],
                        mut_sigma=float(job.get("mut_sigma", 0.30)),
                        pop_size=pop_size,
                        elite_frac=float(job.get("elite_frac", 0.20)),
                        crossover_rate=float(job.get("crossover_rate", 0.75)),
                        brood_size=brood_size,
                    )
                    mummy_ws.apply_goal_brood(
                        room, goal_name, json.dumps(brood_doc, separators=(",", ":")), int(top_score)
                    )
                else:
                    next_genomes, top_score = evolve_population(
                        job["genomes"],
                        job["scores"],
                        job["up_counts"],
                        mut_sigma=float(job.get("mut_sigma", 0.30)),
                        pop_size=pop_size,
                        elite_frac=float(job.get("elite_frac", 0.20)),
                        crossover_rate=float(job.get("crossover_rate", 0.75)),
                    )
                    mummy_ws.apply_goal_evolve(
                        room, goal_name, json.dumps(next_genomes, separators=(",", ":")), int(top_score)
                    )
            processed += 1
        except Exception as exc:
            import sys
            print(f"[ws_bridge] Evo job failed ({kind} {room}/{goal_name}): {exc}", file=sys.stderr, flush=True)
            if kind == "evolve" and room and goal_name:
                try:
                    mummy_ws.cancel_goal_evolve(room, goal_name)
                except Exception:
                    pass
            elif kind == "create" and room and goal_name:
                try:
                    mummy_ws.cancel_goal_create(room, goal_name)
                except Exception:
                    pass
            elif kind == "init_room" and room:
                try:
                    mummy_ws.apply_room_init(room, "[]")
                except Exception:
                    pass
    return processed


def _worker_loop(
    poll_interval_sec: float = 0.02,
    state_flush_interval_sec: float = 0.25,
) -> None:
    last_state_flush = 0.0
    while not _stop_event.wait(poll_interval_sec):
        try:
            flush_evo_jobs()
        except Exception:
            pass
        try:
            flush_pending_messages()
        except Exception:
            pass
        if _persist_to_db:
            now = time.monotonic()
            if now - last_state_flush >= state_flush_interval_sec:
                last_state_flush = now
                try:
                    flush_room_states()
                except Exception:
                    pass


def is_running() -> bool:
    with _bridge_lock:
        return _started


def start_ws_server(
    host: str = "127.0.0.1",
    port: int = 8001,
    seed: bool = True,
    pop_size: int | None = None,
    pop_size_override: bool | None = None,
    min_votes: int | None = None,
    min_votes_override: bool = False,
    countdown_ms: int = 8000,
    max_votes: int = 20,
    mut_sigma: float = 0.30,
    elite_frac: float = 0.20,
    crossover_rate: float = 0.75,
    brood_size: int = 1,
    admin_password: str | None = None,
    evolve_token: str | None = None,
) -> None:
    """Start the Nim Mummy WebSocket server and background PyTorch/ORM bridge loop."""
    global _worker_thread, _started, _persist_to_db
    with _bridge_lock:
        if _started:
            return
        _persist_to_db = bool(seed)
        eff_default_pop = 30 if pop_size is None else int(pop_size)
        eff_pop_override = (
            int(pop_size)
            if (pop_size is not None and pop_size_override is not False)
            else None
        )
        resolved_admin_pw = os.getenv("DET_ADMIN_PASSWORD", "") if admin_password is None else admin_password
        mummy_ws.register_task_configs(json.dumps(get_nim_task_configs_payload()))
        mummy_ws.set_admin_password(resolved_admin_pw)
        mummy_ws.set_evolve_token(evolve_token or "")
        if min_votes is not None or countdown_ms != 8000:
            mummy_ws.set_round_config(
                0 if min_votes is None else int(min_votes),
                int(countdown_ms),
                bool(min_votes_override),
            )
        mummy_ws.set_default_population_size(eff_default_pop)
        mummy_ws.set_default_max_votes(int(max_votes))
        mummy_ws.set_default_ga_params(float(mut_sigma), float(elite_frac), float(crossover_rate))
        mummy_ws.set_default_brood_size(max(MIN_BROOD_SIZE, min(MAX_BROOD_SIZE, int(brood_size))))
        if seed:
            seed_from_db(default_pop_size=eff_pop_override, default_max_votes=int(max_votes))
        if not mummy_ws.has_room_evolution("lobby"):
            mummy_ws.seed_room_goals(
                "lobby",
                _build_goals_payload(pop_size=eff_pop_override, image_pop=eff_default_pop),
            )
        if _persist_to_db:
            flush_room_states()
        _stop_event.clear()
        mummy_ws.start_server(host, int(port))
        print(f"Mummy WebSocket server listening on ws://{host}:{port}", flush=True)
        _worker_thread = threading.Thread(
            target=_worker_loop,
            name="mummy-bridge-worker",
            daemon=True,
        )
        _worker_thread.start()
        _started = True


def stop_ws_server(join_timeout_sec: float = 2.0) -> None:
    """Flush remaining messages/evolution state and stop the Nim Mummy WebSocket server."""
    global _worker_thread, _started, _persist_to_db
    with _bridge_lock:
        if not _started:
            return
        persist = _persist_to_db
        _persist_to_db = False
        _stop_event.set()
        thread = _worker_thread
        _worker_thread = None
        _started = False

    if thread is not None:
        thread.join(timeout=join_timeout_sec)
    try:
        flush_pending_messages()
        if persist:
            flush_room_states()
    finally:
        mummy_ws.stop_server()


atexit.register(stop_ws_server)
