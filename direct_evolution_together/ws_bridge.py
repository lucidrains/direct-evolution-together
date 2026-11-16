import atexit
import json
import os
import threading

import nimporter_plus  # noqa: F401 - registers Nim import hook
from django.db import close_old_connections

from direct_evolution_together import mummy_ws
from direct_evolution_together.cppn import (
    evolve_population,
    resize_population,
    sync_genome_counter,
)
from direct_evolution_together.environments import (
    clamp_pop_for_task,
    create_task_population,
    default_elite_for_task,
    default_pop_for_task,
    evaluate_control_genome,
    get_task_spec,
    normalize_task_id,
)
from direct_evolution_together.models import ChatMessage, RoomEvolutionState

HISTORY_LIMIT = 100
DEFAULT_GOAL_SPECS: tuple[tuple[str, str, int | None], ...] = (
    ("CartPole", "cartpole", 30),
    ("Butterfly", "image", None),
    ("Galaxy", "image", None),
    ("Creature", "image", None),
)
DEFAULT_GOALS = tuple(name for name, _, _ in DEFAULT_GOAL_SPECS)

_bridge_lock = threading.Lock()
_stop_event = threading.Event()
_worker_thread: threading.Thread | None = None
_started = False
_persist_to_db = False


def _build_lineage_from_genomes(genomes: list[dict], gen: int = 0) -> list[dict]:
    return [
        {
            "id": int(g["id"]),
            "gen": gen,
            "parents": [int(p) for p in g.get("parents", [])],
            "op": str(g.get("op", "random")),
            "weights": g["weights"],
            "biases": g["biases"],
            "activations": g["activations"],
        }
        for g in genomes
    ]


def _build_goal_entry(
    name: str,
    task: str,
    pop_size: int,
) -> dict:
    clean_task = normalize_task_id(task)
    spec = get_task_spec(clean_task)
    eff_pop = clamp_pop_for_task(clean_task, pop_size)
    entry: dict = {
        "name": name,
        "task": clean_task,
        "minVotes": spec.default_min_votes,
        "genomes": create_task_population(clean_task, eff_pop),
    }
    if spec.default_elite_frac is not None:
        entry["eliteFrac"] = float(spec.default_elite_frac)
    return entry


def _build_goals_payload(
    goal_specs: tuple[tuple[str, str, int | None], ...] | tuple[str, ...] = DEFAULT_GOAL_SPECS,
    pop_size: int = 30,
) -> str:
    entries: list[dict] = []
    for item in goal_specs:
        if isinstance(item, tuple):
            name, task, cap_pop = item
            target_pop = min(pop_size, cap_pop) if cap_pop is not None else pop_size
        else:
            name, task = str(item), "image"
            target_pop = pop_size
        entries.append(_build_goal_entry(name, task, target_pop))
    return json.dumps(entries)


def build_room_state_json(
    goal_name: str,
    genomes: list[dict],
    *,
    task: str = "image",
    min_votes: int | None = None,
    max_votes_per_user: int = 20,
    mut_sigma: float = 0.30,
    elite_frac: float | None = None,
    crossover_rate: float = 0.75,
) -> str:
    """Serialize initial Generation-0 room state matching Mummy's RoomEvolutionState schema."""
    clean_task = normalize_task_id(task)
    spec = get_task_spec(clean_task)
    eff_min_votes = spec.default_min_votes if min_votes is None else int(min_votes)
    if spec.kind == "control":
        eff_min_votes = max(1, min(eff_min_votes, int(max_votes_per_user)))
    eff_elite_frac = (
        default_elite_for_task(clean_task, 0.20)
        if elite_frac is None
        else max(0.0, min(0.5, float(elite_frac)))
    )
    params = {
        "maxVotesPerUser": int(max_votes_per_user),
        "minVotes": int(eff_min_votes),
        "mutSigma": float(mut_sigma),
        "eliteFrac": float(eff_elite_frac),
        "crossoverRate": float(crossover_rate),
    }
    lineage = _build_lineage_from_genomes(genomes, gen=0)
    return json.dumps(
        {
            "defaultParams": params,
            "goals": [
                {
                    "name": goal_name,
                    "task": clean_task,
                    "generation": 0,
                    "genomes": genomes,
                    "lineage": lineage,
                    "params": params,
                }
            ],
        }
    )


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
    default_pop_size: int = 30,
    default_max_votes: int | None = None,
) -> tuple[str, bool]:
    """Ensure newly added default lobby goals are seeded and vote caps match runtime config."""
    try:
        parsed = json.loads(state_json)
    except ValueError:
        return state_json, False
    if not isinstance(parsed, dict) or not isinstance(parsed.get("goals"), list):
        return state_json, False

    existing = {str(g.get("name", "")) for g in parsed["goals"] if isinstance(g, dict)}
    base_params = parsed.get("defaultParams") if isinstance(parsed.get("defaultParams"), dict) else {}
    changed = False
    if (
        default_max_votes is not None
        and isinstance(base_params, dict)
        and int(base_params.get("maxVotesPerUser", 0)) < int(default_max_votes)
    ):
        base_params["maxVotesPerUser"] = int(default_max_votes)
        changed = True
    for g in parsed["goals"]:
        if not isinstance(g, dict):
            continue
        p = g.get("params")
        if (
            default_max_votes is not None
            and isinstance(p, dict)
            and int(p.get("maxVotesPerUser", 0)) < int(default_max_votes)
        ):
            p["maxVotesPerUser"] = int(default_max_votes)
            changed = True
        if g.get("task") == "cartpole" and int(g.get("generation", 0)) == 0:
            cp_spec = get_task_spec("cartpole")
            p = g.get("params")
            if isinstance(p, dict):
                cp_max_votes = max(
                    1,
                    int(p.get("maxVotesPerUser", base_params.get("maxVotesPerUser", 20))),
                )
                cp_min_votes = max(1, min(cp_spec.default_min_votes, cp_max_votes))
                if int(p.get("minVotes", 0)) != cp_min_votes:
                    p["minVotes"] = cp_min_votes
                    changed = True
                if float(p.get("eliteFrac", 0.0)) != 0.0:
                    p["eliteFrac"] = 0.0
                    changed = True
            cur_genomes = g.get("genomes")
            if isinstance(cur_genomes, list) and cur_genomes:
                target_pop = clamp_pop_for_task(
                    "cartpole",
                    min(default_pop_size, cp_spec.default_pop_size),
                )
                needs_resize = len(cur_genomes) != target_pop
                needs_refresh = any(
                    evaluate_control_genome("cartpole", cg, seeds=cp_spec.eval_seeds)["meanSteps"] >= 60.0
                    for cg in cur_genomes
                )
                if needs_resize or needs_refresh:
                    fresh = create_task_population(
                        "cartpole",
                        target_pop if needs_resize else len(cur_genomes),
                    )
                    g["genomes"] = fresh
                    g["lineage"] = _build_lineage_from_genomes(fresh, gen=0)
                    changed = True
    added: list[dict] = []
    default_goal_max_votes = max(
        1,
        int(base_params.get("maxVotesPerUser", default_max_votes or 20)),
    )
    for name, task, cap_pop in DEFAULT_GOAL_SPECS:
        if name in existing:
            continue
        spec = get_task_spec(task)
        t_pop = min(default_pop_size, cap_pop) if cap_pop is not None else default_pop_size
        genomes = create_task_population(task, t_pop)
        params = {
            "maxVotesPerUser": default_goal_max_votes,
            "minVotes": max(0, min(int(spec.default_min_votes), default_goal_max_votes)),
            "mutSigma": float(base_params.get("mutSigma", 0.30)),
            "eliteFrac": default_elite_for_task(task, float(base_params.get("eliteFrac", 0.20))),
            "crossoverRate": float(base_params.get("crossoverRate", 0.75)),
        }
        added.append(
            {
                "name": name,
                "task": task,
                "generation": 0,
                "genomes": genomes,
                "lineage": _build_lineage_from_genomes(genomes, gen=0),
                "params": params,
            }
        )
    if not added and not changed:
        return state_json, False

    if added:
        # Place newly added control defaults at the front so visitors immediately see them.
        parsed["goals"] = added + parsed["goals"]
    return json.dumps(parsed), True


def seed_from_db(default_pop_size: int = 30, default_max_votes: int | None = None) -> None:
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
        state_json = row.state_json
        max_id = max(max_id, _max_genome_id_in_state(state_json))
        sync_genome_counter(max_id)
        if row.room == "lobby":
            updated_json, changed = _ensure_lobby_default_goals(
                state_json,
                default_pop_size=default_pop_size,
                default_max_votes=default_max_votes,
            )
            if changed:
                state_json = updated_json
                row.state_json = state_json
                row.save(update_fields=["state_json", "updated_at"])
                max_id = max(max_id, _max_genome_id_in_state(state_json))
        mummy_ws.restore_room_state(row.room, state_json)
    sync_genome_counter(max_id)


def flush_room_states() -> int:
    """Drain dirty room evolution snapshots from Mummy and persist them via Django ORM."""
    raw = mummy_ws.drain_dirty_room_states()
    items = json.loads(raw)
    if not items:
        return 0

    close_old_connections()
    for item in items:
        RoomEvolutionState.objects.update_or_create(
            room=item["room"],
            defaults={"state_json": item["stateJson"]},
        )
    return len(items)


def flush_pending_messages() -> int:
    """Drain queued chat messages from Mummy and persist them via Django ORM."""
    raw = mummy_ws.drain_pending_messages()
    items = json.loads(raw)
    if not items:
        return 0

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
    jobs = json.loads(raw)
    if not jobs:
        return 0

    for job in jobs:
        kind = job["kind"]
        room = job["room"]
        task = normalize_task_id(job.get("task", "image"))
        pop_size = clamp_pop_for_task(task, int(job.get("populationSize") or default_pop_for_task(task)))
        if kind == "init_room":
            if _persist_to_db:
                close_old_connections()
                saved_row = RoomEvolutionState.objects.filter(room=room).first()
                if saved_row is not None:
                    sync_genome_counter(_max_genome_id_in_state(saved_row.state_json))
                    mummy_ws.restore_room_state(room, saved_row.state_json)
                    continue
            initial_goal = (job.get("goal") or "").strip()
            if initial_goal:
                goal_specs: tuple[tuple[str, str, int | None], ...] = ((initial_goal, task, pop_size),)
            elif room != "lobby":
                goal_specs = ()
            else:
                goal_specs = DEFAULT_GOAL_SPECS
            mummy_ws.apply_room_init(
                room, _build_goals_payload(goal_specs, pop_size=pop_size)
            )
        elif kind == "create":
            genomes = create_task_population(task, pop_size)
            mummy_ws.apply_goal_create(
                room, job["goal"], job["username"], json.dumps(genomes)
            )
        elif kind == "evolve":
            next_genomes, top_score = evolve_population(
                job["genomes"],
                job["scores"],
                job["upCounts"],
                mut_sigma=float(job.get("mutSigma", 0.30)),
                pop_size=pop_size,
                elite_frac=float(job.get("eliteFrac", 0.20)),
                crossover_rate=float(job.get("crossoverRate", 0.75)),
            )
            mummy_ws.apply_goal_evolve(
                room, job["goal"], json.dumps(next_genomes), int(top_score)
            )
        elif kind == "resize":
            resized = resize_population(
                job["genomes"],
                pop_size,
                job["scores"],
                job["upCounts"],
                mut_sigma=float(job.get("mutSigma", 0.30)),
            )
            mummy_ws.apply_goal_resize(
                room, job["goal"], job["username"], json.dumps(resized)
            )
    return len(jobs)


def _worker_loop() -> None:
    while not _stop_event.wait(0.02):
        try:
            flush_evo_jobs()
        except Exception:
            pass
        try:
            flush_pending_messages()
        except Exception:
            pass
        if _persist_to_db:
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
    pop_size: int = 30,
    min_votes: int | None = None,
    min_votes_override: bool = False,
    countdown_ms: int = 8000,
    max_votes: int = 20,
    mut_sigma: float = 0.30,
    elite_frac: float = 0.20,
    crossover_rate: float = 0.75,
    admin_password: str | None = None,
    evolve_token: str | None = None,
) -> None:
    """Start the Nim Mummy WebSocket server and background PyTorch/ORM bridge loop."""
    global _worker_thread, _started, _persist_to_db
    with _bridge_lock:
        if _started:
            return
        _persist_to_db = bool(seed)
        resolved_admin_pw = os.getenv("DET_ADMIN_PASSWORD", "") if admin_password is None else admin_password
        mummy_ws.set_admin_password(resolved_admin_pw)
        mummy_ws.set_evolve_token(evolve_token or "")
        if min_votes is not None or countdown_ms != 8000:
            mummy_ws.set_round_config(
                0 if min_votes is None else int(min_votes),
                int(countdown_ms),
                bool(min_votes_override),
            )
        mummy_ws.set_default_population_size(int(pop_size))
        mummy_ws.set_default_max_votes(int(max_votes))
        mummy_ws.set_default_ga_params(float(mut_sigma), float(elite_frac), float(crossover_rate))
        if seed:
            seed_from_db(default_pop_size=int(pop_size), default_max_votes=int(max_votes))
        if not mummy_ws.has_room_evolution("lobby"):
            mummy_ws.seed_room_goals("lobby", _build_goals_payload(pop_size=int(pop_size)))
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


def stop_ws_server() -> None:
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
        thread.join(timeout=2.0)
    try:
        flush_pending_messages()
        if persist:
            flush_room_states()
    finally:
        mummy_ws.stop_server()


atexit.register(stop_ws_server)
