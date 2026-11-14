import atexit
import json
import os
import threading

import nimporter_plus  # noqa: F401 - registers Nim import hook
from django.db import close_old_connections

from direct_evolution_together import mummy_ws
from direct_evolution_together.cppn import (
    create_random_population,
    evolve_population,
    resize_population,
    sync_genome_counter,
)
from direct_evolution_together.models import ChatMessage, RoomEvolutionState

HISTORY_LIMIT = 100
DEFAULT_GOALS = ("Butterfly", "Galaxy", "Creature")

_bridge_lock = threading.Lock()
_stop_event = threading.Event()
_worker_thread: threading.Thread | None = None
_started = False
_persist_to_db = False


def _build_goals_payload(
    goal_names: tuple[str, ...] = DEFAULT_GOALS,
    pop_size: int = 30,
) -> str:
    return json.dumps(
        [
            {"name": name, "genomes": create_random_population(pop_size=pop_size)}
            for name in goal_names
        ]
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


def seed_from_db() -> None:
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
        max_id = max(max_id, _max_genome_id_in_state(row.state_json))
        mummy_ws.restore_room_state(row.room, row.state_json)
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
        pop_size = int(job.get("populationSize") or 30)
        if kind == "init_room":
            initial_goal = (job.get("goal") or "").strip()
            if initial_goal:
                goal_names: tuple[str, ...] = (initial_goal,)
            elif room != "lobby":
                goal_names = ()
            else:
                goal_names = DEFAULT_GOALS
            mummy_ws.apply_room_init(
                room, _build_goals_payload(goal_names, pop_size=pop_size)
            )
        elif kind == "create":
            genomes = create_random_population(pop_size=pop_size)
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
    max_votes: int = 20,
    mut_sigma: float = 0.30,
    elite_frac: float = 0.20,
    crossover_rate: float = 0.75,
    admin_password: str | None = None,
) -> None:
    """Start the Nim Mummy WebSocket server and background PyTorch/ORM bridge loop."""
    global _worker_thread, _started, _persist_to_db
    with _bridge_lock:
        if _started:
            return
        _persist_to_db = bool(seed)
        resolved_admin_pw = os.getenv("DET_ADMIN_PASSWORD", "") if admin_password is None else admin_password
        mummy_ws.set_admin_password(resolved_admin_pw)
        if min_votes is not None:
            mummy_ws.set_round_config(int(min_votes), 8000)
        mummy_ws.set_default_population_size(int(pop_size))
        mummy_ws.set_default_max_votes(int(max_votes))
        mummy_ws.set_default_ga_params(float(mut_sigma), float(elite_frac), float(crossover_rate))
        if seed:
            seed_from_db()
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
