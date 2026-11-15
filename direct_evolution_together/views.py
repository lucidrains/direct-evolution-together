import base64
import hmac
import io
import json
import os
import secrets
import urllib.error
import urllib.parse
import urllib.request
import zipfile

from django.conf import settings
from django.core.exceptions import RequestDataTooBig, TooManyFilesSent
from django.http import HttpRequest, HttpResponse, HttpResponseBadRequest, JsonResponse
from django.shortcuts import render
from django.utils.text import slugify
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST, require_http_methods

from direct_evolution_together import mummy_ws, ws_bridge
from direct_evolution_together.cppn import (
    MAX_ARCHIVE_BYTES,
    MAX_CHECKPOINT_BYTES,
    MAX_POPULATION_SIZE,
    MAX_TOTAL_UPLOAD_BYTES,
    MAX_UPLOAD_FILES,
    MIN_POPULATION_SIZE,
    build_runner_script,
    create_random_population,
    create_seeded_population,
    parse_uploaded_strain_bytes,
    save_cppn_bytes,
    slugify_policy_name,
)
from direct_evolution_together.models import Room, RoomEvolutionState


def _resolve_base_path(request: HttpRequest) -> str:
    configured = getattr(settings, "SUBPATH", "")
    if configured:
        return configured
    forwarded = request.META.get("HTTP_X_FORWARDED_PREFIX", "").strip().strip("/")
    return f"/{forwarded}" if forwarded else ""


def _room_slug(name: str) -> str:
    base = slugify(name)[:48].strip("-") or f"p-{secrets.token_hex(4)}"
    if base == "lobby":
        base = "lobby-2"
    slug = base
    suffix = 2
    while Room.objects.filter(slug=slug).exists():
        slug = f"{base}-{suffix}"
        suffix += 1
    return slug


def index(request: HttpRequest, room: str = "lobby") -> HttpResponse:
    """Serve the minimal collaborative evolution and chat page."""
    if not room or len(room) > 64:
        return HttpResponseBadRequest("Invalid room name")
    room_title = "Lobby" if room == "lobby" else room
    room_description = ""
    if room != "lobby":
        meta = Room.objects.filter(slug=room).values("name", "description").first()
        if meta:
            room_title = meta["name"]
            room_description = meta["description"]
    initial_goal = request.GET.get("goal", "").strip()[:32]
    initial_pop = settings.POPULATION_SIZE
    raw_pop = request.GET.get("pop", "").strip()
    if raw_pop:
        try:
            initial_pop = max(3, min(64, int(raw_pop)))
        except ValueError:
            initial_pop = settings.POPULATION_SIZE
    initial_max_votes = settings.MAX_VOTES_PER_USER
    raw_max_votes = request.GET.get("max_votes", "").strip()
    if raw_max_votes:
        try:
            initial_max_votes = max(1, min(64, int(raw_max_votes)))
        except ValueError:
            initial_max_votes = settings.MAX_VOTES_PER_USER
    initial_mut_sigma = settings.MUTATION_SIGMA
    raw_sigma = request.GET.get("sigma", "").strip()
    if raw_sigma:
        try:
            initial_mut_sigma = max(0.05, min(1.0, float(raw_sigma)))
        except ValueError:
            initial_mut_sigma = settings.MUTATION_SIGMA
    initial_elite_frac = settings.ELITE_FRAC
    raw_elite = request.GET.get("elite_frac", "").strip()
    if raw_elite:
        try:
            initial_elite_frac = max(0.0, min(0.5, float(raw_elite)))
        except ValueError:
            initial_elite_frac = settings.ELITE_FRAC
    initial_crossover_rate = settings.CROSSOVER_RATE
    raw_crossover = request.GET.get("crossover", "").strip()
    if raw_crossover:
        try:
            initial_crossover_rate = max(0.0, min(1.0, float(raw_crossover)))
        except ValueError:
            initial_crossover_rate = settings.CROSSOVER_RATE
    base_path = _resolve_base_path(request)
    ws_via_proxy = bool(getattr(settings, "WS_VIA_PROXY", False) or base_path)
    return render(
        request,
        "direct_evolution_together/index.html",
        {
            "base_path": base_path,
            "ws_via_proxy": ws_via_proxy,
            "ws_host": settings.WS_HOST,
            "ws_port": settings.WS_PORT,
            "room_name": room,
            "room_title": room_title,
            "room_description": room_description,
            "is_private": room != "lobby",
            "initial_goal": initial_goal,
            "initial_pop": initial_pop,
            "initial_max_votes": initial_max_votes,
            "initial_mut_sigma": f"{initial_mut_sigma:.2f}",
            "initial_elite_frac": f"{initial_elite_frac:.2f}",
            "initial_crossover_rate": f"{initial_crossover_rate:.2f}",
        },
    )


def health(request: HttpRequest) -> JsonResponse:
    """Basic health check endpoint."""
    return JsonResponse({"status": "ok"})


def _truncate_utf8(text: str, max_bytes: int = 32) -> str:
    cleaned = (text or "").strip()
    encoded = cleaned.encode("utf-8")
    if len(encoded) <= max_bytes:
        return cleaned
    return encoded[:max_bytes].decode("utf-8", errors="ignore").strip()


def _parse_clamped_float_param(raw: object, fallback: float, lo: float, hi: float) -> float:
    if raw is None or str(raw).strip() == "":
        return float(fallback)
    try:
        return max(lo, min(hi, float(raw)))
    except (TypeError, ValueError):
        return float(fallback)


def _parse_clamped_int_param(raw: object, fallback: int, lo: int, hi: int) -> int:
    if raw is None or str(raw).strip() == "":
        return int(fallback)
    try:
        return max(lo, min(hi, int(raw)))
    except (TypeError, ValueError):
        return int(fallback)


def _too_large_error() -> str:
    max_mb = MAX_TOTAL_UPLOAD_BYTES // (1024 * 1024)
    return f"Upload payload exceeds the {max_mb} MB maximum request size."


def _check_request_content_length(request: HttpRequest) -> str | None:
    """Reject oversized HTTP bodies before Django's multipart parser streams them."""
    raw_len = request.META.get("CONTENT_LENGTH")
    if not raw_len:
        return None
    try:
        content_len = int(raw_len)
    except (TypeError, ValueError):
        return None
    return _too_large_error() if content_len > MAX_TOTAL_UPLOAD_BYTES else None


def _parse_multipart_upload(request: HttpRequest) -> tuple[dict, list, tuple[int, str] | None]:
    """Safely extract POST metadata and uploaded files under upload caps."""
    if size_err := _check_request_content_length(request):
        return {}, [], (413, size_err)
    try:
        post_data = request.POST
        files = request.FILES.getlist("weights") or request.FILES.getlist("files")
    except TooManyFilesSent:
        return {}, [], (400, f"Too many uploaded files (max {MAX_UPLOAD_FILES}).")
    except RequestDataTooBig:
        return {}, [], (413, _too_large_error())
    if len(files) > MAX_UPLOAD_FILES:
        return {}, [], (400, f"Too many uploaded files (max {MAX_UPLOAD_FILES}).")
    return post_data, files, None


def _read_uploaded_file_bounded(up: object, total_remaining: int) -> bytes:
    """Read an uploaded file with strict per-file and cumulative byte caps."""
    filename = str(getattr(up, "name", "") or "upload")
    is_zip = filename.lower().endswith(".zip")
    per_file_cap = MAX_ARCHIVE_BYTES if is_zip else MAX_CHECKPOINT_BYTES
    limit_label = f"{MAX_ARCHIVE_BYTES // (1024 * 1024)} MB" if is_zip else f"{MAX_CHECKPOINT_BYTES // 1024} KB"
    max_mb = MAX_TOTAL_UPLOAD_BYTES // (1024 * 1024)

    def _verify_size(n: int) -> None:
        if n > per_file_cap:
            raise ValueError(f"{filename}: File exceeds the {limit_label} size limit.")
        if n > total_remaining:
            raise ValueError(f"Combined uploaded files exceed the {max_mb} MB limit.")

    if isinstance(reported := getattr(up, "size", None), int):
        _verify_size(reported)

    data = getattr(up, "read")(max(0, min(per_file_cap, total_remaining)) + 1)
    _verify_size(len(data))
    return data


def _extract_uploaded_strains(
    uploaded_files: list,
    remaining_slots: int,
) -> list[tuple[str, dict, bytes]]:
    """Read, bound-check, and parse uploaded `.pt`/`.zip` strain files."""
    strains: list[tuple[str, dict, bytes]] = []
    bytes_remaining = MAX_TOTAL_UPLOAD_BYTES
    for up in uploaded_files:
        raw_bytes = _read_uploaded_file_bounded(up, bytes_remaining)
        bytes_remaining -= len(raw_bytes)
        parsed = parse_uploaded_strain_bytes(
            up.name,
            raw_bytes,
            assign_ids=False,
            max_strains=remaining_slots,
        )
        remaining_slots -= len(parsed)
        strains.extend(parsed)
    return strains


@require_POST
def api_validate_weights(request: HttpRequest) -> JsonResponse:
    """Validate uploaded `.pt` or `.zip` CPPN checkpoints and return preview genomes."""
    post_data, uploaded_files, err = _parse_multipart_upload(request)
    if err:
        return JsonResponse({"ok": False, "error": err[1]}, status=err[0])
    if not uploaded_files:
        return JsonResponse(
            {"ok": False, "error": "Please select at least one .pt or .zip weight file."},
            status=400,
        )

    existing_count = _parse_clamped_int_param(
        post_data.get("existing_count"), 0, 0, MAX_POPULATION_SIZE
    )
    remaining_slots = MAX_POPULATION_SIZE - existing_count
    if remaining_slots < 1:
        return JsonResponse(
            {
                "ok": False,
                "error": (
                    f"Cannot exceed {MAX_POPULATION_SIZE} uploaded strains "
                    f"(max population size is {MAX_POPULATION_SIZE})."
                ),
            },
            status=400,
        )

    try:
        parsed = _extract_uploaded_strains(uploaded_files, remaining_slots)
    except ValueError as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)

    strains = [
        {
            "label": label,
            "genome": genome,
            "pt_b64": base64.b64encode(pt_bytes).decode("ascii"),
        }
        for label, genome, pt_bytes in parsed
    ]
    return JsonResponse({"ok": True, "strains": strains})


@require_POST
def api_create_room(request: HttpRequest) -> JsonResponse:
    """Create a named private room, optionally reseeding Gen 0 with uploaded CPPN weights."""
    is_multipart = (request.content_type or "").lower().startswith("multipart/form-data")
    if is_multipart:
        payload, uploaded_files, err = _parse_multipart_upload(request)
        if err:
            return JsonResponse({"error": err[1]}, status=err[0])
    else:
        if size_err := _check_request_content_length(request):
            return JsonResponse({"error": size_err}, status=413)
        try:
            payload = json.loads(request.body.decode("utf-8"))
        except RequestDataTooBig:
            return JsonResponse({"error": _too_large_error()}, status=413)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return JsonResponse({"error": "Invalid JSON."}, status=400)
        if not isinstance(payload, dict):
            return JsonResponse({"error": "Invalid room payload."}, status=400)
        uploaded_files = []

    name = str(payload.get("name") or "").strip()[:64]
    description = str(payload.get("description") or "").strip()[:280]
    if not name:
        return JsonResponse({"error": "Room name is required."}, status=400)

    raw_pop = payload.get("population_size")
    if raw_pop is not None and str(raw_pop).strip() != "":
        try:
            pop_size = int(str(raw_pop).strip())
        except ValueError:
            return JsonResponse(
                {
                    "error": (
                        f"Population size must be an integer between "
                        f"{MIN_POPULATION_SIZE} and {MAX_POPULATION_SIZE}."
                    )
                },
                status=400,
            )
        if not (MIN_POPULATION_SIZE <= pop_size <= MAX_POPULATION_SIZE):
            return JsonResponse(
                {
                    "error": (
                        f"Population size must be between "
                        f"{MIN_POPULATION_SIZE} and {MAX_POPULATION_SIZE}."
                    )
                },
                status=400,
            )
    else:
        pop_size = int(settings.POPULATION_SIZE)

    base_path = _resolve_base_path(request)
    seeded_meta: dict | None = None

    if uploaded_files:
        try:
            seed_genomes = [
                genome
                for _, genome, _ in _extract_uploaded_strains(uploaded_files, MAX_POPULATION_SIZE)
            ]
        except ValueError as exc:
            return JsonResponse({"error": str(exc)}, status=400)

        if len(seed_genomes) > pop_size:
            return JsonResponse(
                {
                    "error": (
                        f"Uploaded strains ({len(seed_genomes)}) cannot exceed "
                        f"total population size ({pop_size})."
                    )
                },
                status=400,
            )

        try:
            genomes = create_seeded_population(seed_genomes, pop_size=pop_size)
        except ValueError as exc:
            return JsonResponse({"error": str(exc)}, status=400)

        seeded_meta = {
            "genomes": genomes,
            "seeded": len(seed_genomes),
            "max_votes": _parse_clamped_int_param(
                payload.get("max_votes_per_user"), settings.MAX_VOTES_PER_USER, 1, 64
            ),
            "mut_sigma": _parse_clamped_float_param(
                payload.get("mutation_sigma"), settings.MUTATION_SIGMA, 0.05, 1.0
            ),
            "elite_frac": _parse_clamped_float_param(
                payload.get("elite_frac"), settings.ELITE_FRAC, 0.0, 0.5
            ),
            "crossover_rate": _parse_clamped_float_param(
                payload.get("crossover_rate"), settings.CROSSOVER_RATE, 0.0, 1.0
            ),
        }

    room = Room.objects.create(
        slug=_room_slug(name),
        name=name,
        description=description,
    )
    resp_data: dict[str, object] = {
        "ok": True,
        "slug": room.slug,
        "name": room.name,
        "description": room.description,
        "url": f"{base_path}/p/{room.slug}/",
    }

    if seeded_meta is not None:
        raw_goal = str(payload.get("goal") or "").strip()
        goal_name = _truncate_utf8(raw_goal or name, 32) or room.slug[:32]
        state_json = ws_bridge.build_room_state_json(
            goal_name,
            seeded_meta["genomes"],
            max_votes_per_user=seeded_meta["max_votes"],
            mut_sigma=seeded_meta["mut_sigma"],
            elite_frac=seeded_meta["elite_frac"],
            crossover_rate=seeded_meta["crossover_rate"],
        )
        RoomEvolutionState.objects.update_or_create(
            room=room.slug,
            defaults={"state_json": state_json},
        )
        if ws_bridge.is_running():
            mummy_ws.restore_room_state(room.slug, state_json)
        resp_data.update(
            {
                "seeded": seeded_meta["seeded"],
                "population_size": pop_size,
                "goal": goal_name,
            }
        )

    return JsonResponse(resp_data)


def _parse_export_payload(request: HttpRequest) -> tuple[dict, str]:
    payload = json.loads(request.body.decode("utf-8"))
    genome = payload.get("genome")
    if not isinstance(genome, dict) or not {"weights", "biases", "activations"} <= set(genome):
        raise ValueError("Invalid genome payload")
    genome = dict(genome)
    genome.setdefault("id", 0)
    goal = str(payload.get("goal", "policy"))
    generation = int(payload.get("generation", 0))
    index = int(payload.get("index", 0))
    return genome, slugify_policy_name(goal, generation, index)


@require_POST
def export_weights(request: HttpRequest) -> HttpResponse:
    """Download a trained/evolved CPPN genome as a save_load .pt checkpoint."""
    try:
        genome, slug = _parse_export_payload(request)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return HttpResponseBadRequest(str(exc))

    data = save_cppn_bytes(genome)
    resp = HttpResponse(data, content_type="application/octet-stream")
    resp["Content-Disposition"] = f'attachment; filename="{slug}.pt"'
    return resp


@require_GET
def export_script(request: HttpRequest) -> HttpResponse:
    """Download the <=10 LOC uv runner script for CPPN checkpoints."""
    script = build_runner_script("weights.pt")
    resp = HttpResponse(script, content_type="text/x-python; charset=utf-8")
    resp["Content-Disposition"] = 'attachment; filename="run.py"'
    return resp


@require_POST
def export_bundle(request: HttpRequest) -> HttpResponse:
    """Download a zip containing weights.pt and the <=10 LOC uv runner script."""
    try:
        genome, slug = _parse_export_payload(request)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return HttpResponseBadRequest(str(exc))

    weights_bytes = save_cppn_bytes(genome)
    script_text = build_runner_script("weights.pt")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{slug}/weights.pt", weights_bytes)
        zf.writestr(f"{slug}/run.py", script_text)

    resp = HttpResponse(buf.getvalue(), content_type="application/zip")
    resp["Content-Disposition"] = f'attachment; filename="{slug}.zip"'
    return resp


def _configured_admin_password() -> str:
    return os.getenv("DET_ADMIN_PASSWORD", getattr(settings, "ADMIN_PASSWORD", ""))


def _verify_admin_password(submitted: str) -> tuple[bool, int, str]:
    configured = _configured_admin_password()
    if not configured:
        return False, 503, "Admin reset is disabled: set DET_ADMIN_PASSWORD in the environment or .env."
    if not submitted or not hmac.compare_digest(submitted, configured):
        return False, 403, "Invalid admin password."
    return True, 200, ""


def _ws_internal_request(
    method: str,
    path: str,
    password: str,
    body: dict | None = None,
) -> tuple[int, str]:
    ws_host = "127.0.0.1" if settings.WS_HOST == "0.0.0.0" else settings.WS_HOST
    url = f"http://{ws_host}:{settings.WS_PORT}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"X-Admin-Password": password}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        return 502, f"WebSocket server unreachable: {exc.reason}"


def _fetch_room_pop_sizes(room: str, configured: str) -> tuple[int, dict[str, int], str]:
    if ws_bridge.is_running():
        return 200, json.loads(mummy_ws.get_room_goal_pop_sizes(room)), ""
    status, text = _ws_internal_request(
        "GET",
        f"/internal/room-goals/{urllib.parse.quote(room, safe='')}",
        configured,
    )
    if status != 200:
        return status, {}, text or "Failed to query room goals."
    return 200, json.loads(text or "{}"), ""


def _apply_goal_reset_dispatch(room: str, goal_name: str, genomes: list[dict], configured: str) -> tuple[int, str]:
    genomes_json = json.dumps(genomes)
    if ws_bridge.is_running():
        mummy_ws.apply_goal_reset(room, goal_name, "admin", genomes_json)
        return 200, ""
    return _ws_internal_request(
        "POST",
        "/internal/goal-reset",
        configured,
        {"room": room, "goal": goal_name, "username": "admin", "genomes": genomes},
    )


def _apply_room_clear_dispatch(room: str, goals_payload: list[dict], goal: str, configured: str) -> tuple[int, str]:
    goals_json = json.dumps(goals_payload)
    if ws_bridge.is_running():
        mummy_ws.clear_room_evolution(room, goals_json, goal, "admin")
        return 200, ""
    return _ws_internal_request(
        "POST",
        "/internal/room-clear",
        configured,
        {"room": room, "goal": goal, "username": "admin", "goals": goals_payload},
    )


def _db_room_goal_names(room: str) -> list[str]:
    row = RoomEvolutionState.objects.filter(room=room).first()
    if row is None:
        return []
    try:
        parsed = json.loads(row.state_json)
    except ValueError:
        return []
    if not isinstance(parsed, dict):
        return []
    return [
        str(g.get("name"))
        for g in parsed.get("goals", [])
        if isinstance(g, dict) and g.get("name")
    ]


def _clear_persisted_room_state(room: str, goal: str) -> None:
    if not goal or goal == "*":
        RoomEvolutionState.objects.filter(room=room).delete()
        return
    row = RoomEvolutionState.objects.filter(room=room).first()
    if row is None:
        return
    try:
        parsed = json.loads(row.state_json)
    except ValueError:
        RoomEvolutionState.objects.filter(room=room).delete()
        return
    if not isinstance(parsed, dict):
        return
    remaining = [
        g for g in parsed.get("goals", []) if isinstance(g, dict) and g.get("name") != goal
    ]
    if not remaining and room != "lobby":
        row.delete()
        return
    parsed["goals"] = remaining
    row.state_json = json.dumps(parsed)
    row.save(update_fields=["state_json", "updated_at"])


def _perform_admin_reset(room: str, goal: str, configured: str) -> tuple[bool, int, str, list[str]]:
    room = (room or "lobby").strip() or "lobby"
    goal = (goal or "").strip()
    status, pop_map, err = _fetch_room_pop_sizes(room, configured)
    if status != 200:
        return False, status, err, []
    if not pop_map:
        return False, 404, f"Room '{room}' has no initialized goals.", []

    if not goal or goal == "*":
        targets = list(pop_map.items())
    elif goal in pop_map:
        targets = [(goal, pop_map[goal])]
    else:
        available = ", ".join(pop_map.keys())
        return False, 404, f"Goal '{goal}' not found in room '{room}' (available: {available}).", []

    reset_goals: list[str] = []
    for goal_name, pop_size in targets:
        genomes = create_random_population(pop_size=int(pop_size))
        r_status, r_err = _apply_goal_reset_dispatch(room, goal_name, genomes, configured)
        if r_status != 200:
            return False, r_status, r_err or "Failed to reset goal.", reset_goals
        reset_goals.append(goal_name)
    return True, 200, f"Reset {', '.join(reset_goals)} in '{room}' to gen 0.", reset_goals


def _perform_admin_clear(room: str, goal: str, configured: str) -> tuple[bool, int, str, list[str]]:
    room = (room or "lobby").strip() or "lobby"
    goal = (goal or "").strip()
    is_all = not goal or goal == "*"

    status, pop_map, err = _fetch_room_pop_sizes(room, configured)
    if status != 200:
        return False, status, err, []

    if room == "lobby":
        known_goals = list(pop_map.keys()) if pop_map else list(ws_bridge.DEFAULT_GOALS)
        if is_all:
            goals_payload = [
                {
                    "name": g_name,
                    "genomes": create_random_population(
                        pop_size=int(pop_map.get(g_name, settings.POPULATION_SIZE))
                    ),
                }
                for g_name in known_goals
            ]
            _clear_persisted_room_state(room, "")
            c_status, c_err = _apply_room_clear_dispatch(room, goals_payload, "", configured)
            if c_status != 200:
                return False, c_status, c_err or "Failed to clear lobby room state.", []
            return True, 200, f"Cleared {', '.join(known_goals)} in '{room}' to fresh gen 0.", known_goals

        if goal not in known_goals:
            available = ", ".join(known_goals)
            return False, 404, f"Goal '{goal}' not found in room '{room}' (available: {available}).", []
        genomes = create_random_population(
            pop_size=int(pop_map.get(goal, settings.POPULATION_SIZE))
        )
        _clear_persisted_room_state(room, goal)
        c_status, c_err = _apply_room_clear_dispatch(room, genomes, goal, configured)
        if c_status != 200:
            return False, c_status, c_err or "Failed to clear lobby goal.", []
        return True, 200, f"Cleared '{goal}' in '{room}' to fresh gen 0.", [goal]

    db_goals = _db_room_goal_names(room)
    all_goals = list(dict.fromkeys(list(pop_map.keys()) + db_goals))
    if not is_all and goal not in all_goals:
        if all_goals:
            return False, 404, f"Goal '{goal}' not found in room '{room}' (available: {', '.join(all_goals)}).", []
        return False, 404, f"Room '{room}' has no initialized or persisted goals.", []

    cleared = list(all_goals) if is_all else [goal]
    _clear_persisted_room_state(room, "" if is_all else goal)
    c_status, c_err = _apply_room_clear_dispatch(room, [], "" if is_all else goal, configured)
    if c_status != 200:
        return False, c_status, c_err or "Failed to clear private room state.", []
    if is_all:
        label = ", ".join(cleared) if cleared else "(empty)"
        return True, 200, f"Cleared evolution state for room '{room}' ({label}).", cleared
    return True, 200, f"Cleared goal '{goal}' from room '{room}'.", cleared


def _collect_room_summaries() -> list[dict]:
    meta_by_slug = {r.slug: r.name for r in Room.objects.all()}
    saved_by_room: dict[str, list[dict]] = {}
    for row in RoomEvolutionState.objects.order_by("room"):
        try:
            parsed = json.loads(row.state_json)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            saved_by_room[row.room] = [
                {"name": str(g.get("name", "")), "gen": int(g.get("generation", 0))}
                for g in parsed.get("goals", [])
                if isinstance(g, dict) and g.get("name")
            ]

    ordered_rooms = list(dict.fromkeys(["lobby"] + list(meta_by_slug.keys()) + list(saved_by_room.keys())))
    summaries = []
    for slug in ordered_rooms:
        goals = saved_by_room.get(slug, [])
        if slug == "lobby" and not goals:
            goals = [{"name": g, "gen": 0} for g in ws_bridge.DEFAULT_GOALS]
        summaries.append({
            "slug": slug,
            "label": "Front Lobby" if slug == "lobby" else meta_by_slug.get(slug, slug),
            "goals": goals,
        })
    return summaries


@require_http_methods(["GET", "POST"])
def admin_reset_page(request: HttpRequest) -> HttpResponse:
    """Unlinked admin-only page for resetting or clearing room evolution state."""
    configured = _configured_admin_password()
    room = "lobby"
    goal = ""
    status_msg = ""
    ok = False
    status_code = 200

    if request.method == "POST":
        room = request.POST.get("room", "lobby").strip() or "lobby"
        goal = request.POST.get("goal", "").strip()
        action = request.POST.get("action", "reset").strip().lower()
        password = request.POST.get("password", "")
        authed, status_code, err = _verify_admin_password(password)
        if not authed:
            status_msg = err
        elif action == "clear":
            ok, status_code, status_msg, _ = _perform_admin_clear(room, goal, configured)
        else:
            ok, status_code, status_msg, _ = _perform_admin_reset(room, goal, configured)

    return render(
        request,
        "direct_evolution_together/admin_reset.html",
        {
            "base_path": _resolve_base_path(request),
            "configured": bool(configured),
            "room": room,
            "goal": goal,
            "status_msg": status_msg,
            "ok": ok,
            "room_summaries": _collect_room_summaries(),
        },
        status=status_code,
    )


def _handle_admin_json_action(request: HttpRequest, default_action: str) -> JsonResponse:
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return JsonResponse({"ok": False, "error": "Invalid JSON body."}, status=400)
    if not isinstance(payload, dict):
        return JsonResponse({"ok": False, "error": "JSON object required."}, status=400)

    auth_header = request.headers.get("Authorization", "")
    bearer = auth_header[7:].strip() if auth_header.startswith("Bearer ") else ""
    password = str(payload.get("password") or bearer or "")
    authed, status_code, err = _verify_admin_password(password)
    if not authed:
        return JsonResponse({"ok": False, "error": err}, status=status_code)

    room = str(payload.get("room") or "lobby")
    goal = str(payload.get("goal") or "")
    action = str(payload.get("action") or default_action).strip().lower()
    if action == "clear":
        ok, status_code, msg, affected = _perform_admin_clear(
            room, goal, _configured_admin_password()
        )
    else:
        ok, status_code, msg, affected = _perform_admin_reset(
            room, goal, _configured_admin_password()
        )
    if not ok:
        return JsonResponse({"ok": False, "error": msg}, status=status_code)
    return JsonResponse({"ok": True, "room": room, "reset": affected, "cleared": affected, "message": msg})


@csrf_exempt
@require_POST
def api_reset(request: HttpRequest) -> JsonResponse:
    """Admin JSON endpoint to reset or clear goal(s)/room state via DET_ADMIN_PASSWORD."""
    return _handle_admin_json_action(request, "reset")


@csrf_exempt
@require_POST
def api_clear(request: HttpRequest) -> JsonResponse:
    """Admin JSON endpoint to clear room/goal evolution state via DET_ADMIN_PASSWORD."""
    return _handle_admin_json_action(request, "clear")
