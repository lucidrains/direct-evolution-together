import hmac
import io
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import zipfile

from django.conf import settings
from django.http import HttpRequest, HttpResponse, HttpResponseBadRequest, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST, require_http_methods

from direct_evolution_together import mummy_ws, ws_bridge
from direct_evolution_together.cppn import (
    build_runner_script,
    create_random_population,
    save_cppn_bytes,
    slugify_policy_name,
)


def _resolve_base_path(request: HttpRequest) -> str:
    configured = getattr(settings, "SUBPATH", "")
    if configured:
        return configured
    forwarded = request.META.get("HTTP_X_FORWARDED_PREFIX", "").strip().strip("/")
    return f"/{forwarded}" if forwarded else ""


def index(request: HttpRequest, room: str = "lobby") -> HttpResponse:
    """Serve the minimal collaborative evolution and chat page."""
    if not room or len(room) > 64:
        return HttpResponseBadRequest("Invalid room name")
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
            initial_max_votes = max(1, min(20, int(raw_max_votes)))
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


@require_http_methods(["GET", "POST"])
def admin_reset_page(request: HttpRequest) -> HttpResponse:
    """Unlinked admin-only page for resetting a room's goal(s) with DET_ADMIN_PASSWORD."""
    configured = _configured_admin_password()
    room = "lobby"
    goal = ""
    status_msg = ""
    ok = False
    status_code = 200

    if request.method == "POST":
        room = request.POST.get("room", "lobby").strip() or "lobby"
        goal = request.POST.get("goal", "").strip()
        password = request.POST.get("password", "")
        authed, status_code, err = _verify_admin_password(password)
        if not authed:
            status_msg = err
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
        },
        status=status_code,
    )


@csrf_exempt
@require_POST
def api_reset(request: HttpRequest) -> JsonResponse:
    """Admin JSON endpoint to reset goal(s) to gen 0 via DET_ADMIN_PASSWORD."""
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
    ok, status_code, msg, reset_goals = _perform_admin_reset(
        room, goal, _configured_admin_password()
    )
    if not ok:
        return JsonResponse({"ok": False, "error": msg}, status=status_code)
    return JsonResponse({"ok": True, "room": room, "reset": reset_goals, "message": msg})
