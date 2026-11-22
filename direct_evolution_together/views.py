import base64
from dataclasses import dataclass
import gzip
import hashlib
import hmac
import io
import json
import mimetypes
import os
from pathlib import Path, PurePosixPath
import secrets
import threading
import urllib.error
import urllib.parse
import urllib.request
import zipfile

from django.conf import settings
from django.core.exceptions import RequestDataTooBig, TooManyFilesSent
from django.http import (
    FileResponse,
    Http404,
    HttpRequest,
    HttpResponse,
    HttpResponseBadRequest,
    HttpResponseNotModified,
    JsonResponse,
)
from django.shortcuts import render
from django.utils.text import slugify
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST, require_http_methods

from direct_evolution_together import mummy_ws, ws_bridge
from direct_evolution_together.config import get_config
from direct_evolution_together.crowd_trust import (
    MAX_SHARD_CLAIM_BATCH,
    SQLITE_SYNC_LOCK,
    claim_crowd_eval_shard_indices,
    compute_goal_eval_rankings,
    count_verified_population,
    fetch_room_evals,
    get_cached_gen_eval_state,
    is_contributor_quarantined,
    reset_goal_shard_state,
    resolve_effective_quorum,
    submit_eval_attestations,
)
from direct_evolution_together.cppn import (
    MAX_ARCHIVE_BYTES,
    MAX_CHECKPOINT_BYTES,
    MAX_POPULATION_SIZE,
    MAX_TOTAL_UPLOAD_BYTES,
    MAX_UPLOAD_FILES,
    MIN_POPULATION_SIZE,
    build_runner_script,
    create_seeded_population,
    ensure_genome_uuid,
    genome_content_hash,
    normalize_short_uuid,
    parse_uploaded_strain_bytes,
    save_cppn_bytes,
    slugify_policy_name,
    validate_genome_dict,
)
from direct_evolution_together.environments import (
    TASK_ADAPTERS,
    build_crowd_worker_script,
    build_worker_script,
    create_task_population,
    default_elite_for_task,
    evaluate_control_batch,
    extract_task_env_overrides,
    get_lean_task_specs_payload,
    get_scene_manifest,
    get_task_spec,
    get_task_specs_payload,
    lunarlander_eval_seeds_for_generation,
    normalize_task_id,
    render_task_media,
    rollout_control_episode,
    sanitize_task_env_params,
)
from direct_evolution_together.models import ExternalEvalResult, Room, RoomEvolutionState
from direct_evolution_together.populora_engine import (
    MAX_POPULORA_CHECKPOINT_BYTES,
    create_populora_population,
    get_or_create_base_model,
    load_populora_checkpoint_bytes,
    reset_populora_goal_runtime,
    save_populora_checkpoint_bytes,
    validate_populora_genome_dict,
)


def _validate_task_genome(spec: object, raw_genome: object) -> dict:
    if getattr(spec, "genome_kind", "cppn") == "populora" and getattr(spec, "policy_spec", None) is not None:
        return validate_populora_genome_dict(raw_genome, getattr(spec, "policy_spec"))
    return validate_genome_dict(raw_genome)


_LOOPBACK_ADDRS = frozenset({"127.0.0.1", "::1", "::ffff:127.0.0.1"})


def _is_loopback_request(request: HttpRequest) -> bool:
    remote_addr = str(request.META.get("REMOTE_ADDR", "")).strip()
    if remote_addr not in _LOOPBACK_ADDRS:
        return False
    forwarded = str(request.META.get("HTTP_X_FORWARDED_FOR", "")).strip()
    if forwarded:
        client_ip = forwarded.split(",", 1)[0].strip()
        if client_ip not in _LOOPBACK_ADDRS:
            return False
    return True


def _resolve_evolve_now_access(request: HttpRequest) -> bool:
    if not settings.EVOLVE_NOW_TOKEN:
        return False
    access_key = settings.EVOLVE_NOW_SECRET or settings.ADMIN_PASSWORD
    raw_param = request.GET.get("evolve", "").strip()
    if raw_param == "0":
        request.session.pop("det_evolve_token", None)
        return False
    if access_key and raw_param and hmac.compare_digest(raw_param, access_key):
        request.session["det_evolve_token"] = settings.EVOLVE_NOW_TOKEN
        return True
    if settings.EVOLVE_NOW_LOCAL and _is_loopback_request(request):
        return True
    sess_token = str(request.session.get("det_evolve_token", ""))
    return bool(access_key and sess_token and hmac.compare_digest(sess_token, settings.EVOLVE_NOW_TOKEN))


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


_CONTRIBUTOR_COOKIE = "det_cid"
_CONTRIBUTOR_SALT = "det.contributor"


def _resolve_contributor_id(request: HttpRequest, explicit_peer_id: str = "") -> tuple[str, bool]:
    """Return ``(contributor_id, is_new)`` backed by a tamper-evident signed cookie or CLI peer token."""
    try:
        cid = str(request.get_signed_cookie(_CONTRIBUTOR_COOKIE, salt=_CONTRIBUTOR_SALT, default="") or "").strip()
    except Exception:
        cid = ""
    if len(cid) >= 8:
        return cid[:64], False
    raw_peer = (explicit_peer_id or request.headers.get("X-Crowd-Peer", "")).strip()
    if 6 <= len(raw_peer) <= 56 and all(ch.isalnum() or ch in ("_", "-") for ch in raw_peer):
        return f"cli_{raw_peer}", False
    return f"peer_{secrets.token_hex(10)}", True


def _attach_contributor_cookie(resp: HttpResponse, cid: str, is_new: bool) -> HttpResponse:
    if is_new:
        resp.set_signed_cookie(
            _CONTRIBUTOR_COOKIE,
            cid,
            salt=_CONTRIBUTOR_SALT,
            max_age=365 * 86400,
            httponly=True,
            samesite="Lax",
        )
    return resp


def index(
    request: HttpRequest,
    room: str = "lobby",
    *,
    initial_goal_override: str = "",
    initial_ind_override: str = "",
) -> HttpResponse:
    """Serve the minimal collaborative evolution and chat page."""
    if not room or len(room) > 64:
        return HttpResponseBadRequest("Invalid room name")
    room_title = "Lobby" if room == "lobby" else room
    room_description = ""
    room_has_worker = False
    if room != "lobby":
        meta = Room.objects.filter(slug=room).values("name", "description", "worker_token_hash").first()
        if meta:
            room_title = meta["name"]
            room_description = meta["description"]
            room_has_worker = bool(meta["worker_token_hash"])
    initial_goal = (initial_goal_override or request.GET.get("goal", "")).strip()[:32]
    initial_ind = normalize_short_uuid(initial_ind_override or request.GET.get("ind", ""))
    initial_task = normalize_task_id(request.GET.get("task", ""))
    raw_query_overrides = extract_task_env_overrides(initial_task, request.GET)
    initial_env_params = sanitize_task_env_params(initial_task, raw_query_overrides) if raw_query_overrides else {}
    task_spec = get_task_spec(initial_task)
    default_pop = (
        min(settings.POPULATION_SIZE, task_spec.default_pop_size)
        if task_spec.kind == "control"
        else settings.POPULATION_SIZE
    )
    default_min_votes = task_spec.default_min_votes or settings.MIN_VOTES
    raw_min_votes = request.GET.get("min_votes", "").strip()
    initial_min_votes_explicit = bool(raw_min_votes) or task_spec.default_min_votes > 0
    default_elite = default_elite_for_task(initial_task, settings.ELITE_FRAC)

    initial_pop = _parse_clamped_int_param(
        request.GET.get("pop"), default_pop, MIN_POPULATION_SIZE, task_spec.max_pop_size
    )
    initial_min_votes = _parse_clamped_int_param(raw_min_votes, default_min_votes, 1, 1024)
    initial_max_votes = _parse_clamped_int_param(
        request.GET.get("max_votes"), settings.MAX_VOTES_PER_USER, 1, 256
    )
    initial_mut_sigma = _parse_clamped_float_param(
        request.GET.get("sigma"), settings.MUTATION_SIGMA, 0.05, 1.0
    )
    initial_elite_frac = _parse_clamped_float_param(
        request.GET.get("elite_frac"), default_elite, 0.0, 0.5
    )
    initial_crossover_rate = _parse_clamped_float_param(
        request.GET.get("crossover"), settings.CROSSOVER_RATE, 0.0, 1.0
    )
    default_brood = task_spec.default_brood_size if task_spec.default_brood_size > 1 else settings.BROOD_SIZE
    initial_brood_size = _parse_clamped_int_param(
        request.GET.get("brood_size"), default_brood, 1, 8
    )
    base_path = _resolve_base_path(request)
    ws_via_proxy = bool(getattr(settings, "WS_VIA_PROXY", False) or base_path)
    can_evolve_now = _resolve_evolve_now_access(request)
    cfg = get_config()
    contributor_id, cid_is_new = _resolve_contributor_id(request)
    fe_dir = Path(__file__).resolve().parent / "frontend"
    fe_sig = hashlib.sha256()
    for fe_p in sorted([*fe_dir.glob("*.js"), *fe_dir.glob("*.css")]):
        st = fe_p.stat()
        fe_sig.update(f"{fe_p.name}:{st.st_size}:{st.st_mtime_ns}|".encode("ascii"))
    asset_ver = fe_sig.hexdigest()[:10]
    mujoco_wasm_url = cfg.mujoco_wasm_url or settings.MUJOCO_WASM_URL
    if not mujoco_wasm_url:
        vendor_mujoco = Path(__file__).resolve().parent / "assets" / "vendor" / "mujoco" / "mujoco.js"
        if vendor_mujoco.is_file():
            mujoco_wasm_url = f"{base_path}/assets/vendor/mujoco/mujoco.js?v={asset_ver}"
        else:
            mujoco_wasm_url = "https://cdn.jsdelivr.net/npm/@mujoco/mujoco@4.0.1/mujoco.js"
    resp = render(
        request,
        "direct_evolution_together/index.html",
        {
            "base_path": base_path,
            "asset_ver": asset_ver,
            "ws_via_proxy": ws_via_proxy,
            "ws_host": settings.WS_HOST,
            "ws_port": settings.WS_PORT,
            "room_name": room,
            "room_title": room_title,
            "room_description": room_description,
            "room_has_worker": room_has_worker,
            "is_private": room != "lobby",
            "can_evolve_now": can_evolve_now,
            "evolve_token": settings.EVOLVE_NOW_TOKEN if can_evolve_now else "",
            "initial_goal": initial_goal,
            "initial_ind": initial_ind,
            "initial_env_params": initial_env_params,
            "initial_task": initial_task,
            "initial_pop": initial_pop,
            "initial_min_votes": initial_min_votes,
            "initial_min_votes_explicit": initial_min_votes_explicit,
            "default_min_votes": settings.MIN_VOTES,
            "initial_max_votes": initial_max_votes,
            "initial_mut_sigma": f"{initial_mut_sigma:.2f}",
            "initial_elite_frac": f"{initial_elite_frac:.2f}",
            "initial_crossover_rate": f"{initial_crossover_rate:.2f}",
            "initial_brood_size": initial_brood_size,
            "mujoco_wasm_url": mujoco_wasm_url,
            "task_specs": get_lean_task_specs_payload(),
            "crowd_config": {
                **cfg.trust.to_client_dict(),
                "contributor_id": contributor_id,
            },
        },
    )
    resp["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return _attach_contributor_cookie(resp, contributor_id, cid_is_new)


def health(request: HttpRequest) -> JsonResponse:
    """Basic health check endpoint."""
    return JsonResponse({"status": "ok"})


_ASSET_CONTENT_TYPES = {
    ".js": "application/javascript; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".xml": "application/xml; charset=utf-8",
    ".urdf": "application/xml; charset=utf-8",
    ".wasm": "application/wasm",
    ".json": "application/json; charset=utf-8",
    ".bin": "application/octet-stream",
    ".stl": "model/stl",
    ".obj": "text/plain; charset=utf-8",
    ".png": "image/png",
}
_PKG_DIR = Path(__file__).resolve().parent
_ASSET_CACHE_LOCK = threading.Lock()
_ASSET_CACHE: dict[tuple[str, int, int], tuple[bytes, bytes | None, str, str]] = {}


def _get_cached_asset(candidate: Path) -> tuple[bytes, bytes | None, str, str]:
    st = candidate.stat()
    key = (str(candidate), st.st_mtime_ns, st.st_size)
    with _ASSET_CACHE_LOCK:
        hit = _ASSET_CACHE.get(key)
    if hit is not None:
        return hit

    raw = candidate.read_bytes()
    gz_sidecar = candidate.with_name(candidate.name + ".gz")
    gz: bytes | None = None
    if gz_sidecar.is_file() and gz_sidecar.stat().st_mtime_ns == st.st_mtime_ns:
        gz = gz_sidecar.read_bytes()
    elif len(raw) >= 1024 and candidate.suffix.lower() != ".png":
        gz = gzip.compress(raw, compresslevel=6)
    if gz is not None and len(gz) >= len(raw):
        gz = None
    base_tag = f"{st.st_mtime_ns:x}-{st.st_size:x}"
    entry = (raw, gz, f'"{base_tag}"', f'"{base_tag}-gz"')
    with _ASSET_CACHE_LOCK:
        if len(_ASSET_CACHE) > 48:
            _ASSET_CACHE.clear()
        _ASSET_CACHE[key] = entry
    return entry


def _asset_search_dirs() -> tuple[Path, ...]:
    staged = os.getenv("DET_ASSETS_DIR", "").strip()
    if staged:
        return (Path(staged), _PKG_DIR / "frontend", _PKG_DIR / "assets")
    return (_PKG_DIR / "frontend", _PKG_DIR / "assets")


@require_GET
def serve_asset(request: HttpRequest, asset_name: str) -> HttpResponse:
    """Serve bundled frontend scripts and simulation assets with cached gzip & ETag."""
    clean = PurePosixPath(asset_name)
    if (
        clean.is_absolute()
        or ".." in clean.parts
        or not clean.parts
        or clean.suffix.lower() not in _ASSET_CONTENT_TYPES
    ):
        raise Http404("Asset not found")

    for base_dir in _asset_search_dirs():
        if not base_dir.is_dir():
            continue
        root = base_dir.resolve()
        candidate = (root / Path(*clean.parts)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.is_file():
            suffix = candidate.suffix.lower()
            ctype = _ASSET_CONTENT_TYPES[suffix]
            raw, gz, raw_tag, gz_tag = _get_cached_asset(candidate)
            accept_enc = (request.META.get("HTTP_ACCEPT_ENCODING") or "").lower()
            use_gz = gz is not None and "gzip" in accept_enc
            etag = gz_tag if use_gz else raw_tag
            cache_ctrl = (
                "public, max-age=31536000, immutable"
                if (
                    "vendor/" in asset_name
                    or suffix == ".wasm"
                    or bool(request.GET.get("v"))
                )
                else "public, max-age=3600"
            )
            if request.META.get("HTTP_IF_NONE_MATCH") == etag:
                resp = HttpResponseNotModified()
                resp["ETag"] = etag
                resp["Cache-Control"] = cache_ctrl
                resp["Vary"] = "Accept-Encoding"
                return resp

            resp = HttpResponse(gz if use_gz else raw, content_type=ctype)
            if use_gz:
                resp["Content-Encoding"] = "gzip"
            resp["ETag"] = etag
            resp["Cache-Control"] = cache_ctrl
            resp["Vary"] = "Accept-Encoding"
            resp["X-Content-Type-Options"] = "nosniff"
            return resp
    raise Http404("Asset not found")


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


def _extract_uploaded_populora_strains(
    uploaded_files: list,
    task_id: str,
    policy_spec,
    remaining_slots: int,
) -> list[tuple[str, dict, bytes]]:
    """Read, bound-check, and parse uploaded `.pt`/`.zip` PopuLoRA checkpoints."""
    strains: list[tuple[str, dict, bytes]] = []
    max_total = max(MAX_TOTAL_UPLOAD_BYTES, MAX_POPULORA_CHECKPOINT_BYTES)
    bytes_remaining = max_total
    for up in uploaded_files:
        if remaining_slots <= 0:
            break
        filename = str(getattr(up, "name", "") or "weights.pt")
        cap = min(MAX_POPULORA_CHECKPOINT_BYTES, bytes_remaining)
        raw_bytes = getattr(up, "read")(cap + 1)
        if len(raw_bytes) > cap:
            raise ValueError(f"{filename}: Checkpoint exceeds upload limit.")
        bytes_remaining -= len(raw_bytes)

        if filename.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                    pt_members = [
                        m for m in zf.infolist()
                        if not m.is_dir() and m.filename.lower().endswith(".pt")
                    ]
                    if not pt_members:
                        raise ValueError(f"{filename}: No .pt checkpoint found inside zip archive.")
                    for info in pt_members[:remaining_slots]:
                        if info.file_size > MAX_POPULORA_CHECKPOINT_BYTES:
                            raise ValueError(f"{info.filename}: Checkpoint inside zip exceeds size limit.")
                        pt_b = zf.read(info)
                        genome = load_populora_checkpoint_bytes(
                            pt_b,
                            policy_spec,
                            task_id=task_id,
                            source_name=info.filename,
                            assign_id=False,
                        )
                        stem = PurePosixPath(info.filename).stem or "populora_seed"
                        strains.append((stem[:48], genome, pt_b))
                        remaining_slots -= 1
            except zipfile.BadZipFile as exc:
                raise ValueError(f"{filename}: Invalid zip archive.") from exc
        else:
            genome = load_populora_checkpoint_bytes(
                raw_bytes,
                policy_spec,
                task_id=task_id,
                source_name=filename,
                assign_id=False,
            )
            stem = PurePosixPath(filename).stem or "populora_seed"
            strains.append((stem[:48], genome, raw_bytes))
            remaining_slots -= 1
    return strains


@require_POST
def api_validate_weights(request: HttpRequest) -> JsonResponse:
    """Validate uploaded `.pt` or `.zip` checkpoints (CPPN or PopuLoRA) and return preview genomes."""
    post_data, uploaded_files, err = _parse_multipart_upload(request)
    if err:
        return JsonResponse({"ok": False, "error": err[1]}, status=err[0])
    if not uploaded_files:
        return JsonResponse(
            {"ok": False, "error": "Please select at least one .pt or .zip weight file."},
            status=400,
        )

    task_id = normalize_task_id(str(post_data.get("task") or "image"))
    task_spec = get_task_spec(task_id)
    max_slots = task_spec.max_pop_size if task_spec.genome_kind == "populora" else MAX_POPULATION_SIZE
    existing_count = _parse_clamped_int_param(
        post_data.get("existing_count"), 0, 0, max_slots
    )
    remaining_slots = max_slots - existing_count
    if remaining_slots < 1:
        return JsonResponse(
            {
                "ok": False,
                "error": (
                    f"Cannot exceed {max_slots} uploaded strains "
                    f"(max population size is {max_slots})."
                ),
            },
            status=400,
        )

    try:
        if task_spec.genome_kind == "populora" and task_spec.policy_spec is not None:
            parsed = _extract_uploaded_populora_strains(
                uploaded_files, task_id, task_spec.policy_spec, remaining_slots
            )
        else:
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
    task = normalize_task_id(str(payload.get("task") or "image"))
    task_spec = get_task_spec(task)
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
                        f"{MIN_POPULATION_SIZE} and {task_spec.max_pop_size}."
                    )
                },
                status=400,
            )
        if not (MIN_POPULATION_SIZE <= pop_size <= task_spec.max_pop_size):
            return JsonResponse(
                {
                    "error": (
                        f"Population size must be between "
                        f"{MIN_POPULATION_SIZE} and {task_spec.max_pop_size}."
                    )
                },
                status=400,
            )
    else:
        pop_size = (
            min(int(settings.POPULATION_SIZE), task_spec.default_pop_size)
            if task_spec.kind == "control"
            else int(settings.POPULATION_SIZE)
        )

    raw_env_override = extract_task_env_overrides(task, payload)
    clean_env_params = sanitize_task_env_params(task, raw_env_override or None)
    env_policy = get_config().get_env(task)

    base_path = _resolve_base_path(request)
    seeded_meta: dict | None = None

    raw_brood = payload.get("brood_size")
    default_brood = task_spec.default_brood_size if task_spec.default_brood_size > 1 else settings.BROOD_SIZE
    brood_size_val = (
        _parse_clamped_int_param(raw_brood, default_brood, 1, 8)
        if (raw_brood is not None and str(raw_brood).strip() != "")
        else default_brood
    )
    if uploaded_files or task != "image" or brood_size_val > 1:
        if uploaded_files:
            if task_spec.genome_kind == "populora" and task_spec.policy_spec is not None:
                try:
                    seed_genomes = [
                        genome
                        for _, genome, _ in _extract_uploaded_populora_strains(
                            uploaded_files, task, task_spec.policy_spec, task_spec.max_pop_size
                        )
                    ]
                except ValueError as exc:
                    return JsonResponse({"error": str(exc)}, status=400)
                if len(seed_genomes) > pop_size:
                    return JsonResponse(
                        {"error": f"Uploaded strains ({len(seed_genomes)}) cannot exceed total population size ({pop_size})."},
                        status=400,
                    )
                genomes = create_populora_population(
                    task, task_spec.policy_spec, pop_size=pop_size, seed_genomes=seed_genomes
                )
                seeded_count = len(seed_genomes)
            else:
                try:
                    seed_genomes = [
                        genome
                        for _, genome, _ in _extract_uploaded_strains(uploaded_files, task_spec.max_pop_size)
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
                seeded_count = len(seed_genomes)
        else:
            genomes = create_task_population(task, pop_size)
            seeded_count = 0

        default_elite = default_elite_for_task(task, settings.ELITE_FRAC)
        raw_min_votes = payload.get("min_votes")
        min_votes_val = (
            _parse_clamped_int_param(raw_min_votes, task_spec.default_min_votes, 1, 1024)
            if (raw_min_votes is not None and str(raw_min_votes).strip() != "")
            else task_spec.default_min_votes
        )
        max_votes_val = _parse_clamped_int_param(
            payload.get("max_votes_per_user"), settings.MAX_VOTES_PER_USER, 1, 256
        )
        scaled = env_policy.constrain_room_scale(
            pop_size,
            min_votes_val,
            max_votes_val,
            is_control=(task_spec.kind == "control"),
        )
        seeded_meta = {
            "genomes": genomes,
            "seeded": seeded_count,
            "min_votes": int(scaled["min_votes"]),
            "max_votes": int(scaled["max_votes_per_user"]),
            "mut_sigma": _parse_clamped_float_param(
                payload.get("mutation_sigma"), settings.MUTATION_SIGMA, 0.05, 1.0
            ),
            "elite_frac": _parse_clamped_float_param(
                payload.get("elite_frac"), default_elite, 0.0, 0.5
            ),
            "crossover_rate": _parse_clamped_float_param(
                payload.get("crossover_rate"), settings.CROSSOVER_RATE, 0.0, 1.0
            ),
            "brood_size": brood_size_val,
        }

    enable_worker = task == "external" or str(payload.get("enable_worker") or "").lower() in ("1", "true", "yes")
    raw_worker_token = secrets.token_urlsafe(24) if enable_worker else ""
    worker_token_hash = hashlib.sha256(raw_worker_token.encode("utf-8")).hexdigest() if raw_worker_token else ""

    room = Room.objects.create(
        slug=_room_slug(name),
        name=name,
        description=description,
        worker_token_hash=worker_token_hash,
    )
    resp_data: dict[str, object] = {
        "ok": True,
        "slug": room.slug,
        "name": room.name,
        "description": room.description,
        "task": task,
        "url": f"{base_path}/p/{room.slug}/",
    }
    if raw_worker_token:
        resp_data["worker_token"] = raw_worker_token
        resp_data["worker_script_url"] = f"{base_path}/export/worker-script/?room={urllib.parse.quote(room.slug)}"

    if seeded_meta is not None:
        raw_goal = str(payload.get("goal") or "").strip()
        goal_name = _truncate_utf8(raw_goal or name, 32) or room.slug[:32]
        raw_auto_ev = payload.get("auto_evolve")
        auto_ev_opt = (
            str(raw_auto_ev).strip().lower() in ("1", "true", "yes", "on")
            if raw_auto_ev is not None and str(raw_auto_ev).strip() != ""
            else None
        )
        state_json = ws_bridge.build_room_state_json(
            goal_name,
            seeded_meta["genomes"],
            task=task,
            env_params=clean_env_params,
            min_votes=seeded_meta["min_votes"],
            max_votes_per_user=seeded_meta["max_votes"],
            mut_sigma=seeded_meta["mut_sigma"],
            elite_frac=seeded_meta["elite_frac"],
            crossover_rate=seeded_meta["crossover_rate"],
            brood_size=seeded_meta["brood_size"],
            auto_evolve=auto_ev_opt,
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
        if clean_env_params:
            resp_data["env_params"] = clean_env_params

    return JsonResponse(resp_data)


def _parse_export_payload(request: HttpRequest) -> tuple[dict, str, str, dict[str, object]]:
    payload = json.loads(request.body.decode("utf-8"))
    genome = payload.get("genome")
    is_populora = isinstance(genome, dict) and genome.get("kind") == "populora" and isinstance(genome.get("adapters"), dict)
    is_cppn = isinstance(genome, dict) and {"weights", "biases", "activations"} <= set(genome)
    if not (is_populora or is_cppn):
        raise ValueError("Invalid genome payload")
    genome = dict(genome)
    genome.setdefault("id", 0)
    goal = str(payload.get("goal", "policy"))
    task = normalize_task_id(str(payload.get("task") or (genome.get("task") if is_populora else "image") or ("microduck" if is_populora else "image")))
    generation = int(payload.get("generation", 0))
    index = int(payload.get("index", 0))
    raw_env = extract_task_env_overrides(task, payload)
    env_params = sanitize_task_env_params(task, raw_env or None)
    return genome, slugify_policy_name(goal, generation, index), task, env_params


def _serialize_policy_checkpoint(genome: dict, task: str) -> bytes:
    if genome.get("kind") == "populora":
        spec = get_task_spec(task)
        if spec.policy_spec is None:
            raise ValueError(f"Task '{task}' has no PopuLoRA policy specification.")
        return save_populora_checkpoint_bytes(genome, spec.policy_spec)
    return save_cppn_bytes(genome)


@require_POST
def export_weights(request: HttpRequest) -> HttpResponse:
    """Download a trained/evolved CPPN or PopuLoRA genome as a .pt checkpoint."""
    try:
        genome, slug, task, _ = _parse_export_payload(request)
        data = _serialize_policy_checkpoint(genome, task)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return HttpResponseBadRequest(str(exc))

    resp = HttpResponse(data, content_type="application/octet-stream")
    resp["Content-Disposition"] = f'attachment; filename="{slug}.pt"'
    return resp


@require_GET
def export_script(request: HttpRequest) -> HttpResponse:
    """Download the standalone uv runner script for checkpoints."""
    task = normalize_task_id(request.GET.get("task", "image"))
    raw_env = extract_task_env_overrides(task, request.GET)
    env_params = sanitize_task_env_params(task, raw_env or None) if raw_env else {}
    script = build_runner_script("weights.pt", task=task, env_params=env_params or None)
    resp = HttpResponse(script, content_type="text/x-python; charset=utf-8")
    resp["Content-Disposition"] = 'attachment; filename="run.py"'
    return resp


@require_POST
def export_bundle(request: HttpRequest) -> HttpResponse:
    """Download a zip containing weights.pt and the standalone uv runner script."""
    try:
        genome, slug, task, env_params = _parse_export_payload(request)
        weights_bytes = _serialize_policy_checkpoint(genome, task)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return HttpResponseBadRequest(str(exc))

    script_text = build_runner_script("weights.pt", task=task, env_params=env_params or None)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr(f"{slug}/weights.pt", weights_bytes)
        zf.writestr(f"{slug}/run.py", script_text)
        spec = get_task_spec(task)
        resolved_env = spec.resolve_env_params(env_params)
        eff_scene_id = str(resolved_env.get("scene_id") or spec.scene_id or "")
        urdf_rel = resolved_env.get("urdf_asset")
        if spec.mjcf_asset:
            xml_path = _PKG_DIR / "assets" / spec.mjcf_asset
            if xml_path.is_file():
                zf.writestr(f"{slug}/{spec.mjcf_asset}", xml_path.read_bytes())
        if isinstance(urdf_rel, str) and urdf_rel:
            urdf_path = _PKG_DIR / "assets" / urdf_rel
            if urdf_path.is_file():
                zf.writestr(f"{slug}/{Path(urdf_rel).name}", urdf_path.read_bytes())
        if eff_scene_id:
            lic_path = _PKG_DIR / "assets" / "scenes" / eff_scene_id / "LICENSE"
            if lic_path.is_file():
                zf.writestr(f"{slug}/SCENE_LICENSE", lic_path.read_bytes())

    resp = HttpResponse(buf.getvalue(), content_type="application/zip")
    resp["Content-Disposition"] = f'attachment; filename="{slug}.zip"'
    return resp


_ALLOWED_MEDIA_SUFFIXES = {".mp4", ".webm", ".png", ".webp", ".jpg", ".jpeg", ".json"}


@require_POST
def api_render_video(request: HttpRequest) -> JsonResponse:
    """Render and cache a control rollout video + poster PNG lazily on demand."""
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return JsonResponse({"ok": False, "error": "Invalid JSON."}, status=400)
    if not isinstance(payload, dict):
        return JsonResponse({"ok": False, "error": "Invalid render payload."}, status=400)

    task = normalize_task_id(str(payload.get("task") or "cartpole"))
    spec = get_task_spec(task)
    if not spec.supports_video or not spec.allow_server_video:
        return JsonResponse({"ok": False, "error": f"Server video rendering is disabled for '{task}'."}, status=400)

    try:
        genome = _validate_task_genome(spec, payload.get("genome"))
        seed = max(0, min(10_000_000, int(payload.get("seed", 0))))
    except (ValueError, TypeError) as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)

    res = render_task_media(
        task,
        genome,
        seed=seed,
        media_root=settings.MEDIA_ROOT,
        max_cache_bytes=settings.MAX_MEDIA_CACHE_BYTES,
    )
    base_path = _resolve_base_path(request)
    out_payload: dict[str, object] = {
        "ok": True,
        "video_url": f"{base_path}/media/{res['mp4_rel']}",
        "poster_url": f"{base_path}/media/{res['png_rel']}",
        "steps": res["steps"],
        "return": res["return"],
        "seed": res["seed"],
    }
    if "mean_abs_x" in res:
        out_payload["mean_abs_x"] = res["mean_abs_x"]
    if "distance" in res:
        out_payload["distance"] = res["distance"]
    return JsonResponse(out_payload)


@require_POST
def api_sim_eval_batch(request: HttpRequest) -> JsonResponse:
    """Run (or return LRU-cached) server-side evaluations + primary pose trajectories."""
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return JsonResponse({"ok": False, "error": "Invalid JSON."}, status=400)
    if not isinstance(payload, dict):
        return JsonResponse({"ok": False, "error": "Invalid simulation payload."}, status=400)

    task = normalize_task_id(str(payload.get("task") or ""))
    if task not in TASK_ADAPTERS:
        return JsonResponse({"ok": False, "error": f"Task '{task}' does not support server simulation."}, status=400)
    spec = get_task_spec(task)
    if "server" not in spec.sim_modes:
        return JsonResponse({"ok": False, "error": f"Task '{task}' is locked to client-side simulation."}, status=400)

    raw_genomes = payload.get("genomes")
    if not isinstance(raw_genomes, list) or not raw_genomes:
        return JsonResponse({"ok": False, "error": "Non-empty genomes list is required."}, status=400)
    max_batch = max(MAX_POPULATION_SIZE, int(spec.max_pop_size))
    if len(raw_genomes) > max_batch:
        return JsonResponse({"ok": False, "error": f"Too many genomes (max {max_batch})."}, status=400)

    try:
        genomes = [_validate_task_genome(spec, g) for g in raw_genomes]
    except ValueError as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)

    raw_seeds = payload.get("seeds")
    seeds: tuple[int, ...] | None = None
    if isinstance(raw_seeds, list) and raw_seeds:
        if len(raw_seeds) > 12:
            return JsonResponse({"ok": False, "error": "At most 12 seeds may be evaluated per request."}, status=400)
        try:
            seeds = tuple(max(0, min(10_000_000, int(s))) for s in raw_seeds)
        except (TypeError, ValueError):
            return JsonResponse({"ok": False, "error": "Invalid seed list."}, status=400)

    include_rollout = bool(payload.get("include_rollout", True))
    items = evaluate_control_batch(
        task,
        genomes,
        seeds=seeds,
        include_primary_rollout=include_rollout,
    )
    return JsonResponse({"ok": True, "engine": "server", "task": task, "items": items})


@require_POST
def api_sim_rollout(request: HttpRequest) -> JsonResponse:
    """Run (or retrieve cached) single episode rollout for server-mode fullscreen playback."""
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return JsonResponse({"ok": False, "error": "Invalid JSON."}, status=400)
    if not isinstance(payload, dict):
        return JsonResponse({"ok": False, "error": "Invalid rollout payload."}, status=400)

    task = normalize_task_id(str(payload.get("task") or ""))
    if task not in TASK_ADAPTERS:
        return JsonResponse({"ok": False, "error": f"Task '{task}' does not support server simulation."}, status=400)
    spec = get_task_spec(task)
    if "server" not in spec.sim_modes:
        return JsonResponse({"ok": False, "error": f"Task '{task}' is locked to client-side simulation."}, status=400)

    try:
        genome = _validate_task_genome(spec, payload.get("genome"))
        seed = max(0, min(10_000_000, int(payload.get("seed", 0))))
    except (ValueError, TypeError) as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)

    raw_nudge = str(payload.get("nudge") or "").strip().lower()
    nudge = raw_nudge if raw_nudge in ("left", "right", "up", "back", "fwd", "hop") else None
    raw_steps = payload.get("max_steps")
    max_steps = int(raw_steps) if isinstance(raw_steps, int) else None
    include_geoms = bool(payload.get("include_geoms", True))

    rollout = rollout_control_episode(
        task,
        genome,
        seed=seed,
        max_steps=max_steps,
        nudge=nudge,
        include_geoms=include_geoms,
    )
    return JsonResponse({"ok": True, "engine": "server", "task": task, "rollout": rollout})


@require_GET
def api_sim_base_model(request: HttpRequest) -> JsonResponse:
    """Serve shared PopuLoRA base model weights (cached immutably by content hash)."""
    task = normalize_task_id(request.GET.get("task", "microduck"))
    spec = get_task_spec(task)
    if spec.genome_kind != "populora" or spec.policy_spec is None:
        return JsonResponse({"ok": False, "error": f"Task '{task}' has no PopuLoRA base model."}, status=400)
    req_hash = request.GET.get("hash", "").strip()[:32]
    try:
        base_seed = int(request.GET.get("seed", "1"))
    except ValueError:
        base_seed = 1
    _, resolved_hash, payload = get_or_create_base_model(
        spec.policy_spec, base_seed=base_seed, base_hash=req_hash or None
    )
    resp = JsonResponse({"ok": True, "task": spec.task_id, **payload})
    if req_hash and req_hash == resolved_hash:
        resp["Cache-Control"] = "public, max-age=31536000, immutable"
    else:
        resp["Cache-Control"] = "public, max-age=3600"
    return resp


@require_GET
def api_sim_scene_manifest(request: HttpRequest, scene_id: str) -> JsonResponse:
    """Serve compiled MuJoCo scene manifest (physics VFS + visual meshpack hashes)."""
    manifest = get_scene_manifest(scene_id)
    if manifest is None:
        return JsonResponse({"ok": False, "error": f"Scene '{scene_id}' not found."}, status=404)
    return JsonResponse({"ok": True, **manifest})


def _verify_worker_token(request: HttpRequest, room_obj: Room | None) -> bool:
    if room_obj is None or not room_obj.worker_token_hash:
        return False
    supplied = request.headers.get("X-Worker-Token", "").strip()
    if not supplied:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            supplied = auth[7:].strip()
    if not supplied:
        return False
    digest = hashlib.sha256(supplied.encode("utf-8")).hexdigest()
    return hmac.compare_digest(digest, room_obj.worker_token_hash)


@dataclass(frozen=True)
class _GoalIndex:
    stubs: tuple[dict, ...]
    hashes: tuple[str, ...]
    hash_set: frozenset[str]
    base_hash: str


_GOAL_INDEX_LOCK = threading.Lock()
_GOAL_INDEX_CACHE: dict[tuple[str, str, int, int, str], _GoalIndex] = {}


def _invalidate_goal_index_cache(room: str, goal: str = "") -> None:
    clean_goal = (goal or "").strip()
    match_all = not clean_goal or clean_goal == "*"
    with _GOAL_INDEX_LOCK:
        for k in [k for k in _GOAL_INDEX_CACHE if k[0] == room and (match_all or k[1] == clean_goal)]:
            _GOAL_INDEX_CACHE.pop(k, None)


def _goal_from_snap(snap: dict | None, goal: str = "") -> dict | None:
    goals = (snap or {}).get("goals") or []
    return next((g for g in goals if isinstance(g, dict) and (not goal or g.get("name") == goal)), None)


def _fetch_room_snapshot(
    room: str,
    goal: str = "",
    *,
    stubs_only: bool = False,
    head_only: bool = False,
) -> dict | None:
    clean_goal = (goal or "").strip()
    if ws_bridge.is_running():
        if head_only:
            raw = mummy_ws.get_room_goal_head_json(room, clean_goal)
        elif stubs_only:
            raw = mummy_ws.get_room_goal_stubs_json(room, clean_goal)
        else:
            raw = (
                mummy_ws.get_room_goal_snapshot_json(room, clean_goal)
                if clean_goal
                else mummy_ws.get_room_snapshot_json(room)
            )
        return json.loads(raw)
    configured = _configured_admin_password()
    if configured:
        qparts: list[str] = []
        if clean_goal:
            qparts.append(f"goal={urllib.parse.quote(clean_goal, safe='')}")
        if stubs_only:
            qparts.append("stubs=1")
        if head_only:
            qparts.append("head=1")
        qs = f"?{'&'.join(qparts)}" if qparts else ""
        status, text = _ws_internal_request(
            "GET",
            f"/internal/room-snapshot/{urllib.parse.quote(room, safe='')}{qs}",
            configured,
        )
        if status == 200 and text:
            return json.loads(text)
    row = RoomEvolutionState.objects.filter(room=room).first()
    if row is not None:
        try:
            parsed = json.loads(row.state_json)
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            pass
    return None


def _get_cached_goal_index(
    room: str,
    target_head: dict,
    *,
    hash_length: int = 20,
    max_cache_entries: int = 128,
) -> _GoalIndex:
    """Return cached `_GoalIndex` for `(room, goal, generation)`, parsing stubs once per generation."""
    goal_name = str(target_head.get("name") or "")
    generation = int(target_head.get("generation", 0))
    head_genomes = target_head.get("genomes")
    has_inline = isinstance(head_genomes, list) and len(head_genomes) > 0
    pop_size = int(target_head.get("population_size") or (len(head_genomes) if isinstance(head_genomes, list) else 0))
    head_uuid = str(
        target_head.get("head_uuid")
        or ((head_genomes[0].get("uuid") if isinstance(head_genomes[0], dict) else "") if has_inline else "")
    )
    cache_key = (room, goal_name, generation, pop_size, head_uuid)
    if head_uuid:
        with _GOAL_INDEX_LOCK:
            if (hit := _GOAL_INDEX_CACHE.get(cache_key)) is not None:
                return hit

    if has_inline:
        raw_genomes = [g for g in head_genomes if isinstance(g, dict)]
    else:
        stub_goal = _goal_from_snap(_fetch_room_snapshot(room, goal_name, stubs_only=True), goal_name)
        raw_genomes = [g for g in ((stub_goal or {}).get("genomes") or []) if isinstance(g, dict)]

    stubs: list[dict] = []
    hashes: list[str] = []
    base_hash = ""
    for g in raw_genomes:
        gh = str(g.get("hash") or genome_content_hash(g, length=hash_length))
        guuid = ensure_genome_uuid(g)
        if not base_hash and g.get("base_hash"):
            base_hash = str(g["base_hash"])
        stubs.append({"id": int(g.get("id", 0) or 0), "uuid": guuid, "hash": gh})
        hashes.append(gh)

    resolved_head_uuid = head_uuid or (stubs[0]["uuid"] if stubs else "")
    hash_tuple = tuple(hashes)
    entry = _GoalIndex(
        stubs=tuple(stubs),
        hashes=hash_tuple,
        hash_set=frozenset(h for h in hash_tuple if h),
        base_hash=base_hash,
    )
    if resolved_head_uuid:
        with _GOAL_INDEX_LOCK:
            if len(_GOAL_INDEX_CACHE) > max_cache_entries:
                _GOAL_INDEX_CACHE.pop(next(iter(_GOAL_INDEX_CACHE)), None)
            _GOAL_INDEX_CACHE[(room, goal_name, generation, len(stubs), str(resolved_head_uuid))] = entry
    return entry


def _parse_slice_response(raw: str) -> dict[int, dict] | None:
    parsed = json.loads(raw)
    if not (isinstance(parsed, dict) and parsed.get("ok") and isinstance(parsed.get("items"), list)):
        return None
    return {
        int(it["index"]): it["genome"]
        for it in parsed["items"]
        if isinstance(it, dict) and isinstance(it.get("index"), int) and isinstance(it.get("genome"), dict)
    }


def _fetch_room_goal_slice(room: str, goal: str, generation: int, indices: list[int]) -> dict[int, dict]:
    """Fetch full genome dicts ONLY for `indices` in O(len(indices)) without reparsing all P genomes."""
    if not indices:
        return {}
    if ws_bridge.is_running():
        hit = _parse_slice_response(mummy_ws.get_room_goal_slice_json(room, goal, int(generation), json.dumps(indices)))
        if hit is not None:
            return hit
    if configured := _configured_admin_password():
        status, text = _ws_internal_request(
            "POST",
            "/internal/room-slice",
            configured,
            {"room": room, "goal": goal, "generation": int(generation), "indices": indices},
        )
        if status == 200 and text and (hit := _parse_slice_response(text)) is not None:
            return hit
    g_obj = _goal_from_snap(_fetch_room_snapshot(room, goal), goal)
    if g_obj is None or int(g_obj.get("generation", -1)) != int(generation):
        return {}
    all_g = g_obj.get("genomes") or []
    return {idx: all_g[idx] for idx in indices if 0 <= idx < len(all_g) and isinstance(all_g[idx], dict)}


def _get_live_room_peer_count(room: str, *, timeout_sec: float = 1.5) -> int | None:
    """Return active WS peer count in `room` when WS server is active, else None."""
    if ws_bridge.is_running():
        cnt = int(mummy_ws.get_room_peer_count(room))
        return cnt if cnt >= 1 else None
    configured = _configured_admin_password()
    if configured:
        status, text = _ws_internal_request(
            "GET",
            f"/internal/room-peers/{urllib.parse.quote(room, safe='')}",
            configured,
            timeout=timeout_sec,
        )
        if status == 200 and text:
            try:
                cnt = int(json.loads(text).get("peers", 0))
                return cnt if cnt >= 1 else None
            except Exception:
                pass
    return None


def _notify_goal_sim_progress(
    room: str,
    goal_name: str,
    generation: int,
    verified_count: int,
    *,
    timeout_sec: float = 1.5,
) -> str:
    if ws_bridge.is_running():
        return str(mummy_ws.notify_goal_sim_progress(room, goal_name, int(generation), int(verified_count)))
    configured = _configured_admin_password()
    if configured:
        status, text = _ws_internal_request(
            "POST",
            "/internal/sim-progress",
            configured,
            {
                "room": room,
                "goal": goal_name,
                "generation": int(generation),
                "verified_count": int(verified_count),
            },
            timeout=timeout_sec,
        )
        if status == 200 and text:
            try:
                return str(json.loads(text).get("phase", ""))
            except Exception:
                pass
    return ""


def _sync_goal_sim_phase(
    room: str,
    goal_name: str,
    generation: int,
    *,
    goal_index: _GoalIndex | None = None,
) -> tuple[int, str]:
    """Count verified genomes belonging to the active goal population and advance phase if ready."""
    if goal_index is None:
        target = _goal_from_snap(_fetch_room_snapshot(room, goal_name, head_only=True), goal_name)
        if target is None or int(target.get("generation", -1)) != int(generation):
            return 0, ""
        goal_index = _get_cached_goal_index(room, target)
    if not goal_index.hashes:
        return 0, ""
    verified_hashes, _ = get_cached_gen_eval_state(room, goal_name, generation)
    verified_in_pop = count_verified_population(verified_hashes, goal_index.hashes, goal_index.hash_set)
    phase = _notify_goal_sim_progress(room, goal_name, generation, verified_in_pop)
    return verified_in_pop, phase


def _broadcast_room_event(
    room: str,
    payload: dict,
    *,
    goal: str = "",
    timeout_sec: float = 1.5,
) -> None:
    clean_goal = (goal or "").strip()
    if ws_bridge.is_running():
        raw_payload = json.dumps(payload, separators=(",", ":"))
        if clean_goal:
            mummy_ws.broadcast_goal_payload(room, clean_goal, raw_payload)
        else:
            mummy_ws.broadcast_room_payload(room, raw_payload)
        return
    configured = _configured_admin_password()
    if configured:
        body: dict[str, object] = {"room": room, "payload": payload}
        if clean_goal:
            body["goal"] = clean_goal
        _ws_internal_request(
            "POST",
            "/internal/room-broadcast",
            configured,
            body,
            timeout=timeout_sec,
        )


@require_GET
def api_worker_jobs(request: HttpRequest, room: str) -> JsonResponse:
    """Return unevaluated CPPN genomes in `room` for an authenticated researcher worker script."""
    room_obj = Room.objects.filter(slug=room).first()
    if not _verify_worker_token(request, room_obj):
        return JsonResponse({"ok": False, "error": "Invalid or missing worker token."}, status=403)

    req_goal = request.GET.get("goal", "").strip()
    snapshot = _fetch_room_snapshot(room, req_goal)
    goals = (snapshot or {}).get("goals") or []
    if not goals:
        if req_goal:
            return JsonResponse({"ok": False, "error": f"Goal '{req_goal}' not found."}, status=404)
        return JsonResponse({"ok": True, "room": room, "goal": None, "generation": 0, "pending": [], "total_count": 0})

    target = next((g for g in goals if g.get("name") == req_goal), None) if req_goal else goals[0]
    if target is None:
        return JsonResponse({"ok": False, "error": f"Goal '{req_goal}' not found."}, status=404)

    goal_name = str(target.get("name") or "")
    generation = int(target.get("generation", 0))
    genomes = target.get("genomes") or []
    done_hashes = set(
        ExternalEvalResult.objects.filter(room=room, goal=goal_name, generation=generation)
        .values_list("genome_hash", flat=True)
    )

    pending = []
    for idx, g in enumerate(genomes):
        gh = genome_content_hash(g, length=20)
        if gh not in done_hashes:
            pending.append({"index": idx, "id": g.get("id", 0), "hash": gh, "genome": g})

    return JsonResponse(
        {
            "ok": True,
            "room": room,
            "goal": goal_name,
            "task": target.get("task", "external"),
            "generation": generation,
            "total_count": len(genomes),
            "completed_count": len(genomes) - len(pending),
            "pending": pending,
        }
    )


_MAX_WORKER_POSTER_B64 = 128 * 1024


@csrf_exempt
@require_POST
def api_worker_results(request: HttpRequest, room: str) -> JsonResponse:
    """Ingest evaluation telemetry & optional visual frames/posters from a researcher's local script."""
    room_obj = Room.objects.filter(slug=room).first()
    if not _verify_worker_token(request, room_obj):
        return JsonResponse({"ok": False, "error": "Invalid or missing worker token."}, status=403)
    if size_err := _check_request_content_length(request):
        return JsonResponse({"ok": False, "error": size_err}, status=413)

    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return JsonResponse({"ok": False, "error": "Invalid JSON."}, status=400)
    if not isinstance(payload, dict):
        return JsonResponse({"ok": False, "error": "Invalid worker payload."}, status=400)

    goal_name = str(payload.get("goal") or "").strip()[:64]
    try:
        generation = int(payload.get("generation", 0))
    except (TypeError, ValueError):
        generation = 0
    results = payload.get("results")
    if not goal_name or not isinstance(results, list) or not results:
        return JsonResponse({"ok": False, "error": "goal, generation, and non-empty results are required."}, status=400)

    outcome = submit_eval_attestations(
        room=room,
        goal=goal_name,
        generation=generation,
        results=results,
        contributor_id=f"worker:{room}",
        authority=True,
    )
    stored_items = outcome["items"]
    if stored_items:
        ws_items = [{k: v for k, v in it.items() if k != "traj_sample"} for it in stored_items]
        _broadcast_room_event(
            room,
            {
                "type": "worker_eval_updated",
                "goal": goal_name,
                "generation": generation,
                "items": ws_items,
            },
            goal=goal_name,
        )
        _sync_goal_sim_phase(room, goal_name, generation)
    return JsonResponse({"ok": True, "accepted": len(stored_items)})


@require_GET
def api_config(request: HttpRequest) -> JsonResponse:
    """Return active centralized environment, execution-lock, and crowd-trust config."""
    payload: dict[str, object] = {"ok": True, **get_config().to_public_dict()}
    if request.GET.get("task_specs", "").strip().lower() in ("1", "true", "yes", "full"):
        payload["task_specs"] = get_task_specs_payload()
    return JsonResponse(payload)


@require_GET
def api_worker_evals(request: HttpRequest, room: str) -> JsonResponse:
    """Fetch cached researcher worker evaluation badges/posters for a room goal & generation."""
    goal_name = request.GET.get("goal", "").strip()[:64]
    try:
        generation = int(request.GET.get("generation", "0"))
    except ValueError:
        generation = 0
    if not goal_name:
        return JsonResponse({"ok": True, "room": room, "goal": "", "generation": generation, "items": []})
    items = fetch_room_evals(room, goal_name, generation, trusted_only=False, include_traj=False)
    return JsonResponse({"ok": True, "room": room, "goal": goal_name, "generation": generation, "items": items})


def _is_goal_sharded(spec: object, pop_size: int) -> bool:
    return bool(getattr(spec, "sim_phase_enabled", False) and pop_size > int(getattr(spec, "crowd_shard_threshold", 16)))


def _strip_traj_sample(ev_rec: dict | None) -> dict | None:
    if isinstance(ev_rec, dict) and "traj_sample" in ev_rec:
        return {k: v for k, v in ev_rec.items() if k != "traj_sample"}
    return ev_rec


def _format_uuid_hit(
    *,
    clean_uuid: str,
    room_slug: str,
    goal_name: str,
    task_id: str,
    env_params: dict,
    entry_gen: int,
    idx: int | None,
    is_live: bool,
    entry: dict,
    gh: str,
) -> dict[str, object]:
    evs = (
        fetch_room_evals(room_slug, goal_name, entry_gen, trusted_only=False, include_traj=False, genome_hashes=[gh])
        if gh
        else []
    )
    return {
        "uuid": clean_uuid,
        "id": int(entry.get("id", 0) or 0),
        "room": room_slug,
        "goal": goal_name,
        "task": task_id,
        "env_params": env_params,
        "generation": entry_gen,
        "index": idx,
        "is_live": is_live,
        "op": str(entry.get("op") or "random"),
        "parents": [int(p) for p in (entry.get("parents") or []) if isinstance(p, int)],
        "hash": gh,
        "genome": entry,
        "eval": _strip_traj_sample(evs[0] if evs else None),
    }


def _search_goals_for_uuid(room_slug: str, goals: list, clean_uuid: str) -> dict[str, object] | None:
    for goal in goals:
        if not isinstance(goal, dict):
            continue
        g_name = str(goal.get("name") or "")
        task_id = normalize_task_id(str(goal.get("task") or "image"))
        gen_num = int(goal.get("generation", 0))
        env_params = goal.get("env_params") if isinstance(goal.get("env_params"), dict) else {}

        for idx, g in enumerate(goal.get("genomes") or []):
            if isinstance(g, dict) and normalize_short_uuid(g.get("uuid")) == clean_uuid:
                gh = str(g.get("hash") or genome_content_hash(g, length=20))
                return _format_uuid_hit(
                    clean_uuid=clean_uuid, room_slug=room_slug, goal_name=g_name, task_id=task_id,
                    env_params=env_params, entry_gen=gen_num, idx=idx, is_live=True, entry=g, gh=gh,
                )

        for node in goal.get("lineage") or []:
            if isinstance(node, dict) and normalize_short_uuid(node.get("uuid")) == clean_uuid:
                return _format_uuid_hit(
                    clean_uuid=clean_uuid, room_slug=room_slug, goal_name=g_name, task_id=task_id,
                    env_params=env_params, entry_gen=int(node.get("gen", 0)), idx=None, is_live=False,
                    entry=node, gh=str(node.get("hash") or ""),
                )
    return None


def _pack_uuid_hit_from_raw(raw_hit: dict, clean_uuid: str) -> dict[str, object] | None:
    entry = raw_hit.get("genome")
    if not isinstance(entry, dict):
        return None
    raw_idx = raw_hit.get("index")
    gh = str(
        entry.get("hash")
        or (genome_content_hash(entry, length=20) if ("weights" in entry or "adapters" in entry) else "")
    )
    return _format_uuid_hit(
        clean_uuid=clean_uuid,
        room_slug=str(raw_hit.get("room") or ""),
        goal_name=str(raw_hit.get("goal") or ""),
        task_id=normalize_task_id(str(raw_hit.get("task") or "image")),
        env_params=raw_hit.get("env_params") if isinstance(raw_hit.get("env_params"), dict) else {},
        entry_gen=int(raw_hit.get("generation", 0)),
        idx=int(raw_idx) if isinstance(raw_idx, int) else None,
        is_live=bool(raw_hit.get("is_live", False)),
        entry=entry,
        gh=gh,
    )


def _find_individual_by_uuid(ind_uuid: str, room_hint: str = "") -> dict[str, object] | None:
    clean_uuid = normalize_short_uuid(ind_uuid)
    if not clean_uuid:
        return None
    hint = (room_hint or "").strip()

    checked_live: set[str] = set()
    if ws_bridge.is_running():
        try:
            active_rooms = json.loads(mummy_ws.get_active_rooms_json() or "[]")
        except ValueError:
            active_rooms = []
        ordered_live = ([hint] if hint else []) + [
            str(r) for r in active_rooms if isinstance(r, str) and str(r) != hint
        ]
        for r_slug in ordered_live:
            if not r_slug or r_slug in checked_live:
                continue
            checked_live.add(r_slug)
            try:
                fast_raw = mummy_ws.get_room_individual_by_uuid_json(r_slug, clean_uuid)
                if fast_raw:
                    fast_parsed = json.loads(fast_raw)
                    if isinstance(fast_parsed, dict) and fast_parsed.get("ok"):
                        packed = _pack_uuid_hit_from_raw(fast_parsed, clean_uuid)
                        if packed is not None:
                            return packed
            except ValueError:
                continue

    elif hint:
        snap = _fetch_room_snapshot(hint)
        if isinstance(snap, dict):
            hit = _search_goals_for_uuid(hint, snap.get("goals") or [], clean_uuid)
            if hit is not None:
                return hit

    db_qs = RoomEvolutionState.objects.all()
    rows = list(db_qs)
    rows.sort(key=lambda r: (0 if (hint and r.room == hint) else (1 if r.room == "lobby" else 2), r.room))
    for row in rows:
        try:
            parsed = json.loads(row.state_json)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            hit = _search_goals_for_uuid(row.room, parsed.get("goals") or [], clean_uuid)
            if hit is not None:
                return hit
    return None


def _build_individual_permalink(base_path: str, room_slug: str, ind_uuid: str) -> str:
    if room_slug and room_slug != "lobby":
        return f"{base_path}/p/{urllib.parse.quote(room_slug, safe='')}/i/{ind_uuid}/"
    return f"{base_path}/i/{ind_uuid}/"


@require_GET
def individual_permalink(request: HttpRequest, ind_uuid: str, room: str = "") -> HttpResponse:
    """Direct researcher permalink `/i/<uuid>/` or `/p/<room>/i/<uuid>/` opening that individual."""
    found = _find_individual_by_uuid(ind_uuid, room_hint=room)
    if found is None:
        raise Http404("Individual not found")
    target_room = str(found["room"])
    target_goal = str(found["goal"])
    return index(
        request,
        room=target_room,
        initial_goal_override=target_goal,
        initial_ind_override=str(found["uuid"]),
    )


@require_GET
def api_individual_detail(request: HttpRequest, ind_uuid: str, room: str = "") -> JsonResponse:
    """Return full genome weights/adapters, lineage metadata, and permalink for an individual's short UUID."""
    found = _find_individual_by_uuid(ind_uuid, room_hint=room)
    if found is None:
        return JsonResponse({"ok": False, "error": f"Individual '{ind_uuid}' not found."}, status=404)
    base_path = _resolve_base_path(request)
    permalink = _build_individual_permalink(base_path, str(found["room"]), str(found["uuid"]))
    return JsonResponse({"ok": True, **found, "permalink": permalink})


@csrf_exempt
@require_http_methods(["GET", "POST"])
def api_crowd_claim(request: HttpRequest, room: str) -> JsonResponse:
    """Claim a disjoint shard of unevaluated MuJoCo individuals for local client simulation."""
    payload: dict = {}
    if request.method == "POST" and request.body:
        try:
            parsed = json.loads(request.body.decode("utf-8"))
            if isinstance(parsed, dict):
                payload = parsed
        except (UnicodeDecodeError, json.JSONDecodeError):
            return JsonResponse({"ok": False, "error": "Invalid JSON."}, status=400)

    req_goal = str(payload.get("goal") or request.GET.get("goal", "")).strip()[:64]
    explicit_peer = str(payload.get("peer_id") or request.GET.get("peer_id", "")).strip()
    cid, cid_is_new = _resolve_contributor_id(request, explicit_peer_id=explicit_peer)

    head_snap = _fetch_room_snapshot(room, req_goal, head_only=True)
    goals = (head_snap or {}).get("goals") or []
    if not goals:
        resp = JsonResponse(
            {
                "ok": True,
                "room": room,
                "goal": req_goal or None,
                "generation": 0,
                "shard": [],
                "total_count": 0,
                "completed_count": 0,
                "remaining_count": 0,
            }
        )
        return _attach_contributor_cookie(resp, cid, cid_is_new)

    if req_goal:
        target = next((g for g in goals if g.get("name") == req_goal), None)
    else:
        target = next(
            (g for g in goals if get_task_spec(str(g.get("task") or "image")).sim_phase_enabled),
            goals[0],
        )
    if target is None:
        return JsonResponse({"ok": False, "error": f"Goal '{req_goal}' not found."}, status=404)

    goal_name = str(target.get("name") or "")
    task_id = normalize_task_id(str(target.get("task") or "image"))
    spec = get_task_spec(task_id)
    raw_batch = payload.get("batch_size", request.GET.get("batch_size", spec.crowd_shard_size))
    batch_size = _parse_clamped_int_param(raw_batch, spec.crowd_shard_size, 1, MAX_SHARD_CLAIM_BATCH)
    generation = int(target.get("generation", 0))
    goal_env_raw = target.get("env_params") if isinstance(target.get("env_params"), dict) else None
    resolved_env = spec.resolve_env_params(goal_env_raw)

    g_index = _get_cached_goal_index(room, target)
    stubs = g_index.stubs
    pop_total = len(stubs)

    if is_contributor_quarantined(room, cid):
        done_set, _ = get_cached_gen_eval_state(room, goal_name, generation)
        completed_cnt = count_verified_population(done_set, g_index.hashes, g_index.hash_set)
        chosen_idxs: list[int] = []
    else:
        chosen_idxs, completed_cnt = claim_crowd_eval_shard_indices(
            room,
            goal_name,
            generation,
            g_index.hashes,
            peer_id=cid,
            batch_size=batch_size,
            pop_hash_set=g_index.hash_set,
        )

    slice_map = _fetch_room_goal_slice(room, goal_name, generation, chosen_idxs) if chosen_idxs else {}
    shard = [
        {
            "index": idx,
            "id": stubs[idx]["id"],
            "uuid": stubs[idx]["uuid"],
            "hash": stubs[idx]["hash"],
            "genome": g_full,
        }
        for idx in chosen_idxs
        if idx < pop_total and isinstance(g_full := slice_map.get(idx), dict)
    ]

    resp = JsonResponse(
        {
            "ok": True,
            "room": room,
            "goal": goal_name,
            "task": task_id,
            "env_params": resolved_env,
            "generation": generation,
            "phase": str(target.get("phase") or "simulating"),
            "sharded": _is_goal_sharded(spec, pop_total),
            "crowd_shard_threshold": spec.crowd_shard_threshold,
            "crowd_shard_size": spec.crowd_shard_size,
            "eval_seeds": (
                list(lunarlander_eval_seeds_for_generation(generation))
                if task_id == "lunarlander"
                else list(spec.eval_seeds or [0, 1])
            ),
            "base_hash": g_index.base_hash,
            "contributor_id": cid,
            "total_count": pop_total,
            "completed_count": completed_cnt,
            "remaining_count": max(0, pop_total - completed_cnt),
            "shard": shard,
        }
    )
    return _attach_contributor_cookie(resp, cid, cid_is_new)


@require_GET
def api_crowd_evals(request: HttpRequest, room: str) -> HttpResponse:
    """Fetch crowd WASM/WebGPU/local-shard consensus results, rankings, and contributor trust status."""
    cfg = get_config().trust
    cid, cid_is_new = _resolve_contributor_id(request)
    goal_name = request.GET.get("goal", "").strip()[:64]
    try:
        generation = int(request.GET.get("generation", "0"))
    except ValueError:
        generation = 0
    trusted_only = request.GET.get("trusted_only", "").strip().lower() in ("1", "true", "yes")
    active_only = request.GET.get("active_only", "").strip().lower() in ("1", "true", "yes")

    g_stubs: tuple[dict, ...] = ()
    pop_hashes: tuple[str, ...] | None = None
    rankings: list[dict[str, object]] = []
    sharded = False
    if goal_name:
        g_obj = _goal_from_snap(_fetch_room_snapshot(room, goal_name, head_only=True), goal_name)
        if g_obj is not None and int(g_obj.get("generation", -1)) == generation:
            g_index = _get_cached_goal_index(room, g_obj)
            g_stubs = g_index.stubs
            spec = get_task_spec(str(g_obj.get("task") or "image"))
            sharded = _is_goal_sharded(spec, len(g_stubs))
            if active_only and g_stubs:
                pop_hashes = g_index.hashes

    items = (
        fetch_room_evals(
            room,
            goal_name,
            generation,
            trusted_only=trusted_only,
            include_traj=False,
            genome_hashes=pop_hashes,
        )
        if goal_name
        else []
    )
    if g_stubs:
        eval_map = {str(it["hash"]): it for it in items if isinstance(it, dict) and it.get("hash")}
        rankings = compute_goal_eval_rankings(list(g_stubs), eval_map)

    live_peers = _get_live_room_peer_count(room)
    eff_quorum = resolve_effective_quorum(cfg, active_peers=live_peers, sharded=sharded)
    payload_obj = {
        "ok": True,
        "room": room,
        "goal": goal_name,
        "generation": generation,
        "sharded": sharded,
        "quorum": eff_quorum,
        "contributor_id": cid,
        "items": items,
        "rankings": rankings,
    }
    raw_bytes = json.dumps(payload_obj, separators=(",", ":")).encode("utf-8")
    accept_enc = (request.META.get("HTTP_ACCEPT_ENCODING") or "").lower()
    etag_suffix = "-gz" if "gzip" in accept_enc else ""
    etag = f'W/"{hashlib.sha1(raw_bytes).hexdigest()[:16]}{etag_suffix}"'
    if not cid_is_new and request.headers.get("If-None-Match") == etag:
        not_mod = HttpResponseNotModified()
        not_mod["ETag"] = etag
        not_mod["Cache-Control"] = "private, no-cache"
        not_mod["Vary"] = "Accept-Encoding"
        return not_mod
    resp = HttpResponse(raw_bytes, content_type="application/json")
    resp["ETag"] = etag
    resp["Cache-Control"] = "private, no-cache"
    resp["Vary"] = "Accept-Encoding"
    return _attach_contributor_cookie(resp, cid, cid_is_new)


@csrf_exempt
@require_POST
def api_crowd_results(request: HttpRequest, room: str) -> JsonResponse:
    """Submit browser WASM or local-runner MuJoCo simulation attestations toward room consensus."""
    trust_cfg = get_config().trust
    if not trust_cfg.enabled:
        return JsonResponse({"ok": True, "accepted": 0, "items": [], "disabled": True})
    if size_err := _check_request_content_length(request):
        return JsonResponse({"ok": False, "error": size_err}, status=413)
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return JsonResponse({"ok": False, "error": "Invalid JSON."}, status=400)
    if not isinstance(payload, dict):
        return JsonResponse({"ok": False, "error": "Invalid crowd payload."}, status=400)

    goal_name = str(payload.get("goal") or "").strip()[:64]
    try:
        generation = int(payload.get("generation", 0))
    except (TypeError, ValueError):
        generation = 0
    results = payload.get("results")
    if not goal_name or not isinstance(results, list) or not results:
        return JsonResponse({"ok": False, "error": "goal, generation, and non-empty results are required."}, status=400)

    explicit_peer = str(payload.get("peer_id") or "").strip()
    cid, cid_is_new = _resolve_contributor_id(request, explicit_peer_id=explicit_peer)

    goal_obj = _goal_from_snap(_fetch_room_snapshot(room, goal_name, head_only=True), goal_name)
    g_idx = _get_cached_goal_index(room, goal_obj) if goal_obj is not None else None
    goal_stubs = g_idx.stubs if g_idx is not None else ()
    task_spec = get_task_spec(str((goal_obj or {}).get("task") or "image"))
    pop_total = len(goal_stubs)
    is_sharded = bool(payload.get("sharded")) or _is_goal_sharded(task_spec, pop_total)

    live_peers = _get_live_room_peer_count(room)
    outcome = submit_eval_attestations(
        room=room,
        goal=goal_name,
        generation=generation,
        results=results,
        contributor_id=cid,
        authority=False,
        active_peers=live_peers,
        sharded=is_sharded,
    )
    items = outcome["items"]
    changed_items = outcome.get("changed_items") or []
    if changed_items:
        ws_items = [{k: v for k, v in it.items() if k != "traj_sample"} for it in changed_items]
        _broadcast_room_event(
            room,
            {
                "type": "worker_eval_updated",
                "goal": goal_name,
                "generation": generation,
                "items": ws_items,
            },
            goal=goal_name,
        )
    active_gen_idx = g_idx if goal_obj is not None and int(goal_obj.get("generation", -1)) == generation else None
    pop_verified, phase = _sync_goal_sim_phase(room, goal_name, generation, goal_index=active_gen_idx)
    include_rankings = payload.get("include_rankings", True) is not False
    rankings: list[dict[str, object]] = []
    if include_rankings and active_gen_idx is not None and goal_stubs:
        _, cached_eval_map = get_cached_gen_eval_state(room, goal_name, generation)
        rankings = compute_goal_eval_rankings(list(goal_stubs), cached_eval_map)

    ack_items = [
        {
            "hash": it["hash"],
            "summary": it["summary"],
            "trust": it["trust"],
        }
        for it in items
    ]
    resp = JsonResponse({
        "ok": True,
        "accepted": len(ack_items),
        "sharded": is_sharded,
        "verified_count": outcome["verified_count"],
        "goal_verified_count": pop_verified,
        "completed_count": pop_verified,
        "total_count": pop_total,
        "phase": phase,
        "quorum": outcome.get("quorum", trust_cfg.quorum),
        "contributor_reputation": outcome["contributor_reputation"],
        "quarantined": outcome["quarantined"],
        "items": ack_items,
        "rankings": rankings,
    })
    return _attach_contributor_cookie(resp, cid, cid_is_new)


def _recommended_shard_batch(
    default_shard_size: int,
    pop_size: int,
    *,
    scale_threshold: int = 128,
    pop_divisor: int = 32,
    min_scaled: int = 4,
    max_scaled: int = 32,
) -> int:
    if pop_size < scale_threshold:
        return default_shard_size
    return max(default_shard_size, min(max_scaled, max(min_scaled, pop_size // pop_divisor)))


@require_GET
def export_crowd_worker_script(request: HttpRequest) -> HttpResponse:
    """Download a standalone PEP 723 ``uv run`` local MuJoCo crowd shard simulator script."""
    room = request.GET.get("room", "lobby").strip()[:64] or "lobby"
    goal = request.GET.get("goal", "").strip()[:64]
    g_obj = _goal_from_snap(_fetch_room_snapshot(room, goal, head_only=True), goal)
    spec = get_task_spec(str((g_obj or {}).get("task") or "microduck"))
    pop_sz = int((g_obj or {}).get("population_size") or spec.default_pop_size)
    rec_batch = _recommended_shard_batch(spec.crowd_shard_size, pop_sz)
    base_path = _resolve_base_path(request)
    scheme = request.scheme or "http"
    host = request.get_host()
    server_url = f"{scheme}://{host}{base_path}"
    script = build_crowd_worker_script(room, server_url, goal_name=goal, default_batch_size=rec_batch)
    resp = HttpResponse(script, content_type="text/x-python; charset=utf-8")
    resp["Content-Disposition"] = f'attachment; filename="crowd_worker_{slugify(room) or "room"}.py"'
    return resp


@require_GET
def export_worker_script(request: HttpRequest) -> HttpResponse:
    """Download a pre-configured uv external researcher simulation worker template."""
    room = request.GET.get("room", "my-room").strip()[:64] or "my-room"
    base_path = _resolve_base_path(request)
    scheme = request.scheme or "http"
    host = request.get_host()
    server_url = f"{scheme}://{host}{base_path}"
    script = build_worker_script(room, server_url)
    resp = HttpResponse(script, content_type="text/x-python; charset=utf-8")
    resp["Content-Disposition"] = f'attachment; filename="worker_{slugify(room) or "room"}.py"'
    return resp


def _parse_single_byte_range(header: str, total_size: int) -> tuple[int, int] | None:
    if not header or not header.startswith("bytes="):
        return None
    spec = header[len("bytes="):].strip()
    if "," in spec or "-" not in spec:
        return None
    start_s, end_s = spec.split("-", 1)
    if not start_s and not end_s:
        return None
    if not start_s:
        suffix = int(end_s)
        if suffix <= 0:
            raise ValueError("Invalid suffix byte range")
        start = max(0, total_size - suffix)
        end = total_size - 1
        return start, end
    start = int(start_s)
    end = int(end_s) if end_s else total_size - 1
    if start < 0 or start >= total_size or end < start:
        raise ValueError("Byte range out of bounds")
    return start, min(end, total_size - 1)


def _attach_media_cache_headers(resp: HttpResponse, etag: str) -> HttpResponse:
    resp["Accept-Ranges"] = "bytes"
    resp["ETag"] = etag
    resp["Cache-Control"] = "public, max-age=31536000, immutable"
    resp["X-Content-Type-Options"] = "nosniff"
    return resp


@require_GET
def serve_media(request: HttpRequest, rel_path: str) -> HttpResponse:
    """Serve cached media assets with byte-range seeking and immutable cache headers."""
    clean_rel = PurePosixPath(rel_path)
    if (
        clean_rel.is_absolute()
        or ".." in clean_rel.parts
        or not clean_rel.parts
        or clean_rel.suffix.lower() not in _ALLOWED_MEDIA_SUFFIXES
    ):
        raise Http404("Media not found")

    media_root = Path(settings.MEDIA_ROOT).resolve()
    target = (media_root / Path(*clean_rel.parts)).resolve()
    try:
        target.relative_to(media_root)
    except ValueError as exc:
        raise Http404("Media not found") from exc
    if not target.is_file():
        raise Http404("Media not found")

    stat = target.stat()
    size = stat.st_size
    suffix_lower = target.suffix.lower()
    gz_sidecar = target.with_name(target.name + ".gz") if suffix_lower == ".json" else None
    accept_enc = (request.META.get("HTTP_ACCEPT_ENCODING") or "").lower()
    use_gz_sidecar = (
        gz_sidecar is not None
        and "gzip" in accept_enc
        and gz_sidecar.is_file()
        and gz_sidecar.stat().st_mtime_ns == stat.st_mtime_ns
    )
    etag = f'W/"{size:x}-{stat.st_mtime_ns:x}{"-gz" if use_gz_sidecar else ""}"'
    if request.headers.get("If-None-Match") == etag:
        not_mod = _attach_media_cache_headers(HttpResponse(status=304), etag)
        if suffix_lower == ".json":
            not_mod["Vary"] = "Accept-Encoding"
        return not_mod

    content_type, _ = mimetypes.guess_type(target.name)
    if suffix_lower == ".mp4":
        content_type = "video/mp4"
    elif suffix_lower == ".webm":
        content_type = "video/webm"
    elif suffix_lower == ".json":
        content_type = "application/json; charset=utf-8"
    content_type = content_type or "application/octet-stream"

    if use_gz_sidecar and gz_sidecar is not None:
        gz_bytes = gz_sidecar.read_bytes()
        resp = HttpResponse(gz_bytes, content_type=content_type)
        resp["Content-Encoding"] = "gzip"
        resp["Content-Length"] = str(len(gz_bytes))
        resp["Vary"] = "Accept-Encoding"
        return _attach_media_cache_headers(resp, etag)

    range_header = request.headers.get("Range", "").strip()
    if range_header and size > 0:
        try:
            byte_range = _parse_single_byte_range(range_header, size)
        except ValueError:
            resp = HttpResponse(status=416)
            resp["Content-Range"] = f"bytes */{size}"
            return resp
        if byte_range is not None:
            start, end = byte_range
            length = end - start + 1
            with target.open("rb") as fh:
                fh.seek(start)
                chunk = fh.read(length)
            resp = HttpResponse(chunk, status=206, content_type=content_type)
            resp["Content-Range"] = f"bytes {start}-{end}/{size}"
            resp["Content-Length"] = str(length)
            return _attach_media_cache_headers(resp, etag)

    resp = FileResponse(target.open("rb"), content_type=content_type)
    resp["Content-Length"] = str(size)
    return _attach_media_cache_headers(resp, etag)


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
    *,
    timeout: float = 5.0,
) -> tuple[int, str]:
    ws_host = "127.0.0.1" if settings.WS_HOST == "0.0.0.0" else settings.WS_HOST
    url = f"http://{ws_host}:{settings.WS_PORT}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"X-Admin-Password": password}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
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


def _room_goal_tasks(room: str) -> dict[str, str]:
    tasks: dict[str, str] = (
        {name: task for name, task, _ in ws_bridge.DEFAULT_GOAL_SPECS}
        if room == "lobby"
        else {}
    )
    row = RoomEvolutionState.objects.filter(room=room).first()
    if row is None:
        return tasks
    try:
        parsed = json.loads(row.state_json)
    except ValueError:
        return tasks
    if isinstance(parsed, dict) and isinstance(parsed.get("goals"), list):
        for g in parsed["goals"]:
            if isinstance(g, dict) and g.get("name"):
                tasks[str(g["name"])] = normalize_task_id(str(g.get("task", "image")))
    return tasks


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

    task_map = _room_goal_tasks(room)
    reset_goals: list[str] = []
    for goal_name, pop_size in targets:
        reset_populora_goal_runtime(room, goal_name)
        reset_goal_shard_state(room, goal_name)
        _invalidate_goal_index_cache(room, goal_name)
        genomes = create_task_population(task_map.get(goal_name, "image"), int(pop_size))
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

    reset_goal_shard_state(room, "" if is_all else goal)
    _invalidate_goal_index_cache(room, "" if is_all else goal)
    if room == "lobby":
        lobby_specs = get_config().get_lobby_goal_specs()
        known_goals = list(pop_map.keys()) if pop_map else [name for name, _, _ in lobby_specs]
        task_by_goal = _room_goal_tasks(room)
        cap_by_goal = {name: cap for name, _, cap in lobby_specs}
        if is_all:
            goals_payload = [
                ws_bridge._build_goal_entry(
                    g_name,
                    task_by_goal.get(g_name, "image"),
                    int(pop_map.get(g_name, cap_by_goal.get(g_name) or settings.POPULATION_SIZE)),
                )
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
        default_p = cap_by_goal.get(goal) or settings.POPULATION_SIZE
        genomes = create_task_population(
            task_by_goal.get(goal, "image"), int(pop_map.get(goal, default_p))
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
    for g_cl in cleared:
        reset_populora_goal_runtime(room, g_cl)
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
