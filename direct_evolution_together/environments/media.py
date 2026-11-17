"""Lazy MP4 + poster rendering with an on-disk LRU cache."""

from __future__ import annotations

import io
import os
from pathlib import Path
import threading

import imageio_ffmpeg
from PIL import Image

from direct_evolution_together.cppn import genome_content_hash

from .adapters import TASK_ADAPTERS, get_task_adapter
from .evaluation import _rollout_cached
from .registry import get_task_spec
from .types import ControlTaskAdapter


def _render_poster_png(
    adapter: ControlTaskAdapter,
    rollout: dict[str, object],
    max_steps: int,
    width: int = 320,
    height: int = 240,
) -> bytes:
    last_idx = max(0, int(rollout["steps"]))
    frame = adapter.render_frame(rollout, last_idx, max_steps, width, height)
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


_MEDIA_RENDER_LOCKS: dict[str, threading.Lock] = {}
_MEDIA_LOCKS_GUARD = threading.Lock()


def _get_render_lock(key: str) -> threading.Lock:
    with _MEDIA_LOCKS_GUARD:
        lock = _MEDIA_RENDER_LOCKS.get(key)
        if lock is None:
            if len(_MEDIA_RENDER_LOCKS) > 512:
                unlocked = [k for k, lk in _MEDIA_RENDER_LOCKS.items() if not lk.locked()]
                for k in unlocked[:256]:
                    _MEDIA_RENDER_LOCKS.pop(k, None)
            lock = threading.Lock()
            _MEDIA_RENDER_LOCKS[key] = lock
        return lock


def _prune_media_cache(renders_dir: Path, max_bytes: int) -> None:
    if max_bytes <= 0 or not renders_dir.exists():
        return
    files = [
        p for p in renders_dir.iterdir()
        if p.is_file() and not p.name.startswith(".") and ".tmp" not in p.name
    ]
    total = sum(p.stat().st_size for p in files)
    if total <= max_bytes:
        return
    files.sort(key=lambda p: p.stat().st_mtime)
    for p in files:
        if total <= max_bytes:
            break
        try:
            sz = p.stat().st_size
            p.unlink(missing_ok=True)
            total -= sz
        except OSError:
            pass


def render_task_media(
    task_id: str,
    genome: dict,
    *,
    seed: int = 0,
    media_root: Path,
    max_cache_bytes: int = 256 * 1024 * 1024,
) -> dict[str, object]:
    """Generate and cache a fast-start H.264 MP4 + PNG poster for a control policy on demand."""
    spec = get_task_spec(task_id)
    if not spec.supports_video or spec.task_id not in TASK_ADAPTERS:
        raise ValueError(f"Task '{task_id}' does not support video rendering.")

    adapter = get_task_adapter(spec.task_id)
    seed = int(seed)
    g_hash = genome_content_hash(genome, length=20)
    key = f"{spec.task_id}_{g_hash}_s{seed}"
    renders_dir = Path(media_root) / "renders"
    if not renders_dir.is_dir():
        renders_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(Path(media_root), 0o755)
        os.chmod(renders_dir, 0o755)

    mp4_path = renders_dir / f"{key}.mp4"
    png_path = renders_dir / f"{key}.png"

    with _get_render_lock(key):
        rollout = _rollout_cached(spec.task_id, genome, seed=seed, max_steps=spec.max_steps)
        if not png_path.is_file():
            png_bytes = _render_poster_png(adapter, rollout, spec.max_steps, width=320, height=240)
            tmp_png = renders_dir / f".{key}.{os.getpid()}.png.tmp"
            tmp_png.write_bytes(png_bytes)
            os.chmod(tmp_png, 0o644)
            os.replace(tmp_png, png_path)
        else:
            png_path.touch()

        if not mp4_path.is_file():
            width, height = 480, 320
            fps = max(10, min(60, round(1.0 / max(1e-3, spec.dt))))
            tmp_mp4 = renders_dir / f".{key}.{os.getpid()}.tmp.mp4"
            total_frames = int(rollout["steps"]) + 1
            writer = imageio_ffmpeg.write_frames(
                str(tmp_mp4),
                (width, height),
                fps=fps,
                codec="libx264",
                pix_fmt_in="rgb24",
                pix_fmt_out="yuv420p",
                output_params=["-crf", "24", "-preset", "veryfast", "-movflags", "+faststart"],
            )
            writer.send(None)
            try:
                for step_idx in range(total_frames):
                    frame = adapter.render_frame(rollout, step_idx, spec.max_steps, width, height)
                    writer.send(frame)
            finally:
                writer.close()
            os.chmod(tmp_mp4, 0o644)
            os.replace(tmp_mp4, mp4_path)
        else:
            mp4_path.touch()

        _prune_media_cache(renders_dir, max_cache_bytes)

    out: dict[str, object] = {
        "key": key,
        "mp4_rel": f"renders/{key}.mp4",
        "png_rel": f"renders/{key}.png",
        "steps": int(rollout["steps"]),
        "return": round(float(rollout["return"]), 2),
        "seed": seed,
    }
    if "mean_abs_x" in rollout:
        out["meanAbsX"] = round(float(rollout["mean_abs_x"]), 3)
    if "distance" in rollout:
        out["distance"] = round(float(rollout["distance"]), 2)
    return out
