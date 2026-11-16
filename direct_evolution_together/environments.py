"""Generalized task & control environment registry backed by Stanley's CPPN + HyperNEAT."""

from __future__ import annotations

from dataclasses import dataclass
import io
import math
import os
from pathlib import Path
import threading
from typing import Callable, Iterable, Sequence

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw
import torch

from direct_evolution_together.cppn import (
    CPPN,
    MAX_POPULATION_SIZE,
    MIN_POPULATION_SIZE,
    POPULATION_SIZE,
    build_uv_script_header,
    create_random_population,
    genome_content_hash,
)

CARTPOLE_GRAVITY = 9.8
CARTPOLE_MASS_CART = 1.0
CARTPOLE_MASS_POLE = 0.1
CARTPOLE_TOTAL_MASS = CARTPOLE_MASS_CART + CARTPOLE_MASS_POLE
CARTPOLE_HALF_LENGTH = 0.5
CARTPOLE_POLEMASS_LENGTH = CARTPOLE_MASS_POLE * CARTPOLE_HALF_LENGTH
CARTPOLE_FORCE_MAG = 10.0
CARTPOLE_TAU = 0.02
CARTPOLE_THETA_THRESHOLD = 12.0 * 2.0 * math.pi / 360.0
CARTPOLE_X_THRESHOLD = 2.4
CARTPOLE_MAX_STEPS = 500
CARTPOLE_OBS_SCALE = (CARTPOLE_X_THRESHOLD, 3.0, CARTPOLE_THETA_THRESHOLD, 3.0)

CONTROL_MAX_POP_SIZE = 32
DEFAULT_CONTROL_POP_SIZE = 16
DEFAULT_CONTROL_MIN_VOTES = 30
DEFAULT_CONTROL_ELITE_FRAC = 0.0


def mulberry32(seed: int) -> Callable[[], float]:
    """Deterministic 32-bit PRNG matching JavaScript Math.imul bit-for-bit."""
    a = int(seed) & 0xFFFFFFFF

    def _next() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((a ^ (a >> 15)) * (1 | a)) & 0xFFFFFFFF
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) ^ t) & 0xFFFFFFFF
        t = (t ^ (t >> 14)) & 0xFFFFFFFF
        return t / 4294967296.0

    return _next


class HyperNEATSubstrate:
    """Maps any fixed-topology 4-input CPPN onto an arbitrary MLP substrate (obs -> hidden -> act)."""

    def __init__(self, dims: tuple[int, ...]) -> None:
        if len(dims) < 2:
            raise ValueError("Substrate must have at least input and output layers.")
        self.dims = tuple(int(d) for d in dims)

        queries: list[list[float]] = []
        w_slices: list[torch.Tensor] = []
        b_slices: list[torch.Tensor] = []
        num_layers = len(self.dims) - 1
        zs = np.linspace(-1.0, 1.0, num_layers + 1, dtype=np.float32)
        layer_xs = [
            np.array([0.0], dtype=np.float32)
            if d == 1
            else np.linspace(-1.0, 1.0, d, dtype=np.float32)
            for d in self.dims
        ]

        for l_idx, (d_in, d_out) in enumerate(zip(self.dims[:-1], self.dims[1:])):
            z_mid = float(zs[l_idx] + zs[l_idx + 1]) * 0.5
            w_idx = torch.zeros((d_in, d_out), dtype=torch.long)
            for i in range(d_in):
                for j in range(d_out):
                    w_idx[i, j] = len(queries)
                    queries.append(
                        [float(layer_xs[l_idx][i]), float(layer_xs[l_idx + 1][j]), z_mid, 1.0]
                    )
            b_idx = torch.zeros((d_out,), dtype=torch.long)
            for j in range(d_out):
                b_idx[j] = len(queries)
                queries.append(
                    [0.0, float(layer_xs[l_idx + 1][j]), float(zs[l_idx + 1]), -1.0]
                )
            w_slices.append(w_idx)
            b_slices.append(b_idx)

        self.coords = torch.tensor(queries, dtype=torch.float32)
        self.weight_slices = w_slices
        self.bias_slices = b_slices

    @torch.no_grad()
    def decode(self, genome_or_net: dict | CPPN) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Query the CPPN across all substrate coordinates with variance-preserving normalization."""
        net = genome_or_net if isinstance(genome_or_net, CPPN) else CPPN.from_dict(genome_or_net)
        # Center channel 0 from [0, 1] -> [-1, 1]
        raw = net(self.coords)[:, 0].cpu().numpy().astype(np.float64) * 2.0 - 1.0

        weights: list[np.ndarray] = []
        biases: list[np.ndarray] = []
        for w_idx, b_idx in zip(self.weight_slices, self.bias_slices):
            w = raw[w_idx.numpy()]
            mean = float(w.mean())
            std = float(np.sqrt(np.mean((w - mean) ** 2)))
            fan_in = w.shape[0]
            w_norm = (w - mean) / (std + 1e-6) * math.sqrt(1.0 / fan_in)
            b_norm = raw[b_idx.numpy()] * 0.1
            weights.append(w_norm.astype(np.float64))
            biases.append(b_norm.astype(np.float64))
        return weights, biases

    @staticmethod
    def forward(
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        obs: np.ndarray | Iterable[float],
    ) -> np.ndarray:
        """Evaluate the decoded substrate MLP, emitting continuous actions in [-1, 1]."""
        h = np.asarray(obs, dtype=np.float64)
        for idx, (w, b) in enumerate(zip(weights, biases)):
            h = h @ w + b
            if idx < len(weights) - 1:
                h = np.tanh(h)
        return np.tanh(h)


_SUBSTRATE_CACHE: dict[tuple[int, ...], HyperNEATSubstrate] = {}
_SUBSTRATE_LOCK = threading.Lock()


def get_substrate(dims: tuple[int, ...]) -> HyperNEATSubstrate:
    key = tuple(int(d) for d in dims)
    with _SUBSTRATE_LOCK:
        sub = _SUBSTRATE_CACHE.get(key)
        if sub is None:
            sub = HyperNEATSubstrate(key)
            _SUBSTRATE_CACHE[key] = sub
        return sub


@dataclass(frozen=True)
class TaskSpec:
    """Unified descriptor for image and control evolution environments."""

    task_id: str
    kind: str  # "image" | "control"
    label: str
    description: str
    default_pop_size: int
    max_pop_size: int
    default_min_votes: int
    client_sim: bool
    supports_video: bool
    default_elite_frac: float | None = None
    obs_dim: int = 0
    hidden_dims: tuple[int, ...] = ()
    action_dim: int = 0
    continuous: bool = True
    max_steps: int = 0
    eval_seeds: tuple[int, ...] = ()

    @property
    def substrate_dims(self) -> tuple[int, ...]:
        if self.kind != "control":
            return ()
        return (self.obs_dim, *self.hidden_dims, self.action_dim)

    def to_client_dict(self) -> dict[str, object]:
        return {
            "task": self.task_id,
            "kind": self.kind,
            "label": self.label,
            "description": self.description,
            "defaultPopSize": self.default_pop_size,
            "maxPopSize": self.max_pop_size,
            "defaultMinVotes": self.default_min_votes,
            "defaultEliteFrac": self.default_elite_frac,
            "clientSim": self.client_sim,
            "supportsVideo": self.supports_video,
            "obsDim": self.obs_dim,
            "hiddenDims": list(self.hidden_dims),
            "actionDim": self.action_dim,
            "continuous": self.continuous,
            "maxSteps": self.max_steps,
            "evalSeeds": list(self.eval_seeds),
        }


TASK_SPECS: dict[str, TaskSpec] = {
    "image": TaskSpec(
        task_id="image",
        kind="image",
        label="Image (Picbreeder)",
        description="2D CPPN pattern synthesis evaluated over (x, y, r, 1) coordinates.",
        default_pop_size=POPULATION_SIZE,
        max_pop_size=MAX_POPULATION_SIZE,
        default_min_votes=0,
        client_sim=True,
        supports_video=False,
    ),
    "cartpole": TaskSpec(
        task_id="cartpole",
        kind="control",
        label="CartPole (Continuous Control)",
        description="Balance a pole via continuous horizontal force using HyperNEAT on the CPPN.",
        default_pop_size=DEFAULT_CONTROL_POP_SIZE,
        max_pop_size=CONTROL_MAX_POP_SIZE,
        default_min_votes=DEFAULT_CONTROL_MIN_VOTES,
        client_sim=True,
        supports_video=True,
        default_elite_frac=DEFAULT_CONTROL_ELITE_FRAC,
        obs_dim=4,
        hidden_dims=(8,),
        action_dim=1,
        continuous=True,
        max_steps=CARTPOLE_MAX_STEPS,
        eval_seeds=(0, 1, 2),
    ),
}


def normalize_task_id(raw: str | None) -> str:
    cleaned = (raw or "").strip().lower()
    return cleaned if cleaned in TASK_SPECS else "image"


def get_task_spec(task_id: str | None) -> TaskSpec:
    return TASK_SPECS[normalize_task_id(task_id)]


def get_task_specs_payload() -> dict[str, dict[str, object]]:
    return {k: spec.to_client_dict() for k, spec in TASK_SPECS.items()}


def clamp_pop_for_task(task_id: str | None, pop_size: int) -> int:
    spec = get_task_spec(task_id)
    return max(MIN_POPULATION_SIZE, min(spec.max_pop_size, int(pop_size)))


def default_pop_for_task(task_id: str | None, fallback: int | None = None) -> int:
    spec = get_task_spec(task_id)
    if spec.kind == "control":
        return spec.default_pop_size
    base = spec.default_pop_size if fallback is None else int(fallback)
    return max(MIN_POPULATION_SIZE, min(spec.max_pop_size, base))


def default_elite_for_task(task_id: str | None, fallback: float) -> float:
    spec = get_task_spec(task_id)
    raw = spec.default_elite_frac if spec.default_elite_frac is not None else fallback
    return max(0.0, min(0.5, float(raw)))


def create_task_population(
    task_id: str | None,
    pop_size: int,
    *,
    max_gen0_steps: float = 60.0,
) -> list[dict]:
    """Generate a Gen-0 population for `task_id`, rejecting pre-solved control founders."""
    clean_task = normalize_task_id(task_id)
    spec = get_task_spec(clean_task)
    eff_pop = clamp_pop_for_task(clean_task, pop_size)
    if spec.kind != "control":
        return create_random_population(pop_size=eff_pop)

    founders: list[dict] = []
    max_attempts = eff_pop * 12
    attempts = 0
    while len(founders) < eff_pop and attempts < max_attempts:
        attempts += 1
        cand = CPPN.random().to_dict()
        ev = evaluate_control_genome(clean_task, cand, seeds=spec.eval_seeds)
        if ev["meanSteps"] < max_gen0_steps:
            founders.append(cand)
    while len(founders) < eff_pop:
        founders.append(CPPN.random().to_dict())
    return founders


def cartpole_reset(seed: int) -> tuple[float, float, float, float]:
    """Sample Gymnasium-range [-0.05, 0.05] initial CartPole state using mulberry32."""
    rng = mulberry32(seed)
    return (
        (rng() * 2.0 - 1.0) * 0.05,
        (rng() * 2.0 - 1.0) * 0.05,
        (rng() * 2.0 - 1.0) * 0.05,
        (rng() * 2.0 - 1.0) * 0.05,
    )


def cartpole_step(
    state: tuple[float, float, float, float],
    action_u: float,
) -> tuple[tuple[float, float, float, float], float, bool]:
    """Advance continuous-force CartPole one 20ms Euler step."""
    x, x_dot, theta, theta_dot = state
    u = max(-1.0, min(1.0, float(action_u)))
    force = u * CARTPOLE_FORCE_MAG
    costheta = math.cos(theta)
    sintheta = math.sin(theta)

    temp = (force + CARTPOLE_POLEMASS_LENGTH * theta_dot * theta_dot * sintheta) / CARTPOLE_TOTAL_MASS
    thetaacc = (CARTPOLE_GRAVITY * sintheta - costheta * temp) / (
        CARTPOLE_HALF_LENGTH
        * (4.0 / 3.0 - CARTPOLE_MASS_POLE * costheta * costheta / CARTPOLE_TOTAL_MASS)
    )
    xacc = temp - CARTPOLE_POLEMASS_LENGTH * thetaacc * costheta / CARTPOLE_TOTAL_MASS

    nx = x + CARTPOLE_TAU * x_dot
    nx_dot = x_dot + CARTPOLE_TAU * xacc
    ntheta = theta + CARTPOLE_TAU * theta_dot
    ntheta_dot = theta_dot + CARTPOLE_TAU * thetaacc
    terminated = (
        nx < -CARTPOLE_X_THRESHOLD
        or nx > CARTPOLE_X_THRESHOLD
        or ntheta < -CARTPOLE_THETA_THRESHOLD
        or ntheta > CARTPOLE_THETA_THRESHOLD
    )
    return (nx, nx_dot, ntheta, ntheta_dot), force, terminated


def cartpole_rollout(
    genome_or_net: dict | CPPN,
    *,
    seed: int = 0,
    max_steps: int = CARTPOLE_MAX_STEPS,
) -> dict[str, object]:
    """Run one deterministic continuous CartPole episode and return full trace + telemetry."""
    spec = TASK_SPECS["cartpole"]
    substrate = get_substrate(spec.substrate_dims)
    weights, biases = substrate.decode(genome_or_net)

    state = cartpole_reset(seed)
    states: list[tuple[float, float, float, float]] = [state]
    forces: list[float] = []
    abs_x_sum = 0.0
    abs_theta_sum = 0.0

    for _ in range(max_steps):
        x, x_dot, theta, theta_dot = state
        obs = (
            x / CARTPOLE_OBS_SCALE[0],
            x_dot / CARTPOLE_OBS_SCALE[1],
            theta / CARTPOLE_OBS_SCALE[2],
            theta_dot / CARTPOLE_OBS_SCALE[3],
        )
        u = float(substrate.forward(weights, biases, obs)[0])
        next_state, force, terminated = cartpole_step(state, u)
        forces.append(force)
        states.append(next_state)
        abs_x_sum += abs(x)
        abs_theta_sum += abs(theta)
        state = next_state
        if terminated:
            break

    steps = len(forces)
    return {
        "seed": int(seed),
        "steps": steps,
        "return": float(steps),
        "mean_abs_x": abs_x_sum / max(1, steps),
        "mean_abs_theta": abs_theta_sum / max(1, steps),
        "states": states,
        "forces": forces,
    }


def evaluate_control_genome(
    task_id: str,
    genome_or_net: dict | CPPN,
    seeds: Sequence[int] | None = None,
) -> dict[str, object]:
    """Evaluate a CPPN genome across deterministic seeds for a registered control task."""
    spec = get_task_spec(task_id)
    if spec.task_id != "cartpole":
        raise ValueError(f"Task '{task_id}' does not support control evaluation.")
    eval_seeds = tuple(spec.eval_seeds if seeds is None else seeds)
    episodes = [cartpole_rollout(genome_or_net, seed=s, max_steps=spec.max_steps) for s in eval_seeds]
    step_list = [int(ep["steps"]) for ep in episodes]
    mean_steps = sum(step_list) / max(1, len(step_list))
    min_steps = min(step_list) if step_list else 0
    mean_abs_x = sum(float(ep["mean_abs_x"]) for ep in episodes) / max(1, len(episodes))
    return {
        "task": spec.task_id,
        "seeds": list(eval_seeds),
        "stepsPerSeed": step_list,
        "meanSteps": round(mean_steps, 1),
        "minSteps": int(min_steps),
        "meanAbsX": round(mean_abs_x, 3),
        "solved": min_steps >= 490,
    }


def _render_cartpole_frame(
    state: tuple[float, float, float, float],
    force: float,
    step_idx: int,
    max_steps: int,
    width: int = 480,
    height: int = 320,
) -> np.ndarray:
    """Render one crisp CartPole frame as an RGB uint8 numpy array."""
    img = Image.new("RGB", (width, height), (9, 9, 11))
    draw = ImageDraw.Draw(img)

    x, _, theta, _ = state
    track_y = int(height * 0.68)
    margin = 36
    usable_w = width - 2 * margin
    scale = usable_w / (2.0 * CARTPOLE_X_THRESHOLD)
    cx = width * 0.5 + x * scale

    # Subtle background grid
    for gx in range(margin, width - margin + 1, max(24, usable_w // 8)):
        draw.line([(gx, 38), (gx, track_y + 24)], fill=(26, 26, 32), width=1)
    # Track and failure rails
    draw.line([(margin, track_y), (width - margin, track_y)], fill=(82, 82, 91), width=2)
    draw.line([(width * 0.5, track_y - 6), (width * 0.5, track_y + 6)], fill=(113, 113, 122), width=1)
    for bx in (margin, width - margin):
        draw.line([(bx, track_y - 12), (bx, track_y + 12)], fill=(251, 113, 133), width=2)

    # Cart body
    cart_w, cart_h = 52, 26
    draw.rounded_rectangle(
        [cx - cart_w * 0.5, track_y - cart_h * 0.5, cx + cart_w * 0.5, track_y + cart_h * 0.5],
        radius=5,
        fill=(228, 228, 231),
        outline=(250, 250, 250),
        width=1,
    )

    # Pole (visual length proportional to height)
    pole_px = int(height * 0.36)
    pivot_y = track_y - cart_h * 0.15
    tip_x = cx + math.sin(theta) * pole_px
    tip_y = pivot_y - math.cos(theta) * pole_px
    pole_color = (52, 211, 153) if abs(theta) < 0.08 else ((245, 196, 81) if abs(theta) < 0.16 else (251, 113, 133))
    draw.line([(cx, pivot_y), (tip_x, tip_y)], fill=pole_color, width=7)
    draw.ellipse([cx - 5, pivot_y - 5, cx + 5, pivot_y + 5], fill=(24, 24, 27), outline=(250, 250, 250), width=2)

    # Force arrow
    if abs(force) > 0.2:
        arrow_len = (force / CARTPOLE_FORCE_MAG) * 44.0
        ay = track_y + cart_h * 0.5 + 14
        ax_end = cx + arrow_len
        draw.line([(cx, ay), (ax_end, ay)], fill=(56, 189, 248), width=3)

    # Simple progress bar at top
    bar_w = int((width - 2 * margin) * min(1.0, step_idx / max(1, max_steps)))
    draw.rectangle([margin, 14, width - margin, 20], fill=(31, 31, 38))
    if bar_w > 0:
        bar_col = (52, 211, 153) if step_idx >= 490 else (245, 196, 81)
        draw.rectangle([margin, 14, margin + bar_w, 20], fill=bar_col)
    draw.text((margin, 26), f"step {step_idx}/{max_steps}  x={x:+.2f}  th={math.degrees(theta):+.1f}deg  F={force:+.1f}N", fill=(161, 161, 170))
    return np.asarray(img, dtype=np.uint8)


def _render_cartpole_poster_png(rollout: dict[str, object], width: int = 320, height: int = 240) -> bytes:
    states = rollout["states"]
    forces = rollout["forces"]
    steps = int(rollout["steps"])
    last_state = states[-1]
    last_force = forces[-1] if forces else 0.0
    frame = _render_cartpole_frame(last_state, last_force, steps, CARTPOLE_MAX_STEPS, width=width, height=height)
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
    if not spec.supports_video or spec.task_id != "cartpole":
        raise ValueError(f"Task '{task_id}' does not support video rendering.")

    seed = int(seed)
    g_hash = genome_content_hash(genome, length=20)
    key = f"{spec.task_id}_{g_hash}_s{seed}"
    renders_dir = Path(media_root) / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(Path(media_root), 0o755)
    os.chmod(renders_dir, 0o755)

    mp4_path = renders_dir / f"{key}.mp4"
    png_path = renders_dir / f"{key}.png"

    with _get_render_lock(key):
        rollout = cartpole_rollout(genome, seed=seed, max_steps=spec.max_steps)
        if not png_path.is_file():
            png_bytes = _render_cartpole_poster_png(rollout, width=320, height=240)
            tmp_png = renders_dir / f".{key}.{os.getpid()}.png.tmp"
            tmp_png.write_bytes(png_bytes)
            os.chmod(tmp_png, 0o644)
            os.replace(tmp_png, png_path)
        else:
            png_path.touch()

        if not mp4_path.is_file():
            width, height, fps = 480, 320, 50
            tmp_mp4 = renders_dir / f".{key}.{os.getpid()}.tmp.mp4"
            states = rollout["states"]
            forces = rollout["forces"]
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
                for step_idx in range(len(states)):
                    force = forces[min(step_idx, len(forces) - 1)] if forces else 0.0
                    frame = _render_cartpole_frame(
                        states[step_idx],
                        force,
                        min(step_idx, int(rollout["steps"])),
                        spec.max_steps,
                        width=width,
                        height=height,
                    )
                    writer.send(frame)
            finally:
                writer.close()
            os.chmod(tmp_mp4, 0o644)
            os.replace(tmp_mp4, mp4_path)
        else:
            mp4_path.touch()

        _prune_media_cache(renders_dir, max_cache_bytes)

    return {
        "key": key,
        "mp4_rel": f"renders/{key}.mp4",
        "png_rel": f"renders/{key}.png",
        "steps": int(rollout["steps"]),
        "return": float(rollout["return"]),
        "meanAbsX": round(float(rollout["mean_abs_x"]), 3),
        "seed": seed,
    }


def build_control_runner_script(task_id: str, weights_filename: str = "weights.pt") -> str:
    """Return a standalone uv script that evaluates an exported CPPN policy on a control task."""
    spec = get_task_spec(task_id)
    return "\n".join(
        [
            *build_uv_script_header(extra_deps=("imageio-ffmpeg",)),
            "import sys",
            "from pathlib import Path",
            "from direct_evolution_together.cppn import CPPN",
            "from direct_evolution_together.environments import evaluate_control_genome, render_task_media",
            "",
            f'net = CPPN.init_and_load(sys.argv[1] if len(sys.argv) > 1 else "{weights_filename}")',
            f'summary = evaluate_control_genome("{spec.task_id}", net, seeds=tuple(range(10)))',
            'print(f"Evaluated 10 held-out episodes: mean={summary[\'meanSteps\']}/500, min={summary[\'minSteps\']}/500, solved={summary[\'solved\']}")',
            f'media = render_task_media("{spec.task_id}", net.to_dict(), seed=0, media_root=Path("."))',
            'Path("out.mp4").write_bytes((Path(".") / media["mp4_rel"]).read_bytes())',
            'print("Voila! Saved out.mp4")',
            "",
        ]
    )
