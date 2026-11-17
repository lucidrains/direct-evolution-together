"""Generalized task & control environment registry backed by Stanley's CPPN + HyperNEAT."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
import io
import math
import os
from pathlib import Path
import threading
from typing import Callable, Iterable, Protocol, Sequence

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw
import torch

from direct_evolution_together.config import get_config

from direct_evolution_together.cppn import (
    CPPN,
    MAX_POPULATION_SIZE,
    MIN_POPULATION_SIZE,
    POPULATION_SIZE,
    build_uv_script_header,
    create_random_population,
    genome_content_hash,
)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

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

HALFCHEETAH_XML_PATH = ASSETS_DIR / "half_cheetah.xml"
HALFCHEETAH_FRAME_SKIP = 5
HALFCHEETAH_DT = 0.01 * HALFCHEETAH_FRAME_SKIP  # 0.05s (20 Hz control)
HALFCHEETAH_MAX_STEPS = 200  # 10.0s sprint
HALFCHEETAH_PHASE_FREQ = 2.0
HALFCHEETAH_PHASE_HARMONICS = 2
HALFCHEETAH_OBS_SCALE = (
    0.5, 1.0,  # rootz-0.7, rooty
    1.0, 1.0, 1.0,  # bthigh, bshin, bfoot
    1.0, 1.0, 1.0,  # fthigh, fshin, ffoot
    5.0, 2.0, 5.0,  # vx, vz, pitch_vel
    10.0, 10.0, 10.0, 10.0, 10.0, 10.0,  # joint vels
)
# Anatomical left-to-right (rear -> trunk/clock -> front) substrate coordinates
HALFCHEETAH_IN_COORDS = (
    0.0, 0.0,
    -0.85, -0.65, -0.45,
    0.45, 0.65, 0.85,
    0.0, 0.0, 0.0,
    -0.85, -0.65, -0.45,
    0.45, 0.65, 0.85,
    -0.25, 0.25, -0.12, 0.12,
)
HALFCHEETAH_OUT_COORDS = (-0.85, -0.65, -0.45, 0.45, 0.65, 0.85)
HALFCHEETAH_OBS_LABELS = (
    "z", "pitch",
    "bthigh", "bshin", "bfoot",
    "fthigh", "fshin", "ffoot",
    "vx", "vz", "omega_y",
    "vbthigh", "vbshin", "vbfoot",
    "vfthigh", "vfshin", "vffoot",
    "sin(wt)", "cos(wt)", "sin(2wt)", "cos(2wt)",
)
HALFCHEETAH_ACTION_LABELS = ("bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot")

CONTROL_MAX_POP_SIZE = 32
DEFAULT_CONTROL_POP_SIZE = 30
DEFAULT_CONTROL_MIN_VOTES = 20
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


def _default_layer_xs(dim: int) -> tuple[float, ...]:
    if dim <= 1:
        return (0.0,)
    return tuple(float(x) for x in np.linspace(-1.0, 1.0, dim, dtype=np.float32))


class HyperNEATSubstrate:
    """Maps any fixed-topology 4-input CPPN onto an arbitrary MLP substrate (obs -> hidden -> act)."""

    def __init__(
        self,
        dims: tuple[int, ...],
        *,
        layer_coords: tuple[tuple[float, ...] | None, ...] | None = None,
    ) -> None:
        if len(dims) < 2:
            raise ValueError("Substrate must have at least input and output layers.")
        self.dims = tuple(int(d) for d in dims)

        resolved_coords: list[tuple[float, ...]] = []
        for idx, dim in enumerate(self.dims):
            custom = layer_coords[idx] if (layer_coords is not None and idx < len(layer_coords)) else None
            if custom is not None:
                if len(custom) != dim:
                    raise ValueError(f"Expected {dim} coords for substrate layer {idx}, got {len(custom)}.")
                resolved_coords.append(tuple(float(v) for v in custom))
            else:
                resolved_coords.append(_default_layer_xs(dim))
        self.layer_coords = tuple(resolved_coords)

        queries: list[list[float]] = []
        w_slices: list[torch.Tensor] = []
        b_slices: list[torch.Tensor] = []
        num_layers = len(self.dims) - 1
        zs = np.linspace(-1.0, 1.0, num_layers + 1, dtype=np.float32)

        for l_idx, (d_in, d_out) in enumerate(zip(self.dims[:-1], self.dims[1:])):
            z_mid = float(zs[l_idx] + zs[l_idx + 1]) * 0.5
            xs_in = self.layer_coords[l_idx]
            xs_out = self.layer_coords[l_idx + 1]
            w_idx = torch.zeros((d_in, d_out), dtype=torch.long)
            for i in range(d_in):
                for j in range(d_out):
                    w_idx[i, j] = len(queries)
                    queries.append([xs_in[i], xs_out[j], z_mid, 1.0])
            b_idx = torch.zeros((d_out,), dtype=torch.long)
            for j in range(d_out):
                b_idx[j] = len(queries)
                queries.append([0.0, xs_out[j], float(zs[l_idx + 1]), -1.0])
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
        for w, b in zip(weights, biases):
            h = np.tanh(h @ w + b)
        return h

    @staticmethod
    def forward_with_trace(
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        obs: np.ndarray | Iterable[float],
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Evaluate the substrate MLP and return per-layer activations for visualization."""
        h = np.asarray(obs, dtype=np.float64)
        activations: list[np.ndarray] = [h]
        for w, b in zip(weights, biases):
            h = np.tanh(h @ w + b)
            activations.append(h)
        return h, activations


_SUBSTRATE_CACHE: dict[tuple[tuple[int, ...], tuple[tuple[float, ...], ...]], HyperNEATSubstrate] = {}
_SUBSTRATE_LOCK = threading.Lock()


def get_substrate(
    dims: tuple[int, ...],
    layer_coords: tuple[tuple[float, ...] | None, ...] | None = None,
) -> HyperNEATSubstrate:
    norm_dims = tuple(int(d) for d in dims)
    resolved_key_coords = tuple(
        tuple(float(v) for v in layer_coords[i])
        if (layer_coords is not None and i < len(layer_coords) and layer_coords[i] is not None)
        else _default_layer_xs(d)
        for i, d in enumerate(norm_dims)
    )
    key = (norm_dims, resolved_key_coords)
    with _SUBSTRATE_LOCK:
        sub = _SUBSTRATE_CACHE.get(key)
        if sub is None:
            sub = HyperNEATSubstrate(norm_dims, layer_coords=resolved_key_coords)
            _SUBSTRATE_CACHE[key] = sub
        return sub


@dataclass(frozen=True)
class TaskSpec:
    """Unified descriptor for image, built-in control, and external simulation tasks."""

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
    enabled: bool = True
    execution_policy: str = "client_preferred"
    crowd_contribute: bool = False
    allow_server_video: bool = True
    sim_modes: tuple[str, ...] = ("client",)
    default_sim_mode: str = "client"
    obs_dim: int = 0
    hidden_dims: tuple[int, ...] = ()
    action_dim: int = 0
    continuous: bool = True
    max_steps: int = 0
    dt: float = 0.02
    eval_seeds: tuple[int, ...] = ()
    heldout_seeds: tuple[int, ...] = tuple(range(100, 110))
    layer_coords: tuple[tuple[float, ...] | None, ...] | None = None
    phase_freq: float = 0.0
    phase_harmonics: int = 0
    obs_labels: tuple[str, ...] = ()
    action_labels: tuple[str, ...] = ()
    mjcf_asset: str | None = None
    primary_metric: str = "meanSteps"
    sort_label: str = "Steps"
    solved_label: str = "Solved"
    telemetry_fields: tuple[dict[str, str], ...] = ()
    nudges: tuple[dict[str, object], ...] = ()

    @property
    def substrate_dims(self) -> tuple[int, ...]:
        if self.kind != "control" or self.obs_dim <= 0 or self.action_dim <= 0:
            return ()
        return (self.obs_dim, *self.hidden_dims, self.action_dim)

    def resolved_layer_coords(self) -> list[list[float]]:
        dims = self.substrate_dims
        if not dims:
            return []
        return [list(c) for c in get_substrate(dims, self.layer_coords).layer_coords]

    def to_nim_dict(self) -> dict[str, object]:
        return {
            "task": self.task_id,
            "isControl": self.kind == "control",
            "defaultPopSize": self.default_pop_size,
            "maxPopSize": self.max_pop_size,
            "defaultMinVotes": self.default_min_votes,
            "defaultEliteFrac": self.default_elite_frac,
        }

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
            "enabled": self.enabled,
            "executionPolicy": self.execution_policy,
            "crowdContribute": self.crowd_contribute,
            "allowServerVideo": self.allow_server_video,
            "simModes": list(self.sim_modes),
            "defaultSimMode": self.default_sim_mode,
            "supportsVideo": self.supports_video,
            "obsDim": self.obs_dim,
            "hiddenDims": list(self.hidden_dims),
            "actionDim": self.action_dim,
            "continuous": self.continuous,
            "maxSteps": self.max_steps,
            "dt": self.dt,
            "evalSeeds": list(self.eval_seeds),
            "heldoutSeeds": list(self.heldout_seeds),
            "layerCoords": self.resolved_layer_coords(),
            "phaseFreq": self.phase_freq,
            "phaseHarmonics": self.phase_harmonics,
            "obsLabels": list(self.obs_labels),
            "actionLabels": list(self.action_labels),
            "mjcfAsset": self.mjcf_asset,
            "primaryMetric": self.primary_metric,
            "sortLabel": self.sort_label,
            "solvedLabel": self.solved_label,
            "telemetryFields": [dict(f) for f in self.telemetry_fields],
            "nudges": [dict(n) for n in self.nudges],
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
        sim_modes=("client",),
        default_sim_mode="client",
        obs_labels=("x", "y", "r", "1"),
        action_labels=("R", "G", "B"),
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
        sim_modes=("client", "server"),
        default_sim_mode="client",
        obs_dim=4,
        hidden_dims=(8,),
        action_dim=1,
        continuous=True,
        max_steps=CARTPOLE_MAX_STEPS,
        dt=CARTPOLE_TAU,
        eval_seeds=(0, 1, 2),
        obs_labels=("x", "vx", "theta", "omega"),
        action_labels=("force",),
        primary_metric="meanSteps",
        sort_label="Steps",
        solved_label="all 10 unseen initial states >= 490/500",
        telemetry_fields=(
            {"key": "step", "label": "step"},
            {"key": "x", "label": "x"},
            {"key": "theta", "label": "\u03b8"},
            {"key": "force", "label": "F"},
            {"key": "status", "label": ""},
        ),
        nudges=(
            {"id": "left", "label": "\u2190 Nudge", "title": "Apply leftward perturbation"},
            {"id": "right", "label": "Nudge \u2192", "title": "Apply rightward perturbation"},
        ),
    ),
    "halfcheetah": TaskSpec(
        task_id="halfcheetah",
        kind="control",
        label="HalfCheetah (MuJoCo Gait)",
        description="6-DOF MuJoCo planar locomotion with anatomical + phase-oscillator HyperNEAT.",
        default_pop_size=16,
        max_pop_size=24,
        default_min_votes=DEFAULT_CONTROL_MIN_VOTES,
        client_sim=True,
        supports_video=True,
        default_elite_frac=0.05,
        sim_modes=("client", "server"),
        default_sim_mode="client",
        obs_dim=21,
        hidden_dims=(12,),
        action_dim=6,
        continuous=True,
        max_steps=HALFCHEETAH_MAX_STEPS,
        dt=HALFCHEETAH_DT,
        eval_seeds=(0, 1),
        layer_coords=(HALFCHEETAH_IN_COORDS, None, HALFCHEETAH_OUT_COORDS),
        phase_freq=HALFCHEETAH_PHASE_FREQ,
        phase_harmonics=HALFCHEETAH_PHASE_HARMONICS,
        obs_labels=HALFCHEETAH_OBS_LABELS,
        action_labels=HALFCHEETAH_ACTION_LABELS,
        mjcf_asset="half_cheetah.xml",
        primary_metric="meanDistance",
        sort_label="Distance",
        solved_label="mean >= 8.0m upright sprint across 10 seeds",
        telemetry_fields=(
            {"key": "step", "label": "step"},
            {"key": "dx", "label": "dx"},
            {"key": "vx", "label": "vx"},
            {"key": "upright", "label": "up"},
            {"key": "status", "label": ""},
        ),
        nudges=(
            {"id": "back", "label": "\u2190 Push", "title": "Apply backward impulse"},
            {"id": "fwd", "label": "Push \u2192", "title": "Apply forward boost"},
            {"id": "hop", "label": "Hop \u2191", "title": "Apply upward perturbation"},
        ),
    ),
    "external": TaskSpec(
        task_id="external",
        kind="control",
        label="Custom Researcher Script (External Worker)",
        description="Connect your own local Python/MuJoCo/Gymnasium simulation script via room worker API.",
        default_pop_size=DEFAULT_CONTROL_POP_SIZE,
        max_pop_size=MAX_POPULATION_SIZE,
        default_min_votes=DEFAULT_CONTROL_MIN_VOTES,
        client_sim=False,
        supports_video=False,
        default_elite_frac=0.10,
        sim_modes=("external",),
        default_sim_mode="external",
        obs_dim=0,
        hidden_dims=(),
        action_dim=0,
        continuous=True,
        max_steps=200,
        dt=0.05,
        eval_seeds=(0,),
        primary_metric="primaryValue",
        sort_label="Score",
        solved_label="Evaluated by researcher script",
        telemetry_fields=(
            {"key": "step", "label": "step"},
            {"key": "score", "label": "score"},
            {"key": "status", "label": ""},
        ),
    ),
}


def normalize_task_id(raw: str | None) -> str:
    cleaned = (raw or "").strip().lower()
    return cleaned if cleaned in TASK_SPECS else "image"


def get_task_spec(task_id: str | None) -> TaskSpec:
    clean = normalize_task_id(task_id)
    base = TASK_SPECS[clean]
    env_cfg = get_config().get_env(clean)
    return replace(
        base,
        enabled=env_cfg.enabled,
        execution_policy=env_cfg.execution_policy,
        sim_modes=env_cfg.sim_modes,
        default_sim_mode=env_cfg.default_sim_mode,
        client_sim=env_cfg.client_sim,
        crowd_contribute=env_cfg.crowd_contribute,
        allow_server_video=env_cfg.allow_server_video,
        default_pop_size=env_cfg.default_pop_size,
        max_pop_size=env_cfg.max_pop_size,
        default_min_votes=env_cfg.default_min_votes,
        default_elite_frac=env_cfg.default_elite_frac,
    )


def get_task_specs_payload() -> dict[str, dict[str, object]]:
    return {k: get_task_spec(k).to_client_dict() for k in TASK_SPECS}


def get_nim_task_configs_payload() -> list[dict[str, object]]:
    return [get_task_spec(k).to_nim_dict() for k in TASK_SPECS]


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


# ---------------------------------------------------------------------------
# CartPole continuous control environment
# ---------------------------------------------------------------------------


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
    nudge: str | None = None,
) -> dict[str, object]:
    """Run one deterministic continuous CartPole episode and return full trace + telemetry."""
    spec = TASK_SPECS["cartpole"]
    substrate = get_substrate(spec.substrate_dims, spec.layer_coords)
    weights, biases = substrate.decode(genome_or_net)

    state = cartpole_reset(seed)
    if nudge in ("left", "right"):
        sign = -1.0 if nudge == "left" else 1.0
        state = (state[0], state[1] + sign * 0.35, state[2], state[3] + sign * 0.55)
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

    for gx in range(margin, width - margin + 1, max(24, usable_w // 8)):
        draw.line([(gx, 38), (gx, track_y + 24)], fill=(26, 26, 32), width=1)
    draw.line([(margin, track_y), (width - margin, track_y)], fill=(82, 82, 91), width=2)
    draw.line([(width * 0.5, track_y - 6), (width * 0.5, track_y + 6)], fill=(113, 113, 122), width=1)
    for bx in (margin, width - margin):
        draw.line([(bx, track_y - 12), (bx, track_y + 12)], fill=(251, 113, 133), width=2)

    cart_w, cart_h = 52, 26
    draw.rounded_rectangle(
        [cx - cart_w * 0.5, track_y - cart_h * 0.5, cx + cart_w * 0.5, track_y + cart_h * 0.5],
        radius=5,
        fill=(228, 228, 231),
        outline=(250, 250, 250),
        width=1,
    )

    pole_px = int(height * 0.36)
    pivot_y = track_y - cart_h * 0.15
    tip_x = cx + math.sin(theta) * pole_px
    tip_y = pivot_y - math.cos(theta) * pole_px
    pole_color = (
        (52, 211, 153)
        if abs(theta) < 0.08
        else ((245, 196, 81) if abs(theta) < 0.16 else (251, 113, 133))
    )
    draw.line([(cx, pivot_y), (tip_x, tip_y)], fill=pole_color, width=7)
    draw.ellipse([cx - 5, pivot_y - 5, cx + 5, pivot_y + 5], fill=(24, 24, 27), outline=(250, 250, 250), width=2)

    if abs(force) > 0.2:
        arrow_len = (force / CARTPOLE_FORCE_MAG) * 44.0
        ay = track_y + cart_h * 0.5 + 14
        ax_end = cx + arrow_len
        draw.line([(cx, ay), (ax_end, ay)], fill=(56, 189, 248), width=3)

    bar_w = int((width - 2 * margin) * min(1.0, step_idx / max(1, max_steps)))
    draw.rectangle([margin, 14, width - margin, 20], fill=(31, 31, 38))
    if bar_w > 0:
        bar_col = (52, 211, 153) if step_idx >= 490 else (245, 196, 81)
        draw.rectangle([margin, 14, margin + bar_w, 20], fill=bar_col)
    draw.text(
        (margin, 26),
        f"step {step_idx}/{max_steps}  x={x:+.2f}  th={math.degrees(theta):+.1f}deg  F={force:+.1f}N",
        fill=(161, 161, 170),
    )
    return np.asarray(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# HalfCheetah (MuJoCo) continuous locomotion environment
# ---------------------------------------------------------------------------

_CHEETAH_MODEL: mujoco.MjModel | None = None
_CHEETAH_MODEL_LOCK = threading.Lock()


def get_halfcheetah_model() -> mujoco.MjModel:
    global _CHEETAH_MODEL
    with _CHEETAH_MODEL_LOCK:
        if _CHEETAH_MODEL is None:
            _CHEETAH_MODEL = mujoco.MjModel.from_xml_path(str(HALFCHEETAH_XML_PATH))
        return _CHEETAH_MODEL


def halfcheetah_reset(model: mujoco.MjModel, data: mujoco.MjData, seed: int) -> None:
    """Reset HalfCheetah with deterministic mulberry32 initial perturbations."""
    mujoco.mj_resetData(model, data)
    rng = mulberry32(seed)
    for i in range(model.nq):
        data.qpos[i] += (rng() * 2.0 - 1.0) * 0.05
    for i in range(model.nv):
        data.qvel[i] += (rng() * 2.0 - 1.0) * 0.05
    mujoco.mj_forward(model, data)


def _cheetah_obs(data: mujoco.MjData, step_idx: int) -> np.ndarray:
    raw = np.empty(21, dtype=np.float64)
    raw[0] = (float(data.qpos[1]) - 0.7) / HALFCHEETAH_OBS_SCALE[0]
    for i in range(1, 8):
        raw[i] = float(data.qpos[i + 1]) / HALFCHEETAH_OBS_SCALE[i]
    for i in range(9):
        raw[8 + i] = float(data.qvel[i]) / HALFCHEETAH_OBS_SCALE[8 + i]
    t = step_idx * HALFCHEETAH_DT
    ang = 2.0 * math.pi * HALFCHEETAH_PHASE_FREQ * t
    raw[17] = math.sin(ang)
    raw[18] = math.cos(ang)
    raw[19] = math.sin(2.0 * ang)
    raw[20] = math.cos(2.0 * ang)
    return raw


def halfcheetah_rollout(
    genome_or_net: dict | CPPN,
    *,
    seed: int = 0,
    max_steps: int = HALFCHEETAH_MAX_STEPS,
    nudge: str | None = None,
) -> dict[str, object]:
    """Run one deterministic HalfCheetah MuJoCo rollout and capture compact 2D pose frames."""
    spec = TASK_SPECS["halfcheetah"]
    substrate = get_substrate(spec.substrate_dims, spec.layer_coords)
    weights, biases = substrate.decode(genome_or_net)

    model = get_halfcheetah_model()
    data = mujoco.MjData(model)
    halfcheetah_reset(model, data, seed)
    if nudge == "back":
        data.qvel[0] -= 2.2
    elif nudge == "fwd":
        data.qvel[0] += 2.2
    elif nudge == "hop":
        data.qvel[1] += 2.6

    start_x = float(data.qpos[0])
    bfoot_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bfoot")
    ffoot_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ffoot")

    def _capture_frame(ctrl_rms: float) -> list[float]:
        q = data.qpos
        g1 = data.contact.geom1
        g2 = data.contact.geom2
        bf_c = 1.0 if ((g1 == bfoot_geom).any() or (g2 == bfoot_geom).any()) else 0.0
        ff_c = 1.0 if ((g1 == ffoot_geom).any() or (g2 == ffoot_geom).any()) else 0.0
        return [
            round(float(q[0]), 4),
            round(float(q[1]), 4),
            round(float(q[2]), 4),
            round(float(q[3]), 4),
            round(float(q[4]), 4),
            round(float(q[5]), 4),
            round(float(q[6]), 4),
            round(float(q[7]), 4),
            round(float(q[8]), 4),
            round(ctrl_rms, 3),
            bf_c,
            ff_c,
        ]

    frames: list[list[float]] = [_capture_frame(0.0)]
    total_return = 0.0
    upright_sum = 0.0
    energy_sum = 0.0
    vx_trace: list[float] = [float(data.qvel[0])]

    for step_idx in range(max_steps):
        obs = _cheetah_obs(data, step_idx)
        act = substrate.forward(weights, biases, obs)
        x_before = float(data.qpos[0])
        data.ctrl[:] = act
        for _ in range(HALFCHEETAH_FRAME_SKIP):
            mujoco.mj_step(model, data)
        x_after = float(data.qpos[0])
        vx = (x_after - x_before) / HALFCHEETAH_DT
        ctrl_sq = float(np.mean(act * act))
        total_return += vx - 0.1 * float(np.sum(act * act))
        upright_sum += math.cos(float(data.qpos[2]))
        energy_sum += ctrl_sq
        vx_trace.append(round(vx, 3))
        frames.append(_capture_frame(math.sqrt(ctrl_sq)))

    steps = len(frames) - 1
    dx = float(data.qpos[0]) - start_x
    mean_vx = dx / max(HALFCHEETAH_DT, steps * HALFCHEETAH_DT)
    mean_upright = upright_sum / max(1, steps)
    return {
        "seed": int(seed),
        "steps": steps,
        "distance": dx,
        "mean_vx": mean_vx,
        "mean_upright": mean_upright,
        "mean_energy": energy_sum / max(1, steps),
        "return": total_return,
        "frames": frames,
        "vx_trace": vx_trace,
    }


def _rot_xz(x: float, z: float, angle: float) -> tuple[float, float]:
    c, s = math.cos(angle), math.sin(angle)
    return x * c + z * s, -x * s + z * c


def _cheetah_fk_segments(frame: Sequence[float]) -> list[tuple[str, tuple[float, float], tuple[float, float]]]:
    """Compute exact world XZ capsule segments matching `half_cheetah.xml`."""
    rx, rz, ry, bth, bsh, bft, fth, fsh, fft = frame[:9]
    rz_w = rz + 0.7
    t1 = _rot_xz(-0.5, 0.0, ry)
    t2 = _rot_xz(0.5, 0.0, ry)
    hc = _rot_xz(0.6, 0.1, ry)
    ha = _rot_xz(math.sin(0.87), math.cos(0.87), ry)
    segs: list[tuple[str, tuple[float, float], tuple[float, float]]] = [
        ("torso", (rx + t1[0], rz_w + t1[1]), (rx + t2[0], rz_w + t2[1])),
        (
            "head",
            (rx + hc[0] - 0.15 * ha[0], rz_w + hc[1] - 0.15 * ha[1]),
            (rx + hc[0] + 0.15 * ha[0], rz_w + hc[1] + 0.15 * ha[1]),
        ),
    ]

    def _append_leg(
        px: float,
        pz: float,
        pa: float,
        specs: list[tuple[str, float, float, float, float, float, float, float]],
    ) -> None:
        x, z, a = px, pz, pa
        for name, ox, oz, dq, gx, gz, ga, half_len in specs:
            dx, dz = _rot_xz(ox, oz, a)
            x += dx
            z += dz
            a += dq
            cx, cz = _rot_xz(gx, gz, a)
            ax, az = _rot_xz(math.sin(ga), math.cos(ga), a)
            segs.append(
                (
                    name,
                    (x + cx - half_len * ax, z + cz - half_len * az),
                    (x + cx + half_len * ax, z + cz + half_len * az),
                )
            )

    _append_leg(
        rx,
        rz_w,
        ry,
        [
            ("bthigh", -0.5, 0.0, bth, 0.1, -0.13, -3.8, 0.145),
            ("bshin", 0.16, -0.25, bsh, -0.14, -0.07, -2.03, 0.15),
            ("bfoot", -0.28, -0.14, bft, 0.03, -0.097, -0.27, 0.094),
        ],
    )
    _append_leg(
        rx,
        rz_w,
        ry,
        [
            ("fthigh", 0.5, 0.0, fth, -0.07, -0.12, 0.52, 0.133),
            ("fshin", -0.14, -0.24, fsh, 0.065, -0.09, -0.6, 0.106),
            ("ffoot", 0.13, -0.18, fft, 0.045, -0.07, -0.6, 0.07),
        ],
    )
    return segs


def _render_cheetah_frame(
    frames: list[list[float]],
    step_idx: int,
    max_steps: int,
    width: int = 480,
    height: int = 320,
) -> np.ndarray:
    """Render one crisp 2D sagittal HalfCheetah frame as an RGB uint8 numpy array."""
    img = Image.new("RGB", (width, height), (9, 9, 11))
    draw = ImageDraw.Draw(img)

    cur = frames[min(step_idx, len(frames) - 1)]
    cam_x = float(cur[0])
    ground_y = int(height * 0.74)
    px_per_m = height * 0.36
    margin = 28

    def _to_px(wx: float, wz: float) -> tuple[float, float]:
        return width * 0.44 + (wx - cam_x) * px_per_m, ground_y - wz * px_per_m

    # Scrolling world distance markers every 1m
    m_start = math.floor(cam_x - 3.0)
    m_end = math.ceil(cam_x + 3.0)
    for meter in range(m_start, m_end + 1):
        gx, _ = _to_px(float(meter), 0.0)
        if 0 <= gx <= width:
            draw.line([(gx, 38), (gx, ground_y)], fill=(24, 24, 30), width=1)
            tick_h = 8 if meter % 2 == 0 else 4
            tick_col = (113, 113, 122) if meter % 2 == 0 else (63, 63, 70)
            draw.line([(gx, ground_y), (gx, ground_y + tick_h)], fill=tick_col, width=1)
            if meter % 2 == 0:
                draw.text((gx + 3, ground_y + 6), f"{meter}m", fill=(113, 113, 122))

    # Ground plane
    draw.line([(0, ground_y), (width, ground_y)], fill=(82, 82, 91), width=2)

    # Recent COM trail
    trail_start = max(0, step_idx - 35)
    trail_pts = [_to_px(frames[i][0], frames[i][1] + 0.7) for i in range(trail_start, step_idx + 1)]
    if len(trail_pts) >= 2:
        draw.line(trail_pts, fill=(56, 189, 248), width=2)

    # Capsule segments (back leg amber, torso zinc, front leg cyan)
    colors = {
        "bthigh": (217, 155, 58),
        "bshin": (245, 196, 81),
        "bfoot": (251, 211, 121),
        "torso": (228, 228, 231),
        "head": (250, 250, 250),
        "fthigh": (56, 189, 248),
        "fshin": (103, 212, 252),
        "ffoot": (165, 243, 252),
    }
    segs = _cheetah_fk_segments(cur)
    seg_w = max(5, int(px_per_m * 0.088))
    for name, p1, p2 in segs:
        x1, y1 = _to_px(p1[0], p1[1])
        x2, y2 = _to_px(p2[0], p2[1])
        col = colors.get(name, (228, 228, 231))
        draw.line([(x1, y1), (x2, y2)], fill=col, width=seg_w)
        r = seg_w * 0.48
        draw.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill=col)
        draw.ellipse([x2 - r, y2 - r, x2 + r, y2 + r], fill=col)

    # Progress bar + telemetry
    dx = float(cur[0]) - float(frames[0][0])
    bar_w = int((width - 2 * margin) * min(1.0, step_idx / max(1, max_steps)))
    draw.rectangle([margin, 14, width - margin, 20], fill=(31, 31, 38))
    if bar_w > 0:
        bar_col = (52, 211, 153) if dx >= 8.0 else ((245, 196, 81) if dx >= 2.5 else (251, 113, 133))
        draw.rectangle([margin, 14, margin + bar_w, 20], fill=bar_col)
    up = math.cos(float(cur[2]))
    vx_est = (float(cur[0]) - float(frames[max(0, step_idx - 1)][0])) / HALFCHEETAH_DT if step_idx > 0 else 0.0
    draw.text(
        (margin, 26),
        f"step {step_idx}/{max_steps}  dx={dx:+.2f}m  vx={vx_est:+.2f}m/s  up={up:+.2f}",
        fill=(161, 161, 170),
    )
    return np.asarray(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Generalized Task Adapter Registry + LRU Rollout Cache
# ---------------------------------------------------------------------------


class ControlTaskAdapter(Protocol):
    task_id: str

    def rollout(
        self,
        genome_or_net: dict | CPPN,
        *,
        seed: int = 0,
        max_steps: int | None = None,
        nudge: str | None = None,
    ) -> dict[str, object]: ...

    def evaluate(
        self,
        genome_or_net: dict | CPPN,
        *,
        seeds: Sequence[int] | None = None,
    ) -> dict[str, object]: ...

    def is_gen0_presolved(self, summary: dict[str, object]) -> bool: ...

    def render_frame(
        self,
        rollout: dict[str, object],
        step_idx: int,
        max_steps: int,
        width: int,
        height: int,
    ) -> np.ndarray: ...


class CartPoleAdapter:
    task_id = "cartpole"

    def rollout(
        self,
        genome_or_net: dict | CPPN,
        *,
        seed: int = 0,
        max_steps: int | None = None,
        nudge: str | None = None,
    ) -> dict[str, object]:
        spec = TASK_SPECS[self.task_id]
        return cartpole_rollout(
            genome_or_net,
            seed=seed,
            max_steps=spec.max_steps if max_steps is None else int(max_steps),
            nudge=nudge,
        )

    def evaluate(
        self,
        genome_or_net: dict | CPPN,
        *,
        seeds: Sequence[int] | None = None,
    ) -> dict[str, object]:
        spec = TASK_SPECS[self.task_id]
        eval_seeds = tuple(spec.eval_seeds if seeds is None else seeds)
        episodes = [self.rollout(genome_or_net, seed=s, max_steps=spec.max_steps) for s in eval_seeds]
        step_list = [int(ep["steps"]) for ep in episodes]
        mean_steps = sum(step_list) / max(1, len(step_list))
        min_steps = min(step_list) if step_list else 0
        mean_abs_x = sum(float(ep["mean_abs_x"]) for ep in episodes) / max(1, len(episodes))
        solved = min_steps >= 490
        mean_rounded = round(mean_steps, 1)
        val_txt = str(int(mean_rounded)) if float(mean_rounded).is_integer() else f"{mean_rounded:.1f}"
        return {
            "task": spec.task_id,
            "seeds": list(eval_seeds),
            "stepsPerSeed": step_list,
            "meanSteps": mean_rounded,
            "minSteps": int(min_steps),
            "meanAbsX": round(mean_abs_x, 3),
            "solved": solved,
            "primaryValue": mean_rounded,
            "primaryDisplay": f"{val_txt}/{spec.max_steps}",
            "badgeText": f"{val_txt}/{spec.max_steps}",
            "tier": "solved" if solved else ("mid" if mean_steps >= 150 else ""),
            "secondaryText": f"min {min_steps}/{spec.max_steps}, |x|={mean_abs_x:.2f}m",
        }

    def is_gen0_presolved(self, summary: dict[str, object]) -> bool:
        return float(summary.get("meanSteps", 0.0)) >= 60.0

    def render_frame(
        self,
        rollout: dict[str, object],
        step_idx: int,
        max_steps: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        states = rollout["states"]
        forces = rollout["forces"]
        state = states[min(step_idx, len(states) - 1)]
        force = forces[min(step_idx, len(forces) - 1)] if forces else 0.0
        return _render_cartpole_frame(
            state,
            float(force),
            min(step_idx, int(rollout["steps"])),
            max_steps,
            width=width,
            height=height,
        )


class HalfCheetahAdapter:
    task_id = "halfcheetah"

    def rollout(
        self,
        genome_or_net: dict | CPPN,
        *,
        seed: int = 0,
        max_steps: int | None = None,
        nudge: str | None = None,
    ) -> dict[str, object]:
        spec = TASK_SPECS[self.task_id]
        return halfcheetah_rollout(
            genome_or_net,
            seed=seed,
            max_steps=spec.max_steps if max_steps is None else int(max_steps),
            nudge=nudge,
        )

    def evaluate(
        self,
        genome_or_net: dict | CPPN,
        *,
        seeds: Sequence[int] | None = None,
    ) -> dict[str, object]:
        spec = TASK_SPECS[self.task_id]
        eval_seeds = tuple(spec.eval_seeds if seeds is None else seeds)
        episodes = [self.rollout(genome_or_net, seed=s, max_steps=spec.max_steps) for s in eval_seeds]
        dists = [round(float(ep["distance"]), 2) for ep in episodes]
        mean_dx = sum(dists) / max(1, len(dists))
        min_dx = min(dists) if dists else 0.0
        mean_vx = sum(float(ep["mean_vx"]) for ep in episodes) / max(1, len(episodes))
        mean_up = sum(float(ep["mean_upright"]) for ep in episodes) / max(1, len(episodes))
        mean_ret = sum(float(ep["return"]) for ep in episodes) / max(1, len(episodes))
        solved = mean_dx >= 8.0 and mean_up >= 0.35
        tier = "solved" if solved else ("mid" if (mean_dx >= 2.5 and mean_up >= 0.2) else "")
        return {
            "task": spec.task_id,
            "seeds": list(eval_seeds),
            "distancesPerSeed": dists,
            "meanDistance": round(mean_dx, 2),
            "minDistance": round(min_dx, 2),
            "meanVx": round(mean_vx, 2),
            "meanUpright": round(mean_up, 2),
            "return": round(mean_ret, 1),
            "solved": solved,
            "primaryValue": round(mean_dx, 2),
            "primaryDisplay": f"{mean_dx:+.1f}m",
            "badgeText": f"{mean_dx:+.1f}m",
            "tier": tier,
            "secondaryText": f"vx={mean_vx:+.2f}m/s, up={mean_up:+.2f}",
        }

    def is_gen0_presolved(self, summary: dict[str, object]) -> bool:
        return float(summary.get("meanDistance", 0.0)) >= 4.5

    def render_frame(
        self,
        rollout: dict[str, object],
        step_idx: int,
        max_steps: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        return _render_cheetah_frame(
            rollout["frames"],
            step_idx,
            max_steps,
            width=width,
            height=height,
        )


TASK_ADAPTERS: dict[str, ControlTaskAdapter] = {
    "cartpole": CartPoleAdapter(),
    "halfcheetah": HalfCheetahAdapter(),
}


def get_task_adapter(task_id: str) -> ControlTaskAdapter:
    spec = get_task_spec(task_id)
    adapter = TASK_ADAPTERS.get(spec.task_id)
    if adapter is None:
        raise ValueError(f"Task '{task_id}' does not support server control simulation.")
    return adapter


_ROLLOUT_CACHE: OrderedDict[tuple[str, str, int, int], dict[str, object]] = OrderedDict()
_ROLLOUT_CACHE_LOCK = threading.Lock()
_MAX_ROLLOUT_CACHE = 512


def _rollout_cached(
    task_id: str,
    genome: dict,
    *,
    seed: int,
    max_steps: int,
    nudge: str | None = None,
) -> dict[str, object]:
    adapter = get_task_adapter(task_id)
    if nudge:
        return adapter.rollout(genome, seed=seed, max_steps=max_steps, nudge=nudge)
    g_hash = genome_content_hash(genome, length=20)
    key = (adapter.task_id, g_hash, int(seed), int(max_steps))
    with _ROLLOUT_CACHE_LOCK:
        hit = _ROLLOUT_CACHE.get(key)
        if hit is not None:
            _ROLLOUT_CACHE.move_to_end(key)
            return hit
    result = adapter.rollout(genome, seed=seed, max_steps=max_steps)
    with _ROLLOUT_CACHE_LOCK:
        _ROLLOUT_CACHE[key] = result
        _ROLLOUT_CACHE.move_to_end(key)
        while len(_ROLLOUT_CACHE) > _MAX_ROLLOUT_CACHE:
            _ROLLOUT_CACHE.popitem(last=False)
    return result


def create_task_population(
    task_id: str | None,
    pop_size: int,
) -> list[dict]:
    """Generate a Gen-0 population for `task_id`, rejecting pre-solved control founders."""
    clean_task = normalize_task_id(task_id)
    spec = get_task_spec(clean_task)
    eff_pop = clamp_pop_for_task(clean_task, pop_size)
    adapter = TASK_ADAPTERS.get(clean_task)
    if spec.kind != "control" or adapter is None:
        return create_random_population(pop_size=eff_pop)

    founders: list[dict] = []
    max_attempts = eff_pop * 12
    attempts = 0
    while len(founders) < eff_pop and attempts < max_attempts:
        attempts += 1
        cand = CPPN.random().to_dict()
        ev = adapter.evaluate(cand, seeds=spec.eval_seeds)
        if not adapter.is_gen0_presolved(ev):
            founders.append(cand)
    while len(founders) < eff_pop:
        founders.append(CPPN.random().to_dict())
    return founders


def is_gen0_founder_presolved(task_id: str, genome: dict) -> bool:
    adapter = TASK_ADAPTERS.get(normalize_task_id(task_id))
    if adapter is None:
        return False
    spec = get_task_spec(task_id)
    return adapter.is_gen0_presolved(adapter.evaluate(genome, seeds=spec.eval_seeds))


def evaluate_control_genome(
    task_id: str,
    genome_or_net: dict | CPPN,
    seeds: Sequence[int] | None = None,
) -> dict[str, object]:
    """Evaluate a CPPN genome across deterministic seeds for a registered control task."""
    adapter = get_task_adapter(task_id)
    return adapter.evaluate(genome_or_net, seeds=seeds)


def rollout_control_episode(
    task_id: str,
    genome: dict,
    *,
    seed: int = 0,
    max_steps: int | None = None,
    nudge: str | None = None,
    include_geoms: bool = False,
) -> dict[str, object]:
    """Run (or retrieve from LRU cache) a single control rollout for browser replay/eval."""
    spec = get_task_spec(task_id)
    eff_steps = spec.max_steps if max_steps is None else max(1, min(spec.max_steps, int(max_steps)))
    ro = _rollout_cached(spec.task_id, genome, seed=int(seed), max_steps=eff_steps, nudge=nudge)
    if include_geoms and spec.task_id == "halfcheetah":
        return attach_halfcheetah_geoms(ro)
    return ro


def attach_halfcheetah_geoms(rollout: dict[str, object]) -> dict[str, object]:
    """Attach exact MuJoCo `geom_xpos` / `geom_xmat` 3D geometry poses from qpos frames on demand."""
    frames = rollout.get("frames") or []
    if not isinstance(frames, list) or not frames:
        return rollout
    model = get_halfcheetah_model()
    data = mujoco.MjData(model)
    geom_specs = []
    non_plane_ids = []
    for gid in range(model.ngeom):
        gtype = int(model.geom_type[gid])
        if gtype == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        non_plane_ids.append(gid)
        geom_specs.append({
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"g{gid}",
            "type": gtype,
            "size": [round(float(x), 4) for x in model.geom_size[gid]],
            "rgba": [round(float(x), 3) for x in model.geom_rgba[gid]],
        })

    geom_poses: list[list[float]] = []
    nq = int(model.nq)
    for fr in frames:
        for i in range(min(nq, len(fr))):
            data.qpos[i] = float(fr[i])
        mujoco.mj_kinematics(model, data)
        row: list[float] = []
        for gid in non_plane_ids:
            pos = data.geom_xpos[gid]
            mat = data.geom_xmat[gid]
            row.extend([round(float(pos[0]), 4), round(float(pos[1]), 4), round(float(pos[2]), 4)])
            row.extend([round(float(mat[k]), 4) for k in range(9)])
        geom_poses.append(row)

    out = dict(rollout)
    out["geomSpecs"] = geom_specs
    out["geomPoses"] = geom_poses
    return out


def evaluate_control_batch(
    task_id: str,
    genomes: Sequence[dict],
    *,
    seeds: Sequence[int] | None = None,
    include_primary_rollout: bool = True,
) -> list[dict[str, object]]:
    """Server-side cached batch evaluation for control grids and server-mode fallback."""
    spec = get_task_spec(task_id)
    adapter = get_task_adapter(spec.task_id)
    eval_seeds = tuple(spec.eval_seeds if seeds is None else seeds)
    results: list[dict[str, object]] = []
    for genome in genomes:
        summary = adapter.evaluate(genome, seeds=eval_seeds)
        entry: dict[str, object] = {
            "hash": genome_content_hash(genome, length=20),
            "summary": summary,
        }
        if include_primary_rollout and eval_seeds:
            entry["primaryRollout"] = _rollout_cached(
                spec.task_id,
                genome,
                seed=int(eval_seeds[0]),
                max_steps=spec.max_steps,
            )
        results.append(entry)
    return results


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


def build_control_runner_script(task_id: str, weights_filename: str = "weights.pt") -> str:
    """Return a standalone uv script that evaluates an exported CPPN policy on a control task."""
    spec = get_task_spec(task_id)
    if spec.task_id not in TASK_ADAPTERS:
        return "\n".join(
            [
                *build_uv_script_header(),
                "import sys",
                "from direct_evolution_together.cppn import CPPN",
                "",
                f'net = CPPN.init_and_load(sys.argv[1] if len(sys.argv) > 1 else "{weights_filename}")',
                'print("Loaded CPPN genome:", net.genome_id, "layers:", [tuple(l.weight.shape) for l in net.layers])',
                "",
            ]
        )

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
            'print(f"Evaluated 10 held-out episodes: primary={summary[\'primaryDisplay\']}, solved={summary[\'solved\']} ({summary[\'secondaryText\']})")',
            f'media = render_task_media("{spec.task_id}", net.to_dict(), seed=0, media_root=Path("."))',
            'Path("out.mp4").write_bytes((Path(".") / media["mp4_rel"]).read_bytes())',
            'print("Voila! Saved out.mp4")',
            "",
        ]
    )


def build_worker_script(
    room_slug: str,
    server_url: str,
    worker_token: str = "<YOUR_WORKER_TOKEN>",
) -> str:
    """Return a standalone PEP 723 uv worker script so researchers can plug in any custom simulator."""
    clean_base = server_url.rstrip("/")
    return "\n".join(
        [
            "# /// script",
            '# dependencies = ["numpy", "torch", "pillow", "direct-evolution-together"]',
            "# ///",
            '"""External simulation worker for direct-evolution-together.',
            "",
            "Run your own MuJoCo / Gymnasium / Microduck / custom simulator locally and stream",
            "per-genome telemetry, 2D/3D pose frames, or rendered PNG/MP4 previews to the room.",
            '"""',
            "import base64",
            "import io",
            "import json",
            "import time",
            "import urllib.request",
            "import numpy as np",
            "from PIL import Image, ImageDraw",
            "from direct_evolution_together.environments import HyperNEATSubstrate",
            "",
            f'BASE_URL = "{clean_base}"',
            f'ROOM_SLUG = "{room_slug}"',
            f'WORKER_TOKEN = "{worker_token}"',
            "SUBSTRATE = HyperNEATSubstrate((8, 12, 4))  # Customize obs_dim -> hidden -> act_dim",
            "",
            "def _req(method, path, payload=None):",
            "    data = json.dumps(payload).encode('utf-8') if payload is not None else None",
            "    headers = {'X-Worker-Token': WORKER_TOKEN}",
            "    if data is not None:",
            "        headers['Content-Type'] = 'application/json'",
            "    req = urllib.request.Request(f'{BASE_URL}{path}', data=data, headers=headers, method=method)",
            "    with urllib.request.urlopen(req, timeout=15) as resp:",
            "        return json.loads(resp.read().decode('utf-8'))",
            "",
            "def evaluate_genome(genome: dict) -> dict:",
            "    # Replace this body with your own simulator (e.g. MicroduckGaitEnv, Gymnasium, etc.)",
            "    weights, biases = SUBSTRATE.decode(genome)",
            "    obs = np.zeros(SUBSTRATE.dims[0], dtype=np.float64)",
            "    act = SUBSTRATE.forward(weights, biases, obs)",
            "    score = float(np.sum(np.abs(act)))",
            "    img = Image.new('RGB', (160, 160), (9, 9, 11))",
            "    draw = ImageDraw.Draw(img)",
            "    draw.text((14, 70), f'score {score:.2f}', fill=(52, 211, 153))",
            "    buf = io.BytesIO()",
            "    img.save(buf, format='PNG')",
            "    return {",
            "        'summary': {",
            "            'primaryValue': round(score, 2),",
            "            'badgeText': f'{score:.2f}',",
            "            'tier': 'solved' if score > 2.5 else ('mid' if score > 1.2 else ''),",
            "            'secondaryText': 'evaluated by local researcher script',",
            "        },",
            "        'posterPngB64': base64.b64encode(buf.getvalue()).decode('ascii'),",
            "    }",
            "",
            "def main():",
            "    print(f'Worker connected to {BASE_URL}/p/{ROOM_SLUG}/')",
            "    while True:",
            "        job = _req('GET', f'/api/rooms/{ROOM_SLUG}/worker/jobs/')",
            "        pending = job.get('pending', [])",
            "        if pending:",
            "            results = []",
            "            for item in pending:",
            "                res = evaluate_genome(item['genome'])",
            "                results.append({'hash': item['hash'], **res})",
            "            _req('POST', f'/api/rooms/{ROOM_SLUG}/worker/results/', {",
            "                'goal': job['goal'],",
            "                'generation': job['generation'],",
            "                'results': results,",
            "            })",
            "            print(f\"Submitted {len(results)} evals for {job['goal']} gen {job['generation']}\")",
            "        time.sleep(1.0)",
            "",
            "if __name__ == '__main__':",
            "    main()",
            "",
        ]
    )
