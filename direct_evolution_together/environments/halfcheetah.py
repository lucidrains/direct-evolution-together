"""HalfCheetah MuJoCo locomotion: rollout, scoring, kinematics, and 2D rendering."""

from __future__ import annotations

import math
from pathlib import Path
import threading
from typing import Sequence

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from direct_evolution_together.cppn import CPPN

from .rng import mulberry32
from .substrate import get_substrate

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

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

HALFCHEETAH_OBS_DIM = 21
HALFCHEETAH_HIDDEN_DIMS = (12,)
HALFCHEETAH_ACTION_DIM = 6
HALFCHEETAH_SUBSTRATE_DIMS = (
    HALFCHEETAH_OBS_DIM,
    *HALFCHEETAH_HIDDEN_DIMS,
    HALFCHEETAH_ACTION_DIM,
)
HALFCHEETAH_LAYER_COORDS = (HALFCHEETAH_IN_COORDS, None, HALFCHEETAH_OUT_COORDS)
HALFCHEETAH_EVAL_SEEDS = (0, 1)

HALFCHEETAH_REWARD_CTRL_COST = 0.1
HALFCHEETAH_PRESOLVED_DISTANCE = 4.5
HALFCHEETAH_SOLVED_DISTANCE = 8.0
HALFCHEETAH_SOLVED_UPRIGHT = 0.35
HALFCHEETAH_TIER_DISTANCE = 2.5
HALFCHEETAH_TIER_UPRIGHT = 0.2

_cheetah_model: mujoco.MjModel | None = None
_cheetah_bfoot_geom: int = -1
_cheetah_ffoot_geom: int = -1
_cheetah_model_LOCK = threading.Lock()
_CHEETAH_TLS = threading.local()

_INV_OBS_SCALE = np.asarray([1.0 / s for s in HALFCHEETAH_OBS_SCALE], dtype=np.float64)
_INV_OBS_SCALE_0 = float(_INV_OBS_SCALE[0])
_INV_OBS_SCALE_POS = _INV_OBS_SCALE[1:8]
_INV_OBS_SCALE_VEL = _INV_OBS_SCALE[8:17]
_PHASE_ANGLES = 2.0 * math.pi * HALFCHEETAH_PHASE_FREQ * (np.arange(HALFCHEETAH_MAX_STEPS, dtype=np.float64) * HALFCHEETAH_DT)
_PHASE_TABLE = np.ascontiguousarray(
    np.column_stack(
        [
            np.sin(_PHASE_ANGLES),
            np.cos(_PHASE_ANGLES),
            np.sin(2.0 * _PHASE_ANGLES),
            np.cos(2.0 * _PHASE_ANGLES),
        ]
    ),
    dtype=np.float64,
)


def get_halfcheetah_model() -> mujoco.MjModel:
    global _cheetah_model, _cheetah_bfoot_geom, _cheetah_ffoot_geom
    model = _cheetah_model
    if model is not None:
        return model
    with _cheetah_model_LOCK:
        if _cheetah_model is None:
            m = mujoco.MjModel.from_xml_path(str(HALFCHEETAH_XML_PATH))
            _cheetah_bfoot_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "bfoot")
            _cheetah_ffoot_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ffoot")
            _cheetah_model = m
        return _cheetah_model


def get_thread_halfcheetah_data(model: mujoco.MjModel | None = None) -> mujoco.MjData:
    """Return a reusable per-thread `MjData` to avoid 6.3 MB arena malloc/free per rollout."""
    eff_model = get_halfcheetah_model() if model is None else model
    cached_model = getattr(_CHEETAH_TLS, "model", None)
    cached_data = getattr(_CHEETAH_TLS, "data", None)
    if cached_data is None or cached_model is not eff_model:
        cached_data = mujoco.MjData(eff_model)
        _CHEETAH_TLS.model = eff_model
        _CHEETAH_TLS.data = cached_data
    return cached_data


def halfcheetah_reset(model: mujoco.MjModel, data: mujoco.MjData, seed: int) -> None:
    """Reset HalfCheetah with deterministic mulberry32 initial perturbations."""
    mujoco.mj_resetData(model, data)
    rng = mulberry32(seed)
    qpos = data.qpos
    qvel = data.qvel
    for i in range(model.nq):
        qpos[i] += (rng() * 2.0 - 1.0) * 0.05
    for i in range(model.nv):
        qvel[i] += (rng() * 2.0 - 1.0) * 0.05
    mujoco.mj_forward(model, data)


def _fill_cheetah_obs(obs: np.ndarray, qpos: np.ndarray, qvel: np.ndarray, step_idx: int) -> None:
    obs[0] = (float(qpos[1]) - 0.7) * _INV_OBS_SCALE_0
    obs[1:8] = qpos[2:9] * _INV_OBS_SCALE_POS
    obs[8:17] = qvel[:9] * _INV_OBS_SCALE_VEL
    if step_idx < len(_PHASE_TABLE):
        obs[17:21] = _PHASE_TABLE[step_idx]
    else:
        ang = 2.0 * math.pi * HALFCHEETAH_PHASE_FREQ * (step_idx * HALFCHEETAH_DT)
        obs[17] = math.sin(ang)
        obs[18] = math.cos(ang)
        obs[19] = math.sin(2.0 * ang)
        obs[20] = math.cos(2.0 * ang)


def _cheetah_obs(data: mujoco.MjData, step_idx: int) -> np.ndarray:
    raw = np.empty(HALFCHEETAH_OBS_DIM, dtype=np.float64)
    _fill_cheetah_obs(raw, data.qpos, data.qvel, step_idx)
    return raw


def halfcheetah_rollout(
    genome_or_net: dict | CPPN,
    *,
    seed: int = 0,
    max_steps: int = HALFCHEETAH_MAX_STEPS,
    nudge: str | None = None,
    capture_frames: bool = True,
) -> dict[str, object]:
    """Run one deterministic HalfCheetah MuJoCo rollout and capture compact 2D pose frames."""
    substrate = get_substrate(HALFCHEETAH_SUBSTRATE_DIMS, HALFCHEETAH_LAYER_COORDS)
    weights, biases = substrate.decode(genome_or_net)
    w0, w1 = weights[0], weights[1]
    b0, b1 = biases[0], biases[1]

    model = get_halfcheetah_model()
    data = get_thread_halfcheetah_data(model)
    halfcheetah_reset(model, data, seed)
    qpos = data.qpos
    qvel = data.qvel
    ctrl = data.ctrl
    if nudge == "back":
        qvel[0] -= 2.2
    elif nudge == "fwd":
        qvel[0] += 2.2
    elif nudge == "hop":
        qvel[1] += 2.6

    start_x = float(qpos[0])
    bfoot_geom = _cheetah_bfoot_geom
    ffoot_geom = _cheetah_ffoot_geom
    contact = data.contact

    def _capture_frame(ctrl_rms: float) -> list[float]:
        ncon = int(data.ncon)
        if ncon > 0:
            g1 = contact.geom1[:ncon]
            g2 = contact.geom2[:ncon]
            bf_c = 1.0 if ((g1 == bfoot_geom).any() or (g2 == bfoot_geom).any()) else 0.0
            ff_c = 1.0 if ((g1 == ffoot_geom).any() or (g2 == ffoot_geom).any()) else 0.0
        else:
            bf_c = 0.0
            ff_c = 0.0
        return [
            round(float(qpos[0]), 4),
            round(float(qpos[1]), 4),
            round(float(qpos[2]), 4),
            round(float(qpos[3]), 4),
            round(float(qpos[4]), 4),
            round(float(qpos[5]), 4),
            round(float(qpos[6]), 4),
            round(float(qpos[7]), 4),
            round(float(qpos[8]), 4),
            round(ctrl_rms, 3),
            bf_c,
            ff_c,
        ]

    frames: list[list[float]] = [_capture_frame(0.0)] if capture_frames else []
    vx_trace: list[float] = [float(qvel[0])] if capture_frames else []
    obs = np.empty(HALFCHEETAH_OBS_DIM, dtype=np.float64)
    total_return = 0.0
    upright_sum = 0.0
    energy_sum = 0.0
    inv_dt = 1.0 / HALFCHEETAH_DT
    inv_act_dim = 1.0 / HALFCHEETAH_ACTION_DIM

    for step_idx in range(max_steps):
        _fill_cheetah_obs(obs, qpos, qvel, step_idx)
        act = np.tanh(np.tanh(obs @ w0 + b0) @ w1 + b1)
        x_before = float(qpos[0])
        ctrl[:] = act
        mujoco.mj_step(model, data, nstep=HALFCHEETAH_FRAME_SKIP)
        x_after = float(qpos[0])
        vx = (x_after - x_before) * inv_dt
        act_sq_sum = float(act @ act)
        ctrl_sq = act_sq_sum * inv_act_dim
        total_return += vx - HALFCHEETAH_REWARD_CTRL_COST * act_sq_sum
        upright_sum += math.cos(float(qpos[2]))
        energy_sum += ctrl_sq
        if capture_frames:
            vx_trace.append(round(vx, 3))
            frames.append(_capture_frame(math.sqrt(ctrl_sq)))

    steps = (len(frames) - 1) if capture_frames else max(0, int(max_steps))
    dx = float(qpos[0]) - start_x
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
        bar_col = (
            (52, 211, 153)
            if dx >= HALFCHEETAH_SOLVED_DISTANCE
            else ((245, 196, 81) if dx >= HALFCHEETAH_TIER_DISTANCE else (251, 113, 133))
        )
        draw.rectangle([margin, 14, margin + bar_w, 20], fill=bar_col)
    up = math.cos(float(cur[2]))
    vx_est = (float(cur[0]) - float(frames[max(0, step_idx - 1)][0])) / HALFCHEETAH_DT if step_idx > 0 else 0.0
    draw.text(
        (margin, 26),
        f"step {step_idx}/{max_steps}  dx={dx:+.2f}m  vx={vx_est:+.2f}m/s  up={up:+.2f}",
        fill=(161, 161, 170),
    )
    return np.asarray(img, dtype=np.uint8)


class HalfCheetahAdapter:
    task_id = "halfcheetah"

    def rollout(
        self,
        genome_or_net: dict | CPPN,
        *,
        seed: int = 0,
        max_steps: int | None = None,
        nudge: str | None = None,
        capture_frames: bool = True,
    ) -> dict[str, object]:
        return halfcheetah_rollout(
            genome_or_net,
            seed=seed,
            max_steps=HALFCHEETAH_MAX_STEPS if max_steps is None else int(max_steps),
            nudge=nudge,
            capture_frames=capture_frames,
        )

    def evaluate(
        self,
        genome_or_net: dict | CPPN,
        *,
        seeds: Sequence[int] | None = None,
        capture_frames: bool = True,
        episodes: Sequence[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        eval_seeds = tuple(HALFCHEETAH_EVAL_SEEDS if seeds is None else seeds)
        if episodes is None:
            episodes = [
                self.rollout(
                    genome_or_net,
                    seed=s,
                    max_steps=HALFCHEETAH_MAX_STEPS,
                    capture_frames=capture_frames,
                )
                for s in eval_seeds
            ]
        dists = [round(float(ep["distance"]), 2) for ep in episodes]
        mean_dx = sum(dists) / max(1, len(dists))
        min_dx = min(dists) if dists else 0.0
        mean_vx = sum(float(ep["mean_vx"]) for ep in episodes) / max(1, len(episodes))
        mean_up = sum(float(ep["mean_upright"]) for ep in episodes) / max(1, len(episodes))
        mean_ret = sum(float(ep["return"]) for ep in episodes) / max(1, len(episodes))
        solved = mean_dx >= HALFCHEETAH_SOLVED_DISTANCE and mean_up >= HALFCHEETAH_SOLVED_UPRIGHT
        tier = (
            "solved"
            if solved
            else (
                "mid"
                if (mean_dx >= HALFCHEETAH_TIER_DISTANCE and mean_up >= HALFCHEETAH_TIER_UPRIGHT)
                else ""
            )
        )
        return {
            "task": self.task_id,
            "seeds": list(eval_seeds),
            "distances_per_seed": dists,
            "mean_distance": round(mean_dx, 2),
            "min_distance": round(min_dx, 2),
            "mean_vx": round(mean_vx, 2),
            "mean_upright": round(mean_up, 2),
            "return": round(mean_ret, 1),
            "solved": solved,
            "primary_value": round(mean_dx, 2),
            "primary_display": f"{mean_dx:+.1f}m",
            "badge_text": f"{mean_dx:+.1f}m",
            "tier": tier,
            "secondary_text": f"vx={mean_vx:+.2f}m/s, up={mean_up:+.2f}",
        }

    def is_gen0_presolved(self, summary: dict[str, object]) -> bool:
        return float(summary.get("mean_distance", 0.0)) >= HALFCHEETAH_PRESOLVED_DISTANCE

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
