"""Continuous-force CartPole simulation, scoring, and crisp 2D rendering."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from direct_evolution_together.cppn import CPPN

from .rng import mulberry32
from .substrate import get_substrate

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

CARTPOLE_OBS_DIM = 4
CARTPOLE_HIDDEN_DIMS = (8,)
CARTPOLE_ACTION_DIM = 1
CARTPOLE_SUBSTRATE_DIMS = (CARTPOLE_OBS_DIM, *CARTPOLE_HIDDEN_DIMS, CARTPOLE_ACTION_DIM)
CARTPOLE_EVAL_SEEDS = (0, 1, 2)
CARTPOLE_PRESOLVED_STEPS = 60.0


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
    substrate = get_substrate(CARTPOLE_SUBSTRATE_DIMS)
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
        return cartpole_rollout(
            genome_or_net,
            seed=seed,
            max_steps=CARTPOLE_MAX_STEPS if max_steps is None else int(max_steps),
            nudge=nudge,
        )

    def evaluate(
        self,
        genome_or_net: dict | CPPN,
        *,
        seeds: Sequence[int] | None = None,
    ) -> dict[str, object]:
        eval_seeds = tuple(CARTPOLE_EVAL_SEEDS if seeds is None else seeds)
        episodes = [self.rollout(genome_or_net, seed=s, max_steps=CARTPOLE_MAX_STEPS) for s in eval_seeds]
        step_list = [int(ep["steps"]) for ep in episodes]
        mean_steps = sum(step_list) / max(1, len(step_list))
        min_steps = min(step_list) if step_list else 0
        mean_abs_x = sum(float(ep["mean_abs_x"]) for ep in episodes) / max(1, len(episodes))
        solved = min_steps >= 490
        mean_rounded = round(mean_steps, 1)
        val_txt = str(int(mean_rounded)) if float(mean_rounded).is_integer() else f"{mean_rounded:.1f}"
        return {
            "task": self.task_id,
            "seeds": list(eval_seeds),
            "steps_per_seed": step_list,
            "mean_steps": mean_rounded,
            "min_steps": int(min_steps),
            "mean_abs_x": round(mean_abs_x, 3),
            "solved": solved,
            "primary_value": mean_rounded,
            "primary_display": f"{val_txt}/{CARTPOLE_MAX_STEPS}",
            "badge_text": f"{val_txt}/{CARTPOLE_MAX_STEPS}",
            "tier": "solved" if solved else ("mid" if mean_steps >= 150 else ""),
            "secondary_text": f"min {min_steps}/{CARTPOLE_MAX_STEPS}, |x|={mean_abs_x:.2f}m",
        }

    def is_gen0_presolved(self, summary: dict[str, object]) -> bool:
        return float(summary.get("mean_steps", 0.0)) >= CARTPOLE_PRESOLVED_STEPS

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
