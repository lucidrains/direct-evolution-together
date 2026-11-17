"""Static task registry: mechanics + labels live here, policy defaults live in `config`."""

from __future__ import annotations

from direct_evolution_together.config import DEFAULT_ENV_POLICIES

from .cartpole import (
    CARTPOLE_ACTION_DIM,
    CARTPOLE_EVAL_SEEDS,
    CARTPOLE_HIDDEN_DIMS,
    CARTPOLE_MAX_STEPS,
    CARTPOLE_OBS_DIM,
    CARTPOLE_TAU,
)
from .halfcheetah import (
    HALFCHEETAH_ACTION_DIM,
    HALFCHEETAH_ACTION_LABELS,
    HALFCHEETAH_DT,
    HALFCHEETAH_EVAL_SEEDS,
    HALFCHEETAH_HIDDEN_DIMS,
    HALFCHEETAH_LAYER_COORDS,
    HALFCHEETAH_MAX_STEPS,
    HALFCHEETAH_OBS_DIM,
    HALFCHEETAH_OBS_LABELS,
    HALFCHEETAH_PHASE_FREQ,
    HALFCHEETAH_PHASE_HARMONICS,
)
from .types import TaskSpec


def _task_spec(task_id: str, **mechanics: object) -> TaskSpec:
    """Compose a task descriptor from its policy defaults and static mechanics."""
    policy = DEFAULT_ENV_POLICIES[task_id]
    return TaskSpec(
        task_id=task_id,
        enabled=policy.enabled,
        execution_policy=policy.execution_policy,
        sim_modes=policy.sim_modes,
        default_sim_mode=policy.default_sim_mode,
        client_sim=policy.client_sim,
        crowd_contribute=policy.crowd_contribute,
        allow_server_video=policy.allow_server_video,
        default_pop_size=policy.default_pop_size,
        max_pop_size=policy.max_pop_size,
        default_min_votes=policy.default_min_votes,
        default_elite_frac=policy.default_elite_frac,
        **mechanics,
    )


TASK_SPECS: dict[str, TaskSpec] = {
    "image": _task_spec(
        "image",
        kind="image",
        label="Image (Picbreeder)",
        description="2D CPPN pattern synthesis evaluated over (x, y, r, 1) coordinates.",
        supports_video=False,
        obs_labels=("x", "y", "r", "1"),
        action_labels=("R", "G", "B"),
    ),
    "cartpole": _task_spec(
        "cartpole",
        kind="control",
        label="CartPole (Continuous Control)",
        description="Balance a pole via continuous horizontal force using HyperNEAT on the CPPN.",
        supports_video=True,
        obs_dim=CARTPOLE_OBS_DIM,
        hidden_dims=CARTPOLE_HIDDEN_DIMS,
        action_dim=CARTPOLE_ACTION_DIM,
        continuous=True,
        max_steps=CARTPOLE_MAX_STEPS,
        dt=CARTPOLE_TAU,
        eval_seeds=CARTPOLE_EVAL_SEEDS,
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
    "halfcheetah": _task_spec(
        "halfcheetah",
        kind="control",
        label="HalfCheetah (MuJoCo Gait)",
        description="6-DOF MuJoCo planar locomotion with anatomical + phase-oscillator HyperNEAT.",
        supports_video=True,
        obs_dim=HALFCHEETAH_OBS_DIM,
        hidden_dims=HALFCHEETAH_HIDDEN_DIMS,
        action_dim=HALFCHEETAH_ACTION_DIM,
        continuous=True,
        max_steps=HALFCHEETAH_MAX_STEPS,
        dt=HALFCHEETAH_DT,
        eval_seeds=HALFCHEETAH_EVAL_SEEDS,
        layer_coords=HALFCHEETAH_LAYER_COORDS,
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
    "external": _task_spec(
        "external",
        kind="control",
        label="Custom Researcher Script (External Worker)",
        description="Connect your own local Python/MuJoCo/Gymnasium simulation script via room worker API.",
        supports_video=False,
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
