"""SO-101 / SO-ARM102 6-DOF robotic arm tabletop cube pick-and-lift constants & PopuLoRA policy spec."""

from __future__ import annotations

from typing import Any, Mapping

SO101_CONTROL_HZ = 50
SO101_N_SUBSTEPS = 5
SO101_PHYSICS_DT = 1.0 / (SO101_CONTROL_HZ * SO101_N_SUBSTEPS)  # 0.004s
SO101_DT = SO101_PHYSICS_DT * SO101_N_SUBSTEPS  # 0.02s (50 Hz)
SO101_MAX_STEPS = 250  # 5.0s manipulation episode

SO101_OBS_DIM = 33
SO101_ACTION_DIM = 6
SO101_HIDDEN_DIM = 96
SO101_HEAD_DIM = 96
SO101_LOW_RANK = 6
SO101_EVAL_SEEDS = (0, 1)

SO101_LORA_TARGETS = (
    "net.encoder.0",
    "net.encoder.2",
    "net.cell.linear_ih",
    "net.cell.linear_hh",
    "net.decoder.0",
    "net.decoder.2",
)

SO101_ACTION_LABELS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

SO101_OBS_LABELS = (
    *(f"q_{j}" for j in SO101_ACTION_LABELS),
    *(f"dq_{j}" for j in SO101_ACTION_LABELS),
    "ee_x", "ee_y", "ee_z",
    "cube_x", "cube_y", "cube_z",
    "rel_x", "rel_y", "rel_z",
    "goal_dx", "goal_dy", "goal_dz",
    "cube_vx", "cube_vy", "cube_vz",
    *(f"act_{j}" for j in SO101_ACTION_LABELS),
)

SO101_POLICY_SPEC: dict[str, object] = {
    "arch": "gru_mlp",
    "obs_dim": SO101_OBS_DIM,
    "action_dim": SO101_ACTION_DIM,
    "hidden_dim": SO101_HIDDEN_DIM,
    "head_dim": SO101_HEAD_DIM,
    "low_rank": SO101_LOW_RANK,
    "lora_targets": list(SO101_LORA_TARGETS),
    "action_dist": "squashed_gaussian",
    "action_bounds": [-1.0, 1.0],
    "oscillatory_init": False,
    "init_std_up": 0.35,
}

SO102_STL_ASSETS: tuple[str, ...] = (
    "scenes/so102/meshes/base_link.stl",
    "scenes/so102/meshes/link1_1.stl",
    "scenes/so102/meshes/link2_1.stl",
    "scenes/so102/meshes/link3_1.stl",
    "scenes/so102/meshes/link4_1.stl",
    "scenes/so102/meshes/link5_1.stl",
    "scenes/so102/meshes/clamp1_1.stl",
    "scenes/so102/meshes/clamp2_1.stl",
)
SO102_URDF_ASSET = "scenes/so102/so_arm_102.urdf"

SO101_ANGULAR_ENV_PARAMS: dict[str, object] = {
    "arm_model": "so101",
    "scene_id": "so101",
    "short_label": "SO-101",
    "gripper_kind": "angular",
    "urdf_asset": None,
    "stl_assets": [],
    "physics_dt": SO101_PHYSICS_DT,
    "n_substeps": SO101_N_SUBSTEPS,
    "ctrl_dt": SO101_DT,
    "action_scales": [0.65, 0.65, 0.65, 0.65, 0.55, 0.95],
    "goal_pos": [0.23, 0.012, 0.14],
    "cube_rest_z": 0.016,
    "reach_weight": 12.0,
    "reach_progress_weight": 45.0,
    "pre_grasp_align_weight": 5.0,
    "single_touch_weight": 2.5,
    "pinch_grasp_weight": 16.0,
    "lift_weight": 220.0,
    "lift_high_water_weight": 650.0,
    "goal_hold_weight": 24.0,
    "knock_penalty_weight": 15.0,
    "floor_scrape_weight": 8.0,
    "action_rate_weight": 0.04,
}

SO102_ENV_PARAMS: dict[str, object] = {
    **SO101_ANGULAR_ENV_PARAMS,
    "arm_model": "so102",
    "scene_id": "so102",
    "short_label": "SO-ARM102",
    "gripper_kind": "parallel",
    "urdf_asset": SO102_URDF_ASSET,
    "stl_assets": list(SO102_STL_ASSETS),
    "action_scales": [0.65, 0.65, 0.65, 0.65, 0.55, 0.028],
    "goal_pos": [0.238, 0.0, 0.14],
}

# Default arm environment hyperparameters point at SO-ARM102 (parallel 2-jaw gripper)
SO101_ENV_PARAMS: dict[str, object] = dict(SO102_ENV_PARAMS)

SO_ARM_VARIANTS: dict[str, dict[str, Any]] = {
    "so102": dict(SO102_ENV_PARAMS),
    "so101": dict(SO101_ANGULAR_ENV_PARAMS),
}

SO101_ALIASES = ("so101", "so_101", "so_arm_101", "soarm101")
SO102_ALIASES = ("so102", "so_102", "so_arm_102", "soarm102")

SO101_HYPERPARAM_SPECS: tuple[dict[str, Any], ...] = (
    {
        "key": "arm_model",
        "aliases": ["arm_model", "scene_id"],
        "label": "Arm Hardware Model",
        "type": "select",
        "default": "so102",
        "description": "Defaults to SO-ARM102 (parallel 2-jaw prismatic gripper backed by so_arm_102.urdf + STL meshes), with SO-101 angular jaw available as an option.",
        "options": [
            {
                "value": "so102",
                "aliases": list(SO102_ALIASES),
                "label": "SO-ARM102 (Parallel Gripper · URDF/STL)",
                "scene_id": "so102",
                "urdf_asset": SO102_URDF_ASSET,
            },
            {
                "value": "so101",
                "aliases": list(SO101_ALIASES),
                "label": "SO-101 (Angular Jaw)",
                "scene_id": "so101",
                "urdf_asset": None,
            },
        ],
    },
)


def normalize_so_arm_model(raw: object) -> str:
    """Normalize an arm model identifier to ``'so102'`` (default) or ``'so101'``."""
    text = str(raw or "").strip().lower().replace("-", "_")
    return "so101" if text in SO101_ALIASES else "so102"


def resolve_so_arm_env_params(overrides: Mapping[str, object] | None = None) -> dict[str, Any]:
    """Return canonical arm environment hyperparameters pointing at SO-101 or SO-102 scene/URDF/STL files."""
    ov = overrides or {}
    arm_model = normalize_so_arm_model(ov.get("arm_model") or ov.get("arm_model") or ov.get("scene_id"))
    resolved = dict(SO_ARM_VARIANTS[arm_model])
    return {
        **resolved,
        "action_scales": list(resolved["action_scales"]),
        "goal_pos": list(resolved["goal_pos"]),
        "stl_assets": list(resolved["stl_assets"]),
    }
