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
    "obsDim": SO101_OBS_DIM,
    "actionDim": SO101_ACTION_DIM,
    "hiddenDim": SO101_HIDDEN_DIM,
    "headDim": SO101_HEAD_DIM,
    "lowRank": SO101_LOW_RANK,
    "loraTargets": list(SO101_LORA_TARGETS),
    "actionDist": "squashed_gaussian",
    "actionBounds": [-1.0, 1.0],
    "oscillatoryInit": False,
    "initStdUp": 0.35,
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
    "sceneId": "so101",
    "shortLabel": "SO-101",
    "gripperKind": "angular",
    "urdfAsset": None,
    "stlAssets": [],
    "physicsDt": SO101_PHYSICS_DT,
    "nSubsteps": SO101_N_SUBSTEPS,
    "ctrlDt": SO101_DT,
    "actionScales": [0.65, 0.65, 0.65, 0.65, 0.55, 0.95],
    "goalPos": [0.23, 0.012, 0.14],
    "cubeRestZ": 0.016,
    "reachWeight": 12.0,
    "reachProgressWeight": 45.0,
    "preGraspAlignWeight": 5.0,
    "singleTouchWeight": 2.5,
    "pinchGraspWeight": 16.0,
    "liftWeight": 220.0,
    "liftHighWaterWeight": 650.0,
    "goalHoldWeight": 24.0,
    "knockPenaltyWeight": 15.0,
    "floorScrapeWeight": 8.0,
    "actionRateWeight": 0.04,
}

SO102_ENV_PARAMS: dict[str, object] = {
    **SO101_ANGULAR_ENV_PARAMS,
    "arm_model": "so102",
    "sceneId": "so102",
    "shortLabel": "SO-ARM102",
    "gripperKind": "parallel",
    "urdfAsset": SO102_URDF_ASSET,
    "stlAssets": list(SO102_STL_ASSETS),
    "actionScales": [0.65, 0.65, 0.65, 0.65, 0.55, 0.028],
    "goalPos": [0.238, 0.0, 0.14],
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
        "aliases": ["armModel", "sceneId"],
        "label": "Arm Hardware Model",
        "type": "select",
        "default": "so102",
        "description": "Defaults to SO-ARM102 (parallel 2-jaw prismatic gripper backed by so_arm_102.urdf + STL meshes), with SO-101 angular jaw available as an option.",
        "options": [
            {
                "value": "so102",
                "aliases": list(SO102_ALIASES),
                "label": "SO-ARM102 (Parallel Gripper · URDF/STL)",
                "sceneId": "so102",
                "urdfAsset": SO102_URDF_ASSET,
            },
            {
                "value": "so101",
                "aliases": list(SO101_ALIASES),
                "label": "SO-101 (Angular Jaw)",
                "sceneId": "so101",
                "urdfAsset": None,
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
    arm_model = normalize_so_arm_model(ov.get("arm_model") or ov.get("armModel") or ov.get("sceneId"))
    resolved = dict(SO_ARM_VARIANTS[arm_model])
    return {
        **resolved,
        "actionScales": list(resolved["actionScales"]),
        "goalPos": list(resolved["goalPos"]),
        "stlAssets": list(resolved["stlAssets"]),
    }
