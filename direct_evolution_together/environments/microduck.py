"""Microduck bipedal walking gait task constants and PopuLoRA policy specification."""

from __future__ import annotations

MICRODUCK_CONTROL_HZ = 50
MICRODUCK_N_SUBSTEPS = 5
MICRODUCK_PHYSICS_DT = 1.0 / (MICRODUCK_CONTROL_HZ * MICRODUCK_N_SUBSTEPS)  # 0.004s
MICRODUCK_DT = MICRODUCK_PHYSICS_DT * MICRODUCK_N_SUBSTEPS  # 0.02s (50 Hz)
MICRODUCK_MAX_STEPS = 400  # 8.0s episode

MICRODUCK_OBS_DIM = 55
MICRODUCK_ACTION_DIM = 14
MICRODUCK_HIDDEN_DIM = 128
MICRODUCK_HEAD_DIM = 128
MICRODUCK_LOW_RANK = 8
MICRODUCK_EVAL_SEEDS = (0, 1)

MICRODUCK_LORA_TARGETS = (
    "net.encoder.0",
    "net.encoder.2",
    "net.cell.linear_ih",
    "net.cell.linear_hh",
    "net.decoder.0",
    "net.decoder.2",
)

MICRODUCK_ACTION_LABELS = (
    "l_hip_yaw",
    "l_hip_roll",
    "l_hip_pitch",
    "l_knee",
    "l_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "r_hip_yaw",
    "r_hip_roll",
    "r_hip_pitch",
    "r_knee",
    "r_ankle",
)

MICRODUCK_OBS_LABELS = (
    "gx", "gy", "gz",
    "vx_loc", "vy_loc", "vz_loc",
    "wx_loc", "wy_loc", "wz_loc",
    "vx_w", "vy_w",
    *(f"q_{j}" for j in MICRODUCK_ACTION_LABELS),
    *(f"dq_{j}" for j in MICRODUCK_ACTION_LABELS),
    *(f"act_{j}" for j in MICRODUCK_ACTION_LABELS),
    "cmd_vx", "cmd_vy",
)

MICRODUCK_POLICY_SPEC: dict[str, object] = {
    "arch": "gru_mlp",
    "obsDim": MICRODUCK_OBS_DIM,
    "actionDim": MICRODUCK_ACTION_DIM,
    "hiddenDim": MICRODUCK_HIDDEN_DIM,
    "headDim": MICRODUCK_HEAD_DIM,
    "lowRank": MICRODUCK_LOW_RANK,
    "loraTargets": list(MICRODUCK_LORA_TARGETS),
    "actionDist": "squashed_gaussian",
    "actionBounds": [-1.0, 1.0],
    "oscillatoryInit": True,
}

MICRODUCK_ENV_PARAMS: dict[str, object] = {
    "sceneId": "microduck",
    "physicsDt": MICRODUCK_PHYSICS_DT,
    "nSubsteps": MICRODUCK_N_SUBSTEPS,
    "ctrlDt": MICRODUCK_DT,
    "actionScale": 0.5,
    "kpMult": 3.0,
    "kv": 0.1,
    "command": [0.15, 0.0],
    "progressWeight": 350.0,
    "clearanceWeight": 900.0,
    "clearanceTarget": 0.008,
    "footDragWeight": 160.0,
    "altStepBonus": 400.0,
    "swayWeight": 20.0,
    "bodyAngVelWeight": 0.04,
    "actionRateWeight": 0.10,
    "fallPenalty": 3500.0,
    "minTrunkZ": 0.04,
    "minUprightness": 0.35,
    "legJoints": [0, 1, 2, 3, 4, 9, 10, 11, 12, 13],
}

MICRODUCK_HYPERPARAM_SPECS: tuple[dict[str, object], ...] = (
    {
        "key": "actionScale",
        "label": "Joint Action Scale",
        "type": "float",
        "default": 0.50,
        "min": 0.20,
        "max": 0.85,
        "step": 0.05,
        "description": "Per-step joint target amplitude (rad) around standing pose.",
    },
)
