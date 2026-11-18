"""Unitree Go2 12-DOF quadrupedal locomotion task constants and PopuLoRA policy specification."""

from __future__ import annotations

UNITREE_GO2_CONTROL_HZ = 50
UNITREE_GO2_N_SUBSTEPS = 5
UNITREE_GO2_PHYSICS_DT = 1.0 / (UNITREE_GO2_CONTROL_HZ * UNITREE_GO2_N_SUBSTEPS)  # 0.004s
UNITREE_GO2_DT = UNITREE_GO2_PHYSICS_DT * UNITREE_GO2_N_SUBSTEPS  # 0.02s (50 Hz)
UNITREE_GO2_MAX_STEPS = 400  # 8.0s episode

UNITREE_GO2_OBS_DIM = 49
UNITREE_GO2_ACTION_DIM = 12
UNITREE_GO2_HIDDEN_DIM = 96
UNITREE_GO2_HEAD_DIM = 96
UNITREE_GO2_LOW_RANK = 6
UNITREE_GO2_EVAL_SEEDS = (0, 1)

UNITREE_GO2_LORA_TARGETS = (
    "net.encoder.0",
    "net.encoder.2",
    "net.cell.linear_ih",
    "net.cell.linear_hh",
    "net.decoder.0",
    "net.decoder.2",
)

UNITREE_GO2_ACTION_LABELS = (
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
)

UNITREE_GO2_OBS_LABELS = (
    "gx", "gy", "gz",
    "vx_loc", "vy_loc", "vz_loc",
    "wx_loc", "wy_loc", "wz_loc",
    "vx_w", "vy_w",
    *(f"q_{j}" for j in UNITREE_GO2_ACTION_LABELS),
    *(f"dq_{j}" for j in UNITREE_GO2_ACTION_LABELS),
    *(f"act_{j}" for j in UNITREE_GO2_ACTION_LABELS),
    "cmd_vx", "cmd_vy",
)

UNITREE_GO2_POLICY_SPEC: dict[str, object] = {
    "arch": "gru_mlp",
    "obsDim": UNITREE_GO2_OBS_DIM,
    "actionDim": UNITREE_GO2_ACTION_DIM,
    "hiddenDim": UNITREE_GO2_HIDDEN_DIM,
    "headDim": UNITREE_GO2_HEAD_DIM,
    "lowRank": UNITREE_GO2_LOW_RANK,
    "loraTargets": list(UNITREE_GO2_LORA_TARGETS),
    "actionDist": "squashed_gaussian",
    "actionBounds": [-1.0, 1.0],
    "oscillatoryInit": True,
}

UNITREE_GO2_ENV_PARAMS: dict[str, object] = {
    "sceneId": "unitree_go2",
    "physicsDt": UNITREE_GO2_PHYSICS_DT,
    "nSubsteps": UNITREE_GO2_N_SUBSTEPS,
    "ctrlDt": UNITREE_GO2_DT,
    "actionScale": 0.42,
    "angVelObsScale": 0.25,
    "jointVelObsScale": 0.05,
    "command": [0.65, 0.0],
    "footRadius": 0.022,
    "nominalBaseZ": 0.268,
    "minBaseZ": 0.145,
    "minUprightness": 0.45,
    "signedProgressWeight": 140.0,
    "highWaterWeight": 60.0,
    "velTrackWeight": 16.0,
    "clearanceWeight": 24.0,
    "clearanceTarget": 0.055,
    "stepStrideWeight": 35.0,
    "footSlipWeight": 1.2,
    "trotBonusWeight": 8.0,
    "orientWeight": 8.0,
    "lateralDriftWeight": 12.0,
    "heightWeight": 20.0,
    "bodyAngVelWeight": 0.35,
    "vertVelWeight": 1.5,
    "hipPoseWeight": 3.0,
    "actionRateWeight": 0.08,
    "fallPenalty": 120.0,
    "hipJoints": [0, 3, 6, 9],
}
