"""SO-101 6-DOF robotic arm tabletop cube pick-and-lift constants & PopuLoRA policy spec."""

from __future__ import annotations

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

SO101_ENV_PARAMS: dict[str, object] = {
    "sceneId": "so101",
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
