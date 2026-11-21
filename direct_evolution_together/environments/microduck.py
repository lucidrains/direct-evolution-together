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
    "obs_dim": MICRODUCK_OBS_DIM,
    "action_dim": MICRODUCK_ACTION_DIM,
    "hidden_dim": MICRODUCK_HIDDEN_DIM,
    "head_dim": MICRODUCK_HEAD_DIM,
    "low_rank": MICRODUCK_LOW_RANK,
    "lora_targets": list(MICRODUCK_LORA_TARGETS),
    "action_dist": "squashed_gaussian",
    "action_bounds": [-1.0, 1.0],
    "oscillatory_init": True,
}

MICRODUCK_ENV_PARAMS: dict[str, object] = {
    "scene_id": "microduck",
    "physics_dt": MICRODUCK_PHYSICS_DT,
    "n_substeps": MICRODUCK_N_SUBSTEPS,
    "ctrl_dt": MICRODUCK_DT,
    "action_scale": 0.5,
    "kp_mult": 3.0,
    "kv": 0.1,
    "command": [0.15, 0.0],
    "progress_weight": 350.0,
    "clearance_weight": 900.0,
    "clearance_target": 0.008,
    "foot_drag_weight": 160.0,
    "alt_step_bonus": 400.0,
    "sway_weight": 20.0,
    "body_ang_vel_weight": 0.04,
    "action_rate_weight": 0.10,
    "fall_penalty": 3500.0,
    "min_trunk_z": 0.04,
    "min_uprightness": 0.35,
    "leg_joints": [0, 1, 2, 3, 4, 9, 10, 11, 12, 13],
}

MICRODUCK_HYPERPARAM_SPECS: tuple[dict[str, object], ...] = (
    {
        "key": "action_scale",
        "label": "Joint Action Scale",
        "type": "float",
        "default": 0.50,
        "min": 0.20,
        "max": 0.85,
        "step": 0.05,
        "description": "Per-step joint target amplitude (rad) around standing pose.",
    },
)
