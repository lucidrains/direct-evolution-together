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
    "obs_dim": UNITREE_GO2_OBS_DIM,
    "action_dim": UNITREE_GO2_ACTION_DIM,
    "hidden_dim": UNITREE_GO2_HIDDEN_DIM,
    "head_dim": UNITREE_GO2_HEAD_DIM,
    "low_rank": UNITREE_GO2_LOW_RANK,
    "lora_targets": list(UNITREE_GO2_LORA_TARGETS),
    "action_dist": "squashed_gaussian",
    "action_bounds": [-1.0, 1.0],
    "oscillatory_init": True,
}

UNITREE_GO2_ENV_PARAMS: dict[str, object] = {
    "scene_id": "unitree_go2",
    "physics_dt": UNITREE_GO2_PHYSICS_DT,
    "n_substeps": UNITREE_GO2_N_SUBSTEPS,
    "ctrl_dt": UNITREE_GO2_DT,
    "action_scale": 0.42,
    "ang_vel_obs_scale": 0.25,
    "joint_vel_obs_scale": 0.05,
    "command": [0.65, 0.0],
    "foot_radius": 0.022,
    "nominal_base_z": 0.268,
    "min_base_z": 0.145,
    "min_uprightness": 0.45,
    "signed_progress_weight": 140.0,
    "high_water_weight": 60.0,
    "vel_track_weight": 16.0,
    "clearance_weight": 24.0,
    "clearance_target": 0.055,
    "step_stride_weight": 35.0,
    "foot_slip_weight": 1.2,
    "trot_bonus_weight": 8.0,
    "orient_weight": 8.0,
    "lateral_drift_weight": 12.0,
    "height_weight": 20.0,
    "body_ang_vel_weight": 0.35,
    "vert_vel_weight": 1.5,
    "hip_pose_weight": 3.0,
    "action_rate_weight": 0.08,
    "fall_penalty": 120.0,
    "hip_joints": [0, 3, 6, 9],
}

UNITREE_GO2_HYPERPARAM_SPECS: tuple[dict[str, object], ...] = (
    {
        "key": "action_scale",
        "label": "Joint Action Scale",
        "type": "float",
        "default": 0.42,
        "min": 0.20,
        "max": 0.75,
        "step": 0.02,
        "description": "Per-step joint target amplitude (rad) around home stance.",
    },
)
