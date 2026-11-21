"""Native Python MuJoCo + PopuLoRA simulators for crowdsourced local shard evaluation.

Mirrors ``frontend/sim_worker.js`` across ``microduck``, ``unitree_go2``, and ``so101``
(including the ``arm_model='so102'`` parallel-gripper variant backed by ``so_arm_102.urdf``
and STL meshes) so researchers and automated clients can download a shard of a large
population, simulate & rank it locally with native ``mujoco``, and submit attestations back.
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import copy
import math
import threading
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np

from direct_evolution_together.populora_engine import (
    get_or_create_base_model,
    populora_genome_hash,
)

from .registry import get_task_spec
from .rng import mulberry32
from .scene_assets import ASSETS_DIR, get_scene_manifest, unpack_vfs_entries


# helpers

def exists(val: Any) -> bool:
    return val is not None


def default(val: Any, d: Any) -> Any:
    return val if exists(val) else d


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _linear(layer: tuple[np.ndarray, np.ndarray | None], x: np.ndarray) -> np.ndarray:
    w, b = layer
    return w @ x + b if exists(b) else w @ x


def _linear_tanh(layer: tuple[np.ndarray, np.ndarray | None], x: np.ndarray) -> np.ndarray:
    return np.tanh(_linear(layer, x))


def _signed(val: float, fmt: str = ".0f") -> str:
    clean = 0.0 if val == 0 else val
    return f"{clean:+{fmt}}"

_SCENE_LOCK = threading.Lock()
_SCENE_MODEL_CACHE: dict[str, mujoco.MjModel] = {}
_DECODED_BASE_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def _decode_f16_np(b64_str: str) -> np.ndarray:
    raw = base64.b64decode(b64_str)
    return np.frombuffer(raw, dtype="<f2").astype(np.float32)


def _decode_f32_np(b64_str: str) -> np.ndarray:
    raw = base64.b64decode(b64_str)
    return np.frombuffer(raw, dtype="<f4").astype(np.float32).copy()


def _load_scene_model(scene_id: str) -> mujoco.MjModel:
    clean = str(scene_id).strip().lower()
    with _SCENE_LOCK:
        proto = _SCENE_MODEL_CACHE.get(clean)
        if proto is None:
            manifest = get_scene_manifest(clean)
            if not manifest:
                raise ValueError(f"Unknown MuJoCo scene '{scene_id}'.")
            phys_path = ASSETS_DIR / str(manifest["physics_asset"])
            vfs_files = unpack_vfs_entries(phys_path.read_bytes())
            entry_name = str(manifest["entry_xml"])
            entry_xml = vfs_files[entry_name].decode("utf-8")
            assets = {k: v for k, v in vfs_files.items() if k != entry_name}
            proto = mujoco.MjModel.from_xml_string(entry_xml, assets=assets)
            _SCENE_MODEL_CACHE[clean] = proto
    return copy.copy(proto)


def _parse_base_model_tensors(
    task_id: str,
    genome: Mapping[str, Any],
    supplied_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    req_hash = str(
        (supplied_payload.get("base_hash") if supplied_payload else None)
        or genome.get("base_hash")
        or ""
    )
    if req_hash:
        with _SCENE_LOCK:
            cached_bm = _DECODED_BASE_CACHE.get((task_id, req_hash))
        if cached_bm is not None:
            return cached_bm

    if supplied_payload and isinstance(supplied_payload.get("tensors"), Mapping):
        raw: Mapping[str, Any] = supplied_payload
    else:
        spec = get_task_spec(task_id)
        if not spec.policy_spec:
            raise ValueError(f"Task '{task_id}' has no PopuLoRA policy_spec.")
        _, _, raw = get_or_create_base_model(
            spec.policy_spec,
            base_seed=int(genome.get("base_seed", 1) or 1),
            base_hash=str(genome.get("base_hash") or "") or None,
        )
    tensors_raw = raw.get("tensors") or {}
    decoded: dict[str, np.ndarray] = {}
    if isinstance(tensors_raw, Mapping):
        for tname, entry in tensors_raw.items():
            if isinstance(entry, Mapping):
                b64 = str(entry.get("b64") or "")
                decoded[str(tname)] = (
                    _decode_f32_np(b64) if entry.get("dtype") == "f32" else _decode_f16_np(b64)
                )
    out_bm = {
        "arch": str(raw.get("arch") or "gru_mlp"),
        "obs_dim": int(raw["obs_dim"]),
        "action_dim": int(raw["action_dim"]),
        "hidden_dim": int(raw.get("hidden_dim") or 128),
        "head_dim": int(raw.get("head_dim") or 128),
        "low_rank": int(raw.get("low_rank") or 8),
        "tensors": decoded,
    }
    resolved_h = str(raw.get("base_hash") or req_hash)
    if resolved_h:
        with _SCENE_LOCK:
            if len(_DECODED_BASE_CACHE) > 32:
                _DECODED_BASE_CACHE.clear()
            _DECODED_BASE_CACHE[(task_id, resolved_h)] = out_bm
    return out_bm


class _CompiledPopuloraPolicy:
    """Fast NumPy recurrent/feedforward PopuLoRA policy runner matching ``sim_worker.js``."""

    def __init__(self, base_model: Mapping[str, Any], genome: Mapping[str, Any]) -> None:
        self.arch = str(base_model["arch"])
        obs_dim = int(base_model["obs_dim"])
        act_dim = int(base_model["action_dim"])
        hidden_dim = int(base_model["hidden_dim"])
        head_dim = int(base_model["head_dim"])
        r = int(genome.get("low_rank") or base_model.get("low_rank") or 8)
        tensors: Mapping[str, np.ndarray] = base_model["tensors"]
        adapters = genome.get("adapters") if isinstance(genome.get("adapters"), Mapping) else {}

        def _merge(path: str, out_d: int, in_d: int, has_bias: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
            bw = tensors.get(f"{path}.weight")
            w = bw.reshape(out_d, in_d).copy() if exists(bw) else np.zeros((out_d, in_d), dtype=np.float32)
            bb = tensors.get(f"{path}.bias") if has_bias else None
            b = bb.reshape(out_d).copy() if exists(bb) else None
            ad = adapters.get(path.replace(".", "_")) if isinstance(adapters, Mapping) else None
            if isinstance(ad, Mapping) and ad.get("down") and ad.get("up"):
                down = _decode_f16_np(str(ad["down"])).reshape(in_d, r)
                up = _decode_f16_np(str(ad["up"])).reshape(out_d, r)
                w += up @ down.T
            return w, b

        self.hidden_dim = hidden_dim
        if self.arch == "mlp":
            self.enc0 = _merge("encoder.0", hidden_dim, obs_dim, True)
            self.enc2 = _merge("encoder.2", head_dim, hidden_dim, True)
            self.head = _merge("head", act_dim, head_dim, True)
            self.init_mem = np.zeros(hidden_dim, dtype=np.float32)
        else:
            init_m = tensors.get("init_memory")
            self.init_mem = (
                init_m.reshape(-1)[:hidden_dim].copy()
                if exists(init_m)
                else np.zeros(hidden_dim, dtype=np.float32)
            )
            self.enc0 = _merge("net.encoder.0", hidden_dim, obs_dim, True)
            self.enc2 = _merge("net.encoder.2", hidden_dim, hidden_dim, True)
            self.ih = _merge("net.cell.linear_ih", 3 * hidden_dim, hidden_dim, True)
            self.hh = _merge("net.cell.linear_hh", 3 * hidden_dim, hidden_dim, False)
            self.dec0 = _merge("net.decoder.0", head_dim, hidden_dim, True)
            self.dec2 = _merge("net.decoder.2", act_dim, head_dim, True)
        self.mem = self.init_mem.copy()

    def reset(self) -> None:
        self.mem = self.init_mem.copy()

    def step(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        if self.arch == "mlp":
            h = _linear_tanh(self.enc0, x)
            h = _linear_tanh(self.enc2, h)
            return _linear_tanh(self.head, h)

        h_dim = self.hidden_dim
        enc = _linear_tanh(self.enc2, _linear_tanh(self.enc0, x))
        g_ih = _linear(self.ih, enc)
        g_hh = _linear(self.hh, self.mem)
        r_gate = _sigmoid(g_ih[:h_dim] + g_hh[:h_dim])
        z_gate = _sigmoid(g_ih[h_dim : 2 * h_dim] + g_hh[h_dim : 2 * h_dim])
        n_cand = np.tanh(g_ih[2 * h_dim :] + r_gate * g_hh[2 * h_dim :])
        self.mem = (1.0 - z_gate) * n_cand + z_gate * self.mem
        return _linear_tanh(self.dec2, _linear_tanh(self.dec0, self.mem))


def _capture_body_poses(model: mujoco.MjModel, data: mujoco.MjData) -> list[float]:
    row: list[float] = []
    for b in range(1, int(model.nbody)):
        px, py, pz = data.xpos[b]
        qw, qx, qy, qz = data.xquat[b]
        row.extend(
            [
                round(float(px), 4),
                round(float(py), 4),
                round(float(pz), 4),
                round(float(qw), 4),
                round(float(qx), 4),
                round(float(qy), 4),
                round(float(qz), 4),
            ]
        )
    return row


def _subsample_trajectory(rows: list[list[float]], num_sub: int = 16) -> list[list[float]]:
    if not rows:
        return []
    out: list[list[float]] = []
    n = len(rows)
    for k in range(num_sub):
        r_idx = round((k * (n - 1)) / max(1, num_sub - 1))
        out.append([round(float(v), 4) for v in rows[r_idx]])
    return out


def _tail_percentile_95(samples: Sequence[float], skip_head: int = 20) -> float:
    if not samples:
        return 0.0
    tail = list(samples[min(skip_head, max(0, len(samples) - 1)) :])
    if not tail:
        return 0.0
    tail.sort()
    idx = min(len(tail) - 1, int(math.floor(0.95 * (len(tail) - 1))))
    return float(tail[idx])


def _ctrl_bounds(model: mujoco.MjModel, fallback_abs: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    nu = int(model.nu)
    cmin = np.empty(nu, dtype=np.float64)
    cmax = np.empty(nu, dtype=np.float64)
    for a in range(nu):
        limited = bool(model.actuator_ctrllimited[a])
        cmin[a] = float(model.actuator_ctrlrange[a, 0]) if limited else -fallback_abs
        cmax[a] = float(model.actuator_ctrlrange[a, 1]) if limited else fallback_abs
    return cmin, cmax


def microduck_rollout(
    genome: Mapping[str, Any],
    *,
    seed: int = 0,
    max_steps: int = 400,
    capture_poses: bool = True,
    pose_stride: int = 2,
    env_params: Mapping[str, Any] | None = None,
    base_model_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    spec = get_task_spec("microduck")
    p: dict[str, Any] = spec.resolve_env_params(env_params)
    model = _load_scene_model("microduck")
    data = mujoco.MjData(model)

    physics_dt = float(p.get("physics_dt", 0.004))
    kp_mult = float(p.get("kp_mult", 3.0))
    kv = float(p.get("kv", 0.1))
    model.opt.timestep = physics_dt
    for a in range(int(model.nu)):
        model.actuator_gainprm[a, 0] *= kp_mult
        model.actuator_biasprm[a, 1] = -model.actuator_gainprm[a, 0]
        model.actuator_biasprm[a, 2] = -kv

    stand_idx = max(0, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND"))
    home_qpos = model.key_qpos[stand_idx].copy()
    home_ctrl = model.key_ctrl[stand_idx].copy()
    floor_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_geoms = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision"),
    ]
    foot_sites = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot"),
    ]
    foot_bodies = {int(model.geom_bodyid[foot_geoms[0]]), int(model.geom_bodyid[foot_geoms[1]])}
    non_foot_bodies = [b for b in range(1, int(model.nbody)) if b not in foot_bodies]

    bm = _parse_base_model_tensors("microduck", genome, base_model_payload)
    policy = _CompiledPopuloraPolicy(bm, genome)
    policy.reset()

    n_substeps = int(p.get("n_substeps", 5))
    ctrl_dt = float(p.get("ctrl_dt", 0.02))
    action_scale = float(p.get("action_scale", 0.5))
    cmd = list(p.get("command") or [0.15, 0.0])
    cmd_speed = math.hypot(float(cmd[0]), float(cmd[1]))
    has_cmd = cmd_speed > 0.01
    cmd_dir_x = float(cmd[0]) / cmd_speed if has_cmd else 1.0
    cmd_dir_y = float(cmd[1]) / cmd_speed if has_cmd else 0.0

    progress_w = float(p.get("progress_weight", 350.0))
    clearance_w = float(p.get("clearance_weight", 900.0))
    clearance_target = float(p.get("clearance_target", 0.008))
    foot_drag_w = float(p.get("foot_drag_weight", 160.0))
    alt_step_bonus = float(p.get("alt_step_bonus", 400.0))
    sway_w = float(p.get("sway_weight", 20.0))
    body_ang_w = float(p.get("body_ang_vel_weight", 0.04))
    action_rate_w = float(p.get("action_rate_weight", 0.10))
    fall_penalty = float(p.get("fall_penalty", 3500.0))
    min_trunk_z = float(p.get("min_trunk_z", 0.04))
    min_upright = float(p.get("min_uprightness", 0.35))
    leg_joints = [int(x) for x in (p.get("leg_joints") or [0, 1, 2, 3, 4, 9, 10, 11, 12, 13])]

    mujoco.mj_resetData(model, data)
    rng = mulberry32(seed)
    data.qpos[:] = home_qpos
    data.qpos[2] += rng() * 0.002
    for j in range(int(model.nu)):
        data.qpos[7 + j] += (rng() * 2.0 - 1.0) * 0.005
    for v in range(int(model.nv)):
        data.qvel[v] = (rng() * 2.0 - 1.0) * 0.01
    data.ctrl[:] = home_ctrl
    mujoco.mj_forward(model, data)

    start_x = float(data.qpos[0])
    start_y = float(data.qpos[1])
    last_action = np.zeros(14, dtype=np.float32)
    prev_foot_xy = [
        [float(data.site_xpos[foot_sites[0], 0]), float(data.site_xpos[foot_sites[0], 1])],
        [float(data.site_xpos[foot_sites[1], 0]), float(data.site_xpos[foot_sites[1], 1])],
    ]
    max_foot_proj = [
        (prev_foot_xy[0][0] - start_x) * cmd_dir_x + (prev_foot_xy[0][1] - start_y) * cmd_dir_y,
        (prev_foot_xy[1][0] - start_x) * cmd_dir_x + (prev_foot_xy[1][1] - start_y) * cmd_dir_y,
    ]
    td_proj = [max_foot_proj[0], max_foot_proj[1]]
    lifted_adv = [0.0, 0.0]
    swing_steps = [0, 0]
    swing_peak = [0.0, 0.0]
    last_foot = -1
    max_deff = 0.0
    ssup_steps = 0
    total_return = 0.0
    min_roll = float("inf")
    max_roll = float("-inf")
    foot_clears_mm: list[tuple[float, float]] = []
    raw_gait_rows: list[list[float]] = []
    disp_int_x = 0.0
    disp_int_y = 0.0
    vel_local = np.zeros(6, dtype=np.float64)
    vel_world = np.zeros(6, dtype=np.float64)
    obs = np.zeros(55, dtype=np.float32)

    body_poses = [_capture_body_poses(model, data)] if capture_poses else None
    vx_trace = [0.0] if capture_poses else None
    steps = 0
    terminated = False

    for s in range(max_steps):
        xmat1 = data.xmat[1]
        obs[0] = -float(xmat1[6])
        obs[1] = -float(xmat1[7])
        obs[2] = -float(xmat1[8])
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, 1, vel_local, 1)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, 1, vel_world, 0)
        obs[3:6] = vel_local[3:6]
        obs[6:9] = vel_local[0:3]
        obs[9] = vel_world[3]
        obs[10] = vel_world[4]
        obs[11:25] = data.qpos[7:21] - home_qpos[7:21]
        obs[25:39] = data.qvel[6:20]
        obs[39:53] = last_action
        obs[53] = float(cmd[0])
        obs[54] = float(cmd[1])

        disp_int_x += float(obs[9]) * ctrl_dt
        disp_int_y += float(obs[10]) * ctrl_dt
        gait_row = [
            disp_int_x,
            disp_int_y,
            float(obs[9]),
            float(obs[10]),
            float(obs[0]),
            float(obs[1]),
            float(obs[2]),
            float(obs[6]),
            float(obs[7]),
            float(obs[8]),
            *(float(obs[11 + lj]) for lj in leg_joints[:10]),
        ]
        raw_gait_rows.append(gait_row)

        action = policy.step(obs)
        act_eff = np.clip(np.where(np.isfinite(action), action, 0.0), -1.0, 1.0)
        action_rate = float(np.mean((act_eff - last_action) ** 2))
        data.ctrl[:14] = home_ctrl[:14] + action_scale * act_eff
        last_action[:] = act_eff

        for _ in range(n_substeps):
            mujoco.mj_step(model, data)
        steps += 1

        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            total_return += -10.0 - fall_penalty
            terminated = True
            break

        xmat1 = data.xmat[1]
        uprightness = float(xmat1[8])
        roll_deg = math.degrees(math.atan2(float(xmat1[7]), uprightness))
        min_roll = min(min_roll, roll_deg)
        max_roll = max(max_roll, roll_deg)
        min_nf_z = min(float(data.xpos[b, 2]) for b in non_foot_bodies)
        floor_clip = max(0.0, -min_nf_z)
        terminated = uprightness < min_upright or float(data.qpos[2]) < min_trunk_z or floor_clip > 0.015

        v_wx = float(data.cvel[1, 3])
        v_wy = float(data.cvel[1, 4])
        disp_x = float(data.qpos[0]) - start_x
        disp_y = float(data.qpos[1]) - start_y
        v_proj = v_wx * cmd_dir_x + v_wy * cmd_dir_y if has_cmd else 0.0
        v_lat = math.hypot(v_wx - v_proj * cmd_dir_x, v_wy - v_proj * cmd_dir_y)
        fwd_dist = disp_x * cmd_dir_x + disp_y * cmd_dir_y if has_cmd else 0.0
        lat_drift = math.hypot(disp_x - fwd_dist * cmd_dir_x, disp_y - fwd_dist * cmd_dir_y)
        heading_cos = float(xmat1[0]) * cmd_dir_x + float(xmat1[3]) * cmd_dir_y if has_cmd else 1.0

        d_eff = min(0.22, max(0.0, fwd_dist - 0.03 - 0.55 * lat_drift))
        frontier_vel = max(0.0, d_eff - max_deff) / ctrl_dt
        max_deff = max(max_deff, d_eff)

        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, 1, vel_local, 1)
        roll_yaw_sq = float(vel_local[0] ** 2 + vel_local[2] ** 2)
        pose_gate = math.exp(
            -12.0 * max(0.0, 1.0 - uprightness)
            - 10.0 * max(0.0, 1.0 - heading_cos)
            - 55.0 * (lat_drift * lat_drift)
        )
        sway_gate = math.exp(-10.0 * (v_lat * v_lat) - 0.06 * roll_yaw_sq)
        step_reward = (
            pose_gate * math.sqrt(sway_gate) * (progress_w * frontier_vel + 0.25 * progress_w * min(v_proj, cmd_speed))
            - sway_w * (v_lat * v_lat + 0.80 * lat_drift * lat_drift)
            - body_ang_w * roll_yaw_sq
            - action_rate_w * action_rate
        )

        if has_cmd and clearance_w > 0.0:
            c0 = float(mujoco.mj_geomDistance(model, data, floor_geom, foot_geoms[0], 0.10, None))
            c1 = float(mujoco.mj_geomDistance(model, data, floor_geom, foot_geoms[1], 0.10, None))
            c = [c0, c1]
            fxy0 = [float(data.site_xpos[foot_sites[0], 0]), float(data.site_xpos[foot_sites[0], 1])]
            fxy1 = [float(data.site_xpos[foot_sites[1], 0]), float(data.site_xpos[foot_sites[1], 1])]
            fspd0 = math.hypot(fxy0[0] - prev_foot_xy[0][0], fxy0[1] - prev_foot_xy[0][1]) / ctrl_dt
            fspd1 = math.hypot(fxy1[0] - prev_foot_xy[1][0], fxy1[1] - prev_foot_xy[1][1]) / ctrl_dt
            prev_foot_xy[0], prev_foot_xy[1] = fxy0, fxy1

            fp0 = (fxy0[0] - start_x) * cmd_dir_x + (fxy0[1] - start_y) * cmd_dir_y - 0.50 * lat_drift
            fp1 = (fxy1[0] - start_x) * cmd_dir_x + (fxy1[1] - start_y) * cmd_dir_y - 0.50 * lat_drift
            fp = [fp0, fp1]
            fvel0 = max(0.0, fp0 - max_foot_proj[0]) / ctrl_dt
            fvel1 = max(0.0, fp1 - max_foot_proj[1]) / ctrl_dt
            max_foot_proj[0] = max(max_foot_proj[0], fp0)
            max_foot_proj[1] = max(max_foot_proj[1], fp1)

            u0 = max(0.0, min(1.0, (c0 + 0.0003) / clearance_target))
            u1 = max(0.0, min(1.0, (c1 + 0.0003) / clearance_target))
            opp_g0 = max(0.0, min(1.0, 1.0 - max(0.0, c1 - 0.0005) / 0.0015))
            opp_g1 = max(0.0, min(1.0, 1.0 - max(0.0, c0 - 0.0005) / 0.0015))
            sw0 = u0 * opp_g0 * fvel0
            sw1 = u1 * opp_g1 * fvel1
            prev_min = min(lifted_adv[0], lifted_adv[1])
            lifted_adv[0] += sw0 * ctrl_dt
            lifted_adv[1] += sw1 * ctrl_dt
            sym_vel = (min(lifted_adv[0], lifted_adv[1]) - prev_min) / ctrl_dt

            c_scale = clearance_w / 900.0
            drag_ramp = max(0.0, min(1.0, (fwd_dist - 0.16) / 0.12))
            drag_term = (1.0 - u0) ** 2 * fspd0 + (1.0 - u1) ** 2 * fspd1
            step_reward += pose_gate * clearance_w * (sw0 + sw1 + 1.8 * sym_vel) - c_scale * foot_drag_w * drag_ramp * drag_term

            for i in range(2):
                if c[i] > 0.0010 and c[1 - i] <= 0.0008:
                    swing_steps[i] += 1
                    swing_peak[i] = max(swing_peak[i], c[i])
                elif c[i] <= 0.0006:
                    if (
                        swing_steps[i] >= 2
                        and swing_peak[i] >= 0.0015
                        and fp[i] > max(fp[1 - i] + 0.004, td_proj[i] + 0.008)
                        and last_foot != i
                    ):
                        step_reward += (
                            pose_gate
                            * c_scale
                            * alt_step_bonus
                            * min(swing_steps[i] / 5.0, 1.0)
                            * min(swing_peak[i] / clearance_target, 1.0)
                        )
                        td_proj[i] = fp[i]
                        last_foot = i
                    swing_steps[i] = 0
                    swing_peak[i] = 0.0

            opp_gr0 = c1 <= 0.0008
            opp_gr1 = c0 <= 0.0008
            if (c0 > 0.0010 and opp_gr0) or (c1 > 0.0010 and opp_gr1):
                ssup_steps += 1
            cl0 = max(0.0, c0) * 1000.0 if opp_gr0 else 0.0
            cl1 = max(0.0, c1) * 1000.0 if opp_gr1 else 0.0
            foot_clears_mm.append((cl0, cl1))

        if terminated:
            step_reward -= fall_penalty
        total_return += step_reward

        if capture_poses and (s % max(1, pose_stride) == 0 or terminated or s == max_steps - 1):
            assert body_poses is not None and vx_trace is not None
            body_poses.append(_capture_body_poses(model, data))
            vx_trace.append(round(v_proj, 3))

        if terminated:
            break

    dx = float(data.qpos[0]) - start_x
    abs_y = abs(float(data.qpos[1]) - start_y)
    roll_ptp = (max_roll - min_roll) if min_roll <= max_roll else 0.0
    ssup_pct = (100.0 * ssup_steps) / max(1, steps)
    clear_mm = _tail_percentile_95([max(r[0], r[1]) for r in foot_clears_mm], 20)
    min_lr_mm = min(
        _tail_percentile_95([r[0] for r in foot_clears_mm], 20),
        _tail_percentile_95([r[1] for r in foot_clears_mm], 20),
    )
    return {
        "seed": seed,
        "steps": steps,
        "terminated": terminated,
        "distance": dx,
        "abs_y": abs_y,
        "roll_deg": roll_ptp,
        "clear_mm": clear_mm,
        "min_lr_mm": min_lr_mm,
        "ssup_pct": ssup_pct,
        "return": total_return,
        "pose_stride": max(1, pose_stride),
        "body_poses": body_poses,
        "vx_trace": vx_trace,
        "traj_sample": _subsample_trajectory(raw_gait_rows, 16),
    }


def unitree_go2_rollout(
    genome: Mapping[str, Any],
    *,
    seed: int = 0,
    max_steps: int = 400,
    capture_poses: bool = True,
    pose_stride: int = 2,
    env_params: Mapping[str, Any] | None = None,
    base_model_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    spec = get_task_spec("unitree_go2")
    p: dict[str, Any] = spec.resolve_env_params(env_params)
    model = _load_scene_model("unitree_go2")
    data = mujoco.MjData(model)
    model.opt.timestep = float(p.get("physics_dt", 0.004))

    home_idx = max(0, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home"))
    home_qpos = model.key_qpos[home_idx].copy()
    home_ctrl = model.key_ctrl[home_idx].copy()
    floor_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("FL", "FR", "RL", "RR")]
    foot_sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in ("FL_foot", "FR_foot", "RL_foot", "RR_foot")]
    foot_body_set = {int(model.geom_bodyid[gid]) for gid in foot_geoms}
    non_foot_bodies = [b for b in range(1, int(model.nbody)) if b not in foot_body_set]
    ctrl_min, ctrl_max = _ctrl_bounds(model, 10.0)

    bm = _parse_base_model_tensors("unitree_go2", genome, base_model_payload)
    policy = _CompiledPopuloraPolicy(bm, genome)
    policy.reset()

    n_substeps = int(p.get("n_substeps", 5))
    ctrl_dt = float(p.get("ctrl_dt", 0.02))
    action_scale = float(p.get("action_scale", 0.42))
    ang_vel_scale = float(p.get("ang_vel_obs_scale", 0.25))
    jnt_vel_scale = float(p.get("joint_vel_obs_scale", 0.05))
    cmd = list(p.get("command") or [0.65, 0.0])
    cmd_vx = float(cmd[0])
    foot_radius = float(p.get("foot_radius", 0.022))
    nominal_z = float(p.get("nominal_base_z", 0.268))
    min_base_z = float(p.get("min_base_z", 0.145))
    min_upright = float(p.get("min_uprightness", 0.45))
    hip_joints = [int(x) for x in (p.get("hip_joints") or [0, 3, 6, 9])]

    mujoco.mj_resetData(model, data)
    rng = mulberry32(seed)
    data.qpos[:] = home_qpos
    data.qpos[2] += (rng() * 2.0 - 1.0) * 0.002
    for j in range(12):
        data.qpos[7 + j] += (rng() * 2.0 - 1.0) * 0.01
    for v in range(int(model.nv)):
        data.qvel[v] = (rng() * 2.0 - 1.0) * 0.02
    data.ctrl[:] = home_ctrl
    mujoco.mj_forward(model, data)

    start_x = float(data.qpos[0])
    start_y = float(data.qpos[1])
    prev_x = start_x
    max_x = start_x
    last_action = np.zeros(12, dtype=np.float32)
    prev_fx = np.array([float(data.site_xpos[s, 0]) for s in foot_sites], dtype=np.float64)
    prev_fy = np.array([float(data.site_xpos[s, 1]) for s in foot_sites], dtype=np.float64)
    td_fx = prev_fx.copy()
    air_time = np.zeros(4, dtype=np.float64)
    swing_peak = np.zeros(4, dtype=np.float64)
    vel_local = np.zeros(6, dtype=np.float64)
    vel_world = np.zeros(6, dtype=np.float64)
    obs = np.zeros(49, dtype=np.float32)

    total_return = 0.0
    trot_steps = 0
    swing_clears: list[float] = []
    raw_gait_rows: list[list[float]] = []
    disp_int_x = 0.0
    disp_int_y = 0.0
    body_poses = [_capture_body_poses(model, data)] if capture_poses else None
    vx_trace = [0.0] if capture_poses else None
    steps = 0
    terminated = False

    for s in range(max_steps):
        xmat1 = data.xmat[1]
        obs[0:3] = -xmat1[6:9]
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, 1, vel_local, 1)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, 1, vel_world, 0)
        obs[3:6] = vel_local[3:6]
        obs[6:9] = vel_local[0:3] * ang_vel_scale
        obs[9] = vel_world[3]
        obs[10] = vel_world[4]
        obs[11:23] = data.qpos[7:19] - home_qpos[7:19]
        obs[23:35] = data.qvel[6:18] * jnt_vel_scale
        obs[35:47] = last_action
        obs[47] = float(cmd[0])
        obs[48] = float(cmd[1])

        disp_int_x += float(obs[9]) * ctrl_dt
        disp_int_y += float(obs[10]) * ctrl_dt
        raw_gait_rows.append(
            [
                disp_int_x,
                disp_int_y,
                float(obs[9]),
                float(obs[10]),
                float(obs[0]),
                float(obs[1]),
                float(obs[2]),
                float(obs[6]),
                float(obs[7]),
                float(obs[8]),
                *(float(obs[11 + j]) for j in range(12)),
            ]
        )

        action = policy.step(obs)
        act_eff = np.clip(np.where(np.isfinite(action), action, 0.0), -1.0, 1.0)
        action_rate = float(np.mean((act_eff - last_action) ** 2))
        data.ctrl[:12] = np.clip(home_ctrl[:12] + action_scale * act_eff, ctrl_min, ctrl_max)
        last_action[:] = act_eff

        for _ in range(n_substeps):
            mujoco.mj_step(model, data)
        steps += 1

        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            total_return -= 120.0
            terminated = True
            break

        contacts = [
            float(mujoco.mj_geomDistance(model, data, floor_geom, fg, 0.04, None)) <= 0.0015
            for fg in foot_geoms
        ]
        min_nf_z = min(float(data.xpos[b, 2]) for b in non_foot_bodies)
        base_z = float(data.qpos[2])
        up_z = float(data.xmat[1, 8])
        if min_nf_z < 0.035 or base_z < min_base_z or up_z < min_upright:
            terminated = True

        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, 1, vel_local, 1)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, 1, vel_world, 0)
        vx_w, vy_w, vz_w = float(vel_world[3]), float(vel_world[4]), float(vel_world[5])
        wx_loc, wy_loc = float(vel_local[0]), float(vel_local[1])

        cur_x = float(data.qpos[0])
        dx_step = cur_x - prev_x
        prev_x = cur_x
        d_new = max(0.0, cur_x - max_x)
        max_x = max(max_x, cur_x)

        r_prog = 140.0 * dx_step + 60.0 * d_new
        fwd_gate = max(0.0, min(1.0, (vx_w + 0.02) / 0.25))
        vx_err = (vx_w - cmd_vx) ** 2 + 1.5 * vy_w * vy_w
        r_vel = 16.0 * math.exp(-vx_err / 0.22) * ctrl_dt

        r_feet = 0.0
        step_max_clr_mm = 0.0
        for f in range(4):
            fx, fy, fz = (float(v) for v in data.site_xpos[foot_sites[f]])
            dfx, dfy = fx - prev_fx[f], fy - prev_fy[f]
            foot_v = math.hypot(dfx, dfy) / ctrl_dt
            foot_vx = dfx / ctrl_dt
            prev_fx[f], prev_fy[f] = fx, fy
            clr = max(0.0, fz - foot_radius)
            if contacts[f]:
                stride_fwd = max(0.0, fx - td_fx[f])
                if 0.06 <= air_time[f] <= 0.50 and stride_fwd > 0.01 and vx_w > 0.0:
                    r_feet += 35.0 * min(stride_fwd, 0.18) * min(swing_peak[f] / 0.03, 1.5)
                air_time[f] = 0.0
                swing_peak[f] = 0.0
                td_fx[f] = fx
                r_feet -= 1.2 * foot_v * foot_v * ctrl_dt
            else:
                air_time[f] += ctrl_dt
                swing_peak[f] = max(swing_peak[f], clr)
                step_max_clr_mm = max(step_max_clr_mm, clr * 1000.0)
                if foot_vx > 0.02:
                    r_feet += 24.0 * min(clr, 0.055) * min(foot_vx, 1.5) * ctrl_dt * fwd_gate
        swing_clears.append(step_max_clr_mm)

        diag1 = contacts[0] and contacts[3] and not contacts[1] and not contacts[2]
        diag2 = contacts[1] and contacts[2] and not contacts[0] and not contacts[3]
        if diag1 or diag2:
            trot_steps += 1
            if vx_w > 0.08:
                r_feet += 8.0 * ctrl_dt

        gx, gy = -float(data.xmat[1, 6]), -float(data.xmat[1, 7])
        lat_drift = abs(float(data.qpos[1]) - start_y)
        hip_err = sum(float((data.qpos[7 + hj] - home_qpos[7 + hj]) ** 2) for hj in hip_joints)
        r_pen = (
            8.0 * (gx * gx + gy * gy) * ctrl_dt
            + 12.0 * lat_drift * ctrl_dt
            + 20.0 * ((base_z - nominal_z) ** 2) * ctrl_dt
            + 0.35 * (wx_loc * wx_loc + wy_loc * wy_loc) * ctrl_dt
            + 1.5 * vz_w * vz_w * ctrl_dt
            + 3.0 * hip_err * ctrl_dt
            + 0.08 * action_rate
        )
        step_reward = r_prog + r_vel + r_feet - r_pen
        if terminated:
            step_reward -= 120.0
        total_return += step_reward

        if capture_poses and (s % max(1, pose_stride) == 0 or terminated or s == max_steps - 1):
            assert body_poses is not None and vx_trace is not None
            body_poses.append(_capture_body_poses(model, data))
            vx_trace.append(round(vx_w, 3))

        if terminated:
            break

    dx = float(data.qpos[0]) - start_x
    mean_vx = dx / max(ctrl_dt, steps * ctrl_dt)
    trot_pct = (100.0 * trot_steps) / max(1, steps)
    clear_mm = _tail_percentile_95(swing_clears, 20)
    return {
        "seed": seed,
        "steps": steps,
        "terminated": terminated,
        "distance": dx,
        "mean_vx": mean_vx,
        "abs_y": abs(float(data.qpos[1]) - start_y),
        "clear_mm": clear_mm,
        "trot_pct": trot_pct,
        "return": total_return,
        "pose_stride": max(1, pose_stride),
        "body_poses": body_poses,
        "vx_trace": vx_trace,
        "traj_sample": _subsample_trajectory(raw_gait_rows, 16),
    }


def so101_rollout(
    genome: Mapping[str, Any],
    *,
    seed: int = 0,
    max_steps: int = 250,
    capture_poses: bool = True,
    pose_stride: int = 2,
    env_params: Mapping[str, Any] | None = None,
    base_model_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run SO-101 (angular jaw) or SO-ARM102 (parallel 2-jaw slide gripper) pick & lift rollout."""
    spec = get_task_spec("so101")
    p: dict[str, Any] = spec.resolve_env_params(env_params)
    scene_id = str(p.get("scene_id") or spec.scene_id or "")
    arm_model = str(p.get("arm_model") or scene_id)

    model = _load_scene_model(scene_id)
    data = mujoco.MjData(model)
    model.opt.timestep = float(p.get("physics_dt", 0.004))

    home_idx = max(0, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home"))
    home_qpos = model.key_qpos[home_idx].copy()
    home_ctrl = model.key_ctrl[home_idx].copy()
    fixed_jaw_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_pad")
    moving_jaw_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_pad")
    cube_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_col")
    fixed_tip_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "fixed_tip")
    moving_tip_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "moving_tip")

    cube_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
    if cube_jnt < 0:
        cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cube_jnt = int(model.body_jntadr[cube_body]) if cube_body >= 0 else -1
    cube_qadr = int(model.jnt_qposadr[cube_jnt]) if cube_jnt >= 0 else 6
    cube_vadr = int(model.jnt_dofadr[cube_jnt]) if cube_jnt >= 0 else 6

    gripper_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    gripper_is_slide = (
        gripper_jnt >= 0 and int(model.jnt_type[gripper_jnt]) == int(mujoco.mjtJoint.mjJNT_SLIDE)
    )
    mirror_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_mirror")
    mirror_qadr = int(model.jnt_qposadr[mirror_jnt]) if mirror_jnt >= 0 else -1

    ctrl_min, ctrl_max = _ctrl_bounds(model, 3.14)
    bm = _parse_base_model_tensors("so101", genome, base_model_payload)
    policy = _CompiledPopuloraPolicy(bm, genome)
    policy.reset()

    n_substeps = int(p.get("n_substeps", 5))
    ctrl_dt = float(p.get("ctrl_dt", 0.02))
    action_scales = [float(x) for x in (p.get("action_scales") or [0.65, 0.65, 0.65, 0.65, 0.55, 0.95])]
    goal_pos = [float(x) for x in (p.get("goal_pos") or [0.23, 0.012, 0.14])]
    cube_rest_z = float(p.get("cube_rest_z", 0.016))
    reach_w = float(p.get("reach_weight", 12.0))
    reach_prog_w = float(p.get("reach_progress_weight", 45.0))
    pre_grasp_w = float(p.get("pre_grasp_align_weight", 5.0))
    single_touch_w = float(p.get("single_touch_weight", 2.5))
    pinch_grasp_w = float(p.get("pinch_grasp_weight", 16.0))
    lift_w = float(p.get("lift_weight", 220.0))
    lift_hw_w = float(p.get("lift_high_water_weight", 650.0))
    goal_hold_w = float(p.get("goal_hold_weight", 24.0))
    knock_pen_w = float(p.get("knock_penalty_weight", 15.0))
    floor_scrape_w = float(p.get("floor_scrape_weight", 8.0))
    action_rate_w = float(p.get("action_rate_weight", 0.04))

    mujoco.mj_resetData(model, data)
    rng = mulberry32(seed)
    data.qpos[:] = home_qpos
    for j in range(6):
        pert = (rng() * 2.0 - 1.0) * (0.0015 if (j == 5 and gripper_is_slide) else 0.015)
        data.qpos[j] += pert
    if gripper_is_slide:
        data.qpos[5] = float(np.clip(data.qpos[5], ctrl_min[5], ctrl_max[5]))
        if mirror_qadr >= 0:
            data.qpos[mirror_qadr] = -data.qpos[5]
    if seed > 0:
        data.qpos[cube_qadr] += (rng() * 2.0 - 1.0) * 0.015
        data.qpos[cube_qadr + 1] += (rng() * 2.0 - 1.0) * 0.018

    data.ctrl[:] = home_ctrl
    mujoco.mj_forward(model, data)

    start_cube_x = float(data.qpos[cube_qadr])
    start_cube_y = float(data.qpos[cube_qadr + 1])
    last_action = np.zeros(6, dtype=np.float32)
    obs = np.zeros(33, dtype=np.float32)

    def _pinch_and_cube() -> tuple[float, float, float, float, float, float, float]:
        f_pos = data.site_xpos[fixed_tip_site]
        m_pos = data.site_xpos[moving_tip_site]
        px = 0.5 * float(f_pos[0] + m_pos[0])
        py = 0.5 * float(f_pos[1] + m_pos[1])
        pz = 0.5 * float(f_pos[2] + m_pos[2])
        min_tip_z = min(float(f_pos[2]), float(m_pos[2]))
        cx = float(data.qpos[cube_qadr])
        cy = float(data.qpos[cube_qadr + 1])
        cz = float(data.qpos[cube_qadr + 2])
        return px, py, pz, min_tip_z, cx, cy, cz

    px0, py0, pz0, _, cx0, cy0, cz0 = _pinch_and_cube()
    min_reach_dist = math.hypot(cx0 - px0, cy0 - py0, cz0 - pz0)
    max_lift_mm = 0.0
    max_lift_m = 0.0
    pinch_steps = 0
    total_return = 0.0
    raw_traj_rows: list[list[float]] = []
    body_poses = [_capture_body_poses(model, data)] if capture_poses else None
    vx_trace = [0.0] if capture_poses else None
    steps = 0
    final_dist_cm = min_reach_dist * 100.0
    grip_pos_obs_scale = 25.0 if gripper_is_slide else 1.0
    grip_vel_obs_scale = 2.5 if gripper_is_slide else 0.10

    for s in range(max_steps):
        px, py, pz, _, cx, cy, cz = _pinch_and_cube()
        for j in range(5):
            obs[j] = float(data.qpos[j] - home_qpos[j])
            obs[6 + j] = float(data.qvel[j]) * 0.10
        obs[5] = float(data.qpos[5] - home_qpos[5]) * grip_pos_obs_scale
        obs[11] = float(data.qvel[5]) * grip_vel_obs_scale
        obs[12] = (px - 0.22) * 4.0
        obs[13] = py * 4.0
        obs[14] = (pz - 0.08) * 4.0
        obs[15] = (cx - 0.22) * 4.0
        obs[16] = cy * 4.0
        obs[17] = (cz - 0.08) * 4.0
        obs[18] = (cx - px) * 5.0
        obs[19] = (cy - py) * 5.0
        obs[20] = (cz - pz) * 5.0
        obs[21] = (goal_pos[0] - cx) * 4.0
        obs[22] = (goal_pos[1] - cy) * 4.0
        obs[23] = (goal_pos[2] - cz) * 4.0
        obs[24] = float(data.qvel[cube_vadr]) * 0.5
        obs[25] = float(data.qvel[cube_vadr + 1]) * 0.5
        obs[26] = float(data.qvel[cube_vadr + 2]) * 0.5
        obs[27:33] = last_action

        raw_traj_rows.append(
            [
                px, py, pz,
                cx, cy, cz,
                cx - px, cy - py, cz - pz,
                *(float(obs[j]) for j in range(6)),
            ]
        )

        action = policy.step(obs)
        act_eff = np.clip(np.where(np.isfinite(action), action, 0.0), -1.0, 1.0)
        action_rate = float(np.mean((act_eff - last_action) ** 2))
        for j in range(6):
            raw_target = float(home_ctrl[j]) + action_scales[j] * float(act_eff[j])
            data.ctrl[j] = max(ctrl_min[j], min(ctrl_max[j], raw_target))
        last_action[:] = act_eff

        for _ in range(n_substeps):
            mujoco.mj_step(model, data)
        steps += 1

        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            total_return -= 50.0
            break

        d_fix = float(mujoco.mj_geomDistance(model, data, fixed_jaw_geom, cube_geom, 0.08, None))
        d_mov = float(mujoco.mj_geomDistance(model, data, moving_jaw_geom, cube_geom, 0.08, None))
        px, py, pz, min_tip_z, cx, cy, cz = _pinch_and_cube()
        dist = math.hypot(cx - px, cy - py, cz - pz)
        final_dist_cm = dist * 100.0
        d_reach_prog = max(0.0, min_reach_dist - dist)
        min_reach_dist = min(min_reach_dist, dist)

        both_touch = d_fix <= 0.0025 and d_mov <= 0.0025
        any_touch = d_fix <= 0.003 or d_mov <= 0.003
        if both_touch:
            pinch_steps += 1

        lift_m = max(0.0, cz - cube_rest_z)
        lift_clamped = min(lift_m, 0.15)
        d_lift_new = max(0.0, lift_clamped - max_lift_m)
        max_lift_m = max(max_lift_m, lift_clamped)
        cur_lift_mm = lift_m * 1000.0
        max_lift_mm = max(max_lift_mm, cur_lift_mm)

        r_reach = reach_w * (1.0 - math.tanh(6.5 * dist)) * ctrl_dt + reach_prog_w * d_reach_prog
        grip_pos = float(data.qpos[5])
        if gripper_is_slide:
            if dist > 0.042:
                r_grip = 1.5 * max(0.0, min(1.0, (grip_pos - 0.012) / 0.016)) * ctrl_dt
            else:
                r_grip = pre_grasp_w * (1.0 - max(0.0, min(1.0, (grip_pos - 0.002) / 0.024))) * ctrl_dt
        else:
            if dist > 0.042:
                r_grip = 1.5 * max(0.0, min(1.0, (grip_pos - 0.25) / 0.40)) * ctrl_dt
            else:
                r_grip = pre_grasp_w * (1.0 - max(0.0, min(1.0, (grip_pos + 0.12) / 0.75))) * ctrl_dt
        if both_touch:
            r_grip += pinch_grasp_w * ctrl_dt
        elif any_touch and dist < 0.045:
            r_grip += single_touch_w * ctrl_dt

        goal_dist = math.hypot(cx - goal_pos[0], cy - goal_pos[1], cz - goal_pos[2])
        r_lift = (
            lift_w * lift_clamped * ctrl_dt
            + lift_hw_w * d_lift_new
            + (goal_hold_w * (1.0 - math.tanh(5.0 * goal_dist)) * ctrl_dt if lift_m > 0.015 else 0.0)
        )
        cube_wander = math.hypot(cx - start_cube_x, cy - start_cube_y)
        r_pen = (
            (knock_pen_w * ctrl_dt if cube_wander > 0.14 else 0.0)
            + (floor_scrape_w * ctrl_dt if min_tip_z < 0.002 else 0.0)
            + action_rate_w * action_rate
        )
        step_reward = r_reach + r_grip + r_lift - r_pen
        if math.isfinite(step_reward):
            total_return += step_reward

        if capture_poses and (s % max(1, pose_stride) == 0 or s == max_steps - 1):
            assert body_poses is not None and vx_trace is not None
            body_poses.append(_capture_body_poses(model, data))
            vx_trace.append(round(cur_lift_mm, 1))

    return {
        "seed": seed,
        "arm_model": arm_model,
        "scene_id": scene_id,
        "steps": steps,
        "terminated": False,
        "distance": max_lift_m,
        "lift_mm": max_lift_mm,
        "min_dist_cm": min_reach_dist * 100.0,
        "final_dist_cm": final_dist_cm,
        "pinch_pct": (100.0 * pinch_steps) / max(1, steps),
        "return": total_return,
        "pose_stride": max(1, pose_stride),
        "body_poses": body_poses,
        "vx_trace": vx_trace,
        "traj_sample": _subsample_trajectory(raw_traj_rows, 16),
    }


_ROLLOUT_FNS = {
    "microduck": microduck_rollout,
    "unitree_go2": unitree_go2_rollout,
    "so101": so101_rollout,
}


def evaluate_populora_genome(
    task_id: str,
    genome: Mapping[str, Any],
    *,
    seeds: Sequence[int] | None = None,
    max_steps: int | None = None,
    env_params: Mapping[str, Any] | None = None,
    base_model_payload: Mapping[str, Any] | None = None,
    include_replay: bool = True,
) -> dict[str, Any]:
    """Evaluate one PopuLoRA genome locally in native Python MuJoCo."""
    spec = get_task_spec(task_id)
    rollout_fn = _ROLLOUT_FNS.get(spec.task_id)
    if not exists(rollout_fn):
        raise ValueError(f"Unsupported MuJoCo scene task '{task_id}'.")

    eff_seeds = list(default(seeds, spec.eval_seeds or (0, 1)))
    eff_steps = int(default(max_steps, spec.max_steps or 250))
    n = max(1, len(eff_seeds))

    rollouts = [
        rollout_fn(
            genome,
            seed=int(sd),
            max_steps=eff_steps,
            capture_poses=(include_replay and i == 0),
            pose_stride=2,
            env_params=env_params,
            base_model_payload=base_model_payload,
        )
        for i, sd in enumerate(eff_seeds)
    ]

    mean_ret = round(sum(float(r["return"]) for r in rollouts) / n, 1)
    min_steps = min(int(r["steps"]) for r in rollouts)
    disp = _signed(mean_ret)

    if spec.task_id == "microduck":
        mean_x = round(sum(float(r["distance"]) for r in rollouts) / n, 3)
        mean_steps = round(sum(int(r["steps"]) for r in rollouts) / n, 1)
        clear_mm = round(sum(float(r["clear_mm"]) for r in rollouts) / n, 1)
        min_lr_mm = round(sum(float(r["min_lr_mm"]) for r in rollouts) / n, 1)
        ssup_pct = round(sum(float(r["ssup_pct"]) for r in rollouts) / n)
        roll_deg = round(sum(float(r["roll_deg"]) for r in rollouts) / n, 1)
        abs_y = round(sum(float(r["abs_y"]) for r in rollouts) / n, 3)
        solved = min_steps >= eff_steps and mean_ret >= 180.0 and mean_x >= 0.35
        tier = "solved" if solved else ("mid" if (min_steps >= eff_steps * 0.75 and mean_ret >= 20.0) else "")
        dx_txt = f"{_signed(mean_x, '.2f')}m"
        summary: dict[str, Any] = {
            "task": "microduck",
            "seeds": eff_seeds,
            "return": mean_ret,
            "mean_distance": mean_x,
            "mean_steps": mean_steps,
            "min_steps": min_steps,
            "clear_mm": clear_mm,
            "min_lr_mm": min_lr_mm,
            "ssup_pct": ssup_pct,
            "roll_deg": roll_deg,
            "abs_y": abs_y,
            "solved": solved,
            "primary_value": mean_ret,
            "primary_display": f"{disp} ({dx_txt})",
            "badge_text": f"{disp} · {dx_txt}",
            "tier": tier,
            "secondary_text": f"steps {min_steps}/{eff_steps} · clr {clear_mm:.1f}mm · ssup {ssup_pct}%",
        }
    elif spec.task_id == "unitree_go2":
        mean_x = round(sum(float(r["distance"]) for r in rollouts) / n, 3)
        mean_vx = round(sum(float(r["mean_vx"]) for r in rollouts) / n, 2)
        mean_steps = round(sum(int(r["steps"]) for r in rollouts) / n, 1)
        clear_mm = round(sum(float(r["clear_mm"]) for r in rollouts) / n, 1)
        trot_pct = round(sum(float(r["trot_pct"]) for r in rollouts) / n)
        solved = min_steps >= eff_steps and mean_ret >= 320.0 and mean_x >= 1.5
        tier = "solved" if solved else ("mid" if (min_steps >= eff_steps * 0.75 and mean_ret >= 80.0 and mean_x >= 0.25) else "")
        dx_txt = f"{_signed(mean_x, '.2f')}m"
        summary = {
            "task": "unitree_go2",
            "seeds": eff_seeds,
            "return": mean_ret,
            "mean_distance": mean_x,
            "mean_vx": mean_vx,
            "mean_steps": mean_steps,
            "min_steps": min_steps,
            "clear_mm": clear_mm,
            "trot_pct": trot_pct,
            "solved": solved,
            "primary_value": mean_ret,
            "primary_display": f"{disp} ({dx_txt})",
            "badge_text": f"{disp} · {dx_txt}",
            "tier": tier,
            "secondary_text": f"steps {min_steps}/{eff_steps} · vx {_signed(mean_vx, '.2f')}m/s · clr {clear_mm:.1f}mm",
        }
    else:
        lift_mm = round(sum(float(r["lift_mm"]) for r in rollouts) / n, 1)
        min_dist_cm = round(sum(float(r["min_dist_cm"]) for r in rollouts) / n, 1)
        pinch_pct = round(sum(float(r["pinch_pct"]) for r in rollouts) / n)
        solved = lift_mm >= 100.0 and mean_ret >= 160.0
        tier = "solved" if solved else ("mid" if (lift_mm >= 12.0 or (pinch_pct >= 5 and mean_ret >= 45.0)) else "")
        lift_cm_txt = f"{lift_mm / 10.0:.1f}cm"
        summary = {
            "task": "so101",
            "arm_model": str(rollouts[0].get("arm_model") or "so101"),
            "seeds": eff_seeds,
            "return": mean_ret,
            "lift_mm": lift_mm,
            "min_dist_cm": min_dist_cm,
            "pinch_pct": pinch_pct,
            "min_steps": min_steps,
            "solved": solved,
            "primary_value": mean_ret,
            "primary_display": f"{disp} (lift {lift_cm_txt})",
            "badge_text": f"{disp} · ↑{lift_cm_txt}",
            "tier": tier,
            "secondary_text": f"min dist {min_dist_cm:.1f}cm · grip {pinch_pct}% · lift {lift_mm:.1f}mm",
        }

    primary = rollouts[0]
    replay = (
        {
            "body_poses": primary["body_poses"],
            "pose_stride": primary["pose_stride"],
            "vx_trace": primary["vx_trace"],
            "steps": primary["steps"],
        }
        if include_replay and exists(primary.get("body_poses"))
        else None
    )
    return {
        "summary": summary,
        "traj_sample": primary.get("traj_sample"),
        "replay": replay,
    }


def rank_eval_items(
    items: Sequence[Mapping[str, Any]],
    *,
    value_digits: int = 4,
) -> list[dict[str, Any]]:
    """Deterministically rank evaluated candidates by descending ``primaryValue``."""
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for idx, item in enumerate(items):
        summ = item.get("summary") if isinstance(item.get("summary"), Mapping) else {}
        raw_val = summ.get("primary_value", item.get("primary_value", 0.0))
        try:
            pval = float(raw_val)
        except (TypeError, ValueError):
            pval = 0.0
        if not math.isfinite(pval):
            pval = float("-inf")
        slot_idx = int(item.get("index", idx))
        scored.append(
            (
                pval,
                slot_idx,
                {
                    "index": slot_idx,
                    "id": item.get("id"),
                    "uuid": str(item.get("uuid") or ""),
                    "hash": str(item.get("hash") or ""),
                    "primary_value": round(pval, value_digits) if math.isfinite(pval) else 0.0,
                    "badge_text": str(summ.get("badge_text") or ""),
                },
            )
        )
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [{"rank": rank_pos, **entry} for rank_pos, (_, _, entry) in enumerate(scored, start=1)]


def evaluate_populora_shard(
    task_id: str,
    shard_items: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int] | None = None,
    max_steps: int | None = None,
    env_params: Mapping[str, Any] | None = None,
    base_model_payload: Mapping[str, Any] | None = None,
    include_replay: bool = True,
    max_workers: int = 1,
    max_worker_cap: int = 32,
    hash_length: int = 20,
    value_digits: int = 4,
) -> dict[str, Any]:
    """Simulate a claimed shard of PopuLoRA individuals locally (optionally multi-threaded) and rank them."""
    if not shard_items:
        return {"results": [], "shard_ranking": []}

    def _eval_one(item: Mapping[str, Any]) -> dict[str, Any]:
        genome = item.get("genome") if isinstance(item.get("genome"), Mapping) else item
        gh = str(item.get("hash") or genome.get("hash") or populora_genome_hash(dict(genome), length=hash_length))
        ev = evaluate_populora_genome(
            task_id,
            genome,
            seeds=seeds,
            max_steps=max_steps,
            env_params=env_params,
            base_model_payload=base_model_payload,
            include_replay=include_replay,
        )
        entry: dict[str, Any] = {
            "hash": gh,
            "index": int(item.get("index", 0) or 0),
            "id": item.get("id", genome.get("id")),
            "uuid": str(item.get("uuid") or genome.get("uuid") or ""),
            "summary": ev["summary"],
            "traj_sample": ev["traj_sample"],
        }
        if exists(ev.get("replay")):
            entry["replay"] = ev["replay"]
        return entry

    eff_workers = max(1, min(int(max_workers), len(shard_items), max_worker_cap))
    if eff_workers > 1:
        # Warm scene model + base model caches once before spawning worker threads
        first_g = shard_items[0].get("genome") if isinstance(shard_items[0].get("genome"), Mapping) else shard_items[0]
        _parse_base_model_tensors(task_id, first_g, base_model_payload)
        with ThreadPoolExecutor(max_workers=eff_workers) as pool:
            results = list(pool.map(_eval_one, shard_items))
    else:
        results = [_eval_one(it) for it in shard_items]

    return {
        "results": results,
        "shard_ranking": rank_eval_items(results, value_digits=value_digits),
    }
