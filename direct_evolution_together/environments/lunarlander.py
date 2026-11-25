"""Faithful Gymnasium LunarLander-v3 (LunarLanderContinuous-v3) Box2D simulation, PopuLoRA runner, and rendering."""

from __future__ import annotations

import base64
import math
from typing import Any, Callable, Mapping, Sequence

import Box2D
from Box2D.b2 import (
    contactListener,
    edgeShape,
    fixtureDef,
    polygonShape,
    revoluteJointDef,
)
import numpy as np
from PIL import Image, ImageDraw

from direct_evolution_together.cppn import CPPN
from direct_evolution_together.populora_engine import (
    _decode_f16_b64,
    get_or_create_base_model,
)

from .rng import mulberry32
from .substrate import get_substrate

# Official Farama Gymnasium LunarLander-v3 / LunarLanderContinuous-v3 Box2D constants
LUNARLANDER_FPS = 50
LUNARLANDER_DT = 1.0 / LUNARLANDER_FPS  # 0.02s (50 Hz)
LUNARLANDER_SCALE = 30.0
LUNARLANDER_VIEWPORT_W = 600
LUNARLANDER_VIEWPORT_H = 400
LUNARLANDER_WORLD_W = LUNARLANDER_VIEWPORT_W / LUNARLANDER_SCALE  # 20.0
LUNARLANDER_HALF_W = LUNARLANDER_WORLD_W * 0.5  # 10.0
LUNARLANDER_WORLD_H = LUNARLANDER_VIEWPORT_H / LUNARLANDER_SCALE  # 13.3333...

LUNARLANDER_GRAVITY = -10.0
LUNARLANDER_MAIN_ENGINE_POWER = 13.0
LUNARLANDER_SIDE_ENGINE_POWER = 0.6
LUNARLANDER_INITIAL_RANDOM = 1000.0

LUNARLANDER_POLY: tuple[tuple[int, int], ...] = (
    (-14, +17),
    (-17, 0),
    (-17, -10),
    (+17, -10),
    (+17, 0),
    (+14, +17),
)
LUNARLANDER_LEG_AWAY = 20
LUNARLANDER_LEG_DOWN = 18
LUNARLANDER_LEG_W = 2
LUNARLANDER_LEG_H = 8
LUNARLANDER_LEG_SPRING_TORQUE = 40

LUNARLANDER_SIDE_ENGINE_HEIGHT = 14
LUNARLANDER_SIDE_ENGINE_AWAY = 12
LUNARLANDER_MAIN_ENGINE_Y_LOCATION = 4

LUNARLANDER_CHUNKS = 11
LUNARLANDER_TERRAIN_CHUNKS = LUNARLANDER_CHUNKS - 1  # 10 segments across 11 vertices
LUNARLANDER_HELIPAD_Y = LUNARLANDER_WORLD_H / 4.0  # 3.3333...
LUNARLANDER_HELIPAD_X1 = (LUNARLANDER_WORLD_W / (LUNARLANDER_CHUNKS - 1)) * (LUNARLANDER_CHUNKS // 2 - 1)  # 8.0
LUNARLANDER_HELIPAD_X2 = (LUNARLANDER_WORLD_W / (LUNARLANDER_CHUNKS - 1)) * (LUNARLANDER_CHUNKS // 2 + 1)  # 12.0
LUNARLANDER_HELIPAD_HALF_W = (LUNARLANDER_HELIPAD_X2 - LUNARLANDER_HELIPAD_X1) * 0.5  # 2.0
LUNARLANDER_REST_COM_Y = LUNARLANDER_HELIPAD_Y + LUNARLANDER_LEG_DOWN / LUNARLANDER_SCALE  # 3.9333...

LUNARLANDER_MAX_STEPS = 1000
LUNARLANDER_SOLVER_VEL_ITERS = 6 * 30
LUNARLANDER_SOLVER_POS_ITERS = 2 * 30

# Observation / Action / PopuLoRA WeightOnlyMLP policy spec
LUNARLANDER_OBS_DIM = 8
LUNARLANDER_ACTION_DIM = 2
LUNARLANDER_HIDDEN_DIM = 24
LUNARLANDER_HEAD_DIM = 24
LUNARLANDER_LOW_RANK = 8
LUNARLANDER_LORA_TARGETS = ("l0", "l1", "head")

# Legacy HyperNEAT fallback dims (used only if a caller explicitly evaluates a CPPN genome)
LUNARLANDER_HIDDEN_DIMS = (12,)
LUNARLANDER_SUBSTRATE_DIMS = (LUNARLANDER_OBS_DIM, *LUNARLANDER_HIDDEN_DIMS, LUNARLANDER_ACTION_DIM)
LUNARLANDER_IN_COORDS = (-0.60, -0.10, -0.35, 0.10, 0.35, 0.60, -0.85, 0.85)
LUNARLANDER_OUT_COORDS = (-0.45, 0.45)
LUNARLANDER_LAYER_COORDS = (LUNARLANDER_IN_COORDS, None, LUNARLANDER_OUT_COORDS)

LUNARLANDER_OBS_LABELS = ("x", "y", "vx", "vy", "angle", "omega", "leg_l", "leg_r")
LUNARLANDER_ACTION_LABELS = ("main", "side")
LUNARLANDER_EVAL_SEEDS = (0, 1, 2)
LUNARLANDER_SEED_ROTATION_STRIDE = 3
LUNARLANDER_SEED_ROTATION_MOD = 60

LUNARLANDER_PRESOLVED_RETURN = 160.0
LUNARLANDER_SOLVED_RETURN = 200.0
LUNARLANDER_TIER_MID_RETURN = 80.0

LUNARLANDER_POLICY_SPEC: dict[str, object] = {
    "arch": "weight_only_mlp",
    "obs_dim": LUNARLANDER_OBS_DIM,
    "action_dim": LUNARLANDER_ACTION_DIM,
    "hidden_dim": LUNARLANDER_HIDDEN_DIM,
    "head_dim": LUNARLANDER_HEAD_DIM,
    "low_rank": LUNARLANDER_LOW_RANK,
    "lora_targets": list(LUNARLANDER_LORA_TARGETS),
    "action_dist": "squashed_gaussian",
    "action_bounds": [-1.0, 1.0],
    "oscillatory_init": False,
    "init_std_up": None,
    "adaptive_epsilon": True,
    "epsilon_init": 0.15,
    "sigma_granularity": "weight",
    "survive_frac": 0.50,
    "max_num_elites": 1,
    "eta_min": 1.0,
    "eta_max": 1.5,
    "weight_decay": 1e-3,
    "use_dns": True,
    "use_swarm": True,
    "use_champ_backup": False,
}

LUNARLANDER_ENV_PARAMS: dict[str, object] = {
    "gravity": LUNARLANDER_GRAVITY,
    "enable_wind": False,
    "wind_power": 15.0,
    "turbulence_power": 1.5,
}

LUNARLANDER_HYPERPARAM_SPECS: tuple[dict[str, object], ...] = (
    {
        "key": "gravity",
        "label": "Gravity (m/s²)",
        "type": "float",
        "default": LUNARLANDER_GRAVITY,
        "min": -11.9,
        "max": -1.6,
        "step": 0.5,
        "description": "Vertical gravity in Gymnasium LunarLander-v3 (-10.0 default; -1.62 real Moon).",
    },
    {
        "key": "enable_wind",
        "label": "Enable Wind",
        "type": "bool",
        "default": False,
        "description": "Enable Gymnasium LunarLander-v3 non-periodic horizontal wind & rotational turbulence.",
    },
    {
        "key": "wind_power",
        "label": "Wind Power",
        "type": "float",
        "default": 15.0,
        "min": 0.0,
        "max": 20.0,
        "step": 1.0,
        "description": "Maximum linear wind magnitude when wind is enabled (Gymnasium default 15.0).",
    },
    {
        "key": "turbulence_power",
        "label": "Turbulence Power",
        "type": "float",
        "default": 1.5,
        "min": 0.0,
        "max": 2.0,
        "step": 0.1,
        "description": "Maximum rotational torque turbulence when wind is enabled (Gymnasium default 1.5).",
    },
)


def lunarlander_eval_seeds_for_generation(generation: int = 0) -> tuple[int, int, int]:
    """Generation-rotating common-random-number seeds so policies generalize across terrains & impulses."""
    s0 = (max(0, int(generation)) * LUNARLANDER_SEED_ROTATION_STRIDE) % LUNARLANDER_SEED_ROTATION_MOD
    return (s0, s0 + 1, s0 + 2)


def _resolve_lunar_env_params(env_params: Mapping[str, Any] | None = None) -> tuple[float, bool, float, float]:
    p = env_params if isinstance(env_params, Mapping) else {}
    raw_g = float(p.get("gravity", LUNARLANDER_GRAVITY))
    gravity = max(-11.99, min(-0.1, raw_g)) if math.isfinite(raw_g) else LUNARLANDER_GRAVITY
    raw_wind = p.get("enable_wind", False)
    if isinstance(raw_wind, bool):
        enable_wind = raw_wind
    else:
        enable_wind = str(raw_wind).strip().lower() in ("1", "true", "yes", "on")
    raw_wp = float(p.get("wind_power", 15.0))
    wind_power = max(0.0, min(20.0, raw_wp)) if math.isfinite(raw_wp) else 15.0
    raw_tp = float(p.get("turbulence_power", 1.5))
    turbulence_power = max(0.0, min(2.0, raw_tp)) if math.isfinite(raw_tp) else 1.5
    return gravity, enable_wind, wind_power, turbulence_power


def lunarlander_obs_from_body(
    pos_x: float,
    pos_y: float,
    vel_x: float,
    vel_y: float,
    angle: float,
    ang_vel: float,
    leg0_contact: float,
    leg1_contact: float,
    helipad_y: float = LUNARLANDER_HELIPAD_Y,
) -> list[float]:
    """Exact Gymnasium LunarLander-v3 8D observation vector."""
    return [
        (pos_x - LUNARLANDER_HALF_W) / LUNARLANDER_HALF_W,
        (pos_y - (helipad_y + LUNARLANDER_LEG_DOWN / LUNARLANDER_SCALE)) / (LUNARLANDER_WORLD_H * 0.5),
        vel_x * LUNARLANDER_HALF_W / LUNARLANDER_FPS,
        vel_y * (LUNARLANDER_WORLD_H * 0.5) / LUNARLANDER_FPS,
        angle,
        20.0 * ang_vel / LUNARLANDER_FPS,
        1.0 if leg0_contact else 0.0,
        1.0 if leg1_contact else 0.0,
    ]


class _ContactDetector(contactListener):
    def __init__(self, env: "LunarLanderV3Sim") -> None:
        super().__init__()
        self.env = env

    def BeginContact(self, contact: Any) -> None:
        b_a = contact.fixtureA.body
        b_b = contact.fixtureB.body
        if self.env.lander == b_a or self.env.lander == b_b:
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] == b_a or self.env.legs[i] == b_b:
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact: Any) -> None:
        b_a = contact.fixtureA.body
        b_b = contact.fixtureB.body
        for i in range(2):
            if self.env.legs[i] == b_a or self.env.legs[i] == b_b:
                self.env.legs[i].ground_contact = False


class LunarLanderV3Sim:
    """Exact Box2D Gymnasium LunarLander-v3 (continuous) simulator with deterministic mulberry32 RNG."""

    def __init__(
        self,
        *,
        gravity: float = LUNARLANDER_GRAVITY,
        enable_wind: bool = False,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
    ) -> None:
        self.gravity = max(-11.99, min(-0.1, float(gravity)))
        self.enable_wind = bool(enable_wind)
        self.wind_power = max(0.0, min(20.0, float(wind_power)))
        self.turbulence_power = max(0.0, min(2.0, float(turbulence_power)))

        self.world: Box2D.b2World | None = None
        self.moon: Box2D.b2Body | None = None
        self.lander: Box2D.b2Body | None = None
        self.legs: list[Box2D.b2Body] = []
        self.game_over = False
        self.prev_shaping: float | None = None
        self.chunk_x: list[float] = []
        self.smooth_y: list[float] = []
        self.helipad_x1 = LUNARLANDER_HELIPAD_X1
        self.helipad_x2 = LUNARLANDER_HELIPAD_X2
        self.helipad_y = LUNARLANDER_HELIPAD_Y
        self.wind_idx = 0
        self.torque_idx = 0
        self.rng = mulberry32(0)

    def reset(self, seed: int = 0) -> list[float]:
        self.rng = mulberry32(int(seed))
        self.world = Box2D.b2World(gravity=(0.0, self.gravity))
        self.world.contactListener_keepref = _ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        w = LUNARLANDER_WORLD_W
        h = LUNARLANDER_WORLD_H
        chunks = LUNARLANDER_CHUNKS
        height = [self.rng() * (h * 0.5) for _ in range(chunks + 1)]
        chunk_x = [w / (chunks - 1) * i for i in range(chunks)]
        self.helipad_x1 = chunk_x[chunks // 2 - 1]
        self.helipad_x2 = chunk_x[chunks // 2 + 1]
        self.helipad_y = h / 4.0
        for idx in range(chunks // 2 - 2, chunks // 2 + 3):
            height[idx] = self.helipad_y
        smooth_y = [
            0.33 * (height[i - 1] + height[i] + height[i + 1])
            for i in range(chunks)
        ]
        self.chunk_x = chunk_x
        self.smooth_y = smooth_y

        self.moon = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(0.0, 0.0), (w, 0.0)])
        )
        for i in range(chunks - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)

        initial_x = w * 0.5
        initial_y = h
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=[(x / LUNARLANDER_SCALE, y / LUNARLANDER_SCALE) for x, y in LUNARLANDER_POLY]
                ),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,
                restitution=0.0,
            ),
        )
        self.lander.ApplyForceToCenter(
            (
                (self.rng() * 2.0 - 1.0) * LUNARLANDER_INITIAL_RANDOM,
                (self.rng() * 2.0 - 1.0) * LUNARLANDER_INITIAL_RANDOM,
            ),
            True,
        )

        if self.enable_wind:
            self.wind_idx = int(math.floor(self.rng() * 19998.0)) - 9999
            self.torque_idx = int(math.floor(self.rng() * 19998.0)) - 9999

        self.legs = []
        for i in (-1, +1):
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * LUNARLANDER_LEG_AWAY / LUNARLANDER_SCALE, initial_y),
                angle=i * 0.05,
                fixtures=fixtureDef(
                    shape=polygonShape(
                        box=(LUNARLANDER_LEG_W / LUNARLANDER_SCALE, LUNARLANDER_LEG_H / LUNARLANDER_SCALE)
                    ),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001,
                ),
            )
            leg.ground_contact = False
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0.0, 0.0),
                localAnchorB=(
                    i * LUNARLANDER_LEG_AWAY / LUNARLANDER_SCALE,
                    LUNARLANDER_LEG_DOWN / LUNARLANDER_SCALE,
                ),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LUNARLANDER_LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i,
            )
            if i == -1:
                rjd.lowerAngle = +0.9 - 0.5
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        obs, _, _, _, _ = self.step((0.0, 0.0))
        return obs

    def apply_nudge(self, nudge: str) -> None:
        if self.lander is None:
            return
        pos = self.lander.position
        if nudge == "left":
            self.lander.ApplyLinearImpulse((-14.0, 0.0), pos, True)
        elif nudge == "right":
            self.lander.ApplyLinearImpulse((+14.0, 0.0), pos, True)
        elif nudge == "up":
            self.lander.ApplyLinearImpulse((0.0, +16.0), pos, True)

    def step(self, action: Sequence[float]) -> tuple[list[float], float, bool, float, float]:
        assert self.lander is not None and self.world is not None

        if self.enable_wind and not (self.legs[0].ground_contact or self.legs[1].ground_contact):
            wind_mag = (
                math.tanh(
                    math.sin(0.02 * self.wind_idx)
                    + math.sin(math.pi * 0.01 * self.wind_idx)
                )
                * self.wind_power
            )
            self.wind_idx += 1
            self.lander.ApplyForceToCenter((wind_mag, 0.0), True)

            torque_mag = (
                math.tanh(
                    math.sin(0.02 * self.torque_idx)
                    + math.sin(math.pi * 0.01 * self.torque_idx)
                )
                * self.turbulence_power
            )
            self.torque_idx += 1
            self.lander.ApplyTorque(torque_mag, True)

        a0 = max(-1.0, min(1.0, float(action[0])))
        a1 = max(-1.0, min(1.0, float(action[1])))

        ang = float(self.lander.angle)
        tip = (math.sin(ang), math.cos(ang))
        side = (-tip[1], tip[0])

        disp0 = (self.rng() * 2.0 - 1.0) / LUNARLANDER_SCALE
        disp1 = (self.rng() * 2.0 - 1.0) / LUNARLANDER_SCALE

        m_power = 0.0
        if a0 > 0.0:
            m_power = (min(1.0, max(0.0, a0)) + 1.0) * 0.5
            ox = tip[0] * (LUNARLANDER_MAIN_ENGINE_Y_LOCATION / LUNARLANDER_SCALE + 2.0 * disp0) + side[0] * disp1
            oy = -tip[1] * (LUNARLANDER_MAIN_ENGINE_Y_LOCATION / LUNARLANDER_SCALE + 2.0 * disp0) - side[1] * disp1
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            self.lander.ApplyLinearImpulse(
                (-ox * LUNARLANDER_MAIN_ENGINE_POWER * m_power, -oy * LUNARLANDER_MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )

        s_power = 0.0
        s_signed = 0.0
        if abs(a1) > 0.5:
            direction = 1.0 if a1 > 0.0 else -1.0
            s_power = min(1.0, max(0.5, abs(a1)))
            s_signed = direction * s_power
            ox = tip[0] * disp0 + side[0] * (3.0 * disp1 + direction * LUNARLANDER_SIDE_ENGINE_AWAY / LUNARLANDER_SCALE)
            oy = -tip[1] * disp0 - side[1] * (3.0 * disp1 + direction * LUNARLANDER_SIDE_ENGINE_AWAY / LUNARLANDER_SCALE)
            impulse_pos = (
                self.lander.position[0] + ox - tip[0] * 17.0 / LUNARLANDER_SCALE,
                self.lander.position[1] + oy + tip[1] * LUNARLANDER_SIDE_ENGINE_HEIGHT / LUNARLANDER_SCALE,
            )
            self.lander.ApplyLinearImpulse(
                (-ox * LUNARLANDER_SIDE_ENGINE_POWER * s_power, -oy * LUNARLANDER_SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )

        self.world.Step(LUNARLANDER_DT, LUNARLANDER_SOLVER_VEL_ITERS, LUNARLANDER_SOLVER_POS_ITERS)

        pos = self.lander.position
        vel = self.lander.linearVelocity
        obs = lunarlander_obs_from_body(
            float(pos.x),
            float(pos.y),
            float(vel.x),
            float(vel.y),
            float(self.lander.angle),
            float(self.lander.angularVelocity),
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.helipad_y,
        )

        shaping = (
            -100.0 * math.hypot(obs[0], obs[1])
            - 100.0 * math.hypot(obs[2], obs[3])
            - 100.0 * abs(obs[4])
            + 10.0 * obs[6]
            + 10.0 * obs[7]
        )
        reward = 0.0 if self.prev_shaping is None else (shaping - self.prev_shaping)
        self.prev_shaping = shaping
        reward -= m_power * 0.30
        reward -= s_power * 0.03

        terminated = False
        if self.game_over or abs(obs[0]) >= 1.0:
            terminated = True
            reward = -100.0
        if not self.lander.awake:
            terminated = True
            reward = +100.0

        return obs, float(reward), terminated, m_power, s_signed

    def capture_frame(self, m_power: float, s_signed: float, cum_ret: float) -> list[float]:
        assert self.lander is not None
        pos = self.lander.position
        vel = self.lander.linearVelocity
        l0 = self.legs[0]
        l1 = self.legs[1]
        return [
            round(float(pos.x), 4),
            round(float(pos.y), 4),
            round(float(vel.x), 4),
            round(float(vel.y), 4),
            round(float(self.lander.angle), 4),
            round(float(self.lander.angularVelocity), 4),
            1.0 if l0.ground_contact else 0.0,
            1.0 if l1.ground_contact else 0.0,
            round(float(m_power), 3),
            round(float(s_signed), 3),
            round(float(cum_ret), 2),
            round(float(l0.position.x), 4),
            round(float(l0.position.y), 4),
            round(float(l0.angle), 4),
            round(float(l1.position.x), 4),
            round(float(l1.position.y), 4),
            round(float(l1.angle), 4),
        ]


def lunarlander_heuristic(obs: Sequence[float]) -> tuple[float, float]:
    """Official Gymnasium LunarLanderContinuous-v3 heuristic controller (scores ~260–295)."""
    angle_targ = max(-0.4, min(0.4, obs[0] * 0.5 + obs[2] * 1.0))
    hover_targ = 0.55 * abs(obs[0])
    angle_todo = (angle_targ - obs[4]) * 0.5 - obs[5] * 1.0
    hover_todo = (hover_targ - obs[1]) * 0.5 - obs[3] * 0.5
    if obs[6] or obs[7]:
        angle_todo = 0.0
        hover_todo = -obs[3] * 0.5
    return (
        max(-1.0, min(1.0, hover_todo * 20.0 - 1.0)),
        max(-1.0, min(1.0, -angle_todo * 20.0)),
    )


def _decode_base_tensor_np(entry: Mapping[str, Any]) -> np.ndarray:
    shape = tuple(int(d) for d in entry["shape"])
    raw = base64.b64decode(str(entry["b64"]), validate=True)
    dtype = "<f4" if entry.get("dtype") == "f32" else "<f2"
    return np.frombuffer(raw, dtype=dtype).astype(np.float32).reshape(shape).copy()


def _compile_populora_weight_only_mlp(
    genome: Mapping[str, Any],
    base_model_payload: Mapping[str, Any] | None = None,
) -> Callable[[Sequence[float]], tuple[float, float]]:
    """Compile a PopuLoRA WeightOnlyMLP (or standard MLP) genome into a fast NumPy forward function."""
    if base_model_payload is None:
        base_seed = int(genome.get("base_seed", 1))
        base_hash = str(genome.get("base_hash") or "") or None
        _, _, base_model_payload = get_or_create_base_model(
            LUNARLANDER_POLICY_SPEC, base_seed=base_seed, base_hash=base_hash
        )

    arch = str(base_model_payload.get("arch") or "weight_only_mlp")
    tensors_raw = base_model_payload["tensors"]
    low_rank = int(genome.get("low_rank", base_model_payload.get("low_rank", LUNARLANDER_LOW_RANK)))
    adapters = genome.get("adapters") if isinstance(genome.get("adapters"), Mapping) else {}

    def merge_layer(path: str, has_bias: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
        w = _decode_base_tensor_np(tensors_raw[f"{path}.weight"])  # [out, in]
        key = path.replace(".", "_")
        ad = adapters.get(key) if isinstance(adapters, Mapping) else None
        if isinstance(ad, Mapping) and ad.get("down") and ad.get("up"):
            in_dim = w.shape[1]
            out_dim = w.shape[0]
            down = _decode_f16_b64(str(ad["down"]), (in_dim, low_rank)).numpy()
            up = _decode_f16_b64(str(ad["up"]), (out_dim, low_rank)).numpy()
            w = w + (up @ down.T).astype(np.float32)
        b = _decode_base_tensor_np(tensors_raw[f"{path}.bias"]) if (has_bias and f"{path}.bias" in tensors_raw) else None
        return np.ascontiguousarray(w.T, dtype=np.float32), b

    if arch == "weight_only_mlp":
        w0, _ = merge_layer("l0", has_bias=False)
        w1, _ = merge_layer("l1", has_bias=False)
        w2, _ = merge_layer("head", has_bias=False)
        ones = np.ones((1,), dtype=np.float32)

        def fwd_wo(obs: Sequence[float]) -> tuple[float, float]:
            v0 = np.asarray((*obs, 1.0), dtype=np.float32)
            h0 = np.maximum(0.0, v0 @ w0)
            h1 = np.maximum(0.0, np.concatenate((h0, ones)) @ w1)
            out = np.tanh(np.concatenate((h1, ones)) @ w2)
            return float(out[0]), float(out[1])

        return fwd_wo

    w0, b0 = merge_layer("encoder.0", has_bias=True)
    w1, b1 = merge_layer("encoder.2", has_bias=True)
    w2, b2 = merge_layer("head", has_bias=True)
    b0_z = b0 if b0 is not None else np.zeros((w0.shape[1],), dtype=np.float32)
    b1_z = b1 if b1 is not None else np.zeros((w1.shape[1],), dtype=np.float32)
    b2_z = b2 if b2 is not None else np.zeros((w2.shape[1],), dtype=np.float32)

    def fwd_mlp(obs: Sequence[float]) -> tuple[float, float]:
        h0 = np.tanh(np.asarray(obs, dtype=np.float32) @ w0 + b0_z)
        h1 = np.tanh(h0 @ w1 + b1_z)
        out = np.tanh(h1 @ w2 + b2_z)
        return float(out[0]), float(out[1])

    return fwd_mlp


def _compile_policy(
    policy: Mapping[str, Any] | CPPN | Callable[[Sequence[float]], Sequence[float]],
    base_model_payload: Mapping[str, Any] | None = None,
) -> Callable[[Sequence[float]], tuple[float, float]]:
    if callable(policy) and not isinstance(policy, CPPN):
        def fwd_call(obs: Sequence[float]) -> tuple[float, float]:
            act = policy(obs)
            return float(act[0]), float(act[1])
        return fwd_call
    if isinstance(policy, Mapping) and policy.get("kind") == "populora":
        return _compile_populora_weight_only_mlp(policy, base_model_payload=base_model_payload)
    substrate = get_substrate(LUNARLANDER_SUBSTRATE_DIMS, LUNARLANDER_LAYER_COORDS)
    weights, biases = substrate.decode(policy)  # type: ignore[arg-type]

    def fwd_cppn(obs: Sequence[float]) -> tuple[float, float]:
        act = substrate.forward(weights, biases, obs)
        return float(act[0]), float(act[1])

    return fwd_cppn


def lunarlander_rollout(
    genome_or_policy: Mapping[str, Any] | CPPN | Callable[[Sequence[float]], Sequence[float]],
    *,
    seed: int = 0,
    max_steps: int = LUNARLANDER_MAX_STEPS,
    nudge: str | None = None,
    capture_frames: bool = True,
    capture_poses: bool | None = None,
    pose_stride: int = 1,
    env_params: Mapping[str, Any] | None = None,
    base_model_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one deterministic Gymnasium LunarLander-v3 Box2D episode."""
    if capture_poses is not None:
        capture_frames = bool(capture_poses)
    stride = max(1, int(pose_stride))
    gravity, enable_wind, wind_power, turbulence_power = _resolve_lunar_env_params(env_params)

    fwd = _compile_policy(genome_or_policy, base_model_payload=base_model_payload)
    sim = LunarLanderV3Sim(
        gravity=gravity,
        enable_wind=enable_wind,
        wind_power=wind_power,
        turbulence_power=turbulence_power,
    )
    obs = sim.reset(int(seed))

    total_ret = 0.0
    frames: list[list[float]] = []
    traj_sample: list[list[float]] = []
    if capture_frames:
        frames.append(sim.capture_frame(0.0, 0.0, 0.0))

    steps = 0
    outcome = "flying"
    nudge_step = min(22, max(1, int(max_steps) // 6))
    eff_max = max(1, int(max_steps))

    for step_idx in range(eff_max):
        if nudge and step_idx == nudge_step:
            sim.apply_nudge(nudge)
        a0, a1 = fwd(obs)
        obs, reward, terminated, m_power, s_signed = sim.step((a0, a1))
        total_ret += reward
        steps += 1

        if step_idx % 12 == 0 and len(traj_sample) < 32:
            traj_sample.append([round(float(v), 4) for v in obs])

        if capture_frames and (terminated or steps == eff_max or (steps % stride == 0)):
            frames.append(sim.capture_frame(m_power, s_signed, total_ret))

        if terminated:
            assert sim.lander is not None
            if not sim.lander.awake and not sim.game_over:
                on_pad = sim.helipad_x1 <= float(sim.lander.position.x) <= sim.helipad_x2
                outcome = "landed" if on_pad else "settled"
            else:
                outcome = "crashed"
            break

    assert sim.lander is not None
    if outcome == "flying":
        outcome = "timeout"

    pos = sim.lander.position
    on_pad = sim.helipad_x1 <= float(pos.x) <= sim.helipad_x2
    landed_clean = outcome in ("landed", "settled") and not sim.game_over and (not sim.lander.awake)

    return {
        "task": "lunarlander",
        "seed": int(seed),
        "steps": steps,
        "max_steps": eff_max,
        "return": round(float(total_ret), 2),
        "outcome": outcome,
        "landed": bool(landed_clean),
        "on_pad": bool(on_pad),
        "crashed": bool(outcome == "crashed"),
        "final_x": round(float(pos.x), 4),
        "final_y": round(float(pos.y), 4),
        "final_angle": round(float(sim.lander.angle), 4),
        "helipad_x1": round(float(sim.helipad_x1), 4),
        "helipad_x2": round(float(sim.helipad_x2), 4),
        "helipad_y": round(float(sim.helipad_y), 4),
        "terrain_x": [round(float(v), 4) for v in sim.chunk_x],
        "terrain_y": [round(float(v), 4) for v in sim.smooth_y],
        "heights": [round(float(v), 4) for v in sim.smooth_y],
        "pose_stride": stride,
        "frames": frames,
        "traj_sample": traj_sample,
        "nudge": nudge,
    }


def _signed(val: float, fmt: str = ".1f") -> str:
    return f"+{format(val, fmt)}" if val >= 0 else format(val, fmt)


def _summarize_lunarlander_rollouts(rollouts: Sequence[Mapping[str, Any]], seeds: Sequence[int]) -> dict[str, Any]:
    n = max(1, len(rollouts))
    returns = [float(r["return"]) for r in rollouts]
    mean_ret = round(sum(returns) / n, 1)
    min_ret = round(min(returns), 1)
    max_ret = round(max(returns), 1)
    mean_steps = round(sum(int(r["steps"]) for r in rollouts) / n, 1)
    landed_count = sum(1 for r in rollouts if r.get("landed"))
    on_pad_count = sum(1 for r in rollouts if r.get("landed") and r.get("on_pad"))
    crashed_count = sum(1 for r in rollouts if r.get("crashed"))
    solved = mean_ret >= LUNARLANDER_SOLVED_RETURN and landed_count >= max(1, (n * 2) // 3)
    tier = "solved" if solved else ("mid" if mean_ret >= LUNARLANDER_TIER_MID_RETURN or landed_count >= 1 else "")
    disp = _signed(mean_ret, ".0f")
    return {
        "task": "lunarlander",
        "seeds": [int(s) for s in seeds],
        "return": mean_ret,
        "min_return": min_ret,
        "max_return": max_ret,
        "mean_steps": mean_steps,
        "landed_count": landed_count,
        "on_pad_count": on_pad_count,
        "crashed_count": crashed_count,
        "solved": solved,
        "primary_value": mean_ret,
        "primary_display": f"{disp} ({landed_count}/{n} landed)",
        "badge_text": f"{disp} · {landed_count}/{n}",
        "tier": tier,
        "secondary_text": (
            f"mean {mean_ret:+.1f} (min {min_ret:+.1f}) · "
            f"{landed_count}/{n} landed ({on_pad_count} on pad) · {mean_steps:.0f} steps"
        ),
    }


def _render_lunarlander_frame(
    rollout: Mapping[str, Any],
    frame_idx: int,
    max_steps: int,
    width: int = 480,
    height: int = 320,
) -> np.ndarray:
    """Render one crisp Gymnasium LunarLander-v3 frame with articulated legs and exhaust plumes."""
    frames = rollout.get("frames") or []
    idx = min(max(0, int(frame_idx)), max(0, len(frames) - 1))
    fr = frames[idx] if frames else [
        LUNARLANDER_HALF_W,
        LUNARLANDER_WORLD_H,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        LUNARLANDER_HALF_W - LUNARLANDER_LEG_AWAY / LUNARLANDER_SCALE,
        LUNARLANDER_WORLD_H,
        -0.05,
        LUNARLANDER_HALF_W + LUNARLANDER_LEG_AWAY / LUNARLANDER_SCALE,
        LUNARLANDER_WORLD_H,
        +0.05,
    ]
    x, y, _vx, vy, angle, _omega, leg0_c, leg1_c, m_power, s_signed, cum_ret = [float(v) for v in fr[:11]]
    has_leg_poses = len(fr) >= 17

    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Subtle space backdrop
    star_rng = mulberry32(42)
    for _ in range(28):
        sx = int(star_rng() * width)
        sy = int(star_rng() * height * 0.62)
        br = int(75 + star_rng() * 140)
        draw.point((sx, sy), fill=(br, br, min(255, br + 20)))

    def to_px(wx: float, wy: float) -> tuple[float, float]:
        px = (wx / LUNARLANDER_WORLD_W) * width
        py = (1.0 - wy / LUNARLANDER_WORLD_H) * height
        return px, py

    terrain_x = rollout.get("terrain_x") or [
        LUNARLANDER_WORLD_W / (LUNARLANDER_CHUNKS - 1) * i for i in range(LUNARLANDER_CHUNKS)
    ]
    terrain_y = rollout.get("terrain_y") or rollout.get("heights") or [LUNARLANDER_HELIPAD_Y] * LUNARLANDER_CHUNKS
    helipad_x1 = float(rollout.get("helipad_x1", LUNARLANDER_HELIPAD_X1))
    helipad_x2 = float(rollout.get("helipad_x2", LUNARLANDER_HELIPAD_X2))
    helipad_y = float(rollout.get("helipad_y", LUNARLANDER_HELIPAD_Y))

    # Crisp white/lunar terrain matching Gymnasium LunarLander-v3 aesthetics
    poly_pts = [to_px(float(tx), float(ty)) for tx, ty in zip(terrain_x, terrain_y)]
    poly_pts.extend([(float(width), float(height)), (0.0, float(height))])
    draw.polygon(poly_pts, fill=(232, 234, 240))
    ridge_pts = [to_px(float(tx), float(ty)) for tx, ty in zip(terrain_x, terrain_y)]
    if len(ridge_pts) >= 2:
        draw.line(ridge_pts, fill=(148, 156, 174), width=max(1, width // 260))

    # Iconic Gymnasium yellow flags on white poles at helipad_x1 & helipad_x2
    flag_h_world = 50.0 / LUNARLANDER_SCALE
    for fx_w in (helipad_x1, helipad_x2):
        bx, by = to_px(fx_w, helipad_y)
        tx, ty = to_px(fx_w, helipad_y + flag_h_world)
        draw.line([(bx, by), (tx, ty)], fill=(255, 255, 255), width=max(1, width // 360))
        fw = (25.0 / LUNARLANDER_VIEWPORT_W) * width
        fh = (10.0 / LUNARLANDER_VIEWPORT_H) * height
        draw.polygon([(tx, ty), (tx, ty + fh), (tx + fw, ty + fh * 0.5)], fill=(204, 204, 0))

    # Subtle pad line accent
    p1x, p1y = to_px(helipad_x1, helipad_y)
    p2x, p2y = to_px(helipad_x2, helipad_y)
    draw.line([(p1x, p1y), (p2x, p2y)], fill=(52, 211, 153), width=max(2, width // 200))

    # Trajectory trail
    if idx > 1:
        stride_t = max(1, idx // 90)
        trail = [to_px(float(frames[k][0]), float(frames[k][1])) for k in range(0, idx + 1, stride_t)]
        if len(trail) >= 2:
            draw.line(trail, fill=(96, 165, 250), width=max(1, width // 360))

    def rot_world(cx: float, cy: float, ang: float, lx: float, ly: float) -> tuple[float, float]:
        ca = math.cos(ang)
        sa = math.sin(ang)
        return to_px(cx + lx * ca - ly * sa, cy + lx * sa + ly * ca)

    # Exhaust plumes (main + side thrusters)
    if m_power > 0.01:
        plume_len = (0.38 + 0.95 * m_power) * (0.88 + 0.24 * math.sin(idx * 1.9))
        m_y0 = -10.0 / LUNARLANDER_SCALE
        outer = [
            rot_world(x, y, angle, -0.22, m_y0),
            rot_world(x, y, angle, +0.22, m_y0),
            rot_world(x, y, angle, 0.0, m_y0 - plume_len),
        ]
        inner = [
            rot_world(x, y, angle, -0.10, m_y0),
            rot_world(x, y, angle, +0.10, m_y0),
            rot_world(x, y, angle, 0.0, m_y0 - plume_len * 0.58),
        ]
        draw.polygon(outer, fill=(255, 120, 45))
        draw.polygon(inner, fill=(255, 236, 145))

    if abs(s_signed) > 0.01:
        s_dir = 1.0 if s_signed > 0.0 else -1.0
        s_len = 0.28 + 0.48 * abs(s_signed)
        sy = LUNARLANDER_SIDE_ENGINE_HEIGHT / LUNARLANDER_SCALE
        sx = -s_dir * (LUNARLANDER_SIDE_ENGINE_AWAY / LUNARLANDER_SCALE)
        # Side engine shoots particles outward on the firing side
        s_poly = [
            rot_world(x, y, angle, sx, sy - 0.07),
            rot_world(x, y, angle, sx, sy + 0.07),
            rot_world(x, y, angle, sx - s_dir * s_len, sy),
        ]
        draw.polygon(s_poly, fill=(255, 165, 70))

    # Articulated landing legs (drawn at exact Box2D poses!)
    lw = LUNARLANDER_LEG_W / LUNARLANDER_SCALE
    lh = LUNARLANDER_LEG_H / LUNARLANDER_SCALE
    leg_box = ((-lw, -lh), (lw, -lh), (lw, lh), (-lw, lh))
    for li in range(2):
        contact = leg0_c > 0.5 if li == 0 else leg1_c > 0.5
        if has_leg_poses:
            lx_w = float(fr[11 + li * 3])
            ly_w = float(fr[12 + li * 3])
            lang = float(fr[13 + li * 3])
        else:
            side_s = -1.0 if li == 0 else 1.0
            ca = math.cos(angle)
            sa = math.sin(angle)
            ox = side_s * (LUNARLANDER_LEG_AWAY / LUNARLANDER_SCALE) * 0.72
            oy = -(LUNARLANDER_LEG_DOWN / LUNARLANDER_SCALE) * 0.75
            lx_w = x + ox * ca - oy * sa
            ly_w = y + ox * sa + oy * ca
            lang = angle - side_s * 0.45
        l_pts = [rot_world(lx_w, ly_w, lang, vx_b, vy_b) for vx_b, vy_b in leg_box]
        fill_c = (52, 211, 153) if contact else (128, 102, 230)
        out_c = (16, 185, 129) if contact else (77, 77, 128)
        draw.polygon(l_pts, fill=fill_c, outline=out_c)

    # Iconic Gymnasium purple-indigo polygonal lander body (`LANDER_POLY`)
    is_crashed_end = rollout.get("outcome") == "crashed" and idx >= len(frames) - 1
    hull_pts = [
        rot_world(x, y, angle, vx_p / LUNARLANDER_SCALE, vy_p / LUNARLANDER_SCALE)
        for vx_p, vy_p in LUNARLANDER_POLY
    ]
    hull_fill = (248, 113, 113) if is_crashed_end else (128, 102, 230)
    hull_edge = (185, 28, 28) if is_crashed_end else (77, 77, 128)
    draw.polygon(hull_pts, fill=hull_fill, outline=hull_edge)

    # Compact HUD
    step_num = min(int(rollout.get("steps", 0)), idx * int(rollout.get("pose_stride", 1)))
    hud = f"LunarLander-v3  step {step_num}/{max_steps}  ret {cum_ret:+.1f}  vy {vy:+.1f}"
    draw.text((10, 8), hud, fill=(226, 232, 240))
    return np.asarray(img, dtype=np.uint8)


def render_lunarlander_ascii(rollout: Mapping[str, Any], *, cols: int = 64, rows: int = 20) -> str:
    """Render a compact ASCII trajectory plot of a LunarLander-v3 rollout for terminal inspection."""
    grid = [[" " for _ in range(cols)] for _ in range(rows)]
    tx = rollout.get("terrain_x") or [LUNARLANDER_WORLD_W / (LUNARLANDER_CHUNKS - 1) * i for i in range(LUNARLANDER_CHUNKS)]
    ty = rollout.get("terrain_y") or [LUNARLANDER_HELIPAD_Y] * LUNARLANDER_CHUNKS
    hx1 = float(rollout.get("helipad_x1", LUNARLANDER_HELIPAD_X1))
    hx2 = float(rollout.get("helipad_x2", LUNARLANDER_HELIPAD_X2))
    hy = float(rollout.get("helipad_y", LUNARLANDER_HELIPAD_Y))

    def to_cell(wx: float, wy: float) -> tuple[int, int]:
        c = max(0, min(cols - 1, int(round((wx / LUNARLANDER_WORLD_W) * (cols - 1)))))
        r = max(0, min(rows - 1, int(round((1.0 - wy / LUNARLANDER_WORLD_H) * (rows - 1)))))
        return r, c

    # Fill terrain silhouette
    for c in range(cols):
        wx = (c / max(1, cols - 1)) * LUNARLANDER_WORLD_W
        # Linear interpolate terrain height
        u = (wx / LUNARLANDER_WORLD_W) * (len(tx) - 1)
        idx = max(0, min(len(tx) - 2, int(math.floor(u))))
        frac = u - idx
        gy = float(ty[idx]) * (1.0 - frac) + float(ty[idx + 1]) * frac
        gr, _ = to_cell(wx, gy)
        on_pad = hx1 <= wx <= hx2
        grid[gr][c] = "=" if on_pad else "^"
        for rr in range(gr + 1, rows):
            grid[rr][c] = "#"

    # Helipad flags
    for fx in (hx1, hx2):
        fr_pad, fc_pad = to_cell(fx, hy)
        if fr_pad - 1 >= 0:
            grid[fr_pad - 1][fc_pad] = "|"
        if fr_pad - 2 >= 0:
            grid[fr_pad - 2][fc_pad] = "P"

    # Flight trajectory
    frames = rollout.get("frames") or []
    n_f = len(frames)
    for i, fr in enumerate(frames):
        r, c = to_cell(float(fr[0]), float(fr[1]))
        if i == 0:
            grid[r][c] = "S"
        elif i == n_f - 1:
            grid[r][c] = "*" if rollout.get("landed") else "X"
        else:
            if grid[r][c] in (" ", "|"):
                grid[r][c] = "o" if float(fr[8]) > 0.01 else "."

    border = "+" + "-" * cols + "+"
    header = (
        f"LunarLander-v3 | seed={rollout.get('seed')} | steps={rollout.get('steps')} "
        f"| return={float(rollout.get('return', 0.0)):+.2f} | outcome={rollout.get('outcome')} "
        f"| end=({float(rollout.get('final_x', 0.0)):.2f}, {float(rollout.get('final_y', 0.0)):.2f})"
    )
    lines = [header, border]
    for row in grid:
        lines.append("|" + "".join(row) + "|")
    lines.append(border)
    lines.append("Legend: S=start  o=main thrust  .=coast  *=landed  X=crash  P|===|P=helipad")
    return "\n".join(lines)


class LunarLanderAdapter:
    """Server-side ControlTaskAdapter for faithful Gymnasium LunarLander-v3 (Box2D)."""

    task_id = "lunarlander"

    def rollout(
        self,
        genome_or_net: dict | CPPN,
        *,
        seed: int = 0,
        max_steps: int = LUNARLANDER_MAX_STEPS,
        nudge: str | None = None,
        env_params: Mapping[str, Any] | None = None,
    ) -> dict[str, object]:
        return lunarlander_rollout(
            genome_or_net,
            seed=seed,
            max_steps=max_steps,
            nudge=nudge,
            capture_frames=True,
            env_params=env_params,
        )

    def evaluate(
        self,
        genome_or_net: dict | CPPN,
        *,
        seeds: Sequence[int] | None = None,
        env_params: Mapping[str, Any] | None = None,
    ) -> dict[str, object]:
        eval_seeds = tuple(int(s) for s in (seeds if seeds is not None else LUNARLANDER_EVAL_SEEDS))
        rollouts = [
            lunarlander_rollout(
                genome_or_net,
                seed=sd,
                max_steps=LUNARLANDER_MAX_STEPS,
                capture_frames=(i == 0),
                env_params=env_params,
            )
            for i, sd in enumerate(eval_seeds)
        ]
        return _summarize_lunarlander_rollouts(rollouts, eval_seeds)

    def is_gen0_presolved(self, eval_result: dict[str, object]) -> bool:
        return (
            float(eval_result.get("return", -999.0)) >= LUNARLANDER_PRESOLVED_RETURN
            or bool(eval_result.get("solved"))
        )

    def render_frame(
        self,
        rollout: dict[str, object],
        frame_idx: int,
        max_steps: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        return _render_lunarlander_frame(rollout, frame_idx, max_steps, width=width, height=height)
