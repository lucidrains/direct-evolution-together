"""Deployment policy: per-environment defaults plus file/env overrides.

Per-task numbers (population, quorum, elite fraction, execution policy) are declared
exactly once in ``DEFAULT_ENV_POLICIES``. Task mechanics and labels live in
``direct_evolution_together.environments``; runtime lookups merge the two.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import threading

_VALID_POLICIES = {"client_locked", "client_preferred", "server_only", "external_worker"}
_VALID_AGGS = {"median", "weighted_mean"}
_IMAGE_LOBBY_GOALS = ("Butterfly", "Galaxy", "Creature")
LARGE_POP_MAX_SIZE = 1024
SYSTEM_MAX_POPULATION_SIZE = 10_000


@dataclass(frozen=True)
class TrustPolicyConfig:
    enabled: bool = True
    quorum: int = 2
    shard_quorum: int = 1
    tolerance_abs: float = 0.35
    tolerance_rel: float = 0.12
    initial_reputation: float = 0.80
    agree_reward: float = 0.08
    mismatch_penalty: float = 0.35
    min_trusted_reputation: float = 0.35
    aggregation: str = "median"
    spot_check_rate: float = 0.15
    shard_shuffle: bool = True
    max_attestations: int = 5

    def to_client_dict(self) -> dict[str, object]:
        return {
            "enabled": bool(self.enabled),
            "quorum": int(self.quorum),
            "shard_quorum": int(self.shard_quorum),
            "spot_check_rate": float(self.spot_check_rate),
            "shard_shuffle": bool(self.shard_shuffle),
            "aggregation": self.aggregation,
        }


@dataclass(frozen=True)
class EnvPolicyConfig:
    task: str
    enabled: bool = True
    lobby_name: str = ""
    lobby_default: bool = False
    lobby_order: int = 99
    execution_policy: str = "client_preferred"  # client_locked | client_preferred | server_only | external_worker
    crowd_contribute: bool = False
    sim_phase_enabled: bool = False
    default_auto_evolve: bool = False
    allow_server_video: bool = True
    min_pop_size: int = 3
    default_pop_size: int = 30
    max_pop_size: int = 128
    default_min_votes: int = 20
    default_elite_frac: float | None = None
    default_brood_size: int = 1
    crowd_shard_threshold: int = 16
    crowd_shard_size: int = 4
    env_hyperparams: dict[str, object] = field(default_factory=dict)

    @property
    def sim_modes(self) -> tuple[str, ...]:
        return {
            "client_locked": ("client",),
            "client_preferred": ("client", "server"),
            "server_only": ("server",),
            "external_worker": ("external",),
        }.get(self.execution_policy, ("external",))

    @property
    def default_sim_mode(self) -> str:
        if self.execution_policy == "server_only":
            return "server"
        if self.execution_policy == "external_worker":
            return "external"
        return "client"

    @property
    def client_sim(self) -> bool:
        return self.execution_policy in ("client_locked", "client_preferred")

    @property
    def can_shard_population(self) -> bool:
        return (self.crowd_contribute and self.sim_phase_enabled) or self.execution_policy == "external_worker"

    @property
    def effective_max_pop_size(self) -> int:
        if self.task == "image" or self.can_shard_population:
            return max(self.min_pop_size, self.max_pop_size)
        return max(self.min_pop_size, min(self.max_pop_size, self.crowd_shard_threshold * 2))

    def constrain_room_scale(
        self,
        pop_size: int,
        min_votes: int,
        max_votes_per_user: int,
        *,
        is_control: bool = False,
    ) -> dict[str, int | bool]:
        """Enforce centralized population and voting constraints before scaling a room."""
        eff_pop = max(self.min_pop_size, min(self.effective_max_pop_size, int(pop_size)))
        eff_max_votes = max(1, min(256, int(max_votes_per_user)))
        eff_min_votes = max(1, min(1024, int(min_votes)))
        if is_control:
            eff_min_votes = min(eff_min_votes, eff_max_votes)
        sharded = bool(self.sim_phase_enabled and eff_pop > self.crowd_shard_threshold)
        return {
            "population_size": eff_pop,
            "min_votes": eff_min_votes,
            "max_votes_per_user": eff_max_votes,
            "sharded": sharded,
            "crowd_shard_threshold": self.crowd_shard_threshold,
            "crowd_shard_size": self.crowd_shard_size,
        }


@dataclass(frozen=True)
class CentralConfig:
    large_pop: bool = False
    trust: TrustPolicyConfig = field(default_factory=TrustPolicyConfig)
    environments: dict[str, EnvPolicyConfig] = field(default_factory=dict)
    mujoco_wasm_url: str = ""

    def get_env(self, task_id: str) -> EnvPolicyConfig:
        clean = str(task_id or "").strip().lower()
        return self.environments.get(clean) or self.environments["image"]

    def get_lobby_goal_specs(self) -> tuple[tuple[str, str, int | None], ...]:
        items = sorted(
            (c for c in self.environments.values() if c.enabled and c.lobby_default),
            key=lambda c: c.lobby_order,
        )
        specs: list[tuple[str, str, int | None]] = []
        for cfg in items:
            if cfg.task == "image":
                for img_name in _IMAGE_LOBBY_GOALS:
                    specs.append((img_name, "image", None))
            else:
                specs.append((cfg.lobby_name or cfg.task.title(), cfg.task, cfg.default_pop_size))
        return tuple(specs)

    def to_public_dict(self) -> dict[str, object]:
        return {
            "large_pop": self.large_pop,
            "trust": asdict(self.trust),
            "environments": {
                k: {**asdict(v), "sim_modes": list(v.sim_modes), "default_sim_mode": v.default_sim_mode}
                for k, v in self.environments.items()
            },
            "mujoco_wasm_url": self.mujoco_wasm_url,
        }


# Single source of truth for per-environment policy defaults (with `DET_LARGE_POP=0`).
# `DET_LARGE_POP=1` (`--large-pop`) layers `_LARGE_POP_PRESET` on top (250 Microduck gait + 1,024 room cap).
DEFAULT_ENV_POLICIES: dict[str, EnvPolicyConfig] = {
    "image": EnvPolicyConfig(
        task="image",
        lobby_name="Butterfly",
        lobby_default=True,
        lobby_order=0,
        execution_policy="client_locked",
        crowd_contribute=False,
        allow_server_video=False,
        default_pop_size=30,
        max_pop_size=64,
        default_min_votes=0,  # 0 = let the room derive quorum from live votes
        default_elite_frac=None,
    ),
    "lunarlander": EnvPolicyConfig(
        task="lunarlander",
        lobby_name="Lunar Lander",
        lobby_default=True,
        lobby_order=1,
        execution_policy="client_preferred",
        crowd_contribute=True,
        sim_phase_enabled=True,
        default_auto_evolve=False,
        allow_server_video=True,
        default_pop_size=16,
        max_pop_size=128,
        default_min_votes=4,
        default_elite_frac=0.25,
        crowd_shard_threshold=64,
        crowd_shard_size=8,
    ),
    "microduck": EnvPolicyConfig(
        task="microduck",
        lobby_name="Microduck",
        lobby_default=True,
        lobby_order=2,
        execution_policy="client_locked",
        crowd_contribute=True,
        sim_phase_enabled=True,
        default_auto_evolve=False,
        allow_server_video=False,
        default_pop_size=8,
        max_pop_size=128,
        default_min_votes=4,
        default_elite_frac=0.25,
        crowd_shard_threshold=16,
    ),
    "unitree_go2": EnvPolicyConfig(
        task="unitree_go2",
        lobby_name="Unitree Go2",
        lobby_default=True,
        lobby_order=3,
        execution_policy="client_locked",
        crowd_contribute=True,
        sim_phase_enabled=True,
        default_auto_evolve=False,
        allow_server_video=False,
        default_pop_size=8,
        max_pop_size=128,
        default_min_votes=4,
        default_elite_frac=0.25,
        crowd_shard_threshold=16,
    ),
    "so101": EnvPolicyConfig(
        task="so101",
        lobby_name="SO-102 Arm",
        lobby_default=True,
        lobby_order=4,
        execution_policy="client_locked",
        crowd_contribute=True,
        sim_phase_enabled=True,
        default_auto_evolve=False,
        allow_server_video=False,
        default_pop_size=8,
        max_pop_size=128,
        default_min_votes=4,
        default_elite_frac=0.25,
        crowd_shard_threshold=16,
        env_hyperparams={"arm_model": "so102"},
    ),
    "halfcheetah": EnvPolicyConfig(
        task="halfcheetah",
        lobby_name="HalfCheetah",
        lobby_default=True,
        lobby_order=5,
        execution_policy="client_locked",
        crowd_contribute=True,
        allow_server_video=True,
        default_pop_size=16,
        max_pop_size=24,
        default_min_votes=8,
        default_elite_frac=0.05,
    ),
    "cartpole": EnvPolicyConfig(
        task="cartpole",
        lobby_name="CartPole",
        lobby_default=True,
        lobby_order=6,
        execution_policy="client_preferred",
        crowd_contribute=False,
        allow_server_video=True,
        default_pop_size=30,
        max_pop_size=32,
        default_min_votes=20,
        default_elite_frac=0.0,
    ),
    "external": EnvPolicyConfig(
        task="external",
        lobby_name="External",
        lobby_default=False,
        lobby_order=90,
        execution_policy="external_worker",
        crowd_contribute=False,
        allow_server_video=False,
        default_pop_size=30,
        max_pop_size=64,
        default_min_votes=20,
        default_elite_frac=0.10,
    ),
}

_LARGE_POP_PRESET: dict[str, dict[str, object]] = {
    "microduck": {"default_pop_size": 250, "max_pop_size": LARGE_POP_MAX_SIZE, "crowd_shard_size": 8},
    "unitree_go2": {"max_pop_size": LARGE_POP_MAX_SIZE},
    "so101": {"max_pop_size": LARGE_POP_MAX_SIZE},
    "image": {"max_pop_size": LARGE_POP_MAX_SIZE},
    "external": {"max_pop_size": LARGE_POP_MAX_SIZE},
}

# (field, env var template) -- `{task}` expands to the upper-cased task id.
_ENV_VAR_OVERRIDES: tuple[tuple[str, str], ...] = (
    ("enabled", "DET_TASK_ENABLED_{task}"),
    ("crowd_contribute", "DET_CROWD_CONTRIBUTE_{task}"),
    ("sim_phase_enabled", "DET_SIM_PHASE_{task}"),
    ("default_auto_evolve", "DET_AUTO_EVOLVE_{task}"),
    ("allow_server_video", "DET_ALLOW_VIDEO_{task}"),
    ("execution_policy", "DET_EXECUTION_POLICY_{task}"),
    ("default_pop_size", "DET_POP_SIZE_{task}"),
    ("min_pop_size", "DET_MIN_POP_SIZE_{task}"),
    ("max_pop_size", "DET_MAX_POP_SIZE_{task}"),
    ("default_min_votes", "DET_MIN_VOTES_{task}"),
    ("default_elite_frac", "DET_ELITE_FRAC_{task}"),
    ("default_brood_size", "DET_BROOD_SIZE_{task}"),
    ("crowd_shard_threshold", "DET_CROWD_SHARD_THRESHOLD_{task}"),
    ("crowd_shard_size", "DET_CROWD_SHARD_SIZE_{task}"),
    ("env_hyperparams", "DET_HYPERPARAMS_{task}"),
)

_ENV_POLICY_FIELDS = frozenset(EnvPolicyConfig.__dataclass_fields__) - {"task"}

# (field, env_name, lo, hi)
_TRUST_FIELDS: tuple[tuple[str, str, float | None, float | None], ...] = (
    ("enabled", "DET_CROWD_TRUST_ENABLED", None, None),
    ("quorum", "DET_CROWD_QUORUM", 1, 10),
    ("shard_quorum", "DET_CROWD_SHARD_QUORUM", 1, 10),
    ("tolerance_abs", "DET_CROWD_TOLERANCE_ABS", 0.01, 100.0),
    ("tolerance_rel", "DET_CROWD_TOLERANCE_REL", 0.0, 1.0),
    ("initial_reputation", "DET_CROWD_INIT_REPUTATION", 0.1, 1.0),
    ("agree_reward", "DET_CROWD_AGREE_REWARD", 0.0, 0.5),
    ("mismatch_penalty", "DET_CROWD_MISMATCH_PENALTY", 0.0, 1.0),
    ("min_trusted_reputation", "DET_CROWD_MIN_REPUTATION", 0.0, 0.99),
    ("aggregation", "DET_CROWD_AGGREGATION", None, None),
    ("spot_check_rate", "DET_CROWD_SPOT_CHECK_RATE", 0.0, 1.0),
    ("shard_shuffle", "DET_CROWD_SHARD_SHUFFLE", None, None),
    ("max_attestations", "DET_CROWD_MAX_ATTESTATIONS", 1, 20),
)


def _coerce_field(raw: object, default: object, lo: float | None, hi: float | None) -> object:
    if isinstance(default, bool):
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        try:
            v = int(str(raw).strip()) if isinstance(raw, str) else int(raw)  # type: ignore[arg-type]
            return max(int(lo), min(int(hi), v)) if lo is not None else v
        except (TypeError, ValueError):
            return default
    if isinstance(default, float):
        try:
            v = float(raw)  # type: ignore[arg-type]
            return max(float(lo), min(float(hi), v)) if lo is not None else v
        except (TypeError, ValueError):
            return default
    s = str(raw).strip().lower()
    return s if s in _VALID_AGGS else default


def _coerce_bool(raw: object, default: bool) -> bool:
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return default


def _coerce_int(raw: object, default: int, lo: int, hi: int) -> int:
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, value))


def _coerce_env_value(field_name: str, raw: object, default: object) -> object:
    """Validate one per-environment override, falling back to `default` on bad input."""
    if raw is None:
        return default
    if field_name in ("enabled", "crowd_contribute", "sim_phase_enabled", "default_auto_evolve", "allow_server_video", "lobby_default"):
        return _coerce_bool(raw, bool(default))
    if field_name == "execution_policy":
        policy = str(raw).strip().lower()
        return policy if policy in _VALID_POLICIES else default
    if field_name in ("min_pop_size", "default_pop_size", "max_pop_size"):
        return _coerce_int(raw, int(default), 3, SYSTEM_MAX_POPULATION_SIZE)
    if field_name == "crowd_shard_threshold":
        return _coerce_int(raw, int(default), 3, 1024)
    if field_name == "crowd_shard_size":
        return _coerce_int(raw, int(default), 1, 256)
    if field_name == "env_hyperparams":
        if isinstance(raw, dict):
            return {str(k): v for k, v in raw.items()}
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return {str(k): v for k, v in parsed.items()}
            except ValueError:
                pass
        return default
    if field_name == "default_min_votes":
        return _coerce_int(raw, int(default), 0, 1024)
    if field_name == "default_brood_size":
        return _coerce_int(raw, int(default), 1, 8)
    if field_name == "default_elite_frac":
        text = str(raw).strip().lower()
        if text in ("", "none", "null"):
            return None
        try:
            value = float(text)
        except ValueError:
            return default
        return max(0.0, min(0.5, value))
    if field_name == "lobby_order":
        return _coerce_int(raw, int(default), 0, 999)
    if field_name == "lobby_name":
        return str(raw).strip()[:64] or default
    return default


def build_config() -> CentralConfig:
    t_dict = asdict(TrustPolicyConfig())
    envs = dict(DEFAULT_ENV_POLICIES)
    wasm_url = os.getenv("DET_MUJOCO_WASM_URL", "").strip()
    large_pop = False
    loaded: dict[str, object] = {}

    cfg_raw = os.getenv("DET_CONFIG_FILE", "").strip()
    if cfg_raw and Path(cfg_raw).is_file():
        try:
            raw_doc = json.loads(Path(cfg_raw).read_text(encoding="utf-8"))
            if isinstance(raw_doc, dict):
                loaded = raw_doc
        except Exception:
            loaded = {}

    if "large_pop" in loaded:
        large_pop = _coerce_bool(loaded["large_pop"], False)
    if (raw_lp := os.getenv("DET_LARGE_POP")) is not None and raw_lp.strip():
        large_pop = _coerce_bool(raw_lp, large_pop)

    if large_pop:
        for tid, preset in _LARGE_POP_PRESET.items():
            if base := envs.get(tid):
                envs[tid] = EnvPolicyConfig(**{**asdict(base), **preset})

    if loaded:
        if isinstance(loaded.get("mujoco_wasm_url"), str):
            wasm_url = loaded["mujoco_wasm_url"].strip()
        if isinstance(raw_t := loaded.get("trust"), dict):
            for fname, _, lo, hi in _TRUST_FIELDS:
                if fname in raw_t:
                    t_dict[fname] = _coerce_field(raw_t[fname], t_dict[fname], lo, hi)
        if isinstance(raw_e := loaded.get("environments"), dict):
            for tid, overrides in raw_e.items():
                if (base := envs.get(tid)) is None or not isinstance(overrides, dict):
                    continue
                values = asdict(base)
                for field_name, raw in overrides.items():
                    if field_name in _ENV_POLICY_FIELDS:
                        values[field_name] = _coerce_env_value(field_name, raw, values[field_name])
                values["max_pop_size"] = max(int(values["default_pop_size"]), int(values["max_pop_size"]))
                envs[tid] = EnvPolicyConfig(**values)

    for fname, env_key, lo, hi in _TRUST_FIELDS:
        raw_env = os.getenv(env_key)
        if raw_env is not None and raw_env.strip():
            t_dict[fname] = _coerce_field(raw_env, t_dict[fname], lo, hi)

    for tid, base in envs.items():
        values = asdict(base)
        for field_name, template in _ENV_VAR_OVERRIDES:
            raw = os.getenv(template.format(task=tid.upper()))
            if raw is not None and raw.strip():
                values[field_name] = _coerce_env_value(field_name, raw, values[field_name])
        hp_prefix = f"DET_HP_{tid.upper()}_"
        hp = dict(values.get("env_hyperparams") or {})
        for env_k, env_v in os.environ.items():
            if env_k.startswith(hp_prefix) and env_v.strip():
                param_key = env_k[len(hp_prefix) :].lower()
                hp[param_key] = env_v.strip()
        values["env_hyperparams"] = hp
        values["max_pop_size"] = max(int(values["default_pop_size"]), int(values["max_pop_size"]))
        envs[tid] = EnvPolicyConfig(**values)

    return CentralConfig(
        large_pop=large_pop,
        trust=TrustPolicyConfig(**t_dict),
        environments=envs,
        mujoco_wasm_url=wasm_url,
    )


_config_lock = threading.Lock()
_cached_config: CentralConfig | None = None


def get_config() -> CentralConfig:
    global _cached_config
    with _config_lock:
        if _cached_config is None:
            _cached_config = build_config()
        return _cached_config


def reload_config() -> CentralConfig:
    global _cached_config
    with _config_lock:
        _cached_config = build_config()
        return _cached_config
