from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import threading

_VALID_POLICIES = {"client_locked", "client_preferred", "server_only", "external_worker"}
_VALID_AGGS = {"median", "weighted_mean"}


@dataclass(frozen=True)
class TrustPolicyConfig:
    enabled: bool = True
    quorum: int = 2
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
            "spotCheckRate": float(self.spot_check_rate),
            "shardShuffle": bool(self.shard_shuffle),
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
    allow_server_video: bool = True
    default_pop_size: int = 30
    max_pop_size: int = 64
    default_min_votes: int = 20
    default_elite_frac: float | None = None
    default_mut_sigma: float = 0.30
    default_crossover_rate: float = 0.75

    @property
    def sim_modes(self) -> tuple[str, ...]:
        return {
            "client_locked": ("client",),
            "client_preferred": ("client", "server"),
            "server_only": ("server",),
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


@dataclass(frozen=True)
class CentralConfig:
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
                for img_name in ("Butterfly", "Galaxy", "Creature"):
                    specs.append((img_name, "image", None))
            else:
                specs.append((cfg.lobby_name or cfg.task.title(), cfg.task, cfg.default_pop_size))
        return tuple(specs)

    def to_public_dict(self) -> dict[str, object]:
        return {
            "trust": asdict(self.trust),
            "environments": {
                k: {**asdict(v), "simModes": list(v.sim_modes), "defaultSimMode": v.default_sim_mode}
                for k, v in self.environments.items()
            },
            "mujocoWasmUrl": self.mujoco_wasm_url,
        }


_DEFAULT_ENVS: dict[str, EnvPolicyConfig] = {
    "halfcheetah": EnvPolicyConfig(
        task="halfcheetah",
        lobby_name="HalfCheetah",
        lobby_default=True,
        lobby_order=0,
        execution_policy="client_locked",
        crowd_contribute=True,
        allow_server_video=True,
        default_pop_size=16,
        max_pop_size=24,
        default_min_votes=20,
        default_elite_frac=0.05,
    ),
    "cartpole": EnvPolicyConfig(
        task="cartpole",
        lobby_name="CartPole",
        lobby_default=True,
        lobby_order=1,
        execution_policy="client_preferred",
        crowd_contribute=False,
        allow_server_video=True,
        default_pop_size=30,
        max_pop_size=32,
        default_min_votes=20,
        default_elite_frac=0.0,
    ),
    "image": EnvPolicyConfig(
        task="image",
        lobby_name="Butterfly",
        lobby_default=True,
        lobby_order=2,
        execution_policy="client_locked",
        crowd_contribute=False,
        allow_server_video=False,
        default_pop_size=30,
        max_pop_size=64,
        default_min_votes=0,
        default_elite_frac=None,
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

# (field, env_name, lo, hi)
_TRUST_FIELDS: tuple[tuple[str, str, float | None, float | None], ...] = (
    ("enabled", "DET_CROWD_TRUST_ENABLED", None, None),
    ("quorum", "DET_CROWD_QUORUM", 1, 10),
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


def build_config() -> CentralConfig:
    t_dict = asdict(TrustPolicyConfig())
    envs = {k: EnvPolicyConfig(**asdict(v)) for k, v in _DEFAULT_ENVS.items()}
    wasm_url = os.getenv("DET_MUJOCO_WASM_URL", "").strip()

    cfg_raw = os.getenv("DET_CONFIG_FILE", "").strip()
    if cfg_raw and Path(cfg_raw).is_file():
        try:
            loaded = json.loads(Path(cfg_raw).read_text(encoding="utf-8"))
        except Exception:
            loaded = {}
        if isinstance(loaded, dict):
            if isinstance(loaded.get("mujoco_wasm_url"), str):
                wasm_url = loaded["mujoco_wasm_url"].strip()
            raw_t = loaded.get("trust")
            if isinstance(raw_t, dict):
                for fname, _, lo, hi in _TRUST_FIELDS:
                    if fname in raw_t:
                        t_dict[fname] = _coerce_field(raw_t[fname], t_dict[fname], lo, hi)
            raw_e = loaded.get("environments")
            if isinstance(raw_e, dict):
                for tid, ov in raw_e.items():
                    if tid not in envs or not isinstance(ov, dict):
                        continue
                    ek = {**asdict(envs[tid]), **{k: v for k, v in ov.items() if k in envs[tid].__dataclass_fields__}}
                    if ek["execution_policy"] not in _VALID_POLICIES:
                        ek["execution_policy"] = envs[tid].execution_policy
                    ek["enabled"] = bool(ek["enabled"])
                    ek["default_pop_size"] = max(3, min(64, int(ek["default_pop_size"])))
                    ek["max_pop_size"] = max(ek["default_pop_size"], min(64, int(ek["max_pop_size"])))
                    envs[tid] = EnvPolicyConfig(**ek)

    for fname, env_key, lo, hi in _TRUST_FIELDS:
        raw_env = os.getenv(env_key)
        if raw_env is not None and raw_env.strip():
            t_dict[fname] = _coerce_field(raw_env, t_dict[fname], lo, hi)

    for tid, base in list(envs.items()):
        u = tid.upper()
        ek = asdict(base)
        for field_name, env_key in (
            ("enabled", f"DET_TASK_ENABLED_{u}"),
            ("crowd_contribute", f"DET_CROWD_CONTRIBUTE_{u}"),
            ("allow_server_video", f"DET_ALLOW_VIDEO_{u}"),
            ("default_pop_size", f"DET_POP_SIZE_{u}"),
            ("max_pop_size", f"DET_MAX_POP_SIZE_{u}"),
        ):
            raw_val = os.getenv(env_key)
            if raw_val is not None and raw_val.strip():
                lo, hi = (3, 64) if "POP_SIZE" in env_key else (None, None)
                ek[field_name] = _coerce_field(raw_val, ek[field_name], lo, hi)
        pol = os.getenv(f"DET_EXECUTION_POLICY_{u}", "").strip().lower()
        if pol in _VALID_POLICIES:
            ek["execution_policy"] = pol
        ek["max_pop_size"] = max(ek["default_pop_size"], ek["max_pop_size"])
        envs[tid] = EnvPolicyConfig(**ek)

    return CentralConfig(trust=TrustPolicyConfig(**t_dict), environments=envs, mujoco_wasm_url=wasm_url)


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
