"""Runtime task lookup: static specs merged with live deployment policy overrides."""

from __future__ import annotations

from dataclasses import replace

from direct_evolution_together.config import get_config
from direct_evolution_together.cppn import MIN_POPULATION_SIZE

from .specs import TASK_SPECS
from .types import TaskSpec

__all__ = [
    "TASK_SPECS",
    "clamp_pop_for_task",
    "default_elite_for_task",
    "default_pop_for_task",
    "get_nim_task_configs_payload",
    "get_task_spec",
    "get_task_specs_payload",
    "normalize_task_id",
]


def normalize_task_id(raw: str | None) -> str:
    cleaned = (raw or "").strip().lower()
    return cleaned if cleaned in TASK_SPECS else "image"


def get_task_spec(task_id: str | None) -> TaskSpec:
    clean = normalize_task_id(task_id)
    base = TASK_SPECS[clean]
    env_cfg = get_config().get_env(clean)
    return replace(
        base,
        enabled=env_cfg.enabled,
        execution_policy=env_cfg.execution_policy,
        sim_modes=env_cfg.sim_modes,
        default_sim_mode=env_cfg.default_sim_mode,
        client_sim=env_cfg.client_sim,
        crowd_contribute=env_cfg.crowd_contribute,
        allow_server_video=env_cfg.allow_server_video,
        default_pop_size=env_cfg.default_pop_size,
        max_pop_size=env_cfg.max_pop_size,
        default_min_votes=env_cfg.default_min_votes,
        default_elite_frac=env_cfg.default_elite_frac,
    )


def get_task_specs_payload() -> dict[str, dict[str, object]]:
    return {k: get_task_spec(k).to_client_dict() for k in TASK_SPECS}


def get_nim_task_configs_payload() -> list[dict[str, object]]:
    return [get_task_spec(k).to_nim_dict() for k in TASK_SPECS]


def clamp_pop_for_task(task_id: str | None, pop_size: int) -> int:
    spec = get_task_spec(task_id)
    return max(MIN_POPULATION_SIZE, min(spec.max_pop_size, int(pop_size)))


def default_pop_for_task(task_id: str | None, fallback: int | None = None) -> int:
    spec = get_task_spec(task_id)
    if spec.kind == "control":
        return spec.default_pop_size
    base = spec.default_pop_size if fallback is None else int(fallback)
    return max(MIN_POPULATION_SIZE, min(spec.max_pop_size, base))


def default_elite_for_task(task_id: str | None, fallback: float) -> float:
    spec = get_task_spec(task_id)
    raw = spec.default_elite_frac if spec.default_elite_frac is not None else fallback
    return max(0.0, min(0.5, float(raw)))
