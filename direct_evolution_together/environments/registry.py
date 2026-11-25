"""Runtime task lookup: static specs merged with live deployment policy overrides."""

from __future__ import annotations

from dataclasses import replace
import json
from typing import Any, Mapping

from direct_evolution_together.config import get_config
from direct_evolution_together.cppn import MIN_POPULATION_SIZE

from .specs import TASK_SPECS
from .types import TaskSpec

__all__ = [
    "TASK_SPECS",
    "clamp_pop_for_task",
    "default_elite_for_task",
    "default_pop_for_task",
    "extract_task_env_overrides",
    "get_lean_task_specs_payload",
    "get_nim_task_configs_payload",
    "get_task_spec",
    "get_task_specs_payload",
    "normalize_task_id",
    "sanitize_task_env_params",
]


def normalize_task_id(raw: str | None) -> str:
    cleaned = (raw or "").strip().lower()
    return cleaned if cleaned in TASK_SPECS else "image"


def get_task_spec(task_id: str | None) -> TaskSpec:
    clean = normalize_task_id(task_id)
    base = TASK_SPECS[clean]
    cfg = get_config()
    env_cfg = cfg.get_env(clean)
    eff_crowd_contribute = bool(env_cfg.crowd_contribute and cfg.trust.enabled)
    eff_sim_phase = bool(env_cfg.sim_phase_enabled and eff_crowd_contribute)
    resolved_env_params = base.resolve_env_params(env_cfg.env_hyperparams or None) if base.env_params else None
    resolved_scene_id = (
        str(resolved_env_params.get("scene_id") or base.scene_id)
        if resolved_env_params and base.scene_id
        else base.scene_id
    )
    return replace(
        base,
        enabled=env_cfg.enabled,
        lobby_order=env_cfg.lobby_order,
        execution_policy=env_cfg.execution_policy,
        sim_modes=env_cfg.sim_modes,
        default_sim_mode=env_cfg.default_sim_mode,
        client_sim=env_cfg.client_sim,
        crowd_contribute=eff_crowd_contribute,
        sim_phase_enabled=eff_sim_phase,
        default_auto_evolve=env_cfg.default_auto_evolve,
        allow_server_video=env_cfg.allow_server_video,
        min_pop_size=env_cfg.min_pop_size,
        default_pop_size=env_cfg.default_pop_size,
        max_pop_size=env_cfg.effective_max_pop_size,
        default_min_votes=env_cfg.default_min_votes,
        default_elite_frac=env_cfg.default_elite_frac,
        default_brood_size=env_cfg.default_brood_size,
        crowd_shard_threshold=env_cfg.crowd_shard_threshold,
        crowd_shard_size=env_cfg.crowd_shard_size,
        scene_id=resolved_scene_id,
        env_params=resolved_env_params,
    )


def extract_task_env_overrides(task_id: str | None, source: Mapping[str, Any] | None) -> dict[str, object]:
    """Extract declared task hyperparameter overrides from an arbitrary request dict or query map."""
    if not isinstance(source, Mapping):
        return {}
    spec = get_task_spec(task_id)
    out: dict[str, object] = {}
    raw_box = source.get("env_params")
    if isinstance(raw_box, Mapping):
        out.update({str(k): v for k, v in raw_box.items() if v is not None and v != ""})
    elif isinstance(raw_box, str) and raw_box.strip():
        try:
            parsed = json.loads(raw_box)
            if isinstance(parsed, Mapping):
                out.update({str(k): v for k, v in parsed.items() if v is not None and v != ""})
        except ValueError:
            pass
    for hspec in spec.hyperparam_specs:
        hkey = str(hspec.get("key") or "")
        if not hkey:
            continue
        for k in (hkey, *(str(a) for a in (hspec.get("aliases") or ()))):
            val = source.get(k)
            if val is not None and val != "":
                out.setdefault(hkey, val)
                break
    return out


def sanitize_task_env_params(task_id: str | None, raw_overrides: object = None) -> dict[str, object]:
    """Validate and resolve task-specific environment hyperparameters within centralized bounds."""
    spec = get_task_spec(task_id)
    if not spec.hyperparam_specs:
        return {}
    parsed: Mapping[str, object] | None = None
    if isinstance(raw_overrides, Mapping):
        parsed = raw_overrides
    elif isinstance(raw_overrides, str) and raw_overrides.strip():
        try:
            obj = json.loads(raw_overrides)
            if isinstance(obj, Mapping):
                parsed = obj
        except ValueError:
            first_key = str(spec.hyperparam_specs[0].get("key") or "")
            if first_key:
                parsed = {first_key: raw_overrides.strip()}
    full = spec.resolve_env_params(parsed)
    out: dict[str, object] = {}
    for hspec in spec.hyperparam_specs:
        k = str(hspec.get("key") or "")
        if k and k in full:
            out[k] = full[k]
    for extra_key in spec.public_env_keys:
        if extra_key in full and full[extra_key] is not None:
            out[extra_key] = full[extra_key]
    return out


def get_task_specs_payload() -> dict[str, dict[str, object]]:
    return {k: get_task_spec(k).to_client_dict() for k in TASK_SPECS}


def get_lean_task_specs_payload() -> dict[str, dict[str, object]]:
    return {k: get_task_spec(k).to_lean_client_dict() for k in TASK_SPECS}


def get_nim_task_configs_payload() -> list[dict[str, object]]:
    return [get_task_spec(k).to_nim_dict() for k in TASK_SPECS]


def clamp_pop_for_task(task_id: str | None, pop_size: int) -> int:
    spec = get_task_spec(task_id)
    lo = max(MIN_POPULATION_SIZE, int(spec.min_pop_size))
    return max(lo, min(spec.max_pop_size, int(pop_size)))


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
