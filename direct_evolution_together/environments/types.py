"""Task descriptors and the control-task adapter contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from direct_evolution_together.cppn import CPPN

from .scene_assets import get_scene_manifest
from .substrate import get_substrate


# helpers

def exists(val: Any) -> bool:
    return val is not None


def default(val: Any, d: Any) -> Any:
    return val if exists(val) else d


def _norm_param_key(key: Any) -> str:
    return str(key or "").strip().lower().replace("_", "").replace("-", "")


def _lookup_override(ov: Mapping[str, Any], candidates: Sequence[str]) -> Any:
    cand_norms = {_norm_param_key(c) for c in candidates if c}
    for k, v in ov.items():
        if exists(v) and v != "" and (k in candidates or _norm_param_key(k) in cand_norms):
            return v
    return None


def _match_hyperparam_option(hspec: Mapping[str, Any], raw_val: Any) -> str:
    norm = str(raw_val).strip().lower().replace("-", "_")
    for opt in hspec.get("options") or ():
        if not isinstance(opt, Mapping):
            continue
        opt_val = str(opt.get("value") or "").strip().lower()
        aliases = {opt_val, *(str(a).strip().lower() for a in (opt.get("aliases") or ()))}
        if norm in aliases:
            return opt_val
    return ""


def _coerce_hyperparam_value(hspec: Mapping[str, Any], raw_val: Any) -> Any:
    htype = str(hspec.get("type") or "select").strip().lower()
    if htype == "select":
        matched = _match_hyperparam_option(hspec, raw_val)
        return matched if matched else hspec.get("default")
    if htype in ("float", "number"):
        try:
            val = float(raw_val)
        except (TypeError, ValueError):
            return hspec.get("default")
        if not np.isfinite(val):
            return hspec.get("default")
        if exists(hspec.get("min")):
            val = max(float(hspec["min"]), val)
        if exists(hspec.get("max")):
            val = min(float(hspec["max"]), val)
        return round(val, 4)
    if htype == "int":
        try:
            ival = int(round(float(raw_val)))
        except (TypeError, ValueError):
            return hspec.get("default")
        if exists(hspec.get("min")):
            ival = max(int(hspec["min"]), ival)
        if exists(hspec.get("max")):
            ival = min(int(hspec["max"]), ival)
        return ival
    if htype == "bool":
        if isinstance(raw_val, bool):
            return raw_val
        return str(raw_val).strip().lower() in ("1", "true", "yes", "on")
    return raw_val


@dataclass(frozen=True)
class TaskSpec:
    """Unified descriptor for image, built-in control, PopuLoRA robotics, and external tasks."""

    task_id: str
    kind: str  # "image" | "control"
    label: str
    description: str
    default_pop_size: int
    max_pop_size: int
    default_min_votes: int
    client_sim: bool
    supports_video: bool
    min_pop_size: int = 3
    genome_kind: str = "cppn"  # "cppn" | "populora"
    lobby_order: int = 99
    default_elite_frac: float | None = None
    enabled: bool = True
    execution_policy: str = "client_preferred"
    crowd_contribute: bool = False
    sim_phase_enabled: bool = False
    default_auto_evolve: bool = False
    allow_server_video: bool = True
    sim_modes: tuple[str, ...] = ("client",)
    default_sim_mode: str = "client"
    obs_dim: int = 0
    hidden_dims: tuple[int, ...] = ()
    action_dim: int = 0
    continuous: bool = True
    max_steps: int = 0
    dt: float = 0.02
    eval_seeds: tuple[int, ...] = ()
    heldout_seeds: tuple[int, ...] = tuple(range(100, 110))
    layer_coords: tuple[tuple[float, ...] | None, ...] | None = None
    phase_freq: float = 0.0
    phase_harmonics: int = 0
    obs_labels: tuple[str, ...] = ()
    action_labels: tuple[str, ...] = ()
    mjcf_asset: str | None = None
    scene_id: str | None = None
    policy_spec: dict[str, object] | None = None
    env_params: dict[str, object] | None = None
    crowd_shard_threshold: int = 16
    crowd_shard_size: int = 4
    hyperparam_specs: tuple[dict[str, object], ...] = ()
    variants: dict[str, dict[str, object]] | None = None
    public_env_keys: tuple[str, ...] = ("sceneId", "shortLabel", "gripperKind", "urdfAsset", "stlAssets")
    primary_metric: str = "meanSteps"
    sort_label: str = "Steps"
    solved_label: str = "Solved"
    telemetry_fields: tuple[dict[str, str], ...] = ()
    nudges: tuple[dict[str, object], ...] = ()

    @property
    def substrate_dims(self) -> tuple[int, ...]:
        if self.kind != "control" or self.genome_kind != "cppn" or self.obs_dim <= 0 or self.action_dim <= 0:
            return ()
        return (self.obs_dim, *self.hidden_dims, self.action_dim)

    def resolved_layer_coords(self) -> list[list[float]]:
        dims = self.substrate_dims
        if not dims:
            return []
        return [list(c) for c in get_substrate(dims, self.layer_coords).layer_coords]

    def resolve_env_params(self, overrides: Mapping[str, object] | None = None) -> dict[str, object]:
        merged = dict(self.env_params or {})
        ov = overrides if isinstance(overrides, Mapping) else {}

        if self.variants:
            selected = ""
            for hspec in self.hyperparam_specs:
                keys = (str(hspec.get("key") or ""), *(str(a) for a in (hspec.get("aliases") or ())))
                raw_v = _lookup_override(ov, keys)
                if not exists(raw_v):
                    raw_v = default(merged.get(keys[0]), hspec.get("default"))
                if exists(raw_v):
                    cand = _match_hyperparam_option(hspec, raw_v)
                    if cand and cand in self.variants:
                        selected = cand
                        break
            if selected in self.variants:
                merged = {**merged, **dict(self.variants[selected])}

        for hspec in self.hyperparam_specs:
            hkey = str(hspec.get("key") or "")
            if not hkey:
                continue
            keys = (hkey, *(str(a) for a in (hspec.get("aliases") or ())))
            raw_v = _lookup_override(ov, keys)
            if exists(raw_v):
                coerced = _coerce_hyperparam_value(hspec, raw_v)
                if exists(coerced):
                    merged[hkey] = coerced
            elif hkey not in merged and exists(hspec.get("default")):
                merged[hkey] = hspec["default"]
        return merged

    def to_nim_dict(self) -> dict[str, object]:
        return {
            "task": self.task_id,
            "isControl": self.kind == "control",
            "genomeKind": self.genome_kind,
            "simPhaseEnabled": self.sim_phase_enabled,
            "defaultAutoEvolve": self.default_auto_evolve,
            "minPopSize": int(self.min_pop_size),
            "defaultPopSize": self.default_pop_size,
            "maxPopSize": self.max_pop_size,
            "defaultMinVotes": self.default_min_votes,
            "defaultEliteFrac": self.default_elite_frac,
            "crowdShardThreshold": int(self.crowd_shard_threshold),
            "crowdShardSize": int(self.crowd_shard_size),
        }

    def to_client_dict(self) -> dict[str, object]:
        resolved_params = self.resolve_env_params()
        eff_scene_id = str(resolved_params.get("sceneId") or self.scene_id or "") or None
        variants_payload: dict[str, dict[str, object]] | None = None
        if self.variants:
            variants_payload = {}
            for vk, vv in self.variants.items():
                v_scene = str(vv.get("sceneId") or self.scene_id or "") or None
                variants_payload[vk] = {
                    **dict(vv),
                    "sceneManifest": get_scene_manifest(v_scene),
                }
        return {
            "task": self.task_id,
            "kind": self.kind,
            "genomeKind": self.genome_kind,
            "lobbyOrder": self.lobby_order,
            "label": self.label,
            "description": self.description,
            "minPopSize": int(self.min_pop_size),
            "defaultPopSize": self.default_pop_size,
            "maxPopSize": self.max_pop_size,
            "defaultMinVotes": self.default_min_votes,
            "defaultEliteFrac": self.default_elite_frac,
            "crowdShardThreshold": int(self.crowd_shard_threshold),
            "crowdShardSize": int(self.crowd_shard_size),
            "clientSim": self.client_sim,
            "enabled": self.enabled,
            "executionPolicy": self.execution_policy,
            "crowdContribute": self.crowd_contribute,
            "simPhaseEnabled": self.sim_phase_enabled,
            "defaultAutoEvolve": self.default_auto_evolve,
            "allowServerVideo": self.allow_server_video,
            "simModes": list(self.sim_modes),
            "defaultSimMode": self.default_sim_mode,
            "supportsVideo": self.supports_video,
            "obsDim": self.obs_dim,
            "hiddenDims": list(self.hidden_dims),
            "actionDim": self.action_dim,
            "continuous": self.continuous,
            "maxSteps": self.max_steps,
            "dt": self.dt,
            "evalSeeds": list(self.eval_seeds),
            "heldoutSeeds": list(self.heldout_seeds),
            "layerCoords": self.resolved_layer_coords(),
            "phaseFreq": self.phase_freq,
            "phaseHarmonics": self.phase_harmonics,
            "obsLabels": list(self.obs_labels),
            "actionLabels": list(self.action_labels),
            "mjcfAsset": self.mjcf_asset,
            "sceneId": eff_scene_id,
            "sceneManifest": get_scene_manifest(eff_scene_id),
            "policySpec": dict(self.policy_spec) if self.policy_spec else None,
            "envParams": resolved_params if resolved_params else None,
            "hyperparamSpecs": [dict(h) for h in self.hyperparam_specs],
            "variants": variants_payload,
            "primaryMetric": self.primary_metric,
            "sortLabel": self.sort_label,
            "solvedLabel": self.solved_label,
            "telemetryFields": [dict(f) for f in self.telemetry_fields],
            "nudges": [dict(n) for n in self.nudges],
        }


class ControlTaskAdapter(Protocol):
    """Server-side simulator interface for one control task."""

    task_id: str

    def rollout(
        self,
        genome_or_net: dict | CPPN,
        *,
        seed: int = 0,
        max_steps: int | None = None,
        nudge: str | None = None,
    ) -> dict[str, object]: ...

    def evaluate(
        self,
        genome_or_net: dict | CPPN,
        *,
        seeds: Sequence[int] | None = None,
    ) -> dict[str, object]: ...

    def is_gen0_presolved(self, summary: dict[str, object]) -> bool: ...

    def render_frame(
        self,
        rollout: dict[str, object],
        step_idx: int,
        max_steps: int,
        width: int,
        height: int,
    ) -> np.ndarray: ...
