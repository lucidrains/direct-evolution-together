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
    default_brood_size: int = 1
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
    public_env_keys: tuple[str, ...] = ("scene_id", "short_label", "gripper_kind", "urdf_asset", "stl_assets")
    primary_metric: str = "mean_steps"
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
            "is_control": self.kind == "control",
            "genome_kind": self.genome_kind,
            "sim_phase_enabled": self.sim_phase_enabled,
            "default_auto_evolve": self.default_auto_evolve,
            "min_pop_size": int(self.min_pop_size),
            "default_pop_size": self.default_pop_size,
            "max_pop_size": self.max_pop_size,
            "default_min_votes": self.default_min_votes,
            "default_elite_frac": self.default_elite_frac,
            "default_brood_size": int(self.default_brood_size),
            "crowd_shard_threshold": int(self.crowd_shard_threshold),
            "crowd_shard_size": int(self.crowd_shard_size),
        }

    def _base_client_dict(self) -> dict[str, object]:
        return {
            "task": self.task_id,
            "kind": self.kind,
            "genome_kind": self.genome_kind,
            "lobby_order": self.lobby_order,
            "label": self.label,
            "min_pop_size": int(self.min_pop_size),
            "default_pop_size": self.default_pop_size,
            "max_pop_size": self.max_pop_size,
            "default_min_votes": self.default_min_votes,
            "default_elite_frac": self.default_elite_frac,
            "default_brood_size": int(self.default_brood_size),
            "crowd_shard_threshold": int(self.crowd_shard_threshold),
            "crowd_shard_size": int(self.crowd_shard_size),
            "client_sim": self.client_sim,
            "enabled": self.enabled,
            "execution_policy": self.execution_policy,
            "crowd_contribute": self.crowd_contribute,
            "sim_phase_enabled": self.sim_phase_enabled,
            "default_auto_evolve": self.default_auto_evolve,
            "allow_server_video": self.allow_server_video,
            "sim_modes": list(self.sim_modes),
            "default_sim_mode": self.default_sim_mode,
            "supports_video": self.supports_video,
            "obs_dim": self.obs_dim,
            "hidden_dims": list(self.hidden_dims),
            "action_dim": self.action_dim,
            "continuous": self.continuous,
            "max_steps": self.max_steps,
            "dt": self.dt,
            "eval_seeds": list(self.eval_seeds),
            "phase_freq": self.phase_freq,
            "phase_harmonics": self.phase_harmonics,
            "primary_metric": self.primary_metric,
            "sort_label": self.sort_label,
            "solved_label": self.solved_label,
        }

    def to_lean_client_dict(self) -> dict[str, object]:
        """Minimal boot-time descriptor embedded in HTML; full mechanics hydrate on demand."""
        def _lean_manifest(sid: str | None) -> dict[str, object] | None:
            m = get_scene_manifest(sid)
            if not m:
                return None
            return {
                "scene_id": m.get("scene_id"),
                "physics_asset": m.get("physics_asset"),
                "physics_sha256": str(m.get("physics_sha256") or "")[:12],
                "visual_asset": m.get("visual_asset"),
                "visual_sha256": str(m.get("visual_sha256") or "")[:12],
                "nbody": m.get("nbody"),
                "camera_defaults": m.get("camera_defaults"),
            }

        resolved_params = self.resolve_env_params()
        eff_scene_id = str(resolved_params.get("scene_id") or self.scene_id or "") or None
        lean_env_params: dict[str, object] | None = None
        if resolved_params:
            keep_keys = {"scene_id", "short_label", "gripper_kind", *(str(h.get("key") or "") for h in self.hyperparam_specs)}
            filtered = {k: v for k, v in resolved_params.items() if k in keep_keys and v is not None}
            lean_env_params = filtered or None
        lean_variants: dict[str, dict[str, object]] | None = None
        if self.variants:
            lean_variants = {}
            for vk, vv in self.variants.items():
                v_scene = str(vv.get("scene_id") or self.scene_id or "") or None
                v_entry: dict[str, object] = {}
                if v_scene and v_scene != eff_scene_id:
                    v_entry["scene_manifest"] = _lean_manifest(v_scene)
                for k in ("scene_id", "short_label", "gripper_kind"):
                    if k in vv and vv[k] is not None:
                        v_entry[k] = vv[k]
                lean_variants[vk] = v_entry
        lean_hspecs: list[dict[str, object]] = []
        for h in self.hyperparam_specs:
            hs: dict[str, object] = {}
            for k in ("key", "label", "type", "default", "min", "max", "step", "aliases"):
                if k in h and h[k] is not None:
                    hs[k] = h[k]
            if isinstance(h.get("options"), (list, tuple)):
                hs["options"] = [
                    {ok: ov for ok, ov in opt.items() if ok in ("value", "label", "aliases") and ov is not None}
                    for opt in h["options"]
                    if isinstance(opt, Mapping)
                ]
            lean_hspecs.append(hs)
        out: dict[str, object] = {
            **self._base_client_dict(),
            "is_lean": True,
        }
        if self.mjcf_asset:
            out["mjcf_asset"] = self.mjcf_asset
        if eff_scene_id:
            out["scene_id"] = eff_scene_id
            out["scene_manifest"] = _lean_manifest(eff_scene_id)
        if lean_env_params:
            out["env_params"] = lean_env_params
        if lean_hspecs:
            out["hyperparam_specs"] = lean_hspecs
        if lean_variants:
            out["variants"] = lean_variants
        return out

    def to_client_dict(self) -> dict[str, object]:
        resolved_params = self.resolve_env_params()
        eff_scene_id = str(resolved_params.get("scene_id") or self.scene_id or "") or None
        variants_payload: dict[str, dict[str, object]] | None = None
        if self.variants:
            variants_payload = {}
            for vk, vv in self.variants.items():
                v_scene = str(vv.get("scene_id") or self.scene_id or "") or None
                variants_payload[vk] = {
                    **dict(vv),
                    "scene_manifest": get_scene_manifest(v_scene),
                }
        return {
            **self._base_client_dict(),
            "description": self.description,
            "heldout_seeds": list(self.heldout_seeds),
            "layer_coords": self.resolved_layer_coords(),
            "obs_labels": list(self.obs_labels),
            "action_labels": list(self.action_labels),
            "mjcf_asset": self.mjcf_asset,
            "scene_id": eff_scene_id,
            "scene_manifest": get_scene_manifest(eff_scene_id),
            "policy_spec": dict(self.policy_spec) if self.policy_spec else None,
            "env_params": resolved_params if resolved_params else None,
            "hyperparam_specs": [dict(h) for h in self.hyperparam_specs],
            "variants": variants_payload,
            "telemetry_fields": [dict(f) for f in self.telemetry_fields],
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
