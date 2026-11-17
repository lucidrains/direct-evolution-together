"""Task descriptors and the control-task adapter contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from direct_evolution_together.cppn import CPPN

from .substrate import get_substrate


@dataclass(frozen=True)
class TaskSpec:
    """Unified descriptor for image, built-in control, and external simulation tasks."""

    task_id: str
    kind: str  # "image" | "control"
    label: str
    description: str
    default_pop_size: int
    max_pop_size: int
    default_min_votes: int
    client_sim: bool
    supports_video: bool
    default_elite_frac: float | None = None
    enabled: bool = True
    execution_policy: str = "client_preferred"
    crowd_contribute: bool = False
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
    primary_metric: str = "meanSteps"
    sort_label: str = "Steps"
    solved_label: str = "Solved"
    telemetry_fields: tuple[dict[str, str], ...] = ()
    nudges: tuple[dict[str, object], ...] = ()

    @property
    def substrate_dims(self) -> tuple[int, ...]:
        if self.kind != "control" or self.obs_dim <= 0 or self.action_dim <= 0:
            return ()
        return (self.obs_dim, *self.hidden_dims, self.action_dim)

    def resolved_layer_coords(self) -> list[list[float]]:
        dims = self.substrate_dims
        if not dims:
            return []
        return [list(c) for c in get_substrate(dims, self.layer_coords).layer_coords]

    def to_nim_dict(self) -> dict[str, object]:
        return {
            "task": self.task_id,
            "isControl": self.kind == "control",
            "defaultPopSize": self.default_pop_size,
            "maxPopSize": self.max_pop_size,
            "defaultMinVotes": self.default_min_votes,
            "defaultEliteFrac": self.default_elite_frac,
        }

    def to_client_dict(self) -> dict[str, object]:
        return {
            "task": self.task_id,
            "kind": self.kind,
            "label": self.label,
            "description": self.description,
            "defaultPopSize": self.default_pop_size,
            "maxPopSize": self.max_pop_size,
            "defaultMinVotes": self.default_min_votes,
            "defaultEliteFrac": self.default_elite_frac,
            "clientSim": self.client_sim,
            "enabled": self.enabled,
            "executionPolicy": self.execution_policy,
            "crowdContribute": self.crowd_contribute,
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
