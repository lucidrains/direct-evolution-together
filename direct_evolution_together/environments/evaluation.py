"""Rollout caching, population seeding, and batch evaluation."""

from __future__ import annotations

from collections import OrderedDict
import threading
from typing import Sequence

import mujoco

from direct_evolution_together.cppn import (
    CPPN,
    create_random_population,
    genome_content_hash,
)

from .adapters import TASK_ADAPTERS, get_task_adapter
from .halfcheetah import get_halfcheetah_model
from .registry import clamp_pop_for_task, get_task_spec, normalize_task_id


_ROLLOUT_CACHE: OrderedDict[tuple[str, str, int, int], dict[str, object]] = OrderedDict()
_ROLLOUT_CACHE_LOCK = threading.Lock()
_MAX_ROLLOUT_CACHE = 512


def _rollout_cached(
    task_id: str,
    genome: dict,
    *,
    seed: int,
    max_steps: int,
    nudge: str | None = None,
) -> dict[str, object]:
    adapter = get_task_adapter(task_id)
    if nudge:
        return adapter.rollout(genome, seed=seed, max_steps=max_steps, nudge=nudge)
    g_hash = genome_content_hash(genome, length=20)
    key = (adapter.task_id, g_hash, int(seed), int(max_steps))
    with _ROLLOUT_CACHE_LOCK:
        hit = _ROLLOUT_CACHE.get(key)
        if hit is not None:
            _ROLLOUT_CACHE.move_to_end(key)
            return hit
    result = adapter.rollout(genome, seed=seed, max_steps=max_steps)
    with _ROLLOUT_CACHE_LOCK:
        _ROLLOUT_CACHE[key] = result
        _ROLLOUT_CACHE.move_to_end(key)
        while len(_ROLLOUT_CACHE) > _MAX_ROLLOUT_CACHE:
            _ROLLOUT_CACHE.popitem(last=False)
    return result


def create_task_population(
    task_id: str | None,
    pop_size: int,
) -> list[dict]:
    """Generate a Gen-0 population for `task_id`, rejecting pre-solved control founders."""
    clean_task = normalize_task_id(task_id)
    spec = get_task_spec(clean_task)
    eff_pop = clamp_pop_for_task(clean_task, pop_size)
    adapter = TASK_ADAPTERS.get(clean_task)
    if spec.kind != "control" or adapter is None:
        return create_random_population(pop_size=eff_pop)

    founders: list[dict] = []
    max_attempts = eff_pop * 12
    attempts = 0
    while len(founders) < eff_pop and attempts < max_attempts:
        attempts += 1
        cand = CPPN.random().to_dict()
        ev = adapter.evaluate(cand, seeds=spec.eval_seeds)
        if not adapter.is_gen0_presolved(ev):
            founders.append(cand)
    while len(founders) < eff_pop:
        founders.append(CPPN.random().to_dict())
    return founders


def is_gen0_founder_presolved(task_id: str, genome: dict) -> bool:
    adapter = TASK_ADAPTERS.get(normalize_task_id(task_id))
    if adapter is None:
        return False
    spec = get_task_spec(task_id)
    return adapter.is_gen0_presolved(adapter.evaluate(genome, seeds=spec.eval_seeds))


def evaluate_control_genome(
    task_id: str,
    genome_or_net: dict | CPPN,
    seeds: Sequence[int] | None = None,
) -> dict[str, object]:
    """Evaluate a CPPN genome across deterministic seeds for a registered control task."""
    adapter = get_task_adapter(task_id)
    return adapter.evaluate(genome_or_net, seeds=seeds)


def rollout_control_episode(
    task_id: str,
    genome: dict,
    *,
    seed: int = 0,
    max_steps: int | None = None,
    nudge: str | None = None,
    include_geoms: bool = False,
) -> dict[str, object]:
    """Run (or retrieve from LRU cache) a single control rollout for browser replay/eval."""
    spec = get_task_spec(task_id)
    eff_steps = spec.max_steps if max_steps is None else max(1, min(spec.max_steps, int(max_steps)))
    ro = _rollout_cached(spec.task_id, genome, seed=int(seed), max_steps=eff_steps, nudge=nudge)
    if include_geoms and spec.task_id == "halfcheetah":
        return attach_halfcheetah_geoms(ro)
    return ro


def attach_halfcheetah_geoms(rollout: dict[str, object]) -> dict[str, object]:
    """Attach exact MuJoCo `geom_xpos` / `geom_xmat` 3D geometry poses from qpos frames on demand."""
    frames = rollout.get("frames") or []
    if not isinstance(frames, list) or not frames:
        return rollout
    model = get_halfcheetah_model()
    data = mujoco.MjData(model)
    geom_specs = []
    non_plane_ids = []
    for gid in range(model.ngeom):
        gtype = int(model.geom_type[gid])
        if gtype == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        non_plane_ids.append(gid)
        geom_specs.append({
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"g{gid}",
            "type": gtype,
            "size": [round(float(x), 4) for x in model.geom_size[gid]],
            "rgba": [round(float(x), 3) for x in model.geom_rgba[gid]],
        })

    geom_poses: list[list[float]] = []
    nq = int(model.nq)
    for fr in frames:
        for i in range(min(nq, len(fr))):
            data.qpos[i] = float(fr[i])
        mujoco.mj_kinematics(model, data)
        row: list[float] = []
        for gid in non_plane_ids:
            pos = data.geom_xpos[gid]
            mat = data.geom_xmat[gid]
            row.extend([round(float(pos[0]), 4), round(float(pos[1]), 4), round(float(pos[2]), 4)])
            row.extend([round(float(mat[k]), 4) for k in range(9)])
        geom_poses.append(row)

    out = dict(rollout)
    out["geomSpecs"] = geom_specs
    out["geomPoses"] = geom_poses
    return out


def evaluate_control_batch(
    task_id: str,
    genomes: Sequence[dict],
    *,
    seeds: Sequence[int] | None = None,
    include_primary_rollout: bool = True,
) -> list[dict[str, object]]:
    """Server-side cached batch evaluation for control grids and server-mode fallback."""
    spec = get_task_spec(task_id)
    adapter = get_task_adapter(spec.task_id)
    eval_seeds = tuple(spec.eval_seeds if seeds is None else seeds)
    results: list[dict[str, object]] = []
    for genome in genomes:
        summary = adapter.evaluate(genome, seeds=eval_seeds)
        entry: dict[str, object] = {
            "hash": genome_content_hash(genome, length=20),
            "summary": summary,
        }
        if include_primary_rollout and eval_seeds:
            entry["primaryRollout"] = _rollout_cached(
                spec.task_id,
                genome,
                seed=int(eval_seeds[0]),
                max_steps=spec.max_steps,
            )
        results.append(entry)
    return results
