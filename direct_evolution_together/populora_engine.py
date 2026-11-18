"""General PopuLoRA population, policy, serialization, and evolution manager.

Bridges ``populora.Population`` (shared base neural network + per-individual low-rank
LoRA adapters, Firefly PSO, AURORA unsupervised trajectory descriptors, and Dominated
Novelty Search) with browser-crowdsourced MuJoCo WASM simulations.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import hashlib
import io
import json
import math
from pathlib import Path
import threading
from typing import Iterable, Sequence

from einops import einsum, rearrange
import numpy as np
from populora import (
    Aurora,
    CosineAnnealingSchedule,
    Memory,
    Population,
    preserve_rng,
    quality_dominated_novelty,
    sketch_trajectories,
)
import torch
from torch import Tensor
import torch.nn as nn

from direct_evolution_together.cppn import _alloc_genome_id

MAX_POPULORA_CHECKPOINT_BYTES = 4 * 1024 * 1024  # 4 MB cap for merged GRU/MLP .pt files


class ManualGRUCell(nn.Module):
    """Exact pure-Linear GRU cell matching ``populora/train_microduck.py`` for LoRA targeting."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_ih = nn.Linear(input_size, 3 * hidden_size)
        self.linear_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        i_r, i_z, i_n = self.linear_ih(x).chunk(3, dim=-1)
        h_r, h_z, h_n = self.linear_hh(h).chunk(3, dim=-1)
        r = (i_r + h_r).sigmoid()
        z = (i_z + h_z).sigmoid()
        n = (i_n + r * h_n).tanh()
        return (1.0 - z) * n + z * h


class GRUMLPPolicy(nn.Module):
    """Recurrent encoder -> ManualGRUCell -> decoder policy with squashed Gaussian head."""

    def __init__(
        self,
        obs_dim: int = 55,
        action_dim: int = 14,
        hidden_dim: int = 128,
        head_dim: int = 128,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.cell = ManualGRUCell(hidden_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.Tanh(),
            nn.Linear(head_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        with torch.no_grad():
            self.cell.linear_hh.weight.mul_(0.08)
            self.cell.linear_ih.bias.zero_()
            self.cell.linear_ih.bias[:hidden_dim].fill_(0.8)
            self.cell.linear_ih.bias[hidden_dim : 2 * hidden_dim].uniform_(0.3, 1.3)
            for m in (self.encoder[0], self.encoder[2], self.decoder[0], self.decoder[2]):
                m.bias.zero_()
            self.decoder[2].weight.zero_()

    def forward(self, mem: Tensor, obs: Tensor) -> tuple[Tensor, Tensor]:
        next_mem = self.cell(self.encoder(obs), mem)
        mean = self.decoder(next_mem)
        return torch.cat((mean, self.log_std.expand_as(mean)), dim=-1), next_mem


class MLPPolicy(nn.Module):
    """Feedforward MLP policy with squashed Gaussian head for non-recurrent control tasks."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        head_dim: int = 128,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, head_dim),
            nn.Tanh(),
        )
        self.head = nn.Linear(head_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        with torch.no_grad():
            for m in (self.encoder[0], self.encoder[2], self.head):
                m.bias.zero_()
            self.head.weight.zero_()

    def forward(self, obs: Tensor) -> Tensor:
        mean = self.head(self.encoder(obs))
        return torch.cat((mean, self.log_std.expand_as(mean)), dim=-1)


def _encode_f16_b64(tensor: Tensor) -> str:
    arr = np.ascontiguousarray(tensor.detach().cpu().numpy().astype("<f2"))
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _decode_f16_b64(b64_str: str, shape: tuple[int, ...]) -> Tensor:
    raw = base64.b64decode(b64_str, validate=True)
    expected_bytes = int(math.prod(shape)) * 2
    if len(raw) != expected_bytes:
        raise ValueError(f"Invalid float16 buffer length ({len(raw)} != {expected_bytes}).")
    arr = np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(shape)
    if not np.isfinite(arr).all():
        raise ValueError("Adapter tensor contains non-finite values.")
    return torch.from_numpy(arr.copy())


def _encode_f32_b64(tensor: Tensor) -> str:
    arr = np.ascontiguousarray(tensor.detach().cpu().numpy().astype("<f4"))
    return base64.b64encode(arr.tobytes()).decode("ascii")


def build_policy_module(policy_spec: dict[str, object], *, base_seed: int = 1) -> nn.Module:
    """Instantiate a deterministic base policy module from a declarative ``policy_spec``."""
    arch = str(policy_spec.get("arch") or "gru_mlp")
    obs_dim = int(policy_spec["obsDim"])
    act_dim = int(policy_spec["actionDim"])
    hidden_dim = int(policy_spec.get("hiddenDim", 128))
    head_dim = int(policy_spec.get("headDim", hidden_dim))

    with preserve_rng():
        torch.manual_seed(int(base_seed))
        if arch == "gru_mlp":
            net = GRUMLPPolicy(obs_dim=obs_dim, action_dim=act_dim, hidden_dim=hidden_dim, head_dim=head_dim)
            init_mem = 0.18 * torch.randn(1, hidden_dim)
            return Memory(net, init_memory=init_mem)
        if arch == "mlp":
            return MLPPolicy(obs_dim=obs_dim, action_dim=act_dim, hidden_dim=hidden_dim, head_dim=head_dim)
    raise ValueError(f"Unsupported PopuLoRA policy architecture: {arch!r}")


def compute_base_model_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, val in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        arr = np.ascontiguousarray(val.detach().cpu().float().numpy().astype("<f4"))
        h.update(arr.tobytes())
    return h.hexdigest()[:16]


def serialize_base_model_payload(model: nn.Module, policy_spec: dict[str, object], base_hash: str) -> dict[str, object]:
    """Serialize shared base model weights once for browser WASM/JS policy workers."""
    tensors: dict[str, dict[str, object]] = {}
    for name, val in model.state_dict().items():
        shape = [int(d) for d in val.shape]
        # Keep 1D biases / init_memory in exact f32 and 2D weight matrices in compact f16
        use_f32 = val.ndim <= 1
        tensors[name] = {
            "shape": shape,
            "dtype": "f32" if use_f32 else "f16",
            "b64": _encode_f32_b64(val) if use_f32 else _encode_f16_b64(val),
        }
    return {
        "baseHash": base_hash,
        "arch": str(policy_spec.get("arch") or "gru_mlp"),
        "obsDim": int(policy_spec["obsDim"]),
        "actionDim": int(policy_spec["actionDim"]),
        "hiddenDim": int(policy_spec.get("hiddenDim", 128)),
        "headDim": int(policy_spec.get("headDim", 128)),
        "lowRank": int(policy_spec.get("lowRank", 8)),
        "loraTargets": list(policy_spec.get("loraTargets") or []),
        "actionDist": str(policy_spec.get("actionDist") or "squashed_gaussian"),
        "actionBounds": list(policy_spec.get("actionBounds") or [-1.0, 1.0]),
        "tensors": tensors,
    }


_BASE_MODEL_LOCK = threading.Lock()
_BASE_MODEL_STATES: dict[str, dict[str, Tensor]] = {}
_BASE_MODEL_PAYLOADS: dict[str, dict[str, object]] = {}


def _custom_base_dir() -> Path | None:
    try:
        from django.conf import settings
        root = getattr(settings, "MEDIA_ROOT", None)
        if root:
            d = Path(root) / "base_models"
            d.mkdir(parents=True, exist_ok=True)
            return d
    except Exception:
        pass
    return None


def _persist_base_model_static_json(base_hash: str, payload: dict[str, object]) -> None:
    """Write `<MEDIA_ROOT>/base_models/<base_hash>.json(.gz)` so Nginx/media can serve it statically."""
    b_dir = _custom_base_dir()
    if b_dir is None or not base_hash:
        return
    json_path = b_dir / f"{base_hash}.json"
    if json_path.is_file():
        return
    try:
        from direct_evolution_together.public_assets import write_gzip_sidecar

        doc = {"ok": True, **payload}
        raw = json.dumps(doc, separators=(",", ":")).encode("utf-8")
        tmp = json_path.with_name(json_path.name + ".tmp")
        tmp.write_bytes(raw)
        tmp.replace(json_path)
        write_gzip_sidecar(json_path, compresslevel=6)
    except Exception:
        pass


def get_or_create_base_model(
    policy_spec: dict[str, object],
    *,
    base_seed: int = 1,
    base_hash: str | None = None,
    custom_state_dict: dict[str, Tensor] | None = None,
) -> tuple[nn.Module, str, dict[str, object]]:
    """Return ``(model_copy, base_hash, client_payload)`` with deterministic + disk fallback."""
    model = build_policy_module(policy_spec, base_seed=base_seed)
    if custom_state_dict is not None:
        model.load_state_dict(custom_state_dict, strict=False)
        b_hash = compute_base_model_hash(model)
        with _BASE_MODEL_LOCK:
            _BASE_MODEL_STATES[b_hash] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            payload = serialize_base_model_payload(model, policy_spec, b_hash)
            _BASE_MODEL_PAYLOADS[b_hash] = payload
        b_dir = _custom_base_dir()
        if b_dir is not None:
            pt_path = b_dir / f"{b_hash}.pt"
            if not pt_path.exists():
                torch.save(_BASE_MODEL_STATES[b_hash], pt_path)
        _persist_base_model_static_json(b_hash, payload)
        return model, b_hash, payload

    if base_hash:
        with _BASE_MODEL_LOCK:
            cached_state = _BASE_MODEL_STATES.get(base_hash)
            cached_payload = _BASE_MODEL_PAYLOADS.get(base_hash)
        if cached_state is not None and cached_payload is not None:
            model.load_state_dict(cached_state, strict=True)
            _persist_base_model_static_json(base_hash, cached_payload)
            return model, base_hash, cached_payload
        b_dir = _custom_base_dir()
        if b_dir is not None:
            pt_path = b_dir / f"{base_hash}.pt"
            if pt_path.is_file():
                loaded = torch.load(pt_path, map_location="cpu", weights_only=True)
                model.load_state_dict(loaded, strict=False)
                payload = serialize_base_model_payload(model, policy_spec, base_hash)
                with _BASE_MODEL_LOCK:
                    _BASE_MODEL_STATES[base_hash] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    _BASE_MODEL_PAYLOADS[base_hash] = payload
                _persist_base_model_static_json(base_hash, payload)
                return model, base_hash, payload

    b_hash = compute_base_model_hash(model)
    with _BASE_MODEL_LOCK:
        if b_hash not in _BASE_MODEL_PAYLOADS:
            _BASE_MODEL_STATES[b_hash] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            _BASE_MODEL_PAYLOADS[b_hash] = serialize_base_model_payload(model, policy_spec, b_hash)
        payload = _BASE_MODEL_PAYLOADS[b_hash]
    _persist_base_model_static_json(b_hash, payload)
    return model, b_hash, payload


def populora_genome_hash(genome: dict, length: int = 20) -> str:
    """Deterministic SHA-256 content hash of a PopuLoRA genome's baseHash + adapter weights."""
    base_hash = str(genome.get("baseHash") or "")
    adapters = genome.get("adapters") or {}
    parts = [base_hash]
    if isinstance(adapters, dict):
        for k in sorted(adapters.keys()):
            item = adapters[k]
            if isinstance(item, dict):
                parts.append(f"{k}:{item.get('down', '')}:{item.get('up', '')}")
    digest = hashlib.sha256("|".join(parts).encode("ascii")).hexdigest()
    return digest[:length]


@torch.no_grad()
def _seed_oscillatory_population_(pop: Population, has_warmstart: bool = False) -> None:
    """Apply QR orthogonal recurrent basis + oscillatory skew-symmetric modes from train_microduck.py."""
    pop.repopulate_(save_champion=False, std_down=None, std_up=0.04 if has_warmstart else 1.0)
    if has_warmstart:
        return
    if "net_cell_linear_hh" not in pop.weight_down:
        return

    p, r = pop.pop_size, pop.low_rank
    hh_down = pop.weight_down["net_cell_linear_hh"].data
    hh_up = pop.weight_up["net_cell_linear_hh"].data
    hidden = hh_down.shape[1]
    if hh_up.shape[1] != 3 * hidden:
        return

    q, _ = torch.linalg.qr(hh_down)
    hh_down.copy_(q)

    g = torch.randn(1, r, r) + 0.35 * torch.randn(p, r, r)
    w = g - g.transpose(1, 2)
    w = w / w.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-6) * rearrange(
        torch.linspace(3.6, 4.6, p), "p -> p 1 1"
    )
    s = w + rearrange(torch.linspace(1.38, 1.18, p), "p -> p 1 1") * torch.eye(r)

    hh_up[:, : 2 * hidden, :].mul_(0.10)
    hh_up[:, 2 * hidden : 3 * hidden, :].copy_(torch.bmm(q, s))

    for k, scale in (
        ("net_cell_linear_ih", 0.12),
        ("net_encoder_0", 0.14),
        ("net_encoder_2", 0.14),
        ("net_decoder_0", 0.50),
    ):
        if k in pop.weight_up:
            pop.weight_up[k].data.mul_(scale)

    if "net_decoder_2" in pop.weight_up:
        pop.weight_up["net_decoder_2"].data.mul_(
            rearrange(torch.linspace(0.18, 0.44, p), "p -> p 1 1")
        )


@torch.no_grad()
def _quantize_pop_f16_roundtrip_(pop: Population) -> None:
    """Roundtrip LoRA weights through float16 in-place so server & browser share identical numbers."""
    for key in pop.weight_down:
        wd = pop.weight_down[key].data
        wu = pop.weight_up[key].data
        wd.copy_(wd.to(torch.float16).to(wd.dtype))
        wu.copy_(wu.to(torch.float16).to(wu.dtype))


@torch.no_grad()
def _individual_to_genome_dict(
    pop: Population,
    idx: int,
    *,
    task_id: str,
    base_hash: str,
    base_seed: int,
    genome_id: int | None = None,
    parents: list[int] | None = None,
    op: str = "random",
) -> dict[str, object]:
    adapters: dict[str, dict[str, object]] = {}
    for path in pop.lora_targets:
        key = path.replace(".", "_")
        wd = pop.weight_down[key].data[idx]
        wu = pop.weight_up[key].data[idx]
        delta = einsum(wu.float(), wd.float(), "o r, i r -> o i")
        frob = float(delta.norm().item())
        adapters[key] = {
            "down": _encode_f16_b64(wd),
            "up": _encode_f16_b64(wu),
            "inDim": int(wd.shape[0]),
            "outDim": int(wu.shape[0]),
            "norm": round(frob, 4),
        }
    out: dict[str, object] = {
        "id": int(genome_id if genome_id is not None else _alloc_genome_id()),
        "kind": "populora",
        "task": task_id,
        "parents": [int(p) for p in (parents or [])],
        "op": op,
        "baseHash": base_hash,
        "baseSeed": int(base_seed),
        "lowRank": int(pop.low_rank),
        "adapters": adapters,
    }
    out["hash"] = populora_genome_hash(out, length=20)
    return out


@torch.no_grad()
def _load_genomes_into_population(
    genome_dicts: Sequence[dict],
    policy_spec: dict[str, object],
) -> tuple[Population, str, int]:
    if not genome_dicts:
        raise ValueError("Cannot reconstruct empty PopuLoRA population.")
    first = genome_dicts[0]
    base_seed = int(first.get("baseSeed", 1))
    base_hash = str(first.get("baseHash") or "")
    low_rank = int(first.get("lowRank") or policy_spec.get("lowRank", 8))
    lora_targets = tuple(policy_spec.get("loraTargets") or ())

    base_model, resolved_hash, _ = get_or_create_base_model(
        policy_spec, base_seed=base_seed, base_hash=base_hash or None
    )
    pop = Population(
        base_model,
        pop_size=len(genome_dicts),
        low_rank=low_rank,
        lora_targets=lora_targets,
    )
    for idx, g in enumerate(genome_dicts):
        adapters = g.get("adapters") or {}
        for path in lora_targets:
            key = path.replace(".", "_")
            entry = adapters.get(key)
            if not isinstance(entry, dict):
                raise ValueError(f"Missing LoRA adapter '{key}' in genome #{idx}.")
            wd_target = pop.weight_down[key].data[idx]
            wu_target = pop.weight_up[key].data[idx]
            wd_target.copy_(_decode_f16_b64(str(entry["down"]), tuple(wd_target.shape)))
            wu_target.copy_(_decode_f16_b64(str(entry["up"]), tuple(wu_target.shape)))
    return pop, resolved_hash, base_seed


def validate_populora_genome_dict(raw: object, policy_spec: dict[str, object] | None = None) -> dict:
    """Validate an untrusted PopuLoRA genome dictionary."""
    if not isinstance(raw, dict) or raw.get("kind") != "populora":
        raise ValueError("Invalid PopuLoRA genome payload.")
    adapters_raw = raw.get("adapters")
    if not isinstance(adapters_raw, dict) or not adapters_raw:
        raise ValueError("PopuLoRA genome missing adapters.")
    low_rank = int(raw.get("lowRank", 8))
    if not (1 <= low_rank <= 64):
        raise ValueError("Invalid PopuLoRA lowRank.")

    expected_shapes: dict[str, tuple[int, int]] = {}
    if policy_spec is not None:
        base_model = build_policy_module(policy_spec, base_seed=int(raw.get("baseSeed", 1)))
        for path in policy_spec.get("loraTargets") or ():
            lin = base_model.get_submodule(str(path))
            expected_shapes[str(path).replace(".", "_")] = (int(lin.in_features), int(lin.out_features))
        if set(adapters_raw.keys()) != set(expected_shapes.keys()):
            raise ValueError("PopuLoRA adapter keys do not match task policy specification.")

    clean_adapters: dict[str, dict[str, object]] = {}
    for key, val in adapters_raw.items():
        if not isinstance(val, dict):
            raise ValueError(f"Invalid adapter entry '{key}'.")
        if key in expected_shapes:
            in_dim, out_dim = expected_shapes[key]
        else:
            in_dim = int(val.get("inDim", 0))
            out_dim = int(val.get("outDim", 0))
        if in_dim <= 0 or out_dim <= 0 or in_dim > 2048 or out_dim > 2048:
            raise ValueError(f"Invalid adapter dimensions on '{key}'.")
        wd = _decode_f16_b64(str(val.get("down", "")), (in_dim, low_rank))
        wu = _decode_f16_b64(str(val.get("up", "")), (out_dim, low_rank))
        delta = einsum(wu.float(), wd.float(), "o r, i r -> o i")
        clean_adapters[str(key)] = {
            "down": _encode_f16_b64(wd),
            "up": _encode_f16_b64(wu),
            "inDim": in_dim,
            "outDim": out_dim,
            "norm": round(float(delta.norm().item()), 4),
        }

    out: dict[str, object] = {
        "id": int(raw.get("id", 0) or 0),
        "kind": "populora",
        "task": str(raw.get("task") or "microduck"),
        "parents": [int(p) for p in raw.get("parents", []) if isinstance(p, int)],
        "op": str(raw.get("op") or "random"),
        "baseHash": str(raw.get("baseHash") or "")[:32],
        "baseSeed": int(raw.get("baseSeed", 1)),
        "lowRank": low_rank,
        "adapters": clean_adapters,
    }
    out["hash"] = populora_genome_hash(out, length=20)
    return out


@torch.no_grad()
def create_populora_population(
    task_id: str,
    policy_spec: dict[str, object],
    pop_size: int = 8,
    *,
    base_seed: int = 1,
    seed_genomes: Sequence[dict] | None = None,
) -> list[dict[str, object]]:
    """Create a Gen-0 PopuLoRA population (optionally seeded with uploaded strains)."""
    eff_pop = max(3, int(pop_size))
    low_rank = int(policy_spec.get("lowRank", 8))
    lora_targets = tuple(policy_spec.get("loraTargets") or ())

    custom_state = None
    if seed_genomes and len(seed_genomes) == 1 and isinstance(seed_genomes[0].get("_warmstart_state"), dict):
        custom_state = seed_genomes[0]["_warmstart_state"]

    base_model, base_hash, _ = get_or_create_base_model(
        policy_spec, base_seed=base_seed, custom_state_dict=custom_state
    )
    pop = Population(
        base_model,
        pop_size=eff_pop,
        low_rank=low_rank,
        lora_targets=lora_targets,
    )
    if policy_spec.get("oscillatoryInit"):
        _seed_oscillatory_population_(pop, has_warmstart=custom_state is not None)
    else:
        std_up = 0.04 if custom_state is not None else float(policy_spec.get("initStdUp", 0.35))
        pop.repopulate_(save_champion=False, std_down=None, std_up=std_up)

    seeded_slots = 0
    if seed_genomes:
        for s_idx, sg in enumerate(seed_genomes[:eff_pop]):
            if sg.get("kind") == "populora" and isinstance(sg.get("adapters"), dict):
                for path in lora_targets:
                    key = path.replace(".", "_")
                    ent = sg["adapters"].get(key)
                    if isinstance(ent, dict):
                        wd_t = pop.weight_down[key].data[s_idx]
                        wu_t = pop.weight_up[key].data[s_idx]
                        wd_t.copy_(_decode_f16_b64(str(ent["down"]), tuple(wd_t.shape)))
                        wu_t.copy_(_decode_f16_b64(str(ent["up"]), tuple(wu_t.shape)))
                seeded_slots += 1
            elif custom_state is not None and s_idx == 0:
                for key in pop.weight_up:
                    pop.weight_up[key].data[0].zero_()
                seeded_slots += 1

    _quantize_pop_f16_roundtrip_(pop)
    return [
        _individual_to_genome_dict(
            pop,
            i,
            task_id=task_id,
            base_hash=base_hash,
            base_seed=base_seed,
            op="seed" if i < seeded_slots else "random",
        )
        for i in range(eff_pop)
    ]


@dataclass
class _GoalPopuloraRuntime:
    aurora: Aurora | None = None
    aurora_feat_dim: int = 0
    eps_schedule: CosineAnnealingSchedule = field(
        default_factory=lambda: CosineAnnealingSchedule(eps_min=0.005, eps_max=0.015, period=35, decay=0.82)
    )
    best_fitness: float = float("-inf")
    champ_backup: dict | None = None
    champ_parent_id: int = 0


_RUNTIME_LOCK = threading.Lock()
_GOAL_RUNTIMES: dict[tuple[str, str], _GoalPopuloraRuntime] = {}


def reset_populora_goal_runtime(room: str, goal: str) -> None:
    with _RUNTIME_LOCK:
        _GOAL_RUNTIMES.pop((room, goal), None)


def _get_goal_runtime(room: str, goal: str) -> _GoalPopuloraRuntime:
    key = (room, goal)
    with _RUNTIME_LOCK:
        rt = _GOAL_RUNTIMES.get(key)
        if rt is None:
            rt = _GoalPopuloraRuntime()
            _GOAL_RUNTIMES[key] = rt
        return rt


@torch.no_grad()
def evolve_populora_population(
    *,
    room: str,
    goal: str,
    task_id: str,
    generation: int,
    policy_spec: dict[str, object],
    genome_dicts: Iterable[dict],
    scores: list[int],
    up_counts: list[int],
    eval_by_hash: dict[str, dict] | None = None,
    mut_sigma: float = 0.30,
    pop_size: int = 8,
    elite_frac: float = 0.25,
    max_horizon: int = 400,
) -> tuple[list[dict[str, object]], int]:
    """Evolve a PopuLoRA population using crowdsourced WASM fitnesses, AURORA DNS, Firefly PSO, & votes."""
    genomes = list(genome_dicts)
    cur_n = len(genomes)
    target_pop = max(3, int(pop_size))
    if cur_n != target_pop:
        genomes = resize_populora_population(
            genomes,
            target_pop,
            policy_spec=policy_spec,
            task_id=task_id,
            scores=scores,
            up_counts=up_counts,
        )
        scores = (list(scores) + [0] * target_pop)[:target_pop]
        up_counts = (list(up_counts) + [0] * target_pop)[:target_pop]
        cur_n = target_pop

    pop, base_hash, base_seed = _load_genomes_into_population(genomes, policy_spec)
    pop.generation = int(generation)
    rt = _get_goal_runtime(room, goal)

    eval_map = eval_by_hash or {}
    sim_fits = torch.zeros(cur_n, dtype=torch.float32)
    has_sim = torch.zeros(cur_n, dtype=torch.bool)
    traj_tensors: list[Tensor] = []
    inferred_feat_dim = 20

    for i, g in enumerate(genomes):
        gh = str(g.get("hash") or populora_genome_hash(g, length=20))
        rec = eval_map.get(gh)
        if isinstance(rec, dict):
            summ = rec.get("summary") or {}
            ret_val = summ.get("return", summ.get("primaryValue", 0.0))
            try:
                fval = float(ret_val)
            except (TypeError, ValueError):
                fval = 0.0
            if math.isfinite(fval):
                sim_fits[i] = fval
                has_sim[i] = True
            raw_traj = rec.get("trajSample")
            if isinstance(raw_traj, list) and raw_traj and isinstance(raw_traj[0], list):
                arr = np.asarray(raw_traj, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[0] >= 1 and arr.shape[1] >= 2 and np.isfinite(arr).all():
                    inferred_feat_dim = int(arr.shape[1])
                    traj_tensors.append(torch.from_numpy(arr))
                    continue
        traj_tensors.append(torch.empty((0, 0), dtype=torch.float32))

    # Normalize empty fallback trajectories to uniform feature dimension
    traj_tensors = [
        t if (t.ndim == 2 and t.shape[1] == inferred_feat_dim and t.shape[0] > 0)
        else torch.zeros((4, inferred_feat_dim), dtype=torch.float32)
        for t in traj_tensors
    ]

    # Blend crowdsourced simulation fitness with human votes (when human votes were cast)
    vote_t = torch.tensor([float(s) for s in scores[:cur_n]], dtype=torch.float32)
    has_votes = bool((vote_t != 0).any().item())

    if has_sim.any():
        if not has_sim.all():
            floor_val = float(sim_fits[has_sim].min().item()) - 25.0
            sim_fits[~has_sim] = floor_val
        fits = sim_fits.clone()
        if has_votes:
            sim_std = float(sim_fits.std().item()) if cur_n > 1 else 0.0
            vote_weight = max(25.0, 0.75 * sim_std)
            fits = fits + vote_weight * vote_t
    else:
        fits = vote_t * 50.0 + torch.tensor([float(u) * 5.0 for u in up_counts[:cur_n]], dtype=torch.float32)

    # Behavioral descriptors via PopuLoRA AURORA (with sketch_trajectories fallback)
    has_real_trajs = bool(has_sim.any())
    if has_real_trajs:
        try:
            if rt.aurora is None or rt.aurora_feat_dim != inferred_feat_dim:
                rt.aurora = Aurora(
                    latent_dim=min(8, max(4, cur_n)),
                    hidden_dim=64,
                    batch_size=min(64, cur_n),
                    update_steps=4,
                    max_horizon=max_horizon,
                )
                rt.aurora_feat_dim = inferred_feat_dim
            with preserve_rng():
                rt.aurora.fit(traj_tensors)
                descriptors = rt.aurora.encode(traj_tensors).to(pop.device)
        except Exception:
            descriptors = sketch_trajectories(traj_tensors, max_horizon=max_horizon, device=pop.device)
    else:
        descriptors = None

    # Update champion backup before evolving so best-ever individual is never lost
    best_idx = int(fits.argmax().item())
    best_fit = float(fits[best_idx].item())
    if has_sim[best_idx] and best_fit > rt.best_fitness:
        rt.best_fitness = best_fit
        rt.champ_backup = pop.backup_individual(best_idx)
        rt.champ_parent_id = int(genomes[best_idx]["id"])

    old_ids = [int(g["id"]) for g in genomes]
    ranked_old = [old_ids[i] for i in fits.argsort(descending=True).tolist()]

    use_dns = descriptors is not None and cur_n >= 4
    comp_fits = (
        quality_dominated_novelty(
            fits,
            descriptors,
            k=min(5, cur_n - 1),
            weight=0.65,
            quantile=0.30,
            sharpness=2.0,
        )
        if use_dns
        else None
    )

    clamped_elite = max(0.0, min(0.5, float(elite_frac)))
    n_elites = max(1, min(cur_n - 1, round(cur_n * clamped_elite))) if clamped_elite > 0 else 1
    # Scale PopuLoRA epsilon schedule gently by room mutationSigma knob (default 0.30 -> 1.0x)
    sigma_mult = max(0.25, min(3.0, float(mut_sigma) / 0.30))
    eff_eps = float(rt.eps_schedule(generation)) * sigma_mult

    sel = pop.evolve_(
        fits,
        competition_fitnesses=comp_fits,
        descriptors=descriptors if use_dns else None,
        raw_elites=n_elites if use_dns else None,
        survive_frac=0.70 if cur_n >= 5 else 0.50,
        elite_frac=clamped_elite if clamped_elite > 0 else (1.0 / cur_n),
        num_elites=n_elites,
        selection_type="deterministic",
        parent_selection_type="assortative" if use_dns else "tournament",
        tournament_size=min(4, max(2, cur_n // 2)),
        niche_k=min(8, max(1, cur_n - 1)),
        crossover_type="extrapolative",
        eta_min=0.20,
        eta_max=1.25,
        mutation_type="full_gaussian",
        epsilon=eff_eps,
        weight_decay=1e-4,
        consolidate_alpha=0.0,
        swarm_type="firefly" if cur_n >= 4 else None,
        swarm_bottom_frac=0.60 if use_dns else 1.0,
        swarm_kwargs=dict(
            beta0=0.50,
            gamma=1.0,
            alpha=0.035 * sigma_mult,
            align_lora_sign=True,
            align_lora_procrustes=True,
            preserve_norm=True,
            normalize_weights=True,
            preserve_distance_decay=True,
            fitness_weighted=True,
            scale_invariant_distance=True,
        ),
    )

    elite_set = set(sel.elites.tolist())
    culled_set = set(sel.culled.tolist())
    top_elite_id = old_ids[sel.elites[0].item()] if len(sel.elites) > 0 else ranked_old[0]
    breed_parents = ranked_old[:2] if len(ranked_old) >= 2 else [ranked_old[0]]

    champ_in_elites = False
    if isinstance(rt.champ_backup, dict) and "down" in rt.champ_backup and "up" in rt.champ_backup:
        champ_down = rt.champ_backup["down"]
        champ_up = rt.champ_backup["up"]
        for e_slot in elite_set:
            if all(
                torch.equal(pop.weight_down[k].data[e_slot], champ_down[k])
                and torch.equal(pop.weight_up[k].data[e_slot], champ_up[k])
                for k in pop.weight_down
            ):
                champ_in_elites = True
                break

    champ_restored_slot0 = False
    if rt.champ_backup is not None and not champ_in_elites and 0 not in elite_set:
        pop.restore_individual(0, rt.champ_backup)
        champ_restored_slot0 = True

    _quantize_pop_f16_roundtrip_(pop)

    next_genomes: list[dict[str, object]] = []
    for slot in range(cur_n):
        if champ_restored_slot0 and slot == 0:
            parents, op = [rt.champ_parent_id], "elite"
        elif slot in elite_set:
            parents, op = [old_ids[slot]], "elite"
        elif slot in culled_set:
            parents, op = list(breed_parents), "crossover"
        else:
            parents = [old_ids[slot]] if old_ids[slot] == top_elite_id else [old_ids[slot], top_elite_id]
            op = "mutation"
        next_genomes.append(
            _individual_to_genome_dict(
                pop,
                slot,
                task_id=task_id,
                base_hash=base_hash,
                base_seed=base_seed,
                parents=parents,
                op=op,
            )
        )

    top_score = int(round(best_fit)) if has_sim.any() else (max(scores) if scores else 0)
    return next_genomes, top_score


@torch.no_grad()
def resize_populora_population(
    genome_dicts: Iterable[dict],
    target_size: int,
    *,
    policy_spec: dict[str, object],
    task_id: str,
    scores: list[int] | None = None,
    up_counts: list[int] | None = None,
) -> list[dict[str, object]]:
    existing = list(genome_dicts)
    target = max(3, int(target_size))
    if target <= len(existing):
        return existing[:target]
    if not existing:
        return create_populora_population(task_id, policy_spec, pop_size=target)

    extra_needed = target - len(existing)
    fresh = create_populora_population(
        task_id,
        policy_spec,
        pop_size=max(3, extra_needed),
        base_seed=int(existing[0].get("baseSeed", 1)),
    )
    return existing + fresh[:extra_needed]


@torch.no_grad()
def save_populora_checkpoint_bytes(genome: dict, policy_spec: dict[str, object]) -> bytes:
    """Export a merged PyTorch state_dict ``.pt`` compatible with ``populora/train_microduck.py``."""
    pop, _, _ = _load_genomes_into_population([genome], policy_spec)
    _, (w_down, w_up) = pop.individual_weights(0)
    state = {k: v.detach().cpu().clone() for k, v in pop.model.state_dict().items()}
    for path in pop.lora_targets:
        key = path.replace(".", "_")
        delta = einsum(w_up[key].float(), w_down[key].float(), "o r, i r -> o i")
        w = state[f"{path}.weight"]
        w.add_(delta.to(w.dtype))
    buf = io.BytesIO()
    torch.save(state, buf)
    return buf.getvalue()


@torch.no_grad()
def load_populora_checkpoint_bytes(
    raw_bytes: bytes,
    policy_spec: dict[str, object],
    *,
    task_id: str = "microduck",
    source_name: str = "weights.pt",
    assign_id: bool = True,
) -> dict[str, object]:
    """Load a merged ``.pt`` policy checkpoint and project its residual onto a rank-``r`` PopuLoRA genome."""
    if not raw_bytes:
        raise ValueError(f"{source_name}: Checkpoint file is empty.")
    if len(raw_bytes) > MAX_POPULORA_CHECKPOINT_BYTES:
        raise ValueError(f"{source_name}: Checkpoint exceeds {MAX_POPULORA_CHECKPOINT_BYTES // (1024 * 1024)} MB limit.")

    try:
        state = torch.load(io.BytesIO(raw_bytes), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"{source_name}: Not a valid PyTorch weights_only checkpoint.") from exc

    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    if not isinstance(state, dict):
        raise ValueError(f"{source_name}: Expected a PyTorch state_dict.")

    base_model, base_hash, _ = get_or_create_base_model(policy_spec, base_seed=1)
    expected = base_model.state_dict()
    lora_targets = tuple(policy_spec.get("loraTargets") or ())
    low_rank = int(policy_spec.get("lowRank", 8))

    for path in lora_targets:
        w_key = f"{path}.weight"
        if w_key not in state or tuple(state[w_key].shape) != tuple(expected[w_key].shape):
            raise ValueError(f"{source_name}: Missing or incompatible tensor '{w_key}'.")

    pop = Population(base_model, pop_size=1, low_rank=low_rank, lora_targets=lora_targets)
    for path in lora_targets:
        key = path.replace(".", "_")
        w_key = f"{path}.weight"
        delta = (state[w_key].float() - expected[w_key].float())
        u, s, vh = torch.linalg.svd(delta, full_matrices=False)
        r_eff = min(low_rank, s.shape[0])
        sqrt_s = s[:r_eff].clamp(min=0.0).sqrt()
        pop.weight_up[key].data[0].zero_()
        pop.weight_down[key].data[0].zero_()
        pop.weight_up[key].data[0, :, :r_eff].copy_(u[:, :r_eff] * sqrt_s.unsqueeze(0))
        pop.weight_down[key].data[0, :, :r_eff].copy_(vh[:r_eff, :].T * sqrt_s.unsqueeze(0))

    _quantize_pop_f16_roundtrip_(pop)
    return _individual_to_genome_dict(
        pop,
        0,
        task_id=task_id,
        base_hash=base_hash,
        base_seed=1,
        genome_id=None if assign_id else 0,
        op="seed",
    )
