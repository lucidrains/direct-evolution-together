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
    quality_gated_fitness,
    sketch_trajectories,
)
from populora._utils import default, exists
from populora.operators import _efficient_svd_of_lora
import torch
from torch import Tensor
import torch.nn as nn
from torch_einops_utils import z_score

from direct_evolution_together.cppn import _alloc_genome_id, alloc_short_uuid, normalize_short_uuid

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
        *,
        hh_init_scale: float = 0.08,
        reset_bias_init: float = 0.8,
        update_bias_bounds: tuple[float, float] = (0.3, 1.3),
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
            self.cell.linear_hh.weight.mul_(hh_init_scale)
            self.cell.linear_ih.bias.zero_()
            self.cell.linear_ih.bias[:hidden_dim].fill_(reset_bias_init)
            self.cell.linear_ih.bias[hidden_dim : 2 * hidden_dim].uniform_(*update_bias_bounds)
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


# serialization & tensor helpers


def _encode_f16_b64(tensor: Tensor) -> str:
    arr = np.ascontiguousarray(tensor.detach().cpu().numpy().astype("<f2"))
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _encode_f16_batch_b64(tensor: Tensor) -> list[str]:
    # encode all (p, d, r) float16 matrices in a single C base64 pass via RFC-4648 3-byte alignment
    raw = np.ascontiguousarray(tensor.detach().cpu().numpy().astype("<f2")).view(np.uint8)
    raw = raw.reshape(raw.shape[0], -1)
    p, n_bytes = raw.shape
    pad = (3 - n_bytes % 3) % 3
    if pad:
        raw = np.pad(raw, ((0, 0), (0, pad)), mode="constant")
    stride = 4 * (raw.shape[1] // 3)
    keep = stride - pad
    suffix = "=" * pad
    joined = base64.b64encode(raw.tobytes()).decode("ascii")
    return [joined[i * stride : i * stride + keep] + suffix for i in range(p)]


def _decode_f16_batch_b64(b64_list: Sequence[str], shape: tuple[int, int]) -> Tensor:
    # decode all p float16 adapter matrices in a single C base64 pass
    p = len(b64_list)
    dim, r = shape
    n_bytes = dim * r * 2
    pad = (3 - n_bytes % 3) % 3
    stride = 4 * ((n_bytes + pad) // 3)
    suffix = "=" * pad

    for s in b64_list:
        if len(s) != stride or (pad and (not s.endswith(suffix) or "=" in s[:-pad])) or (not pad and "=" in s):
            raise ValueError("Invalid float16 base64 buffer.")

    joined = "".join(b64_list)
    if pad:
        joined = joined.replace("=", "A")

    raw = np.frombuffer(base64.b64decode(joined, validate=True), dtype=np.uint8)
    raw = np.ascontiguousarray(raw.reshape(p, n_bytes + pad)[:, :n_bytes])
    arr = raw.view("<f2").astype(np.float32).reshape(p, dim, r)
    if not np.isfinite(arr).all():
        raise ValueError("Adapter tensor contains non-finite values.")
    return torch.from_numpy(arr)


def _lora_frob_norms(wd: Tensor, wu: Tensor) -> Tensor:
    # cyclic trace identity: ||wu @ wd.T||_F^2 = <wu.T @ wu, wd.T @ wd>_F
    wu_f, wd_f = wu.float(), wd.float()
    return ((wu_f.transpose(-1, -2) @ wu_f) * (wd_f.transpose(-1, -2) @ wd_f)).sum(dim=(-2, -1)).clamp(min=0.0).sqrt()


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
    obs_dim = int(policy_spec["obs_dim"])
    act_dim = int(policy_spec["action_dim"])
    hidden_dim = int(policy_spec.get("hidden_dim", 128))
    head_dim = int(policy_spec.get("head_dim", hidden_dim))

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
        "base_hash": base_hash,
        "arch": str(policy_spec.get("arch") or "gru_mlp"),
        "obs_dim": int(policy_spec["obs_dim"]),
        "action_dim": int(policy_spec["action_dim"]),
        "hidden_dim": int(policy_spec.get("hidden_dim", 128)),
        "head_dim": int(policy_spec.get("head_dim", 128)),
        "low_rank": int(policy_spec.get("low_rank", 8)),
        "lora_targets": list(policy_spec.get("lora_targets") or []),
        "action_dist": str(policy_spec.get("action_dist") or "squashed_gaussian"),
        "action_bounds": list(policy_spec.get("action_bounds") or [-1.0, 1.0]),
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


def _persist_base_model_static_json(
    base_hash: str,
    payload: dict[str, object],
    *,
    compresslevel: int = 6,
) -> None:
    """Write `<MEDIA_ROOT>/base_models/<base_hash>.json(.gz)` so Nginx/media can serve it statically."""
    b_dir = _custom_base_dir()
    if b_dir is None or not base_hash:
        return
    json_path = b_dir / f"{base_hash}.json"
    if json_path.is_file():
        try:
            if b'"obs_dim"' in json_path.read_bytes():
                return
        except OSError:
            pass
    try:
        from direct_evolution_together.public_assets import write_gzip_sidecar

        doc = {"ok": True, **payload}
        raw = json.dumps(doc, separators=(",", ":")).encode("utf-8")
        tmp = json_path.with_name(json_path.name + ".tmp")
        tmp.write_bytes(raw)
        tmp.replace(json_path)
        write_gzip_sidecar(json_path, compresslevel=compresslevel)
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
    base_hash = str(genome.get("base_hash") or "")
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
def _rescale_init_up_(pop: Population, std_up: float) -> None:
    # Population.__init__ already draws w_down ~ N(0, d^-0.5) and w_up ~ N(0, r^-0.5)
    scale = float(std_up) * math.sqrt(pop.low_rank)
    for wu in pop.weight_up.values():
        wu.data.mul_(scale)


@torch.no_grad()
def _seed_oscillatory_population_(
    pop: Population,
    has_warmstart: bool = False,
    *,
    warmstart_std_up: float = 0.04,
    init_std_up: float = 1.0,
    noise_scale: float = 0.35,
    freq_range: tuple[float, float] = (3.6, 4.6),
    diag_range: tuple[float, float] = (1.38, 1.18),
    gate_scale: float = 0.10,
    ih_scale: float = 0.12,
    enc_scale: float = 0.14,
    dec_hidden_scale: float = 0.50,
    dec_out_range: tuple[float, float] = (0.18, 0.44),
    norm_eps: float = 1e-6,
) -> None:
    """Apply QR orthogonal recurrent basis + oscillatory skew-symmetric modes from train_microduck.py."""
    _rescale_init_up_(pop, warmstart_std_up if has_warmstart else init_std_up)
    if has_warmstart or "net_cell_linear_hh" not in pop.weight_down:
        return

    p, r = pop.pop_size, pop.low_rank
    hh_down = pop.weight_down["net_cell_linear_hh"].data
    hh_up = pop.weight_up["net_cell_linear_hh"].data
    hidden = hh_down.shape[1]
    if hh_up.shape[1] != 3 * hidden:
        return

    q, _ = torch.linalg.qr(hh_down)
    hh_down.copy_(q)

    g = torch.randn(1, r, r) + noise_scale * torch.randn(p, r, r)
    w = g - g.transpose(1, 2)
    w = w / w.norm(dim=(-2, -1), keepdim=True).clamp(min=norm_eps) * rearrange(
        torch.linspace(freq_range[0], freq_range[1], p), "p -> p 1 1"
    )
    s = w + rearrange(torch.linspace(diag_range[0], diag_range[1], p), "p -> p 1 1") * torch.eye(r)

    hh_up[:, : 2 * hidden, :].mul_(gate_scale)
    hh_up[:, 2 * hidden : 3 * hidden, :].copy_(torch.bmm(q, s))

    for k, scale in (
        ("net_cell_linear_ih", ih_scale),
        ("net_encoder_0", enc_scale),
        ("net_encoder_2", enc_scale),
        ("net_decoder_0", dec_hidden_scale),
    ):
        if k in pop.weight_up:
            pop.weight_up[k].data.mul_(scale)

    if "net_decoder_2" in pop.weight_up:
        pop.weight_up["net_decoder_2"].data.mul_(
            rearrange(torch.linspace(dec_out_range[0], dec_out_range[1], p), "p -> p 1 1")
        )


@torch.no_grad()
def _quantize_pop_f16_roundtrip_(pop: Population) -> None:
    """Roundtrip LoRA weights through float16 in-place so server & browser share identical numbers."""
    for key in pop.weight_down:
        wd = pop.weight_down[key].data
        wu = pop.weight_up[key].data
        wd.copy_(wd.to(torch.float16).to(wd.dtype))
        wu.copy_(wu.to(torch.float16).to(wu.dtype))


# lora gauge canonicalization & population serialization


@torch.no_grad()
def _canonicalize_lora_gauge_(pop: Population) -> None:
    # place each individual's LoRA factors into canonical singular-mode gauge in O(P),
    # preserving wu @ wd.T while eliminating O(F * E) pairwise Procrustes SVDs in Firefly
    if pop.pop_size <= 0:
        return

    for key, wd_param in pop.weight_down.items():
        wu_param = pop.weight_up[key]
        wd, wu = wd_param.data, wu_param.data
        u, s, v = _efficient_svd_of_lora(wd, wu)
        sign = torch.where(v.gather(1, v.abs().argmax(dim=1, keepdim=True)) >= 0.0, 1.0, -1.0)
        scale = sign * rearrange(s.clamp(min=0.0).sqrt(), "p r -> p 1 r")
        wd.copy_((u * scale).to(wd.dtype))
        wu.copy_((v * scale).to(wu.dtype))


@torch.no_grad()
def _population_to_genome_dicts(
    pop: Population,
    *,
    task_id: str,
    base_hash: str,
    base_seed: int,
    indices: Sequence[int] | None = None,
    genome_ids: Sequence[int | None] | None = None,
    uuids: Sequence[str | None] | None = None,
    parents_per_slot: Sequence[list[int]] | None = None,
    ops_per_slot: Sequence[str] | None = None,
    norm_digits: int = 4,
    hash_length: int = 20,
) -> list[dict[str, object]]:
    slots = list(default(indices, range(int(pop.pop_size))))
    if not slots:
        return []

    low_rank = int(pop.low_rank)
    idx_t = None if not exists(indices) else torch.as_tensor(slots, dtype=torch.long, device=pop.device)
    target_keys = [path.replace(".", "_") for path in pop.lora_targets]
    hash_key_order = sorted(target_keys)

    encoded_layers: dict[str, tuple[list[str], list[str], int, int, list[float]]] = {}
    for key in target_keys:
        wd = pop.weight_down[key].data
        wu = pop.weight_up[key].data
        if exists(idx_t):
            wd, wu = wd[idx_t], wu[idx_t]
        norms = _lora_frob_norms(wd, wu).cpu().tolist()
        encoded_layers[key] = (
            _encode_f16_batch_b64(wd),
            _encode_f16_batch_b64(wu),
            int(wd.shape[1]),
            int(wu.shape[1]),
            norms,
        )

    sha256 = hashlib.sha256
    out: list[dict[str, object]] = []

    for pos, _ in enumerate(slots):
        adapters = {
            key: {
                "down": downs[pos],
                "up": ups[pos],
                "in_dim": in_dim,
                "out_dim": out_dim,
                "norm": round(float(norms[pos]), norm_digits),
            }
            for key, (downs, ups, in_dim, out_dim, norms) in encoded_layers.items()
        }
        digest_input = "|".join(
            [base_hash, *(f"{hk}:{adapters[hk]['down']}:{adapters[hk]['up']}" for hk in hash_key_order)]
        )
        gid = genome_ids[pos] if exists(genome_ids) else None
        raw_uuid = uuids[pos] if exists(uuids) else None
        out.append({
            "id": int(gid if exists(gid) else _alloc_genome_id()),
            "uuid": normalize_short_uuid(raw_uuid) or alloc_short_uuid(),
            "kind": "populora",
            "task": task_id,
            "parents": [int(p) for p in parents_per_slot[pos]] if exists(parents_per_slot) else [],
            "op": str(ops_per_slot[pos]) if exists(ops_per_slot) else "random",
            "base_hash": base_hash,
            "base_seed": int(base_seed),
            "low_rank": low_rank,
            "adapters": adapters,
            "hash": sha256(digest_input.encode("ascii")).hexdigest()[:hash_length],
        })

    return out


@torch.no_grad()
def _individual_to_genome_dict(
    pop: Population,
    idx: int,
    *,
    task_id: str,
    base_hash: str,
    base_seed: int,
    genome_id: int | None = None,
    uuid: str | None = None,
    parents: list[int] | None = None,
    op: str = "random",
) -> dict[str, object]:
    return _population_to_genome_dicts(
        pop,
        task_id=task_id,
        base_hash=base_hash,
        base_seed=base_seed,
        indices=[idx],
        genome_ids=[genome_id],
        uuids=[uuid],
        parents_per_slot=[list(default(parents, []))],
        ops_per_slot=[op],
    )[0]


@torch.no_grad()
def _load_genomes_into_population(
    genome_dicts: Sequence[dict],
    policy_spec: dict[str, object],
    *,
    cached_pop: Population | None = None,
    cached_base_hash: str = "",
) -> tuple[Population, str, int]:
    if not genome_dicts:
        raise ValueError("Cannot reconstruct empty PopuLoRA population.")

    first = genome_dicts[0]
    base_seed = int(first.get("base_seed", 1))
    base_hash = str(first.get("base_hash") or "")
    low_rank = int(first.get("low_rank") or policy_spec.get("low_rank", 8))
    lora_targets = tuple(policy_spec.get("lora_targets") or ())
    p_cnt = len(genome_dicts)

    if (
        exists(cached_pop)
        and cached_pop.pop_size == p_cnt
        and cached_pop.low_rank == low_rank
        and tuple(cached_pop.lora_targets) == lora_targets
        and cached_base_hash
        and (not base_hash or base_hash == cached_base_hash)
    ):
        pop = cached_pop
        resolved_hash = cached_base_hash
    else:
        base_model, resolved_hash, _ = get_or_create_base_model(
            policy_spec, base_seed=base_seed, base_hash=base_hash or None
        )
        pop = Population(base_model, pop_size=p_cnt, low_rank=low_rank, lora_targets=lora_targets)

    for path in lora_targets:
        key = path.replace(".", "_")
        wd_all, wu_all = pop.weight_down[key].data, pop.weight_up[key].data
        in_dim, r_d = int(wd_all.shape[1]), int(wd_all.shape[2])
        out_dim, _ = int(wu_all.shape[1]), int(wu_all.shape[2])
        downs, ups = [], []

        for idx, g in enumerate(genome_dicts):
            adapters = g.get("adapters")
            entry = adapters.get(key) if isinstance(adapters, dict) else None
            if not isinstance(entry, dict):
                raise ValueError(f"Missing LoRA adapter '{key}' in genome #{idx}.")
            downs.append(str(entry["down"]))
            ups.append(str(entry["up"]))

        wd_all.copy_(_decode_f16_batch_b64(downs, (in_dim, r_d)))
        wu_all.copy_(_decode_f16_batch_b64(ups, (out_dim, r_d)))

    return pop, resolved_hash, base_seed


def validate_populora_genome_dict(
    raw: object,
    policy_spec: dict[str, object] | None = None,
    *,
    default_low_rank: int = 8,
    max_low_rank: int = 64,
    max_dim: int = 2048,
    norm_digits: int = 4,
    max_base_hash_len: int = 32,
    hash_length: int = 20,
) -> dict:
    """Validate an untrusted PopuLoRA genome dictionary."""
    if not isinstance(raw, dict) or raw.get("kind") != "populora":
        raise ValueError("Invalid PopuLoRA genome payload.")
    adapters_raw = raw.get("adapters")
    if not isinstance(adapters_raw, dict) or not adapters_raw:
        raise ValueError("PopuLoRA genome missing adapters.")
    low_rank = int(raw.get("low_rank", default_low_rank))
    if not (1 <= low_rank <= max_low_rank):
        raise ValueError("Invalid PopuLoRA lowRank.")

    expected_shapes: dict[str, tuple[int, int]] = {}
    if policy_spec is not None:
        base_model = build_policy_module(policy_spec, base_seed=int(raw.get("base_seed", 1)))
        for path in policy_spec.get("lora_targets") or ():
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
            in_dim = int(val.get("in_dim", 0))
            out_dim = int(val.get("out_dim", 0))
        if in_dim <= 0 or out_dim <= 0 or in_dim > max_dim or out_dim > max_dim:
            raise ValueError(f"Invalid adapter dimensions on '{key}'.")
        wd = _decode_f16_b64(str(val.get("down", "")), (in_dim, low_rank))
        wu = _decode_f16_b64(str(val.get("up", "")), (out_dim, low_rank))
        clean_adapters[str(key)] = {
            "down": _encode_f16_b64(wd),
            "up": _encode_f16_b64(wu),
            "in_dim": in_dim,
            "out_dim": out_dim,
            "norm": round(float(_lora_frob_norms(wd, wu).item()), norm_digits),
        }

    out: dict[str, object] = {
        "id": int(raw.get("id", 0) or 0),
        "uuid": normalize_short_uuid(raw.get("uuid")) or alloc_short_uuid(),
        "kind": "populora",
        "task": str(raw.get("task") or "microduck"),
        "parents": [int(p) for p in raw.get("parents", []) if isinstance(p, int)],
        "op": str(raw.get("op") or "random"),
        "base_hash": str(raw.get("base_hash") or "")[:max_base_hash_len],
        "base_seed": int(raw.get("base_seed", 1)),
        "low_rank": low_rank,
        "adapters": clean_adapters,
    }
    out["hash"] = populora_genome_hash(out, length=hash_length)
    return out


@torch.no_grad()
def create_populora_population(
    task_id: str,
    policy_spec: dict[str, object],
    pop_size: int = 8,
    *,
    base_seed: int = 1,
    seed_genomes: Sequence[dict] | None = None,
    min_pop_size: int = 3,
    default_low_rank: int = 8,
    warmstart_std_up: float = 0.04,
    default_init_std_up: float = 0.35,
) -> list[dict[str, object]]:
    """Create a Gen-0 PopuLoRA population (optionally seeded with uploaded strains)."""
    eff_pop = max(min_pop_size, int(pop_size))
    low_rank = int(policy_spec.get("low_rank", default_low_rank))
    lora_targets = tuple(policy_spec.get("lora_targets") or ())

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
    if policy_spec.get("oscillatory_init"):
        _seed_oscillatory_population_(pop, has_warmstart=exists(custom_state), warmstart_std_up=warmstart_std_up)
    else:
        _rescale_init_up_(
            pop,
            warmstart_std_up if exists(custom_state) else float(policy_spec.get("init_std_up", default_init_std_up)),
        )

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
    ops = ["seed" if i < seeded_slots else "random" for i in range(eff_pop)]
    return _population_to_genome_dicts(
        pop,
        task_id=task_id,
        base_hash=base_hash,
        base_seed=base_seed,
        ops_per_slot=ops,
    )


@dataclass
class _GoalPopuloraRuntime:
    pop: Population | None = None
    pop_base_hash: str = ""
    aurora: Aurora | None = None
    aurora_feat_dim: int = 0
    eps_min: float = 0.005
    eps_max: float = 0.015
    eps_period: int = 35
    eps_decay: float = 0.82
    eps_schedule: CosineAnnealingSchedule | None = None
    best_fitness: float = float("-inf")
    champ_backup: dict | None = None
    champ_parent_id: int = 0

    def __post_init__(self) -> None:
        if self.eps_schedule is None:
            self.eps_schedule = CosineAnnealingSchedule(
                eps_min=self.eps_min,
                eps_max=self.eps_max,
                period=self.eps_period,
                decay=self.eps_decay,
            )


_RUNTIME_LOCK = threading.Lock()
_GOAL_RUNTIMES: dict[tuple[str, str], _GoalPopuloraRuntime] = {}


def reset_populora_goal_runtime(room: str, goal: str) -> None:
    with _RUNTIME_LOCK:
        _GOAL_RUNTIMES.pop((room, goal), None)


def _get_goal_runtime(
    room: str,
    goal: str,
    *,
    eps_min: float = 0.005,
    eps_max: float = 0.015,
    eps_period: int = 35,
    eps_decay: float = 0.82,
) -> _GoalPopuloraRuntime:
    key = (room, goal)
    with _RUNTIME_LOCK:
        rt = _GOAL_RUNTIMES.get(key)
        if rt is None:
            rt = _GoalPopuloraRuntime(
                eps_min=eps_min,
                eps_max=eps_max,
                eps_period=eps_period,
                eps_decay=eps_decay,
            )
            _GOAL_RUNTIMES[key] = rt
        return rt


# dominated novelty search (tiled for 1k - 10k+)


@torch.no_grad()
def _sketch_population_trajectories(
    traj_tensors: list[Tensor],
    *,
    num_subsamples: int = 8,
    max_horizon: int = 400,
    device: torch.device | str | None = None,
) -> Tensor:
    # fast batched trajectory sketching when trajectories share a uniform (T, D) shape
    shapes = {tuple(t.shape) for t in traj_tensors}
    if len(shapes) == 1:
        t_len, _ = next(iter(shapes))
        if t_len >= 1:
            stacked = torch.stack(traj_tensors).float()
            idx = torch.linspace(0, max(t_len - 1, 0), steps=num_subsamples).round().long()
            sub = rearrange(stacked[:, idx], "p s d -> p (s d)")
            mean_s = stacked.mean(dim=1)
            std_s = stacked.std(dim=1, unbiased=False) if t_len > 1 else torch.zeros_like(mean_s)
            len_col = torch.full((stacked.shape[0], 1), t_len / max(float(max_horizon), 1.0))
            out = torch.cat((sub, mean_s, std_s, len_col), dim=-1)
            return out.to(device) if exists(device) else out

    return sketch_trajectories(traj_tensors, num_subsamples=num_subsamples, max_horizon=max_horizon, device=device)


@torch.no_grad()
def _tiled_quality_dominated_novelty(
    fitnesses: Tensor,
    descriptors: Tensor,
    *,
    k: int = 5,
    weight: float = 0.65,
    quantile: float = 0.30,
    sharpness: float = 2.0,
    chunk_size: int = 512,
    exact_threshold: int = 1024,
    non_dominated_tiebreak: float = 0.01,
    eps: float = 1e-6,
) -> Tensor:
    p = int(fitnesses.shape[0])
    if p <= exact_threshold:
        return quality_dominated_novelty(
            fitnesses,
            descriptors,
            k=min(k, max(1, p - 1)),
            weight=weight,
            quantile=quantile,
            sharpness=sharpness,
        )

    device = fitnesses.device
    desc = z_score(rearrange(descriptors.float(), "n ... -> n (...)").to(device), dim=0)
    f = fitnesses.float().to(device)
    eff_k = min(k, p - 1)

    dns = torch.zeros(p, dtype=torch.float32, device=device)
    dominated = torch.zeros(p, dtype=torch.bool, device=device)
    rank_idx = rearrange(torch.arange(eff_k, device=device), "k -> 1 k")

    for start in range(0, p, chunk_size):
        end = min(p, start + chunk_size)
        d_chunk = torch.cdist(desc[start:end], desc, p=2.0)
        fitter = rearrange(f, "j -> 1 j") > rearrange(f[start:end], "i -> i 1")
        num_fitter = fitter.sum(dim=-1)
        dominated[start:end] = num_fitter > 0
        nearest = d_chunk.masked_fill(~fitter, float("inf")).topk(k=eff_k, dim=-1, largest=False).values
        valid = rank_idx < rearrange(num_fitter, "i -> i 1")
        dns[start:end] = torch.where(valid, nearest, 0.0).sum(dim=-1) / num_fitter.clamp(min=1, max=eff_k).float()

    if dominated.any():
        dom_vals = dns[dominated]
        max_dom, span = dom_vals.max(), (dom_vals.max() - dom_vals.min()).clamp(min=1.0)
    else:
        max_dom = span = torch.tensor(1.0, device=device)

    f_norm = (f - f.min()) / f.std(unbiased=False).clamp(min=eps)
    nov = torch.where(dominated, dns, max_dom + span + 1.0 + non_dominated_tiebreak * f_norm)
    return quality_gated_fitness(f, nov, weight=weight, quantile=quantile, sharpness=sharpness)


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
    min_pop_size: int = 3,
    max_elite_frac: float = 0.50,
    hash_length: int = 20,
    default_traj_dim: int = 20,
    default_traj_steps: int = 4,
    unsimulated_penalty: float = 25.0,
    min_vote_weight: float = 25.0,
    vote_std_scale: float = 0.75,
    fallback_vote_scale: float = 50.0,
    fallback_up_scale: float = 5.0,
    aurora_latent_bounds: tuple[int, int] = (4, 8),
    aurora_hidden_dim: int = 64,
    aurora_buf_bounds: tuple[int, int] = (512, 4096),
    aurora_batch_small: int = 64,
    aurora_batch_large: int = 128,
    aurora_update_steps: int = 4,
    large_pop_threshold: int = 64,
    exact_dns_threshold: int = 1024,
    min_dns_pop: int = 4,
    dns_k_small: int = 5,
    dns_k_large: int = 8,
    dns_weight: float = 0.65,
    dns_quantile: float = 0.30,
    dns_sharpness: float = 2.0,
    dns_chunk_size: int = 512,
    ref_mut_sigma: float = 0.30,
    sigma_mult_bounds: tuple[float, float] = (0.25, 3.0),
    survive_frac: float = 0.70,
    small_pop_survive_frac: float = 0.50,
    small_pop_cutoff: int = 5,
    tournament_size_small: int = 4,
    tournament_size_large: int = 6,
    niche_k_small: int = 8,
    niche_k_large: int = 12,
    eta_min: float = 0.20,
    eta_max: float = 1.25,
    weight_decay: float = 1e-4,
    consolidate_alpha: float = 0.0,
    swarm_bottom_dns: float = 0.60,
    swarm_beta0: float = 0.50,
    swarm_gamma: float = 1.0,
    swarm_alpha: float = 0.035,
    max_swarm_elites: int = 32,
) -> tuple[list[dict[str, object]], int]:
    """Evolve a PopuLoRA population using crowdsourced WASM fitnesses, AURORA DNS, Firefly PSO, & votes."""
    genomes = list(genome_dicts)
    cur_n = len(genomes)
    target_pop = max(min_pop_size, int(pop_size))
    if cur_n != target_pop:
        genomes = resize_populora_population(
            genomes,
            target_pop,
            policy_spec=policy_spec,
            task_id=task_id,
            scores=scores,
            up_counts=up_counts,
            min_pop_size=min_pop_size,
        )
        scores = (list(scores) + [0] * target_pop)[:target_pop]
        up_counts = (list(up_counts) + [0] * target_pop)[:target_pop]
        cur_n = target_pop

    rt = _get_goal_runtime(room, goal)
    pop, base_hash, base_seed = _load_genomes_into_population(
        genomes, policy_spec, cached_pop=rt.pop, cached_base_hash=rt.pop_base_hash
    )
    rt.pop = pop
    rt.pop_base_hash = base_hash
    pop.generation = int(generation)

    eval_map = eval_by_hash or {}
    sim_fits = torch.zeros(cur_n, dtype=torch.float32)
    has_sim = torch.zeros(cur_n, dtype=torch.bool)
    traj_tensors: list[Tensor] = []
    inferred_feat_dim = default_traj_dim

    for i, g in enumerate(genomes):
        gh = str(g.get("hash") or populora_genome_hash(g, length=hash_length))
        rec = eval_map.get(gh)
        if isinstance(rec, dict):
            summ = rec.get("summary") or {}
            ret_val = summ.get("return", summ.get("primary_value", 0.0))
            try:
                fval = float(ret_val)
            except (TypeError, ValueError):
                fval = 0.0
            if math.isfinite(fval):
                sim_fits[i] = fval
                has_sim[i] = True
            raw_traj = rec.get("traj_sample")
            if isinstance(raw_traj, list) and raw_traj and isinstance(raw_traj[0], list):
                arr = np.asarray(raw_traj, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[0] >= 1 and arr.shape[1] >= 2 and np.isfinite(arr).all():
                    inferred_feat_dim = int(arr.shape[1])
                    traj_tensors.append(torch.from_numpy(arr))
                    continue
        traj_tensors.append(torch.empty((0, 0), dtype=torch.float32))

    # Normalize empty fallback trajectories to uniform feature dimension
    default_t_len = next(
        (t.shape[0] for t in traj_tensors if t.ndim == 2 and t.shape[1] == inferred_feat_dim and t.shape[0] > 0),
        default_traj_steps,
    )
    traj_tensors = [
        t if (t.ndim == 2 and t.shape[1] == inferred_feat_dim and t.shape[0] > 0)
        else torch.zeros((default_t_len, inferred_feat_dim), dtype=torch.float32)
        for t in traj_tensors
    ]

    # Blend crowdsourced simulation fitness with human votes (when human votes were cast)
    vote_t = torch.tensor([float(s) for s in scores[:cur_n]], dtype=torch.float32)
    has_votes = bool((vote_t != 0).any().item())

    if has_sim.any():
        if not has_sim.all():
            floor_val = float(sim_fits[has_sim].min().item()) - unsimulated_penalty
            sim_fits[~has_sim] = floor_val
        fits = sim_fits.clone()
        if has_votes:
            sim_std = float(sim_fits.std().item()) if cur_n > 1 else 0.0
            vote_weight = max(min_vote_weight, vote_std_scale * sim_std)
            fits = fits + vote_weight * vote_t
    else:
        fits = (
            vote_t * fallback_vote_scale
            + torch.tensor([float(u) * fallback_up_scale for u in up_counts[:cur_n]], dtype=torch.float32)
        )

    large_pop_mode = cur_n > large_pop_threshold

    # Behavioral descriptors via PopuLoRA AURORA (with sketch_trajectories fallback)
    if bool(has_sim.any()):
        sketches = _sketch_population_trajectories(traj_tensors, max_horizon=max_horizon, device=pop.device)
        try:
            if not exists(rt.aurora) or rt.aurora_feat_dim != inferred_feat_dim:
                lat_lo, lat_hi = aurora_latent_bounds
                buf_lo, buf_hi = aurora_buf_bounds
                rt.aurora = Aurora(
                    latent_dim=min(lat_hi, max(lat_lo, cur_n)),
                    hidden_dim=aurora_hidden_dim,
                    buffer_capacity=max(buf_lo, min(buf_hi, cur_n)),
                    batch_size=min(aurora_batch_large if large_pop_mode else aurora_batch_small, cur_n),
                    update_steps=aurora_update_steps,
                    max_horizon=max_horizon,
                )
                rt.aurora_feat_dim = inferred_feat_dim
            with preserve_rng():
                rt.aurora.fit(sketches)
                descriptors = rt.aurora.encode(sketches).to(pop.device)
        except Exception:
            descriptors = sketches
    else:
        descriptors = None

    if large_pop_mode:
        _canonicalize_lora_gauge_(pop)

    # Update champion backup before evolving so best-ever individual is never lost
    best_idx = int(fits.argmax().item())
    best_fit = float(fits[best_idx].item())
    if has_sim[best_idx] and best_fit > rt.best_fitness:
        rt.best_fitness = best_fit
        rt.champ_backup = pop.backup_individual(best_idx)
        rt.champ_parent_id = int(genomes[best_idx]["id"])

    old_ids = [int(g["id"]) for g in genomes]
    ranked_old = [old_ids[i] for i in fits.argsort(descending=True).tolist()]

    use_dns = descriptors is not None and cur_n >= min_dns_pop
    comp_fits = (
        _tiled_quality_dominated_novelty(
            fits,
            descriptors,
            k=min(dns_k_large if large_pop_mode else dns_k_small, cur_n - 1),
            weight=dns_weight,
            quantile=dns_quantile,
            sharpness=dns_sharpness,
            chunk_size=dns_chunk_size,
            exact_threshold=exact_dns_threshold,
        )
        if use_dns
        else None
    )

    clamped_elite = max(0.0, min(max_elite_frac, float(elite_frac)))
    n_elites = max(1, min(cur_n - 1, round(cur_n * clamped_elite))) if clamped_elite > 0 else 1
    assert rt.eps_schedule is not None
    mult_lo, mult_hi = sigma_mult_bounds
    sigma_mult = max(mult_lo, min(mult_hi, float(mut_sigma) / max(ref_mut_sigma, 1e-6)))
    eff_eps = float(rt.eps_schedule(generation)) * sigma_mult
    swarm_bottom = swarm_bottom_dns if use_dns else 1.0
    swarm_cfg = dict(
        beta0=swarm_beta0,
        gamma=swarm_gamma,
        alpha=swarm_alpha * sigma_mult,
        align_lora_sign=True,
        align_lora_procrustes=not large_pop_mode,
        preserve_norm=True,
        normalize_weights=True,
        preserve_distance_decay=True,
        fitness_weighted=True,
        scale_invariant_distance=True,
    )

    sel = pop.evolve_(
        fits,
        competition_fitnesses=comp_fits,
        descriptors=descriptors if (use_dns and cur_n <= exact_dns_threshold) else None,
        raw_elites=n_elites if use_dns else None,
        survive_frac=survive_frac if cur_n >= small_pop_cutoff else small_pop_survive_frac,
        elite_frac=clamped_elite if clamped_elite > 0 else (1.0 / cur_n),
        num_elites=n_elites,
        selection_type="deterministic",
        parent_selection_type="assortative" if (use_dns and cur_n <= exact_dns_threshold) else "tournament",
        tournament_size=min(tournament_size_large if large_pop_mode else tournament_size_small, max(2, cur_n // 2)),
        niche_k=min(niche_k_large if large_pop_mode else niche_k_small, max(1, cur_n - 1)),
        crossover_type="extrapolative",
        eta_min=eta_min,
        eta_max=eta_max,
        mutation_type="full_gaussian",
        epsilon=eff_eps,
        weight_decay=weight_decay,
        consolidate_alpha=consolidate_alpha,
        swarm_type=None if large_pop_mode else ("firefly" if cur_n >= min_dns_pop else None),
        swarm_bottom_frac=swarm_bottom,
        swarm_kwargs=swarm_cfg,
    )

    if large_pop_mode and cur_n >= min_dns_pop and len(sel.survivors) > len(sel.elites):
        non_elite_surv = sel.survivors[~torch.isin(sel.survivors, sel.elites)]
        if len(non_elite_surv) > 0:
            order = fits[non_elite_surv].argsort(descending=False)
            n_follow = max(1, int(round(len(non_elite_surv) * swarm_bottom)))
            followers = non_elite_surv[order[:n_follow]]
            swarm_elites = sel.elites[: min(max_swarm_elites, len(sel.elites))] if len(sel.elites) > 0 else sel.survivors[:1]
            pop.swarm_(
                "firefly",
                fits,
                individuals=followers,
                elites=swarm_elites,
                **swarm_cfg,
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

    parents_per_slot: list[list[int]] = []
    ops_per_slot: list[str] = []
    for slot in range(cur_n):
        if champ_restored_slot0 and slot == 0:
            parents_per_slot.append([rt.champ_parent_id])
            ops_per_slot.append("elite")
        elif slot in elite_set:
            parents_per_slot.append([old_ids[slot]])
            ops_per_slot.append("elite")
        elif slot in culled_set:
            parents_per_slot.append(list(breed_parents))
            ops_per_slot.append("crossover")
        else:
            parents_per_slot.append(
                [old_ids[slot]] if old_ids[slot] == top_elite_id else [old_ids[slot], top_elite_id]
            )
            ops_per_slot.append("mutation")

    next_genomes = _population_to_genome_dicts(
        pop,
        task_id=task_id,
        base_hash=base_hash,
        base_seed=base_seed,
        parents_per_slot=parents_per_slot,
        ops_per_slot=ops_per_slot,
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
    min_pop_size: int = 3,
) -> list[dict[str, object]]:
    existing = list(genome_dicts)
    target = max(min_pop_size, int(target_size))
    if target <= len(existing):
        return existing[:target]
    if not existing:
        return create_populora_population(task_id, policy_spec, pop_size=target, min_pop_size=min_pop_size)

    extra_needed = target - len(existing)
    fresh = create_populora_population(
        task_id,
        policy_spec,
        pop_size=max(min_pop_size, extra_needed),
        base_seed=int(existing[0].get("base_seed", 1)),
        min_pop_size=min_pop_size,
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
    lora_targets = tuple(policy_spec.get("lora_targets") or ())
    low_rank = int(policy_spec.get("low_rank", 8))

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
