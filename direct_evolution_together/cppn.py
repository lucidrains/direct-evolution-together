import hashlib
import io
import itertools
import math
from pathlib import Path, PurePosixPath
import re
import secrets
import tempfile
import threading
from typing import Iterable, Sequence
import zipfile

from populora.operators import parent_select_tournament
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import leaky_relu
from torch_einops_utils.save_load import save_load

LAYER_DIMS = (4, 12, 12, 12, 3)
NUM_WEIGHTS = sum(a * b for a, b in zip(LAYER_DIMS[:-1], LAYER_DIMS[1:]))  # 372
NUM_BIASES = sum(LAYER_DIMS[1:])  # 39
NUM_HIDDEN = sum(LAYER_DIMS[1:-1])  # 36
NUM_ACTIVATIONS = 5  # 0=tanh, 1=sin, 2=gauss, 3=abs, 4=leaky_relu
POPULATION_SIZE = 30
MIN_POPULATION_SIZE = 3
MAX_POPULATION_SIZE = 1024
SYSTEM_MAX_POPULATION_SIZE = 10_000
MAX_CHECKPOINT_BYTES = 128 * 1024  # 128 KB (~4.5 KB typical CPPN .pt)
MAX_CHECKPOINT_UNCOMPRESSED_BYTES = 256 * 1024  # 256 KB max internal .pt zip payload
MAX_PT_ARCHIVE_ENTRIES = 64
MAX_ARCHIVE_BYTES = 2 * 1024 * 1024  # 2 MB max .zip upload
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 4 * 1024 * 1024  # 4 MB max extracted .pt bytes per .zip
MAX_ZIP_ENTRIES = 256
MAX_TOTAL_UPLOAD_BYTES = 4 * 1024 * 1024  # 4 MB max total HTTP upload body
MAX_UPLOAD_FILES = 256
DEFAULT_MUT_SIGMA = 0.30
DEFAULT_ELITE_FRAC = 0.20
DEFAULT_CROSSOVER_RATE = 0.75
MIN_BROOD_SIZE = 1
MAX_BROOD_SIZE = 8
DEFAULT_BROOD_SIZE = 1

_counter_lock = threading.Lock()
_genome_counter = itertools.count(1)
SHORT_UUID_RE = re.compile(r"^[0-9a-f]{8}$")


def _alloc_genome_id() -> int:
    with _counter_lock:
        return next(_genome_counter)


def alloc_short_uuid() -> str:
    """Allocate a compact 8-character lowercase hex UUID without perturbing PyTorch RNG."""
    return secrets.token_hex(4)


def normalize_short_uuid(raw: object) -> str:
    """Validate and normalize an 8-character hex short UUID (accepting optional leading '#')."""
    if isinstance(raw, str):
        cand = raw.strip().lower().lstrip("#")
        if SHORT_UUID_RE.match(cand):
            return cand
    return ""


def ensure_genome_uuid(genome: dict) -> str:
    """Ensure ``genome['uuid']`` is a valid 8-char short UUID in-place and return it."""
    u = normalize_short_uuid(genome.get("uuid"))
    if not u:
        u = alloc_short_uuid()
        genome["uuid"] = u
    return u


def sync_genome_counter(max_existing_id: int) -> None:
    """Advance the global genome ID counter past any restored IDs."""
    global _genome_counter
    if max_existing_id <= 0:
        return
    with _counter_lock:
        nxt = next(_genome_counter)
        if nxt <= max_existing_id:
            _genome_counter = itertools.count(max_existing_id + 1)
        else:
            # Keep nxt unused or step back 1 cleanly
            _genome_counter = itertools.count(nxt)


def _apply_mixed_activations(z: torch.Tensor, act_codes: torch.Tensor) -> torch.Tensor:
    """Apply per-neuron Picbreeder activation functions along the last dimension."""
    out = torch.empty_like(z)
    mask0 = act_codes == 0
    mask1 = act_codes == 1
    mask2 = act_codes == 2
    mask3 = act_codes == 3
    mask4 = act_codes == 4

    if mask0.any():
        out[..., mask0] = torch.tanh(z[..., mask0])
    if mask1.any():
        out[..., mask1] = torch.sin(z[..., mask1] * 1.5)
    if mask2.any():
        out[..., mask2] = torch.exp(-(z[..., mask2] ** 2))
    if mask3.any():
        out[..., mask3] = torch.abs(z[..., mask3]) - 0.5
    if mask4.any():
        out[..., mask4] = leaky_relu(z[..., mask4], negative_slope=0.1)
    return out


@save_load
class CPPN(nn.Module):
    """Fixed-topology PyTorch Compositional Pattern Producing Network (CPPN)."""

    def __init__(
        self,
        genome_id: int | None = None,
        parents: list[int] | None = None,
        op: str = "random",
    ) -> None:
        super().__init__()
        self.genome_id = genome_id if genome_id is not None else _alloc_genome_id()
        self.uuid: str = alloc_short_uuid()
        self.parents: list[int] = list(parents) if parents else []
        self.op: str = op
        self.layers = nn.ModuleList(
            [
                nn.Linear(in_f, out_f)
                for in_f, out_f in zip(LAYER_DIMS[:-1], LAYER_DIMS[1:])
            ]
        )
        self.register_buffer(
            "activations",
            torch.zeros(NUM_HIDDEN, dtype=torch.long),
        )

    @classmethod
    @torch.no_grad()
    def random(cls, genome_id: int | None = None) -> "CPPN":
        net = cls(genome_id=genome_id, parents=[], op="random")
        for layer_idx, layer in enumerate(net.layers):
            in_f, out_f = layer.weight.shape[1], layer.weight.shape[0]
            # Start sparse like early NEAT graphs so gen 0 has crisp geometric & symmetric forms.
            keep_prob = 0.55 if layer_idx == 0 else (0.35 if layer_idx < len(net.layers) - 1 else 0.45)
            std = 1.6 if layer_idx == 0 else (1.1 if layer_idx < len(net.layers) - 1 else 1.0)
            mask = (torch.rand(out_f, in_f) < keep_prob).float()
            # Ensure every output unit receives at least one active input connection.
            for row in range(out_f):
                if mask[row].sum() == 0:
                    mask[row, int(torch.randint(0, in_f, (1,)).item())] = 1.0
            layer.weight.normal_(0.0, std).mul_(mask)
            layer.bias.normal_(0.0, 0.4 if layer_idx < len(net.layers) - 1 else 0.25)
        net.activations.copy_(
            torch.randint(0, NUM_ACTIVATIONS, (NUM_HIDDEN,), dtype=torch.long)
        )
        return net

    @classmethod
    @torch.no_grad()
    def crossover_and_mutate(
        cls,
        parent_a: "CPPN",
        parent_b: "CPPN",
        *,
        use_crossover: bool = True,
        mut_sigma: float = 0.35,
        mut_rate: float = 0.35,
        act_flip_rate: float = 0.08,
    ) -> "CPPN":
        """Create a child CPPN via neuron-unit crossover and sparse Gaussian mutation.

        Each hidden neuron's incoming weight row, bias, and activation function are
        inherited together as a single functional unit from either parent, avoiding
        mixed/inconsistent neuron representations.
        """
        child = cls(
            parents=(
                [parent_a.genome_id]
                if not use_crossover or parent_a.genome_id == parent_b.genome_id
                else [parent_a.genome_id, parent_b.genome_id]
            ),
            op="crossover" if use_crossover else "mutation",
        )
        acts = parent_a.activations.clone()
        act_offset = 0

        for layer_idx, (c_layer, a_layer, b_layer) in enumerate(
            zip(child.layers, parent_a.layers, parent_b.layers)
        ):
            out_dim = c_layer.weight.shape[0]
            if use_crossover:
                unit_mask = torch.rand(out_dim) < 0.5
                w = torch.where(unit_mask.unsqueeze(-1), b_layer.weight, a_layer.weight)
                b = torch.where(unit_mask, b_layer.bias, a_layer.bias)
                if layer_idx < len(child.layers) - 1:
                    acts[act_offset : act_offset + out_dim] = torch.where(
                        unit_mask,
                        parent_b.activations[act_offset : act_offset + out_dim],
                        parent_a.activations[act_offset : act_offset + out_dim],
                    )
            else:
                w = a_layer.weight.clone()
                b = a_layer.bias.clone()

            if layer_idx < len(child.layers) - 1:
                act_offset += out_dim

            if mut_sigma > 0.0 and mut_rate > 0.0:
                w_mut = (torch.rand_like(w) < mut_rate).float() * torch.randn_like(w) * mut_sigma
                b_mut = (
                    (torch.rand_like(b) < mut_rate).float()
                    * torch.randn_like(b)
                    * (mut_sigma * 0.6)
                )
                w = (w + w_mut).clamp(-6.0, 6.0)
                b = (b + b_mut).clamp(-4.0, 4.0)
            c_layer.weight.copy_(w)
            c_layer.bias.copy_(b)

        if act_flip_rate > 0.0:
            flip_mask = torch.rand(NUM_HIDDEN) < act_flip_rate
            if flip_mask.any():
                acts[flip_mask] = torch.randint(
                    0, NUM_ACTIVATIONS, (int(flip_mask.sum().item()),), dtype=torch.long
                )
        child.activations.copy_(acts)
        return child

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Map (..., 4) coordinates [x, y, r, 1] to (..., 3) RGB values in [0, 1]."""
        h = coords
        act_offset = 0
        for layer in self.layers[:-1]:
            z = layer(h)
            width = z.shape[-1]
            h = _apply_mixed_activations(z, self.activations[act_offset : act_offset + width])
            act_offset += width
        out = self.layers[-1](h)
        return 0.5 + 0.5 * torch.tanh(out)

    @torch.no_grad()
    def render(self, size: int = 64) -> torch.Tensor:
        """Render the CPPN into a (size, size, 3) uint8 RGB tensor."""
        lin = torch.linspace(-1.0, 1.0, size)
        yy, xx = torch.meshgrid(lin, lin, indexing="ij")
        rr = torch.sqrt(xx * xx + yy * yy)
        ones = torch.ones_like(xx)
        coords = torch.stack([xx, yy, rr, ones], dim=-1)
        rgb = self(coords)
        return (rgb * 255.0).round().clamp(0, 255).to(torch.uint8)

    def to_dict(self, *, digits: int = 4) -> dict:
        def _round_params(tensors: Iterable[torch.Tensor]) -> list[float | int]:
            vals = np.round(np.concatenate([t.detach().cpu().numpy().astype(np.float64).ravel() for t in tensors]), digits).tolist()
            return [0 if v == 0.0 else v for v in vals]

        weights = _round_params(layer.weight for layer in self.layers)
        biases = _round_params(layer.bias for layer in self.layers)
        return {
            "id": self.genome_id,
            "uuid": self.uuid,
            "parents": list(self.parents),
            "op": self.op,
            "weights": weights,
            "biases": biases,
            "activations": [int(a) for a in self.activations.tolist()],
        }

    @classmethod
    @torch.no_grad()
    def from_dict(cls, data: dict) -> "CPPN":
        net = cls(
            genome_id=int(data["id"]),
            parents=[int(p) for p in data.get("parents", [])],
            op=str(data.get("op", "random")),
        )
        restored_uuid = normalize_short_uuid(data.get("uuid"))
        if restored_uuid:
            net.uuid = restored_uuid
        w_flat = data["weights"]
        b_flat = data["biases"]
        w_offset = 0
        b_offset = 0
        for layer in net.layers:
            w_num = layer.weight.numel()
            b_num = layer.bias.numel()
            w_slice = torch.tensor(
                w_flat[w_offset : w_offset + w_num], dtype=torch.float32
            ).view_as(layer.weight)
            b_slice = torch.tensor(
                b_flat[b_offset : b_offset + b_num], dtype=torch.float32
            ).view_as(layer.bias)
            layer.weight.copy_(w_slice)
            layer.bias.copy_(b_slice)
            w_offset += w_num
            b_offset += b_num
        net.activations.copy_(torch.tensor(data["activations"], dtype=torch.long))
        return net


def create_random_population(pop_size: int = POPULATION_SIZE) -> list[dict]:
    """Generate `pop_size` fresh random PyTorch CPPN genomes."""
    return [CPPN.random().to_dict() for _ in range(pop_size)]


def _sample_parents_batch(
    pool: list[int],
    scores: list[int],
    num_children: int,
    crossover_rate: float = DEFAULT_CROSSOVER_RATE,
    *,
    num_parents_per_child: int = 2,
    max_tournament_size: int = 3,
) -> list[tuple[int, int, bool]]:
    """Sample `(idx_a, idx_b, use_crossover)` for `num_children` in one vectorized tournament call."""
    if num_children <= 0:
        return []
    if len(pool) == 1:
        only = pool[0]
        return [(only, only, False) for _ in range(num_children)]
    fitnesses = torch.tensor([float(scores[i]) for i in pool], dtype=torch.float32)
    picked = parent_select_tournament(
        fitnesses,
        num_children=num_children,
        num_parents_per_child=num_parents_per_child,
        tournament_size=min(max_tournament_size, len(pool)),
    )
    clamped_rate = max(0.0, min(1.0, float(crossover_rate)))
    cross_flags = (torch.rand(num_children) < clamped_rate).tolist()
    pairs = picked.tolist()
    return [
        (pool[int(pair[0])], pool[int(pair[1])], bool(use_c))
        for pair, use_c in zip(pairs, cross_flags)
    ]


def _sample_parents(
    pool: list[int],
    scores: list[int],
    crossover_rate: float = DEFAULT_CROSSOVER_RATE,
) -> tuple[int, int, bool]:
    """Sample parent indices using PopuLoRA's tournament parent selector."""
    return _sample_parents_batch(pool, scores, 1, crossover_rate=crossover_rate)[0]


def _evolve_population_slots(
    genome_dicts: Iterable[dict],
    scores: list[int],
    up_counts: list[int],
    mut_sigma: float = DEFAULT_MUT_SIGMA,
    pop_size: int = POPULATION_SIZE,
    elite_frac: float = DEFAULT_ELITE_FRAC,
    crossover_rate: float = DEFAULT_CROSSOVER_RATE,
    *,
    brood_size: int = DEFAULT_BROOD_SIZE,
    max_elite_frac: float = 0.5,
    liked_immigrant_divisor: int = 10,
    max_liked_immigrants: int = 2,
    fine_sigma_scale: float = 0.42,
    broad_sigma_scale: float = 0.90,
    fine_mut_rate: float = 0.14,
    broad_mut_rate: float = 0.28,
    fine_act_flip_rate: float = 0.02,
    broad_act_flip_rate: float = 0.06,
    neutral_seed_sigma_scale: float = 0.75,
    neutral_seed_mut_rate: float = 0.22,
    neutral_seed_act_flip_rate: float = 0.05,
    neutral_immigrant_divisor: int = 8,
    max_neutral_immigrants: int = 3,
    neutral_breed_mut_rate: float = 0.26,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    """Generate fixed next-gen slots (elites/immigrants) and bred brood groups.

    Implements Tackett & Carmi (1994) brood selection when `brood_size > 1`: each
    bred slot produces `brood_size` distinct offspring sharing the same parent(s)
    so an extra selection phase can choose the strongest sibling.
    """
    eff_brood = max(MIN_BROOD_SIZE, min(MAX_BROOD_SIZE, int(brood_size)))
    parents = [CPPN.from_dict(g) for g in genome_dicts]
    ranked = sorted(
        range(len(parents)),
        key=lambda i: (-scores[i], -up_counts[i], i),
    )

    liked = [i for i in ranked if scores[i] > 0]
    neutral_or_better = [i for i in ranked if scores[i] >= 0]
    clamped_elite_frac = max(0.0, min(max_elite_frac, float(elite_frac)))
    clamped_crossover = max(0.0, min(1.0, float(crossover_rate)))

    if clamped_elite_frac <= 0.0 or pop_size <= 1:
        target_elites = 0
    else:
        target_elites = min(max(1, round(pop_size * clamped_elite_frac)), pop_size - 1)

    def _clone_elite(src: CPPN) -> CPPN:
        elite = CPPN.from_dict(src.to_dict())
        elite.genome_id = _alloc_genome_id()
        elite.uuid = alloc_short_uuid()
        elite.parents = [src.genome_id]
        elite.op = "elite"
        return elite

    fixed: list[dict[str, object]] = []
    broods: list[dict[str, object]] = []
    slot_idx = 0

    if liked:
        # Keep at least target_elites, and avoid dropping tied top-scorers by index bias.
        top_key = (scores[liked[0]], up_counts[liked[0]])
        top_tied_count = sum(1 for i in liked if (scores[i], up_counts[i]) == top_key)
        max_tie_elites = max(target_elites, pop_size // 2)
        num_elites = (
            min(len(liked), max(target_elites, min(top_tied_count, max_tie_elites)))
            if target_elites > 0
            else 0
        )
        for e_idx in range(num_elites):
            fixed.append({"slot": slot_idx, "genome": _clone_elite(parents[liked[e_idx]]).to_dict()})
            slot_idx += 1

        num_immigrants = min(max_liked_immigrants, max(0, (pop_size - num_elites) // liked_immigrant_divisor))
        num_breed = max(0, pop_size - num_elites - num_immigrants)
        sampled_liked = _sample_parents_batch(liked, scores, num_breed, crossover_rate=clamped_crossover)

        for breed_i, (idx_a, idx_b, use_cross) in enumerate(sampled_liked):
            # Alternate between fine local polishers (keep silhouette) and broader explorers
            is_fine = (breed_i % 2 == 0)
            eff_sigma = mut_sigma * (fine_sigma_scale if is_fine else broad_sigma_scale)
            eff_rate = fine_mut_rate if is_fine else broad_mut_rate
            eff_flip = fine_act_flip_rate if is_fine else broad_act_flip_rate
            candidates = [
                CPPN.crossover_and_mutate(
                    parents[idx_a],
                    parents[idx_b],
                    use_crossover=use_cross,
                    mut_sigma=eff_sigma,
                    mut_rate=eff_rate,
                    act_flip_rate=eff_flip,
                ).to_dict()
                for _ in range(eff_brood)
            ]
            if eff_brood > 1:
                parent_ids = (
                    [parents[idx_a].genome_id, parents[idx_b].genome_id]
                    if use_cross and parents[idx_a].genome_id != parents[idx_b].genome_id
                    else [parents[idx_a].genome_id]
                )
                broods.append({
                    "slot": slot_idx,
                    "parents": parent_ids,
                    "op": "crossover" if use_cross else "mutation",
                    "candidates": candidates,
                })
            else:
                fixed.append({"slot": slot_idx, "genome": candidates[0]})
            slot_idx += 1

        neutral_only = [i for i in ranked if scores[i] == 0]
        for imm in range(num_immigrants):
            if imm == 0 and neutral_only:
                pick = int(torch.randint(0, len(neutral_only), (1,)).item())
                seed_parent = parents[neutral_only[pick]]
                imm_genome = CPPN.crossover_and_mutate(
                    seed_parent,
                    seed_parent,
                    use_crossover=False,
                    mut_sigma=mut_sigma * neutral_seed_sigma_scale,
                    mut_rate=neutral_seed_mut_rate,
                    act_flip_rate=neutral_seed_act_flip_rate,
                ).to_dict()
            else:
                imm_genome = CPPN.random().to_dict()
            fixed.append({"slot": slot_idx, "genome": imm_genome})
            slot_idx += 1
    elif neutral_or_better:
        # Only dislikes or 0-ties were cast: preserve top non-disliked elites before breeding away.
        num_elites = min(len(neutral_or_better), target_elites) if target_elites > 0 else 0
        for e_idx in range(num_elites):
            fixed.append({"slot": slot_idx, "genome": _clone_elite(parents[neutral_or_better[e_idx]]).to_dict()})
            slot_idx += 1
        num_immigrants = min(max_neutral_immigrants, max(0, (pop_size - num_elites) // neutral_immigrant_divisor))
        num_breed = max(0, pop_size - num_elites - num_immigrants)
        sampled_neutral = _sample_parents_batch(
            neutral_or_better,
            scores,
            num_breed,
            crossover_rate=clamped_crossover,
        )
        for idx_a, idx_b, use_cross in sampled_neutral:
            candidates = [
                CPPN.crossover_and_mutate(
                    parents[idx_a],
                    parents[idx_b],
                    use_crossover=use_cross,
                    mut_sigma=mut_sigma,
                    mut_rate=neutral_breed_mut_rate,
                ).to_dict()
                for _ in range(eff_brood)
            ]
            if eff_brood > 1:
                parent_ids = (
                    [parents[idx_a].genome_id, parents[idx_b].genome_id]
                    if use_cross and parents[idx_a].genome_id != parents[idx_b].genome_id
                    else [parents[idx_a].genome_id]
                )
                broods.append({
                    "slot": slot_idx,
                    "parents": parent_ids,
                    "op": "crossover" if use_cross else "mutation",
                    "candidates": candidates,
                })
            else:
                fixed.append({"slot": slot_idx, "genome": candidates[0]})
            slot_idx += 1
        while slot_idx < pop_size:
            fixed.append({"slot": slot_idx, "genome": CPPN.random().to_dict()})
            slot_idx += 1
    else:
        while slot_idx < pop_size:
            fixed.append({"slot": slot_idx, "genome": CPPN.random().to_dict()})
            slot_idx += 1

    top_score = scores[ranked[0]] if ranked else 0
    return fixed, broods, top_score


def evolve_population_broods(
    genome_dicts: Iterable[dict],
    scores: list[int],
    up_counts: list[int],
    mut_sigma: float = DEFAULT_MUT_SIGMA,
    pop_size: int = POPULATION_SIZE,
    elite_frac: float = DEFAULT_ELITE_FRAC,
    crossover_rate: float = DEFAULT_CROSSOVER_RATE,
    *,
    brood_size: int = 2,
    **kwargs: object,
) -> tuple[dict[str, object], int]:
    """Generate a brood-selection bundle (`fixed` elites/immigrants + `broods` sibling groups)."""
    eff_brood = max(MIN_BROOD_SIZE, min(MAX_BROOD_SIZE, int(brood_size)))
    fixed, broods, top_score = _evolve_population_slots(
        genome_dicts,
        scores,
        up_counts,
        mut_sigma=mut_sigma,
        pop_size=pop_size,
        elite_frac=elite_frac,
        crossover_rate=crossover_rate,
        brood_size=eff_brood,
        **kwargs,  # type: ignore[arg-type]
    )
    return {
        "brood_size": eff_brood,
        "pop_size": int(pop_size),
        "fixed": fixed,
        "broods": broods,
    }, top_score


def evolve_population(
    genome_dicts: Iterable[dict],
    scores: list[int],
    up_counts: list[int],
    mut_sigma: float = DEFAULT_MUT_SIGMA,
    pop_size: int = POPULATION_SIZE,
    elite_frac: float = DEFAULT_ELITE_FRAC,
    crossover_rate: float = DEFAULT_CROSSOVER_RATE,
    *,
    max_elite_frac: float = 0.5,
    liked_immigrant_divisor: int = 10,
    max_liked_immigrants: int = 2,
    fine_sigma_scale: float = 0.42,
    broad_sigma_scale: float = 0.90,
    fine_mut_rate: float = 0.14,
    broad_mut_rate: float = 0.28,
    fine_act_flip_rate: float = 0.02,
    broad_act_flip_rate: float = 0.06,
    neutral_seed_sigma_scale: float = 0.75,
    neutral_seed_mut_rate: float = 0.22,
    neutral_seed_act_flip_rate: float = 0.05,
    neutral_immigrant_divisor: int = 8,
    max_neutral_immigrants: int = 3,
    neutral_breed_mut_rate: float = 0.26,
) -> tuple[list[dict], int]:
    """Evolve a PyTorch CPPN population using thumbs-up/down scores.

    Preserves a configurable fraction of top liked elites, blends fine local refinement
    with medium crossover exploration, and injects a small number of neutral/random
    immigrants so larger crowds don't prematurely collapse diversity.
    """
    fixed, _, top_score = _evolve_population_slots(
        genome_dicts,
        scores,
        up_counts,
        mut_sigma=mut_sigma,
        pop_size=pop_size,
        elite_frac=elite_frac,
        crossover_rate=crossover_rate,
        brood_size=1,
        max_elite_frac=max_elite_frac,
        liked_immigrant_divisor=liked_immigrant_divisor,
        max_liked_immigrants=max_liked_immigrants,
        fine_sigma_scale=fine_sigma_scale,
        broad_sigma_scale=broad_sigma_scale,
        fine_mut_rate=fine_mut_rate,
        broad_mut_rate=broad_mut_rate,
        fine_act_flip_rate=fine_act_flip_rate,
        broad_act_flip_rate=broad_act_flip_rate,
        neutral_seed_sigma_scale=neutral_seed_sigma_scale,
        neutral_seed_mut_rate=neutral_seed_mut_rate,
        neutral_seed_act_flip_rate=neutral_seed_act_flip_rate,
        neutral_immigrant_divisor=neutral_immigrant_divisor,
        max_neutral_immigrants=max_neutral_immigrants,
        neutral_breed_mut_rate=neutral_breed_mut_rate,
    )
    ordered = sorted(fixed, key=lambda item: int(item["slot"]))
    return [item["genome"] for item in ordered[:pop_size]], top_score  # type: ignore[misc]


def resize_population(
    genome_dicts: Iterable[dict],
    target_size: int,
    scores: list[int],
    up_counts: list[int],
    mut_sigma: float = 0.35,
) -> list[dict]:
    """Resize a population while keeping existing genomes (growing via breeding or shrinking by rank)."""
    existing = list(genome_dicts)
    if target_size <= len(existing):
        return existing[:target_size]

    parents = [CPPN.from_dict(g) for g in existing]
    ranked = sorted(
        range(len(parents)),
        key=lambda i: (-scores[i], -up_counts[i], i),
    )
    liked = [i for i in ranked if scores[i] > 0]
    pool = liked or [i for i in ranked if scores[i] >= 0]

    need_extra = target_size - len(existing)
    extra: list[dict] = []
    if pool:
        for idx_a, idx_b, use_cross in _sample_parents_batch(pool, scores, need_extra):
            child = CPPN.crossover_and_mutate(
                parents[idx_a],
                parents[idx_b],
                use_crossover=use_cross,
                mut_sigma=mut_sigma,
            )
            extra.append(child.to_dict())
    else:
        extra = [CPPN.random().to_dict() for _ in range(need_extra)]
    return existing + extra


def slugify_policy_name(goal: str, generation: int, index: int) -> str:
    """Build a safe filesystem stem for an exported policy checkpoint."""
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", (goal or "policy").strip()).strip("_").lower()
    return f"{clean or 'policy'}_g{generation}_{index}"


def validate_genome_dict(raw: object) -> dict:
    """Validate an untrusted genome dict for topology, finiteness, and activation range."""
    if not isinstance(raw, dict):
        raise ValueError("Invalid genome payload.")
    if raw.get("kind") == "populora":
        from direct_evolution_together.populora_engine import validate_populora_genome_dict
        return validate_populora_genome_dict(raw)
    weights_raw = raw.get("weights")
    biases_raw = raw.get("biases")
    acts_raw = raw.get("activations")
    if (
        not isinstance(weights_raw, list)
        or not isinstance(biases_raw, list)
        or not isinstance(acts_raw, list)
        or len(weights_raw) != NUM_WEIGHTS
        or len(biases_raw) != NUM_BIASES
        or len(acts_raw) != NUM_HIDDEN
    ):
        raise ValueError("Invalid CPPN genome dimensions.")

    weights = [float(v) for v in weights_raw]
    biases = [float(v) for v in biases_raw]
    if not all(math.isfinite(v) for v in weights) or not all(math.isfinite(v) for v in biases):
        raise ValueError("Genome parameters must be finite.")

    acts: list[int] = []
    for a in acts_raw:
        if isinstance(a, bool) or not isinstance(a, int):
            raise ValueError("Activation codes must be integers.")
        if a < 0 or a >= NUM_ACTIVATIONS:
            raise ValueError(f"Activation codes must be in [0, {NUM_ACTIVATIONS - 1}].")
        acts.append(a)

    return {
        "id": int(raw.get("id", 0) or 0),
        "uuid": normalize_short_uuid(raw.get("uuid")) or alloc_short_uuid(),
        "parents": [int(p) for p in raw.get("parents", []) if isinstance(p, int)],
        "op": str(raw.get("op", "random")),
        "weights": weights,
        "biases": biases,
        "activations": acts,
    }


def genome_content_hash(genome: dict, length: int = 16) -> str:
    """Deterministic content hash of a CPPN or PopuLoRA genome."""
    if isinstance(genome, dict) and genome.get("kind") == "populora":
        from direct_evolution_together.populora_engine import populora_genome_hash
        return populora_genome_hash(genome, length=length)
    w = ",".join(f"{float(v):.4f}" for v in genome["weights"])
    b = ",".join(f"{float(v):.4f}" for v in genome["biases"])
    a = ",".join(str(int(v)) for v in genome["activations"])
    digest = hashlib.sha256(f"{w}|{b}|{a}".encode("ascii")).hexdigest()
    return digest[:length]


def save_cppn_bytes(data: dict) -> bytes:
    """Serialize and deflate-compress a CPPN genome dict using `torch_einops_utils.save_load`."""
    net = CPPN.from_dict(data)
    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
        net.save(tmp.name)
        raw = Path(tmp.name).read_bytes()
    if not zipfile.is_zipfile(io.BytesIO(raw)):
        return raw
    dst = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(raw), "r") as zin,
        zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zout,
    ):
        for info in zin.infolist():
            zout.writestr(info.filename, zin.read(info.filename))
    return dst.getvalue()


def _clean_strain_label(raw: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", (raw or "").strip()).strip("_.")
    return (cleaned[:48] or "strain")


def _read_zip_member_bounded(
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    max_bytes: int,
    error_message: str,
) -> bytes:
    """Read a zip entry up to `max_bytes`, guarding against spoofed ZipInfo sizes."""
    if info.file_size > max_bytes:
        raise ValueError(error_message)
    chunks: list[bytes] = []
    read_total = 0
    with zf.open(info, "r") as member:
        while chunk := member.read(min(65536, max_bytes - read_total + 1)):
            read_total += len(chunk)
            if read_total > max_bytes:
                raise ValueError(error_message)
            chunks.append(chunk)
    return b"".join(chunks)


def _verify_pt_container_bounds(raw_bytes: bytes, source_name: str) -> None:
    """Guard against PyTorch `.pt` zip/decompression bombs before calling `torch.load`."""
    if not zipfile.is_zipfile(io.BytesIO(raw_bytes)):
        return
    try:
        zf_ctx = zipfile.ZipFile(io.BytesIO(raw_bytes), "r")
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{source_name}: Corrupt checkpoint archive.") from exc

    err_msg = (
        f"{source_name}: Checkpoint decompressed size exceeds the "
        f"{MAX_CHECKPOINT_UNCOMPRESSED_BYTES // 1024} KB safety limit."
    )
    with zf_ctx as zf:
        infos = zf.infolist()
        if len(infos) > MAX_PT_ARCHIVE_ENTRIES:
            raise ValueError(
                f"{source_name}: Checkpoint contains too many internal entries ({len(infos)} > {MAX_PT_ARCHIVE_ENTRIES})."
            )
        budget = MAX_CHECKPOINT_UNCOMPRESSED_BYTES
        for info in infos:
            if info.is_dir():
                continue
            member_bytes = _read_zip_member_bounded(zf, info, budget, err_msg)
            budget -= len(member_bytes)


@torch.no_grad()
def load_cppn_bytes(
    raw_bytes: bytes,
    *,
    source_name: str = "weights.pt",
    assign_id: bool = True,
) -> dict:
    """Safely validate and deserialize a downloaded CPPN `.pt` checkpoint into a genome dict.

    Enforces strict compressed & decompressed byte caps before deserializing, uses
    `weights_only=True` so untrusted uploads never execute arbitrary pickle payloads,
    and strictly verifies layer topology, tensor shapes, dtypes, finiteness, and activations.
    """
    if not raw_bytes:
        raise ValueError(f"{source_name}: Checkpoint file is empty.")
    if len(raw_bytes) > MAX_CHECKPOINT_BYTES:
        raise ValueError(
            f"{source_name}: Checkpoint exceeds the {MAX_CHECKPOINT_BYTES // 1024} KB size limit."
        )

    _verify_pt_container_bounds(raw_bytes, source_name)

    try:
        pkg = torch.load(io.BytesIO(raw_bytes), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"{source_name}: Not a valid PyTorch checkpoint.") from exc

    if isinstance(pkg, dict) and isinstance(pkg.get("model"), dict):
        state_dict = pkg["model"]
    elif isinstance(pkg, dict):
        state_dict = pkg
    else:
        raise ValueError(f"{source_name}: Invalid checkpoint structure (expected CPPN state dict).")

    net = CPPN(genome_id=(None if assign_id else 0), parents=[], op="seed")
    expected = net.state_dict()
    if set(state_dict.keys()) != set(expected.keys()):
        dims_str = "→".join(str(d) for d in LAYER_DIMS)
        raise ValueError(
            f"{source_name}: Incompatible network architecture (expected CPPN layers {dims_str})."
        )

    validated: dict[str, torch.Tensor] = {}
    for key, exp_tensor in expected.items():
        val = state_dict[key]
        if not isinstance(val, torch.Tensor):
            raise ValueError(f"{source_name}: Entry '{key}' is not a tensor.")
        if tuple(val.shape) != tuple(exp_tensor.shape):
            raise ValueError(
                f"{source_name}: Incompatible shape for '{key}': got {tuple(val.shape)}, expected {tuple(exp_tensor.shape)}."
            )
        if key == "activations":
            if val.dtype == torch.bool or torch.is_floating_point(val) or torch.is_complex(val):
                raise ValueError(f"{source_name}: 'activations' must contain integer codes.")
            acts = val.to(dtype=torch.long)
            if int(acts.min().item()) < 0 or int(acts.max().item()) >= NUM_ACTIVATIONS:
                raise ValueError(
                    f"{source_name}: Activation codes out of range [0, {NUM_ACTIVATIONS - 1}]."
                )
            validated[key] = acts
        else:
            if not torch.is_floating_point(val):
                raise ValueError(f"{source_name}: Parameter '{key}' must be floating-point.")
            param = val.to(dtype=torch.float32)
            if not bool(torch.isfinite(param).all().item()):
                raise ValueError(f"{source_name}: Parameter '{key}' contains NaN or Inf values.")
            validated[key] = param

    net.load_state_dict(validated, strict=True)
    return net.to_dict()


def parse_uploaded_strain_bytes(
    filename: str,
    raw_bytes: bytes,
    *,
    assign_ids: bool = False,
    max_strains: int = MAX_POPULATION_SIZE,
) -> list[tuple[str, dict, bytes]]:
    """Extract and validate one or more CPPN strains from an uploaded `.pt` or exported `.zip`."""
    safe_name = Path(filename or "upload").name or "upload"
    lower_name = safe_name.lower()
    if max_strains < 1:
        raise ValueError(
            f"Maximum of {MAX_POPULATION_SIZE} seed strains reached (cannot exceed max population {MAX_POPULATION_SIZE})."
        )

    if lower_name.endswith(".pt"):
        genome = load_cppn_bytes(raw_bytes, source_name=safe_name, assign_id=assign_ids)
        return [(_clean_strain_label(Path(safe_name).stem), genome, raw_bytes)]

    if lower_name.endswith(".zip"):
        if not raw_bytes:
            raise ValueError(f"{safe_name}: Zip archive is empty.")
        if len(raw_bytes) > MAX_ARCHIVE_BYTES:
            raise ValueError(
                f"{safe_name}: Zip archive exceeds the {MAX_ARCHIVE_BYTES // (1024 * 1024)} MB size limit."
            )
        try:
            zf_ctx = zipfile.ZipFile(io.BytesIO(raw_bytes), "r")
        except zipfile.BadZipFile as exc:
            raise ValueError(f"{safe_name}: Corrupt or invalid .zip archive.") from exc

        with zf_ctx as zf:
            infos = zf.infolist()
            if len(infos) > MAX_ZIP_ENTRIES:
                raise ValueError(
                    f"{safe_name}: Zip archive contains too many entries ({len(infos)} > {MAX_ZIP_ENTRIES})."
                )

            pt_infos = []
            for info in infos:
                if info.is_dir():
                    continue
                rel = PurePosixPath(info.filename)
                if ".." in rel.parts:
                    raise ValueError(f"{safe_name}: Unsafe path inside zip archive.")
                if "__MACOSX" in rel.parts or rel.name.startswith("._"):
                    continue
                if rel.name.lower().endswith(".zip"):
                    raise ValueError(f"{safe_name}: Nested .zip archives are not permitted.")
                if rel.name.lower().endswith(".pt"):
                    pt_infos.append(info)

            if not pt_infos:
                raise ValueError(f"{safe_name}: Zip archive contains no .pt weight files.")
            if len(pt_infos) > max_strains:
                raise ValueError(
                    f"{safe_name}: Contains {len(pt_infos)} checkpoints, exceeding remaining slot limit ({max_strains})."
                )

            results: list[tuple[str, dict, bytes]] = []
            zip_stem = Path(safe_name).stem
            total_extracted = 0
            for info in pt_infos:
                pt_bytes = _read_zip_member_bounded(
                    zf,
                    info,
                    MAX_CHECKPOINT_BYTES,
                    f"{safe_name} ({info.filename}): Checkpoint exceeds the {MAX_CHECKPOINT_BYTES // 1024} KB size limit.",
                )
                total_extracted += len(pt_bytes)
                if total_extracted > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                    raise ValueError(
                        f"{safe_name}: Extracted checkpoints exceed the {MAX_ARCHIVE_UNCOMPRESSED_BYTES // (1024 * 1024)} MB safety limit."
                    )
                rel = PurePosixPath(info.filename)
                if rel.stem.lower() == "weights" and rel.parent.name:
                    raw_label = rel.parent.name
                elif rel.stem.lower() == "weights":
                    raw_label = zip_stem
                else:
                    raw_label = rel.stem
                genome = load_cppn_bytes(
                    pt_bytes,
                    source_name=f"{safe_name} ({rel.name})",
                    assign_id=assign_ids,
                )
                results.append((_clean_strain_label(raw_label), genome, pt_bytes))
            return results

    raise ValueError(
        f"{safe_name}: Unsupported file format (upload downloaded .pt weights or .zip bundles)."
    )


def create_seeded_population(
    seed_genomes: Sequence[dict],
    pop_size: int = POPULATION_SIZE,
) -> list[dict]:
    """Create a Generation-0 population seeded with archived strains plus fresh random founders.

    Allows `1 <= len(seed_genomes) <= pop_size` so uploaded strains can partially or fully
    seed the population without exceeding `pop_size`.
    """
    pop_size = int(pop_size)
    if pop_size < MIN_POPULATION_SIZE or pop_size > SYSTEM_MAX_POPULATION_SIZE:
        raise ValueError(
            f"Population size must be between {MIN_POPULATION_SIZE} and {SYSTEM_MAX_POPULATION_SIZE}."
        )
    if not seed_genomes:
        raise ValueError("At least one seed strain is required to seed a population.")
    if len(seed_genomes) > pop_size:
        raise ValueError(
            f"Uploaded strains ({len(seed_genomes)}) cannot exceed total population size ({pop_size})."
        )

    seeded: list[dict] = []
    for g in seed_genomes:
        if (
            not isinstance(g, dict)
            or len(g.get("weights", ())) != NUM_WEIGHTS
            or len(g.get("biases", ())) != NUM_BIASES
            or len(g.get("activations", ())) != NUM_HIDDEN
        ):
            raise ValueError("Invalid seed genome dimensions.")
        net = CPPN.from_dict(
            {
                **g,
                "id": _alloc_genome_id(),
                "uuid": alloc_short_uuid(),
                "parents": [],
                "op": "seed",
            }
        )
        seeded.append(net.to_dict())

    remaining = pop_size - len(seeded)
    random_fill = create_random_population(pop_size=remaining) if remaining > 0 else []
    return seeded + random_fill


def build_uv_script_header(extra_deps: Sequence[str] = ()) -> list[str]:
    """Build PEP 723 uv inline script header referencing this package."""
    repo_root = Path(__file__).resolve().parent.parent
    deps = ["torch-einops-utils", "numpy", "packaging", "pillow", *extra_deps, "direct-evolution-together"]
    deps_json = ", ".join(f'"{d}"' for d in deps)
    lines = ["# /// script", f"# dependencies = [{deps_json}]"]
    if (repo_root / "pyproject.toml").exists():
        lines.extend([
            "# [tool.uv.sources]",
            f'# direct-evolution-together = {{ path = "{repo_root.as_posix()}", editable = true }}',
        ])
    lines.append("# ///")
    return lines


def build_runner_script(
    weights_filename: str = "weights.pt",
    task: str = "image",
    env_params: dict[str, object] | None = None,
) -> str:
    """Return a standalone uv script that loads a CPPN checkpoint and runs its task."""
    if task != "image":
        from direct_evolution_together.environments import build_control_runner_script

        return build_control_runner_script(task, weights_filename, env_params=env_params)

    return "\n".join(
        [
            *build_uv_script_header(),
            "import sys",
            "from PIL import Image",
            "from direct_evolution_together.cppn import CPPN",
            "",
            f'net = CPPN.init_and_load(sys.argv[1] if len(sys.argv) > 1 else "{weights_filename}")',
            'Image.fromarray(net.render(256).numpy()).save("out.png")',
            'print("Voila! Saved out.png")',
            "",
        ]
    )
