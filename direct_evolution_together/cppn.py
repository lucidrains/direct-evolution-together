import itertools
from pathlib import Path
import re
import tempfile
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_einops_utils.save_load import save_load

LAYER_DIMS = (4, 12, 12, 12, 3)
NUM_HIDDEN = sum(LAYER_DIMS[1:-1])  # 36
NUM_ACTIVATIONS = 5  # 0=tanh, 1=sin, 2=gauss, 3=abs, 4=leaky_relu
POPULATION_SIZE = 30
DEFAULT_MUT_SIGMA = 0.35
DEFAULT_ELITE_FRAC = 0.10
DEFAULT_CROSSOVER_RATE = 0.75

_genome_counter = itertools.count(1)


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
        out[..., mask4] = F.leaky_relu(z[..., mask4], negative_slope=0.1)
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
        self.genome_id = genome_id if genome_id is not None else next(_genome_counter)
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

    def to_dict(self) -> dict:
        weights: list[float] = []
        biases: list[float] = []
        for layer in self.layers:
            weights.extend(round(v, 4) for v in layer.weight.detach().flatten().tolist())
            biases.extend(round(v, 4) for v in layer.bias.detach().flatten().tolist())
        return {
            "id": self.genome_id,
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


def _sample_parents(
    pool: list[int],
    scores: list[int],
    crossover_rate: float = DEFAULT_CROSSOVER_RATE,
) -> tuple[int, int, bool]:
    """Sample fitness-weighted parent indices, pairing distinct parents when possible."""
    if len(pool) == 1:
        return pool[0], pool[0], False
    min_s = min(scores[i] for i in pool)
    weights = torch.tensor([float(scores[i] - min_s + 1) for i in pool])
    picks = torch.multinomial(weights, 2, replacement=False)
    use_cross = bool(torch.rand(1).item() < max(0.0, min(1.0, float(crossover_rate))))
    return pool[int(picks[0].item())], pool[int(picks[1].item())], use_cross


def evolve_population(
    genome_dicts: Iterable[dict],
    scores: list[int],
    up_counts: list[int],
    mut_sigma: float = DEFAULT_MUT_SIGMA,
    pop_size: int = POPULATION_SIZE,
    elite_frac: float = DEFAULT_ELITE_FRAC,
    crossover_rate: float = DEFAULT_CROSSOVER_RATE,
) -> tuple[list[dict], int]:
    """Evolve a PyTorch CPPN population using thumbs-up/down scores.

    Preserves a configurable fraction of top liked elites, blends fine local refinement
    with medium crossover exploration, and injects a small number of neutral/random
    immigrants so larger crowds don't prematurely collapse diversity.
    """
    parents = [CPPN.from_dict(g) for g in genome_dicts]
    ranked = sorted(
        range(len(parents)),
        key=lambda i: (-scores[i], -up_counts[i], i),
    )

    liked = [i for i in ranked if scores[i] > 0]
    neutral_or_better = [i for i in ranked if scores[i] >= 0]
    clamped_elite_frac = max(0.0, min(0.5, float(elite_frac)))
    clamped_crossover = max(0.0, min(1.0, float(crossover_rate)))

    next_gen: list[CPPN] = []
    if liked:
        # Preserve configurable fraction of liked top individuals as elites
        if clamped_elite_frac <= 0.0 or pop_size <= 1:
            target_elites = 0
        else:
            target_elites = min(max(1, round(pop_size * clamped_elite_frac)), pop_size - 1)
        num_elites = min(len(liked), target_elites)
        for e_idx in range(num_elites):
            elite_src = parents[liked[e_idx]]
            elite = CPPN.from_dict(elite_src.to_dict())
            elite.parents = [elite_src.genome_id]
            elite.op = "elite"
            next_gen.append(elite)

        num_immigrants = min(2, max(0, (pop_size - num_elites) // 10))
        num_breed = max(0, pop_size - num_elites - num_immigrants)

        for slot in range(num_breed):
            idx_a, idx_b, use_cross = _sample_parents(liked, scores, crossover_rate=clamped_crossover)
            # Alternate between fine local polishers (keep silhouette) and broader explorers
            is_fine = (slot % 2 == 0)
            eff_sigma = mut_sigma * (0.42 if is_fine else 0.9)
            eff_rate = 0.14 if is_fine else 0.28
            eff_flip = 0.02 if is_fine else 0.06
            next_gen.append(
                CPPN.crossover_and_mutate(
                    parents[idx_a],
                    parents[idx_b],
                    use_crossover=use_cross,
                    mut_sigma=eff_sigma,
                    mut_rate=eff_rate,
                    act_flip_rate=eff_flip,
                )
            )

        neutral_only = [i for i in ranked if scores[i] == 0]
        for imm in range(num_immigrants):
            if imm == 0 and neutral_only:
                pick = int(torch.randint(0, len(neutral_only), (1,)).item())
                seed_parent = parents[neutral_only[pick]]
                next_gen.append(
                    CPPN.crossover_and_mutate(
                        seed_parent,
                        seed_parent,
                        use_crossover=False,
                        mut_sigma=mut_sigma * 0.75,
                        mut_rate=0.22,
                        act_flip_rate=0.05,
                    )
                )
            else:
                next_gen.append(CPPN.random())
    elif neutral_or_better:
        # Only dislikes were cast: steer away by breeding neutral survivors + fresh explorers
        num_immigrants = min(3, max(1, pop_size // 8))
        for _ in range(max(0, pop_size - num_immigrants)):
            idx_a, idx_b, use_cross = _sample_parents(
                neutral_or_better,
                scores,
                crossover_rate=clamped_crossover,
            )
            next_gen.append(
                CPPN.crossover_and_mutate(
                    parents[idx_a],
                    parents[idx_b],
                    use_crossover=use_cross,
                    mut_sigma=mut_sigma,
                    mut_rate=0.26,
                )
            )
        while len(next_gen) < pop_size:
            next_gen.append(CPPN.random())
    else:
        next_gen = [CPPN.random() for _ in range(pop_size)]

    top_score = scores[ranked[0]] if ranked else 0
    return [g.to_dict() for g in next_gen[:pop_size]], top_score


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

    extra: list[dict] = []
    for _ in range(target_size - len(existing)):
        if pool:
            idx_a, idx_b, use_cross = _sample_parents(pool, scores)
            child = CPPN.crossover_and_mutate(
                parents[idx_a],
                parents[idx_b],
                use_crossover=use_cross,
                mut_sigma=mut_sigma,
            )
            extra.append(child.to_dict())
        else:
            extra.append(CPPN.random().to_dict())
    return existing + extra


def slugify_policy_name(goal: str, generation: int, index: int) -> str:
    """Build a safe filesystem stem for an exported policy checkpoint."""
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", (goal or "policy").strip()).strip("_").lower()
    return f"{clean or 'policy'}_g{generation}_{index}"


def save_cppn_bytes(data: dict) -> bytes:
    """Serialize a CPPN genome dict using `torch_einops_utils.save_load`."""
    net = CPPN.from_dict(data)
    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
        net.save(tmp.name)
        return Path(tmp.name).read_bytes()


def build_runner_script(weights_filename: str = "weights.pt") -> str:
    """Return a standalone <=10 LOC uv script that loads and renders a CPPN."""
    repo_root = Path(__file__).resolve().parent.parent
    pkg_spec = (
        f"direct-evolution-together @ {repo_root.as_uri()}"
        if (repo_root / "pyproject.toml").exists()
        else "direct-evolution-together"
    )
    deps = (
        '["torch-einops-utils", "numpy", "packaging", "pillow", '
        f'"{pkg_spec}"]'
    )
    return "\n".join(
        [
            "# /// script",
            f"# dependencies = {deps}",
            "# ///",
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
