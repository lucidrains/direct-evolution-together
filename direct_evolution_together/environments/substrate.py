"""CPPN -> MLP substrate decoding shared by every continuous-control task."""

from __future__ import annotations

import math
import threading
from typing import Iterable

import numpy as np
import torch

from direct_evolution_together.cppn import CPPN


def _default_layer_xs(dim: int) -> tuple[float, ...]:
    if dim <= 1:
        return (0.0,)
    return tuple(float(x) for x in np.linspace(-1.0, 1.0, dim, dtype=np.float32))


class HyperNEATSubstrate:
    """Maps any fixed-topology 4-input CPPN onto an arbitrary MLP substrate (obs -> hidden -> act)."""

    def __init__(
        self,
        dims: tuple[int, ...],
        *,
        layer_coords: tuple[tuple[float, ...] | None, ...] | None = None,
    ) -> None:
        if len(dims) < 2:
            raise ValueError("Substrate must have at least input and output layers.")
        self.dims = tuple(int(d) for d in dims)

        resolved_coords: list[tuple[float, ...]] = []
        for idx, dim in enumerate(self.dims):
            custom = layer_coords[idx] if (layer_coords is not None and idx < len(layer_coords)) else None
            if custom is not None:
                if len(custom) != dim:
                    raise ValueError(f"Expected {dim} coords for substrate layer {idx}, got {len(custom)}.")
                resolved_coords.append(tuple(float(v) for v in custom))
            else:
                resolved_coords.append(_default_layer_xs(dim))
        self.layer_coords = tuple(resolved_coords)

        queries: list[list[float]] = []
        w_slices: list[torch.Tensor] = []
        b_slices: list[torch.Tensor] = []
        num_layers = len(self.dims) - 1
        zs = np.linspace(-1.0, 1.0, num_layers + 1, dtype=np.float32)

        for l_idx, (d_in, d_out) in enumerate(zip(self.dims[:-1], self.dims[1:])):
            z_mid = float(zs[l_idx] + zs[l_idx + 1]) * 0.5
            xs_in = self.layer_coords[l_idx]
            xs_out = self.layer_coords[l_idx + 1]
            w_idx = torch.zeros((d_in, d_out), dtype=torch.long)
            for i in range(d_in):
                for j in range(d_out):
                    w_idx[i, j] = len(queries)
                    queries.append([xs_in[i], xs_out[j], z_mid, 1.0])
            b_idx = torch.zeros((d_out,), dtype=torch.long)
            for j in range(d_out):
                b_idx[j] = len(queries)
                queries.append([0.0, xs_out[j], float(zs[l_idx + 1]), -1.0])
            w_slices.append(w_idx)
            b_slices.append(b_idx)

        self.coords = torch.tensor(queries, dtype=torch.float32)
        self.weight_slices = w_slices
        self.bias_slices = b_slices

    @torch.no_grad()
    def decode(self, genome_or_net: dict | CPPN) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Query the CPPN across all substrate coordinates with variance-preserving normalization."""
        net = genome_or_net if isinstance(genome_or_net, CPPN) else CPPN.from_dict(genome_or_net)
        # Center channel 0 from [0, 1] -> [-1, 1]
        raw = net(self.coords)[:, 0].cpu().numpy().astype(np.float64) * 2.0 - 1.0

        weights: list[np.ndarray] = []
        biases: list[np.ndarray] = []
        for w_idx, b_idx in zip(self.weight_slices, self.bias_slices):
            w = raw[w_idx.numpy()]
            mean = float(w.mean())
            std = float(np.sqrt(np.mean((w - mean) ** 2)))
            fan_in = w.shape[0]
            w_norm = (w - mean) / (std + 1e-6) * math.sqrt(1.0 / fan_in)
            b_norm = raw[b_idx.numpy()] * 0.1
            weights.append(w_norm.astype(np.float64))
            biases.append(b_norm.astype(np.float64))
        return weights, biases

    @staticmethod
    def forward(
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        obs: np.ndarray | Iterable[float],
    ) -> np.ndarray:
        """Evaluate the decoded substrate MLP, emitting continuous actions in [-1, 1]."""
        h = np.asarray(obs, dtype=np.float64)
        for w, b in zip(weights, biases):
            h = np.tanh(h @ w + b)
        return h

    @staticmethod
    def forward_with_trace(
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        obs: np.ndarray | Iterable[float],
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Evaluate the substrate MLP and return per-layer activations for visualization."""
        h = np.asarray(obs, dtype=np.float64)
        activations: list[np.ndarray] = [h]
        for w, b in zip(weights, biases):
            h = np.tanh(h @ w + b)
            activations.append(h)
        return h, activations


_SUBSTRATE_CACHE: dict[tuple[tuple[int, ...], tuple[tuple[float, ...], ...]], HyperNEATSubstrate] = {}
_SUBSTRATE_LOCK = threading.Lock()


def get_substrate(
    dims: tuple[int, ...],
    layer_coords: tuple[tuple[float, ...] | None, ...] | None = None,
) -> HyperNEATSubstrate:
    norm_dims = tuple(int(d) for d in dims)
    resolved_key_coords = tuple(
        tuple(float(v) for v in layer_coords[i])
        if (layer_coords is not None and i < len(layer_coords) and layer_coords[i] is not None)
        else _default_layer_xs(d)
        for i, d in enumerate(norm_dims)
    )
    key = (norm_dims, resolved_key_coords)
    with _SUBSTRATE_LOCK:
        sub = _SUBSTRATE_CACHE.get(key)
        if sub is None:
            sub = HyperNEATSubstrate(norm_dims, layer_coords=resolved_key_coords)
            _SUBSTRATE_CACHE[key] = sub
        return sub
