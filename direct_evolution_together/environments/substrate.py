"""CPPN -> MLP substrate decoding shared by every continuous-control task."""

from __future__ import annotations

from collections import OrderedDict
import math
import threading
from typing import Iterable, Sequence

import numpy as np
import torch

from direct_evolution_together.cppn import CPPN, LAYER_DIMS, genome_content_hash


def _default_layer_xs(dim: int) -> tuple[float, ...]:
    if dim <= 1:
        return (0.0,)
    return tuple(float(x) for x in np.linspace(-1.0, 1.0, dim, dtype=np.float32))


def _apply_mixed_activations_np(z: np.ndarray, act_codes: np.ndarray) -> np.ndarray:
    """Vectorized NumPy equivalent of `cppn._apply_mixed_activations`."""
    out = np.empty_like(z)
    m0 = act_codes == 0
    m1 = act_codes == 1
    m2 = act_codes == 2
    m3 = act_codes == 3
    m4 = act_codes == 4
    if m0.any():
        out[:, m0] = np.tanh(z[:, m0])
    if m1.any():
        out[:, m1] = np.sin(z[:, m1] * np.float32(1.5))
    if m2.any():
        out[:, m2] = np.exp(-(z[:, m2] * z[:, m2]))
    if m3.any():
        out[:, m3] = np.abs(z[:, m3]) - np.float32(0.5)
    if m4.any():
        z4 = z[:, m4]
        out[:, m4] = np.where(z4 >= 0.0, z4, z4 * np.float32(0.1))
    return out


def _eval_cppn_ch0_np(
    coords_np: np.ndarray,
    w_flat: Sequence[float] | np.ndarray,
    b_flat: Sequence[float] | np.ndarray,
    acts_flat: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Fast pure-NumPy CPPN channel-0 evaluation without PyTorch module overhead."""
    w_arr = np.asarray(w_flat, dtype=np.float32)
    b_arr = np.asarray(b_flat, dtype=np.float32)
    acts_arr = np.asarray(acts_flat, dtype=np.int64)

    h = coords_np
    w_off = 0
    b_off = 0
    a_off = 0
    num_cppn_layers = len(LAYER_DIMS) - 1
    for l_idx in range(num_cppn_layers - 1):
        in_f = LAYER_DIMS[l_idx]
        out_f = LAYER_DIMS[l_idx + 1]
        w_num = out_f * in_f
        w_mat = w_arr[w_off : w_off + w_num].reshape(out_f, in_f)
        b_vec = b_arr[b_off : b_off + out_f]
        z = h @ w_mat.T + b_vec
        h = _apply_mixed_activations_np(z, acts_arr[a_off : a_off + out_f])
        w_off += w_num
        b_off += out_f
        a_off += out_f

    # Output channel 0 only (R channel used for HyperNEAT weight/bias decoding)
    in_last = LAYER_DIMS[-2]
    w0_row = w_arr[w_off : w_off + in_last]
    b0 = b_arr[b_off]
    out0 = h @ w0_row + b0
    rgb0 = np.float32(0.5) + np.float32(0.5) * np.tanh(out0)
    return rgb0.astype(np.float64) * 2.0 - 1.0


_MAX_DECODED_CACHE = 1024


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
        w_slices_np: list[np.ndarray] = []
        b_slices_np: list[np.ndarray] = []
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
            w_slices_np.append(w_idx.numpy())
            b_slices_np.append(b_idx.numpy())

        self.coords_np = np.asarray(queries, dtype=np.float32)
        self.coords = torch.from_numpy(self.coords_np)
        self.weight_slices = w_slices
        self.bias_slices = b_slices
        self._weight_slices_np = w_slices_np
        self._bias_slices_np = b_slices_np
        self._decode_cache: OrderedDict[str, tuple[list[np.ndarray], list[np.ndarray]]] = OrderedDict()
        self._decode_lock = threading.Lock()

    def decode(self, genome_or_net: dict | CPPN) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Query the CPPN across all substrate coordinates with variance-preserving normalization."""
        cache_key: str | None = None
        if isinstance(genome_or_net, dict):
            cache_key = genome_content_hash(genome_or_net, length=20)
            with self._decode_lock:
                hit = self._decode_cache.get(cache_key)
                if hit is not None:
                    self._decode_cache.move_to_end(cache_key)
                    return hit
            raw = _eval_cppn_ch0_np(
                self.coords_np,
                genome_or_net["weights"],
                genome_or_net["biases"],
                genome_or_net["activations"],
            )
        else:
            with torch.no_grad():
                raw = genome_or_net(self.coords)[:, 0].cpu().numpy().astype(np.float64) * 2.0 - 1.0

        weights: list[np.ndarray] = []
        biases: list[np.ndarray] = []
        for w_idx, b_idx in zip(self._weight_slices_np, self._bias_slices_np):
            w = raw[w_idx]
            mean = float(w.mean())
            diff = w - mean
            std = float(math.sqrt(float(np.mean(diff * diff))))
            fan_in = w.shape[0]
            w_norm = np.ascontiguousarray(diff * (math.sqrt(1.0 / fan_in) / (std + 1e-6)), dtype=np.float64)
            b_norm = np.ascontiguousarray(raw[b_idx] * 0.1, dtype=np.float64)
            w_norm.flags.writeable = False
            b_norm.flags.writeable = False
            weights.append(w_norm)
            biases.append(b_norm)

        decoded = (weights, biases)
        if cache_key is not None:
            with self._decode_lock:
                self._decode_cache[cache_key] = decoded
                self._decode_cache.move_to_end(cache_key)
                while len(self._decode_cache) > _MAX_DECODED_CACHE:
                    self._decode_cache.popitem(last=False)
        return decoded

    @staticmethod
    def forward(
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        obs: np.ndarray | Iterable[float],
    ) -> np.ndarray:
        """Evaluate the decoded substrate MLP, emitting continuous actions in [-1, 1]."""
        h = obs if (isinstance(obs, np.ndarray) and obs.dtype == np.float64) else np.asarray(obs, dtype=np.float64)
        if len(weights) == 2:
            return np.tanh(np.tanh(h @ weights[0] + biases[0]) @ weights[1] + biases[1])
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
