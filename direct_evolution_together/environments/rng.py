"""Deterministic PRNG mirrored by the browser worker (`mulberry32`)."""

from __future__ import annotations

from typing import Callable


def mulberry32(seed: int) -> Callable[[], float]:
    """Deterministic 32-bit PRNG matching JavaScript Math.imul bit-for-bit."""
    a = int(seed) & 0xFFFFFFFF

    def _next() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((a ^ (a >> 15)) * (1 | a)) & 0xFFFFFFFF
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) ^ t) & 0xFFFFFFFF
        t = (t ^ (t >> 14)) & 0xFFFFFFFF
        return t / 4294967296.0

    return _next
