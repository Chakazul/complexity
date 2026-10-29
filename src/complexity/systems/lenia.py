"""Lenia (Chan 2019), single channel and kernel, continuous or integer (PLAN.md §4.2).

The quantization level P selects the variant. Without P, classic continuous Lenia: FFT
convolution, float states in [0, 1]. With P, Integer Lenia: states 0..P, integer kernel and
potential (FFT, rounded back to exact integers), growth by lookup table. Kernel shells and
growth are Gaussian, the standard for Lenia experiments here. The model and the patterns,
with their parameters (R, T, P, b, m, s), come from the hand-coded Lenium
(`vendor/lenium/`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from complexity.registry import register_system
from complexity.systems.base import System, place_center
from complexity.vendor.lenium import model
from complexity.vendor.lenium.patterns import patterns as CREATURES

# pattern -> (creature, random soup?)
PATTERNS = {
    "min_orbium": ("min-orbium", False),
    "random": ("min-orbium", True),
    "orbium": ("orbium", False),
    "random_continuous": ("orbium", True),
}


@partial(jax.tree_util.register_dataclass, data_fields=["A", "K", "g"],
         meta_fields=["P", "T", "m", "s"])
@dataclass(frozen=True)
class LeniaState:
    A: jax.Array  # int32 0..P (integer) or float32 [0, 1] (continuous)
    K: jax.Array  # rfft2 of the kernel: integer weights (integer) or normalized (continuous)
    g: jax.Array | None  # growth lookup table (integer only)
    P: int  # 0 = continuous
    T: float
    m: float = field(default=0.0)
    s: float = field(default=0.0)


@register_system
class Lenia(System):
    key = "lenia"
    name = "Lenia"
    patterns = {
        "min_orbium": "Minimum Orbium (Integer Lenia, P=12)",
        "random": "Random soup (Integer Lenia, P=12)",
        "orbium": "Orbium (continuous)",
        "random_continuous": "Random soup (continuous, Orbium params)",
    }

    def n_states(self, pattern: str) -> int | None:
        P = CREATURES[PATTERNS[pattern][0]].get("P")
        return P + 1 if P else None

    def init(self, pattern: str, size: tuple[int, int], rng: jax.Array) -> LeniaState:
        name, soup = PATTERNS[pattern]
        c = CREATURES[name]
        R, T, P = c["R"], c["T"], c.get("P", 0)
        (kern,) = c["kernels"]

        if soup:
            h, w = size
            cells = jax.random.uniform(rng, (h * 3 // 4, w * 3 // 4))
            cells = model.discretize(cells, P) if P else cells
        else:
            (cells,) = c["cells"]  # one channel
        A = place_center(jnp.zeros(size, jnp.int32 if P else jnp.float32), cells)

        if P:
            K = model.integer_kernel(R, P, kern["b"])
            g = model.growth_lookup(K, P, kern["m"], kern["s"])
            return LeniaState(A, model.integer_kernel_fft(K, size), g, P, T)
        return LeniaState(A, model.kernel_fft(size, R, kern["b"]), None, 0, T, kern["m"], kern["s"])

    def step(self, st: LeniaState) -> LeniaState:
        if st.P:
            A = model.integer_update(st.A, model.potential_fft(st.A, st.K), st.g, st.T, st.P)
        else:
            A = model.step(st.A, st.K, st.m, st.s, st.T)
        return LeniaState(A, st.K, st.g, st.P, st.T, st.m, st.s)

    def observe(self, st: LeniaState) -> jax.Array:
        if not st.P:
            return st.A
        # A lookup, not A / P: under jit XLA multiplies by 1/P, which is off by an ulp from
        # the exact levels k / P that discrete frames promise (PLAN.md §2).
        return jnp.asarray(np.arange(st.P + 1, dtype=np.float32) / np.float32(st.P))[st.A]
