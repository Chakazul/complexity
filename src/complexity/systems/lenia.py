"""Lenia (Chan 2019), single channel and kernel, continuous or integer (PLAN.md §4.2).

The quantization level P selects the variant. P = 0 is classic continuous Lenia: FFT
convolution, float states in [0, 1]. P > 0 is Integer Lenia: states 0..P, integer kernel and
potential (FFT, rounded back to exact integers), growth by lookup table
(`vendor/lenium/integer.py`). Each pattern carries its own parameters (R, T, P, m, s, kernel and growth profiles), taken from Lenium's creature library.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from complexity.registry import register_system
from complexity.systems.base import System, place_center
from complexity.vendor.lenium import integer, profiles
from complexity.vendor.lenium.rle import rle2arr

CREATURES = json.loads((Path(integer.__file__).parent / "creatures.json").read_text())

# pattern -> (creature, random soup?)
PATTERNS = {
    "min_orbium": ("min_orbium", False),
    "random": ("min_orbium", True),
    "orbium": ("orbium", False),
    "random_continuous": ("orbium", True),
}


@partial(jax.tree_util.register_dataclass, data_fields=["A", "K", "g"],
         meta_fields=["P", "T", "m", "s", "growth"])
@dataclass(frozen=True)
class LeniaState:
    A: jax.Array  # int32 0..P (integer) or float32 [0, 1] (continuous)
    K: jax.Array  # rfft2 of the kernel: integer weights (integer) or normalized (continuous)
    g: jax.Array | None  # growth lookup table (integer only)
    P: int
    T: float
    m: float = field(default=0.0)
    s: float = field(default=0.0)
    growth: str = "gaussian"


def _profiles(creature: dict) -> tuple[str, str]:
    functions = creature.get("functions", {})
    return functions.get("kernel", "gaussian"), functions.get("growth", "gaussian")


def _continuous_kernel_fft(size, R, rings, profile) -> jax.Array:
    h, w = size
    y, x = jnp.ogrid[-(h // 2) : h - h // 2, -(w // 2) : w - w // 2]
    D = jnp.sqrt(x**2 + y**2) / R
    K = sum(ring["b"] * profiles.kernel_core(profile, D, ring["r"], ring["w"]) for ring in rings)
    K = K * (D < 1)
    return jnp.fft.rfft2(jnp.fft.ifftshift(K / K.sum()))


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
        P = CREATURES[PATTERNS[pattern][0]]["P"]
        return P + 1 if P else None

    def init(self, pattern: str, size: tuple[int, int], rng: jax.Array) -> LeniaState:
        name, soup = PATTERNS[pattern]
        c = CREATURES[name]
        R, T, P = c["R"], c["T"], c["P"]
        (kern,) = c["kernels"]
        kernel_profile, growth_profile = _profiles(c)

        if soup:
            h, w = size
            cells = jax.random.uniform(rng, (h * 3 // 4, w * 3 // 4))
            cells = integer.discretize(cells, P) if P else cells
        elif P:
            (cells,) = c["cells"]  # integer cells, one channel
        else:
            cells = rle2arr(c["cells"])
        A = place_center(jnp.zeros(size, jnp.int32 if P else jnp.float32), cells)

        if P:
            K = integer.kernel(R, P, b=[ring["b"] for ring in kern["rings"]])
            g = integer.growth_lookup(K, P, kern["m"], kern["s"])
            return LeniaState(A, integer.kernel_fft(K, size), g, P, T)
        fK = _continuous_kernel_fft(size, R, kern["rings"], kernel_profile)
        return LeniaState(A, fK, None, 0, T, kern["m"], kern["s"], growth_profile)

    def step(self, st: LeniaState) -> LeniaState:
        if st.P:
            A = integer.update(st.A, integer.potential_fft(st.A, st.K), st.g, st.T, st.P)
        else:
            U = jnp.fft.irfft2(st.K * jnp.fft.rfft2(st.A), s=st.A.shape)
            G = profiles.growth_core(st.growth, U, st.m, st.s) * 2 - 1
            A = jnp.clip(st.A + G / st.T, 0, 1)
        return LeniaState(A, st.K, st.g, st.P, st.T, st.m, st.s, st.growth)

    def observe(self, st: LeniaState) -> jax.Array:
        if not st.P:
            return st.A
        # A lookup, not A / P: under jit XLA multiplies by 1/P, which is off by an ulp from
        # the exact levels k / P that discrete frames promise (PLAN.md §2).
        return jnp.asarray(np.arange(st.P + 1, dtype=np.float32) / np.float32(st.P))[st.A]
