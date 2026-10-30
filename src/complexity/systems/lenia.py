"""Lenia (Chan 2019), single channel and kernel: `Lenia` (continuous) and `IntegerLenia`
(PLAN.md §4.2).

The pattern's quantization level P selects the variant. Without P, classic continuous Lenia: FFT
convolution, float states in [0, 1]. With P, Integer Lenia: states 0..P, integer kernel and
potential (FFT, rounded back to exact integers), growth by lookup table. Kernel shells and
growth are Gaussian, the standard for Lenia experiments here. The model and the Orbium
patterns, with their parameters (R, T, P, b, m, s), come from the hand-coded Lenium
(`vendor/lenium/`); the other creatures come from the original Lenia catalogue
(`vendor/lenia/`), found there with a bump kernel core but run here on the Gaussian shell.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from complexity.registry import register_system
from complexity.systems.base import System, place_center
from complexity.vendor.lenia import ANIMALS
from complexity.vendor.lenia.rle import rle2arr
from complexity.vendor.lenium import model
from complexity.vendor.lenium.patterns import patterns as LENIUM

# pattern -> index in the original Lenia catalogue (PLAN.md §4.2)
CATALOGUE = {
    "gyrorbium_gyrans": 11, "scutium_solidus": 28, "helicium_solidus": 164,
    "hydrogeminium_natans": 233, "paraptera_arcus_labens": 89,
    "paraptera_cavus_furiosus": 96, "tetralapillium_metamorpha": 498, "kronium_solidus": 283,
    "hexadentium_rotans": 348, "amoebium_dentalis": 588,
}


def from_catalogue(a: dict) -> dict:
    """A catalogue entry in Lenium's pattern format (one kernel, one channel)."""
    p = a["params"]
    b = [float(Fraction(f)) for f in p["b"].split(",")]
    return {"name": a["name"], "R": p["R"], "T": p["T"],
            "kernels": [{"b": b, "m": p["m"], "s": p["s"]}], "cells": [rle2arr(a["cells"])]}


CREATURES = {**LENIUM, **{k: from_catalogue(ANIMALS[i]) for k, i in CATALOGUE.items()}}



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
        "orbium": "Orbium",
        "random": "Random soup (Orbium params)",
        **{k: CREATURES[k]["name"] for k in CATALOGUE},
    }
    # pattern -> (creature, random soup?)
    sources = {
        "orbium": ("orbium", False),
        "random": ("orbium", True),
        **{k: (k, False) for k in CATALOGUE},
    }

    def n_states(self, pattern: str) -> int | None:
        P = CREATURES[self.sources[pattern][0]].get("P")
        return P + 1 if P else None

    def init(self, pattern: str, size: tuple[int, int], rng: jax.Array) -> LeniaState:
        name, soup = self.sources[pattern]
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


@register_system
class IntegerLenia(Lenia):
    """Integer Lenia: the same model on patterns with a quantization level P."""

    key = "integer_lenia"
    name = "Integer Lenia"
    patterns = {
        "min_orbium": "Minimum Orbium",
        "random": "Random soup (minimum Orbium params)",
    }
    sources = {
        "min_orbium": ("min-orbium", False),
        "random": ("min-orbium", True),
    }
