"""Lenia (Chan 2019), single channel and kernel: `Lenia` (continuous) and `IntegerLenia`
(PLAN.md §4.2).

The pattern's quantization level P selects the variant. Without P, classic continuous Lenia: FFT
convolution, float states in [0, 1]. With P, Integer Lenia: states 0..P, integer kernel and
potential (FFT, rounded back to exact integers), growth by lookup table. Growth is Gaussian.
Kernel shells are Gaussian, the standard for Lenia experiments here, unless the pattern was
found with the original Lenia's bump core (`kernel_core = "bump"`): the catalogue creatures.
The model and the Orbium patterns, with their parameters (R, T, P, b, m, s), come from the
hand-coded Lenium (`vendor/lenium/`); the other creatures come from the original Lenia
catalogue (`vendor/lenia/`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from complexity.registry import register_system
from complexity.systems.base import System, place_center, random_active_cell
from complexity.vendor.lenia import ANIMALS
from complexity.vendor.lenia.rle import rle2arr
from complexity.vendor.lenium import model
from complexity.vendor.lenium.patterns import patterns as LENIUM

# pattern -> index in the original Lenia catalogue (PLAN.md §4.2); where several entries
# share a name, the index picks one (Helicium serratus turbulentus: the 3rd, Asterium vagus
# and Amoebium foliatus: the 2nd)
CATALOGUE = {
    "gyrorbium_gyrans": 11, "triscutium_valvatus_laterale": 44,
    "helicium_serratus_turbulentus": 173, "hydrogeminium_natans": 233,
    "paraptera_arcus_pedes": 91, "paraptera_cavus_furiosus": 96,
    "tetralapillium_metamorpha": 498, "florakronium_arcus": 313,
    "heptadentium_scintillans": 355, "asterium_vagus": 404, "asterium_metamorpha_currens": 408,
    "amoebium_foliatus": 587, "tetradecactenium": 257, "urium_longus": 267,
}

# the original Lenia's kernel cores (`kn`) that the model implements
KERNEL_CORES = {1: "bump"}


def from_catalogue(a: dict) -> dict:
    """A catalogue entry in Lenium's pattern format (one kernel, one channel), with its
    kernel core."""
    p = a["params"]
    b = [float(Fraction(f)) for f in p["b"].split(",")]
    assert p["gn"] == 1, "only Gaussian growth is implemented"
    return {"name": a["name"], "R": p["R"], "T": p["T"], "kernel_core": KERNEL_CORES[p["kn"]],
            "kernels": [{"b": b, "m": p["m"], "s": p["s"]}], "cells": [rle2arr(a["cells"])]}


# shown after the name in the pattern list
BEHAVIOUR = {
    "gyrorbium_gyrans": "gyrating", "paraptera_cavus_furiosus": "chaotic",
    "tetralapillium_metamorpha": "chaotic", "asterium_vagus": "chaotic",
    "asterium_metamorpha_currens": "chaotic", "tetradecactenium": "big", "urium_longus": "big",
}


def display_name(key: str) -> str:
    """The pattern's catalogue name and its behaviour."""
    return CREATURES[key]["name"] + (f" ({BEHAVIOUR[key]})" if key in BEHAVIOUR else "")

CREATURES = {**LENIUM, **{k: from_catalogue(ANIMALS[i]) for k, i in CATALOGUE.items()}}



# m and s are data, not static metadata, so runs with different growth parameters share one
# compiled step and can be batched.
@partial(jax.tree_util.register_dataclass, data_fields=["A", "K", "g", "m", "s"],
         meta_fields=["P", "T"])
@dataclass(frozen=True)
class LeniaState:
    A: jax.Array  # int32 0..P (integer) or float32 [0, 1] (continuous)
    K: jax.Array  # rfft2 of the kernel: integer weights (integer) or normalized (continuous)
    g: jax.Array | None  # growth lookup table (integer only; built from m, s)
    P: int  # 0 = continuous
    T: float
    m: jax.Array = field(default_factory=lambda: jnp.float32(0))  # growth (continuous only)
    s: jax.Array = field(default_factory=lambda: jnp.float32(0))


@register_system
class Lenia(System):
    key = "lenia"
    name = "Lenia"
    patterns = {
        "orbium": "Orbium",
        "random": "Random soup (Orbium params)",
        **{k: display_name(k) for k in CATALOGUE},
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
        return self.init_with(pattern, size, rng)

    def init_with(self, pattern: str, size: tuple[int, int], rng: jax.Array, *,
                  cells: str = "pattern", m: float | None = None,
                  s: float | None = None) -> LeniaState:
        """`init`, optionally with other growth parameters and other cells: "pattern" (the
        pattern's own), "soup" (random cells in the central 3/4 with the pattern's rule),
        "scramble" (the pattern's bounding box filled with iid uniform states; a soup for
        soups) or "uniform" (every cell iid uniform over the states)."""
        name, soup = self.sources[pattern]
        c = CREATURES[name]
        R, T, P = c["R"], c["T"], c.get("P", 0)
        core = c.get("kernel_core", "gaussian")
        (kern,) = c["kernels"]
        m = kern["m"] if m is None else m
        s = kern["s"] if s is None else s

        if cells == "uniform":
            A = (jax.random.randint(rng, size, 0, P + 1) if P
                 else jax.random.uniform(rng, size))
        else:
            if soup or cells == "soup":
                h, w = size
                init = jax.random.uniform(rng, (h * 3 // 4, w * 3 // 4))
                init = model.discretize(init, P) if P else init
            else:
                (init,) = c["cells"]  # one channel
                if cells == "scramble":
                    shape = np.shape(init)
                    init = (jax.random.randint(rng, shape, 0, P + 1) if P
                            else jax.random.uniform(rng, shape))
            A = place_center(jnp.zeros(size, jnp.int32 if P else jnp.float32), init)

        if P:
            K = model.integer_kernel(R, P, kern["b"], core=core)
            g = model.growth_lookup(K, P, m, s)
            return LeniaState(A, model.integer_kernel_fft(K, size), g, P, T)
        return LeniaState(A, model.kernel_fft(size, R, kern["b"], core=core), None, 0, T,
                          jnp.float32(m), jnp.float32(s))

    def reach(self, pattern: str) -> int:
        return CREATURES[self.sources[pattern][0]]["R"]

    def soup(self, pattern: str, size: tuple[int, int], rng: jax.Array) -> LeniaState:
        return self.init_with(pattern, size, rng, cells="soup")

    def uniform(self, pattern: str, size: tuple[int, int], rng: jax.Array) -> LeniaState:
        return self.init_with(pattern, size, rng, cells="uniform")

    def scramble(self, pattern: str, size: tuple[int, int], rng: jax.Array) -> LeniaState:
        return self.init_with(pattern, size, rng, cells="scramble")

    def step(self, st: LeniaState) -> LeniaState:
        if st.P:
            A = model.integer_update(st.A, model.potential_fft(st.A, st.K), st.g, st.T, st.P)
        else:
            A = model.step(st.A, st.K, st.m, st.s, st.T)
        return LeniaState(A, st.K, st.g, st.P, st.T, st.m, st.s)

    def perturb(self, st: LeniaState, rng: jax.Array) -> LeniaState:
        """Integer: one active cell up or down one level. Continuous: Gaussian noise of
        standard deviation 1e-5 on every cell, clipped to [0, 1]."""
        if st.P:
            k1, k2 = jax.random.split(rng)
            y, x = random_active_cell(st.A > 0, k1)
            a = st.A[y, x]
            up = jnp.where(a == 0, True, jnp.where(a == st.P, False, jax.random.bernoulli(k2)))
            A = st.A.at[y, x].set(jnp.where(up, a + 1, a - 1))
        else:
            A = jnp.clip(st.A + 1e-5 * jax.random.normal(rng, st.A.shape), 0, 1)
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
