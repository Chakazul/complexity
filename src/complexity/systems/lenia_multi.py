"""Multi-channel, multi-kernel Lenia (Chan 2020): the Aquarium (PLAN.md §4.2, M3b).

The model and the Aquarium (Tessellatium gyrans: 3 channels, 15 kernels, Gaussian ring
kernels and growth) come from the hand-coded Lenium (`vendor/lenium/`). Measures see one
frame, the mean of the channels.

Two parameter sets, a hair apart (PLAN.md §4.2): Lenium's Aquarium self-replicates on its
own; with the values Lenium keeps as comments (Leniabreeder's VT049W) a lone unit glides
without dividing, but units replicate when they collide. Both grow into a network of ~100
separate units on 256² (rotating rings and moving C-shapes), not into goo.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from complexity.registry import register_system
from complexity.systems.base import System, place_center
from complexity.vendor.lenium import model
from complexity.vendor.lenium.patterns import patterns as LENIUM


def vt049w(aquarium: dict) -> dict:
    """The Aquarium with the two kernels that Lenium keeps as comments restored: kernel 3
    rings b = [0, 1] (Lenium: [1/12, 1]) and kernel 13 r = 0.82, b = [1/6, 1, 0] (Lenium:
    0.55, [1/6, 1]; the zero third ring makes these nearly the same kernel, out to 0.547 R).
    These are the values of Leniabreeder's VT049W "Aquarium (self-replicating)"
    (~/github/lenia-ref/code/Leniabreeder/lenia/patterns.py), whose cells equal Lenium's.
    Run here, a lone unit does not divide; colliding units do."""
    p = copy.deepcopy(aquarium)
    p["name"] = "Aquarium (VT049W)"
    p["kernels"][3]["b"] = [0, 1]
    p["kernels"][13].update(r=0.82, b=[1 / 6, 1, 0])
    return p


CREATURES = {"aquarium": LENIUM["aquarium"], "vt049w": vt049w(LENIUM["aquarium"])}


def scatter(cells: np.ndarray, size: tuple[int, int], n: int, seed: int = 0) -> np.ndarray:
    """`n` copies of `cells` (y x c) in quarter turns, on a jittered lattice of 2 rows."""
    rng = np.random.default_rng(seed)
    h, w = size
    A = np.zeros((h, w, cells.shape[-1]), np.float32)
    cols = -(-n // 2)
    for i in range(n):
        c = np.rot90(cells, rng.integers(4))
        y = (i // cols) * (h // 2) + rng.integers(0, max(1, h // 2 - c.shape[0]))
        x = (i % cols) * (w // cols) + rng.integers(0, max(1, w // cols - c.shape[1]))
        A[y : y + c.shape[0], x : x + c.shape[1]] = c[: h - y, : w - x]
    return A


@partial(jax.tree_util.register_dataclass, data_fields=["A", "fK", "m", "s", "h", "Mk", "Mc"],
         meta_fields=["T"])
@dataclass(frozen=True)
class MultiLeniaState:
    A: jax.Array  # float32 (y x c) in [0, 1]
    fK: jax.Array  # rfft2 of the normalized kernels (y x' k)
    m: jax.Array  # growth per kernel (k)
    s: jax.Array
    h: jax.Array
    Mk: jax.Array  # channel -> kernel (c k)
    Mc: jax.Array  # kernel -> channel (k c)
    T: float


@register_system
class MultiLenia(System):
    key = "lenia_multi"
    name = "Multi-channel Lenia"
    patterns = {
        "aquarium": "Aquarium (self-replicating)",
        "aquarium_unit": "Aquarium unit (VT049W, does not divide alone)",
        "aquarium_scene": "Aquarium, 8 units (VT049W, divide on collision)",
    }
    default_size = (256, 256)
    # pattern -> (creature, how to place its cells); random soups die out under both rules
    sources = {"aquarium": ("aquarium", "pattern"), "aquarium_unit": ("vt049w", "pattern"),
               "aquarium_scene": ("vt049w", "scene")}

    def init(self, pattern: str, size: tuple[int, int], rng: jax.Array) -> MultiLeniaState:
        return self._init(pattern, size, rng, self.sources[pattern][1])

    def _init(self, pattern, size, rng, cells) -> MultiLeniaState:
        c = CREATURES[self.sources[pattern][0]]
        init = np.moveaxis(np.asarray(c["cells"], np.float32), 0, -1)  # (y x c)
        channels = init.shape[-1]
        h, w = size
        if cells == "uniform":
            A = jax.random.uniform(rng, (h, w, channels))
        elif cells == "scramble":  # the pattern's box, iid uniform
            box = jax.random.uniform(rng, init.shape)
            A = jnp.stack([place_center(jnp.zeros(size, jnp.float32), box[..., i])
                           for i in range(channels)], -1)
        elif cells == "scene":
            A = jnp.asarray(scatter(init, size, 8))
        elif cells == "soup":
            A = jnp.zeros((h, w, channels), jnp.float32)
            soup = jax.random.uniform(rng, (h * 3 // 4, w * 3 // 4, channels))
            A = jnp.stack([place_center(A[..., i], soup[..., i]) for i in range(channels)], -1)
        else:
            A = jnp.stack([place_center(jnp.zeros(size, jnp.float32), init[..., i])
                           for i in range(channels)], -1)
        m, s, hh, Mk, Mc = model.multi_params(c["kernels"], channels)
        return MultiLeniaState(A, model.multi_kernel_fft(size, c["R"], c["kernels"]),
                               m, s, hh, Mk, Mc, c["T"])

    def soup(self, pattern, size, rng) -> MultiLeniaState:
        return self._init(pattern, size, rng, "soup")

    def uniform(self, pattern, size, rng) -> MultiLeniaState:
        return self._init(pattern, size, rng, "uniform")

    def scramble(self, pattern, size, rng) -> MultiLeniaState:
        return self._init(pattern, size, rng, "scramble")

    def reach(self, pattern) -> int:
        return CREATURES[self.sources[pattern][0]]["R"]

    def step(self, st: MultiLeniaState) -> MultiLeniaState:
        A = model.multi_step(st.A, st.fK, st.m, st.s, st.h, st.Mk, st.Mc, st.T)
        return MultiLeniaState(A, st.fK, st.m, st.s, st.h, st.Mk, st.Mc, st.T)

    def potential(self, st: MultiLeniaState) -> jax.Array:
        """The mean of the kernels' potentials K_k ∗ A_c0(k) (each kernel normalized)."""
        fA = jnp.fft.rfft2(st.A, axes=(0, 1))
        return jnp.fft.irfft2(fA @ st.Mk * st.fK, s=st.A.shape[:2], axes=(0, 1)).mean(-1)

    def observe(self, st: MultiLeniaState) -> jax.Array:
        return st.A.mean(-1)

    def render(self, st: MultiLeniaState) -> jax.Array:
        """The 3 channels as red, green and blue."""
        return st.A[..., :3]

    def perturb(self, st: MultiLeniaState, rng: jax.Array) -> MultiLeniaState:
        """Gaussian noise of standard deviation 1e-5 on every cell, clipped to [0, 1]."""
        A = jnp.clip(st.A + 1e-5 * jax.random.normal(rng, st.A.shape), 0, 1)
        return MultiLeniaState(A, st.fK, st.m, st.s, st.h, st.Mk, st.Mc, st.T)
