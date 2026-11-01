"""Conway's Game of Life (B3/S23) on a torus."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from complexity.registry import register_system
from complexity.systems.base import System, place_center, random_active_cell
from complexity.vendor.lenia import ANIMALS
from complexity.vendor.lenia.rle import rle2arr

# Standard Life RLE (b = dead, o = alive, $ = end of row, ! = end).
RLE = {"acorn": "bo5b$3bo3b$2o2b3o!"}

# pattern -> index in the original Lenia catalogue (vendor/lenia/), whose last 16 entries
# are Life patterns
CATALOGUE = {
    "glider": 598, "gosper_glider_gun": 602, "lwss": 599, "mwss": 600, "hwss": 601,
    "r_pentomino": 603, "copperhead": 604, "galaxy": 605, "weekender": 606, "lobster": 607,
    "sir_robin": 608, "pufferfish": 609, "switch_engine": 610, "max": 611,
    "sidecar_gun": 612, "star_gate": 613,
}


def cells(pattern: str) -> np.ndarray:
    """The pattern's uint8 0/1 cells."""
    if pattern in CATALOGUE:
        return (rle2arr(ANIMALS[CATALOGUE[pattern]]["cells"]) > 0).astype(np.uint8)
    return (rle2arr(RLE[pattern]) > 0).astype(np.uint8)


@register_system
class GameOfLife(System):
    """State: uint8 (height, width) of 0/1."""

    key = "life"
    name = "Game of Life"
    patterns = {  # catalogue order, but the two methuselahs together before the soup
        **{k: ANIMALS[i]["name"] for k, i in CATALOGUE.items() if k != "r_pentomino"},
        "gosper_glider_gun": "Gosper glider gun (gun)",
        "r_pentomino": ANIMALS[CATALOGUE["r_pentomino"]]["name"],
        "acorn": "Acorn (methuselah 5206)",
        "random": "Random soup (50%)",
    }

    def n_states(self, pattern: str) -> int:
        return 2

    def init(self, pattern: str, size: tuple[int, int], rng: jax.Array) -> jax.Array:
        if pattern == "random":
            return (jax.random.uniform(rng, size) < 0.5).astype(jnp.uint8)
        return place_center(jnp.zeros(size, jnp.uint8), cells(pattern))

    def step(self, grid: jax.Array) -> jax.Array:
        n = sum(
            jnp.roll(grid, (dy, dx), (0, 1))
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if dy or dx
        )
        return ((n == 3) | ((grid == 1) & (n == 2))).astype(jnp.uint8)

    def cells(self, pattern: str) -> np.ndarray:
        return cells(pattern)

    def with_cells(self, pattern: str, A) -> jax.Array:
        return jnp.asarray(A, jnp.uint8)

    def replace_cells(self, grid: jax.Array, A) -> jax.Array:
        return jnp.asarray(A).astype(jnp.uint8)

    def potential(self, grid: jax.Array) -> jax.Array:
        """The live-neighbour count over 8."""
        n = sum(jnp.roll(grid, (dy, dx), (0, 1)) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if dy or dx)
        return n.astype(jnp.float32) / 8

    def observe(self, grid: jax.Array) -> jax.Array:
        return grid.astype(jnp.float32)

    def uniform(self, pattern: str, size: tuple[int, int], rng: jax.Array) -> jax.Array:
        return self.init("random", size, rng)  # 50 % soup over the whole grid

    def scramble(self, pattern: str, size: tuple[int, int], rng: jax.Array) -> jax.Array:
        if pattern == "random":
            return self.init("random", size, rng)
        box = jax.random.bernoulli(rng, 0.5, cells(pattern).shape).astype(jnp.uint8)
        return place_center(jnp.zeros(size, jnp.uint8), box)

    def perturb(self, grid: jax.Array, rng: jax.Array) -> jax.Array:
        """Flip one live cell (any cell if none is alive)."""
        y, x = random_active_cell(grid == 1, rng)
        return grid.at[y, x].set(1 - grid[y, x])
