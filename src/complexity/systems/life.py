"""Conway's Game of Life (B3/S23) on a torus."""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np

from complexity.registry import register_system
from complexity.systems.base import System, place_center

# Standard Life RLE (b = dead, o = alive, $ = end of row, ! = end).
RLE = {
    "glider": "bob$2bo$3o!",
    "gosper_glider_gun": (
        "24bo$22bobo$12b2o6b2o12b2o$11bo3bo4b2o12b2o$2o8bo5bo3b2o$2o8bo3bob2o4bobo$"
        "10bo5bo7bo$11bo3bo$12b2o!"
    ),
    "r_pentomino": "b2o$2ob$bo!",
    "acorn": "bo5b$3bo3b$2o2b3o!",
    "pulsar": (
        "2b3o3b3o2b2$o4bobo4bo$o4bobo4bo$o4bobo4bo$2b3o3b3o2b2$2b3o3b3o2b$"
        "o4bobo4bo$o4bobo4bo$o4bobo4bo2$2b3o3b3o!"
    ),
}


def rle2arr(text: str) -> np.ndarray:
    """Decode a Life RLE string to a uint8 array of 0/1."""
    rows, row = [], []
    for count, tag in re.findall(r"(\d*)([bo$!])", text):
        n = int(count) if count else 1
        if tag in "bo":
            row.extend([int(tag == "o")] * n)
        elif tag == "$":
            rows.append(row)
            rows.extend([[]] * (n - 1))
            row = []
        else:
            break
    rows.append(row)
    width = max(len(r) for r in rows)
    return np.array([r + [0] * (width - len(r)) for r in rows], dtype=np.uint8)


@register_system
class GameOfLife(System):
    """State: uint8 (height, width) of 0/1."""

    key = "life"
    name = "Game of Life"
    patterns = {
        "gosper_glider_gun": "Gosper glider gun",
        "glider": "Glider",
        "pulsar": "Pulsar (period 3)",
        "r_pentomino": "R-pentomino (methuselah)",
        "acorn": "Acorn (methuselah)",
        "random": "Random soup (50%)",
    }

    def n_states(self, pattern: str) -> int:
        return 2

    def init(self, pattern: str, size: tuple[int, int], rng: jax.Array) -> jax.Array:
        if pattern == "random":
            return (jax.random.uniform(rng, size) < 0.5).astype(jnp.uint8)
        return place_center(jnp.zeros(size, jnp.uint8), rle2arr(RLE[pattern]))

    def step(self, grid: jax.Array) -> jax.Array:
        n = sum(
            jnp.roll(grid, (dy, dx), (0, 1))
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if dy or dx
        )
        return ((n == 3) | ((grid == 1) & (n == 2))).astype(jnp.uint8)

    def observe(self, grid: jax.Array) -> jax.Array:
        return grid.astype(jnp.float32)
