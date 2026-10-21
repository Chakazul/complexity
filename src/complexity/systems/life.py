"""Conway's Game of Life on a torus."""

from __future__ import annotations

import re

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
    key = "life"
    name = "Game of Life"
    state_type = "discrete"
    patterns = {
        "gosper_glider_gun": "Gosper glider gun",
        "glider": "Glider",
        "pulsar": "Pulsar (period 3)",
        "r_pentomino": "R-pentomino (methuselah)",
        "acorn": "Acorn (methuselah)",
        "random": "Random soup (50%)",
    }

    def reset(self, pattern: str) -> None:
        self.grid = np.zeros(self.size, dtype=np.uint8)
        if pattern == "random":
            self.grid[:] = self.rng.random(self.size) < 0.5
        else:
            place_center(self.grid, rle2arr(RLE[pattern]))

    def step(self) -> None:
        g = self.grid
        n = sum(
            np.roll(np.roll(g, dy, 0), dx, 1)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if dy or dx
        )
        self.grid = ((n == 3) | ((g == 1) & (n == 2))).astype(np.uint8)

    def frame(self) -> np.ndarray:
        return self.grid.astype(np.float32)
