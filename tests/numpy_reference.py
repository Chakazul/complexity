"""The M0 NumPy Game of Life step, kept as the reference for the JAX parity test
(PLAN.md §6) after it was removed from the package."""

import numpy as np


def life_step(g: np.ndarray) -> np.ndarray:
    n = sum(
        np.roll(np.roll(g, dy, 0), dx, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if dy or dx
    )
    return ((n == 3) | ((g == 1) & (n == 2))).astype(np.uint8)

