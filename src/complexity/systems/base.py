"""The interface every dynamical system implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class System(ABC):
    """A dynamical system on a 2-D grid, observed one frame per step.

    Subclasses set the class attributes and implement `reset`, `step` and `frame`.
    Measures see only `frame()`, so a system with a different internal state (particles,
    multiple channels) renders itself to a grid there.
    """

    key: str  # registry key, e.g. "life"
    name: str  # display name
    state_type: str  # "discrete" or "continuous"; measures may accept only one
    patterns: dict[str, str]  # pattern key -> display name; the first is the default
    default_size: tuple[int, int] = (128, 128)  # (height, width)

    def __init__(self, size: tuple[int, int] | None = None, seed: int = 0):
        self.size = tuple(size or self.default_size)
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def reset(self, pattern: str) -> None:
        """Set the initial state to the named pattern."""

    @abstractmethod
    def step(self) -> None:
        """Advance the system by one time step."""

    @abstractmethod
    def frame(self) -> np.ndarray:
        """The current observation: float32 array (height, width) with values in [0, 1]."""


def place_center(grid: np.ndarray, pattern: np.ndarray) -> None:
    """Write `pattern` into the center of `grid` in place."""
    h, w = pattern.shape
    y, x = (grid.shape[0] - h) // 2, (grid.shape[1] - w) // 2
    grid[y : y + h, x : x + w] = pattern
