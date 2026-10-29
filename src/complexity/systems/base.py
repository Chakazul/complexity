"""The interface every dynamical system implements (PLAN.md §3.2).

A system is a set of pure functions over a state pytree, so that JAX can compile the step
once and scan or vmap it: `init` builds a state, `step` advances it, `observe` renders it to
a frame. The system object itself holds only static, hashable parameters; it is part of the
jit cache key, which is why systems compare equal by type and parameters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.numpy as jnp

State = Any  # a pytree of arrays (plus static metadata, see jax.tree_util.register_dataclass)


class System(ABC):
    """A dynamical system on a 2-D grid, observed one frame per step.

    Subclasses set the class attributes and implement `init`, `step` and `observe`.
    Measures see only frames, so a system with a different internal state (particles,
    multiple channels) renders itself to a grid in `observe`.
    """

    key: str  # registry key, e.g. "life"
    name: str  # display name
    patterns: dict[str, str]  # pattern key -> display name; the first is the default
    default_size: tuple[int, int] = (128, 128)  # (height, width)

    def __init__(self, **params):
        self.params = params  # static and hashable; part of the jit cache key

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and self.params == other.params

    def __hash__(self) -> int:
        return hash((type(self), tuple(sorted(self.params.items()))))

    def n_states(self, pattern: str) -> int | None:
        """Number of discrete cell states (frames take values k / (n - 1)), or None if
        continuous. A method, since one system may run discrete and continuous patterns."""
        return None

    @abstractmethod
    def init(self, pattern: str, size: tuple[int, int], rng: jax.Array) -> State:
        """The initial state for the named pattern on a (height, width) torus."""

    @abstractmethod
    def step(self, state: State) -> State:
        """Advance by one time step. Pure and jit-able."""

    @abstractmethod
    def observe(self, state: State) -> jax.Array:
        """The current frame: float32 (height, width) with values in [0, 1]. Pure and jit-able."""


def place_center(grid: jax.Array, pattern) -> jax.Array:
    """`grid` with `pattern` written into its center."""
    pattern = jnp.asarray(pattern, grid.dtype)
    h, w = pattern.shape
    y, x = (grid.shape[0] - h) // 2, (grid.shape[1] - w) // 2
    return grid.at[y : y + h, x : x + w].set(pattern)
