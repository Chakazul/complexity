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
import numpy as np

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

    def configuration(self, state: State) -> jax.Array:
        """The dynamical state alone as a frame, for measures that compare states (transient
        length, Langton's λ): the frame itself, unless the frame shows more than the state
        (an elementary CA's space-time diagram: its current row). Pure and jit-able."""
        return self.observe(state)

    def render(self, state: State) -> jax.Array | None:
        """An RGB picture (height, width, 3) in [0, 1] for display, or None to show the
        frame in a colour map. Measures never see it. Optional."""
        return None

    def reach(self, pattern: str) -> int:
        """How far a cell's next state looks, in cells (the neighbourhood's radius): 1 for
        Life, R for Lenia. Rule-level measures scale grid sizes by it."""
        return 1

    def soup(self, pattern: str, size: tuple[int, int], rng: jax.Array) -> State:
        """A random initial state under the pattern's rule, for rule-level measures (transient
        length). Default: the "random" pattern if there is one, else the pattern itself."""
        return self.init("random" if "random" in self.patterns else pattern, size, rng)

    def uniform(self, pattern: str, size: tuple[int, int], rng: jax.Array) -> State:
        """Every cell iid uniform over the states, under the pattern's rule (Langton's
        lambda). Optional, like `perturb`."""
        raise NotImplementedError

    def scramble(self, pattern: str, size: tuple[int, int], rng: jax.Array) -> State:
        """A random configuration of the pattern's size: its bounding box (for a soup, the
        soup's area) filled with iid uniform states, placed as `init` places the pattern,
        under the pattern's rule (functional information). Optional, like `perturb`."""
        raise NotImplementedError

    def rule_table(self, pattern: str) -> np.ndarray:
        """A 1-D binary rule's outputs, indexed by neighbourhood value (the left cell the
        high bit), length 2^K for K cells (Wuensche's Z). Optional, like `perturb`."""
        raise NotImplementedError

    def transition_map(self, pattern: str, n: int) -> np.ndarray:
        """Each configuration's successor on a lattice of `n` cells, configurations numbered
        0..states^n - 1: the whole state-transition graph of a tiny lattice (thermodynamic
        depth). Optional, like `perturb`."""
        raise NotImplementedError

    def cells(self, pattern: str) -> np.ndarray:
        """The pattern's own cells, as the state holds them ((h, w), or (h, w, channels)),
        for building scenes (collision experiments). Optional."""
        raise NotImplementedError

    def with_cells(self, pattern: str, A) -> State:
        """A state with the pattern's rule and cells A (the shape of a full grid, as
        `cells` returns them). Optional, with `cells`."""
        raise NotImplementedError

    def replace_cells(self, state: State, A) -> State:
        """`state` with its cells replaced by A, the rule's parts (kernels) kept: cheap and
        traceable, so a batch of cell arrays can share one state under `vmap` (twin runs of
        the goal measure). Optional, with `with_cells`."""
        raise NotImplementedError

    def potential(self, state: State) -> jax.Array:
        """The neighbour sum K ∗ A that the rule reads, (height, width) in [0, 1]: how the
        model "sees" the world (object detection, MEASURES.md §5.4). Optional; measures
        fall back to the frame. Pure and jit-able."""
        raise NotImplementedError

    def perturb(self, state: State, rng: jax.Array) -> State:
        """A slightly different state, for measures of the dynamics (Lyapunov exponent):
        one active cell changed by one level if discrete, small noise if continuous.
        Optional; measures that need it skip systems without it. Pure and jit-able."""
        raise NotImplementedError


def random_active_cell(active: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
    """(row, column) of a random cell where `active` is True, or of any cell if none is."""
    logits = jnp.where(active.any(), jnp.where(active, 0.0, -jnp.inf), 0.0).ravel()
    return jnp.unravel_index(jax.random.categorical(rng, logits), active.shape)


def place_center(grid: jax.Array, pattern) -> jax.Array:
    """`grid` with `pattern` written into its center. A pattern larger than the grid is
    cropped to its central part."""
    pattern = jnp.asarray(pattern, grid.dtype)
    (h, H), (w, W) = zip(pattern.shape, grid.shape)
    pattern = pattern[max(0, (h - H) // 2) :][:H, max(0, (w - W) // 2) :][:, :W]
    h, w = pattern.shape
    y, x = (grid.shape[0] - h) // 2, (grid.shape[1] - w) // 2
    return grid.at[y : y + h, x : x + w].set(pattern)
