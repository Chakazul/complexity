"""Reference "systems" that calibrate every measure (PLAN.md §4.3): an empty grid, one
random frame held still, and a fresh random frame every step.

Noise comes binary (as Game of Life), with 13 levels (as Integer Lenia's minimum Orbium) or
continuous (as Lenia), so each system can be compared with noise of its own kind. The level
count is part of the pattern, so the UI and CLI choose it like any other pattern.

1-D iid noise (`iid_noise_1d_binary`) is the reference for elementary CA: a space-time
diagram of iid random rows that scrolls up one row per step, a fresh row at the bottom, as
an elementary CA's frame does.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from complexity.registry import register_system
from complexity.systems.base import System

# noise pattern suffix -> discrete levels (0 = continuous)
LEVELS = {"binary": 2, "13": 13, "continuous": 0}
LEVEL_NAMES = {"binary": "binary", "13": "13 levels", "continuous": "continuous"}


def parse(pattern: str) -> tuple[str, int]:
    """(kind, levels) of a pattern key, e.g. "iid_noise_13" -> ("iid_noise", 13)."""
    if pattern == "empty":
        return "empty", 2
    kind, suffix = pattern.rsplit("_", 1)
    return kind, LEVELS[suffix]


def noise(rng: jax.Array, size: tuple[int, int], levels: int) -> jax.Array:
    """Uniform noise in [0, 1): continuous, or the values k / (levels - 1)."""
    if levels == 0:
        return jax.random.uniform(rng, size)
    return jax.random.randint(rng, size, 0, levels).astype(jnp.float32) / (levels - 1)


@partial(jax.tree_util.register_dataclass, data_fields=["A", "rng"],
         meta_fields=["levels", "fresh", "rows"])
@dataclass(frozen=True)
class BaselineState:
    A: jax.Array  # float32 frame
    rng: jax.Array
    levels: int  # 0 = continuous
    fresh: bool  # new noise every step
    rows: bool = False  # 1-D: only a new bottom row, the rest scrolled up


@register_system
class Baseline(System):
    key = "baseline"
    name = "Baseline"
    patterns = {
        "empty": "Empty",
        **{f"{kind}_{lv}": f"{label} ({LEVEL_NAMES[lv]})"
           for kind, label in (("iid_noise", "IID noise"), ("frozen_noise", "Frozen noise"))
           for lv in LEVELS},
        "iid_noise_1d_binary": "IID noise, 1-D rows (binary; for elementary CA)",
    }

    def n_states(self, pattern: str) -> int | None:
        return parse(pattern)[1] or None

    def init(self, pattern: str, size: tuple[int, int], rng: jax.Array) -> BaselineState:
        kind, levels = parse(pattern)
        rng, sub = jax.random.split(rng)
        A = jnp.zeros(size, jnp.float32) if kind == "empty" else noise(sub, size, levels)
        return BaselineState(A, rng, levels, kind.startswith("iid_noise"), kind == "iid_noise_1d")

    def step(self, state: BaselineState) -> BaselineState:
        if not state.fresh:
            return state
        rng, sub = jax.random.split(state.rng)
        if state.rows:
            row = noise(sub, state.A.shape[1:], state.levels)
            return replace(state, A=jnp.roll(state.A, -1, 0).at[-1].set(row), rng=rng)
        return replace(state, A=noise(sub, state.A.shape, state.levels), rng=rng)

    def configuration(self, state: BaselineState) -> jax.Array:
        return state.A[-1:] if state.rows else state.A

    def observe(self, state: BaselineState) -> jax.Array:
        return state.A

    def cells(self, pattern: str) -> np.ndarray:
        return np.zeros((1, 1), np.float32)

    def with_cells(self, pattern: str, A) -> BaselineState:
        """The pattern's noise kind with cells A, and a noise stream of its own: a copy
        with other cells is a different run (iid noise diverges at once)."""
        st = self.init(pattern, np.shape(A), jax.random.key(1))
        return replace(st, A=jnp.asarray(A, jnp.float32))

    def replace_cells(self, state: BaselineState, A) -> BaselineState:
        """Other cells and another noise stream, as `with_cells`."""
        return replace(state, A=jnp.asarray(A, jnp.float32), rng=jax.random.fold_in(state.rng, 1))

    def perturb(self, state: BaselineState, rng: jax.Array) -> BaselineState:
        """One cell to the next level (discrete) or small noise (continuous), and a different
        noise stream, so a perturbed iid noise run diverges completely."""
        if state.levels:
            y, x = jax.random.randint(rng, (2,), 0, jnp.array(state.A.shape))
            step = 1 / (state.levels - 1)
            A = state.A.at[y, x].set(jnp.where(state.A[y, x] + step > 1 + 1e-6, 0.0,
                                               state.A[y, x] + step))
        else:
            A = jnp.clip(state.A + 1e-5 * jax.random.normal(rng, state.A.shape), 0, 1)
        return replace(state, A=A, rng=jax.random.fold_in(state.rng, 1))
