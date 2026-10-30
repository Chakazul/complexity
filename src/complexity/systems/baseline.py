"""Reference "systems" that calibrate every measure (PLAN.md §4.3): an empty grid, one
random frame held still, and a fresh random frame every step.

Noise comes binary (as Game of Life), with 13 levels (as Integer Lenia's minimum Orbium) or
continuous (as Lenia), so each system can be compared with noise of its own kind. The level
count is part of the pattern, so the UI and CLI choose it like any other pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

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
         meta_fields=["levels", "fresh"])
@dataclass(frozen=True)
class BaselineState:
    A: jax.Array  # float32 frame
    rng: jax.Array
    levels: int  # 0 = continuous
    fresh: bool  # new noise every step


@register_system
class Baseline(System):
    key = "baseline"
    name = "Baseline"
    patterns = {
        "empty": "Empty",
        **{f"{kind}_{lv}": f"{label} ({LEVEL_NAMES[lv]})"
           for kind, label in (("iid_noise", "IID noise"), ("frozen_noise", "Frozen noise"))
           for lv in LEVELS},
    }

    def n_states(self, pattern: str) -> int | None:
        return parse(pattern)[1] or None

    def init(self, pattern: str, size: tuple[int, int], rng: jax.Array) -> BaselineState:
        kind, levels = parse(pattern)
        rng, sub = jax.random.split(rng)
        A = jnp.zeros(size, jnp.float32) if kind == "empty" else noise(sub, size, levels)
        return BaselineState(A, rng, levels, kind == "iid_noise")

    def step(self, state: BaselineState) -> BaselineState:
        if not state.fresh:
            return state
        rng, sub = jax.random.split(state.rng)
        return BaselineState(noise(sub, state.A.shape, state.levels), rng, state.levels, True)

    def observe(self, state: BaselineState) -> jax.Array:
        return state.A
