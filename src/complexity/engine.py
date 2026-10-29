"""Compiled rollouts of a system: `lax.scan` over steps, `vmap` over a batch (PLAN.md §3.1).

The system is a static argument, so each (system, grid size, pattern parameters) compiles
once and is cached by jit.
"""

from __future__ import annotations

from functools import partial

import jax
from jax import lax

from complexity.systems.base import State, System


def _rollout(system: System, state: State, steps: int, every: int):
    def chunk(s, _):
        s = lax.fori_loop(0, every, lambda _, s: system.step(s), s)
        return s, system.observe(s)

    return lax.scan(chunk, state, length=steps // every)


@partial(jax.jit, static_argnums=(0, 2, 3))
def rollout(system: System, state: State, steps: int, every: int = 1):
    """Advance `steps` steps; returns (final state, frames observed every `every` steps),
    frames float32 (steps // every, height, width)."""
    return _rollout(system, state, steps, every)


@partial(jax.jit, static_argnums=(0, 2, 3))
def rollout_batch(system: System, states: State, steps: int, every: int = 1):
    """`rollout` over a leading batch axis of `states`; frames (batch, steps // every, h, w)."""
    return jax.vmap(lambda s: _rollout(system, s, steps, every))(states)


@partial(jax.jit, static_argnums=0)
def advance(system: System, state: State, steps) -> State:
    """Advance `steps` steps without observing. `steps` is traced: no recompile per count."""
    return lax.fori_loop(0, steps, lambda _, s: system.step(s), state)


def init_batch(system: System, pattern: str, size: tuple[int, int], rngs: jax.Array) -> State:
    """States for a batch of seeds, stacked on a leading axis (e.g. random soups)."""
    return jax.vmap(lambda rng: system.init(pattern, size, rng))(rngs)
