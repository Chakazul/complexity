"""Measures of the dynamics rather than the frames (MEASURES.md §5.6): the Lyapunov exponent
(LE), from a twin run started from a perturbed copy of the current state.

These need the system and its state, so they set `needs_state` and implement
`compute_state`; the system must implement `perturb`.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from complexity import engine
from complexity.measures.base import NO_CONTEXT, Measure
from complexity.registry import register_measure
from complexity.systems.base import System


@partial(jax.jit, static_argnums=(0, 3, 4, 5))
def divergence_rate(system: System, state, rng: jax.Array, steps: int, burn_in: int,
                    discrete: bool):
    """Growth rate of a small perturbation, per step.

    Continuous: ln(d_steps / d_burn_in) / (steps - burn_in), d the Euclidean distance
    between the frames. Most of a random perturbation lies in stable directions and dies out
    within a few steps (Lenia: ~4x smaller by step 8), so the rate is taken after a burn-in,
    once the surviving perturbation points along the fastest-growing direction. The result is
    the larger of the rates during and after the burn-in, so a perturbation that saturates
    within the burn-in (iid noise decorrelates in one step) still counts as fast growth.
    d is floored at 1e-6 of its previous value, so a perturbation that dies out gives a
    finite negative rate. Discrete: log2(1 + cells that differ after `steps`) / steps, the damage-spreading
    rate of one changed cell (0 when the damage heals).
    """
    twin = system.perturb(state, rng)
    if discrete:
        a = system.observe(engine.advance(system, state, steps))
        b = system.observe(engine.advance(system, twin, steps))
        return jnp.log2(1 + jnp.sum(a != b)) / steps
    def distance(floor):
        return jnp.maximum(jnp.linalg.norm(system.observe(twin) - system.observe(state)), floor)

    d_init = distance(1e-30)
    state, twin = engine.advance(system, state, burn_in), engine.advance(system, twin, burn_in)
    d_burn = distance(1e-6 * d_init)
    state = engine.advance(system, state, steps - burn_in)
    twin = engine.advance(system, twin, steps - burn_in)
    d = distance(1e-6 * d_burn)
    return jnp.maximum(jnp.log(d_burn / d_init) / burn_in, jnp.log(d / d_burn) / (steps - burn_in))


@partial(jax.jit, static_argnums=(0, 3, 4))
def separation_field(system: System, state, rng: jax.Array, steps: int, discrete: bool):
    """Where the twin run has diverged after `steps` steps, 0 to 1: 1 where cells differ
    (discrete); log10(1 + |difference| / 1e-5) / 5 (continuous: the orders of magnitude
    grown beyond the perturbation's size, out of the 5 up to a difference of 1)."""
    twin = system.perturb(state, rng)
    a = system.observe(engine.advance(system, state, steps))
    b = system.observe(engine.advance(system, twin, steps))
    if discrete:
        return (a != b).astype(jnp.float32)
    return jnp.log10(1 + jnp.abs(b - a) / 1e-5) / 5


@register_measure
class Lyapunov(Measure):
    key = "lyapunov"
    icon = "🦋"
    equation = r"\lambda = \lim_{t\to\infty} \frac{1}{t} \ln \frac{|\Delta x(t)|}{|\Delta x(0)|}"
    reference = "Lyapunov 1892; for Lenia: Yevenko et al. 2025"
    name = "Lyapunov exponent"
    unit = "1/step"
    group = "Dynamics"
    description = (
        "How fast a small perturbation grows, from a twin run of `steps` steps started at "
        "the current state. Continuous states: finite-time estimate of the maximal "
        "Lyapunov exponent, ln(separation growth) per step after a `burn_in` in which the "
        "stable part of a 1e-5 Gaussian perturbation dies out (or during it, if larger: "
        "noise saturates at once); negative when perturbations die out. Discrete states: damage spreading, log2(1 + cells that differ) / steps "
        "after changing one active cell. Positive = chaotic."
    )
    needs_state = True
    map = "local"
    map_unit = "divergence (0-1)"
    steps = 64
    burn_in = 8
    params = ("steps", "burn_in")
    granularity = "rule"  # the exponent belongs to the dynamics (MEASURES.md §2.3)
    subject = "trajectory"  # a finite-time estimate from the current state
    estimated_from = "run"
    applies_to = "systems that can perturb their state (all built-in ones)"
    temporal = "dynamic"

    def accepts(self, system) -> bool:
        return type(system).perturb is not System.perturb

    def compute(self, frames, ctx=NO_CONTEXT) -> float:
        raise TypeError("lyapunov needs the system state: use compute_state")

    def compute_state_map(self, system, state, rng, ctx=NO_CONTEXT) -> np.ndarray:
        """The twin's separation field after `steps` steps (see `separation_field`)."""
        discrete = ctx.n_states is not None
        return np.asarray(separation_field(system, state, rng, self.steps, discrete))

    def compute_state(self, system, state, rng, ctx=NO_CONTEXT) -> float:
        discrete = ctx.n_states is not None
        return float(divergence_rate(system, state, rng, self.steps, self.burn_in, discrete))
