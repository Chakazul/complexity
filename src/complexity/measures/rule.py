"""Rule-level measures (PLAN.md §5.3): properties of a rule, from runs of their own.

Transient length (TL) and Langton's lambda (LL) apply to any rule; functional information
(FI) varies Lenia's growth parameters. Each is computed once per (system, pattern) and
cached by the run (`session.rule_value`).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from complexity import engine
from complexity.measures.base import NO_CONTEXT, RuleMeasure
from complexity.registry import register_measure
from complexity.systems.base import System


@partial(jax.jit, static_argnums=(0, 2, 3))
def state_hashes(system: System, states, steps: int, levels: int) -> jax.Array:
    """Two uint32 hashes of every frame, from the initial one through `steps` steps, for a
    batch of states: (batch, steps + 1, 2). Frames are quantized to `levels` levels first
    (k / (levels - 1) exactly for a discrete system with `levels = n_states`)."""
    h, w = system.observe(jax.tree.map(lambda x: x[0], states)).shape
    r = jax.random.bits(jax.random.key(0), (2, h, w), jnp.uint32)

    def hash_(frame):
        q = jnp.round(frame * (levels - 1)).astype(jnp.uint32)
        return jnp.sum(q * r, axis=(1, 2), dtype=jnp.uint32)  # wraps mod 2^32

    def run(state):
        def body(s, _):
            s = system.step(s)
            return s, hash_(system.observe(s))

        _, hs = lax.scan(body, state, length=steps)
        return jnp.concatenate([hash_(system.observe(state))[None], hs])

    return jax.vmap(run)(states)


def transient(hashes: np.ndarray) -> int | None:
    """Steps before the first repeated state (the start of its cycle), or None if no state
    repeats. `hashes` is (time, 2)."""
    keys = hashes[:, 0].astype(np.uint64) << np.uint64(32) | hashes[:, 1].astype(np.uint64)
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    repeats = np.nonzero(first[inverse] < np.arange(len(keys)))[0]
    return int(first[inverse[repeats[0]]]) if len(repeats) else None


@register_measure
class TransientLength(RuleMeasure):
    key = "transient_length"
    name = "Transient length"
    unit = "log2 steps"
    description = (
        "How long random initial states take to settle: `batch` random soups of the rule "
        "on a `size`² torus run until a state repeats (exactly for discrete states; to 1e-3 "
        "for continuous ones), and the mean of log2(1 + steps before the cycle) is "
        "reported. A soup that never repeats within `max_steps` counts as `max_steps` (a "
        "lower bound). 0 for still rules, highest for chaotic ones and noise."
    )
    size = 64
    batch = 8
    max_steps = 4096
    params = ("size", "batch", "max_steps")
    temporal = "dynamic"

    def compute_rule(self, system, pattern, ctx=NO_CONTEXT) -> float:
        rngs = jax.random.split(jax.random.key(0), self.batch)
        states = jax.vmap(lambda k: system.soup(pattern, (self.size, self.size), k))(rngs)
        levels = ctx.n_states or 1001
        hashes = np.asarray(state_hashes(system, states, self.max_steps, levels))
        lengths = [transient(h) for h in hashes]
        return float(np.mean([np.log2(1 + (self.max_steps if t is None else t)) for t in lengths]))


@register_measure
class LangtonLambda(RuleMeasure):
    key = "langton_lambda"
    name = "Langton's λ"
    unit = "fraction"
    description = (
        "Langton's lambda: the fraction of the rule table that leads to a non-quiescent "
        "(nonzero) state, i.e. the probability that a uniformly random neighbourhood "
        "produces a live cell. Estimated by stepping a `size`² grid of iid uniform states "
        "once (Life: 140/512 = 0.273). Discrete rules only. Near 0: frozen; around 0.3 for "
        "binary rules: complex; toward 1 - 1/states: chaotic."
    )
    size = 256
    params = ("size",)

    def accepts(self, system) -> bool:
        return (type(system).uniform is not System.uniform
                and all(system.n_states(p) for p in system.patterns))

    def compute_rule(self, system, pattern, ctx=NO_CONTEXT) -> float:
        state = system.uniform(pattern, (self.size, self.size), jax.random.key(0))
        return float(jnp.mean(system.observe(system.step(state)) > 0))


@partial(jax.jit, static_argnums=(0, 2))
def _final_frames(system: System, states, steps: int) -> jax.Array:
    return jax.vmap(lambda s: system.observe(engine.advance(system, s, steps)))(states)


@register_measure
class FunctionalInformation(RuleMeasure):
    key = "functional_information"
    name = "Functional information"
    unit = "bits"
    description = (
        "Hazen's functional information, -log2 of the fraction of rules that perform a "
        "function: `samples` random growth parameters m and s, each within ±`spread` "
        "(relative) of the pattern's own (the rest of the rule and the cells are the "
        "pattern's), run for `steps` steps on a `size`² torus; the function is that the "
        "pattern lives on (mass within 1/2 to 2x the start, area at most 2x). High = only "
        "a narrow range of rules keeps the pattern alive. None found counts as half of one "
        "(a lower bound)."
    )
    peaks_at = "structure"
    samples = 128
    spread = 0.5
    steps = 500
    size = 128
    params = ("samples", "spread", "steps", "size")
    temporal = "dynamic"

    def accepts(self, system) -> bool:
        return hasattr(system, "growth_params")

    def compute_rule(self, system, pattern, ctx=NO_CONTEXT) -> float:
        size, key = (self.size, self.size), jax.random.key(0)
        m0, s0 = (float(v) for v in system.growth_params(pattern))
        rng = np.random.default_rng(0)
        ms = m0 * rng.uniform(1 - self.spread, 1 + self.spread, self.samples)
        ss = s0 * rng.uniform(1 - self.spread, 1 + self.spread, self.samples)
        states = [system.init_with(pattern, size, key, m=m, s=s) for m, s in zip(ms, ss)]
        states = jax.tree.map(lambda *x: jnp.stack(x), *states)
        start = np.asarray(system.observe(jax.tree.map(lambda x: x[0], states)))
        end = np.asarray(_final_frames(system, states, self.steps))
        mass = end.sum(axis=(1, 2)) / start.sum()
        area = (end > 0.1).sum(axis=(1, 2)) / max((start > 0.1).sum(), 1)
        works = (mass >= 0.5) & (mass <= 2) & (area <= 2)
        return float(-np.log2(max(works.sum(), 0.5) / self.samples))
