"""Rule-level measures (MEASURES.md §5.7): properties of a rule, from runs of their own.

Transient length (TL) and Langton's lambda (LL) apply to any rule; functional information
(FI) needs random patterns of the pattern's size (`System.scramble`). Each is computed once per (system, pattern) and
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
def state_hashes(system: System, states, steps: int, levels: int):
    """Advance a batch of states by `steps` and hash every frame on the way: (states,
    hashes of shape (batch, steps, 2)), two uint32 hashes per frame. Frames are quantized to
    `levels` levels first (k / (levels - 1) exactly for a discrete system with
    `levels = n_states`)."""
    h, w = system.observe(jax.tree.map(lambda x: x[0], states)).shape
    r = jax.random.bits(jax.random.key(0), (2, h, w), jnp.uint32)

    def hash_(frame):
        q = jnp.round(frame * (levels - 1)).astype(jnp.uint32)
        return jnp.sum(q * r, axis=(1, 2), dtype=jnp.uint32)  # wraps mod 2^32

    def run(state):
        def body(s, _):
            s = system.step(s)
            return s, hash_(system.observe(s))

        return lax.scan(body, state, length=steps)

    return jax.vmap(run)(states)


@partial(jax.jit, static_argnums=(0, 2))
def _initial_hashes(system: System, states, levels: int) -> jax.Array:
    h, w = system.observe(jax.tree.map(lambda x: x[0], states)).shape
    r = jax.random.bits(jax.random.key(0), (2, h, w), jnp.uint32)
    q = jnp.round(jax.vmap(system.observe)(states) * (levels - 1)).astype(jnp.uint32)
    return jnp.sum(q[:, None] * r, axis=(2, 3), dtype=jnp.uint32)[:, None]


def transients(system: System, states, max_steps: int, levels: int,
               chunk: int = 256) -> list[int | None]:
    """The transient of each state in a batch (None if no state repeats within
    `max_steps`), run in chunks that stop once every state has repeated."""
    hashes = [np.asarray(_initial_hashes(system, states, levels))]
    done = 0
    while done < max_steps:
        states, hs = state_hashes(system, states, chunk, levels)
        hashes.append(np.asarray(hs))
        done += chunk
        lengths = [transient(h[: max_steps + 1]) for h in np.concatenate(hashes, axis=1)]
        if None not in lengths:
            break
    return lengths


def transient(hashes: np.ndarray) -> int | None:
    """Steps before the first repeated state (the start of its cycle), or None if no state
    repeats. `hashes` is (time, 2)."""
    keys = hashes[:, 0].astype(np.uint64) << np.uint64(32) | hashes[:, 1].astype(np.uint64)
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    repeats = np.nonzero(first[inverse] < np.arange(len(keys)))[0]
    return int(first[inverse[repeats[0]]]) if len(repeats) else None


GROWTH_CLASSES = ("bounded", "log", "linear", "polynomial", "exponential")


def growth_class(sizes, means) -> int:
    """Hudcová & Mikolov's transient class of mean transients `means` at grid sides `sizes`:
    an index into GROWTH_CLASSES. Bounded if the means change by less than a step or a
    quarter of their mean; otherwise the best R² (in transient steps) among log, linear,
    polynomial (power law) and exponential fits."""
    n, mu = np.asarray(sizes, float), np.asarray(means, float)
    if np.ptp(mu) < max(1.0, 0.25 * mu.mean()):
        return 0

    def r2(pred):
        return 1 - np.sum((mu - pred) ** 2) / np.sum((mu - mu.mean()) ** 2)

    def fit(x, y):  # least squares y = a + b x, predicted y
        return np.polyval(np.polyfit(x, y, 1), x)

    scores = [
        r2(fit(np.log(n), mu)),
        r2(fit(n, mu)),
        r2(np.expm1(fit(np.log(n), np.log1p(mu)))),
        r2(np.expm1(fit(n, np.log1p(mu)))),
    ]
    return 1 + int(np.argmax(scores))


@register_measure
class TransientLength(RuleMeasure):
    key = "transient_length"
    icon = "⏳"
    equation = r"t_u = \min\{\,i : \exists\, j > i,\ F^i(u) = F^j(u)\,\},\qquad \mu_n = \mathbb{E}_u[t_u] \sim 1,\ \log n,\ n,\ n^k,\ e^{bn}"
    reference = "Hudcová & Mikolov 2020"
    name = "Transient length"
    unit = "class"
    description = (
        "Hudcová & Mikolov's transient classification: how the mean transient (steps before "
        "the state first repeats) of random soups grows "
        "with the grid side n, over `n_sizes` sides from `min_size` to `max_size` (for a "
        "rule with a wider neighbourhood, from just above its width, same ratio). "
        "`report` class (default): 0 bounded, 1 log, 2 linear, 3 polynomial, 4 exponential, "
        "the best R² fit; or exponent: the slope of log(1 + mean) against log n. Soups come "
        "in batches of `batch` until the mean is known to 10 % (95 % confidence) or "
        "`max_batches` are run. A soup that does not repeat within `max_steps` counts as "
        "`max_steps`; sizes where more than a quarter do are left out of the fit, and with "
        "fewer than 3 sizes left the class is exponential. Chaotic rules are exponential; Life is linear; complex 2-D rules are "
        "linear (steep) or polynomial."
    )
    min_size = 16
    max_size = 96
    n_sizes = 6
    batch = 8
    max_batches = 4
    max_steps = 4096
    report = "class"
    params = ("report", "min_size", "max_size", "n_sizes", "batch", "max_batches", "max_steps")
    choices = {"report": ("class", "exponent")}
    temporal = "dynamic"
    applies_to = "discrete rules (Game of Life, Integer Lenia): continuous states never repeat"

    def accepts(self, system) -> bool:
        return all(system.n_states(p) for p in system.patterns)

    def sizes(self, reach: int = 1) -> list[int]:
        """Grid sides from `min_size` to `max_size`, geometrically spaced; for a rule that
        looks further than a 3x3 neighbourhood the range starts just above its
        neighbourhood's width (2 reach + 2) and keeps its ratio."""
        lo = max(self.min_size, 2 * reach + 2)
        hi = self.max_size * lo / self.min_size
        return sorted({round(float(n)) for n in np.geomspace(lo, hi, self.n_sizes)})

    def mean_transient(self, system, pattern, n, levels) -> tuple[float, float]:
        """The mean transient at side n and the fraction of soups that hit `max_steps`."""
        lengths = []
        for b in range(self.max_batches):
            rngs = jax.random.split(jax.random.key(b), self.batch)
            states = jax.vmap(lambda k: system.soup(pattern, (n, n), k))(rngs)
            lengths += transients(system, states, self.max_steps, levels)
            t = np.array([self.max_steps if x is None else x for x in lengths], float)
            if t.mean() == 0 or 1.96 * t.std(ddof=1) / np.sqrt(len(t)) / t.mean() < 0.1:
                break
        return float(t.mean()), lengths.count(None) / len(lengths)

    def compute_rule(self, system, pattern, ctx=NO_CONTEXT) -> float:
        levels = ctx.n_states
        sizes = self.sizes(system.reach(pattern))
        results = [self.mean_transient(system, pattern, n, levels) for n in sizes]
        means = [m for m, _ in results]
        if self.report == "exponent":
            return float(np.polyfit(np.log(sizes), np.log1p(means), 1)[0])
        free = [(n, m) for n, (m, capped) in zip(sizes, results) if capped <= 0.25]
        if len(free) < 3:
            return 4.0
        return float(growth_class(*zip(*free)))


@register_measure
class LangtonLambda(RuleMeasure):
    key = "langton_lambda"
    icon = "🎚️"
    equation = r"\lambda = \frac{K^N - n_q}{K^N}"
    reference = "Langton 1990"
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
    applies_to = "discrete rules (Game of Life, Integer Lenia)"

    def accepts(self, system) -> bool:
        return (type(system).uniform is not System.uniform
                and all(system.n_states(p) for p in system.patterns))

    def compute_rule(self, system, pattern, ctx=NO_CONTEXT) -> float:
        state = system.uniform(pattern, (self.size, self.size), jax.random.key(0))
        return float(jnp.mean(system.observe(system.step(state)) > 0))


@partial(jax.jit, static_argnums=0, static_argnames=("steps", "checks"))
def survival(system: System, states, steps: int, checks: int, max_area) -> jax.Array:
    """Hazen's degree of function for a batch of states: the number of steps (in multiples
    of steps / checks) each lives on as a bounded object — mass within 1/2 to 2x its own
    after settling for steps / checks steps, and at most `max_area` cells above 0.1 — before
    it first fails a check. A state that is empty after settling scores 0."""
    every = steps // checks

    def run(state):
        state = engine.advance(system, state, every)
        mass0 = system.observe(state).sum()

        def body(carry, _):
            s, alive = carry
            s = engine.advance(system, s, every)
            f = system.observe(s)
            ok = (f.sum() >= mass0 / 2) & (f.sum() <= 2 * mass0) & ((f > 0.1).sum() <= max_area)
            alive = alive & ok
            return (s, alive), alive

        _, alive = lax.scan(body, (state, mass0 > 0), length=checks)
        return every * alive.sum()

    return jax.vmap(run)(states)


@register_measure
class FunctionalInformation(RuleMeasure):
    key = "functional_information"
    icon = "📚"
    equation = r"I(E_x) = -\log_2 F(E_x) = -\log_2 \frac{M(E_x)}{N}"
    reference = "Hazen et al. 2007"
    name = "Functional information"
    unit = "bits"
    description = (
        "Hazen et al.'s functional information of the pattern: -log2 of the fraction of "
        "configurations whose degree of function is at least the pattern's own. "
        "Configurations: `samples` random fillings of the pattern's bounding box (iid "
        "uniform states; for a soup, other soups) under the pattern's rule, plus the "
        "pattern itself. Degree of function: steps it lives on as a bounded object (mass "
        "within 1/2 to 2x its own after settling for steps / checks steps; area at most 2x "
        "the pattern's bounding box), checked `checks` times over `steps` steps on a `size`² torus. At most log2(1 + samples); high = few random patterns "
        "of that size do what the pattern does."
    )
    peaks_at = "structure"
    granularity = "system"  # Hazen: a fraction of an ensemble of configurations
    subject = "ensemble"  # random configurations of the pattern's size
    samples = 512
    steps = 500
    checks = 10
    size = 128
    params = ("samples", "steps", "checks", "size")
    applies_to = "Game of Life, Lenia and Integer Lenia (random patterns of the pattern's size)"
    temporal = "dynamic"

    def accepts(self, system) -> bool:
        return type(system).scramble is not System.scramble

    def compute_rule(self, system, pattern, ctx=NO_CONTEXT) -> float:
        size = (self.size, self.size)
        own = system.init(pattern, size, jax.random.key(0))
        rows, cols = (np.flatnonzero(np.asarray(system.observe(own)).any(axis)) for axis in (1, 0))
        box = (np.ptp(rows) + 1) * (np.ptp(cols) + 1) if len(rows) else 1
        run = partial(survival, system, steps=self.steps, checks=self.checks, max_area=2 * box)
        degree = [np.asarray(run(jax.tree.map(lambda x: x[None], own)))]
        rngs = jax.random.split(jax.random.key(1), self.samples)
        for i in range(0, self.samples, 64):  # in batches, to bound device memory
            batch = jax.vmap(lambda k: system.scramble(pattern, size, k))(rngs[i : i + 64])
            degree.append(np.asarray(run(batch)))
        degree = np.concatenate(degree)
        return float(np.log2(1 / np.mean(degree >= degree[0])))
