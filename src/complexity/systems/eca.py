"""Elementary cellular automata (Wolfram 1983): binary cells on a ring, each next state a
function of the cell and its two neighbours, one of 256 rules (PLAN.md §4.4).

The dynamical state is one row; the frame is its space-time diagram, the last `height` rows
with the newest at the bottom, scrolling up one row per step. Frame-based measures therefore
see space-time diagrams (as compression and entropy of ECA usually are); measures that compare
dynamical states (transient length, Langton's λ) read the current row (`configuration`).

Pattern keys name the rule and the initial row: "rule110" (a random row, 50 %) or
"rule110_seed" (one live cell). Every rule 0..255 is accepted; the curated ones are listed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from complexity.registry import register_system
from complexity.systems.base import System, random_active_cell

# rule -> behaviour from a random row, in Wolfram's classes
CURATED = {
    110: "class 4", 54: "class 4", 30: "class 3", 18: "class 3", 90: "class 3, additive; dies on 2^k cells",
    184: "class 2, traffic", 108: "class 2", 232: "class 2, majority", 160: "class 1",
}
SEEDED = (110, 30, 90)  # also listed from one live cell


def parse(pattern: str) -> tuple[int, bool]:
    """(rule, seeded) of a pattern key, e.g. "rule30_seed" -> (30, True)."""
    match = re.fullmatch(r"rule(\d+)(_seed)?", pattern)
    if not match or int(match[1]) > 255:
        raise KeyError(f"not an elementary CA pattern: {pattern!r}")
    return int(match[1]), bool(match[2])


def rule_table(rule: int) -> np.ndarray:
    """The rule's outputs, indexed by neighbourhood 4 left + 2 centre + right (Wolfram's
    numbering: bit v of the rule number is the output for neighbourhood v)."""
    return np.array([(rule >> v) & 1 for v in range(8)], np.uint8)


def next_row(row, table):
    """One step of a ring of cells (NumPy or JAX arrays, rows along the last axis)."""
    xp = jnp if isinstance(row, jax.Array) else np
    v = 4 * xp.roll(row, 1, -1) + 2 * row + xp.roll(row, -1, -1)
    return table[v.astype(xp.int32)]


@partial(jax.tree_util.register_dataclass, data_fields=["A", "table"], meta_fields=[])
@dataclass(frozen=True)
class ECAState:
    A: jax.Array  # uint8 (height, width): the space-time diagram, current row last
    table: jax.Array  # uint8 (8,): the rule's outputs; data, so rules share one compile


@register_system
class ElementaryCA(System):
    key = "eca"
    name = "Elementary CA"
    patterns = {
        **{f"rule{r}": f"Rule {r} ({cls})" for r, cls in CURATED.items()},
        **{f"rule{r}_seed": f"Rule {r} from one cell" for r in SEEDED},
    }

    def n_states(self, pattern: str) -> int:
        return 2

    def rule_table(self, pattern: str) -> np.ndarray:
        return rule_table(parse(pattern)[0])

    def transition_map(self, pattern: str, n: int) -> np.ndarray:
        """On a ring of `n` cells; configuration x has cell i as bit i."""
        x = np.arange(2**n, dtype=np.int64)
        bits = (x[:, None] >> np.arange(n)) & 1
        out = next_row(bits.astype(np.uint8), rule_table(parse(pattern)[0]))
        return (out.astype(np.int64) << np.arange(n)).sum(axis=1)

    def _state(self, pattern: str, size, row) -> ECAState:
        A = jnp.zeros(size, jnp.uint8).at[-1].set(row)
        return ECAState(A, jnp.asarray(self.rule_table(pattern)))

    def init(self, pattern: str, size: tuple[int, int], rng: jax.Array) -> ECAState:
        _, seeded = parse(pattern)
        if seeded:
            row = jnp.zeros(size[1], jnp.uint8).at[size[1] // 2].set(1)
        else:
            row = jax.random.bernoulli(rng, 0.5, (size[1],)).astype(jnp.uint8)
        return self._state(pattern, size, row)

    def step(self, state: ECAState) -> ECAState:
        row = next_row(state.A[-1], state.table)
        return ECAState(jnp.roll(state.A, -1, 0).at[-1].set(row), state.table)

    def observe(self, state: ECAState) -> jax.Array:
        return state.A.astype(jnp.float32)

    def configuration(self, state: ECAState) -> jax.Array:
        return state.A[-1:].astype(jnp.float32)

    def soup(self, pattern: str, size: tuple[int, int], rng: jax.Array) -> ECAState:
        return self.init(f"rule{parse(pattern)[0]}", size, rng)

    def uniform(self, pattern: str, size: tuple[int, int], rng: jax.Array) -> ECAState:
        return self.soup(pattern, size, rng)

    def perturb(self, state: ECAState, rng: jax.Array) -> ECAState:
        """Flip one live cell of the current row (any cell if none is alive)."""
        _, x = random_active_cell(state.A[-1:] == 1, rng)
        return ECAState(state.A.at[-1, x].set(1 - state.A[-1, x]), state.table)
