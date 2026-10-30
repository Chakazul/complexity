"""Statistical complexity (SC) from local causal states (docs/measures.md §5.3.1).

Shalizi et al. (2006) define causal states for fields: two past light cones are equivalent
when they predict the same distribution of future light cones. Here the past cone is a
cell's 3x3 neighbourhood at time t and the future cone the cell and its 4 von Neumann
neighbours at t+1, on discretized frames. Pasts are merged into causal states greedily
(most frequent first, as in CSSR) when a G-test cannot tell their future distributions
apart; the statistical complexity is the entropy of the causal state over all cells and
steps, in bits per cell.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from complexity.measures.base import NO_CONTEXT, Measure
from complexity.registry import register_measure

_PAST = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
_FUTURE = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]


def _code(frames: np.ndarray, offsets, n_levels: int) -> np.ndarray:
    code = np.zeros(frames.shape, np.int64)
    for dy, dx in offsets:
        code = code * n_levels + np.roll(frames, (-dy, -dx), (1, 2))
    return code


def _g_stat(a: np.ndarray, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """G statistics of homogeneity between one count vector and each row of `states`, and
    their degrees of freedom (future values seen in either, minus one)."""
    total = a + states
    n_a, n_s, n = a.sum(), states.sum(1, keepdims=True), total.sum(1, keepdims=True)
    g = np.zeros(len(states))
    for obs, n_obs in ((np.broadcast_to(a, states.shape), n_a), (states, n_s)):
        exp = total * n_obs / n
        with np.errstate(divide="ignore", invalid="ignore"):
            g += 2 * np.sum(np.where(obs > 0, obs * np.log(obs / exp), 0), 1)
    return g, (total > 0).sum(1) - 1


def causal_states(symbols: np.ndarray, n_levels: int, alpha: float):
    """(causal state of each distinct past, how often each past occurs, which distinct past
    each cell and step has). `symbols` is (time, height, width) on a torus."""
    past = _code(symbols[:-1], _PAST, n_levels).ravel()
    future = _code(symbols[1:], _FUTURE, n_levels).ravel()
    pasts, past_idx = np.unique(past, return_inverse=True)
    futures, future_idx = np.unique(future, return_inverse=True)
    table = np.zeros((len(pasts), len(futures)))
    np.add.at(table, (past_idx, future_idx), 1)
    counts = table.sum(1)
    state = np.full(len(pasts), -1)
    sums = np.zeros((0, len(futures)))  # future counts of each causal state
    for i in np.argsort(-counts, kind="stable"):
        g, dof = _g_stat(table[i], sums)
        same = np.nonzero((dof < 1) | (g <= chi2.ppf(1 - alpha, np.maximum(dof, 1))))[0]
        if len(same):  # the first (most frequent) state that cannot be told apart
            state[i] = same[0]
            sums[same[0]] += table[i]
        else:
            state[i] = len(sums)
            sums = np.vstack([sums, table[i]])
    return state, counts, past_idx


@register_measure
class StatisticalComplexity(Measure):
    key = "statistical_complexity"
    name = "Statistical complexity"
    unit = "bits/cell"
    group = "Information theory"
    description = (
        "Crutchfield's statistical complexity, local form (Shalizi et al. 2006): cells whose "
        "3x3 neighbourhoods predict the same distribution of their next 5-cell cross share a "
        "causal state (merged by a G-test at `alpha`); the measure is the entropy of the "
        "causal states over the window, the memory needed to predict the field. 0 for "
        "uniform frames and for noise (every past predicts the same), high for structure. "
        "At most 2 levels by default (the 3x3 pasts have L⁹ values)."
    )
    window = 16
    needs_discrete = True
    peaks_at = "structure"
    alpha = 1e-3
    max_levels = 2
    params = ("window", "alpha", "n_levels")
    scope = "local"
    temporal = "dynamic"
    subject = "trajectory"  # local causal states of this run's window
    estimated_from = "run"
    map = "local"

    def levels(self, n_states):
        return self.n_levels or min(n_states or self.max_levels, self.max_levels)

    def compute(self, frames, ctx=NO_CONTEXT) -> float:
        symbols = np.stack(frames)
        if len(symbols) < 2:
            return 0.0
        state, counts, _ = causal_states(symbols, ctx.n_levels, self.alpha)
        p = np.bincount(state, weights=counts) / counts.sum()
        p = p[p > 0]
        return float(max(-np.sum(p * np.log2(p)), 0.0))  # not -0.0

    def compute_map(self, frames, ctx=NO_CONTEXT) -> np.ndarray:
        """Local statistical complexity, -log2 P(causal state) per cell (Shalizi et al.
        2006), averaged over the window; its mean is the measure's value."""
        symbols = np.stack(frames)
        if len(symbols) < 2:
            return np.zeros(symbols.shape[1:], np.float32)
        state, counts, past_idx = causal_states(symbols, ctx.n_levels, self.alpha)
        p = np.bincount(state, weights=counts) / counts.sum()
        local = -np.log2(p[state[past_idx]]).reshape(symbols[:-1].shape)
        return np.maximum(local.mean(0), 0).astype(np.float32)
