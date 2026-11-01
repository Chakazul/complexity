"""Information decompositions of a cell's next state (MEASURES.md §5.3): partial information
decomposition (PID, Williams & Beer 2010) and functional information decomposition (FID,
Bohm et al. 2025).

Both use the same variables: target T = a cell's next state y_{t+1}, sources the cell's own
state Y = y_t and a von Neumann neighbour's X = x_t, on discretized frames, averaged over the
4 neighbours. Transfer entropy is part of this picture: I(T; X | Y) = unique(X) + synergy.
"""

from __future__ import annotations

import numpy as np

from complexity.measures.base import NO_CONTEXT, Measure
from complexity.registry import register_measure

# for the 4 neighbours: (shift of the neighbour's frame onto the cell)
_SHIFTS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def joint_counts(symbols: np.ndarray, n_levels: int, shift) -> np.ndarray:
    """Counts of (target y_{t+1}, own y_t, neighbour x_t), shape (L, L, L)."""
    nxt, cur = symbols[1:], symbols[:-1]
    nb = np.roll(cur, shift, (1, 2))
    idx = (nxt * n_levels + cur) * n_levels + nb
    return np.bincount(idx.ravel(), minlength=n_levels**3).reshape((n_levels,) * 3)


def _mi(p_ts: np.ndarray) -> float:
    """I(T; S) in bits from a joint table p(t, s) (s may be several axes, flattened)."""
    p = p_ts.reshape(p_ts.shape[0], -1)
    pt, ps = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float(np.sum(p[nz] * np.log2(p[nz] / (pt @ ps)[nz])))


def pid(counts: np.ndarray) -> dict[str, float]:
    """Williams-Beer PID of p(t, y, x) with redundancy I_min, in bits:
    R = Σ_t p(t) min_i I_spec(t; S_i), I_spec(t; S) = Σ_s p(s|t) log2(p(t|s) / p(t))."""
    p = counts / counts.sum()
    pt = p.sum((1, 2))
    specific = []
    for axis in (2, 1):  # marginalize the other source: sources Y, then X
        pts = p.sum(axis)  # p(t, s)
        ps = pts.sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(pts > 0, (pts / ps) / pt[:, None], 1)
            spec = np.sum(np.where(pts > 0, pts / pt[:, None] * np.log2(ratio), 0), 1)
        specific.append(np.where(pt > 0, spec, 0))
    redundancy = float(np.sum(pt * np.minimum(*specific)))
    i_y, i_x, i_yx = _mi(p.sum(2)), _mi(p.sum(1)), _mi(p)
    return {"redundancy": redundancy, "unique_self": i_y - redundancy,
            "unique_neighbour": i_x - redundancy,
            "synergy": i_yx - i_y - i_x + redundancy}


def fid(counts: np.ndarray) -> dict[str, float]:
    """Functional information decomposition of the map (y, x) -> t, in bits: the data are
    reweighted to uniform inputs over every combination of the values each source takes
    (an unseen combination gives a uniform output over the observed targets), then
    individual information I(T; S_i) and synergy I(T; Y, X) - Σ_i I(T; S_i). As
    `fid-tools`' `TPM.from_data(...).neutral_completion_fid()` (0.1.2, read, not run)."""
    counts = counts[:, counts.sum((0, 2)) > 0][:, :, counts.sum((0, 1)) > 0]
    seen_t = counts.sum((1, 2)) > 0
    rows = counts.sum(0)  # (y, x) -> count
    with np.errstate(invalid="ignore"):
        p_t_given = np.where(rows > 0, counts / rows, seen_t[:, None, None] / seen_t.sum())
    p = p_t_given / p_t_given[0].size  # uniform over (y, x)
    i_y, i_x, i_yx = _mi(p.sum(2)), _mi(p.sum(1)), _mi(p)
    return {"self": i_y, "neighbour": i_x, "synergy": i_yx - i_y - i_x}


class _Decomposition(Measure):
    """Shared by PID and FID: averaged over the 4 neighbours at distance `offset`."""

    unit = "bits/cell"
    group = "Information theory"
    window = 16
    needs_discrete = True
    peaks_at = "structure"
    offset = 1
    max_levels = 4
    granularity = "system"
    scope = "local"
    temporal = "dynamic"
    subject = "trajectory"
    estimated_from = "run"
    map = "tile"

    def levels(self, n_states):
        """At most `max_levels` symbols: the tables have L³ entries."""
        return self.n_levels or min(n_states or self.max_levels, self.max_levels)

    def compute(self, frames, ctx=NO_CONTEXT) -> float:
        symbols = np.stack(frames)
        if len(symbols) < 2:
            return 0.0
        L = ctx.n_levels
        values = [self.decompose(joint_counts(symbols, L, (dy * self.offset, dx * self.offset)))
                  for dy, dx in _SHIFTS]
        return float(np.mean([v[self.component] for v in values]))


@register_measure
class PartialInformation(_Decomposition):
    key = "pid"
    icon = "🤝"
    name = "Partial information (PID)"
    ratio = "change"
    ratio_unit = "bits/Δmass"
    description = (
        "Williams & Beer's partial information decomposition of what a cell's own state and "
        "a neighbour's state (at distance `offset`) tell about the cell's next state, with "
        "I_min redundancy; reports `component`: synergy (default: information only the two "
        "together give), redundancy, unique_self or unique_neighbour. Averaged over the 4 "
        "neighbours and the window; at most 4 levels. ~0 for empty frames and noise."
    )
    component = "synergy"
    params = ("component", "window", "offset", "n_levels")
    choices = {"component": ("synergy", "redundancy", "unique_self", "unique_neighbour")}

    def decompose(self, counts):
        return pid(counts)


@register_measure
class FunctionalDecomposition(_Decomposition):
    key = "fid"
    icon = "🤝"
    name = "Functional information (FID)"
    description = (
        "Bohm et al.'s functional information decomposition of the map from (a cell's own "
        "state, a neighbour's state) to the cell's next state, weighting every input pattern "
        "equally: synergy = I(next; both) - I(next; own) - I(next; neighbour), never negative "
        "(equally weighted inputs are independent, so there is no redundancy). `component`: "
        "synergy (default), self or neighbour. "
        "Averaged over the 4 neighbours and the window; at most 4 levels."
    )
    component = "synergy"
    params = ("component", "window", "offset", "n_levels")
    choices = {"component": ("synergy", "self", "neighbour")}
    granularity = "rule"  # a property of the input-output function (MEASURES.md §2.3)
    subject = "rule"

    def decompose(self, counts):
        return fid(counts)
