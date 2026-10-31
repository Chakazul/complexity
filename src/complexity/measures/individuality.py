"""Information theory of individuality (ITI; Krakauer et al. 2020), in two forms
(MEASURES.md §5.4).

Krakauer's measures split what determines a system S's next state between S itself and
its environment E:
  organismal individuality  A* = I(S_{n+1}; S_n)
  colonial individuality    A  = I(S_{n+1}; S_n | E_n)
  environmental determination nC = I(S_{n+1}; E_n | S_n)

`iti_features` follows the paper: S and E are coarse-grained features (mass) of a segmented
creature and of everything else, as time series over the window. `iti_local` is a local
adaptation: per cell of a creature, S = the cell and its neighbours in the same creature,
E = its other neighbours (background or other creatures); it estimates well and gives a map.
Creatures are connected components above a threshold, on the torus.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from complexity.measures.base import NO_CONTEXT, Measure
from complexity.registry import register_measure

_EIGHT = np.ones((3, 3), bool)
_NEIGHBOURS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
COMPONENTS = ("organismal", "colonial", "driven")


def torus_label(mask: np.ndarray) -> np.ndarray:
    """8-connected components of `mask` on a torus: ndimage.label, then labels that touch
    across opposite edges merged. 0 = background, labels 1..n (not consecutive)."""
    labels, n = ndimage.label(mask, _EIGHT)
    if n == 0:
        return labels
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a_edge, b_edge in ((labels[0], labels[-1]), (labels[:, 0], labels[:, -1])):
        for shift in (-1, 0, 1):  # diagonal neighbours across the edge too
            b = np.roll(b_edge, shift)
            for x, y in zip(a_edge[(a_edge > 0) & (b > 0)], b[(a_edge > 0) & (b > 0)]):
                parent[find(x)] = find(y)
    roots = np.array([find(i) for i in range(n + 1)])
    return roots[labels]


def _entropy(counts: np.ndarray) -> float:
    c = counts[counts > 0].astype(float)
    n = c.sum()
    return float(np.log2(n) - np.sum(c * np.log2(c)) / n) if n else 0.0


def decompose(joint: np.ndarray) -> dict[str, float]:
    """A*, A and nC from counts over (S', S, E), plug-in, in bits."""
    def h(axes):
        return _entropy((joint.sum(axis=axes) if axes else joint).ravel())

    h_s1s, h_s, h_s1se, h_se = h((2,)), h((0, 2)), h(()), h((0,))
    h_s1, h_s1e, h_e = h((1, 2)), h((1,)), h((0, 1))
    a_star = h_s1 + h_s - h_s1s  # I(S'; S)
    a = (h_s1e - h_e) - (h_s1se - h_se)  # I(S'; S | E) = H(S'|E) - H(S'|S,E)
    nc = (h_s1s - h_s) - (h_s1se - h_se)  # I(S'; E | S) = H(S'|S) - H(S'|S,E)
    return {"organismal": max(a_star, 0.0), "colonial": max(a, 0.0), "driven": max(nc, 0.0)}


def local_samples(symbols: np.ndarray, n_levels: int, radius: int):
    """Per member cell of a creature at each step: (S', S, E) codes and the sample mask.
    S = (own state, summed state of its 8 neighbours in the same creature: the live
    neighbour count for Life). E = summed state of other creatures' cells within `radius`
    (a disk), in units of the top level, capped at `n_levels`: 8-connected creatures
    never touch, so other creatures are at least 2 cells away."""
    L = n_levels
    cur, nxt = symbols[:-1], symbols[1:]
    labels = np.stack([torus_label(f > 0) for f in cur])
    same_sum = np.zeros(cur.shape, int)
    for dy, dx in _NEIGHBOURS:
        same = np.roll(labels, (dy, dx), (1, 2)) == labels
        same_sum += np.where(same, np.roll(cur, (dy, dx), (1, 2)), 0)
    other_sum = np.zeros(cur.shape, int)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx <= radius * radius:
                other = np.roll(labels, (dy, dx), (1, 2)) != labels
                other_sum += np.where(other, np.roll(cur, (dy, dx), (1, 2)), 0)
    e = np.minimum(-(-other_sum // (L - 1)), L)  # ceil: any contact counts
    return nxt, cur * (8 * (L - 1) + 1) + same_sum, e, labels > 0


def creature_features(frame: np.ndarray, threshold: float) -> tuple[float, float]:
    """(mass of the largest creature, mass of everything else) in a frame."""
    labels = torus_label(frame > threshold)
    if labels.max() == 0:
        return 0.0, float(frame.sum())
    masses = ndimage.sum(frame, labels, np.arange(labels.max() + 1))
    masses[0] = 0
    target = float(masses.max())
    return target, float(frame.sum()) - target


def quantize(x: np.ndarray, q: int) -> np.ndarray:
    """Levels 0..q-1 by the series' own quantiles; a constant series is one level."""
    edges = np.unique(np.quantile(x, np.linspace(0, 1, q + 1)[1:-1]))
    return np.digitize(x, edges)


@register_measure
class FeatureIndividuality(Measure):
    key = "iti_features"
    name = "Individuality (features)"
    unit = "bits"
    group = "Individuality"
    description = (
        "Krakauer et al.'s information theory of individuality on coarse-grained features, "
        "as in the paper: S = the mass of the largest creature (connected cells above "
        "`threshold`, on the torus), E = the mass of everything else, each cut into "
        "`levels_q` quantile levels over the window's time series; `component`: organismal "
        "A* = I(S'; S) (default), colonial A = I(S'; S | E), driven nC = I(S'; E | S), "
        "less their mean over `shuffles` time-shuffled S' (removes the bias of short "
        "series). For a lone creature E is empty and A* is the mass's "
        "predictability; meant for scenes of several creatures (e.g. the Aquarium)."
    )
    window = 128
    peaks_at = "structure"
    component = "organismal"
    threshold = 0.1
    levels_q = 4
    shuffles = 20
    params = ("component", "window", "threshold", "levels_q", "shuffles")
    choices = {"component": COMPONENTS}
    scope = "local"
    temporal = "dynamic"
    subject = "trajectory"
    estimated_from = "run"

    def __init__(self, **params):
        super().__init__(**params)
        self._cache: dict[int, tuple] = {}  # id(frame) -> (frame, features)

    def _features(self, frame) -> tuple[float, float]:
        hit = self._cache.get(id(frame))
        if hit is None or hit[0] is not frame:
            hit = (frame, creature_features(frame, self.threshold))
            self._cache[id(frame)] = hit
        return hit[1]

    def compute(self, frames, ctx=NO_CONTEXT) -> float:
        feats = np.array([self._features(f) for f in frames])
        live = {id(f) for f in frames}
        self._cache = {k: v for k, v in self._cache.items() if k in live}
        if len(feats) < 3:
            return 0.0
        s, e = quantize(feats[:, 0], self.levels_q), quantize(feats[:, 1], self.levels_q)
        q, rng = self.levels_q, np.random.default_rng(0)

        def value(nxt):
            joint = np.zeros((q, q, q))
            np.add.at(joint, (nxt, s[:-1], e[:-1]), 1)
            return decompose(joint)[self.component]

        surrogate = np.mean([value(rng.permutation(s[1:])) for _ in range(self.shuffles)])
        return max(value(s[1:]) - surrogate, 0.0)


@register_measure
class LocalIndividuality(Measure):
    key = "iti_local"
    name = "Individuality (local)"
    unit = "bits/cell"
    group = "Individuality"
    description = (
        "A local adaptation of Krakauer et al.'s information theory of individuality: for "
        "every cell of a creature (connected cells above the lowest level), how much its "
        "next state is determined by itself and its neighbours in the same creature (S) "
        "versus its other neighbours, background or other creatures (E). `component`: "
        "organismal A* = I(S'; S) (default), colonial A = I(S'; S | E), driven "
        "nC = I(S'; E | S). S is the cell's state and the summed state of its neighbours in "
        "the creature, E the state of other creatures' cells within `radius`. Per creature "
        "cell, over the window; at most 4 levels. 0 for empty frames, ~0 for noise. In Life "
        "nC is 0: a cell's next state depends on its 3x3 only, all in its own creature."
    )
    window = 16
    needs_discrete = True
    peaks_at = "structure"
    component = "organismal"
    max_levels = 4
    radius = 3
    params = ("component", "window", "radius", "n_levels")
    choices = {"component": COMPONENTS}
    scope = "local"
    temporal = "dynamic"
    subject = "trajectory"
    estimated_from = "run"
    map = "local"

    def levels(self, n_states):
        return self.n_levels or min(n_states or self.max_levels, self.max_levels)

    def _joint(self, frames, L):
        s1, s, e, member = local_samples(np.stack(frames), L, self.radius)
        joint = np.zeros((L, L * (8 * (L - 1) + 1), L + 1))
        np.add.at(joint, (s1[member], s[member], e[member]), 1)
        return joint, (s1, s, e, member)

    def compute(self, frames, ctx=NO_CONTEXT) -> float:
        if len(frames) < 2:
            return 0.0
        joint, _ = self._joint(frames, ctx.n_levels)
        return decompose(joint)[self.component] if joint.sum() else 0.0

    def compute_map(self, frames, ctx=NO_CONTEXT) -> np.ndarray:
        """Pointwise values (log2 of the probability ratios inside the chosen component),
        averaged over the steps each cell is part of a creature; 0 elsewhere."""
        if len(frames) < 2:
            return np.zeros(np.shape(frames[-1]), np.float32)
        joint, (s1, s, e, member) = self._joint(frames, ctx.n_levels)
        if not joint.sum():
            return np.zeros(s1.shape[1:], np.float32)
        p = joint / joint.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            p_s1s, p_s, p_s1 = p.sum(2), p.sum((0, 2)), p.sum((1, 2))
            p_s1se, p_se, p_s1e, p_e = p, p.sum(0), p.sum(1), p.sum((0, 1))
            ratio = {
                "organismal": p_s1s[s1, s] / p_s[s] / p_s1[s1],
                "colonial": (p_s1se[s1, s, e] / p_se[s, e]) / (p_s1e[s1, e] / p_e[e]),
                "driven": (p_s1se[s1, s, e] / p_se[s, e]) / (p_s1s[s1, s] / p_s[s]),
            }[self.component]
            local = np.where(member, np.log2(np.where(member, ratio, 1)), 0)
        return (local.sum(0) / np.maximum(member.sum(0), 1)).astype(np.float32)
