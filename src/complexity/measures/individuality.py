"""Information theory of individuality (ITI; Krakauer et al. 2020), in two forms
(MEASURES.md §5.4).

Krakauer's measures split what determines a system S's next state between S itself and
its environment E:
  organismal individuality  A* = I(S_{n+1}; S_n)
  colonial individuality    A  = I(S_{n+1}; S_n | E_n)
  environmental determination nC = I(S_{n+1}; E_n | S_n)

`iti_features` follows the paper: S and E are coarse-grained features (mass) of a tracked
creature and of its environment, as time series over the window. Creatures are found in
the potential K ∗ A, how the model sees the world (`System.potential`): the catchment
basins of its peaks, neighbouring basins merged across shallow saddles; a target is followed
through the window by overlap (MEASURES.md §5.4). `iti_local` is a local
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


def basins(U: np.ndarray, threshold: float) -> np.ndarray:
    """Catchment basins of U's peaks on the torus: every cell points to its highest 3x3
    neighbour (itself if none is higher), and following the pointers ends at a peak, whose
    flat index + 1 labels the cell. Cells at or below `threshold` are 0."""
    h, w = U.shape
    idx = np.arange(h * w).reshape(h, w)
    best_u, best_i = U.copy(), idx.copy()
    for dy, dx in _NEIGHBOURS:
        u, i = np.roll(U, (dy, dx), (0, 1)), np.roll(idx, (dy, dx), (0, 1))
        higher = u > best_u
        best_u, best_i = np.where(higher, u, best_u), np.where(higher, i, best_i)
    ptr = best_i.ravel()
    while not np.array_equal(nxt := ptr[ptr], ptr):  # pointer jumping to the peaks
        ptr = nxt
    return np.where(U > threshold, ptr.reshape(h, w) + 1, 0)


def merge_shallow(U: np.ndarray, labels: np.ndarray, saddle: float) -> np.ndarray:
    """Basins merged with a neighbour when the saddle between them (the highest point of
    their common border) is at least `saddle` × the lower peak: bumps on one creature, not
    separate creatures. Highest saddles first; a merged basin keeps the higher peak."""
    ids = np.unique(labels[labels > 0])
    if len(ids) < 2:
        return labels
    peak = dict(zip(ids, U.ravel()[ids - 1]))
    a, b, height = [], [], []  # every border crossing, in 4 directions
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        other, u_other = np.roll(labels, (-dy, -dx), (0, 1)), np.roll(U, (-dy, -dx), (0, 1))
        m = (labels > 0) & (other > 0) & (labels != other)
        a.append(np.minimum(labels[m], other[m]))
        b.append(np.maximum(labels[m], other[m]))
        height.append(np.minimum(U[m], u_other[m]))
    a, b, height = np.concatenate(a), np.concatenate(b), np.concatenate(height)
    order = np.lexsort((height, b, a))  # per pair, its highest crossing last
    pairs = dict(zip(zip(a[order], b[order]), height[order]))
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for (x, y), v in sorted(pairs.items(), key=lambda kv: -kv[1]):
        rx, ry = find(x), find(y)
        if rx != ry and v >= saddle * min(peak[rx], peak[ry]):
            hi, lo = (rx, ry) if peak[rx] >= peak[ry] else (ry, rx)
            parent[lo] = hi
    lut = np.zeros(labels.max() + 1, labels.dtype)
    lut[ids] = [find(i) for i in ids]
    return lut[labels]


def detect(frame: np.ndarray, potential: np.ndarray | None, threshold: float, saddle: float,
           min_area: float) -> np.ndarray:
    """Creatures as labels (0 = background). With the potential: its basins above
    `threshold`, merged across shallow saddles, those smaller than `min_area` cells dropped.
    Without: 8-connected cells of the frame above `threshold`."""
    if potential is None:
        return torus_label(frame > threshold)
    labels = merge_shallow(potential, basins(potential, threshold), saddle)
    sizes = np.bincount(labels.ravel())
    return np.where(sizes[labels] >= min_area, labels, 0) * (labels > 0)


MIN_OVERLAP = 0.5  # of the target's cells, for its successor; else the target is lost


def disk_mask(radius: float) -> np.ndarray:
    r = int(np.ceil(radius))
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    return x * x + y * y <= radius * radius


def follow_target(frames, label_maps, target_kind: str = "typical"):
    """The target creature's mask in each frame (None once lost): chosen in the first frame
    (the one closest to the median mass, "typical", or the most massive, "largest"), then
    followed by overlap: the creature sharing most cells with the previous target, at
    least MIN_OVERLAP of them; when it divides, the larger part."""
    target = None
    for i, (frame, labels) in enumerate(zip(frames, label_maps)):
        if i == 0:
            ids = np.unique(labels[labels > 0])
            if len(ids):
                masses = np.bincount(labels.ravel(), weights=frame.ravel())
                if target_kind == "largest":
                    pick = ids[np.argmax(masses[ids])]
                else:
                    pick = ids[np.argmin(np.abs(masses[ids] - np.median(masses[ids])))]
                target = labels == pick
        elif target is not None:
            overlap = labels[target & (labels > 0)]
            best = np.bincount(overlap).argmax() if len(overlap) else 0
            keep = best and np.count_nonzero(overlap == best) >= MIN_OVERLAP * target.sum()
            target = labels == best if keep else None
        yield target


def track_features(frames, label_maps, environment: str, reach: float,
                   target_kind: str = "typical") -> np.ndarray:
    """(S, E) per frame: S = the mass of the target creature (`follow_target`), E = the
    mass of the other creatures within `reach` of it ("near") or of everything else
    ("all"). A lost target has S = 0."""
    near = disk_mask(reach) if environment == "near" else None
    out = []
    for frame, labels, target in zip(frames, label_maps,
                                     follow_target(frames, label_maps, target_kind)):
        total = float(frame.sum())
        if target is None or not target.any():
            out.append((0.0, total if environment == "all" else 0.0))
            continue
        s = float(frame[target].sum())
        if environment == "all":
            e = total - s
        else:
            r = near.shape[0] // 2
            zone = ndimage.binary_dilation(np.pad(target, r, mode="wrap"), near)[r:-r, r:-r]
            e = float(frame[zone & (labels > 0) & ~target].sum())
        out.append((s, e))
    return np.array(out)


def object_map(labels: np.ndarray, target: np.ndarray | None) -> np.ndarray:
    """Labels recoded for display: 0 background, 1 the target, 2, 3, ... the others."""
    is_other = (labels > 0) if target is None else (labels > 0) & ~target
    others = np.unique(labels[is_other])
    lut = np.zeros(labels.max() + 1, np.float32)
    lut[others] = np.arange(2, 2 + len(others))
    out = lut[labels]
    if target is not None:
        out[target] = 1
    return out


def quantize(x: np.ndarray, q: int) -> np.ndarray:
    """Levels 0..q-1 by the series' own quantiles; a constant series is one level."""
    edges = np.unique(np.quantile(x, np.linspace(0, 1, q + 1)[1:-1]))
    return np.digitize(x, edges)


@register_measure
class FeatureIndividuality(Measure):
    key = "iti_features"
    icon = "🍄"
    equation = r"A^* = I(S_{n+1}; S_n),\quad A = I(S_{n+1}; S_n \mid E_n),\quad nC = I(S_{n+1}; E_n \mid S_n)"
    reference = "Krakauer et al. 2020"
    name = "Individuality (features)"
    unit = "bits"
    group = "Individuality"
    description = (
        "Krakauer et al.'s information theory of individuality on coarse-grained features, "
        "as in the paper: S = the mass of a target creature, E = the mass of its "
        "environment, each cut into `levels_q` quantile levels over the window's time "
        "series. Creatures are found in the potential K ∗ A, how the model sees the world: "
        "the basins of its peaks above `threshold`, merged across saddles at least `saddle` "
        "× the lower peak, at least a disk of radius R/4 (frames above `threshold` for "
        "systems without a potential). `target`: typical (the creature closest to the median "
        "mass at the window's start, default) or largest, followed by overlap. `environment`: near (other creatures within "
        "the rule's reach R of the target, default) or all (everything else). `component`: "
        "organismal A* = I(S'; S) (default), colonial A = I(S'; S | E), driven "
        "nC = I(S'; E | S), less their mean over `shuffles` time-shuffled S' (removes the "
        "bias of short series). For a lone creature E is empty and A* is the mass's "
        "predictability; meant for scenes of several creatures (the Aquarium)."
    )
    window = 128
    peaks_at = "structure"
    needs_potential = True
    component = "organismal"
    environment = "near"
    target = "typical"
    threshold = 0.05
    saddle = 0.8
    levels_q = 4
    shuffles = 20
    params = ("component", "environment", "target", "window", "threshold", "saddle", "levels_q",
              "shuffles")
    choices = {"component": COMPONENTS, "environment": ("near", "all"), "target": ("typical", "largest")}
    scope = "local"
    temporal = "dynamic"
    subject = "trajectory"
    estimated_from = "run"

    def __init__(self, **params):
        super().__init__(**params)
        self._cache: dict[int, tuple] = {}  # id(frame) -> (frame, labels)

    def _labels(self, frame, potential, reach) -> np.ndarray:
        hit = self._cache.get(id(frame))
        if hit is None or hit[0] is not frame:
            min_area = np.pi * (reach / 4) ** 2
            hit = (frame, detect(frame, potential, self.threshold, self.saddle, min_area))
            self._cache[id(frame)] = hit
        return hit[1]

    def objects(self, frames, ctx=NO_CONTEXT, potentials=None) -> np.ndarray:
        """The latest frame's creatures for display (`object_map`), the target tracked over
        the window as `compute` tracks it."""
        reach = ctx.reach or 1
        pots = potentials if potentials is not None else [None] * len(frames)
        labels = [self._labels(f, u, reach) for f, u in zip(frames, pots)]
        *_, target = follow_target(frames, labels, self.target)
        return object_map(labels[-1], target)

    def compute(self, frames, ctx=NO_CONTEXT, potentials=None) -> float:
        reach = ctx.reach or 1
        pots = potentials if potentials is not None else [None] * len(frames)
        labels = [self._labels(f, u, reach) for f, u in zip(frames, pots)]
        live = {id(f) for f in frames}
        self._cache = {k: v for k, v in self._cache.items() if k in live}
        if len(frames) < 3:
            return 0.0
        feats = track_features(frames, labels, self.environment, reach, self.target)
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
    icon = "🍄"
    equation = r"A^* = I(S_{n+1}; S_n),\quad A = I(S_{n+1}; S_n \mid E_n),\quad nC = I(S_{n+1}; E_n \mid S_n)"
    reference = "Krakauer et al. 2020"
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
