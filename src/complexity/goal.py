"""A first prototype of the goal measure, the complexity field C = S × M × I (MEASURES.md
§8.1b, PLAN.md M3b): high for structures that maintain themselves as individuals, low for
empty space, noise, frozen noise and goo.

Per object (the basins of the potential K ∗ A, `individuality.detect`; the frame above a
threshold for systems without a potential), painted onto the object's cells:

- S, structure: how much better the object compresses than its own cells shuffled within
  its outline (matched noise: the same values, no arrangement), at scales 1 and 2.
- M, self-maintenance (interventional): twin runs with the object's living cells
  perturbed at several strengths (Gaussian noise of σ = 0.1-0.4; discrete: each cell reset
  to a random state with probability 0.1-0.3), run `steps` steps; repair and containment
  are averaged over the strengths that lowered the object's similarity by more than 0.02
  below the ceiling (the similarity a negligible hit, σ = 0.02, ends at: a continuous
  object never comes back in exactly the same phase; 1 for discrete systems), and repair
  is measured up to that ceiling, so M is a tolerance. The ceiling also counts as
  stability: M = ceiling × repair × containment (goo is chaotic, its ceiling low): a unit that withstands only the weaker hits scores less (Orbium dies at
  σ = 0.4, the Aquarium unit does not). Repair: the share of the similarity to
  the unperturbed run that the perturbation took and that comes back (similarity at best
  alignment, so a small shift is no damage). Containment: the share of the difference
  between the runs that stays near the object. M = repair × containment. Erasing part of
  the object was tried first: a hole of radius 2 at Orbium's centre (15 % of its mass)
  kills it, so that probes only the strong end of the tolerance curve.
- I, individuality: bounded, isolated and not spreading: the share of a ring around the
  object that is background (not other objects: goo's basins tile the space), over
  1 + area / (5 % of the grid), times 1.5 / growth when its area grows more than 1.5×
  in the unperturbed run during the M test (goo spreads).
"""

from __future__ import annotations

import lzma
from dataclasses import asdict, dataclass

import jax
import numpy as np
from scipy import ndimage

from complexity import engine
from complexity.collisions import state_cells
from complexity.measures.base import to_uint8
from complexity.measures.individuality import FeatureIndividuality, detect, torus_centroid
from complexity.systems.base import System

THRESHOLD, SADDLE = FeatureIndividuality.threshold, FeatureIndividuality.saddle


@dataclass
class ObjectScore:
    label: int
    area: int
    mass: float
    y: float
    x: float
    S: float
    ceiling: float  # stability: the similarity after a negligible hit (1 if discrete)
    repair: float
    containment: float
    M: float
    I: float
    C: float

    def record(self) -> dict:
        return asdict(self)


def compressed_bits(a: np.ndarray) -> int:
    return 8 * len(lzma.compress(to_uint8(a).tobytes(), preset=6))


def structure(frame: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> float:
    """1 − bits(object) / bits(object shuffled within its outline), mean over scales 1 and
    2 (2×2 means), clipped to [0, 1]; 0 for objects of under 9 cells."""
    if mask.sum() < 9:
        return 0.0
    ys, xs = np.nonzero(mask)
    crop = np.where(mask, frame, 0)[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    inside = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    values = []
    for scale in (1, 2):
        c, m = crop, inside
        if scale > 1:
            h, w = (d - d % scale for d in c.shape)
            if min(h, w) < 2:
                continue
            c = c[:h, :w].reshape(h // scale, scale, w // scale, scale).mean((1, 3))
            m = m[:h, :w].reshape(h // scale, scale, w // scale, scale).any((1, 3))
        shuffled = c.copy()
        shuffled[m] = rng.permutation(c[m])
        values.append(1 - compressed_bits(c) / max(compressed_bits(shuffled), 1))
    return float(np.clip(np.mean(values), 0, 1)) if values else 0.0


def aligned_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """The best normalized cross-correlation of two fields over all shifts (on the torus)."""
    a, b = a - a.mean(), b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0 if na == nb else 0.0
    corr = np.fft.irfft2(np.fft.rfft2(a) * np.conj(np.fft.rfft2(b)), s=a.shape)
    return float(corr.max() / (na * nb))


def window(mask: np.ndarray, margin: int) -> np.ndarray:
    return ndimage.binary_dilation(mask, iterations=margin) if margin else mask


def perturb(cells: np.ndarray, where: np.ndarray, strength: float, n_states: int | None,
            rng: np.random.Generator) -> np.ndarray:
    """`cells` with the cells in `where` perturbed: Gaussian noise of σ = `strength`,
    clipped to [0, 1], or, for `n_states` discrete states, each reset to a random state
    with probability `strength`. Channels (multi-channel Lenia) are perturbed alike."""
    out = cells.copy()
    mask = where if cells.ndim == 2 else np.broadcast_to(where[..., None], cells.shape)
    if n_states:
        flip = mask & (rng.random(cells.shape) < strength)
        out[flip] = rng.integers(0, n_states, flip.sum()).astype(cells.dtype)
        if out.dtype.kind == "f":  # frames-as-cells (baselines): levels k / (n - 1)
            out[flip] /= max(n_states - 1, 1)
    else:
        noise = rng.normal(0, strength, cells.shape)
        out = np.where(mask, np.clip(cells + noise, 0, 1), cells).astype(cells.dtype)
    return out


def nearest_object(frame, labels, centre, max_distance) -> np.ndarray | None:
    """The mask of the object whose centre is nearest `centre` (on the torus), if within
    `max_distance`."""
    ids = np.unique(labels[labels > 0])
    if not len(ids):
        return None
    n = np.array(frame.shape, float)
    d = np.array([torus_centroid(labels == i) for i in ids]) - centre
    d = np.abs((d + n / 2) % n - n / 2)
    dist = np.hypot(d[:, 0], d[:, 1])
    return labels == ids[np.argmin(dist)] if dist.min() <= max_distance else None


def object_similarity(frame_a, mask_a, frame_b, mask_b) -> float:
    """The aligned similarity of two objects, each cut out of its frame (zero elsewhere)."""
    if mask_a is None or mask_b is None:
        return 0.0
    return aligned_similarity(np.where(mask_a, frame_a, 0), np.where(mask_b, frame_b, 0))


def complexity_field(system: System, pattern: str, state, steps: int = 60, seed: int = 0,
                     max_objects: int = 64, strengths=None
                     ) -> tuple[np.ndarray, list[ObjectScore]]:
    """C(x, y) at this state, and the scores per object (at most `max_objects`, the most
    massive). See the module docstring."""
    reach = system.reach(pattern)
    n_states = system.n_states(pattern)
    if strengths is None:  # σ of the noise, or the probability of resetting a cell
        strengths = (0.1, 0.2, 0.3) if n_states else (0.1, 0.2, 0.3, 0.4)
    has_potential = type(system).potential is not System.potential
    potential_fn = jax.jit(system.potential) if has_potential else None
    min_area = np.pi * (reach / 4) ** 2

    def look(st):
        frame = np.asarray(system.observe(st), np.float64)
        potential = np.asarray(potential_fn(st), np.float64) if potential_fn else None
        return frame, detect(frame, potential, THRESHOLD, SADDLE, min_area)

    frame, labels = look(state)
    ids = np.unique(labels[labels > 0])
    masses = np.bincount(labels.ravel(), weights=frame.ravel())
    ids = ids[np.argsort(-masses[ids])][:max_objects]
    rng = np.random.default_rng(seed)
    ref_frame, ref_labels = look(engine.advance(system, state, steps))
    cells = state_cells(state)
    travel = reach + steps  # how far an object may move during the test (≤ 1 cell/step)
    grid_area = frame.size
    background = labels == 0  # outside every object (the gaps of goo's stripes are not)
    field = np.zeros(frame.shape, np.float32)
    scores = []
    for i in ids:
        mask = labels == i
        area = int(mask.sum())
        centre = torus_centroid(mask)
        S = structure(frame, mask, rng)
        # the intervention, at each strength: the object's living cells perturbed (at
        # least one cell, so that no change is never read as a full repair)
        living = mask & (frame > (0 if n_states else 0.01))
        ref_obj = nearest_object(ref_frame, ref_labels, centre, travel)

        def twin_after(strength, living=living):
            perturbed = perturb(cells, living, strength, n_states, rng)
            if np.array_equal(perturbed, cells) and living.any():
                ys, xs = np.nonzero(living)
                k = rng.integers(len(ys))
                perturbed[ys[k], xs[k]] = 0
            twin = system.with_cells(pattern, perturbed)
            frame0 = np.asarray(system.observe(twin), np.float64)
            return frame0, *look(engine.advance(system, twin, steps))

        # the ceiling: a continuous object never comes back in exactly the same phase, so
        # the similarity a negligible perturbation ends at is the best repair can reach
        ceiling = 1.0
        if not n_states:
            _, tf, tl = twin_after(0.02)
            ceiling = object_similarity(tf, nearest_object(tf, tl, centre, travel), ref_frame, ref_obj)
        repairs, containments = [], []
        for strength in strengths:
            twin_frame0, twin_frame, twin_labels = twin_after(strength)
            rho0 = object_similarity(twin_frame0, mask, frame, mask)
            if rho0 > ceiling - 0.02:  # too slight to read a repair from
                continue
            # repair: the object against itself, found again in both runs, up to the ceiling
            twin_obj = nearest_object(twin_frame, twin_labels, centre, travel)
            rho = object_similarity(twin_frame, twin_obj, ref_frame, ref_obj)
            repairs.append(float(np.clip((rho - rho0) / (ceiling - rho0), 0, 1)))
            # containment: the difference between the runs near the object's path
            near = window(mask, travel)
            diff = np.abs(twin_frame - ref_frame)
            containments.append(float(diff[near].sum() / diff.sum()) if diff.sum() > 0 else 1.0)
        repair = float(np.mean(repairs)) if repairs else 0.0
        containment = float(np.mean(containments)) if containments else 0.0
        M = max(ceiling, 0.0) * repair * containment  # stability × repair × containment
        # individuality: bounded, isolated (its surroundings are background, not other
        # objects: goo's basins tile the space) and not spreading
        ring = window(mask, max(2, reach // 2)) & ~mask
        isolation = float(background[ring].mean()) if ring.any() else 0.0
        grown = (ref_obj.sum() if ref_obj is not None else 0) / max(area, 1)
        I = isolation / (1 + area / (0.05 * grid_area)) * min(1.0, 1.5 / max(grown, 1e-9))
        score = ObjectScore(int(i), area, float(masses[i]), float(centre[0]), float(centre[1]),
                            S, float(ceiling), repair, containment, M, float(I), float(S * M * I))
        scores.append(score)
        field[mask] = score.C
    return field, scores
