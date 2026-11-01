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
  are averaged over the strengths that changed the object (similarity below 0.995), and
  repair is measured up to the ceiling (the similarity a negligible hit, σ = 0.02, ends
  at: a continuous object never comes back in exactly the same phase, and a chaotic colony
  drifts; 1 for discrete systems); a hit that dents the object less than that drift is
  repaired as far as it gets back to the ceiling, so M is a tolerance. The ceiling also counts as
  stability; and contact, a patch of full-strength noise of the object's size placed
  beside it (the "unit into noise" collision), judged after 3 × `steps` (the patch takes
  time to clear): 0 if the run's mass then exceeds twice the unperturbed run's plus the
  noise added (goo) or the object is gone, else how far it is still itself.
  M = ceiling × repair × containment × contact (Triscutium withstands noise inside but
  turns to goo on any contact; a hit that makes an object grow counts as unrepaired): a unit that withstands only the weaker hits scores less (Orbium dies at
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
from functools import partial

import jax
import numpy as np
from jax import lax
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
    contact: float  # still itself after a patch of noise beside it (0 if it set off growth)
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
    """`mask` grown by `margin` cells, wrapping around the torus (a plain dilation would
    lose an object that crosses the grid's edge)."""
    if not margin:
        return mask
    m = min(margin, max(mask.shape))
    grown = ndimage.binary_dilation(np.pad(mask, m, mode="wrap"), iterations=margin)
    return grown[m:-m, m:-m]


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


BATCH = 16  # twin runs per vmap call (padded to it: one compile per run length)


@partial(jax.jit, static_argnums=(0, 3, 4))
def _twins(system: System, state, cells, steps: int, with_potential: bool):
    """A batch of twin runs that share `state`'s rule and differ in their cells: for each,
    the first frame, the frame after `steps` steps and (if asked) its potential."""
    def one(A):
        start = system.replace_cells(state, A)
        end = lax.fori_loop(0, steps, lambda _, s: system.step(s), start)
        out = (system.observe(start), system.observe(end))
        return out + (system.potential(end),) if with_potential else out

    return jax.vmap(one)(cells)


def run_twins(system: System, state, cells: list, steps: int, with_potential: bool) -> list:
    """`_twins` over any number of cell arrays, BATCH at a time; per twin (frame0, frame,
    potential or None) as float64 NumPy arrays."""
    out = []
    for i in range(0, len(cells), BATCH):
        chunk = cells[i : i + BATCH]
        padded = np.stack(chunk + [chunk[-1]] * (BATCH - len(chunk)))
        result = [np.asarray(r, np.float64) for r in
                  _twins(system, state, padded, steps, with_potential)]
        for j in range(len(chunk)):
            out.append((result[0][j], result[1][j], result[2][j] if with_potential else None))
    return out


def complexity_field(system: System, pattern: str, state, steps: int = 60, seed: int = 0,
                     max_objects: int = 64, strengths=None, trials: int = 1
                     ) -> tuple[np.ndarray, list[ObjectScore]]:
    """C(x, y) at this state, and the scores per object (at most `max_objects`, the most
    massive), each hit drawn `trials` times (more is steadier: where the noise lands
    matters). See the module docstring. All twin runs (every hit of every object) are
    prepared first and run in batches under `vmap`. The contact test runs 3 × `steps`: on
    a grid smaller than about twice the distance an object travels in that time, it may
    meet the noise's remains or itself (Orbium at 128²)."""
    reach = system.reach(pattern)
    n_states = system.n_states(pattern)
    if strengths is None:  # σ of the noise, or the probability of resetting a cell
        strengths = (0.1, 0.2, 0.3) if n_states else (0.1, 0.2, 0.3, 0.4)
    has_potential = type(system).potential is not System.potential
    potential_fn = jax.jit(system.potential) if has_potential else None
    min_area = np.pi * (reach / 4) ** 2

    def labelled(frame, potential):
        return detect(frame, potential, THRESHOLD, SADDLE, min_area)

    def look(st):
        frame = np.asarray(system.observe(st), np.float64)
        potential = np.asarray(potential_fn(st), np.float64) if potential_fn else None
        return frame, labelled(frame, potential)

    frame, labels = look(state)
    ids = np.unique(labels[labels > 0])
    masses = np.bincount(labels.ravel(), weights=frame.ravel())
    ids = ids[np.argsort(-masses[ids])][:max_objects]
    rng = np.random.default_rng(seed)
    ref_frame, ref_labels = look(engine.advance(system, state, steps))
    long_frame, long_labels = look(engine.advance(system, state, 3 * steps))  # for contact
    cells = state_cells(state)
    travel = reach + steps  # how far an object may move during the test (≤ 1 cell/step)
    background = labels == 0  # outside every object (the gaps of goo's stripes are not)

    # 1. every object's hits, as perturbed cells: (object, kind, cells); kinds: "ceiling" (a
    # negligible hit, continuous systems), "inside" (each strength × trials), "contact"
    # (a patch of noise of its size beside it, × trials)
    objects_, short, long = [], [], []
    for i in ids:
        mask = labels == i
        area = int(mask.sum())
        centre = torus_centroid(mask)
        living = mask & (frame > (0 if n_states else 0.01))
        objects_.append((i, mask, area, centre))

        def hit(where, strength, living=living):
            # at least one cell changes, so that no change is never read as a full repair
            perturbed = perturb(cells, where, strength, n_states, rng)
            if np.array_equal(perturbed, cells) and where.any():
                ys, xs = np.nonzero(where)
                k = rng.integers(len(ys))
                perturbed[ys[k], xs[k]] = 0 if where is living else (1 if n_states else 0.5)
            return perturbed

        k = len(objects_) - 1
        if not n_states:
            short.append((k, "ceiling", hit(living, 0.02)))
        for strength in [x for x in strengths for _ in range(trials)]:
            short.append((k, "inside", hit(living, strength)))
        side = max(3, int(np.sqrt(area)))
        for _ in range(trials):
            angle = rng.uniform(0, 2 * np.pi)
            out = np.sqrt(area / np.pi) + side / 2 + 2
            py, px = (centre + out * np.array([np.sin(angle), np.cos(angle)])).round().astype(int)
            yy = (np.arange(side) + py - side // 2) % frame.shape[0]
            xx = (np.arange(side) + px - side // 2) % frame.shape[1]
            patch = np.zeros(frame.shape, bool)
            patch[np.ix_(yy, xx)] = True
            patch &= background
            if patch.any():
                long.append((k, "contact", hit(patch, 1.0)))

    # 2. all twins, batched on the device; objects found again on the host
    results = {}
    for hits, n in ((short, steps), (long, 3 * steps)):
        if hits:
            runs = run_twins(system, state, [c for *_, c in hits], n, has_potential)
            for (k, kind, _), (f0, f, u) in zip(hits, runs):
                results.setdefault(k, []).append((kind, f0, f, labelled(f, u)))

    # 3. the scores
    grid_area = frame.size
    field = np.zeros(frame.shape, np.float32)
    scores = []
    for k, (i, mask, area, centre) in enumerate(objects_):
        S = structure(frame, mask, rng)
        ref_obj = nearest_object(ref_frame, ref_labels, centre, travel)
        long_obj = nearest_object(long_frame, long_labels, centre, 3 * travel)
        ref_area = ref_obj.sum() if ref_obj is not None else area
        near = window(mask, travel)
        # the ceiling: a continuous object never comes back in exactly the same phase, so
        # the similarity a negligible perturbation ends at is the best repair can reach
        ceiling = 1.0
        for kind, _, f, lab in results.get(k, []):
            if kind == "ceiling":
                ceiling = object_similarity(f, nearest_object(f, lab, centre, travel), ref_frame, ref_obj)
        repairs, containments, contacts = [], [], []
        for kind, f0, f, lab in results.get(k, []):
            if kind == "inside":
                twin_obj = nearest_object(f, lab, centre, travel)
                rho = object_similarity(f, twin_obj, ref_frame, ref_obj)
                if twin_obj is not None and twin_obj.sum() > 1.5 * ref_area:
                    repairs.append(0.0)  # it grew: the hit set off growth
                else:
                    rho0 = object_similarity(f0, mask, frame, mask)
                    if rho0 > 0.995:  # the hit did nothing to read a repair from
                        continue
                    # repair: the object against itself, found again in both runs, up to
                    # the ceiling; a hit that dents it less than chaos does (rho0 above
                    # the ceiling, in a chaotic colony) is repaired as far as it gets back
                    if rho0 < ceiling - 0.02:
                        repairs.append(float(np.clip((rho - rho0) / (ceiling - rho0), 0, 1)))
                    else:
                        repairs.append(float(np.clip(rho / max(ceiling, 1e-6), 0, 1)))
                # containment: the difference between the runs near the object's path
                diff = np.abs(f - ref_frame)
                containments.append(float(diff[near].sum() / diff.sum()) if diff.sum() > 0 else 1.0)
            elif kind == "contact":
                # judged after 3 × `steps` (the patch takes time to clear): is it still
                # itself, and did nothing grow?
                added = f0.sum() - frame.sum()
                twin_obj = nearest_object(f, lab, centre, 3 * travel)
                grew = f.sum() > 2 * (long_frame.sum() + added)  # goo
                contacts.append(0.0 if twin_obj is None or grew else float(np.clip(
                    object_similarity(f, twin_obj, long_frame, long_obj)
                    / max(ceiling, 1e-6), 0, 1)))
        contact = float(np.mean(contacts)) if contacts else 1.0
        repair = float(np.mean(repairs)) if repairs else 0.0
        containment = float(np.mean(containments)) if containments else 0.0
        M = max(ceiling, 0.0) * repair * containment * contact
        # individuality: bounded, isolated (its surroundings are background, not other
        # objects: goo's basins tile the space) and not spreading
        ring = window(mask, max(2, reach // 2)) & ~mask
        isolation = float(background[ring].mean()) if ring.any() else 0.0
        grown = (ref_obj.sum() if ref_obj is not None else 0) / max(area, 1)
        I = isolation / (1 + area / (0.05 * grid_area)) * min(1.0, 1.5 / max(grown, 1e-9))
        score = ObjectScore(int(i), area, float(masses[i]), float(centre[0]), float(centre[1]),
                            S, float(ceiling), repair, containment, contact, M, float(I),
                            float(S * M * I))
        scores.append(score)
        field[mask] = score.C
    return field, scores
