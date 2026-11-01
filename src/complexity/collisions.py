"""Collision experiments (MEASURES.md §8.1b, PLAN.md M3b): units of a creature thrown at each
other or at a patch of noise, and the outcome labelled from the objects detected in the
potential K ∗ A (`individuality.detect`) and followed by `individuality.Tracker`.

A unit is first settled alone, then cropped with its heading and speed; scenes place
rotated copies on courses that meet at the grid's centre. The labels (survived, merged,
divided, died, goo, changed) are the ground truth that the goal measure C = S × M × I must
rank: units that survive collisions and stay individuals are the complex ones.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import jax
import numpy as np
from scipy import ndimage

from complexity import engine, registry
from complexity.measures.individuality import FeatureIndividuality, Tracker, detect, torus_centroid
from complexity.measures.statistics import centers

THRESHOLD, SADDLE = FeatureIndividuality.threshold, FeatureIndividuality.saddle
LABELS = ("survived", "merged", "divided", "died", "goo", "missed", "changed")


def state_cells(state) -> np.ndarray:
    """The cells a state holds: `A` for Lenia (with channels for multi-channel Lenia), the
    grid itself for Life."""
    return np.asarray(getattr(state, "A", state))


@dataclass
class Unit:
    """A settled unit: its cells (cropped, zero outside it), heading (degrees, of the
    displacement (dy, dx) in array coordinates, atan2(dy, dx)), speed (cells/step), mass
    and size (the crop's larger side)."""
    system: str
    pattern: str
    cells: np.ndarray
    heading: float
    speed: float
    mass: float
    size: int


def objects(system, state, potential_fn, reach) -> tuple[np.ndarray, np.ndarray]:
    frame = np.asarray(system.observe(state), np.float64)
    potential = np.asarray(potential_fn(state), np.float64)
    return frame, detect(frame, potential, THRESHOLD, SADDLE, np.pi * (reach / 4) ** 2)


def settle(system_key: str, pattern: str, grid: int = 256, steps: int = 300,
           measure_steps: int = 64) -> Unit:
    """Run the pattern alone for `steps`, then crop its object (centred, a margin of R/2
    around its potential basin) and measure its heading and speed over `measure_steps`."""
    system = registry.make_system(system_key)
    reach = system.reach(pattern)
    potential_fn = jax.jit(system.potential)
    state = engine.advance(system, system.init(pattern, (grid, grid), jax.random.key(0)), steps)
    frame, labels = objects(system, state, potential_fn, reach)
    masses = np.bincount(labels.ravel(), weights=frame.ravel())
    masses[0] = 0
    mask = labels == masses.argmax()
    cy, cx = np.round(torus_centroid(mask)).astype(int)
    shift = (grid // 2 - cy, grid // 2 - cx)
    cells = np.roll(state_cells(state), shift, (0, 1))
    mask = np.roll(mask, shift, (0, 1))
    margin = max(1, reach // 2)
    zone = ndimage.binary_dilation(mask, iterations=margin)
    cells = np.where(zone if cells.ndim == 2 else zone[..., None], cells, 0)
    ys, xs = np.nonzero(zone)
    crop = cells[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    path = [np.asarray(system.observe(state), np.float64)]
    for _ in range(measure_steps):
        state = engine.advance(system, state, 1)
        path.append(np.asarray(system.observe(state), np.float64))
    c = centers(path)
    d = c[-1] - c[0]
    return Unit(system_key, pattern, crop, math.degrees(math.atan2(d[0], d[1])),
                float(np.hypot(*d) / measure_steps), float(masses.max()), max(crop.shape[:2]))


def rotate(cells: np.ndarray, degrees: float, discrete: bool) -> np.ndarray:
    """`cells` turned so that a heading θ becomes θ + `degrees` (array coordinates, y down).
    Discrete systems turn in quarter turns only."""
    if discrete:
        return np.rot90(cells, -round(degrees / 90) % 4, (0, 1))
    out = ndimage.rotate(cells, -degrees, axes=(1, 0), reshape=True, order=1)
    return np.clip(out, 0, 1).astype(cells.dtype)


def place(grid: np.ndarray, cells: np.ndarray, centre) -> None:
    """Add `cells` to `grid` (in place) with its centre at `centre`, wrapping around."""
    h, w = cells.shape[:2]
    ys = (np.arange(h) + round(float(centre[0])) - h // 2) % grid.shape[0]
    xs = (np.arange(w) + round(float(centre[1])) - w // 2) % grid.shape[1]
    region = grid[np.ix_(ys, xs)]
    grid[np.ix_(ys, xs)] = np.maximum(region, cells)


@dataclass
class Collision:
    """One collision experiment and its outcome."""
    system: str
    pattern: str
    kind: str  # "pair", "noise" or "alone" (a control: the unit placed with nothing to meet)
    delta: float  # degrees between the two units' headings; 180 = head-on
    offset: float  # impact offset, in unit sizes
    mirror: bool  # the second unit mirrored (the other chirality)
    grid: int
    steps: int
    unit_mass: float
    label: str = ""
    final_units: int = 0  # objects of at least half a unit's mass
    final_masses: list = field(default_factory=list)
    target_masses: list = field(default_factory=list)  # each tracked unit's final mass
    same_object: bool = False  # both tracked units ended in one object
    total_mass_ratio: float = 0.0  # final total mass / initial
    takeovers: int = 0  # tracker takeovers (lost overlaps)
    contact: bool = False  # the units touched: one object, or centres within a unit size

    def record(self) -> dict:
        return asdict(self)


def collide(unit: Unit, kind: str = "pair", delta: float = 180, offset: float = 0.0,
            mirror: bool = False, grid: int | None = None, after: int = 400,
            every: int = 2) -> Collision:
    """Throw two copies of `unit` at each other (`kind` "pair": headings `delta` degrees
    apart, impact offset `offset` unit sizes, the second copy mirrored if `mirror`), or the
    unit at a patch of noise of its own size ("noise"), meeting at the grid's centre, or
    the unit alone ("alone", a control); run until they meet and `after` steps more,
    detecting every `every` steps; label it. The grid (square) is large enough that the
    courses do not wrap around the torus: at least 256 and 2 × start distance + 3 sizes."""
    system = registry.make_system(unit.system)
    discrete = system.n_states(unit.pattern) is not None
    reach = system.reach(unit.pattern)
    potential_fn = jax.jit(system.potential)
    gap = unit.size + max(4, reach)
    sin_half = math.sin(math.radians(delta) / 2) if kind == "pair" else 1.0
    lead = max(1.5 * unit.size, gap / (2 * max(sin_half, 0.2)))  # start distance to the meeting point
    if grid is None:
        grid = max(256, -(-int(2 * lead + 3 * unit.size) // 64) * 64)
    centre = np.array([grid / 2, grid / 2])
    A = np.zeros((grid, grid) + unit.cells.shape[2:], unit.cells.dtype)

    def course(heading: float) -> np.ndarray:
        rad = math.radians(heading)
        return np.array([math.sin(rad), math.cos(rad)])  # (dy, dx)

    base = unit.heading if discrete else 0.0  # discrete units keep their own heading
    a_cells = rotate(unit.cells, base - unit.heading, discrete)
    a_start = centre - lead * course(base)
    place(A, a_cells, a_start)
    starts = [a_start]
    if kind == "pair":
        b = unit.cells[::-1] if mirror else unit.cells
        b_heading = -unit.heading if mirror else unit.heading  # mirroring y flips the heading
        normal = course(base + delta + 90)
        b_start = centre - lead * course(base + delta) + offset * unit.size * normal
        place(A, rotate(b, base + delta - b_heading, discrete), b_start)
        starts.append(b_start)
    elif kind == "noise":  # a patch of noise at the meeting point
        rng = np.random.default_rng(0)
        shape = (unit.size, unit.size) + unit.cells.shape[2:]
        noise = (rng.random(shape) < 0.5).astype(unit.cells.dtype) if discrete else \
            rng.random(shape).astype(unit.cells.dtype)
        if system.n_states(unit.pattern) and system.n_states(unit.pattern) > 2:
            noise = rng.integers(0, system.n_states(unit.pattern), shape).astype(unit.cells.dtype)
        place(A, noise, centre)
    state = system.with_cells(unit.pattern, A)
    steps = int(lead / max(unit.speed, 1e-3)) + after
    frame, labels = objects(system, state, potential_fn, reach)
    initial_mass = frame.sum()
    contact = False
    trackers = []
    for start in starts:  # each tracker starts on the object nearest its unit's start
        tracker = Tracker()
        ids = np.unique(labels[labels > 0])
        d = np.array([torus_centroid(labels == i) for i in ids]) - start
        d = np.minimum(np.abs(d), grid - np.abs(d))
        tracker.target = labels == ids[np.argmin(np.hypot(d[:, 0], d[:, 1]))]
        tracker.last_centre = torus_centroid(tracker.target)
        trackers.append(tracker)
    for _ in range(0, steps, every):
        state = engine.advance(system, state, every)
        frame, labels = objects(system, state, potential_fn, reach)
        for i, tracker in enumerate(trackers):
            others = [t.target for j, t in enumerate(trackers) if j != i and t.target is not None]
            tracker.step(frame, labels, np.any(others, axis=0) if others else None)
        if kind == "pair" and not contact and all(t.target is not None for t in trackers):
            a, b = (t.target for t in trackers)
            d = np.abs(torus_centroid(a) - torus_centroid(b))
            d = np.minimum(d, grid - d)
            contact = bool((a & b).any()) or np.hypot(*d) < unit.size
        elif kind != "pair":
            contact = True  # the patch sits on the unit's course; alone: nothing to miss
    masses = np.bincount(labels.ravel(), weights=frame.ravel())[1:]
    masses = masses[masses > 0]
    result = Collision(unit.system, unit.pattern, kind, delta, offset, mirror, grid, steps,
                       unit.mass)
    result.final_masses = sorted((float(m) for m in masses), reverse=True)
    result.final_units = int((masses >= 0.5 * unit.mass).sum())
    result.target_masses = [float(frame[t.target].sum()) if t.target is not None else 0.0
                            for t in trackers]
    result.same_object = bool(kind == "pair" and all(t.target is not None for t in trackers)
                              and (trackers[0].target & trackers[1].target).any())
    result.total_mass_ratio = float(frame.sum() / initial_mass)
    result.takeovers = sum(t.switches for t in trackers)
    result.contact = bool(contact)
    result.label = label(result, initial_mass)
    return result


def label(c: Collision, initial_mass: float | None = None) -> str:
    """From the final objects, not from the trackers (identity through a merge and a split
    is ambiguous). Objects are unit-sized at ±25 % of a unit's mass, unit-like at 0.4-2×
    (a lone unit may settle larger than units among others: the Aquarium unit at about
    twice), large above 1.25×.
    missed: the two never touched; died: under half the initial mass is left (a pair), or
    the unit under half its mass (noise: the patch's own mass may vanish); divided: more
    unit-like objects than thrown, holding 70 % of the mass (replication); goo: more than
    twice the initial mass otherwise (in Life: growing debris); survived: the thrown units
    are unit-sized objects again, and nothing else of half a unit's mass or more is left;
    merged: fewer unit-sized objects than thrown, with a large one; changed: anything else.
    In Life, still debris (blocks, beehives) is unit-like by mass, so it counts as
    divided."""
    thrown = 2 if c.kind == "pair" else 1
    m = np.array(c.final_masses, float)
    u = c.unit_mass
    unit_sized = int(((m >= 0.75 * u) & (m <= 1.25 * u)).sum())
    unit_like = (m >= 0.4 * u) & (m <= 2 * u)
    large = int((m > 1.25 * u).sum())
    others = int(((m >= 0.5 * u) & (m < 0.75 * u)).sum())
    if c.kind != "pair" and c.target_masses[0] < 0.5 * u and c.total_mass_ratio <= 2:
        return "died"
    if c.kind == "pair" and c.total_mass_ratio < 0.5:
        return "died"
    if not c.contact:
        return "missed"
    if unit_like.sum() > thrown and m[unit_like].sum() >= 0.7 * m.sum():
        return "divided"
    if c.total_mass_ratio > 2:
        return "goo"
    if unit_sized == thrown and large == 0 and others == 0:
        return "survived"
    if unit_sized < thrown and large:
        return "merged"
    return "changed"
