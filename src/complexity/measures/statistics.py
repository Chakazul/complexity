"""Statistics of a frame for reference, not complexity measures (MEASURES.md §5.9): total
mass, moving speed and change rate.

Speed follows the hand-coded Lenium's `stat.calc_basic_stats`, which recenters the world on
the integer part of the center of mass every step so that a creature never straddles the
torus's edge when its center is taken. Here nothing is shifted or stored: each frame's cell
coordinates are wrapped around the previous center, which is the same as rolling the frame
by it, and the center found is carried to the next frame.
"""

from __future__ import annotations

import numpy as np

from complexity.measures.base import NO_CONTEXT, Measure
from complexity.registry import register_measure


def circular_center(mass: np.ndarray, n: int) -> float:
    """A first center along one axis of a torus with n cells: the mass-weighted circular
    mean of the positions, which no edge splits."""
    angle = 2 * np.pi * np.arange(n) / n
    return float(np.arctan2(mass @ np.sin(angle), mass @ np.cos(angle)) * n / (2 * np.pi)) % n


def centers(frames) -> np.ndarray | None:
    """Centers of mass (y, x) of consecutive frames on a torus, unwrapped, so that their
    differences are the displacements: (time, 2). Each frame's coordinates are wrapped to
    within half the grid of the previous center's integer part (for the first frame, of its
    circular center). None if a frame is empty."""
    h, w = frames[0].shape
    first = np.asarray(frames[0], np.float64)
    if first.sum() <= 0:
        return None
    offset = np.array([circular_center(first.sum(1), h), circular_center(first.sum(0), w)])
    out = []
    for frame in frames:
        frame = np.asarray(frame, np.float64)
        total = frame.sum()
        if total <= 0:
            return None
        center = []
        for axis, (n, o) in enumerate(zip((h, w), np.round(offset))):
            d = (np.arange(n) - o + n // 2) % n - n // 2  # coordinates as if rolled by -o
            center.append(o + frame.sum(1 - axis) @ d / total)
        offset = np.array(center)
        out.append(offset)
    return np.array(out)


def change_rate(frames) -> float:
    """Mean total change per step, Σ|A(t+1) − A(t)| over the cells, in units of mass."""
    f = np.asarray(frames, np.float64)
    return float(np.abs(np.diff(f, axis=0)).sum(axis=(1, 2)).mean()) if len(f) > 1 else 0.0


class _Statistic(Measure):
    group = "Statistics"
    peaks_at = "none"  # not a complexity measure


@register_measure
class Mass(_Statistic):
    key = "mass"
    icon = "⚖️"
    name = "Total mass"
    unit = "cells"
    description = (
        "The sum of the frame's cell values (the mean of the channels for multi-channel "
        "Lenia): the number of live cells in Life. A statistic for reference, not a "
        "complexity measure."
    )

    def compute(self, frames, ctx=NO_CONTEXT) -> float:
        return float(np.asarray(frames[-1], np.float64).sum())


@register_measure
class Speed(_Statistic):
    key = "speed"
    icon = "💨"
    name = "Moving speed"
    unit = "cells/step"
    description = (
        "How fast the center of mass moves: its displacement over the window divided by the "
        "steps, on the torus without jumps at the edge (each frame is centered on the "
        "previous center before its center of mass is taken, as in the hand-coded Lenium). "
        "Life's glider: √2/4 = 0.354. 0 for an empty frame. With several objects, the "
        "speed of their common center. A statistic for reference, not a complexity measure."
    )
    window = 33
    params = ("window",)
    temporal = "dynamic"
    subject = "trajectory"
    estimated_from = "run"

    def compute(self, frames, ctx=NO_CONTEXT) -> float:
        if len(frames) < 2:
            return 0.0
        c = centers(frames)
        return 0.0 if c is None else float(np.linalg.norm(c[-1] - c[0]) / (len(frames) - 1))


@register_measure
class ChangeRate(_Statistic):
    key = "change_rate"
    icon = "🔄"
    name = "Change rate"
    unit = "mass/step"
    description = (
        "How much the frame changes per step: Σ|A(t+1) − A(t)| over the cells, averaged over "
        "the window. Covers motion and change in place alike (a still or frozen pattern: 0; "
        "an oscillator or a creature changing shape in place: > 0). The divisor of transfer "
        "entropy's normalized version. A statistic for reference, not a complexity measure."
    )
    window = 16
    params = ("window",)
    temporal = "dynamic"
    subject = "trajectory"
    estimated_from = "run"

    def compute(self, frames, ctx=NO_CONTEXT) -> float:
        return change_rate(frames)
