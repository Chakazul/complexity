"""Geometric measures of a frame (MEASURES.md §5.5): fractal dimension (FD) by box counting."""

from __future__ import annotations

import numpy as np

from complexity.measures.base import NO_CONTEXT, Measure
from complexity.registry import register_measure


def box_counts(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(box sizes 2, 4, ... up to min(h, w) / 4, number of boxes holding a True cell).
    Each size counts boxes over the largest multiple of the size that fits."""
    sizes, counts = [], []
    s = 2
    while s <= min(mask.shape) // 4:
        h, w = (d - d % s for d in mask.shape)
        boxes = mask[:h, :w].reshape(h // s, s, w // s, s).any(axis=(1, 3))
        sizes.append(s)
        counts.append(boxes.sum())
        s *= 2
    return np.array(sizes), np.array(counts)


@register_measure
class FractalDimension(Measure):
    key = "fractal_dimension"
    icon = "❄️"
    name = "Fractal dimension (box counting)"
    unit = "dimension"
    group = "Geometry"
    description = (
        "Box-counting dimension of the cells above `threshold`: the slope of log(boxes "
        "occupied) against log(1 / box size), for boxes of 2, 4, ... up to a quarter of "
        "the grid. 0 for an empty frame or a single point-like object, 1 for curves, 2 for "
        "frames filled throughout (including noise)."
    )
    threshold = 0.1
    params = ("threshold",)

    def compute(self, frames, ctx=NO_CONTEXT) -> float:
        sizes, counts = box_counts(frames[-1] > self.threshold)
        keep = counts > 0
        if keep.sum() < 2:
            return 0.0
        slope = np.polyfit(np.log(sizes[keep]), np.log(counts[keep]), 1)[0]
        return float(max(-slope, 0.0))
