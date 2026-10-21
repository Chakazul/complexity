"""The interface every complexity measure implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class Measure(ABC):
    """A function from the most recent `window` frames of a system to one number.

    `window = 1` makes a static (per-frame) measure; `window > 1` a dynamic one that sees
    a sliding window of consecutive frames, oldest first. Before the window fills, the
    measure receives the frames available so far.
    """

    key: str  # registry key, e.g. "png"
    name: str  # display name
    unit: str  # e.g. "bits/cell"
    description: str = ""
    window: int = 1
    # Classification from Complexity_Measure.pdf §1.1-1.2, shown in the UI.
    granularity: str = "pattern"  # "pattern", "rule" or "system"
    scope: str = "global"  # "local" or "global"
    temporal: str = "static"  # "static" or "dynamic"

    def accepts(self, system) -> bool:
        """Whether this measure applies to `system`; override to restrict it."""
        return True

    @abstractmethod
    def compute(self, frames: Sequence[np.ndarray]) -> float:
        """Frames are float32 (height, width) in [0, 1], oldest first."""


def to_uint8(frame: np.ndarray) -> np.ndarray:
    """Quantize a [0, 1] frame to 8-bit gray, the common input of the compressors."""
    return np.round(np.clip(frame, 0, 1) * 255).astype(np.uint8)
