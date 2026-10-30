"""The interfaces every complexity measure implements (PLAN.md §3.3)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

# Coarse-graining of continuous frames for measures that need symbols, unless the measure's
# `n_levels` says otherwise.
DEFAULT_LEVELS = 8


@dataclass(frozen=True)
class Context:
    """What a measure may know about the run besides its frames."""

    n_states: int | None = None  # the system's discrete states, None if continuous
    n_levels: int | None = None  # symbols in discretized frames (set for `needs_discrete`)
    system: str = ""
    pattern: str = ""


NO_CONTEXT = Context()  # a continuous system, no discretization


class Measure(ABC):
    """A function from the most recent `window` frames of a system to one number.

    `window = 1` makes a static (per-frame) measure; `window > 1` a dynamic one that sees
    a sliding window of consecutive frames, oldest first. Before the window fills, the
    measure receives the frames available so far, and the run flags its value as warming up.

    Orientation: higher value = more complex by this measure's definition (PLAN.md §2). A
    quantity that runs the other way is negated or inverted in `compute`, and the
    `description` says so. `peaks_at` says where the measure is highest: "randomness"
    (compression, entropy), "structure" (transfer entropy) or "mixed" (neither reliably:
    logical depth); the tests check the first two.

    `needs_state` measures (Lyapunov exponent) need the dynamics, not frames: they implement
    `compute_state(system, state, rng, ctx)`, which the run calls instead of `compute`.

    `backend = "host"` measures receive a list of NumPy frames; `backend = "jax"` measures a
    device array (window, height, width). Frames are float32 in [0, 1], or, with
    `needs_discrete`, int32 symbols 0..ctx.n_levels-1 (see `discretize`).

    Attributes named in `params` are tunable: `make_measure(key, window=32)`.
    """

    key: str  # registry key, e.g. "png"
    name: str  # display name
    unit: str  # e.g. "bits/cell"
    description: str = ""
    group: str = "Other"  # heading in the UI's measure list, e.g. "Information theory"
    window: int = 1
    backend: str = "host"  # "host" or "jax"
    needs_discrete: bool = False
    needs_state: bool = False
    n_levels: int | None = None  # None: n_states if discrete, else DEFAULT_LEVELS
    peaks_at: str = "randomness"  # "randomness", "structure" or "mixed"
    params: tuple[str, ...] = ()
    # Classification from Complexity_Measure.pdf §1.1-1.2, shown in the UI.
    granularity: str = "pattern"  # "pattern", "rule" or "system"
    scope: str = "global"  # "local" or "global"
    temporal: str = "static"  # "static" or "dynamic"

    def __init__(self, **params):
        for k, v in params.items():
            if k not in self.params:
                raise ValueError(f"measure {self.key!r} has no parameter {k!r}")
            setattr(self, k, v)

    def config(self) -> dict:
        """Current parameter values, recorded with results."""
        return {k: getattr(self, k) for k in self.params}

    def levels(self, n_states: int | None) -> int:
        """Symbols this measure uses on a system with `n_states` (None if continuous)."""
        return self.n_levels or n_states or DEFAULT_LEVELS

    def accepts(self, system) -> bool:
        """Whether this measure applies to `system`; override to restrict it."""
        return True

    @abstractmethod
    def compute(self, frames, ctx: Context = NO_CONTEXT) -> float:
        """Frames oldest first; see the class docstring for their type."""


class RuleMeasure(Measure):
    """A property of a rule (the system with a pattern's parameters), not of a frame:
    transient length, Langton's lambda, functional information (PLAN.md §5.3). Computed
    once per (system, pattern, parameters) from the rule or from batched runs of their own,
    cached, and reported as a constant on every tick of a run."""

    granularity = "rule"
    group = "Rule-level"

    def compute(self, frames, ctx: Context = NO_CONTEXT) -> float:
        raise TypeError(f"{self.key} is a rule measure: use compute_rule")

    @abstractmethod
    def compute_rule(self, system, pattern: str, ctx: Context = NO_CONTEXT) -> float:
        """The measure's value for `system` running `pattern`'s rule."""


def to_uint8(frame: np.ndarray) -> np.ndarray:
    """Quantize a [0, 1] frame to 8-bit gray, the common input of the compressors."""
    return np.round(np.clip(frame, 0, 1) * 255).astype(np.uint8)


def discretize(frames, n_levels: int):
    """Symbols 0..n_levels-1 from [0, 1] frames, by `n_levels` equal-width bins (PLAN.md §2).

    Exact for a discrete system with `n_levels = n_states`: its values k / (n_states - 1)
    map to k. For a continuous system it is a uniform coarse-graining. Works on NumPy and
    JAX arrays alike and returns the same kind, as int32.
    """
    xp = np if isinstance(frames, np.ndarray) else jnp
    return xp.minimum(xp.floor(xp.asarray(frames) * n_levels), n_levels - 1).astype(xp.int32)
