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
    reach: int | None = None  # how far a cell's next state looks, in cells (System.reach)


NO_CONTEXT = Context()  # a continuous system, no discretization


class Measure(ABC):
    """A function from the most recent `window` frames of a system to one number.

    `window = 1` makes a static (per-frame) measure; `window > 1` a dynamic one that sees
    a sliding window of consecutive frames, oldest first. Before the window fills, the
    measure receives the frames available so far, and the run flags its value as warming up.

    Orientation: higher value = more complex by this measure's definition (MEASURES.md §2). A
    quantity that runs the other way is negated or inverted in `compute`, and the
    `description` says so. `peaks_at` says where the measure is highest: "randomness"
    (compression, entropy), "structure" (transfer entropy), "mixed" (neither reliably:
    logical depth) or "none" (a statistic for reference, not a complexity measure: mass,
    speed, change rate); the tests check the first two.

    `needs_state` measures (Lyapunov exponent) need the dynamics, not frames: they implement
    `compute_state(system, state, rng, ctx)`, which the run calls instead of `compute`.

    `backend = "host"` measures receive a list of NumPy frames; `backend = "jax"` measures a
    device array (window, height, width). Frames are float32 in [0, 1], or, with
    `needs_discrete`, int32 symbols 0..ctx.n_levels-1 (see `discretize`).

    Attributes named in `params` are tunable: `make_measure(key, window=32)`.

    `map` says how the measure gives a spatial map (MEASURES.md §4): "local" measures
    implement `compute_map` (or `compute_state_map`), one value per cell; "tile" measures
    are applied by the run to tiles of the frame; None: no map.

    `ratio` names a normalized, per-object version (MEASURES.md §5.10): the grid total above an empty
    frame ((value − empty value) × cells) divided by the mean "mass", or the "change" rate
    Σ|ΔA| per step, over the measure's window.
    The run reports it instead of the value when asked (`Run(normalized=True)`), in
    `ratio_unit`. The map stays per cell, in `map_unit`.
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
    needs_potential: bool = False  # compute also gets `potentials=`: K ∗ A per frame, or None
    map: str | None = None  # "local", "tile" or None
    map_unit: str = ""  # unit of the map if it differs from `unit`
    n_levels: int | None = None  # None: n_states if discrete, else DEFAULT_LEVELS
    peaks_at: str = "randomness"  # "randomness", "structure", "mixed" or "none"
    ratio: str | None = None  # None, "mass" or "change": see the docstring
    ratio_unit: str = ""
    icon: str = ""  # an emoji before the name in the UI; variants of a measure share one
    equation: str = ""  # LaTeX, shown in the measure's info popup (KaTeX)
    reference: str = ""  # short citations, as in assets/README.md
    params: tuple[str, ...] = ()
    choices: dict[str, tuple[str, ...]] = {}  # allowed values of text parameters
    applies_to: str = ""  # which systems, if not all: shown when a measure is greyed out
    # Classification from Complexity_Measure.pdf §1.1-1.2, as corrected by the reanalysis
    # in MEASURES.md §2.3: what the measure is, in the literature. Shown as UI tags.
    granularity: str = "pattern"  # "pattern", "rule" or "system"
    scope: str = "global"  # "local" or "global"
    temporal: str = "static"  # "static" or "dynamic"
    # What this implementation computes (MEASURES.md §2.3): the property of an
    # "instance" (one frame), a "trajectory" (a stretch of a run), a "rule" or an
    # "ensemble"; estimated from one "object", one "run" or an "ensemble" of runs.
    subject: str = "instance"
    estimated_from: str = "object"

    def __init__(self, **params):
        for k, v in params.items():
            if k not in self.params:
                raise ValueError(f"measure {self.key!r} has no parameter {k!r}")
            setattr(self, k, self._cast(k, v))

    def _cast(self, name: str, value):
        """`value` as the parameter's type (the UI sends text and numbers), checked: numbers
        positive, levels at least 2, text one of `choices`. None keeps an "auto" default."""
        default = getattr(type(self), name)
        if value is None or value == "":
            if default is None:
                return None
            raise ValueError(f"{self.key}: {name} needs a value")
        if name in self.choices:
            if value not in self.choices[name]:
                raise ValueError(f"{self.key}: {name} must be one of {', '.join(self.choices[name])}")
            return value
        if isinstance(default, bool):
            return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes", "on")
        kind = int if default is None else type(default)  # None: n_levels, an int
        try:
            value = kind(value)
        except (TypeError, ValueError):
            raise ValueError(f"{self.key}: {name} must be a{'n integer' if kind is int else ' number'}") from None
        if value <= 0 or (name == "n_levels" and value < 2):
            raise ValueError(f"{self.key}: {name} must be {'at least 2' if name == 'n_levels' else 'positive'}")
        return value

    def config(self) -> dict:
        """Current parameter values, recorded with results."""
        return {k: getattr(self, k) for k in self.params}

    def levels(self, n_states: int | None) -> int:
        """Symbols this measure uses on a system with `n_states` (None if continuous)."""
        return self.n_levels or n_states or DEFAULT_LEVELS

    def accepts(self, system) -> bool:
        """Whether this measure applies to `system`; override to restrict it."""
        return True

    def compute_map(self, frames, ctx: Context = NO_CONTEXT) -> np.ndarray:
        """For `map = "local"`: a (height, width) map whose mean is about `compute`'s value."""
        raise NotImplementedError

    @abstractmethod
    def compute(self, frames, ctx: Context = NO_CONTEXT) -> float:
        """Frames oldest first; see the class docstring for their type."""


class RuleMeasure(Measure):
    """A property of a rule (the system with a pattern's parameters), not of a frame:
    transient length, Langton's lambda, functional information (MEASURES.md §5.7). Computed
    once per (system, pattern, parameters) from the rule or from batched runs of their own,
    cached, and reported as a constant on every tick of a run."""

    granularity = "rule"
    group = "Rule-level"
    subject = "rule"
    estimated_from = "ensemble"

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
