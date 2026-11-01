"""The goal measure (MEASURES.md §8.1d), in the measure list: autopoiesis, the complexity
field C = S × M × I of `complexity.goal` — structure, self-maintenance under perturbation and
individuality, per object. A field costs seconds (twin runs per object), so by default it is
computed in a background thread: each tick reports the latest finished field and starts a
new one once the last is done and `every` ticks have passed; the simulation never waits.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from complexity.measures.base import NO_CONTEXT, Measure
from complexity.registry import register_measure
from complexity.systems.base import System


@register_measure
class Autopoiesis(Measure):
    key = "autopoiesis"
    icon = "🧫"
    equation = r"C = S \times M \times I"
    reference = "the goal measure of this project (MEASURES.md §8.1d); after Maturana & Varela 1980"
    name = "Autopoiesis (goal measure C)"
    unit = "C"
    map_unit = "C"
    group = "Individuality"
    description = (
        "The prototype goal measure: per object (the basins of the potential K ∗ A), "
        "C = S × M × I. S, structure: how much better the object compresses than its cells "
        "shuffled within its outline. M, self-maintenance: twin runs with the object hit by "
        "noise at several strengths, `steps` steps on: how much of its shape comes back "
        "(up to the drift of a negligible hit), how stable it is, and how much of the "
        "difference stays near it. I, individuality: isolated from other objects, bounded, "
        "not spreading. Low for empty space, noise, frozen noise, goo and the Life glider; "
        "highest for the Aquarium. `report`: the largest C over objects (default) or the "
        "mean. Computed in the background every `every` ticks (seconds per field); the map "
        "is the field C(x, y)."
    )
    needs_state = True
    map = "local"
    peaks_at = "structure"
    report = "max"
    every = 10
    steps = 60
    max_objects = 64
    background = True
    params = ("report", "every", "steps", "max_objects", "background")
    choices = {"report": ("max", "mean")}
    granularity = "pattern"
    scope = "local"
    temporal = "dynamic"
    subject = "instance"  # the state now, by intervention
    estimated_from = "run"
    applies_to = "systems whose cells can be set (Lenia, Integer Lenia, multi-channel Lenia, Life, baselines)"

    def __init__(self, **params):
        super().__init__(**params)
        self._pool: ThreadPoolExecutor | None = None
        self._pending: Future | None = None
        self._ticks = 0  # since the last field was started
        self._field: np.ndarray | None = None
        self._scores: list = []

    def accepts(self, system) -> bool:
        return type(system).with_cells is not System.with_cells

    def compute(self, frames, ctx=NO_CONTEXT) -> float:
        raise TypeError("autopoiesis needs the system state: use compute_state")

    def _field_now(self, system, state, ctx):
        from complexity.goal import complexity_field  # imports collisions, which needs the registry

        return complexity_field(system, ctx.pattern, state, steps=self.steps,
                                max_objects=self.max_objects)

    def _update(self, system, state, ctx) -> None:
        if not self.background:
            self._field, self._scores = self._field_now(system, state, ctx)
            return
        if self._pending is not None and self._pending.done():
            self._field, self._scores = self._pending.result()
            self._pending = None
        self._ticks += 1
        if self._pending is None and (self._field is None or self._ticks >= self.every):
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=1)
            self._pending = self._pool.submit(self._field_now, system, state, ctx)
            self._ticks = 0

    def compute_state(self, system, state, rng, ctx=NO_CONTEXT) -> float:
        self._update(system, state, ctx)
        if not self._scores:
            return 0.0
        values = [s.C for s in self._scores]
        return float(max(values) if self.report == "max" else np.mean(values))

    def compute_state_map(self, system, state, rng, ctx=NO_CONTEXT) -> np.ndarray:
        """The latest field C(x, y) (zeros until the first is ready)."""
        if self._field is None:
            return np.zeros(np.asarray(system.observe(state)).shape, np.float32)
        return self._field
