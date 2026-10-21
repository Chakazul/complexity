"""Runs: one system instance plus the measures applied to it, stepped together.

Shared by the interactive server and the headless CLI so both compute identical values.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from complexity import registry
from complexity.measures.base import to_uint8


@dataclass
class RunSpec:
    system: str
    pattern: str
    size: tuple[int, int] | None = None
    seed: int = 0

    @property
    def label(self) -> str:
        cls = registry.systems()[self.system]
        return f"{cls.name} · {cls.patterns[self.pattern]}"


class Run:
    def __init__(self, spec: RunSpec, measure_keys: list[str]):
        self.spec = spec
        self.system = registry.make_system(spec.system, size=spec.size, seed=spec.seed)
        self.system.reset(spec.pattern)
        all_measures = [registry.make_measure(k) for k in measure_keys]
        self.measures = [m for m in all_measures if m.accepts(self.system)]
        self.skipped = [m.key for m in all_measures if not m.accepts(self.system)]
        self.frames = deque([self.system.frame()], maxlen=max([1] + [m.window for m in self.measures]))
        self.t = 0

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self.system.step()
            self.frames.append(self.system.frame())
            self.t += 1

    def measure(self) -> tuple[dict[str, float], dict[str, float]]:
        """Values and compute times (ms) of every measure at the current step."""
        values, ms = {}, {}
        frames = list(self.frames)
        for m in self.measures:
            t0 = time.perf_counter()
            values[m.key] = float(m.compute(frames[-m.window :]))
            ms[m.key] = (time.perf_counter() - t0) * 1000
        return values, ms

    def frame_u8(self) -> np.ndarray:
        return to_uint8(self.frames[-1])
