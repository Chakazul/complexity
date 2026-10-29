"""Runs: one system instance plus the measures applied to it, stepped together.

Shared by the interactive server and the headless CLI so both compute identical values.
The last `max(window)` frames stay on the device; host measures and the UI get them through
one `device_get` per measured tick, of the frames that are new since the last one.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from complexity import engine, registry
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
        self.system = registry.make_system(spec.system)
        self.size = tuple(spec.size or self.system.default_size)
        self.n_states = self.system.n_states(spec.pattern)
        self.state = self.system.init(spec.pattern, self.size, jax.random.key(spec.seed))
        all_measures = [registry.make_measure(k) for k in measure_keys]
        self.measures = [m for m in all_measures if m.accepts(self.system)]
        self.skipped = [m.key for m in all_measures if not m.accepts(self.system)]
        self.window = max([1] + [m.window for m in self.measures])
        self.frames = self.system.observe(self.state)[None]  # device ring buffer, oldest first
        self._host = deque(maxlen=self.window)  # host copies of the same frames
        self._unsynced = 1  # frames at the end of `frames` not yet in `_host`
        self.t = 0

    def step(self, n: int = 1) -> None:
        if n <= 0:
            return
        observed = min(n, self.window)  # older frames would fall out of the window anyway
        if n > observed:
            self.state = engine.advance(self.system, self.state, n - observed)
        self.state, new = engine.rollout(self.system, self.state, observed)
        self.frames = jnp.concatenate([self.frames, new])[-self.window :]
        self._unsynced = min(self._unsynced + observed, self.window)
        self.t += n

    def host_frames(self) -> list[np.ndarray]:
        """The window's frames as NumPy arrays, oldest first. Frames already on the host are
        kept as the same objects, so the compression caches recognise them."""
        if self._unsynced:
            self._host.extend(np.asarray(jax.device_get(self.frames[-self._unsynced :])))
            self._unsynced = 0
        return list(self._host)

    def measure(self) -> tuple[dict[str, float], dict[str, float]]:
        """Values and compute times (ms) of every measure at the current step."""
        values, ms = {}, {}
        frames = self.host_frames()
        for m in self.measures:
            t0 = time.perf_counter()
            values[m.key] = float(m.compute(frames[-m.window :]))
            ms[m.key] = (time.perf_counter() - t0) * 1000
        return values, ms

    def frame_u8(self) -> np.ndarray:
        return to_uint8(self.host_frames()[-1])
