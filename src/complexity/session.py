"""Runs: one system instance plus the measures applied to it, stepped together.

Shared by the interactive server and the headless CLI so both compute identical values.
The last `max(window)` frames stay on the device; host measures and the UI get them through
one `device_get` per measured tick, of the frames that are new since the last one; JAX
measures read the device frames directly. Stepping
calls fixed-shape jitted functions in a Python loop, so changing the steps per tick never
recompiles.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from complexity import engine, registry
from complexity.measures.base import Context, RuleMeasure, discretize, to_uint8
from complexity.measures.statistics import Speed

MIN_SPEED = 1e-3  # cells/step: slower counts as not moving for per-object ratios

# Rule-measure values by (measure key, its parameters, system, pattern): they depend on the
# rule only, and some take seconds, so restarts and other runs of the same rule reuse them.
_RULE_VALUES: dict = {}


def rule_value(m: RuleMeasure, system, pattern: str, ctx: Context) -> float:
    key = (m.key, tuple(sorted(m.config().items())), system, pattern)
    if key not in _RULE_VALUES:
        _RULE_VALUES[key] = float(m.compute_rule(system, pattern, ctx))
    return _RULE_VALUES[key]


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
    def __init__(self, spec: RunSpec, measure_keys: list[str], measure_params: dict | None = None,
                 per_object: bool = False):
        self.spec = spec
        self.per_object = per_object  # report measures' per-object ratios (Measure.ratio)
        self.system = registry.make_system(spec.system)
        self.size = tuple(spec.size or self.system.default_size)
        self.n_states = self.system.n_states(spec.pattern)
        self.rng = jax.random.key(spec.seed)
        self.state = self.system.init(spec.pattern, self.size, self.rng)
        engine.advance(self.system, self.state, 0)  # compile now, not on the first big step
        params = measure_params or {}
        all_measures = [registry.make_measure(k, **params.get(k, {})) for k in measure_keys]
        self.measures = [m for m in all_measures if m.accepts(self.system)]
        self.skipped = [m.key for m in all_measures if not m.accepts(self.system)]
        self.contexts = {
            m.key: Context(self.n_states, m.levels(self.n_states) if m.needs_discrete else None,
                           spec.system, spec.pattern)
            for m in self.measures
        }
        self.window = max([1] + [m.window for m in self.measures])
        # device frames, oldest first
        self.frames = deque([self.system.observe(self.state)], maxlen=self.window)
        self._host = deque(maxlen=self.window)  # host copies of the same frames
        self._unsynced = 1  # frames at the end of `frames` not yet in `_host`
        self.t = 0

    def step(self, n: int = 1) -> None:
        if n <= 0:
            return
        observed = min(n, self.window)  # older frames would fall out of the window anyway
        if n > observed:
            self.state = engine.advance(self.system, self.state, n - observed)
        for _ in range(observed):
            self.state, frame = engine.step_observe(self.system, self.state)
            self.frames.append(frame)
        self._unsynced = min(self._unsynced + observed, self.window)
        self.t += n

    def host_frames(self) -> list[np.ndarray]:
        """The window's frames as NumPy arrays, oldest first. Frames already on the host are
        kept as the same objects, so the compression caches recognise them."""
        if self._unsynced:
            new = list(self.frames)[-self._unsynced :]
            self._host.extend(np.asarray(f) for f in jax.device_get(new))
            self._unsynced = 0
        return list(self._host)

    def _frames_for(self, m):
        ctx = self.contexts[m.key]
        if m.backend == "jax":
            frames = jnp.stack(list(self.frames)[-m.window :])
            return discretize(frames, ctx.n_levels) if m.needs_discrete else frames
        frames = self.host_frames()[-m.window :]
        return [discretize(f, ctx.n_levels) for f in frames] if m.needs_discrete else frames

    def measure(self) -> tuple[dict[str, float], dict[str, float]]:
        """Values and compute times (ms) of every measure at the current step."""
        values, ms = {}, {}
        for m in self.measures:
            t0 = time.perf_counter()
            ctx = self.contexts[m.key]
            if isinstance(m, RuleMeasure):
                values[m.key] = rule_value(m, self.system, self.spec.pattern, ctx)
            elif m.needs_state:
                rng = jax.random.fold_in(self.rng, self.t + 1)  # differs from init's key
                values[m.key] = float(m.compute_state(self.system, self.state, rng, ctx))
            else:
                values[m.key] = float(m.compute(self._frames_for(m), ctx))
            if self.per_object and m.ratio:
                values[m.key] = self._per_object(m, values[m.key])
            ms[m.key] = (time.perf_counter() - t0) * 1000
        return values, ms

    def _per_object(self, m, value: float) -> float:
        """`value` (a grid average) as a total over the grid divided by the mean mass, or
        speed × mass, of the measure's window: MEASURES.md §5.10. 0 if the divisor is (near)
        0: an empty grid, or a pattern that does not move."""
        frames = self.host_frames()[-m.window :]
        mass = float(np.mean([f.sum(dtype=np.float64) for f in frames]))
        divisor = mass
        if m.ratio == "speed_mass":
            speed = Speed().compute(frames)
            divisor = mass * speed if speed >= MIN_SPEED else 0.0
        return value * self.size[0] * self.size[1] / divisor if divisor > 1e-9 else 0.0

    def map_tile(self) -> int:
        """Tile size for per-tile maps: about 16 tiles across, at least 8 cells."""
        return max(8, min(self.size) // 16)

    def map(self, key: str) -> np.ndarray | None:
        """The spatial map of measure `key` at the current step (MEASURES.md §4), float32
        (height, width) for local measures or (height / tile, width / tile) per tile; None
        if the measure has no map or does not run here."""
        m = next((m for m in self.measures if m.key == key), None)
        if m is None or m.map is None:
            return None
        ctx = self.contexts[m.key]
        if m.map == "local":
            if m.needs_state:
                rng = jax.random.fold_in(self.rng, self.t + 1)
                return m.compute_state_map(self.system, self.state, rng, ctx)
            return m.compute_map(self._frames_for(m), ctx)
        # per tile, as the excess over an empty tile: on small tiles a compressor's fixed
        # overhead (~8 bits/cell on 8x8) would otherwise swamp the differences
        frames, t = self._frames_for(m), self.map_tile()
        h, w = (d // t for d in self.size)
        empty = m.compute([np.zeros((t, t), f.dtype) for f in frames], ctx)
        return np.array([[m.compute([f[i * t:(i + 1) * t, j * t:(j + 1) * t] for f in frames], ctx)
                          for j in range(w)] for i in range(h)], np.float32) - empty

    def warming_up(self) -> list[str]:
        """Measures whose window is not full yet: their values cover fewer frames."""
        return [m.key for m in self.measures if len(self.frames) < m.window]

    @property
    def rgb(self) -> bool:
        """Whether the UI gets an RGB picture (`System.render`) instead of the frame."""
        return self.system.render(self.state) is not None

    def frame_u8(self) -> np.ndarray:
        """What the UI draws: the frame (h, w) or, for systems that render, RGB (h, w, 3)."""
        picture = self.system.render(self.state)
        return to_uint8(np.asarray(picture if picture is not None else self.host_frames()[-1]))
