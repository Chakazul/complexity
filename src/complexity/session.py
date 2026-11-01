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
from complexity.measures.individuality import FeatureIndividuality
from complexity.measures.statistics import change_rate
from complexity.systems.base import System

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


# maps that are not measures (`Run.map`): key -> (name, unit or None for labels)
SPECIAL_MAPS = {"potential": ("Neighbour sum (K ∗ A)", "K ∗ A"),
                "objects": ("Objects (target highlighted)", None)}


class Run:
    def __init__(self, spec: RunSpec, measure_keys: list[str], measure_params: dict | None = None,
                 normalized: bool = False):
        self.spec = spec
        self.normalized = normalized  # report measures' normalized versions (Measure.ratio)
        self._empty: dict[tuple[str, int], float] = {}  # measure values on empty frames
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
                           spec.system, spec.pattern, self.system.reach(spec.pattern))
            for m in self.measures
        }
        self.window = max([1] + [m.window for m in self.measures])
        # device frames, oldest first
        self.frames = deque([self.system.observe(self.state)], maxlen=self.window)
        # the potential K ∗ A of the same frames, when a measure needs it and the system has it
        has_potential = type(self.system).potential is not System.potential
        self._potential = (jax.jit(self.system.potential)
                           if has_potential and any(m.needs_potential for m in self.measures) else None)
        self.potentials = deque(maxlen=self.window)
        self._potential_once = None  # for the "potential" and "objects" maps without a measure
        self._host_potentials = deque(maxlen=self.window)
        self._unsynced_potentials = 0
        if self._potential:
            self.potentials.append(self._potential(self.state))
            self._unsynced_potentials = 1
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
            if self._potential:
                self.potentials.append(self._potential(self.state))
        self._unsynced = min(self._unsynced + observed, self.window)
        if self._potential:
            self._unsynced_potentials = min(self._unsynced_potentials + observed, self.window)
        self.t += n

    def host_frames(self) -> list[np.ndarray]:
        """The window's frames as NumPy arrays, oldest first. Frames already on the host are
        kept as the same objects, so the compression caches recognise them."""
        if self._unsynced:
            new = list(self.frames)[-self._unsynced :]
            self._host.extend(np.asarray(f) for f in jax.device_get(new))
            self._unsynced = 0
        return list(self._host)

    def host_potentials(self) -> list[np.ndarray] | None:
        """The window's potentials on the host, aligned with `host_frames`; None if not
        recorded."""
        if not self._potential:
            return None
        if self._unsynced_potentials:
            new = list(self.potentials)[-self._unsynced_potentials :]
            self._host_potentials.extend(np.asarray(u, np.float64) for u in jax.device_get(new))
            self._unsynced_potentials = 0
        return list(self._host_potentials)

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
            elif m.needs_potential:
                pots = self.host_potentials()
                frames = self._frames_for(m)
                values[m.key] = float(m.compute(
                    frames, ctx, potentials=pots[-len(frames):] if pots is not None else None))
            else:
                values[m.key] = float(m.compute(self._frames_for(m), ctx))
            if self.normalized and m.ratio:
                values[m.key] = self._normalize(m, values[m.key])
            ms[m.key] = (time.perf_counter() - t0) * 1000
        return values, ms

    def _normalize(self, m, value: float) -> float:
        """`value` (a grid average) as a total over the grid above an empty frame, divided
        by the mean mass, or the change rate, of the measure's window: MEASURES.md §5.10.
        0 if the divisor is (near) 0: an empty grid, or nothing changing."""
        frames = self.host_frames()[-m.window :]
        if m.ratio == "change":
            divisor = change_rate(frames)
        else:
            divisor = float(np.mean([f.sum(dtype=np.float64) for f in frames]))
        if divisor <= 1e-9:
            return 0.0
        excess = value - self._empty_value(m, len(frames))
        return excess * self.size[0] * self.size[1] / divisor

    def _empty_value(self, m, n: int) -> float:
        """The measure on `n` empty frames of this grid (a compressor's fixed overhead; 0
        for most measures), computed once per measure and window length."""
        key = (m.key, n)
        if key not in self._empty:
            ctx = self.contexts[m.key]
            zeros = np.zeros(self.size, np.float32)
            if m.backend == "jax":
                frames = jnp.zeros((n, *self.size), jnp.float32)
                frames = discretize(frames, ctx.n_levels) if m.needs_discrete else frames
            else:
                frames = [discretize(zeros, ctx.n_levels) if m.needs_discrete else zeros
                          for _ in range(n)]
            self._empty[key] = float(m.compute(frames, ctx))
        return self._empty[key]

    def map_tile(self) -> int:
        """Tile size for per-tile maps: about 16 tiles across, at least 8 cells."""
        return max(8, min(self.size) // 16)

    def map(self, key: str) -> np.ndarray | None:
        """The spatial map of measure `key` at the current step (MEASURES.md §4), float32
        (height, width) for local measures or (height / tile, width / tile) per tile; None
        if the measure has no map or does not run here. Two maps are not measures:
        "potential", the neighbour sum K ∗ A, and "objects", the detected creatures with
        the target as 1 (`individuality.object_map`)."""
        if key in SPECIAL_MAPS:
            return self._special_map(key)
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

    def _special_map(self, key: str) -> np.ndarray | None:
        if type(self.system).potential is System.potential:
            potential = None
        elif self._potential:  # recorded for a measure
            potential = self.host_potentials()[-1]
        else:
            if self._potential_once is None:
                self._potential_once = jax.jit(self.system.potential)
            potential = np.asarray(self._potential_once(self.state), np.float64)
        if key == "potential":
            return None if potential is None else potential.astype(np.float32)
        iti = next((m for m in self.measures if m.key == "iti_features"), None)
        if iti is not None:  # the creatures and target that the measure tracks
            frames = self.host_frames()[-iti.window :]
            pots = self.host_potentials()
            return iti.objects(frames, self.contexts[iti.key],
                               pots[-len(frames):] if pots is not None else None)
        frame = self.host_frames()[-1]
        ctx = Context(self.n_states, None, self.spec.system, self.spec.pattern,
                      self.system.reach(self.spec.pattern))
        return FeatureIndividuality().objects([frame], ctx, None if potential is None else [potential])

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
