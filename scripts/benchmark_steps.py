"""Simulation speed (steps/s, no measures) per system and grid size (results in README).

Run: uv run python scripts/benchmark_steps.py          (default device)
     JAX_PLATFORMS=cpu uv run python scripts/benchmark_steps.py
"""
import time

import jax

from complexity import engine, registry

RUNS = [("life", "random"), ("lenia", "random"), ("lenia", "random_continuous")]
print(f"device: {jax.devices()[0]}")
print(f"{'run':<26}" + "".join(f"{n:>9}²" for n in (128, 256, 512)) + "   (steps/s)")
for key, pattern in RUNS:
    system, row = registry.make_system(key), []
    for n in (128, 256, 512):
        state = system.init(pattern, (n, n), jax.random.key(0))
        jax.block_until_ready(engine.advance(system, state, 10))  # compile
        steps, elapsed = 10, 0.0
        while elapsed < 1.0:
            steps *= 2
            t = time.perf_counter()
            jax.block_until_ready(engine.advance(system, state, steps))
            elapsed = time.perf_counter() - t
        row.append(steps / elapsed)
    print(f"{key + ':' + pattern:<26}" + "".join(f"{v:>10,.0f}" for v in row))
