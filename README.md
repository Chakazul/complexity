# Complexity Lab

An experimentation platform for applying complexity measures from the literature to
dynamical complex systems — cellular automata, continuous CA, particle systems — and for
comparing which measures suit which kinds of system. See
[assets/Complexity_Measure.pdf](assets/Complexity_Measure.pdf) for the measures and
systems in scope.

Current demo: **Game of Life** and **Lenia** (continuous and Integer Lenia), simulated in
**JAX** (CPU, GPU or TPU), measured by **image and video compression** as proxies for
algorithmic (Kolmogorov) complexity.

## Status and roadmap

The design and the milestone checklist are in **[PLAN.md](PLAN.md)**. In brief:

- **M0 (done):** NumPy demo — the systems and measures below, interactive UI, CLI.
- **M1 (done):** JAX engine for GPU/TPU (CPU JAX supported), Integer Lenia with the
  *minimum Orbium* as the default Lenia pattern.
- **M2 (next):** transfer entropy (a local, dynamic measure), noise baselines for calibration,
  measure-expectation tests.
- **M3–M5:** UI improvements, batch experiments and a results site, then sharing
  (CI, contributor guide, hosted demo).

## Conventions

- **Higher value = higher complexity**, by each measure's own definition. A quantity that
  runs the other way is negated or inverted inside the measure. "Complex" therefore means
  different things to different measures: compression rates noise highest, while
  information-flow measures rate it near zero. Comparing these is the point of the platform.
- Information-based measures report **bits per cell**, so values compare across grid sizes.
- Every measure is tagged with the taxonomy from the PDF: granularity
  (pattern / rule / system), scope (local / global), temporal (static / dynamic).

## Install and run

```bash
uv sync                    # or: pip install -e .
uv run complexity serve    # then open http://127.0.0.1:8000
```

The default install runs JAX on the CPU, so it works anywhere. For a GPU or TPU, add the
matching JAX build, then check which device is active:

```bash
uv sync --extra cuda13     # or --extra cuda12; TPU: uv pip install -U "jax[tpu]"
uv run complexity info     # JAX version, backend and devices
```

In the UI, add up to four runs (system + pattern) to compare side by side, tick the
measures, and press **Start**. Each measure shows its current value per run in the table
and its history as a live chart (hover for values). Measures that do not apply to a
selected system are struck through.

Headless runs write CSV, for scripted experiments and for producing results to share:

```bash
uv run complexity list
uv run complexity run life:gosper_glider_gun lenia:orbium --steps 1000 --out results/demo.csv
uv run complexity run life:random --measures png,video_h264 --size 256x256 --every 10
```

## Systems and patterns

| System | Patterns |
|---|---|
| `life` — Game of Life (discrete) | Gosper glider gun, glider, pulsar, R-pentomino, acorn, random soup |
| `lenia` — Lenia (Integer Lenia `P = 12`, or continuous) | minimum Orbium (default, integer), random soup (integer), Orbium (continuous), random soup (continuous) |

Lenia uses Gaussian kernel shells and growth, the standard for experiments here.

The minimum Orbium is an Integer Lenia glider: 13 cell states (0..12), integer kernel,
potential and growth, so it suits both discrete measures (13 symbols) and continuous ones.
Each system declares `n_states(pattern)`: 2 for Life, 13 for Integer Lenia, `None` for
continuous Lenia.

Simulation speed, steps/s without measures (`scripts/benchmark_steps.py`; AMD Ryzen AI 9
HX 370, 24 threads / NVIDIA RTX 5070 Ti Laptop, JAX 0.12):

| Run | CPU 128² | CPU 256² | CPU 512² | GPU 128² | GPU 256² | GPU 512² |
|---|---|---|---|---|---|---|
| `life:random` | 268,058 | 76,348 | 12,531 | 9,773 | 9,819 | 9,887 |
| `lenia:random` (integer) | 20,082 | 1,343 | 861 | 6,937 | 6,968 | 7,459 |
| `lenia:random_continuous` | 20,130 | 1,370 | 921 | 6,547 | 7,177 | 7,338 |

On the GPU every step costs about the same up to 512² (the step is launch-bound); the CPU
wins on small grids. Either way the host-side compression measures, not the simulation,
set the pace in the UI (§5.1 of PLAN.md).

## Measures

Compression measures report **bits per cell** (per frame for video), so values are
comparable across grid sizes: ~0 for a uniform grid, ~1 for incompressible binary noise,
~8 for incompressible 8-bit noise, plus each format's overhead. Frames are quantized to
8-bit gray first.

| Key | Measure | Window | Notes |
|---|---|---|---|
| `png` | PNG size | 1 | DEFLATE with 2-D row prediction |
| `lzma` | LZMA size of raw frame | 1 | 1-D dictionary compression, no image model |
| `video_lzma` | LZMA size of a frame window | 16 | repeats across time compress away |
| `video_h264` | lossless H.264 size of a frame window | 16 | motion compensation captures translating structures |
| `novelty` | best video ÷ mean best image bits per frame | 16 | fraction of each frame that is new: ~0 static, ~1 fresh noise (unit: ratio) |

Each compressor sees a different kind of regularity: PNG sees smooth 2-D variation, LZMA
exact repeats anywhere, H.264 approximate and sub-pixel motion. Where they disagree tells
you what kind of structure a pattern has; the minimum is the tightest bound. Bits per
cell are size-invariant only for space-filling patterns; isolated objects get "cheaper"
on larger grids, so compare them at equal size.

Window measures see fewer frames until the window fills, so their first ~16 steps are a
warm-up transient.

## Adding a system or a measure

Write one class, decorate it, and add its module to `_BUILTIN_MODULES` in
[src/complexity/registry.py](src/complexity/registry.py). The UI and CLI pick it up.

A **system** ([systems/base.py](src/complexity/systems/base.py)) is a set of pure JAX
functions over a state pytree: `init(pattern, size, rng)` builds the state, `step(state)`
advances it and `observe(state)` renders a float32 `(H, W)` frame in `[0, 1]`. The engine
compiles `step` once and runs it with `lax.scan` (`engine.rollout`) or over a batch of
seeds with `vmap` (`engine.rollout_batch`). Systems whose state is not a grid (particles,
multi-channel) render themselves to a frame in `observe`.

```python
@register_system
class MySystem(System):
    key, name = "mine", "My system"
    patterns = {"default": "Default"}
    def n_states(self, pattern): return None  # or the number of discrete levels
    def init(self, pattern, size, rng): ...
    def step(self, state): ...
    def observe(self, state): ...
```

A **measure** ([measures/base.py](src/complexity/measures/base.py)) implements
`compute(frames)`, receiving the last `window` frames (NumPy, on the host), oldest first,
and returns a value where higher means more complex (see Conventions). Override
`accepts(system)` to restrict it. Measures that need symbols use
`discretize(frames, n_levels)` from `measures/base.py`, which is exact for discrete systems
and a uniform coarse-graining for continuous ones. Tag it with `granularity`, `scope` and `temporal`
following the classification in the PDF.

```python
@register_measure
class MyMeasure(Measure):
    key, name, unit = "mine", "My measure", "bits"
    window = 1
    def compute(self, frames): ...
```

Add a test in [tests/](tests/) and run `uv run --extra dev pytest`.

## Layout

```
src/complexity/
  registry.py            system and measure registries
  engine.py              rollout (lax.scan), rollout_batch (vmap), advance
  session.py             Run: one system + its measures + a device frame buffer (UI and CLI)
  cli.py                 complexity serve | list | info | run
  systems/               base.py, life.py, lenia.py (JAX)
  measures/              base.py, compression.py
  vendor/lenium/         code and data copied from Lenium, with source notes
  app/server.py          FastAPI + WebSocket streaming
  app/static/            browser UI (no build step, no JS dependencies)
```

Computation runs in Python on the server; the browser only draws. The same app can be
deployed as a hosted demo (for example on a Hugging Face Space) without changes.

Lenia code and data come from Lenium, the hand-coded JAX Lenia library, which is not
published yet. The parts needed (patterns `orbium` and `min-orbium`, Lenia and Integer
Lenia) are copied into [src/complexity/vendor/lenium/](src/complexity/vendor/lenium/),
whose README records where each file came from. Integer Lenia is bit-identical to Lenium
for 1000 steps of `min_orbium`. Once Lenium is published, the directory is replaced by `import lenium`, and
the golden tests check that behaviour is unchanged.
