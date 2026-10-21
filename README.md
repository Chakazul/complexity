# Complexity Lab

An experimentation platform for applying complexity measures from the literature to
dynamical complex systems — cellular automata, continuous CA, particle systems — and for
comparing which measures suit which kinds of system. See
[assets/Complexity_Measure.pdf](assets/Complexity_Measure.pdf) for the measures and
systems in scope.

Current demo: **Game of Life** and **Lenia**, measured by **image and video compression**
as proxies for algorithmic (Kolmogorov) complexity.

## Status and roadmap

The design and the milestone checklist are in **[PLAN.md](PLAN.md)**. In brief:

- **M0 (done):** NumPy demo — the systems and measures below, interactive UI, CLI.
- **M1 (next):** JAX engine for GPU/TPU (CPU JAX supported), Integer Lenia with the
  *minimum Orbium* as the default Lenia pattern.
- **M2:** transfer entropy (a local, dynamic measure), noise baselines for calibration,
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
| `lenia` — Lenia (continuous) | Orbium, random soup |

## Measures

All compression measures report **bits per cell** (per frame for video), so values are
comparable across grid sizes: ~0 for a uniform grid, ~1 for incompressible binary noise,
~8 for incompressible 8-bit noise, plus each format's overhead. Frames are quantized to
8-bit gray first.

| Key | Measure | Window | Notes |
|---|---|---|---|
| `png` | PNG size | 1 | DEFLATE with 2-D row prediction |
| `lzma` | LZMA size of raw frame | 1 | 1-D dictionary compression, no image model |
| `video_lzma` | LZMA size of a frame window | 16 | repeats across time compress away |
| `video_h264` | lossless H.264 size of a frame window | 16 | motion compensation captures translating structures |

Window measures see fewer frames until the window fills, so their first ~16 steps are a
warm-up transient.

## Adding a system or a measure

Write one class, decorate it, and add its module to `_BUILTIN_MODULES` in
[src/complexity/registry.py](src/complexity/registry.py). The UI and CLI pick it up.

A **system** ([systems/base.py](src/complexity/systems/base.py)) implements
`reset(pattern)`, `step()` and `frame()` — a float32 `(H, W)` array in `[0, 1]`. Systems
whose state is not a grid (particles, multi-channel) render themselves to a frame.

```python
@register_system
class MySystem(System):
    key, name, state_type = "mine", "My system", "continuous"
    patterns = {"default": "Default"}
    def reset(self, pattern): ...
    def step(self): ...
    def frame(self): ...
```

A **measure** ([measures/base.py](src/complexity/measures/base.py)) implements
`compute(frames)`, receiving the last `window` frames, oldest first, and returns a value
where higher means more complex (see Conventions). Override
`accepts(system)` to restrict it (for example, a discrete-only measure returns
`system.state_type == "discrete"`). Tag it with `granularity`, `scope` and `temporal`
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
  session.py             Run: one system + its measures (shared by UI and CLI)
  cli.py                 complexity serve | list | run
  systems/               base.py, life.py, lenia.py
  measures/              base.py, compression.py
  app/server.py          FastAPI + WebSocket streaming
  app/static/            browser UI (no build step, no JS dependencies)
```

Computation runs in Python on the server; the browser only draws. The same app can be
deployed as a hosted demo (for example on a Hugging Face Space) without changes.

Lenia patterns come from Lenium, the JAX Lenia library (not yet published). Until then,
the code and data needed here are copied into this repo with their sources noted;
afterwards they will be replaced by `import lenium`.
