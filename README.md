# Complexity Lab

An experimentation platform for applying complexity measures from the literature to
dynamical complex systems — cellular automata, continuous CA, particle systems — and for
comparing which measures suit which kinds of system. See
[assets/Complexity_Measure.pdf](assets/Complexity_Measure.pdf) for the measures and
systems in scope.

Current demo: **Game of Life** and **Lenia** (continuous and Integer Lenia), simulated in
**JAX** (CPU, GPU or TPU), measured by **image and video compression** as proxies for
algorithmic (Kolmogorov) complexity, and by Lempel–Ziv complexity, block entropy, fractal
dimension, effective complexity, logical depth, the Lyapunov exponent and transfer
entropy, with **noise baselines** to calibrate each measure.

## Status and roadmap

The design and the milestone checklist are in **[PLAN.md](PLAN.md)**; the measures are
described in **[docs/measures.md](docs/measures.md)**. In brief:

- **M0 (done):** NumPy demo — the systems and measures below, interactive UI, CLI.
- **M1 (done):** JAX engine for GPU/TPU (CPU JAX supported), Integer Lenia with the
  *minimum Orbium*.
- **M2 (done):** ten creatures from the original Lenia catalogue (Gyrorbium, Scutium,
  Helicium, Geminium, Paraptera, Tetralapillium, Kronium, Hexadentium, Amoebium), and
  16 Game of Life patterns from the same catalogue.
- **M3 (done):** most measures from the PDF. Done: noise baselines, transfer
  entropy, LZC, block entropy, fractal dimension, effective complexity, logical depth,
  Lyapunov exponent, statistical complexity, PID, FID, and the rule-level transient
  length, Langton's λ and functional information, individuality (ITI) in two forms;
  heatmaps and per-measure settings in the UI.
- **M3b:** multi-scale structural complexity, with multi-channel Lenia's Aquarium as the
  reference case.
- **M4:** more systems: elementary CA, other CA rules, particle systems.
- **M5:** polishing and sharing: UI, batch experiments and a results site, CI,
  contributor guide, hosted demo.

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

In the UI, click up to six systems (system · pattern) in the Systems list to compare side
by side, each outlined in its series colour; click measures in the Measures list; and press
**Start**. Click an item again to deselect it, or **Clear** to deselect a whole list.
**Heatmap** (Run section) shows where one measure's value comes from, as a map per run:
local fields for transfer entropy, statistical complexity, block entropy and the Lyapunov
twin's divergence, per tile for compression, LZC, PID and FID, drawn next to each frame
with a scale shared across runs.

A ⚙ beside a measure opens its settings (window, levels, offset, component, …); changes
apply at the next Start and are remembered in the browser. Hover a greyed-out measure to
see which systems it skips and why.

Each measure shows its current value per run in the table
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
| `lenia` — Lenia (continuous) | Orbium (default), random soup, Gyrorbium gyrans, Scutium solidus, Helicium solidus, Hydrogeminium natans, Paraptera arcus labens, Paraptera cavus furiosus, Tetralapillium metamorpha, Kronium solidus, Hexadentium rotans, Amoebium dentalis |
| `integer_lenia` — Integer Lenia (`P = 12`, 13 states) | minimum Orbium (default), random soup |
| `life` — Game of Life (discrete) | glider (default), Gosper glider gun, lightweight / middleweight / heavyweight spaceships, Copperhead, Galaxy, Weekender, Lobster, Sir Robin, Pufferfish, Switch engine, Max, Sidecar gun, Star gate, R-pentomino, acorn, random soup |
| `lenia_multi` — Multi-channel Lenia (3 channels, 15 kernels) | Aquarium (self-replicating), Aquarium unit (VT049W, does not divide alone), Aquarium with 8 units (VT049W, divide on collision) |
| `baseline` — reference frames | empty (default); iid noise (new every step) and frozen noise (one frame held), each binary, 13-level or continuous |

Multi-channel Lenia is shown in RGB (one colour per channel); measures read the mean of
the channels. Its Aquarium patterns grow into networks of ~100 separate units on 256².

The baselines calibrate the measures: compare a system with noise of its own kind (binary
for Life, 13 levels for Integer Lenia, continuous for Lenia).

Lenia uses Gaussian kernel shells and growth, the standard for experiments here. The
named creatures and Life patterns come from the original Lenia catalogue
(`src/complexity/vendor/lenia/`); the creatures were found with a bump kernel core but
survive on the Gaussian one. A pattern larger than the grid is cropped to its central part;
Amoebium, Sidecar gun and Star gate need 256² to be whole.

The minimum Orbium is an Integer Lenia glider: 13 cell states (0..12), integer kernel,
potential and growth, so it suits both discrete measures (13 symbols) and continuous ones.
Each system declares `n_states(pattern)`: 2 for Life, 13 for Integer Lenia, `None` for
continuous Lenia.

Simulation speed, steps/s without measures (`scripts/benchmark_steps.py`; AMD Ryzen AI 9
HX 370, 24 threads / NVIDIA RTX 5070 Ti Laptop, JAX 0.12):

| Run | CPU 128² | CPU 256² | CPU 512² | GPU 128² | GPU 256² | GPU 512² |
|---|---|---|---|---|---|---|
| `life:random` | 268,058 | 76,348 | 12,531 | 9,773 | 9,819 | 9,887 |
| `integer_lenia:random` | 20,082 | 1,343 | 861 | 6,937 | 6,968 | 7,459 |
| `lenia:random` | 20,130 | 1,370 | 921 | 6,547 | 7,177 | 7,338 |

On the GPU every step costs about the same up to 512² (the step is launch-bound); the CPU
wins on small grids. Either way the host-side compression measures, not the simulation,
set the pace in the UI ([docs/measures.md](docs/measures.md) §5.1).

## Measures

Definitions, categorization (randomness vs structural, taxonomy tags), implementation
decisions, first values and the design Q&A for every measure, including deferred and
proposed ones, are in **[docs/measures.md](docs/measures.md)**. A summary:

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

| `transfer_entropy` | transfer entropy, bits/cell/step | 16 | information a neighbour adds about a cell's next state; see below |

**Transfer entropy** (Schreiber 2000; for CA, Lizier et al. 2008) runs in JAX on the
device, on frames discretized to `n_levels` symbols (Life 2, Integer Lenia 13, continuous
Lenia 8 by default). It is averaged over the 4 neighbours at distance `offset`, all cells
and the window's steps, with history length 1. It is low at both ends: 0 for empty, still
and frozen patterns, ~0 for iid noise (the Miller–Madow bias correction removes the
plug-in bias, which for 13 levels on 128² is as large as the minimum Orbium's signal), and
highest for moving structure. Like compression of an isolated object, its per-cell value
shrinks as the grid grows.

Window measures see fewer frames until the window fills; the UI draws those warm-up values
faint.

Measures from the PDF's other categories (window 1 unless noted):

| Key | PDF | Measure | Unit | Peaks at |
|---|---|---|---|---|
| `lzc` | AC | Lempel–Ziv (1976) phrase count of the discretized frame, normalized `c · log_L(n) / n` | ~0 uniform, ~1 noise | randomness |
| `block_entropy` | SE | Shannon entropy of every cell's `block`×`block` patch (default 2), Miller–Madow corrected | bits/cell | randomness |
| `fractal_dimension` | FD | box-counting slope of the cells above `threshold` (0.1), boxes 2 … N/4 | 0 point, 1 curve, 2 filled | randomness |
| `effective_complexity` | EC | LZMA size of the regular part: blur (σ = 2), then 3 levels below / within / above the band iid noise with the same cell statistics would fill (z = 4) | bits/cell | structure |
| `logical_depth` | LD | LZMA decompression time of the frame, median of 5 (Zenil's proxy) | ns/cell | mixed |
| `lyapunov` | LE | twin run from a perturbed state: continuous, ln(separation growth) per step after an 8-step burn-in, over 64 steps; discrete, damage spreading log2(1 + cells that differ) / 64 | 1/step | randomness |

Effective complexity collapses noise and uniform frames to one level, so it is ~0 for both
and highest for large-scale structure (Lenia soups, Life soups). Logical depth is low for
trivial frames and for noise alike (LZMA stores incompressible data as is), and it is
timing-based, so compare runs measured in the same session. The Lyapunov exponent separates
the chaotic Tetralapillium metamorpha (~0.07) and Lenia soup (~0.02) from the regular
creatures (~0); each value costs 128 extra steps of the system.

**Information decompositions and statistical complexity** look at how a cell's next state
depends on its own state and its neighbours (window 16, discretized to at most 4 levels;
SC to 2):

| Key | PDF | Measure | Unit |
|---|---|---|---|
| `statistical_complexity` | SC | entropy of local causal states (Shalizi et al. 2006): 3×3 pasts grouped when a G-test cannot tell their next-5-cell-cross distributions apart | bits/cell |
| `pid` | PID | Williams–Beer partial information decomposition (I_min) of the next state from own state and a neighbour; synergy by default (`component`: redundancy, unique_self, unique_neighbour) | bits/cell |
| `fid` | FID | Bohm et al.'s functional information decomposition of the same map with every input pattern weighted equally; synergy by default (`component`: self, neighbour) | bits/cell |

All three are 0 for empty frames and iid noise. SC is 5 bits for frozen noise, which is
correct: a static random field is predictable, but only by remembering each cell's
5-cell cross. PID synergy equals transfer entropy on Life (with I_min the neighbour's
unique information is 0). FID weights input patterns equally, so a lone glider (0.086)
scores above a soup (0.038): it describes the local rule as used, not how much happens.

**Individuality** (Krakauer et al. 2020) splits what determines a creature's next state
between the creature itself and its environment: organismal A* = I(S'; S), colonial
A = I(S'; S | E), driven nC = I(S'; E | S) (`component`, A* by default). Creatures are
connected cells above a threshold, joined across the torus's edges.

| Key | Measure | Unit |
|---|---|---|
| `iti_features` | as in the paper: S = mass of the largest creature, E = mass of the rest, over a 128-frame window, less a time-shuffled surrogate; meant for scenes of several creatures (the Aquarium) | bits |
| `iti_local` | local adaptation: per creature cell, S = the cell and its neighbours in the creature, E = other creatures within radius 3; has a heatmap | bits/cell |

**Rule-level measures** describe the rule (the system with a pattern's parameters), not
the frames. Each is computed once from runs of its own, cached, and shown as a constant:

| Key | PDF | Measure | Unit |
|---|---|---|---|
| `transient_length` | TL | mean log2(1 + steps until the state repeats) over 8 random soups of the rule on 64², up to 4096 steps (continuous states compared to 1e-3) | log2 steps |
| `langton_lambda` | LL | fraction of uniformly random neighbourhoods the rule maps to a nonzero state (Life: 140/512 ≈ 0.273); discrete rules only | fraction |
| `functional_information` | FI | −log2 of the fraction of 128 nearby rules (m, s within ±50 % of the pattern's) under which the pattern lives on for 500 steps; Lenia only | bits |

They make the first tick of a run slower (about 12 s for the default four runs), once per
pattern.

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
`compute(frames, ctx)`, receiving the last `window` frames oldest first, and returns a
value where higher means more complex (see Conventions). With `backend = "host"` (default)
the frames are a list of NumPy arrays; with `backend = "jax"` a device array
`(window, H, W)`. With `needs_discrete = True` they arrive as int32 symbols
`0..ctx.n_levels - 1` (`discretize`): exact for discrete systems, a uniform coarse-graining
for continuous ones (`n_levels`, default 8). Set `peaks_at = "structure"` if the measure is
highest for structure rather than randomness; the tests then check that a glider stream
beats noise. Attributes listed in `params` are tunable (`make_measure("transfer_entropy",
offset=2)`). Override `accepts(system)` to restrict it, and tag it with `granularity`,
`scope` and `temporal` following the classification in the PDF.

```python
@register_measure
class MyMeasure(Measure):
    key, name, unit = "mine", "My measure", "bits"
    window = 1
    def compute(self, frames, ctx): ...
```

Add a test in [tests/](tests/) and run `uv run --extra dev pytest`.

## Layout

```
src/complexity/
  registry.py            system and measure registries
  engine.py              rollout (lax.scan), rollout_batch (vmap), advance
  session.py             Run: one system + its measures + a device frame buffer (UI and CLI)
  cli.py                 complexity serve | list | info | run
  systems/               base.py, life.py, lenia.py, baseline.py (JAX)
  measures/              base.py, compression.py, information.py (TE in JAX, block entropy),
                         algorithmic.py (LZC, EC, LD), geometry.py (FD), dynamics.py (LE),
                         rule.py (TL, LL, FI), causal.py (SC), decomposition.py (PID, FID),
                         individuality.py (ITI)
  vendor/lenium/         code and data copied from Lenium, with source notes
  vendor/lenia/          patterns and RLE decoder from the original Lenia, with source notes
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
the golden tests check that behaviour is unchanged. The other creatures and the Life
patterns come from the original Lenia catalogue, copied into
[src/complexity/vendor/lenia/](src/complexity/vendor/lenia/) with its RLE decoder.
