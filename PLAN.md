# Complexity Lab — System Design and Implementation Plan

Living document. Tick boxes as work lands; add a line under **Log** at the end of each
milestone. Background on measures and systems: [assets/Complexity_Measure.pdf](assets/Complexity_Measure.pdf).

## 1. Goals

1. Apply complexity measures from the literature to dynamical complex systems — cellular
   automata, continuous CA, particle systems — in real time and in batch.
2. Compare measures across systems and patterns, to map which measure suits which kind of
   system (static vs. dynamic, local vs. global, discrete vs. continuous).
3. Be easy for other researchers to run, read and extend: one class per system or
   measure, no build step, results reproducible from the repo.

Non-goals for now: 3-D systems, training/evolving systems, a JavaScript reimplementation
of any measure.

## 2. Conventions

These hold for every system and measure; tests enforce them.

**Frames.** A system is observed as a frame: a `float32` array `(H, W)` with values in
`[0, 1]`. Systems whose state is not a grid (particles, multi-channel) render themselves to
a frame. Measures see frames only, never system internals.

**Discreteness.** Every system declares `n_states`: `2` for Game of Life, `P + 1` for
Integer Lenia, `None` for continuous states. A discrete frame takes exactly the values
`k / (n_states - 1)`. Measures that need symbols call `discretize(frames, n_levels)`, which
is exact for discrete systems and a uniform coarse-graining for continuous ones (the
coarse-graining level is a measure parameter, recorded with every result).

**Orientation: higher value = higher complexity.** Every measure returns a number where
larger means more complex *by that measure's definition*. A quantity that runs the other
way is transformed in `compute` — negated, or the reciprocal taken — and the transform is
stated in the measure's `description`. Examples: compression reports compressed bits per
cell (not the compression ratio); a ratio raw/compressed would be inverted to
compressed/raw. Note that "more complex" differs by measure: compression peaks at
randomness, transfer entropy at structured information flow. The noise baselines (§4.3)
make that difference visible instead of hiding it.

**Units.** Information-based measures report bits per cell (per cell per frame for windows),
so values compare across grid sizes. Other measures state their unit.

**Classification.** Every measure is tagged with the PDF's taxonomy: `granularity`
(pattern / rule / system), `scope` (local / global), `temporal` (static / dynamic).

## 3. Architecture

```
            ┌──────────────── device (JAX: GPU / TPU / CPU) ─────────────────┐
 pattern ─▶ │ System.init ─▶ state ─▶ System.step (jit) ─▶ System.observe ─▶ │─▶ frame ring buffer
            │                         lax.scan / vmap for batches            │        │
            │                       JAX measures (jit) ◀─────────────────────│────────┤
            └────────────────────────────────────────────────────────────────┘        │
                                         host measures (zlib / lzma / x264) ◀─ device_get
                                                         │
                         Run / Session ──▶ values ──┬──▶ WebSocket ─▶ browser UI
                                                    └──▶ CSV / Parquet + run metadata
```

### 3.1 Backend: JAX

- Dependency is plain `jax` (CPU wheels), so a fresh install works anywhere. GPU/TPU is
  opt-in: `uv pip install -U "jax[cuda13]"` (or `jax[cuda12]`, `jax[tpu]`); also exposed
  as the extra `complexity[cuda13]`. Nothing may assume a GPU is present.
- `complexity info` prints `jax.__version__`, `jax.default_backend()` and `jax.devices()`;
  the UI header shows the device too.
- `float32` everywhere; x64 stays off.
- System `step` functions are pure and `jax.jit`-compiled once per (system, size, params).
  Changing grid size recompiles, which is expected and shown as "compiling…" in the UI.
- Headless runs use `lax.scan` over steps; experiments over seeds or parameters use
  `jax.vmap` over a batch axis.
- Device → host transfer happens once per measured tick, and only when a host measure or
  the UI needs the frames.

### 3.2 System interface

```python
class System:
    key: str; name: str
    n_states: int | None            # see §2
    patterns: dict[str, str]        # key -> display name; first is default
    default_size: tuple[int, int]
    params: dict                    # static, hashable; part of the jit cache key

    def init(self, pattern: str, size, rng: jax.Array) -> State   # State is a pytree
    def step(self, state: State) -> State                         # pure, jit-able
    def observe(self, state: State) -> jax.Array                  # (H, W) float32 in [0, 1]
```

`complexity.engine.rollout(system, state, steps, every)` returns `(state, frames)` using
`lax.scan`; `rollout_batch` adds `vmap` over a leading batch axis.

### 3.3 Measure interface

```python
class Measure:
    key: str; name: str; unit: str; description: str
    window: int = 1                 # frames needed, oldest first
    backend: str = "host"           # "jax" (jit, runs on device) or "host" (NumPy / C libs)
    needs_discrete: bool = False    # True -> receives discretized frames
    granularity / scope / temporal  # taxonomy tags

    def accepts(self, system) -> bool
    def compute(self, frames, ctx) -> float   # ctx: n_states, n_levels, system key, params
```

JAX measures receive a device array `(window, H, W)`; host measures a NumPy array. Before
the window fills, results are flagged `warming_up` and the UI draws them dimmed instead of
plotting the start-up transient as data.

### 3.4 Runs, sessions, results

- `Run` = one system instance + its measures + a device ring buffer of the last
  `max(window)` frames. Shared by UI and CLI so both produce identical numbers.
- Every result file has a metadata sidecar: git SHA, package and JAX versions, device,
  system params, pattern, seed, grid size, measure params (including `n_levels`).
- Experiments are described by small config files (`experiments/*.toml`): systems ×
  patterns × seeds × measures × steps. Running one writes `results/<name>/` (Parquet or CSV
  + metadata + optional MP4 of each run).

### 3.5 UI

Python serves everything; the browser only draws (current design, kept). Plain HTML/JS
with no build step and no JS dependencies. The same app deploys to a Hugging Face Space as
the online demo. A static results site (GitHub Pages) is generated from `results/`.

## 4. Systems

### 4.1 Game of Life — `life`
Conway B3/S23 on a torus, `n_states = 2`. Step = sum of 8 rolled copies, in JAX.
Patterns: Gosper glider gun, glider, pulsar (period 3), R-pentomino, acorn, random soup.

### 4.2 Lenia — `lenia`
One system covering continuous and Integer Lenia via the quantization level `P`
(`P = 0` → continuous, `n_states = None`; `P > 0` → `n_states = P + 1`).

- **Default pattern: minimum Orbium** (`min_orbium`): Integer Lenia, `R = 12, T = 10,
  P = 12`, one ring (`b = [1]`), `m = 0.15, s = 0.015`, Gaussian kernel shell and growth,
  integer cells `0..12`. Its 13 exact levels make it usable by both discrete measures
  (13 symbols) and continuous ones.
  - Data: `~/github/lenium/src/lenium/io/data/default_creatures.json`, key `"min_orbium"`
    (schema v3, `variant: "integer"`). Lenium generates it from the predecessor library:
    `~/github/lenia-ref/code/lenium/src/lenium/pattern.py`, `patterns["min-orbium"]`.
- **Integer Lenia update**, ported from the predecessor's `IntegerLenia`
  (`~/github/lenia-ref/code/lenium/src/lenium/model.py`, helpers in `func.py`). Lenium itself
  has no Integer Lenia engine yet. Everything is integer:
  - Kernel: `K = round(P · bell(D mod 1, 0.5, 0.15) · [D < 1])` with `D = dist / R`
    (`bell(x, m, s) = exp(-((x − m)/s)² / 2)`), unnormalized integer weights `0..P`.
  - Potential: `U = Σ K · A`, integer, by wrap-padded direct convolution. Crop `K` to
    `(2R+1)²`, since the predecessor's world-size kernel gives the same result but is
    O(N⁴). A float FFT plus `round` is allowed only if a test proves it bit-identical to
    the direct convolution.
  - Growth lookup table indexed by integer `U`: `g[u] = round(P · (2·bell(u / (P·ΣK), m, s) − 1))`.
  - Update: `A ← clip(A + round(g[U] / T), 0, P)`, so each cell changes by at most ±1 per
    step for `min_orbium`.
  - Observation: `frame = A / P`.
  - A float approximation (rounding only the state) was prototyped here. On 64²,
    `min_orbium` survived 500 steps with stable mass, exactly 13 levels, and movement.
    The faithful port must reproduce at least that; the golden tests are set from the port.
- Other patterns: random soup at `P = 12`, and the classic continuous Orbium (`P = 0`,
  polynomial profiles, as in the current NumPy demo) kept as a comparison.
- **Lenium is not published yet**, so its code and data are copied (vendored) into
  `src/complexity/vendor/lenium/`. Each file records where it came from: repo path and
  commit. Copy only what is needed: creature JSON entries, the RLE decoder, profiles,
  Integer Lenia. When Lenium is published, replace the vendored code with `import lenium`
  (add the dependency, delete `vendor/lenium/`, and keep the golden tests as the check
  that behaviour is unchanged).

### 4.3 Baselines — `baseline`
Reference "systems" that calibrate every measure. Expected values are tested (§6).
- `empty` — all zeros.
- `frozen_noise` — one random frame repeated (static randomness).
- `iid_noise` — a new random frame every step (dynamic randomness).
- `noise_levels` parameter selects binary, 13-level or continuous noise to match the
  system being compared.

### 4.4 Later systems (backlog)
Elementary CA (rule space, for LL / WZ / TL), larger-than-life and multi-channel Lenia,
Boids, Particle Life, Particle Lenia, Swarm Chemistry. Particle systems render to a
density frame for frame-based measures and also expose positions/velocities for
particle-native measures.

## 5. Measures

### 5.1 Compression (proxy for algorithmic complexity) — existing
All report compressed bits per cell; higher = less compressible = more complex. Frames
are quantized to 8-bit gray first (`round(255 · v)`).

| Key | Window | Notes |
|---|---|---|
| `png` | 1 | DEFLATE + 2-D row prediction |
| `lzma` | 1 | 1-D dictionary compression of raw bytes |
| `video_lzma` | 16 | temporal repeats compress away |
| `video_h264` | 16 | lossless x264 (qp 0); motion compensation for translating structures |

All run on the host (C libraries), so their frames are transferred from the device.

### 5.2 Transfer entropy (local information flow) — new
Chosen as the third measure because it complements compression on every axis that
matters at this stage:

- It is **local and dynamic**, where compression is global. That puts the first
  cross-category comparison from the PDF's taxonomy in the demo.
- It is **low at both extremes**: zero for empty and still-life patterns (a cell's own
  past predicts its future) and ≈ 0 for iid noise (nothing is predictable). It is high
  for propagating structure such as gliders, glider streams and moving Orbium.
  Compression instead rates noise as most complex. Comparing the two is exactly the
  "which measure suits which system" question.
- It is established for cellular automata (Lizier, Prokopenko & Zomaya 2008: local
  transfer entropy highlights gliders in ECA), is in the PDF (§2.12 TE, Schreiber 2000),
  is cheap, and runs in JAX on the device.
- It exercises the discretization adapter: exact for Game of Life and Integer Lenia,
  coarse-grained for continuous Lenia.

Definition (apparent TE, history length `k = 1`), averaged over all cells `y`, all time
steps in the window, and the 4 von Neumann directions `x = y + δ`, `|δ| = offset`:

```
T_{X→Y} = H(Y_{t+1} | Y_t) − H(Y_{t+1} | Y_t, X_t)
        = Σ p(y', y, x) · log2[ p(y' | y, x) / p(y' | y) ]
```

- Estimator: plug-in from a joint histogram of `(y', y, x)` with `n_levels³` bins
  (`jnp.bincount` with static length), computed in JAX. GoL: 8 bins; min Orbium: 2197 bins
  against ≈ 250 k samples per window at 128².
- Parameters: `window = 16`, `offset = 1`, `n_levels = n_states` for discrete systems
  (default 8 for continuous ones).
- Unit: bits per cell per step. Orientation: higher = more information flow = more complex.
- Known limits, to state in the UI description:
  - `k = 1` mixes information storage into the flow; a larger `k` is a later parameter.
  - Plug-in bias grows with bins / samples; check it against `iid_noise`.
  - For smooth fields (Lenia), `offset = 1` may be too short; `offset` is exposed for
    exploration (e.g. `≈ R/4`).

### 5.3 Later measures (backlog, from the PDF)
In rough order of cost/benefit: Shannon block entropy (SE), Lempel–Ziv complexity (LZC, as
in Adachi et al. 2024), active information storage, fractal dimension (FD), transient
length (TL), Lyapunov exponent (LE; perturbed twin runs, natural in JAX), statistical
complexity (SC, via `emic`), PID (`dit`), assembly index (AT), Langton's λ and Wuensche's Z
for rule-level analysis of ECA / outer-totalistic rules, ITI (Krakauer).

## 6. Validation

- **System goldens.**
  - GoL: glider translates (1, 1) every 4 steps; pulsar has period 3; Gosper gun emits a
    glider every 30 steps.
  - Lenia: `min_orbium` keeps mass within ±10 % and exactly 13 levels over 500 steps, and
    its center of mass moves. Classic Orbium (`P = 0`) keeps mass within ±20 % over 200
    steps.
- **JAX parity.** The JAX GoL is bit-identical to the current NumPy GoL for 200 steps of
  random soup. Lenia matches to `atol = 1e-5` (`P = 0`) or exactly (`P = 12`).
- **Measure expectations on baselines** (the table is the test):

  | Measure | empty | frozen_noise | iid_noise | glider stream |
  |---|---|---|---|---|
  | compression (frame) | ≈ 0 | high | high | low |
  | compression (video) | ≈ 0 | ≈ 0 | high | low |
  | transfer entropy | 0 | 0 | ≈ 0 (< bias bound) | > 0 |

- **Orientation test.** For every registered measure, `iid_noise` scores ≥ `empty`. For
  structure-peaked measures, a declared `peaks_at = "structure"` flag adds a second test:
  glider stream > both noise and empty.
- **Device independence.** CI runs on CPU JAX. GPU runs are checked locally to match CPU
  within tolerance.

## 7. Milestones

### M0 — NumPy demo ✅
- [x] Package layout, `System` / `Measure` base classes, registry
- [x] Game of Life with 6 patterns; Lenia (continuous) with Orbium and random soup
- [x] PNG, LZMA, video-LZMA, video-H.264 measures (bits/cell)
- [x] FastAPI + WebSocket server, browser UI (≤ 4 runs, live table and charts, light/dark)
- [x] `complexity serve | list | run` CLI with CSV output
- [x] Tests: GoL goldens, Orbium survival, compression sanity

### M1 — JAX engine
- [ ] Add `jax` dependency and `cuda13` extra; `complexity info` command
- [ ] Functional `System` API (`init` / `step` / `observe`, pytree state); `engine.rollout`
      (`lax.scan`) and `rollout_batch` (`vmap`)
- [ ] Port Game of Life to JAX; parity test against the NumPy version
- [ ] Vendor from Lenium into `src/complexity/vendor/lenium/` with source notes: creature
      JSON (`min_orbium`, `orbium`), RLE decoder, profiles
- [ ] Integer Lenia in JAX, ported from the predecessor's `IntegerLenia` (integer kernel,
      direct convolution, growth lookup); continuous Lenia (`P = 0`) alongside
- [ ] `min_orbium` as the default Lenia pattern; golden tests (mass, 13 levels, motion)
- [ ] Later, once Lenium is published: replace `vendor/lenium/` with `import lenium`
- [ ] `n_states` on every system; `discretize()` helper with tests
- [ ] Device ring buffer in `Run`; one `device_get` per tick for host measures
- [ ] Remove the NumPy system implementations once parity tests pass
- [ ] Benchmark steps/s on CPU and GPU at 128², 256², 512² and record it in the README

### M2 — Measure conventions and transfer entropy
- [ ] `Measure` gains `backend`, `needs_discrete`, `ctx`; orientation documented in the
      base class
- [ ] Baseline system (`empty`, `frozen_noise`, `iid_noise`, with noise levels)
- [ ] Transfer entropy measure in JAX (`offset`, `n_levels`, `window` parameters)
- [ ] Measure-expectation tests from §6 for all five measures
- [ ] `warming_up` flag for window measures

### M3 — UI update
- [ ] Device and JAX backend shown in the header; "compiling…" state on start
- [ ] Baselines selectable as runs; TE in the measure list with its parameters
- [ ] Per-measure parameter controls (window, `n_levels`, `offset`)
- [ ] Dim warm-up points; export the session as CSV; record the session to MP4
- [ ] Compatibility hints: why a measure is greyed out for a system

### M4 — Experiments and results
- [ ] `experiments/*.toml` config format and `complexity experiment <config>`
- [ ] Batched seeds via `vmap`; results in `results/<name>/` with metadata sidecar
- [ ] First study: the 5 measures × (GoL patterns, Lenia patterns, baselines) × 10 seeds
- [ ] Measure × system compatibility matrix (applicable / informative / misleading) with
      short notes, kept in `docs/compatibility.md`
- [ ] Static results site generated from `results/` (GitHub Pages)

### M5 — Sharing
- [ ] `CONTRIBUTING.md`: adding a system / measure, the conventions in §2, the test checklist
- [ ] GitHub Actions: ruff + pytest on CPU JAX
- [ ] License; citation file
- [ ] Hugging Face Space deployment of the UI
- [ ] Publish the repo; announce to collaborators

### M6+ — Growth (one item per PR, each with tests and a compatibility-matrix entry)
- [ ] Shannon block entropy
- [ ] Lempel–Ziv complexity
- [ ] Lyapunov exponent (twin runs)
- [ ] Transient length
- [ ] Elementary CA + Langton's λ + Wuensche's Z
- [ ] Particle Life, Boids, Particle Lenia (+ density rendering)
- [ ] Further PDF measures (§5.3)

## 8. Risks and open questions

- **GPU availability varies.** The laptop's NVIDIA dGPU is disabled on battery (energy
  profile), so `jax[cuda13]` falls back to CPU then. Develop and test on CPU JAX; benchmark
  GPU only when plugged in. `complexity info` shows which device is active.
- **Host-measure bottleneck.** Compression runs on the CPU regardless of device, so at
  large grids the measures dominate, not the simulation. Mitigations: measure every *k*
  steps, a process pool for host measures, and JAX-native measures where possible.
- **TE parameters for continuous fields.** `offset` and `n_levels` change TE values for
  Lenia; record them with every result and treat their choice as part of the study.
- **x264 availability.** PyAV wheels bundle libx264 on Linux; check macOS and Windows
  wheels in CI, and fall back to `video_lzma` if missing.
- **Lenium alignment.** The vendored code may drift from Lenium as it evolves. When Lenium
  is published (and ships Integer Lenia), switch to `import lenium` and cross-check the
  `min_orbium` trajectory against the golden tests.

## Log
- 2026-10-21 — M0 done: NumPy demo with GoL, Lenia, 4 compression measures, UI and CLI.
