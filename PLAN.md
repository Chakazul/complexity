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

**Discreteness.** Every system declares `n_states(pattern)`: `2` for Game of Life, `P + 1`
for Integer Lenia, `None` for continuous states. It is a method because one system (Lenia)
runs both discrete and continuous patterns. A discrete frame takes exactly the values
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
    patterns: dict[str, str]        # key -> display name; first is default
    default_size: tuple[int, int]
    params: dict                    # static, hashable; part of the jit cache key

    def n_states(self, pattern) -> int | None                     # see §2
    def init(self, pattern: str, size, rng: jax.Array) -> State   # State is a pytree
    def step(self, state: State) -> State                         # pure, jit-able
    def observe(self, state: State) -> jax.Array                  # (H, W) float32 in [0, 1]
```

`complexity.engine.rollout(system, state, steps, every)` returns `(state, frames)` using
`lax.scan`; `rollout_batch` adds `vmap` over a leading batch axis; `advance` steps without
observing (traced step count, so no recompile per count). Systems compare equal by type and
`params`, so the jit cache is shared across instances. Pattern-specific parameters (Lenia's
`R, T, P, m, s`) live in the state pytree as static metadata
(`jax.tree_util.register_dataclass`), so they are part of the cache key too.

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

**Catalogue patterns (M2).** The last 16 entries of the Lenia catalogue
(`animals.json` [598:614]) are Life patterns, stored in Lenia as `kn = 4` rules. Their
cells are standard Life RLE, so the existing decoder in `life.py` reads them as-is:

| Pattern | Kind | Cells |
|---|---|---|
| Glider | diagonal c/4 | 3×3 |
| Lightweight / middleweight / heavyweight spaceship | orthogonal c/2 | 4×5, 4×6, 4×7 |
| Glider gun (Gosper) | gun | 9×36 |
| R-pentomino | methuselah, 1103 steps | 3×3 |
| Copperhead | orthogonal c/10 | 13×11 |
| Galaxy | oscillator, period 8 | 9×9 |
| Weekender | orthogonal 2c/7 | 11×16 |
| Lobster | diagonal c/7 | 26×26 |
| Sir Robin | oblique (2,1)c/6 | 77×31 |
| Pufferfish | puffer c/2 | 12×15 |
| Switch engine | puffer c/12 | 4×16 |
| Max | spacefiller | 27×27 |
| Sidecar gun | gun | 142×213 |
| Star gate | oscillator | 122×155 |

Glider, Gosper gun and R-pentomino already exist; keep the current keys and check the
cells match. Pulsar, acorn and random soup stay. Sir Robin needs ≥ 128²; Sidecar gun and
Star gate need 256² (`min_size`, as for the Lenia creatures). On a torus, puffers and
spacefillers wrap around and eventually collide with their own output; that is expected.

### 4.2 Lenia — `lenia`
One system covering continuous and Integer Lenia via the quantization level `P`
(`P = 0` → continuous, `n_states = None`; `P > 0` → `n_states = P + 1`).

**Base library: the hand-coded Lenium** (`~/github/lenium.old`, commit `9acb864`, identical
to `~/github/lenia-ref/code/lenium`). `~/github/lenium` is a newer, agent-written rewrite
and is not used. **Gaussian kernel shells and growth are the standard** for Lenia
experiments here, unless an experiment says otherwise: kernel rings
`bell(D mod 1, 0.5, 0.15)`, growth `2·bell(U, m, s) − 1`, with
`bell(x, m, s) = exp(-((x − m)/s)² / 2)`.

- **Default pattern: minimum Orbium** (`min_orbium`): Integer Lenia, `R = 12, T = 10,
  P = 12`, one ring (`b = [1]`), `m = 0.15, s = 0.015`, Gaussian kernel shell and growth,
  integer cells `0..12`. Its 13 exact levels make it usable by both discrete measures
  (13 symbols) and continuous ones.
  - Data: `~/github/lenium.old/src/lenium/pattern.py`, `patterns["min-orbium"]`.
- **Integer Lenia update**, ported from the hand-coded `IntegerLenia`
  (`~/github/lenium.old/src/lenium/model.py`, helpers in `func.py`). Everything is integer:
  - Kernel: `K = round(P · bell(D mod 1, 0.5, 0.15) · [D < 1])` with `D = dist / R`
    (`bell(x, m, s) = exp(-((x − m)/s)² / 2)`), unnormalized integer weights `0..P`.
  - Potential: `U = Σ K · A`, integer, by wrap-padded direct convolution. Crop `K` to
    `(2R+1)²`, since the predecessor's world-size kernel gives the same result but is
    O(N⁴). A float FFT plus `round` is allowed only if a test proves it bit-identical to
    the direct convolution. **Done:** the system uses the FFT path (40–150× faster on
    CPU); its error before rounding is ≤ 0.004 against the 0.5 that would change a value,
    and a test checks bit-identity on full, random and trajectory grids of several sizes.
  - Growth lookup table indexed by integer `U`: `g[u] = round(P · (2·bell(u / (P·ΣK − 1), m, s) − 1))`
    for `u < P·ΣK` (the predecessor's `linspace(0, 1, P·ΣK)`), and `g[P·ΣK] = round(P · (2·bell(1, m, s) − 1))`.
  - Update: `A ← clip(A + round(g[U] / T), 0, P)`, so each cell changes by at most ±1 per
    step for `min_orbium`.
  - Observation: `frame = A / P`.
  - A float approximation (rounding only the state) was prototyped here. On 64²,
    `min_orbium` survived 500 steps with stable mass, exactly 13 levels, and movement.
    The faithful port must reproduce at least that; the golden tests are set from the port.
- Other patterns: random soup at `P = 12`, and the continuous Orbium (`P = 0`) of the
  hand-coded Lenium (`patterns["orbium"]`, Gaussian, `R = 13`), with its continuous random
  soup (`random_continuous`). Continuous Lenia is ported from the hand-coded `Lenia` class.
  It replaced M0's polynomial Orbium (2026-10-29), so the M0 NumPy Lenia parity test was
  dropped.
- Patterns are centered at `(N − n) // 2`, as for Life; the predecessor uses
  `N // 2 − n // 2`, one cell different for odd `n`. Comparisons with the predecessor start
  both from the same grid.
- **Lenium is not published yet**, so its code and data are copied (vendored) into
  `src/complexity/vendor/lenium/`. Each file records where it came from: repo path and
  commit. Copy only what is needed: the `orbium` and `min-orbium` pattern entries, and
  Lenia / Integer Lenia as functions (`model.py`). When Lenium is published, replace the
  vendored code with `import lenium` (add the dependency, delete `vendor/lenium/`, and keep
  the golden tests as the check that behaviour is unchanged).

**Creature patterns (M2).** Ten continuous creatures from the original Lenia catalogue,
`~/github/lenia-ref/code/Lenia/Python/animals.json` (614 entries; the Python `LeniaND.py`
is the reference implementation). Where the request named only a genus, the species below
is a proposal (the genus's first catalogue entry, or its best-known one):

| Pattern | Code | R | b | m | s | Cells | Notes |
|---|---|---|---|---|---|---|---|
| Gyrorbium gyrans | OG2g | 13 | 1 | 0.156 | 0.0224 | 23×21 | genus pick (also *revolvens*) |
| Scutium solidus | S1s | 13 | 1 | 0.29 | 0.045 | 20×18 | genus pick (37 entries) |
| Helicium solidus | H3s | 13 | 1 | 0.35 | 0.06 | 29×41 | genus pick (35 entries) |
| Hydrogeminium natans | 3GH2n | 18 | 1/2, 1, 2/3 | 0.26 | 0.036 | 51×55 | no plain "Geminium"; family Geminidae, best-known member |
| Paraptera arcus labens | P4al | 13 | 1 | 0.347 | 0.057 | 40×43 | |
| Paraptera cavus furiosus | P4cf | 13 | 1 | 0.297 | 0.0405 | 36×39 | |
| Tetralapillium metamorpha | 2L4m | 18 | 1, 1/2 | 0.2 | 0.022 | 22×23 | second of two entries; the first dies by step 750 on either kernel core |
| Kronium solidus | K4s | 18 | 1, 1/3 | 0.24 | 0.029 | 23×23 | genus pick (46 entries) |
| Hexadentium rotans | 2D6r | 18 | 1, 3/4 | 0.27 | 0.039 | 35×35 | genus pick |
| Amoebium dentalis | 3Ad | 27 | 1, 5/6, 1/3 | 0.28 | 0.031 | 70×67 | genus pick; needs ≥ 256² |

All have `T = 10`, `kn = 1`, `gn = 1`. What adding them takes:
- **Kernel core.** `kn = 1` is the original *exponential bump* core,
  `exp(4 − 1/(r(1 − r)))`. They run on the standard Gaussian shell `bell(r, 0.5, 0.15)`
  instead, with no parameter changes: in a 1000-step test (2026-10-30, 128², Amoebium at
  256²) all ten keep their shape and motion, with mass within 0.80–1.05 of the start, as
  on the bump core. So no kernel-core parameter is needed. `gn = 1` is the Gaussian growth
  already in use; ring weights `b` (fractions) and the radius convention (`D = dist / R`,
  ring `⌊D · len(b)⌋`) already match.
- **RLE decoder.** Cells are RLE strings, so port `Board.rle2arr` / `ch2val` from
  `LeniaND.py` (2-D only), with its source noted. The decoder vendored in `1588585` came
  from the agent rewrite and is not reused.
- **Grid size.** A creature needs a grid of several kernel diameters; record a minimum
  size per pattern (`min_size`), and have the UI raise the grid size (or warn) when it is
  too small. Amoebium (R = 27) needs 256².
- **Tests.** Each creature survives 500 steps at its minimum size: mass stays within
  ±20 %, and it neither dies out nor fills the grid. A creature that fails gets its source
  parameters checked against `LeniaND.py` before any tuning.

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
| `novelty` | 16 | best video ÷ mean best image bits per frame: the fraction of a frame that is new (unit: ratio) |

All run on the host (C libraries), so their frames are transferred from the device.
Compressed sizes are cached per frame (and per window), so the novelty measure and
the sliding window do not recompress anything.

**Findings so far** (`scripts/compression_by_grid_size.py`, step 100, grids 128²–512²):
- LZMA beats PNG on every run and size, including Lenia random soup, which fills the
  grid. PNG's 2-D prediction does not make up for its weaker entropy coder here, so
  the best image compressor is almost always LZMA.
- Video LZMA beats H.264 everywhere. The gap narrows for continuous Lenia (soup at 512²:
  1.305 vs 1.318 bits/cell/frame). Open question: does H.264 win for larger-R (smoother)
  creatures or longer windows?
- Bits per cell are size-invariant only for space-filling patterns. For a single object
  (gun, Orbium) they shrink as the grid grows (gun LZMA image: 0.098 at 128² → 0.008 at
  512²). Compare isolated patterns at equal grid size, or use total bits.
- `novelty` is stable across sizes: gun ≈ 0.45, Orbium ≈ 0.69, chaotic soups 0.77–0.92.
- Rerun on 2026-10-29 with the Gaussian Orbium: the conclusions are unchanged.
- Cost: all measures on a random soup take ≈ 0.3 s per measured step at 128² and 4–5 s at
  512², so large grids need "measure every k steps".

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

### 5.3 The PDF's measures (M3)
Almost all of the PDF's 18 measures are planned. Non-computable ones get a stated proxy,
and three are deferred as too hard (below). Pattern-level measures fit the frame
interface (§3.3). Rule-level measures (TL, FI, LL, WZ) describe a rule, not a frame, so
they get a `RuleMeasure` interface: computed per (system, params) from batched runs or the
rule table, shown in the UI as a constant per run, and used mainly in experiments.

| PDF | Measure | Implementation | Cost |
|---|---|---|---|
| AC | Algorithmic complexity | done: compression (§5.1). Add **LZC**, Lempel–Ziv 1976 phrase count of the discretized frame, normalized (as in Adachi et al. 2024 for Lenia) | low |
| SE | Shannon entropy | block entropy of `b × b` blocks of the discretized frame, bits/cell; `b = 1` is single-cell entropy | low |
| LE | Lyapunov exponent | twin runs in JAX: perturb by ε, renormalize the separation every step, average `log` growth. Discrete systems use the damage-spreading rate (Hamming distance) instead | low |
| FD | Fractal dimension | box counting of the thresholded frame over box sizes 2…N/4, slope of the log–log fit, in JAX | low |
| EC | Effective complexity | proxy: compressed size of the coarse-grained frame (blurred, then few levels), i.e. the regularities without the noise | low |
| SC | Statistical complexity | local statistical complexity from light cones (Shalizi et al. 2006): group past light cones by their future distributions; `C` = entropy of the local causal states. Check against `emic` on 1-D series | medium |
| LD | Logical depth | proxy: decompression time of the LZMA-compressed frame (Zenil); median of repeats, since timing is noisy | low |
| TD | Thermodynamic depth | **deferred**: for a deterministic system it is the entropy over a state's possible histories, computable only for tiny lattices. Revisit with ECA in M4 | — |
| TL | Transient length | rule-level: from batched random soups, steps until the state repeats (hash for discrete states, tolerance for continuous ones); report the mean log transient | low |
| PID | Partial information decomposition | `dit`, Williams–Beer `I_min`, target `y_{t+1}`, sources `y_t` and neighbour `x_t` on discretized cells (≤ 4 levels for Lenia); report synergy, bits/cell | medium |
| FID | Functional information decomposition | `fid-tools` on the same variables as PID; check the library's maturity first | medium |
| TE | Transfer entropy | §5.2 | low |
| IIT | Integrated information | **too hard**: Φ is exponential in system size, so only ~3×3 binary patches are tractable (`pyphi`), and a patch's Φ is not the system's | — |
| ITI | Individuality | Lenia creatures: segment the individual (largest component above a threshold, tracked), `S` = its cells, `E` = the surroundings; `A* = I(S'; S)`, `A = I(S'; S \| E)`, `nC = I(S'; E \| S)` from coarse-grained features | medium–high |
| AT | Assembly index | **deferred**: the libraries target molecules and strings, and assembly index is exponential in object size. Revisit for small discrete patches | — |
| FI | Functional information | rule-level experiment: `−log2` of the fraction of random parameters (e.g. Lenia `m, s`) or seeds whose run meets a function threshold (a surviving, moving creature) | medium |
| LL | Langton's λ | rule-level, static, from the rule table: Life, outer-totalistic rules, ECA (M4) | low |
| WZ | Wuensche's Z | rule-level, static: ECA only, added with the ECA system in M4 | low |

Order within M3: conventions and baselines first, then TE, then the cheap ones (LZC, SE,
FD, EC, LD, LE), then rule-level (TL, LL, FI), then the medium ones (SC, PID, FID, ITI).

## 6. Validation

- **System goldens.**
  - GoL: glider translates (1, 1) every 4 steps; pulsar has period 3; Gosper gun emits a
    glider every 30 steps.
  - Lenia: `min_orbium` keeps mass within ±10 % and exactly 13 levels over 500 steps, and
    its center of mass moves. Classic Orbium (`P = 0`) keeps mass within ±20 % over 200
    steps. A checksum of `min_orbium` after 500 steps, and the Orbium's mass after 200 steps
    (to 1e-3), guard the switch to `import lenium`.
- **JAX parity.** The JAX GoL is bit-identical to the M0 NumPy GoL for 200 steps of
  random soup (reference step in `tests/numpy_reference.py`). Against the hand-coded Lenium,
  checked when porting (it is not a test dependency): Integer Lenia is bit-identical for
  1000 steps at 64² and 128²; continuous Lenia has an identical kernel FFT and differs by
  ~1e-7 after one step (float rounding order), growing chaotically to O(1) by step ~1000.
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
- [x] Composite compression measure `novelty` (best-of `image_min`/`video_min` dropped); per-frame cache;
      grid-size comparison (`scripts/compression_by_grid_size.py`, findings in §5.1)

### M1 — JAX engine
- [x] Add `jax` dependency and `cuda13` extra; `complexity info` command
- [x] Functional `System` API (`init` / `step` / `observe`, pytree state); `engine.rollout`
      (`lax.scan`) and `rollout_batch` (`vmap`)
- [x] Port Game of Life to JAX; parity test against the NumPy version
- [x] Vendor from the hand-coded Lenium into `src/complexity/vendor/lenium/` with source
      notes: patterns `min-orbium`, `orbium`; Lenia and Integer Lenia as functions
- [x] Integer Lenia in JAX, ported from the predecessor's `IntegerLenia` (integer kernel,
      direct convolution, growth lookup); continuous Lenia (`P = 0`) alongside
- [x] `min_orbium` as the default Lenia pattern; golden tests (mass, 13 levels, motion)
- [ ] Later, once Lenium is published: replace `vendor/lenium/` with `import lenium`
- [x] `n_states` on every system; `discretize()` helper with tests
- [x] Device ring buffer in `Run`; one `device_get` per tick for host measures
- [x] Remove the NumPy system implementations once parity tests pass (the reference steps
      moved to `tests/numpy_reference.py`)
- [x] Benchmark steps/s on CPU and GPU at 128², 256², 512² and record it in the README

Priority from here (set 2026-10-30): Lenia patterns, then measures, then more systems,
then polishing and sharing.

### M2 — Lenia creatures and Life catalogue patterns (§4.1, §4.2)
- [ ] Port the 2-D RLE decoder from `LeniaND.py`, with source notes
- [ ] The ten creatures in the vendored patterns, with `min_size`
- [ ] UI: grid size follows the selected patterns' `min_size` (raise or warn)
- [ ] The 16 Life patterns from the end of `animals.json` (§4.1), with `min_size`
- [ ] Survival tests for each creature at its minimum size; Life goldens for the new
      spaceships (displacement per period) and oscillators (period)

### M3 — Measures (§5.2, §5.3)
- [ ] `Measure` gains `backend`, `needs_discrete`, `ctx`; orientation documented in the
      base class; `RuleMeasure` interface for rule-level measures
- [ ] Baseline system (`empty`, `frozen_noise`, `iid_noise`, with noise levels)
- [ ] Transfer entropy in JAX (`offset`, `n_levels`, `window` parameters)
- [ ] LZC, block entropy (SE), fractal dimension (FD), effective complexity proxy (EC),
      logical depth proxy (LD), Lyapunov exponent (LE)
- [ ] Rule-level: transient length (TL), Langton's λ (LL), functional information (FI)
- [ ] Statistical complexity (SC), PID, FID, individuality (ITI)
- [ ] Measure-expectation tests from §6 for every measure; `warming_up` flag for window
      measures
- [ ] UI: new measures in the list, grouped by taxonomy; per-measure parameter controls
      (window, `n_levels`, `offset`); compatibility hints (why a measure is greyed out)

### M4 — More dynamical complex systems (§4.4, PDF §3)
- [ ] Elementary CA (rule space) + Wuensche's Z; thermodynamic depth on tiny lattices
- [ ] von Neumann / outer-totalistic CA; larger-than-life and multi-channel Lenia
- [ ] Particle systems: Boids, Particle Life, Particle Lenia, Swarm Chemistry (+ density
      rendering, positions for particle-native measures)

### M5 — Polishing and sharing
- [ ] UI: device and JAX backend in the header; "compiling…" state on start; baselines
      selectable as systems; dim warm-up points; export the session as CSV; record to MP4
- [ ] `experiments/*.toml` config format and `complexity experiment <config>`; batched
      seeds via `vmap`; results in `results/<name>/` with metadata sidecar
- [ ] First study: all measures × (GoL, Lenia creatures, baselines, new systems) × 10 seeds
- [ ] Measure × system compatibility matrix (applicable / informative / misleading) with
      short notes, kept in `docs/compatibility.md`
- [ ] Static results site generated from `results/` (GitHub Pages)
- [ ] `CONTRIBUTING.md`: adding a system / measure, the conventions in §2, the test checklist
- [ ] GitHub Actions: ruff + pytest on CPU JAX
- [ ] License; citation file
- [ ] Hugging Face Space deployment of the UI
- [ ] Publish the repo; announce to collaborators

## 8. Risks and open questions

- **GPU availability varies.** The laptop's NVIDIA dGPU is disabled on battery (energy
  profile), so `jax[cuda13]` falls back to CPU then. Develop and test on CPU JAX; benchmark
  GPU only when plugged in. `complexity info` shows which device is active.
- **Host-measure bottleneck.** Compression runs on the CPU regardless of device, so at
  large grids the measures dominate, not the simulation. Mitigations: measure every *k*
  steps, a process pool for host measures, and JAX-native measures where possible.
- **TE parameters for continuous fields.** `offset` and `n_levels` change TE values for
  Lenia; record them with every result and treat their choice as part of the study.
- **GPU step is launch-bound.** On the laptop GPU every system runs ~7–10 k steps/s from
  128² to 512², below the CPU on small grids (README). Batching seeds with `vmap` is the
  way to use the GPU; a single interactive run gains little.
- **x264 availability.** PyAV wheels bundle libx264 on Linux; check macOS and Windows
  wheels in CI, and fall back to `video_lzma` if missing.
- **Lenium alignment.** The vendored code may drift from Lenium as it evolves. When Lenium
  is published (and ships Integer Lenia), switch to `import lenium` and cross-check the
  `min_orbium` trajectory against the golden tests.

## Log
- 2026-10-21 — M0 done: NumPy demo with GoL, Lenia, 4 compression measures, UI and CLI.
- 2026-10-21 — Added best-of and novelty compression measures; grid-size findings in §5.1.
- 2026-10-29 — Lenia base switched to the hand-coded Lenium (`lenium.old`); Gaussian
  Orbium replaces the polynomial one; §5.1 findings rerun.
- 2026-10-29 — M1 done: JAX engine (`rollout` / `rollout_batch` / `advance`), functional
  systems, Integer Lenia with `min_orbium` as default (bit-identical to the predecessor;
  exact FFT potential), vendored Lenium, `discretize`, device frame buffer, benchmarks.
- 2026-10-30 — Reprioritized: M2 Lenia creatures (ten from the original catalogue) and its
  16 Life patterns, M3
  measures (almost all of the PDF's; IIT, TD, AT deferred), M4 more systems, M5 polishing
  and sharing.
