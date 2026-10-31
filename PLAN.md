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
Patterns: the 16 catalogue patterns below, acorn, random soup. (Pulsar was removed from the
list on 2026-10-31; Galaxy and Star gate cover oscillators.)

**Catalogue patterns (M2, done).** The last 16 entries of the Lenia catalogue
(`animals.json` [598:614]) are Life patterns, stored in Lenia as `kn = 4` rules. Their
cells are standard Life RLE, decoded by the vendored Lenia RLE decoder (§4.2):

| Pattern | Kind | Cells |
|---|---|---|
| Glider | diagonal c/4 | 3×3 |
| Lightweight / middleweight / heavyweight spaceship | orthogonal c/2; stored one cell short, they grow into the ship in a few steps | 4×5, 4×6, 4×7 |
| Glider gun (Gosper) | gun | 9×36 |
| R-pentomino | methuselah, 1103 steps | 3×3 |
| Copperhead | orthogonal c/10 | 13×11 |
| Galaxy | oscillator, period 8 | 9×9 |
| Weekender | orthogonal 2c/7 | 11×16 |
| Lobster | diagonal c/7 | 26×26 |
| Sir Robin | oblique (2,1)c/6 | 79×31 |
| Pufferfish | puffer c/2 | 12×15 |
| Switch engine | puffer c/12 | 4×16 |
| Max | spacefiller | 27×27 |
| Sidecar gun | gun | 142×213 |
| Star gate | oscillator, period 60 | 126×155 |

Glider and Gosper gun are cell-for-cell the earlier ones; the R-pentomino is the earlier
one rotated or mirrored. They keep their keys; the glider is the default Life pattern.
Patterns are not restricted by grid size: one larger than the grid is cropped to its
central part (so Sir Robin needs 128² and Sidecar gun and Star gate 256² to be whole). On a
torus, puffers and spacefillers wrap around and eventually collide with their own output;
that is expected.

### 4.2 Lenia — `lenia` and `integer_lenia`
Two systems sharing one model; the pattern's quantization level `P` selects the variant
(`P = 0` → continuous, `n_states = None`; `P > 0` → `n_states = P + 1`). `lenia` holds the
continuous patterns (Orbium, its random soup, the catalogue creatures); `integer_lenia`
holds the minimum Orbium and its random soup. They were one system until 2026-10-30.

**Base library: the hand-coded Lenium** (`~/github/lenium.old`, commit `9acb864`, identical
to `~/github/lenia-ref/code/lenium`). `~/github/lenium` is a newer, agent-written rewrite
and is not used. **Gaussian kernel shells and growth are the standard** for Lenia
experiments here, unless an experiment says otherwise: kernel rings
`bell(D mod 1, 0.5, 0.15)`, growth `2·bell(U, m, s) − 1`, with
`bell(x, m, s) = exp(-((x − m)/s)² / 2)`.

- **Integer Lenia's default pattern: minimum Orbium** (`min_orbium`): `R = 12, T = 10,
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
- Other patterns: random soup at `P = 12` (`integer_lenia:random`), and the continuous Orbium (`P = 0`) of the
  hand-coded Lenium (`patterns["orbium"]`, Gaussian, `R = 13`), with its continuous random
  soup (`lenia:random`, formerly `random_continuous`). Continuous Lenia is ported from the hand-coded `Lenia` class.
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

**Creature patterns (M2, done).** Ten continuous creatures from the original Lenia catalogue,
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
| Amoebium dentalis | 3Ad | 27 | 1, 5/6, 1/3 | 0.28 | 0.031 | 70×67 | genus pick; 256² to be whole and roomy |

All have `T = 10`, `kn = 1`, `gn = 1`. What adding them takes:
- **Kernel core.** `kn = 1` is the original *exponential bump* core,
  `exp(4 − 1/(r(1 − r)))`. They run on the standard Gaussian shell `bell(r, 0.5, 0.15)`
  instead, with no parameter changes: in a 1000-step test (2026-10-30, 128², Amoebium at
  256²) all ten keep their shape and motion, with mass within 0.80–1.05 of the start, as
  on the bump core. So no kernel-core parameter is needed. `gn = 1` is the Gaussian growth
  already in use; ring weights `b` (fractions) and the radius convention (`D = dist / R`,
  ring `⌊D · len(b)⌋`) already match.
- **Vendored** in `src/complexity/vendor/lenia/`: the 26 catalogue entries verbatim
  (`animals.json`, with their original indices) and `Board.rle2arr` / `ch2val` from
  `LeniaND.py`, reduced to 2-D. The decoder vendored in `1588585` came from the agent
  rewrite and is not reused.
- **Grid size.** No minimum: every grid size is allowed, and a pattern larger than the grid
  is cropped to its central part (`place_center`). Whether a cropped or cramped creature
  survives is part of what one can explore. A per-pattern minimum size was tried and
  dropped (2026-10-30).
- **Tests.** Each creature survives 500 steps at 128² (Amoebium 256²): mass within ±20 %
  and under 20 % of the grid above 0.1. Over 2000 steps all ten stay within 0.76–1.08 of
  their start mass. Tetralapillium metamorpha is metastable: at 128² it has
  died between steps 1000 and 2000 in some runs, and at ×2 / ×3 scale it settles into an
  oscillating ring.

### 4.2b Multi-channel Lenia — `lenia_multi` (2026-10-31)

Ported from the hand-coded Lenium's multi-channel `Lenia` (`vendor/lenium/model.py`:
`multi_kernel_fft`, `multi_params`, `multi_step`), with Lenium's `aquarium` (Tessellatium
gyrans: 3 channels, 15 kernels, R = 12, T = 2) vendored verbatim. **Kernels are Gaussian**
rings `bell(D mod 1, 0.5, 0.15) · b` out to R · r, as in Lenium, Chan's Lenia tutorial and
Leniabreeder: the Aquarium was found on Gaussian kernels. Parity: identical kernel FFT,
2.4e-7 after one step on CPU, then chaotic divergence (as for single-channel Lenia).
Frames are the mean of the channels; the UI shows the channels as RGB (`System.render`).

Two parameter sets, a hair apart. Lenium keeps two kernels' original values as comments;
restored, they are Leniabreeder's VT049W "Aquarium (self-replicating)", with Lenium's cells.
Kernel 13 there has rings [1/6, 1, 0] to r = 0.82, i.e. [1/6, 1] to 0.547 against Lenium's
0.55, and kernel 3 an inner ring of 0 against 1/12. Run here (256²):

| Pattern | Rule | Behaviour |
|---|---|---|
| `aquarium` | Lenium | 1 unit self-replicates: 3 by step 100, 24 by 1000, 60 by 2000, ~77 by 8000 |
| `aquarium_unit` | VT049W | a lone glider that never divides (mass settles at 2.3× the stored cells) |
| `aquarium_scene` | VT049W | 8 units divide when they collide: 24 by step 500, ~100 by 4000 |

Both end as networks of ~70–110 separate units over 30–38 % of the grid, rotating rings
and moving C-shapes; neither becomes goo. Uniform random soups die out under both rules
within 200 steps. So the replication switch sits within ~0.5 % of one kernel's radius:
the goo search (M3b) should look around these two parameter sets.

### 4.3 Baselines — `baseline`
Reference "systems" that calibrate every measure. Expected values are tested (docs/measures.md §6).
- `empty` — all zeros.
- `frozen_noise` — one random frame repeated (static randomness).
- `iid_noise` — a new random frame every step (dynamic randomness).
- The noise kind matches the system being compared: binary, 13-level or continuous. It is
  part of the pattern key (`iid_noise_binary`, `frozen_noise_13`, …) rather than a system
  parameter, so the UI and CLI pick it like any pattern (done, M3).

### 4.4 Later systems (backlog)
Elementary CA (rule space, for LL / WZ / TL), larger-than-life and multi-channel Lenia,
Boids, Particle Life, Particle Lenia, Swarm Chemistry. Particle systems render to a
density frame for frame-based measures and also expose positions/velocities for
particle-native measures.

## 5. Measures

The definitions, explanations, categorization and implementation decisions of all
measures — implemented, deferred and proposed — with first values and the design Q&A, are
in **[docs/measures.md](docs/measures.md)**. In brief:

- 20 measures cover all of the PDF's 18 except TD, IIT and AT (deferred) and WZ (with
  elementary CA in M4); ITI comes in two forms, compression in five.
- Each measure peaks at randomness (🎲: compression, LZC, block entropy, FD, LE, TL, LL),
  at structure (🌿: EC, SC, TE, PID, FID, ITI, FI) or neither (logical depth's proxy).
- Frame measures give time series; 11 also give spatial maps (heatmaps), per cell or per
  tile; TL, LL and FI are per rule.
- The goal measure — a dynamic map, low for empty and noise, highest for networks of
  self-contained units above goo — is not among them; multi-scale structural complexity is
  the candidate (M3b, docs/measures.md §8).

## 6. Validation

- **System goldens.**
  - GoL: glider translates (1, 1) every 4 steps; Gosper gun emits a
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
- **Measure expectations** on the noise baselines (orientation and structure tests, and
  per-measure tests): docs/measures.md §6.
- **Device independence.** CI runs on CPU JAX. GPU runs are checked locally to match CPU
  within tolerance.

## 7. Milestones

### M0 — NumPy demo ✅
- [x] Package layout, `System` / `Measure` base classes, registry
- [x] Game of Life with 6 patterns; Lenia (continuous) with Orbium and random soup
- [x] PNG, LZMA, video-LZMA, video-H.264 measures (bits/cell)
- [x] FastAPI + WebSocket server, browser UI (≤ 4 runs, now 6, live table and charts, light/dark)
- [x] `complexity serve | list | run` CLI with CSV output
- [x] Tests: GoL goldens, Orbium survival, compression sanity
- [x] Composite compression measure `novelty` (best-of `image_min`/`video_min` dropped); per-frame cache;
      grid-size comparison (`scripts/compression_by_grid_size.py`, findings in docs/measures.md §5.1)

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
- [x] Port the 2-D RLE decoder from `LeniaND.py`, with source notes
- [x] The ten creatures in the vendored patterns
- [x] Patterns larger than the grid are cropped to their center (no minimum grid size)
- [x] The 16 Life patterns from the end of `animals.json` (§4.1)
- [x] Survival tests for each creature; Life goldens for the new
      spaceships (displacement per period) and oscillators (period)

### M3 — Measures (docs/measures.md) ✅
- [x] `Measure` gains `backend`, `needs_discrete`, `ctx`, `peaks_at` and tunable `params`;
      orientation documented in the base class; `RuleMeasure` interface for rule-level
      measures
- [x] Baseline system (`empty`, `frozen_noise`, `iid_noise`, with noise levels)
- [x] Transfer entropy in JAX (`offset`, `n_levels`, `window` parameters; Miller–Madow
      bias correction)
- [x] LZC, block entropy (SE), fractal dimension (FD), effective complexity proxy (EC),
      logical depth proxy (LD), Lyapunov exponent (LE); systems gain `perturb` for LE
      (docs/measures.md §5.2, §5.6)
- [x] Rule-level: transient length (TL), Langton's λ (LL), functional information (FI)
      (docs/measures.md §5.7)
- [x] Statistical complexity (SC), PID, FID (docs/measures.md §5.3)
- [x] Individuality (ITI) in both forms, `iti_features` and `iti_local` (docs/measures.md §5.4)
- [ ] ITI features: target creature and environment of several creatures, once the
      Aquarium exists (M3b)
- [x] Heatmaps (docs/measures.md §4): a spatial map for every measure with spatial granularity, local
      fields (TE, SC, block entropy, Lyapunov separation) or per tile (compression, LZC,
      PID, FID); shown overlaid on the frame or side by side
- [x] Overlay or side by side: side by side (Bert, 2026-10-30)
- [x] Measure-expectation tests (docs/measures.md §6) for every measure (orientation and structure tests
      over all registered measures, plus per-measure tests); `warming_up` flag for window
      measures, sent per tick and drawn faint in the UI
- [x] UI: measures grouped (by kind, with taxonomy tags and popups); per-measure settings
      behind a ⚙ (typed and checked by the server, kept per browser, applied at the next
      Start); compatibility hints name the skipped systems and what the measure needs

### M3b — Multi-scale structural complexity (docs/measures.md §8), after multi-channel Lenia (M4)
- [ ] Reference cases: empty and noise (low); goo-like structures (self-organizing, one
      level); single creatures (Aquarium unit); **Aquarium** networks: self-replicating
      (`aquarium`) and dividing on collision (`aquarium_scene`) (highest) — the Aquarium
      patterns are in (§4.2b), goo is not found yet
- [ ] Find goo-like structures at parameters near the Aquarium's (experiments by Bert)
- [ ] Collision experiments as the acceptance test: pairs of units and units against noise
      (Orbium, Aquarium unit and colonies in static and moving modes, Scutium, Life glider
      and still lifes, goo); record survival and identity
- [ ] Prototype the complexity field C = S × M × I (docs/measures.md §8.1b): S spatial
      structure across scales; M self-maintenance by perturbation (internal flow, repair,
      containment, tolerance), reusing the Lyapunov twin runs; I individuality
      (segmentation, tracking, ITI); each scaled between empty and matched noise
- [ ] Check it ranks the reference cases: empty, noise < frozen noise < still lifes, glider,
      Scutium < goo < Orbium, Aquarium unit and colonies
- [ ] Hot regions: threshold + connected components in space and time (position, extent,
      peak, integral); boxes in the UI; export
- [ ] Run summaries (mean, max, 95th percentile of C) to compare simulation conditions,
      batched over seeds and parameters (with the M5 experiment runner)
- [ ] Compare with causal emergence / ΦID (Rosas, Mediano et al. 2020) and O-information
      on per-unit macro variables

### M4 — More dynamical complex systems (§4.4, PDF §3)
- [ ] Elementary CA (rule space) + Wuensche's Z; thermodynamic depth on tiny lattices
- [x] Multi-channel Lenia with the Aquarium (§4.2b)
- [ ] von Neumann / outer-totalistic CA; larger-than-life
- [ ] Particle systems: Boids, Particle Life, Particle Lenia, Swarm Chemistry (+ density
      rendering, positions for particle-native measures)

### M5 — Polishing and sharing
- [x] UI: device (CPU, or GPU and its name) in the header; collapsible controls panel (a top bar since M3)
- [x] Without a usable GPU, a CUDA build of JAX falls back to the CPU quietly: the CUDA
      driver is asked first (`complexity/device.py`) and `JAX_PLATFORMS=cpu` set if it has
      no device, instead of the plugin's traceback
- [x] UI starts with Paraptera arcus labens, Paraptera cavus furiosus, Tetralapillium
      metamorpha and continuous iid noise
- [ ] UI: "compiling…" state on start; export the session as CSV; record to MP4
      (baselines are selectable as systems and warm-up points are dimmed since M3)
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
- 2026-10-21 — Added best-of and novelty compression measures; grid-size findings in
  docs/measures.md §5.1.
- 2026-10-29 — Lenia base switched to the hand-coded Lenium (`lenium.old`); Gaussian
  Orbium replaces the polynomial one; compression findings rerun.
- 2026-10-29 — M1 done: JAX engine (`rollout` / `rollout_batch` / `advance`), functional
  systems, Integer Lenia with `min_orbium` as default (bit-identical to the predecessor;
  exact FFT potential), vendored Lenium, `discretize`, device frame buffer, benchmarks.
- 2026-10-30 — Reprioritized: M2 Lenia creatures (ten from the original catalogue) and its
  16 Life patterns, M3
  measures (almost all of the PDF's; IIT, TD, AT deferred), M4 more systems, M5 polishing
  and sharing.
- 2026-10-30 — M2 done: ten Lenia creatures (run on the Gaussian kernel) and 16 Life
  patterns from the original Lenia catalogue, center cropping for oversized patterns,
  tests. Integer Lenia split from `lenia` into its own system, `integer_lenia`. Lenia is
  now the first system; the UI starts with Lenia Orbium and Life glider.
- 2026-10-30 — M3 done: noise baselines and 20 measures (all of the PDF's except TD, IIT,
  AT; ITI in two forms), heatmaps for the 11 with spatial granularity, rule-level measures
  cached per rule, per-measure settings in the UI. Open: ITI features on the Aquarium
  (M3b), multi-scale structural complexity (M3b, after multi-channel Lenia in M4).
- 2026-10-31 — Measure definitions, decisions, values and design Q&A moved from §5 to
  docs/measures.md.
