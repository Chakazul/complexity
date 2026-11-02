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
with no build step and no bundled JS dependencies; the measure info popup renders its
equation with KaTeX from a CDN when online and shows the LaTeX otherwise. The same app deploys to a Hugging Face Space as
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

**Creature patterns (M2; revised 2026-11-01).** Thirteen continuous creatures from the
original Lenia catalogue, `~/github/lenia-ref/code/Lenia/Python/animals.json` (614 entries;
the Python `LeniaND.py` is the reference implementation), chosen by Bert. Where several
entries share a name, the index picks one (the 3rd Helicium serratus turbulentus, the 2nd
Asterium vagus and Amoebium foliatus); names are shown without a number. With the Lenium
Orbium (Gaussian) they make 14 creature patterns.

| Pattern | Index | Code | R | b | m | s | Cells | Notes |
|---|---|---|---|---|---|---|---|---|
| Gyrorbium gyrans | 11 | OG2g | 13 | 1 | 0.156 | 0.0224 | 23×21 | gyrating |
| Triscutium valvatus laterale | 44 | S3vl | 15 | 1 | 0.36 | 0.064 | 35×43 | |
| Helicium serratus turbulentus | 173 | 4H3t | 18 | 1/4, 1/2, 3/4, 1 | 0.2 | 0.02 | 44×65 | 3rd of that name |
| Hydrogeminium natans | 233 | 3GH2n | 18 | 1/2, 1, 2/3 | 0.26 | 0.036 | 51×55 | |
| Paraptera arcus pedes | 91 | P4ap | 13 | 1 | 0.362 | 0.0665 | 43×44 | |
| Paraptera cavus furiosus | 96 | P4cf | 13 | 1 | 0.297 | 0.0405 | 36×39 | chaotic |
| Tetralapillium metamorpha | 498 | 2L4m | 18 | 1, 1/2 | 0.2 | 0.022 | 22×23 | chaotic; second of two entries (the first dies by step 750) |
| Florakronium arcus | 313 | K9a | 18 | 1, 1/2 | 0.28 | 0.035 | 31×40 | |
| Heptadentium scintillans | 355 | 2D7s | 18 | 1, 5/6 | 0.22 | 0.025 | 38×38 | |
| Asterium vagus | 404 | 3R5v | 27 | 1, 1/2, 1 | 0.27 | 0.033 | 42×42 | chaotic; 2nd of that name |
| Asterium metamorpha currens | 408 | 4R5mc | 36 | 1, 1/3, 2/3, 1 | 0.2 | 0.022 | 55×54 | chaotic |
| Amoebium foliatus | 587 | 5Af | 36 | 1, 1/12, 1/12, 1/12, 1/4 | 0.19 | 0.015 | 55×61 | 2nd of that name |
| Urium longus | 267 | 3U4l | 13 | 3/4, 1, 1 | 0.333 | 0.0439 | 65×61 | big |

All have `T = 10`, `kn = 1`, `gn = 1`. Replaced on 2026-11-01: Scutium solidus, Helicium
solidus, Paraptera arcus labens, Kronium solidus,
Hexadentium rotans, Amoebium dentalis.
- **Kernel core.** `kn = 1` is the original *exponential bump* core
  `exp(4 − 1/(r(1 − r)))`, and since 2026-11-01 the creatures run on it, as found: each
  catalogue entry carries `kernel_core = "bump"`,
  and patterns without it (Lenium's Orbium and minimum Orbium, the Aquarium) use the
  Gaussian shell `bell(r, 0.5, 0.15)`, the default. The model takes the core as an argument
  (`model.kernel_core`, a local addition to the vendored Lenium). Earlier they ran on the
  Gaussian shell, which the old ten survived. `gn = 1` is the Gaussian growth already in
  use, and it stays Gaussian with the bump kernel, as in `LeniaND.py` (Bert, 2026-11-01): a
  bump growth matched to its width, 2·bump(0.5 + 0.15 (u − m)/s) − 1, turned 8 of the 14
  creatures into goo within 1000 steps; ring weights `b` (fractions) and the radius convention (`D = dist / R`, ring
  `⌊D · len(b)⌋`) match `LeniaND.py`.
- **Vendored** in `src/complexity/vendor/lenia/`: the 29 catalogue entries in use verbatim
  (`animals.json`, with their original indices) and `Board.rle2arr` / `ch2val` from
  `LeniaND.py`, reduced to 2-D. The decoder vendored in `1588585` came from the agent
  rewrite and is not reused.
- **Grid size.** No minimum: every grid size is allowed, and a pattern larger than the grid
  is cropped to its central part (`place_center`). Whether a cropped or cramped creature
  survives is part of what one can explore. A per-pattern minimum size was tried and
  dropped (2026-10-30). Urium longus (65×61) needs 256² to be roomy.
- **Tests.** Each creature survives 500 steps at 128²: mass within
  ±20 % and under 20 % of the grid above 0.1. Over 2000 steps (128² and 256²) they stay
  within 0.95–1.11 of their start mass, except Tetralapillium metamorpha, which is
  metastable: at 128² it dies by step 1000 on the bump core, at 256² it lives (0.84–0.99).
  The bump core is tested against a float64 NumPy port of `LeniaND.py`'s `kernel_shell`
  and `calc_once`: Gyrorbium gyrans after 200 steps, mass equal to 1e-4, cells to 3e-4.
  Tetradecactenium (index 257) was added and removed on 2026-11-01: under its catalogue
  rule it grows to 13× its mass and a third of the grid by step 1000 (also in the port).

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
Reference "systems" that calibrate every measure. Expected values are tested (MEASURES.md §6).
- `empty` — all zeros.
- `frozen_noise` — one random frame repeated (static randomness).
- `iid_noise` — a new random frame every step (dynamic randomness).
- The noise kind matches the system being compared: binary, 13-level or continuous. It is
  part of the pattern key (`iid_noise_binary`, `frozen_noise_13`, …) rather than a system
  parameter, so the UI and CLI pick it like any pattern (done, M3).

### 4.4 Elementary CA — `eca` (2026-11-02)
Wolfram's 256 rules: binary cells on a ring, the next state a function of a cell and its
two neighbours, `n_states = 2`. The rule table (8 outputs, bit v of the rule number for
neighbourhood v = 4 left + 2 centre + right) is data in the state, so all rules share one
compiled step and can be batched over rules. Pattern keys name a rule and its start:
`rule110` (random row, 50 %, the default) or `rule110_seed` (one live cell); any rule
0–255 is accepted, nine are listed by Wolfram class (110 and 54: 4; 30, 18, 90: 3; 184,
108, 232: 2; 160: 1) and 110, 30, 90 from one cell.
- **Frame = space-time diagram**: the state holds the last `height` rows of a `width`-cell
  ring, the newest at the bottom, scrolling up one row per step. Frame measures see the
  diagram, as ECA compression and entropy usually do; frame-to-frame statistics (speed,
  change rate) mostly see the scroll. A new `System.configuration(state)` gives the
  dynamical state alone (the current row; the frame for every other system), which
  transient length and Langton's λ now compare. Lyapunov perturbs one cell of the current
  row and counts the damage in the diagram.
- `rule_table(pattern)` (Wuensche's Z; Langton's λ, exactly, from the table) and
  `transition_map(pattern, n)` (each configuration's successor on an n-cell ring:
  thermodynamic depth) are new optional `System` methods.
- Rule 90 is nilpotent on rings of 2^k cells: from a random row it dies within a width of
  steps at the UI's square sizes. Its one-cell pattern draws the Sierpinski triangle.
- Tests: rule 30 against its boolean form and its centre column (OEIS A051023), rule 90
  from one cell against Pascal's triangle mod 2, rule 184 conserves cars, the transition
  map agrees with the step.
- Rule space: `scripts/eca_rule_space.py` computes λ, Z, depth and the transient class of
  the 88 equivalence classes (MEASURES.md §5.7.4–5).

### 4.5 Later systems (backlog)
Outer-totalistic and larger-than-life CA, Boids, Particle Life, Particle Lenia, Swarm Chemistry. Particle systems render to a
density frame for frame-based measures and also expose positions/velocities for
particle-native measures.

## 5. Measures

The definitions, explanations, categorization and implementation decisions of all
measures — implemented, deferred and proposed — with first values and the design Q&A, are
in **[MEASURES.md](MEASURES.md)**. In brief:

- 22 measures cover all of the PDF's 18 except IIT and AT (deferred); ITI comes in two
  forms, compression in five. WZ and TD came with elementary CA (M4).
- Three statistics for reference, total mass, moving speed and change rate, are listed
  with them; eight measures have normalized, per-object versions (MEASURES.md §5.10).
- Each measure peaks at randomness (compression, LZC, block entropy, FD, LE, TL, LL), at
  structure (EC, SC, TE, PID, FID, ITI, FI) or neither (logical depth's proxy), shown as a
  tag; an emoji before each name says what it measures (MEASURES.md §2.1).
- Frame measures give time series; 11 also give spatial maps (heatmaps), per cell or per
  tile; TL, LL and FI are per rule.
- The goal measure — a dynamic map, low for empty and noise, highest for networks of
  self-contained units above goo — is not among them; multi-scale structural complexity is
  the candidate (M3b, MEASURES.md §8).

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
  per-measure tests): MEASURES.md §6.
- **Device independence.** CI runs on CPU JAX. GPU runs are checked locally to match CPU
  within tolerance.

## 7. Milestones

### M0 — NumPy demo ✅
- [x] Package layout, `System` / `Measure` base classes, registry
- [x] Game of Life with 6 patterns; Lenia (continuous) with Orbium and random soup
- [x] PNG, LZMA, video-LZMA, video-H.264 measures (bits/cell)
- [x] FastAPI + WebSocket server, browser UI (≤ 4 runs, now 8, live table and charts, light/dark)
- [x] `complexity serve | list | run` CLI with CSV output
- [x] Tests: GoL goldens, Orbium survival, compression sanity
- [x] Composite compression measure `novelty` (best-of `image_min`/`video_min` dropped); per-frame cache;
      grid-size comparison (`scripts/compression_by_grid_size.py`, findings in MEASURES.md §5.1)

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

### M3 — Measures (MEASURES.md) ✅
- [x] `Measure` gains `backend`, `needs_discrete`, `ctx`, `peaks_at` and tunable `params`;
      orientation documented in the base class; `RuleMeasure` interface for rule-level
      measures
- [x] Baseline system (`empty`, `frozen_noise`, `iid_noise`, with noise levels)
- [x] Transfer entropy in JAX (`offset`, `n_levels`, `window` parameters; Miller–Madow
      bias correction)
- [x] LZC, block entropy (SE), fractal dimension (FD), effective complexity proxy (EC),
      logical depth proxy (LD), Lyapunov exponent (LE); systems gain `perturb` for LE
      (MEASURES.md §5.2, §5.6)
- [x] Rule-level: transient length (TL), Langton's λ (LL), functional information (FI)
      (MEASURES.md §5.7)
- [x] Statistical complexity (SC), PID, FID (MEASURES.md §5.3)
- [x] Individuality (ITI) in both forms, `iti_features` and `iti_local` (MEASURES.md §5.4)
- [x] ITI features on scenes (2026-11-01): creatures detected in the potential K ∗ A (basins
      of its peaks, shallow saddles merged), a target tracked by overlap, environment = the
      creatures within reach R or everything else (MEASURES.md §5.4, Q18)
- [x] Heatmaps (MEASURES.md §4): a spatial map for every measure with spatial granularity, local
      fields (TE, SC, block entropy, Lyapunov separation) or per tile (compression, LZC,
      PID, FID); shown overlaid on the frame or side by side
- [x] Overlay or side by side: side by side (Bert, 2026-10-30)
- [x] Measure-expectation tests (MEASURES.md §6) for every measure (orientation and structure tests
      over all registered measures, plus per-measure tests); `warming_up` flag for window
      measures, sent per tick and drawn faint in the UI
- [x] UI: measures grouped (by kind, with taxonomy tags; clicking a tag lists all tags and
      their meanings); an ⓘ per measure opens a popup with its description, equation (KaTeX),
      facts and normalized form; per-measure settings
      behind a ⚙ (typed and checked by the server, kept per browser, applied at the next
      Start); compatibility hints name the skipped systems and what the measure needs

### M3b — Multi-scale structural complexity (MEASURES.md §8) — ⏸ paused 2026-11-02

Paused after the prototype goal measure (🧫 Autopoiesis): it meets the requirements but is
slow for scenes of many creatures and complicated (object detection, real-time
interventions); the next idea is a "physical" measure from autograd (linear response,
causal reach, local stability) with no objects or experiments, for a later session
(MEASURES.md §8.1d, Q19). Work moves to M4.

- [ ] (next idea, later) A physical goal measure: a dense field from `jax.jvp` / `jax.vjp`
      (how far disturbances spread, whether they decay and stay contained), no object
      detection, no real-time experiments
- [ ] Reference cases: empty and noise (low); goo-like structures (self-organizing, one
      level); single creatures (Aquarium unit); **Aquarium** networks: self-replicating
      (`aquarium`) and dividing on collision (`aquarium_scene`) (highest) — the Aquarium
      patterns are in (§4.2b), goo is not found yet
- [ ] Find goo-like structures at parameters near the Aquarium's (experiments by Bert).
      A first goo, found by the collision experiments: Triscutium valvatus laterale grows
      into an expanding labyrinth on any contact (MEASURES.md §8.1c)
- [x] Collision harness and first experiments (2026-11-01, MEASURES.md §8.1c): units
      settled, cropped and thrown at a copy (Δ 180/135/90°, offsets, chirality) or at noise,
      with a control; outcomes labelled from K ∗ A objects (survived, merged, divided, died,
      goo, missed); Orbium, Triscutium, Aquarium unit, Life glider (74 runs)
- [x] A unit into a colony (2026-11-01, MEASURES.md §8.1c): 6 copies settled, an intruder
      from 4 directions × 2 offsets, judged against 5 controls (the colony alone and
      perturbed); the Aquarium's intruders mostly join its replicating colony
- [ ] Collisions still to run: Aquarium static modes; check that survivors separate and
      keep moving
- [x] First prototype of C = S × M × I, named 🧫 Autopoiesis (2026-11-01, MEASURES.md
      §8.1d, `complexity.goal`, measure `autopoiesis` with a heatmap, computed in the
      background; `scripts/goal_ranking.py`): empty, noise, frozen noise, goo, glider <
      gun < Orbium < Aquarium soliton < Aquarium colony, and units ranked as by their
      collision outcomes; open: a stronger S, calibration, speed
- [ ] (planned design) The complexity field C = S × M × I (MEASURES.md §8.1b): S spatial
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

### M4 — More dynamical complex systems (§4.4–4.5, PDF §3)
- [x] Elementary CA (rule space) + Wuensche's Z; thermodynamic depth on tiny lattices
      (§4.4, MEASURES.md §5.7.4–5; 2026-11-02)
- [x] Multi-channel Lenia with the Aquarium (§4.2b)
- [ ] von Neumann / outer-totalistic CA; larger-than-life
- [ ] Particle systems: Boids, Particle Life, Particle Lenia, Swarm Chemistry (+ density
      rendering, positions for particle-native measures)

### M5 — Polishing and sharing
- [x] UI: device (CPU, or GPU and its name) in the header; collapsible controls panel (a top bar since M3)
- [x] Without a usable GPU, a CUDA build of JAX falls back to the CPU quietly: the CUDA
      driver is asked first (`complexity/device.py`) and `JAX_PLATFORMS=cpu` set if it has
      no device, instead of the plugin's traceback
- [x] UI starts with Orbium, Paraptera arcus pedes, Paraptera cavus furiosus,
      Tetralapillium metamorpha, Florakronium arcus and continuous iid noise, and with
      image and video LZMA, temporal novelty, SC, TE, individuality (features) and the
      Lyapunov exponent ticked (2026-11-02); with no measure selected it runs the
      simulations only
- [x] UI presets (2026-11-02): a Preset dropdown under Heatmap for Lenia, Integer Lenia,
      multi-channel Lenia, Game of Life and elementary CA selects a few patterns, noise of
      their kind, the informative measures and a grid size (README)
- [ ] UI: "compiling…" state on start; export the session as CSV; record to MP4
      (baselines are selectable as systems and warm-up points are dimmed since M3)
- [ ] `experiments/*.toml` config format and `complexity experiment <config>`; batched
      seeds via `vmap`; results in `results/<name>/` with metadata sidecar
- [ ] First study: all measures × (GoL, Lenia creatures, baselines, new systems) × 10 seeds
- [ ] Measure × system compatibility matrix (applicable / informative / misleading) with
      short notes, kept in `COMPATIBILITY.md`
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
  MEASURES.md §5.1.
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
  MEASURES.md.
- 2026-11-01 — Measures checked against the papers in `assets/` (MEASURES.md §2.4);
  transient length and functional information replaced by the papers' measures (growth
  class over grid sizes; random patterns of the creature's size with a graded function).
- 2026-11-01 — Lenia creatures revised (13 from the catalogue, PLAN §4.2); catalogue
  creatures run on the bump kernel core they were found with, others on the Gaussian.
- 2026-11-01 — M3b: ITI on scenes (creatures detected in the potential K ∗ A, tracked per
  run), collision experiments (pairs, noise, colonies; Synorbium; Triscutium's goo), and
  the goal-measure prototype C = S × M × I as 🧫 Autopoiesis (batched twin runs).
- 2026-11-02 — M3b paused: Autopoiesis is correct but slow for many creatures and
  complicated; a physical, autograd-based measure is the next idea (later). Next: M4.
- 2026-11-02 — M4: elementary CA (`eca`, frame = scrolling space-time diagram,
  `System.configuration` for state comparisons), Wuensche's Z and thermodynamic depth
  (exact on 12–17-cell rings) as rule measures, λ exact from rule tables; transient
  classes agree with Hudcová & Mikolov on 77 of 88 ECA after requiring polynomial to be
  superlinear (MEASURES.md §5.7).
