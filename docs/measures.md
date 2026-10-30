# Complexity measures — design notes

The definitions, explanations and categorization of the complexity measures in Complexity
Lab: the ones implemented, the ones deferred, and the ones proposed. Also the decisions
behind each implementation, first values, and the questions and answers that shaped them.
The project plan and milestones are in [PLAN.md](../PLAN.md); background on the measures
is in [assets/Complexity_Measure.pdf](../assets/Complexity_Measure.pdf) (cited as "the PDF").

Contents: 1 Purpose · 2 Conventions (2.3 reanalysis of granularity) · 3 Catalogue · 4 Granularity and heatmaps ·
5 Measures in detail · 6 Validation · 7 First values · 8 The goal measure and proposals ·
9 Questions and answers · 10 Open questions

## 1. Purpose

Complexity Lab applies measures from the literature to dynamical systems (Game of Life,
Lenia, Integer Lenia, multi-channel Lenia, noise baselines) to map which measure suits
which kind of system. This file is the reference for what each measure computes, how it is
estimated here, where it is highest, and what it can be used for.

## 2. Conventions

**Orientation: higher value = more complex, by the measure's own definition.** A quantity
that runs the other way is negated or inverted inside the measure, and its description
says so. "More complex" means different things to different measures (§2.1).

**Units.** Information measures report bits per cell (per cell per step for windows), so
values compare across grid sizes; others state their unit. Bits per cell are size-invariant
only for space-filling patterns: for a single object they shrink as the grid grows (§5.1).

**Frames and symbols.** Measures see frames: float32 (H, W) in [0, 1]. Multi-channel Lenia
gives the mean of its channels. Measures that need symbols get `discretize(frames, L)`: `L`
equal-width bins, exact for discrete systems with `L = n_states` (Life 2, Integer Lenia 13),
a coarse-graining for continuous ones (default 8; some measures cap it, e.g. ≤ 4 for PID).

**On a torus.** All local measures read neighbourhoods with wrap-around, as the
simulations do; measures that read a frame row by row see a seam (§9, Q6).

### 2.1 Where a measure peaks: randomness, structural, mixed

Every measure declares `peaks_at`, shown in the UI as a tag and an emoji:

| Kind | Emoji | Meaning | Members |
|---|---|---|---|
| randomness | 🎲 | rises from order to disorder; highest for noise | compression (all), LZC, block entropy, fractal dimension, Lyapunov exponent, transient length, Langton's λ |
| structural | 🌿 | low for both uniform frames and noise, highest in between | effective complexity, statistical complexity, transfer entropy, PID, FID, ITI (both), functional information |
| mixed | — | neither reliably | logical depth (its proxy, §5.2.3) |

This is the classic split between measures of randomness (Kolmogorov complexity is often
called algorithmic randomness) and measures of structured complexity, which are low at both
ends (Grassberger 1986; Crutchfield & Young 1989; Gell-Mann & Lloyd 1996). See §9, Q1.

### 2.2 Tags

Three taxonomy tags say what a measure **is** in the literature (from the PDF, corrected by
the reanalysis in §2.3); two more say what **this implementation computes**.

| Tag | Values | Meaning |
|---|---|---|
| granularity | pattern / rule / system | a property of a configuration / of the dynamical law / of a distribution, ensemble or family |
| scope | local / global | has a pointwise or neighbourhood form / defined on the whole configuration |
| temporal | static / dynamic | one configuration (or a rule table) / needs change over time |
| subject | instance / trajectory / rule / ensemble | what the implementation measures: one frame / a stretch of the run / the rule / an ensemble |
| estimated from | object / run / ensemble | the data it uses: that frame / this run / an ensemble of runs |
| heatmap | — | has a spatial map (§4) |

The UI shows the first three and heatmap as coloured tags (in each pair the value preferred
for the goal in §8 is a step darker: pattern, local, dynamic, structural), and the last two
in each measure's tooltip. Which measures are computed once per rule (and so are off by
default) is a property of the code (`RuleMeasure`, `per_rule` in the catalog), not of the
granularity tag: FID and the Lyapunov exponent are rule-level measures computed per frame.

### 2.3 Reanalysis of granularity (2026-10-31)

The taxonomy was re-derived from the literature, independently of the PDF and of earlier
decisions here, and compared with both.

**Refined axes.** Subject: *instance* (one configuration or object), *trajectory* (a finite
run from a given state), *process or rule* (the dynamics averaged over initial states,
usually as a long-time limit), *ensemble* (a family of configurations or rules). Space:
*global*, *regional* (per object or patch), *pointwise* (a per-cell form exists). Time:
*snapshot*, *window*, *asymptotic*.

**Main distinction.** Algorithmic measures (AC, LZ, LD, EC, AT) are properties of
**individual objects**. Information-theoretic measures (SE, TE, SC, PID, FID, ITI, IIT) are
properties of **probability distributions** — ensembles or stationary processes. Estimated
from one frame or one run, the cells and steps stand in for the ensemble, which assumes
spatial and temporal stationarity and quietly turns "the rule" into "this run". One
granularity tag cannot express that, hence the subject and estimated-from tags.

| Measure | Subject | Space | Time | Literature |
|---|---|---|---|---|
| AC (Kolmogorov, Chaitin) | instance | global; regional via BDM (Zenil) | snapshot (space-time if applied to a run) | defined for a finite object; compression is an upper bound |
| LZ76 | instance (a sequence); asymptotically the source | global, 1-D (2-D needs an arbitrary scan) | snapshot | normalized LZ estimates the entropy rate: a randomness estimate |
| SE (Shannon) | ensemble (a distribution) | global; pointwise as surprisal −log p | snapshot, or entropy rate | block-entropy convergence gives entropy density and excess entropy (Crutchfield & Feldman) |
| LE (Lyapunov) | process or rule (ergodic: almost every orbit); finite-time: a trajectory | global; local / comoving forms (Kaneko; damage spreading, Bagnoli et al.) | asymptotic; finite-time gives a series | a long-time limit of the dynamics |
| FD | instance (a set) | global; pointwise via local dimension, multifractal spectra | snapshot (space-time diagrams have their own) | |
| EC (Gell-Mann & Lloyd) | instance relative to an ensemble | global | snapshot | needs a coarse-graining; cousins: Kolmogorov structure function, sophistication |
| SC, C_μ (Crutchfield & Young) | process | global; pointwise as local SC (Shalizi et al. 2006) | asymptotic (infinite past) | causal states from an ensemble or long run, local values on one realization |
| LD (Bennett) | instance | global | snapshot (it measures computation time; grows over a run) | |
| TD (Lloyd & Pagels) | a state with its history | global | the history | depends on the process that produced the state |
| TL (Hudcová & Mikolov) | rule | global | whole trajectory | classifies rules by how transients grow with system size |
| PID (Williams & Beer) | ensemble (a joint distribution) | as the variables; pointwise variants (Finn & Lizier) | dynamic only with time-lagged variables | rule-level from a rule table, trajectory-level from a run |
| FID (Bohm et al.) | **rule** (the input–output function) | the function's inputs | one-step mapping | uniform inputs make it a property of the function |
| TE (Schreiber; Lizier) | process (a pair of stationary processes) | pointwise (local TE) | window or asymptotic | averages characterize rules, local values locate gliders |
| IIT, Φ (Tononi) | a system in a state | **whole system** (its most integrated subset) | one step | defined over partitions of the whole system |
| ITI (Krakauer et al.) | an individual vs its environment | regional | window or asymptotic | needs an identified individual |
| AT (Cronin, Walker et al.) | instance (an object); with copy number, a population | regional | snapshot (encodes a construction history) | |
| FI (Hazen, Szostak) | ensemble of configurations, for a function | global | snapshot (dynamic if the function is run) | |
| LL (Langton) | rule (table) | global over a local rule | static | |
| WZ (Wuensche) | rule (table) | global over a local rule | static | |

**Comparison with the PDF and these notes before the reanalysis.** Agreement for AC and
compression, LZC, FD, EC, TL, LL, WZ and ITI. Corrections:
- **FID is rule-level** (both said system): it is defined on the function, with uniform inputs.
- **Block entropy is local** (these notes said global, while giving it a per-cell map), and
  its granularity is the distribution's: system, as in the PDF's §1.1.
- **IIT is not local** (the PDF said local): Φ is defined over partitions of the whole system.
- **The Lyapunov exponent is rule-level** (these notes said pattern, after the finite-time
  implementation): the exponent belongs to the dynamics; the implementation's subject is a
  trajectory (now its own tag).
- **Functional information is ensemble-level** (both said rule): Hazen's FI is a fraction of
  an ensemble of configurations; here the ensemble is rules near the pattern's own (system).

Partly right, now resolved by the subject tag: SC and TE (the PDF's rule-level fits the
averages and canonical C_μ; the pattern-level local fields of Shalizi and Lizier are what is
computed here, so they keep pattern); PID (system, dynamic only through time-lagged
variables). The PDF is internally inconsistent: its §1.1 and its per-measure lines disagree
for SE ("system" vs "system / pattern") and TE ("rule" vs "rule / pattern"), and it tags LD
dynamic, mixing computation time with observed dynamics; TD's subject is a state with its
history, and AT is per object (regional), not global.

For the goal (§8), a dynamic spatial map must be the pointwise form of a measure, estimated
from a run (trajectory, window, pointwise): local TE, local SC, local information dynamics.
Their probabilities and causal states would better come from an ensemble of runs of the
rule than from one 16-frame window — a limitation of the current SC.

Caveat: from knowledge of the literature, not re-checked against the papers.

## 3. Catalogue

All 18 measures of the PDF, plus the variants added here and the proposals. Tags are
granularity, scope, temporal (§2.2); "computed" is subject / estimated from. "Map": local =
one value per cell; tile = per tile of the frame (§4).

| PDF | Measure | Key | Kind | Tags | Computed | Map | Status |
|---|---|---|---|---|---|---|---|
| AC | Algorithmic complexity, by compression | `lzma`, `png`, `video_lzma`, `video_h264`, `novelty` | 🎲 | pattern, global, static (video: dynamic) | instance / object (video: trajectory / run) | tile (not H.264, novelty) | implemented |
| AC | Lempel–Ziv complexity | `lzc` | 🎲 | pattern, global, static | instance / object | tile | implemented |
| EC | Effective complexity (proxy) | `effective_complexity` | 🌿 | pattern, global, static | instance / object | — | implemented |
| LD | Logical depth (proxy) | `logical_depth` | mixed | pattern, global, static | instance / object | — | implemented |
| SC | Statistical complexity (local) | `statistical_complexity` | 🌿 | pattern, local, dynamic | trajectory / run | local | implemented |
| TE | Transfer entropy | `transfer_entropy` | 🌿 | pattern, local, dynamic | trajectory / run | local | implemented |
| SE | Shannon (block) entropy | `block_entropy` | 🎲 | system, local, static | instance / object | local | implemented |
| PID | Partial information decomposition | `pid` | 🌿 | system, local, dynamic | trajectory / run | tile | implemented |
| FID | Functional information decomposition | `fid` | 🌿 | rule, local, dynamic | rule / run | tile | implemented |
| ITI | Individuality, as in the paper | `iti_features` | 🌿 | pattern, local, dynamic | trajectory / run | — | implemented |
| ITI | Individuality, local adaptation | `iti_local` | 🌿 | pattern, local, dynamic | trajectory / run | local | implemented |
| FD | Fractal dimension | `fractal_dimension` | 🎲 | pattern, global, static | instance / object | — | implemented |
| LE | Lyapunov exponent | `lyapunov` | 🎲 | rule, global, dynamic | trajectory / run | local | implemented |
| TL | Transient length | `transient_length` | 🎲 | rule, global, dynamic | rule / ensemble | — | implemented |
| LL | Langton's λ | `langton_lambda` | 🎲 | rule, global, static | rule / ensemble | — | implemented |
| FI | Functional information | `functional_information` | 🌿 | system, global, dynamic | ensemble / ensemble | — | implemented |
| TD | Thermodynamic depth | — | — | pattern, global, dynamic | — | — | deferred (§5.8) |
| IIT | Integrated information | — | — | system, global, dynamic | — | — | deferred (§5.8) |
| AT | Assembly index | — | — | pattern, local (regional), static | — | — | deferred (§5.8) |
| WZ | Wuensche's Z | — | — | rule, global, static | — | — | with elementary CA, M4 (§5.8) |
| — | Multi-scale structural complexity | — | 🌿 | pattern, local, dynamic | trajectory / run (better: ensemble) | local | proposed: the goal measure (§8) |
| — | Local information dynamics (storage, transfer, modification) | — | 🌿 | pattern, local, dynamic | trajectory / run | local | proposed (§8.2) |
| — | Causal emergence / integrated information decomposition (ΦID) | — | 🌿 | pattern | trajectory / run | per unit | proposed (§8.2) |
| — | O-information | — | 🌿 | system, local | trajectory / run | local | proposed (§8.2) |
| — | Pointwise PID | — | 🌿 | system, local, dynamic | trajectory / run | local | proposed (§8.2) |
| — | Logical depth, better proxies | — | 🌿? | pattern, global | instance / object | — | proposed (§5.2.3) |
| — | Assembly copy number × index | — | 🌿 | pattern | instance / object | per object | proposed (§8.2) |

Code: `src/complexity/measures/` — `compression.py`, `algorithmic.py` (LZC, EC, LD),
`causal.py` (SC), `information.py` (TE, block entropy), `decomposition.py` (PID, FID),
`individuality.py` (ITI), `geometry.py` (FD), `dynamics.py` (LE), `rule.py` (TL, LL, FI).

## 4. Granularity and heatmaps

**Granularity in space, time and rule** (asked 2026-10-30): can a measure give a spatial map
(possibly coarser than the grid), a time series, or values that differ per rule?

| Measure | Space | Time | Rule |
|---|---|---|---|
| Compression, LZC | per tile | ✓ | indirect (through the patterns) |
| Block entropy | ✓ local surprisal −log2 p(block) | ✓ | indirect |
| Fractal dimension | per tile in principle (too few boxes on small tiles) | ✓ | indirect |
| Effective complexity | per tile in principle (empty on 8×8 tiles) | ✓ | indirect |
| Logical depth | per tile in principle (timing noise) | ✓ | indirect |
| Lyapunov exponent | ✓ the twin's divergence field | ✓ finite-time | indirect |
| Transfer entropy | ✓ local TE per cell per step (Lizier) | ✓ | indirect |
| Statistical complexity | ✓ local complexity field (Shalizi) | ✓ | indirect |
| PID, FID | per tile (I_min has no per-cell form) | ✓ | system |
| ITI | `iti_local` per cell; `iti_features` per individual | ✓ | indirect |
| TL, LL, FI | — | — | ✓ |
| IIT (deferred) | tiny patches only | ✓ | indirect |
| AT (deferred) | per object | ✓ | indirect |
| TD (deferred) | — | — | ✓ |

**What gives a per-cell map** (§9, Q4): being computed from the run's frames and being an
average of a per-cell quantity, so that the map's mean is the measure. Local TE,
−log2 P(causal state) and block surprisal qualify, and their maps average to the measure's
value (tested). The `local` tag is a hint, not a guarantee: PID's I_min takes a minimum over
whole distributions. `structural` and `dynamic` are not needed for a map, but make it
useful for the goal in §8: dark on noise and empty, following what structures do.

**Heatmaps in the UI.** One chosen measure's map per run, beside its frame (`Measure.map`:
"local" or "tile"; `Run.map`):
- **Local**: transfer entropy, statistical complexity, block entropy, Lyapunov exponent,
  `iti_local`. The Lyapunov map is 0 to 1 in both kinds of system: damage (discrete) or
  log10 growth over its 5-decade range beyond the 1e-5 perturbation (continuous).
- **Tile**: LZMA, PNG, video LZMA, LZC, PID, FID. Tiles are about 1/16 of the grid, at least
  8 cells, and each shows its **excess over an empty tile**: on an 8×8 tile LZMA spends
  ~8 bits/cell on format overhead, which swamped every difference.
- Left out: effective complexity (a 2-cell blur leaves nothing on 8×8), fractal dimension
  (2 box sizes), logical depth (timing noise), H.264 and novelty (cost).
- Colour: one orange ramp for magnitude (sequential = one hue), from the surface colour at
  0; the scale is shared across runs per tick; negatives show as 0. Side by side was chosen
  over an overlay, which hid the structure it explained (Bert, 2026-10-30).

## 5. Measures in detail

### 5.1 Compression (AC: algorithmic complexity, by proxy)

**Definition.** Kolmogorov complexity K(x) = min{|p| : U(p) = x}, the shortest program
producing x, is not computable; compressed size is an upper bound. All report compressed
bits per cell (per frame for video); frames are quantized to 8-bit gray, `round(255 · v)`.

| Key | Window | What it sees |
|---|---|---|
| `png` | 1 | DEFLATE with 2-D row prediction: smooth 2-D variation |
| `lzma` | 1 | dictionary compression of raw bytes: exact repeats anywhere |
| `video_lzma` | 16 | repeats across time compress away |
| `video_h264` | 16 | lossless x264 (qp 0): motion compensation, including sub-pixel motion |
| `novelty` | 16 | best video ÷ mean best image bits per frame: the fraction of a frame that is new (ratio) |

Implementation: host-side C libraries; sizes are cached per frame (and window), so novelty
and sliding windows never recompress. Off by default in the UI: PNG and H.264 (LZMA beat
them on every run).

Findings (`scripts/compression_by_grid_size.py`, step 100, 128²–512²):
- LZMA beats PNG on every run and size, including Lenia soup; PNG's 2-D prediction does not
  make up for its weaker entropy coder.
- Video LZMA beats H.264 everywhere; the gap narrows for continuous Lenia (soup at 512²:
  1.305 vs 1.318 bits/cell/frame). Open: does H.264 win for smoother (larger-R) creatures?
- Bits per cell are size-invariant only for space-filling patterns: gun LZMA 0.098 at 128²
  → 0.008 at 512². Compare isolated patterns at equal grid size.
- `novelty` is stable across sizes: gun ≈ 0.45, Orbium ≈ 0.69, chaotic soups 0.77–0.92.
- Cost: ≈ 0.3 s per measured step at 128² and 4–5 s at 512² for all compressors on a soup.

### 5.2 Algorithmic complexity beyond compressed size

#### 5.2.1 Lempel–Ziv complexity (`lzc`, AC)

**Definition.** The number of phrases c in the exhaustive-history parse of the discretized
frame read row by row (Lempel & Ziv 1976; Kaspar & Schuster 1987): each phrase is the
shortest substring not seen before, overlap allowed. Normalized as c · log_L(n) / n: ~0 for
a uniform frame, ~1 for iid noise. Used for Lenia by Adachi et al. 2024.

**Implementation.** A direct parse rescans the history for every phrase (26 s for 512²
noise). Here, for each phrase length ℓ ≤ 48 one vectorized pass of polynomial rolling
hashes marks the positions whose ℓ-substring occurred earlier, so the phrase loop only does
lookups; longer phrases fall back to `bytes.find`. 0.6 s for 512² binary noise, 20 ms at
128²; tested against the direct parse (including Kaspar & Schuster's example, c = 6).

#### 5.2.2 Effective complexity (`effective_complexity`, EC)

**Definition.** Gell-Mann & Lloyd: the algorithmic information of the regularities of an
object, without its random part. **Proxy**: LZMA bits per cell of the frame's regular part:
Gaussian blur (σ = 2 cells), then 3 levels — below, within or above the band that iid noise
with the frame's mean and variance would fill after the blur, mean ± z · std / sqrt(4π σ²),
z = 4. Noise of any kind and uniform frames collapse to one level (~0 bits); structure that
noise cannot explain remains.

**Why the noise band.** Fixed levels after blurring failed both ways: a blurred glider gun
never reached 1/3, while blurred binary noise straddled 1/2 (noise scored above the gun).
With z = 3, binary noise still leaked through (0.27 % of cells); z = 4 fixed it.

#### 5.2.3 Logical depth (`logical_depth`, LD)

**Definition.** Bennett: the running time of the shortest (or near-shortest) program that
produces the object. Trivial and random objects are both shallow; deep objects need long
computation from a short description. **Proxy** (Zenil et al.): time to decompress the
LZMA-compressed frame, median of 5 repeats, ns per cell.

**Why "mixed"** (§9, Q2): the proxy measures LZMA's decoder work, which peaks at
intermediate entropy per byte, not at structure. Incompressible data is stored raw and
decodes as a memory copy (continuous noise: 0.19 ns/cell, below an empty grid's 1.66);
partly compressible data goes through the slow range-coding path (13-level noise: 12.5, the
highest of all). It is exempt from the orientation tests. Also machine- and run-dependent.

**Better proxies (proposed):** time divided by compressed size (time per bit of
description, closer to Bennett); a compressor without raw storage; or a non-timing estimate
from the Block Decomposition Method (Zenil) with step counts.

### 5.3 Information theory

#### 5.3.1 Statistical complexity (`statistical_complexity`, SC)

**Definition.** Crutchfield: C_μ = −Σ P(S) log2 P(S), the entropy of the causal states of a
process's minimal predictive model (ε-machine): the memory needed to predict it. **Local
form for fields** (Shalizi et al. 2006): two past light cones are equivalent if they predict
the same distribution of future light cones; each cell gets a causal state and a local
complexity −log2 P(state), whose mean is C.

**Implementation.** Past = a cell's 3×3 neighbourhood at t; future = the cell and its 4
neighbours at t+1; 2 levels by default (pasts have L⁹ values). Pasts are merged
most-frequent-first into the first causal state a **G-test** (α = 1e-3, vectorized over
states) cannot tell apart. A fixed total-variation threshold split iid noise into many
states (100-odd samples per past over 32 futures already differ by ~0.2). Window 16.

**Notes.** Frozen noise scores 5 bits, which is correct: a static random field is
perfectly predictable, but only by remembering each cell's 5-cell cross. iid noise scores
~0 (up to chance splits among 512 pasts tested at α). Not yet validated against `emic` on
1-D series. Has a local map.

#### 5.3.2 Transfer entropy (`transfer_entropy`, TE)

**Definition.** Schreiber 2000; local TE in cellular automata, Lizier et al. 2008. Apparent
TE with history length k = 1, from a neighbour x at distance `offset` to a cell y, averaged
over all cells, the window's steps and the 4 von Neumann directions:

```
T_{X→Y} = H(Y_{t+1} | Y_t) − H(Y_{t+1} | Y_t, X_t)
        = Σ p(y', y, x) · log2[ p(y' | y, x) / p(y' | y) ]
```

Local TE is the log ratio per sample; its mean is T.

**Why it was the third measure.** It complements compression on every axis: local and
dynamic where compression is global; low at both ends (0 for empty and still patterns,
~0 for iid noise) and high for propagating structure; established for CA; cheap and in JAX.

**Implementation.** Plug-in estimate from a joint histogram of (y′, y, x) with L³ bins
(`jnp.bincount`, static length), in JAX on the device. **Miller–Madow correction** on each
entropy (`correct_bias`, default on): the plug-in bias on 13-level iid noise at 128² is
0.0055 bits — as large as min Orbium's TE (0.006) — and the correction removes it exactly
when all bins are occupied; values are clipped at 0. A filling window is zero-padded to the
full shape with the padding weighted out, so the start of a run does not recompile once
per window length. Parameters: window 16, offset 1, n_levels.

**Limits.** k = 1 mixes information storage into the flow. For smooth fields (Lenia),
offset 1 may be short; try ≈ R/4. Per-cell TE of a lone object shrinks with grid size.

#### 5.3.3 Block entropy (`block_entropy`, SE)

**Definition.** Shannon entropy H = −Σ p log2 p of the discretized frame's `b × b` patches
(every cell's patch, on the torus), per cell: H(blocks) / b². `b = 1` is single-cell entropy;
larger blocks also count spatial correlations. Miller–Madow corrected.

**Notes.** With 13 levels there are more possible 2×2 blocks (28 561) than cells at 128², so
iid noise reads 3.5 of 3.7 bits/cell. Local map: surprisal −log2 p(block) / b².

#### 5.3.4 Partial information decomposition (`pid`, PID)

**Definition.** Williams & Beer 2010: I(T; Y, X) = U(Y) + U(X) + R + S — unique, redundant
and synergistic information that sources Y and X give about target T. Redundancy
I_min = Σ_t p(t) min_i I_spec(t; S_i), with specific information
I_spec(t; S) = Σ_s p(s | t) log2[p(t | s) / p(t)].

**Implementation.** Target = a cell's next state y_{t+1}; sources = its own state y_t and a
neighbour x_t at distance `offset`; ≤ 4 levels; averaged over the 4 neighbours and the
window. `component`: synergy (default), redundancy, unique_self, unique_neighbour.
Implemented directly (a few lines), not with `dit`.

**Notes.** TE = unique(neighbour) + synergy. On Life the synergy equals TE: I_min gives the
neighbour no unique information. On Lenia it is about half of TE.

#### 5.3.5 Functional information decomposition (`fid`, FID)

**Definition.** Bohm et al. 2025: I_tot(Y; X) = I_syn(X → Y) + Σ_i I_ind(Y; X_i), computed on
the input–output map with every input pattern weighted equally — a property of the
function, not of how often its inputs occur.

**Implementation.** The same variables as PID. Following `fid-tools` 0.1.2
(`TPM.from_data(...).neutral_completion_fid()`): every combination of the values each input
takes is weighted equally; an unseen combination gets a uniform output over the observed
targets; then I(T; X_i) and synergy = I(T; X) − Σ I(T; X_i), negative when the inputs are
redundant. `component`: synergy (default), self, neighbour. Tested on XOR, copy and a
completed OR.

**Notes.** Synergy does not depend on how much of the grid is active, so a lone glider
(0.086) scores above a Life soup (0.038): it describes the local rule as used.

### 5.4 Individuality (ITI)

**Definition.** Krakauer et al. 2020, information theory of individuality, for a system S
and its environment E at step n:
- organismal individuality A* = I(S_{n+1}; S_n)
- colonial individuality A = I(S_{n+1}; S_n | E_n)
- environmental determination nC = I(S_{n+1}; E_n | S_n)

`component` selects one (organismal by default). Creatures are 8-connected components above
a threshold, joined across the torus's edges (`torus_label`). Both forms were requested
(Bert, 2026-10-30).

**`iti_features` (as in the paper).** S = mass of the largest creature, E = mass of the rest,
each cut into 4 quantile levels over a 128-frame window of the run's own frames.
Miller–Madow left iid noise at 0.10–0.12 bits (64 bins, 127 samples), so the mean over 20
time-shuffled S′ is subtracted instead (noise: 0.03). For a lone creature E is empty and A*
measures how predictable its mass is. Its intended test bed is the Aquarium: a target
creature among other copies, some moving, some still (to be elaborated, M3b).

**`iti_local` (adaptation).** Per creature cell: S = (own state, summed state of its 8
neighbours in the same creature — the live-neighbour count for Life), E = summed state of
other creatures' cells within radius 3. Pointwise values give a map. Rounded neighbour
*means* erased Life's information (every member is alive), hence sums; E looks beyond the
3×3 because 8-connected creatures never touch. nC is 0 in Life by construction (a cell's
next state depends only on its 3×3, all in its own creature); in Lenia it is non-zero only
when other creatures are near.

### 5.5 Fractal dimension (`fractal_dimension`, FD)

**Definition.** Box-counting dimension of the cells above `threshold` (0.1): the slope of
log(boxes occupied) against log(1 / box size), for boxes of 2, 4, … up to a quarter of the
grid. 0 for a point, 1 for a curve, 2 for a filled frame (noise included). Tested on a
point, a line and a plane.

### 5.6 Lyapunov exponent (`lyapunov`, LE)

**Definition.** The exponential rate at which nearby trajectories separate. Here, a finite
time estimate from a twin run started at a perturbed copy of the current state
(`System.perturb`: Life flips one live cell; Integer Lenia moves one active cell one level;
continuous Lenia adds 1e-5 Gaussian noise; baselines change one cell and their noise
stream).
- Continuous: ln(separation growth) per step after an 8-step burn-in, over 64 steps; the
  larger of the rates during and after the burn-in.
- Discrete: damage spreading, log2(1 + cells that differ) / 64.

**Why the burn-in.** In Lenia a random perturbation first shrinks ~4× within 8 steps (its
stable part dies out); a 16-step estimate came out negative even for chaotic creatures. The
rate during the burn-in still counts, because iid noise decorrelates in one step.

**Notes.** Each value costs 128 extra steps. Its map is the twin's divergence field.

### 5.7 Rule-level measures

`RuleMeasure`: computed once per (measure, parameters, system, pattern) from runs of its own,
cached, and shown as a constant; off by default in the UI (seconds on the first tick).

#### 5.7.1 Transient length (`transient_length`, TL)

Hudcová & Mikolov: how long random initial states take to settle. 8 soups of the rule
(`System.soup`) on 64² run up to 4096 steps; two uint32 hashes per step (in a `lax.scan`,
`vmap`ped over the soups; continuous states quantized to 1e-3); the transient is the index
where the first repeated state first occurred. Reported as the mean of log2(1 + transient);
soups that never repeat count as 4096 (a lower bound).

#### 5.7.2 Langton's λ (`langton_lambda`, LL)

The fraction of the rule table that leads to a non-quiescent state: the probability that a
uniformly random neighbourhood produces a live cell. Estimated by stepping a 256² grid of
iid uniform states once (`System.uniform`). Life: 140/512 = 0.273 (born with 3 of 8
neighbours, 56 tables; survive with 2 or 3, 84). Discrete rules only.

#### 5.7.3 Functional information (`functional_information`, FI)

Hazen, Szostak et al.: −log2 of the fraction of configurations that perform a function.
Here: 128 random growth parameters (m, s), each within ±50 % of the pattern's own, run for
500 steps on 128²; the function is that the pattern lives on (mass within ½ to 2× the start,
area at most 2×). None found counts as half of one. Lenia only (`growth_params`).

An absolute box (m ∈ [0.05, 0.5], s ∈ [0.005, 0.1]) gave ~6 bits for every creature (1 of
64 rules), too coarse to compare; the relative box asks how fragile a creature is to its rule.

### 5.8 Deferred

- **Thermodynamic depth (TD; Lloyd & Pagels).** For a deterministic system it is the
  entropy over a state's possible histories; computable only for tiny lattices. Revisit
  with elementary CA (M4).
- **Integrated information (IIT; Tononi).** Φ is exponential in system size: only ~3×3
  binary patches are tractable (`pyphi`), and a patch's Φ is not the system's. It is a
  whole-system measure, not a local one as the PDF tags it (§2.3).
- **Assembly index (AT; Cronin, Walker).** The libraries target molecules and strings, and
  the index is exponential in object size. Revisit for small discrete patches, with copy
  number (§8.2).
- **Wuensche's Z (WZ).** Rule-level, static, for elementary CA; added with that system (M4).

## 6. Validation

- **Baselines** (`baseline` system): empty; iid noise (new every step) and frozen noise (one
  frame held), each binary, 13-level or continuous, matching Life, Integer Lenia and Lenia.
- **Orientation test**: for every measure except "mixed", iid noise ≥ empty, for each kind
  of noise.
- **Structure test**: every structural measure rates a Gosper glider stream above both
  iid noise and empty.
- **Expectations** (the table is tested):

  | Measure | empty | frozen noise | iid noise | glider stream |
  |---|---|---|---|---|
  | compression (frame) | ≈ 0 | high | high | low |
  | compression (video) | ≈ 0 | ≈ 0 | high | low |
  | transfer entropy | 0 | 0 | ≈ 0 | > 0 |
  | statistical complexity | 0 | 5 bits (binary) | ≈ 0 | > 0 |

- **Per-measure tests**: LZ76 against the direct parse; block entropy on stripes; FD on a
  point, line and plane; PID and FID on textbook gates; ITI on self- and environment-driven
  series; Langton's λ of Life; local maps average to the value; tile maps are zero away from
  structure; parameters from the UI are typed and checked.

## 7. First values

At 128², step 200 unless noted (TE: step 100). Empty and iid noise rows are the baselines.

| Run | LZC | SE | FD | EC | LD ns/cell | LE | TE | SC |
|---|---|---|---|---|---|---|---|---|
| empty | 0.002 | 0 | 0 | 0.059 | 1.66 | 0.016 | 0 | 0 |
| iid noise, binary / 13 / continuous | 1.03 / 0.93 / 0.95 | 1.0 / 3.5 / 3.0 | 2.0 | ≈ empty | 2.7 / 12.5 / 0.19 | 0.20 / 0.22 / > 1 | 0 | 0 / 0.04 / 0.05 |
| Life glider / gun / soup | 0.004 / 0.027 / 0.31 | 0.003 / 0.036 / 0.32 | 0.40 / 0.80 / 1.35 | 0.076 / 0.12 / 0.36 | | 0.065 / 0.084 / 0.11 | 0.0005 / 0.0038 / 0.025 | — / 0.22 / 2.4 |
| min Orbium / Integer Lenia soup | 0.018 / 0.20 | 0.057 / 0.80 | 1.00 / 1.66 | 0.086 / 0.55 | | 0.11 / 0 | 0.0061 / 0.0098 | 0.09 / 3.4 |
| Orbium / Lenia soup | 0.019 / 0.25 | 0.054 / 0.83 | 1.45 / 1.65 | 0.084 / 0.72 | | 0.011 / 0.022 | 0.0054 / 0.0065 | 0.10 / 3.5 |
| Helicium / Paraptera a. l. / P. c. f. / Tetralapillium | 0.04 / 0.05 / 0.04 / 0.02 | 0.14 / 0.16 / 0.14 / 0.06 | 1.39 / 1.46 / 1.38 / 1.05 | 0.10 / 0.11 / 0.11 / 0.09 | | 0.008 / −0.02 / 0.016 / 0.073 | 0.006 / 0.018 / — / 0.0006 | 0.38 / 0.38 / — / 0.20 |

- PID synergy: Life equal to TE; Lenia about half of TE (Paraptera a. l. 0.012 of 0.018).
- FID synergy: glider 0.086, gun 0.053, Life soup 0.038, Paraptera a. l. 0.12, Lenia soup
  0.005; noise and empty 0.
- ITI (organismal A*): `iti_local` 0.9–1.1 for Life and Lenia creatures, 0.4–0.6 for Lenia
  soups, 0 for noise; `iti_features` Orbium 0, gun 0.45, Tetralapillium 1.0, noise 0.03.
- TL: still baselines 0; Life soups 9.3; Orbium-rule soups (die) 7.5; min Orbium 8.6;
  Paraptera, Tetralapillium and noise at the 4096-step cap (12).
- LL: Life 0.269 (exact 0.273); min Orbium 0.84.
- FI: Lenia soup 1.1 bits; min Orbium 3.1, Orbium 3.5, Hydrogeminium and Tetralapillium 3.7,
  Hexadentium 3.8, Paraptera c. f. 4.0, Kronium 4.2, Gyrorbium 4.7, Paraptera a. l. 5.4,
  Scutium and Helicium 7.

Readings: LZC, SE, FD and compression rank soups and noise on top. EC, SC, TE and FID put
noise at the empty-grid level. LE singles out Tetralapillium metamorpha among the creatures;
Paraptera cavus furiosus, listed as chaotic, is only weakly so here. Integer Lenia heals
single-level damage.

## 8. The goal measure and proposals

### 8.1 The goal measure (Bert, 2026-10-30)

A **dynamic spatial map of complexity**, possibly coarser than the simulation in space and
time, that is low for empty and pure noise, high for self-organizing patterns, and
**highest for small self-contained units forming larger networks**, above a patch of goo
(self-organizing, but at one level). In the terms of §2: structural, pattern-level, local,
dynamic, with a map — and also hierarchical, which no measure in §3 is.

Closest today: local statistical complexity (highlights particles and domain walls; ~0 for
noise and uniform regions) and local TE, but both are single-scale: busy goo can score as
high as a network.

**Candidate: multi-scale structural complexity.** Compute a structural measure (EC, local SC
or local TE) on a pyramid of coarse-grainings (1, 2, 4, 8, … cells), each against its own
noise baseline, and score high only when structure exists at several scales (minimum or
product across scales). A network of units has structure at the unit scale (their
internals) and at the network scale (their arrangement); goo has one scale; noise and empty
none. Related to Bar-Yam's complexity profile (Allen, Stacey & Bar-Yam 2017).

**Reference cases** (M3b): empty and noise (lowest); goo-like structures (to be found near
the Aquarium's parameters); a single creature (Aquarium unit); Aquarium networks, which
self-replicate (`aquarium`) or divide on collision (`aquarium_scene`) (highest). See
PLAN.md §4.2b for the Aquarium: Lenium's and Leniabreeder's parameter sets differ by ~0.5 %
in one kernel's radius, and that switches self-replication on and off.

### 8.1b Use of the goal measure (Bert, 2026-10-31)

Measure a simulation, live or offline, and indicate **which regions — spatiotemporal
ranges — have high spatial (structural) and temporal (dynamic) complexity**, in order to:
1. **pick out high-complexity objects** (regions and their tracks), and
2. **compare simulation conditions** by their average and maximum complexity.

Spatial and temporal blurring and delay are acceptable.

**Proposed design** (to prototype in M3b):
- A **complexity field C(x, y, t)** on coarse tiles (e.g. 8×8 cells) over a sliding window
  (e.g. 16–32 frames), updated every few steps. The allowed blur and delay are what make
  per-tile estimates reliable.
- Two components per tile and window, each scaled between its value on empty and on
  matched noise so that values compare across systems:
  - **S, spatial (structural)**: structure that noise with the same statistics cannot
    explain, at several scales (1, 2, 4, 8 cells): the noise-band coarse-graining of EC, or
    local statistical complexity, per scale (§8.1).
  - **D, temporal (dynamic)**: change that is neither absent nor random: local TE or
    predictable change. 0 for still structure and frozen noise, ~0 for iid noise, high for
    coherent motion and transformation.
- **C = S × D**: a region must be both; still structure (high S, no D) and churning noise
  (high D, no S) score low.
- **Estimation**: probabilities pooled over all tiles and the recent window, or over many
  runs of the same conditions; local values then read per tile (§2.3).
- **Outputs**: the C heatmap; **hot regions** (C above a threshold, connected in space and
  time: position, extent, peak, integral; boxes in the UI, exportable); **run summaries**
  (mean, maximum, 95th percentile over space and time) for comparing conditions, batched
  over seeds and parameters (M5 experiments).
- **Validation ranking**, lowest to highest: empty, iid noise, frozen noise; still structure;
  goo / chaotic soup; a single glider or creature; Aquarium networks.

### 8.2 Other proposals

- **Local information dynamics** (Lizier): local active information storage, transfer and
  modification. Particles transfer information, domains store it, collisions modify it; a
  network of interacting units shows all three.
- **Causal emergence / integrated information decomposition, ΦID** (Rosas, Mediano et al.
  2020): whether a macro variable (e.g. per-unit centroids) predicts itself beyond its
  parts — "units forming a network" is macro-level behaviour. Principled, but needs chosen
  macro variables and is harder to estimate.
- **O-information** (Rosas et al. 2019): whether many variables are dominated by synergy or
  redundancy; has a local form; a signature of networks, but not hierarchical by itself.
- **Pointwise PID** (e.g. Finn & Lizier 2018): a per-cell PID, which would give PID a local map.
- **Assembly copy number × index**: many reused building blocks forming larger objects;
  computationally hard on fields.
- **Per-creature ITI plus the number of interacting individuals**: units (individuality)
  and their network directly; needs the target-creature design of `iti_features` (M3b).
- **Logical depth without the timing artefact** (§5.2.3).

## 9. Questions and answers

Collected from the design conversation (2026-10-30 to 2026-10-31).

**Q1. Most measures rate iid noise highest, but effective complexity, logical depth and
transfer entropy rate it low, often near 0. Should those be inverted, to keep "higher =
more complex"?**
No. They measure structured complexity, and the others measure randomness (§2.1). The
principle is "higher = more complex *by the measure's definition*". Inverting a measure that
is low at both ends would make empty grids and noise score highest and structure lowest,
which fits neither meaning. Negation fixes only measures that run the wrong way along the
order-to-disorder axis. Instead, the UI tags each measure 🎲 randomness or 🌿 structural.

**Q2. Why is logical depth "mixed"?**
Bennett's logical depth is structural in theory, but the decompression-time proxy measures
LZMA's decoder work: incompressible noise decodes fastest (stored raw), partly compressible
13-level noise slowest of all, so noise lands at either end depending on its levels (§5.2.3).
It fails both the randomness test and the structure test. Proposed fixes in §5.2.3.

**Q3. Which measures have granularity in space, time and rule?**
§4: all frame measures give time series; TE, SC, block entropy, LE and `iti_local` give
per-cell maps; compression, LZC, PID and FID give per-tile maps; TL, LL and FI are per rule.

**Q4. Do transfer entropy and statistical complexity have maps because they are tagged
structural, pattern, local and dynamic?**
Partly. A map needs pattern-level (computed from the frames) and, more precisely, being an
average of a per-cell quantity; `local` is a hint, not a guarantee (PID is local but has no
per-cell form). `dynamic` and `structural` are not needed (block entropy is static and
randomness-peaked, and has a map), but they make the map useful for the goal: dark on noise
and empty, following what structures do. TE and SC are the two with all four.

**Q5. Does any measure have the ideal property (§8.1)?**
No: structural, local, dynamic measures exist (SC, TE), but none prefers networks of
self-contained units over goo; that needs a multi-scale measure (§8.1), which is planned as
M3b with the Aquarium as the reference case.

**Q6. Lenia creatures often pass through the repeating border. Does that cause problems?**
Barely (measured 2026-10-30, Orbium on 64², 179 of 380 steps straddling; Life glider on
32²). Measures that treat the grid as a torus, as the simulation does (TE, block entropy,
PID, FID, SC, `iti_local`), do not change (exactly 0 for the glider). Measures that read the
frame row by row see the creature cut in pieces: image compression < 1 %; video LZMA,
novelty and LZC +2 % for Orbium and +11 % for the tiny glider. FD and LE vary more from step
to step than with the border. Centering by center of mass would remove the seam for the
image measures without changing their meaning, but re-centering every frame would erase the
motion that video compression and novelty measure; rolling the whole window by one shift
keeps it. Not done: the effect is small next to the differences between patterns.

**Q7. Which ITI: features of a segmented creature, or a local per-cell form?**
Both, as two measures (§5.4). The paper's form needs scenes of several creatures to be
meaningful (for a lone creature it measures the predictability of its mass); the Aquarium
is its test bed. The local form estimates well and gives a map, but is an adaptation.

**Q8. Why are there bias corrections in several measures?**
Plug-in estimates of information are biased upward when bins are many and samples few. TE
uses Miller–Madow (it removes the iid bias exactly when all bins are filled); `iti_features`
uses a shuffled surrogate (Miller–Madow was not enough for 127 samples over 64 bins);
statistical complexity merges states by a G-test rather than a distance threshold.

**Q9. Why is FID implemented here instead of depending on `fid-tools`?**
The library is 0.1.2, beta, from one author (January 2026) and depends on matplotlib; its
decomposition is ~15 lines. It was read (not run) and reimplemented with its conventions
(uniform inputs over observed values, neutral completion over observed outputs).

**Q10. Do the Aquarium and the Lenia creatures use bump or Gaussian kernels?**
Gaussian. The hand-coded Lenium, Chan's Lenia tutorial and Leniabreeder all build
multi-channel kernels as Gaussian rings `bell(D mod 1, 0.5, 0.15) · b`, so the Aquarium was
found on Gaussian kernels. The ten catalogue creatures were found on the bump core
(`kn = 1`) but run here on the Gaussian shell, which they survive (PLAN.md §4.2).

**Q11. Is the granularity right — spatial, temporal, and subject (pattern / rule /
system)?**
Re-derived from the literature on 2026-10-31, without the PDF or these notes (§2.3). Most
tags agreed. Fixed: FID is rule-level; block entropy is local (and system-level); IIT is not
local; the Lyapunov exponent is rule-level; functional information is ensemble-level. The
deeper finding is that information measures are properties of distributions: estimated from
one run they describe the run, not the rule, which one granularity tag cannot say — so each
measure now also states its subject and what it is estimated from.

## 10. Open questions

- Multi-scale structural complexity: which base measure, which combination across scales
  (minimum, product, area under the profile), and how to set each scale's noise baseline.
- `iti_features` on the Aquarium: choosing and tracking the target creature among copies.
- Does H.264 beat video LZMA for smoother (larger-R) creatures or longer windows?
- Transfer entropy for smooth fields: offset ≈ R/4, and history length k > 1.
- Validate statistical complexity against `emic` on 1-D series.
- Logical depth: does dividing time by compressed size make it reliably structural?
- Estimate causal states and probabilities from an ensemble of runs of the rule, then read
  local values on one run (§2.3), instead of from one 16-frame window.
