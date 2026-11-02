# Complexity measures — design notes

The definitions, explanations and categorization of the complexity measures in Complexity
Lab: the ones implemented, the ones deferred, and the ones proposed. Also the decisions
behind each implementation, first values, and the questions and answers that shaped them.
The project plan and milestones are in [PLAN.md](PLAN.md); background on the measures
is in [assets/Complexity_Measure.pdf](assets/Complexity_Measure.pdf) (cited as "the PDF"), and
the papers behind each measure are catalogued in [assets/README.md](assets/README.md) (§2.4).

Contents: 1 Purpose · 2 Conventions (2.3 reanalysis of granularity, 2.4 checked against the papers) · 3 Catalogue · 4 Granularity and heatmaps ·
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

Every measure declares `peaks_at`, shown in the UI as a tag:

| Kind | Meaning | Members |
|---|---|---|
| randomness | rises from order to disorder; highest for noise | compression (all), LZC, block entropy, fractal dimension, Lyapunov exponent, transient length, Langton's λ |
| structural | low for both uniform frames and noise, highest in between | effective complexity, statistical complexity, transfer entropy, PID, FID, ITI (both), functional information |
| mixed | neither reliably | logical depth (its proxy, §5.2.3) |
| statistic | not a complexity measure; for reference | total mass, moving speed, change rate (§5.9) |

Each measure also has an emoji before its name in the UI (`Measure.icon`), by meaning;
variants share one: 🖼️ image compression (LZMA, PNG), 🎬 video compression (LZMA, H.264),
✨ novelty, 🔤 Lempel–Ziv, 💎 effective complexity, ⏱️ logical depth, 🔮 statistical
complexity, 🌊 transfer entropy, 🪟 block entropy, 🔗 PID and FID, 🍄 individuality (both),
❄️ fractal dimension, 🦋 Lyapunov exponent, ⏳ transient length, 🎚️ Langton's λ, 📚
functional information, 🧊 mass, 💨 speed, 🔄 change rate. The ⓘ beside each measure in the UI opens its description, equation
(`Measure.equation`, LaTeX, rendered by KaTeX), facts, reference and normalized form; clicking a tag lists all tags with their meanings.
(Until 2026-11-01 the emoji showed the kind: 🎲 randomness, 🌿 structural.)

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
| discrete | — | defined for discrete states only (a rule table, exact repeats); skips continuous systems |

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
| TL (Hudcová & Mikolov) | rule | global | whole trajectory | classifies rules by how the mean transient grows with grid size: bounded, log, linear, polynomial, exponential |
| PID (Williams & Beer) | ensemble (a joint distribution) | as the variables; pointwise variants (Finn & Lizier) | dynamic only with time-lagged variables | rule-level from a rule table, trajectory-level from a run |
| FID (Bohm et al.) | **rule** (the input–output function) | the function's inputs | one-step mapping | uniform inputs make it a property of the function |
| TE (Schreiber; Lizier) | process (a pair of stationary processes) | pointwise (local TE) | window or asymptotic | averages characterize rules, local values locate gliders |
| IIT, Φ (Tononi) | a system in a state | **regional, not pointwise**: φ is defined over partitions of a candidate set; the exclusion postulate picks maximal sets ("complexes"), recursively, as non-overlapping regions of a larger substrate | one step | interventional: transition probabilities under all interventions (the do-operator) |
| ITI (Krakauer et al.) | an individual vs its environment | regional | window or asymptotic | needs an identified individual |
| AT (Cronin, Walker et al.) | assembly index: instance (an object); assembly A = Σ e^{a_i}(n_i − 1)/N_T: an ensemble of objects with copy numbers n_i | regional | snapshot (encodes a construction history) | the index alone cannot detect selection; index × copy number can |
| FI (Hazen, Szostak) | ensemble of all configurations, for a function and a degree of function E_x | global | snapshot (dynamic if the function is run) | I(E_x) = −log2 F(E_x), a curve over E_x; undefined if no configuration functions |
| LL (Langton) | rule (table) | global over a local rule | static | |
| WZ (Wuensche) | rule (table) | global over a local rule | static | |

**Comparison with the PDF and these notes before the reanalysis.** Agreement for AC and
compression, LZC, FD, EC, TL, LL, WZ and ITI. Corrections:
- **FID is rule-level** (both said system): it is defined on the function, with uniform inputs.
- **Block entropy is local** (these notes said global, while giving it a per-cell map), and
  its granularity is the distribution's: system, as in the PDF's §1.1.
- **IIT is not local** (the PDF said local): φ is defined over partitions of a whole candidate
  system, and the complexes it picks are regions, not per-cell values (§2.4).
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

Re-derived from knowledge of the literature, then checked against the papers (§2.4).

### 2.4 Checked against the papers (2026-10-31)

The 22 papers catalogued in [assets/README.md](assets/README.md) (one per measure, three
for IIT, two each for SC and WZ) were read against these notes. What they confirm, correct
or add:

| Measure | Paper | Result |
|---|---|---|
| FID | Bohm et al. 2026 (arXiv v3) | **Corrected**: synergy is never negative. With every input combination weighted equally the inputs are independent, so I_syn = Σ H(X_i \| Y) − H(X \| Y) = I(X_1; X_2 \| Y) ≥ 0 (their eq. 3); there is no redundancy term. The code's description said "negative when redundant". Their Game of Life example (all 9 cells as inputs: total 0.84635, synergy 0.63765, centre 0.01092, each neighbour 0.02472 bits) is reproduced exactly and now tested. Bohm's "system" is the function itself, so the PDF's tag "system" means what §2.3 calls rule. |
| TL | Hudcová & Mikolov 2020 | **Implementation replaced** (it reported a transient at one size): the measure is the growth class of the mean transient with grid side n (fitted over n = 20–200: bounded, log, linear, polynomial, exponential). ECA: exponential ≈ chaotic; Life is linear (R² 0.98), reproduced here; in 2-D the linear (steep) and polynomial classes hold most complex rules. Defined for discrete systems only. §5.7.1. |
| FI | Hazen et al. 2007 | **Implementation replaced** (it sampled rule parameters with a pass-or-fail function): configurations are configurations of the system — here random patterns of the creature's size — and the degree of function is graded; the FI of one configuration counts those that function at least as well. §5.7.3. |
| LE | Yevenko et al. 2025 (ISAL; the PDF says 2026) | **Adds**: on Asymptotic Lenia they compute the whole Lyapunov spectrum (Benettin's JVP + QR, ~10⁴ steps) and covariant Lyapunov vectors, and propose the Kaplan–Yorke dimension as *the* complexity measure for continuous CA (chaotic soliton: D_KY ≈ 8.5). Symmetry (two translations, rotation, time) gives up to 4 zero exponents: 3 for a translating or rotating soliton, so Orbium's near-zero values here are expected. Nearly all other exponents are negative (mean −1), which they give as the reason creatures withstand perturbation — the repair that M in §8.1b wants to measure. §5.6. |
| IIT | Tononi 2004; Oizumi et al. 2014; Albantakis et al. 2023 (IIT 4.0) | **Refined**: interventional (do-operator), as assumed. Exclusion picks the maximal set of units ("complex"), and complexes can be found one after another over a larger substrate as non-overlapping regions — causal individuation with boundaries, close to Q13. φ is sensitive to "fault lines", weakly coupled parts. The PDF's Φ = D(intact ‖ partitioned) is the IIT 2–3 form; IIT 4.0 uses the intrinsic difference over directional partitions. §5.8. |
| AT | Sharma et al. 2023 | **Refined**: the assembly index is per object; assembly A combines indices with copy numbers over an ensemble. Many copies of a many-step object signal selection — as do Aquarium colonies of dozens of identical units. §5.8, §8.2. |
| ITI | Krakauer et al. 2020 | **Confirmed** (A*, A, nC as in §5.4; their fine decomposition by PID: A* = SI + UI(S), A = CI + UI(S), nC = CI + UI(E)). **Adds**: a boundary search — enlarge S with environment variables while nC falls — and the caveat that A* and A never decrease as S grows, so boundaries need a cost on size. The paper gives no algorithm; its examples are small Markov models. |
| PID | Williams & Beer 2010 | **Confirmed**: I_spec and I_min exactly as §5.3.4. |
| TE | Schreiber 2000 | **Confirmed** (histories k and l; l = 1 recommended). **Adds**: for coarse-grained continuous states TE converges as bins shrink, *except under deterministic coupling, where it diverges* — Lenia is deterministic, so TE grows with `n_levels` and values compare only at equal levels. |
| SC | Crutchfield & Young 1989; Crutchfield & Feldman (SFI WP 1996) | **Confirmed** (C_μ as the entropy of causal states; the 1989 reconstruction merges subtrees within a tolerance, here a G-test). **Adds**: excess entropy E ≤ C_μ, and the complexity–entropy diagram (E or C_μ against entropy density h_μ) as a parameter-free way to compare systems. |
| EC | Gell-Mann & Lloyd 1996 | **Confirmed, with more detail**: EC = K of the ensemble in which the entity is typical; total information Σ = EC + S(ensemble); the best ensemble minimizes Σ, then EC. The noise-band proxy fixes one ensemble instead of minimizing. §5.2.2. |
| LD | Bennett 1988 | **Confirmed, with more detail**: depth at significance s = least time of a program at most s bits longer than the shortest. Bennett treats regional depth in CA, and notes that irreversibility lets noisy systems correct errors. The decompression-time proxy is not in the uploaded Zenil 2020 (it is Zenil, Delahaye & Gaucherel 2012). |
| AC, LZC | Zenil 2020 | **Confirmed**: LZ-family compressors estimate entropy rate, not algorithmic complexity (so randomness). **Adds**: CTM/BDM, BDM(s) = Σ CTM(block) + log2(copies), works on 2-D arrays and gives a per-block map; "algorithmic information dynamics" ranks parts by how perturbing them changes complexity — another interventional precedent (Q15). |
| SE | Shannon 1948 | **Confirmed**: block entropy is Shannon's G_N, the entropy per symbol of N-blocks. |
| FD | Mandelbrot 1967 | **Confirmed** (self-similarity dimension of a set in space). Not the attractor dimension of Yevenko et al. — that one is in phase space. |
| LL | Langton 1990 | **Confirmed**: λ_Life = 0.273, in the transition region for 2-state, 9-cell CA. λ = 1 − 1/K is the most heterogeneous table (Life 0.5, 13 states 0.92), so min Orbium's 0.84 is near it. |
| WZ | Wuensche 1994, 1998 | **Implemented** (§5.7.4; the 1994 worked example reproduced). **Confirmed**: Z = max(Z_left, Z_right), the probability that the next cell of a pre-image is fixed by the rule table; low Z is ordered (many garden-of-Eden states), high Z chaotic, complex in between. **Adds** (1998): the variance of the input entropy (entropy of the rule-table lookup frequencies) over time — high for glider rules. It needs only the run and the lookup table, so it applies to Life and Integer Lenia now (§8.2). |
| TD | Lloyd & Pagels 1988 | **Implemented** exactly on tiny rings (§5.7.5). **Confirmed**: depth of a state = −k ln p(the trajectory that led to it \| the state); the mean over trajectories is their entropy, for Hamiltonian systems coarse- minus fine-grained entropy. |

## 3. Catalogue

All 18 measures of the PDF, plus the variants added here and the proposals. Tags are
granularity, scope, temporal (§2.2); "computed" is subject / estimated from. "Map": local =
one value per cell; tile = per tile of the frame (§4). "discrete" marks measures defined
for discrete states only (§2.2).

| PDF | Measure | Key | Kind | Tags | Computed | Map | Status |
|---|---|---|---|---|---|---|---|
| AC | Algorithmic complexity, by compression | `lzma`, `png`, `video_lzma`, `video_h264`, `novelty` | randomness | pattern, global, static (video: dynamic) | instance / object (video: trajectory / run) | tile (not H.264, novelty) | implemented |
| AC | Lempel–Ziv complexity | `lzc` | randomness | pattern, global, static | instance / object | tile | implemented |
| EC | Effective complexity (proxy) | `effective_complexity` | structural | pattern, global, static | instance / object | — | implemented |
| LD | Logical depth (proxy) | `logical_depth` | mixed | pattern, global, static | instance / object | — | implemented |
| SC | Statistical complexity (local) | `statistical_complexity` | structural | pattern, local, dynamic | trajectory / run | local | implemented |
| TE | Transfer entropy | `transfer_entropy` | structural | pattern, local, dynamic | trajectory / run | local | implemented |
| SE | Shannon (block) entropy | `block_entropy` | randomness | system, local, static | instance / object | local | implemented |
| PID | Partial information decomposition | `pid` | structural | system, local, dynamic | trajectory / run | tile | implemented |
| FID | Functional information decomposition | `fid` | structural | rule, local, dynamic | rule / run | tile | implemented |
| ITI | Individuality, as in the paper | `iti_features` | structural | pattern, local, dynamic | trajectory / run | — | implemented |
| ITI | Individuality, local adaptation | `iti_local` | structural | pattern, local, dynamic | trajectory / run | local | implemented |
| FD | Fractal dimension | `fractal_dimension` | randomness | pattern, global, static | instance / object | — | implemented |
| LE | Lyapunov exponent | `lyapunov` | randomness | rule, global, dynamic | trajectory / run | local | implemented |
| TL | Transient length (growth class) | `transient_length` | randomness | rule, global, dynamic, discrete | rule / ensemble | — | implemented |
| LL | Langton's λ | `langton_lambda` | randomness | rule, global, static, discrete | rule / ensemble | — | implemented |
| WZ | Wuensche's Z | `wuensche_z` | randomness | rule, global, static, discrete | rule / rule table | — | implemented (§5.7.4) |
| FI | Functional information | `functional_information` | structural | system, global, dynamic | ensemble / ensemble | — | implemented |
| TD | Thermodynamic depth (exact, tiny rings) | `thermodynamic_depth` | mixed | pattern, global, dynamic | rule / ensemble (all configurations) | — | implemented (§5.7.5) |
| IIT | Integrated information | — | — | system, global, dynamic | — | — | deferred (§5.8) |
| AT | Assembly index | — | — | pattern, local (regional), static | — | — | deferred (§5.8) |
| — | Total mass (statistic) | `mass` | statistic | pattern, global, static | instance / object | — | implemented (§5.9) |
| — | Moving speed (statistic) | `speed` | statistic | pattern, global, dynamic | trajectory / run | — | implemented (§5.9) |
| — | Change rate (statistic) | `change_rate` | statistic | pattern, global, dynamic | trajectory / run | — | implemented (§5.9) |
| — | Multi-scale structural complexity | — | structural | pattern, local, dynamic | trajectory / run (better: ensemble) | local | proposed: the goal measure (§8) |
| — | Local information dynamics (storage, transfer, modification) | — | structural | pattern, local, dynamic | trajectory / run | local | proposed (§8.2) |
| — | Causal emergence / integrated information decomposition (ΦID) | — | structural | pattern | trajectory / run | per unit | proposed (§8.2) |
| — | O-information | — | structural | system, local | trajectory / run | local | proposed (§8.2) |
| — | Pointwise PID | — | structural | system, local, dynamic | trajectory / run | local | proposed (§8.2) |
| — | Logical depth, better proxies | — | structural? | pattern, global | instance / object | — | proposed (§5.2.3) |
| — | Assembly copy number × index | — | structural | pattern | instance / object | per object | proposed (§8.2) |

Eight measures also have normalized, per-object versions (§5.10).

Code: `src/complexity/measures/` — `compression.py`, `algorithmic.py` (LZC, EC, LD),
`causal.py` (SC), `information.py` (TE, block entropy), `decomposition.py` (PID, FID),
`individuality.py` (ITI), `geometry.py` (FD), `dynamics.py` (LE), `rule.py` (TL, LL, FI, WZ, TD),
`statistics.py` (mass, speed, change rate).

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
| IIT (deferred) | tiny patches only | ✓ | indirect |
| AT (deferred) | per object | ✓ | indirect |
| TL, LL, FI, WZ, TD | — | — | ✓ |

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
- **Not measures**, always listed first: the neighbour sum K ∗ A (`System.potential`, the
  orange ramp) and the detected objects (§5.4: the target in one colour, the other
  creatures in a muted palette; the target is the one `iti_features` tracks when it runs,
  else the run's own tracker, picked and followed the same way). Systems without a potential show no
  neighbour sum, and their objects come from the frame.
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

**Against the definition.** Gell-Mann & Lloyd's EC is the algorithmic information of the
*ensemble* in which the object is typical; the best ensemble minimizes the total information
Σ = EC + S (the entropy left within the ensemble), then EC — a two-part code. The noise band
fixes one ensemble (blur σ = 2, 3 levels) instead of minimizing. A closer proxy: over
coarse-grainings (σ, levels), minimize compressed model bits + residual entropy, and report
the model bits.

**Why the noise band.** Fixed levels after blurring failed both ways: a blurred glider gun
never reached 1/3, while blurred binary noise straddled 1/2 (noise scored above the gun).
With z = 3, binary noise still leaked through (0.27 % of cells); z = 4 fixed it.

#### 5.2.3 Logical depth (`logical_depth`, LD)

**Definition.** Bennett: the running time of the shortest (or near-shortest) program that
produces the object. Trivial and random objects are both shallow; deep objects need long
computation from a short description. Precisely, depth at significance s is the least
running time of a program at most s bits longer than the shortest. **Proxy** (Zenil,
Delahaye & Gaucherel 2012): time to decompress the
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
offset 1 may be short; try ≈ R/4. Lenia is deterministic, and for deterministic coupling
TE diverges as the bins shrink (Schreiber): compare values only at equal `n_levels`. Per-cell TE of a lone object shrinks with grid size.

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
targets; then I(T; X_i) and synergy = I(T; X) − Σ I(T; X_i), never negative (the equally
weighted inputs are independent, §2.4). `component`: synergy (default), self, neighbour.
Tested on XOR, copy, a completed OR, and Bohm et al.'s Game of Life numbers.

**Notes.** Synergy does not depend on how much of the grid is active, so a lone glider
(0.086) scores above a Life soup (0.038): it describes the local rule as used. With only two
of the nine inputs, the other seven are averaged over the run's own frames, so the value
depends on the run, not on the rule alone. Bohm et al. use all 9 cells (512 combinations)
and get one number for Life's rule, synergy 0.638 bits; that full form is exact from the rule
table for Life (and small neighbourhoods), not for Lenia.

### 5.4 Individuality (ITI)

**Definition.** Krakauer et al. 2020, information theory of individuality, for a system S
and its environment E at step n:
- organismal individuality A* = I(S_{n+1}; S_n)
- colonial individuality A = I(S_{n+1}; S_n | E_n)
- environmental determination nC = I(S_{n+1}; E_n | S_n)

`component` selects one (organismal by default). Creatures are 8-connected components above
a threshold, joined across the torus's edges (`torus_label`). Both forms were requested
(Bert, 2026-10-30).

**`iti_features` (as in the paper; scenes since 2026-11-01).** S = mass of a tracked target
creature, E = mass of its environment, each cut into 4 quantile levels over a 128-frame
window. Miller–Madow left iid noise at 0.10–0.12 bits (64 bins, 127 samples), so the mean
over 20 time-shuffled S′ is subtracted instead.
- *Detection, in the potential K ∗ A* (Bert's suggestion: how the model sees the world;
  `System.potential`, recorded by the run beside the frames for measures with
  `needs_potential`): the catchment basins of its peaks above `threshold` (0.05; every cell
  points to its highest 3×3 neighbour, pointers followed to a peak, on the torus), then
  neighbouring basins merged when their saddle is at least `saddle` (0.8) × the lower peak,
  and basins smaller than a disk of radius R/4 dropped (Q18). Systems without a potential
  fall back to the frame above the threshold.
- *Target, one per run* (`Tracker`): picked in the first frame the measure sees
  (`typical`, the creature closest to the median mass, default, or `largest`), then
  followed frame by frame across ticks: its successor is the creature sharing most of its
  cells, at least half of them (when it divides, the larger part); when none does (a
  division, a merge, a lost detection), the creature nearest its last centre takes over.
  Each frame is processed once and keeps its labels, target and (S, E), so a tick costs
  one frame (29 ms on the Aquarium scene). Until 2026-11-01 the target was picked afresh
  at the start of every tick's window: over 60 one-step ticks it changed creature 26
  times and was lost 6 times (the objects heatmap showed it jumping), and each tick's
  value described a different unit; now 1 takeover and no loss. With takeovers in more
  than a quarter of the window's frames the target is no individual and the value is 0:
  on iid noise every frame's creatures are unrelated and a takeover happens almost every
  frame (without the rule, 0.17 bits on 20 frames).
- *Environment*: `near` (the other creatures within the rule's reach R of the target,
  default: what can affect it in Lenia) or `all` (everything else, as in the paper).

Values (256², step; organismal A* / colonial A / driven nC, near; on the GPU runs are not
bit-reproducible, chaotic Lenia drifts apart, and the scene at 600 has also given 0.92):
Aquarium scene at 600
0.69 / 0.39 / 0.09, at 2000 1.31 / 0.63 / 0; self-replicating Aquarium at 1000
0.87 / 0.46 / 0; a lone Aquarium unit 0.12 / 0.12 / 0; Orbium (128²) 0.32 / 0.32 / 0.
With `all`, colonial A is higher (scene 0.55 at 600, 0.98 at 2000) and nC 0. In the
scenes, conditioning on the neighbours (A) removes about half of A*: half of what predicts
a unit's mass is shared with what its neighbours do. A glider stream scores 0: the typical
target is a glider, of constant mass, and A* ≤ H(S) = 0. Paraptera cavus furiosus is
detected as its two units (Q18), so each is the other's environment (nC 0.03).

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

**Against Yevenko et al. 2025** (Asymptotic Lenia). This is one finite-time rate of the
largest exponent from one random perturbation over 64 steps; they compute the whole spectrum
(Benettin: push k tangent vectors through `jvp` and re-orthonormalize by QR each step, ~10⁴
steps) and the covariant Lyapunov vectors, which are spatial maps of the unstable, neutral
and stable directions. Symmetry gives up to 4 zero exponents (3 for a translating or rotating
soliton), so near-zero values for Orbium and other gliders are expected, not a failure of
the estimate. They propose the Kaplan–Yorke dimension D_KY = j + Σ_{i≤j} λ_i / |λ_{j+1}| as the
complexity measure for continuous CA, and read the many negative exponents as the reason
creatures absorb perturbations. Both are direct in JAX (proposed, §8.2); our Lenia clips the
state to [0, 1], so its Jacobian is zero on saturated cells, unlike their smooth system.

### 5.7 Rule-level measures

`RuleMeasure`: computed once per (measure, parameters, system, pattern) from runs of its own,
cached, and shown as a constant; off by default in the UI (seconds on the first tick).

#### 5.7.1 Transient length (`transient_length`, TL)

**Definition.** Hudcová & Mikolov 2020: the transient of a configuration u is the number of
steps before its trajectory first enters a cycle (the smallest i with Fⁱ(u) = Fʲ(u) for some
j > i). The measure is how the mean transient μ_n of random configurations grows with the
grid side n: bounded, logarithmic, linear, polynomial or exponential, by the best R² of
regression fits. Bounded rules compute trivially; exponential ones are chaotic; Life is
linear; in 2-D, linear rules with steep slopes and polynomial ones hold the complex
behaviour. Defined for deterministic systems with finitely many states.

**Implementation.** Sides: 6 values spaced geometrically from 16 to 96 (`min_size`,
`max_size`, `n_sizes`); for a rule that looks further than a 3×3 neighbourhood
(`System.reach`, R for Lenia), from just above its width, 2R + 2, keeping the ratio
(min Orbium: 26–156). At each side, soups of the rule (`System.soup`) in batches of 8 until
the mean is known to 10 % at 95 % confidence, as in the paper, or 4 batches are run. Each
batch runs in chunks of 256 steps, with two uint32 hashes per frame (a `lax.scan`, `vmap`ped
over the soups), and stops once every soup has repeated, up to 4096 steps. Sizes where more
than a quarter of the soups never repeat are left out of the fit; with fewer than 3 sizes
left the class is exponential (the paper's time bound leaves such rules unclassified; most
are exponential). Fits, with R² in transient steps: μ = a + b ln n, μ = a + b n,
1 + μ = c nᵏ, 1 + μ = c e^{bn}; bounded if the means vary by less than a step or a quarter
of their mean. `report`: `class` (0 bounded, 1 log, 2 linear, 3 polynomial, 4 exponential)
or `exponent` (slope of ln(1 + μ) against ln n, a continuous companion: 0 bounded, 1
linear, large exponential).

**Polynomial means faster than linear** (2026-11-02): a power law wins only with an
exponent above 1; otherwise the log and linear fits decide. Without that, a sublinear power
law (rule 6: n^0.4) fitted log growth better over 6–8 sizes, and 16 of the paper's 38 log
ECA came out polynomial. **ECA check** (`scripts/eca_rule_space.py`, sides 20–200 as in the
paper): 77 of 88 classes agree with Hudcová & Mikolov's table — all 27 bounded, all 38 log,
8 of 9 linear (rule 110 is polynomial here, exponent 1.6; the paper calls it linear with a
steep slope); 4 of their 6 exponential (54 and 73 come out polynomial). Their additive "affine" and
"fractal" classes, whose transients oscillate with n's divisors, split between exponential and
linear here. Comparisons of states use `System.configuration` (an ECA's current row, not its
space-time diagram).

**Notes.** Discrete systems only (Life, Integer Lenia, ECA): continuous Lenia never repeats —
creatures drift by fractions of a cell — so the old 1e-3 quantization gave every soup with
a glider the cap. Linear and polynomial are close for Life (exponent ≈ 1.4): 16–96 gives
linear, as in the paper, but 16–64 with fewer soups gave polynomial. The classes order
rules from order to chaos, hence randomness, but the paper places complex rules in the middle
(linear, polynomial), not at the top. The noise baselines are no longer accepted (some of
their patterns are continuous). ~5 s on GPU, ~20 s on CPU for Integer Lenia.

#### 5.7.2 Langton's λ (`langton_lambda`, LL)

The fraction of the rule table that leads to a non-quiescent state: the probability that a
uniformly random neighbourhood produces a live cell. Estimated by stepping a 256² grid of
iid uniform states once (`System.uniform`). Life: 140/512 = 0.273 (born with 3 of 8
neighbours, 56 tables; survive with 2 or 3, 84). Discrete rules only. A system with a rule
table (`System.rule_table`: elementary CA) gives it exactly, as the table's share of
nonzero outputs (rule 110: 5/8); the estimate from one 128-cell row was 0.605.

#### 5.7.3 Functional information (`functional_information`, FI)

**Definition.** Hazen et al. 2007 (after Szostak): for a function x and a degree of function
E_x, I(E_x) = −log2 F(E_x), where F(E_x) is the fraction of all configurations of the system
that achieve at least E_x. The FI of one configuration uses its own degree of function: how
rare it is to do at least as well. 0 when every configuration does as well, log2 N for the
best of N; undefined if nothing functions.

**Implementation.** Configurations: the pattern's own cells, and 512 random fillings of its
bounding box with iid uniform states (`System.scramble`; for a soup, other soups), placed as
the pattern is, under the pattern's rule, on 128². Function: living on as a bounded object.
Degree of function: steps lived, checked every 50 of 500 steps after settling for 50 —
mass within ½ to 2× the mass after settling, and area (cells above 0.1) at most 2× the
pattern's bounding box; a configuration that is empty after settling scores 0. FI =
log2(N / #{E ≥ E_pattern}) over the N = 513 configurations, the pattern included, so at most
log2 513 = 9 bits: "no random pattern of that size did as well" reads 9, a lower bound.
Life, Lenia, Integer Lenia and multi-channel Lenia; batches of 64 bound device memory.

**Notes** (creature names are of the set before 2026-11-01, on the Gaussian shell). The
settling period lets stored cells grow into the creature (the Aquarium unit
settles at 2.3× its stored mass; Life spaceships are stored one cell short). The area limit
is the same for every configuration: measured against each one's own settled state, random
Hydrogeminium boxes that grow into a third of the grid and then hold still passed. The
function is persistence, not shape or motion: a random 3×3 Life box that becomes a block
counts as doing what the glider does, and under Kronium's and Amoebium's rules many random
boxes settle into bounded blobs, so those creatures score low. A growing pattern (the
Gosper gun, a self-replicating Aquarium) fails its own mass limit early and scores low too.
Most Lenia creatures sit at the 9-bit ceiling; telling them apart needs many more samples
(2048 on the GPU in ~5 s: still 11 bits = ceiling for Orbium, Scutium, min Orbium).

#### 5.7.4 Wuensche's Z (`wuensche_z`, WZ; 2026-11-02)

**Definition.** Wuensche 1994: filling in a pre-image of a state cell by cell from the left,
Z_left is the probability that the next cell is fixed by the rule table; Z_right the same
from the right; Z = max(Z_left, Z_right). For K-cell neighbourhoods, R_K is the share of the
table in *deterministic pairs* (two neighbourhoods differing only in the last cell, with
different outputs), R_{K−1} the share in deterministic 4-tuples (the last cell wild, the one
before it deciding the output), and so on to one 2^K-tuple;
Z_left = R_K + R_{K−1}(1 − R_K) + R_{K−2}(1 − R_{K−1})(1 − R_K) + ….
Z = 1: at most 2^(K−1) pre-images (chaotic and reversible rules); low Z: many pre-images and
garden-of-Eden states (order); complex rules in between, often 0.75.

**Implementation.** From the rule table alone (`System.rule_table`, 1-D binary rules: the
elementary CA), in `rule.py` (`z_left`, `mirror`, `wuensche_z`). Reproduces the paper's
worked example exactly (a K = 5 rule: Z_left 0.671875, Z_right 0.5625; tested), and
Hudcová & Mikolov's Z for 87 of the 88 ECA classes. The 88th, rule 3 (output 1 iff left and
centre are 0), they list as 0.25; here 0.5: from the right, the left cell is fixed whenever
the centre is 0 (half the table), so Z_right ≥ 0.5, and Z is the same for rule 3's mirror
and complement (17, 63, 119). Probably a slip in their table.

**Notes.** Over the 88 classes Z takes six values: 0.75 for 36 rules, 0.5 for 27, 1 for 12.
Z = 0.75 holds rules 110, 54 and 62, as Wuensche predicts, but also most log-transient
(ordered) rules, so it narrows the search without deciding it (as Hudcová & Mikolov note).
Z = 1 holds both the chaotic rules (30, 45, 90, 105, 150) and the reversible ones (15, 51,
170, 204), told apart by thermodynamic depth (§5.7.5). The Wuensche 1998 input-entropy
variance (§8.2) is not implemented yet.

#### 5.7.5 Thermodynamic depth (`thermodynamic_depth`, TD; 2026-11-02)

**Definition.** Lloyd & Pagels 1988: the depth of a state is −log P(the history that led to
it | the state); its mean over the histories is their entropy. For a deterministic rule
with every configuration equally likely at the start, the histories of a state s at step t
are its t-step ancestors, so D(s) = log2 N_t(s) and the mean depth is
H(S_0 | S_t) = n − H(S_t): the information about the start that the dynamics has erased.

**Implementation.** Exact, over the whole state-transition graph of tiny rings
(`System.transition_map`: each configuration's successor; elementary CA). Counts of the
configurations' images are pushed through the map (`np.bincount`) until the image stops
shrinking, when every configuration is on its cycle and H(S_t) no longer changes (or for
`steps` steps); the result is (n − H(S_t)) / n in bits per cell. On one ring the value
depends on the ring's number theory (rule 90 is nilpotent on 2^k cells, so it erases all
16 bits at n = 16 but one bit on odd rings; rule 18 jumps to 0.99 at n = 15), so it is
averaged over rings of 12 to 17 cells (`min_size`, `size`; 0.5 s per rule). Exact
values tested: 1 for rule 0, 0 for identity (204) and shift (170).

**Notes.** TD peaks at irreversibility, not at complexity: the rules that erase everything
(depth 1) are exactly Wolfram's class 1 (0, 8, 32, 40, 128, 136, 160; 168 at 0.99), depth 0
are the reversible rules (15, 51, 170, 204; and 154, bijective on these rings). Chaotic rules are shallow (30: 0.24,
45: 0.19, 90: 0.28): they keep almost all of their past, as surjective rules must. The
complex and traffic rules sit in the middle (110: 0.47, 54: 0.46, 184: 0.41). Median depth
by transient class: bounded 0.30 (spread 0–1), log 0.47, linear 0.49, polynomial 0.46,
exponential 0.28. Tagged "mixed". Lloyd & Pagels define it for a state with its history
(pattern-level); the mean over all starts makes this a rule-level value.

### 5.8 Deferred

- **Integrated information (IIT; Tononi).** Φ is exponential in system size: only ~3×3
  binary patches are tractable (`pyphi`), and a patch's Φ is not the system's. It is not a
  local measure as the PDF tags it (§2.3); its complexes are regions with causal boundaries
  (§2.4), the idea behind individuation by intervention (§8.1b), which is affordable
  where IIT is not.
- **Assembly index (AT; Cronin, Walker).** The libraries target molecules and strings, and
  the index is exponential in object size. Revisit for small discrete patches, with copy
  number (§8.2). Assembly A = Σ_i e^{a_i}(n_i − 1)/N_T is an ensemble quantity: unique objects
  i with index a_i and copy number n_i among N_T objects.

### 5.9 Statistics for reference (`mass`, `speed`, `change_rate`)

Not complexity measures: plain statistics to read the others against (is the creature
alive, is it moving), tagged *statistic* and skipped by the orientation tests.

- **Total mass** (`mass`): the sum of the frame's cell values, in cells (live cells in
  Life; the mean of the channels for multi-channel Lenia). Glider 5, Orbium 72.7.
- **Moving speed** (`speed`): displacement of the center of mass over the window (33 frames)
  divided by the steps, in cells per step. On a torus a creature straddling the edge has
  its center of mass split between the two sides, so each frame is first centered: as in
  the hand-coded Lenium (`stat.calc_basic_stats`, which rolls the world by the integer
  part of the center every step), but without shifting or storing the frame — its
  coordinates are wrapped to within half a grid of the previous center's integer part,
  the center of mass taken, and the new center carried to the next frame, so the centers
  are unwrapped and their difference is the true displacement. The first frame of a
  window is centered on its mass-weighted circular mean. Life's glider: exactly √2/4 =
  0.354, on 128² and on 16², where it crosses the edge every ~16 steps; Orbium 0.60 (in
  Lenia's units, × T / R: 0.46); Galaxy and empty 0. With several objects it is the
  speed of their common center, and on iid noise the center jumps (0.6–0.8): read it for
  single creatures.
- **Change rate** (`change_rate`): Σ|A(t+1) − A(t)| over the cells, averaged over the window
  (16 frames), in mass per step: motion and change in place alike. Glider 4 (2 births, 2
  deaths a step); Orbium 14.8, Paraptera pedes 55, Tetralapillium 7.5; still and frozen
  patterns 0. The divisor of TE's and PID's normalized versions (§5.10).

### 5.10 Normalized, per-object versions (option "normalized", on by default in the UI)

Many measures are grid averages, so on a lone creature they mostly measure how much of the
grid it covers and how much it changes (Q17). Their normalized versions divide that out, per
object:
(value − value on empty frames) × cells / divisor, where `cells` is the grid's H × W, so
the numerator is the grid's total above an empty frame (a compressor's fixed overhead
removed; 0 for the information measures), and the divisor is the mean mass or change rate
(§5.9) over the measure's own window of raw frames (SC sees only 2-level symbols).

| Measure | Divisor | Unit (grid average → normalized) |
|---|---|---|
| image compression (LZMA, PNG) | mass | bits/cell → bits/mass |
| video compression (LZMA, H.264) | mass | bits/cell/frame → bits/mass/frame |
| Lempel–Ziv complexity | mass | relative to noise → symbols/mass (c · log_L n symbols) |
| effective complexity | mass | bits/cell → bits/mass |
| block entropy | mass | bits/cell → bits/mass |
| statistical complexity | mass | bits/cell → bits/mass |
| transfer entropy | change rate | bits/cell/step → bits/Δmass: bits per unit of mass changed |
| PID | change rate | bits/cell → bits/Δmass |

The ratio is 0 for an empty grid and, with the change rate, when nothing changes (then TE
and PID are 0 as well). Maps stay per cell (`map_unit`).

**Which measures, which divisor** (2026-11-01). Every frame measure on the five creatures
of Q17, 128² and Orbium on 256²: grid averages fall 3.4–3.9× on the 4× larger grid
(LZMA, PNG, video LZMA, LZC, effective complexity, block entropy, PID, SC, TE); novelty,
FID, `iti_local`, fractal dimension (≈ 1×) and logical depth (1.2×) already describe the
creature, and the Lyapunov exponent (1.8×) is a rate. Mass and change rate correlate 0.8
over the five, so the divisor follows the measure's kind and the outliers: static image
measures divided by change rate make the slowly changing Tetralapillium an outlier (3× the
others); PID follows change rate (log correlation 1.00) like TE. Video compression follows
change rate a little better (0.96 against 0.86) but pays for the first frame's still
content, so divided by change it has the same outlier and would spike when a creature
slows down: it is divided by mass. Above an empty frame, compression per mass does not
depend on the grid size (LZMA Orbium 31.16 on 128² and 256²); LZC and block entropy rise
~10 % from 128² to 256², PID ~20 %, for the reason given below.

**Why change rate, not speed × mass** (2026-11-01). TE was first divided by speed × mass,
which matched it across the five creatures (Q17), but a creature that nearly stops while
changing shape in place keeps its TE while its speed falls toward 0: on Tetralapillium
(256², speed 0.002–0.34 cells/step) TE / (speed × mass) peaked at 39× its median (CV 2.0),
spikes in the chart. The change rate counts motion and change in place alike and is 0 only
when TE is: TE / change rate stays within 1.6× its median (CV 0.24), steadier than raw TE
(2.6×, 0.53), and across the creatures TE follows change rate as closely (log correlation
0.98). TE / mass was in between (2.2×, 0.48).

Implemented in the run (`Measure.ratio`, `ratio_unit`; `Run(normalized=True)`, CLI
`--normalized`, the UI's "normalized" checkbox beside the Measures title, sent at Start;
normalized charts say "(normalized)" in their title); the library
and CLI default to the grid averages. Other measures could declare a ratio the same way.

Values (128², step ~300; LZMA, effective complexity and block entropy from the comparison
above): LZMA Orbium 32.1, Paraptera furiosus 16.8, Paraptera pedes 15.6, Tetralapillium
13.1, Florakronium 12.3; effective complexity Orbium 6.8, Tetralapillium 3.0, Paraptera
furiosus 2.6, Florakronium 2.1, Paraptera pedes 1.7; PID Paraptera pedes 3.8, Paraptera
furiosus 3.3, Florakronium 3.3, Orbium 2.7, Tetralapillium 2.4; SC Orbium 20.1, Tetralapillium 22.0, Paraptera furiosus 16.4,
Florakronium 16.1, Paraptera pedes 14.0; TE Orbium 5.94, Paraptera pedes 5.68, Paraptera
furiosus 5.48, Florakronium 5.14, Tetralapillium 3.28 — Tetralapillium changes the most in
ways its neighbours do not predict. On 256² the grid averages drop ~3.5×, the ratios rise
10–27 %: local SC and TE take probabilities over the whole grid, so a
creature's states are rarer on a larger grid (about log2 4 = 2 more bits per creature cell
at 4× the area). Compare creatures at equal grid size.

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
  series; Langton's λ of Life and rule 110; Wuensche's Z on the paper's example and on
  rules 0, 30, 90, 110, 184; thermodynamic depth on small maps and rules 0, 204, 170;
  local maps average to the value; tile maps are zero away from
  structure; parameters from the UI are typed and checked.

## 7. First values

At 128², step 200 unless noted (TE: step 100). Empty and iid noise rows are the baselines.
Lenia creature values are from before 2026-11-01: the creatures ran on the Gaussian shell,
and several (Scutium, Helicium solidus, Paraptera a. l., Kronium,
Hexadentium, Amoebium dentalis) have since been replaced (PLAN.md §4.2).

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
- TL (growth class, 2026-11-01): Life linear (exponent 1.43); min Orbium's Integer Lenia
  log (0.57). Earlier single-size values (64²) are superseded.
- LL: Life 0.269 (exact 0.273); min Orbium 0.84. ECA (exact, from the table): rule 110
  0.625, 30 / 54 / 90 / 184 0.5, 18 and 160 0.25.
- Elementary CA (2026-11-02, rule-space script; the 88 classes in
  `results/eca_rule_space/rules.csv`, figure `assets/figures/eca_rule_space.png`):

  | Rule | Wolfram class | λ | Z | TD bits/cell | TL class |
  |---|---|---|---|---|---|
  | 110 | 4 | 0.625 | 0.75 | 0.47 | polynomial |
  | 54 | 4 | 0.5 | 0.75 | 0.46 | polynomial |
  | 30 | 3 | 0.5 | 1 | 0.24 | exponential |
  | 18 | 3 | 0.25 | 0.5 | 0.65 | linear |
  | 90 | 3 | 0.5 | 1 | 0.28 | exponential |
  | 184 | 2 | 0.5 | 0.5 | 0.41 | linear |
  | 108 | 2 | 0.5 | 0.75 | 0.23 | bounded |
  | 232 | 2 | 0.5 | 0.5 | 0.40 | log |
  | 160 | 1 | 0.25 | 0.5 | 0.99 | log |

  Frame measures on the space-time diagram (64², step 100): block entropy, TE, SC, PID
  and FID rank rules 110 and 54 above the class-2 rules (TE 0.43 / 0.39 against 0.24 for
  184, 0.22 for 108); compression, LZC and block entropy put rule 30 on top, as for noise.
- FI (random patterns of the pattern's size, 512 samples, ceiling 9 bits; 2026-11-01): at
  the ceiling Orbium, min Orbium, Scutium, Helicium, Hydrogeminium, Paraptera a. l.,
  Hexadentium; Gyrorbium and Paraptera c. f. 7.4; Tetralapillium 2.7; Kronium 1.2;
  Amoebium 0.2; Aquarium unit 2.2, self-replicating Aquarium 3.0 (256²); Life glider 1.1,
  gun 1.3, Galaxy 0.5, R-pentomino 0; soups 0–0.3.

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

**What counts as complex** (Bert, 2026-10-31): a still structure can be highly complex.
Information flows inside its own boundary, so it does not disintegrate on its own
(self-organizing), and it tolerates disruption from other structures or noise
(self-repairing, self-defending). Examples: the Aquarium (static and moving modes) and
Orbium stay stable when they collide with other objects, and can form a **colony** in which
members stay individuals and can leave at any time. Less complex: Scutium and the Life
glider, which disintegrate on collision; goo, which is not made of individuals.

**Consequence: observation is not enough.** A self-maintaining stationary structure and a
frozen pattern give the same frames, so every observational measure here (TE, SC,
compression, ITI) rates them alike. The difference shows only **under intervention**:
perturb a region and follow how the perturbation spreads within it, is corrected, and stays
inside its boundary — causal rather than observational information flow (Ay & Polani 2008;
IIT is interventional for the same reason). The twin-run machinery of the Lyapunov exponent
(`System.perturb`) already supports this.

**Proposed design** (to prototype in M3b): a **complexity field C(x, y, t)** on coarse tiles
or segmented regions over a sliding window, updated every few steps (blur and delay make
the estimates reliable), from three components, each scaled between its value on empty and
on matched noise so values compare across systems:
- **S, structure**: spatial structure that noise with the same statistics cannot explain,
  at several scales (1, 2, 4, 8 cells) — the noise-band coarse-graining of EC, or local
  statistical complexity (§8.1).
- **M, self-maintenance** (interventional): perturb a region at several strengths, up to
  collision-like ones, and measure over the following steps
  - *internal flow*: the perturbation spreads within the structure (its parts are coupled),
  - *repair*: it then decays back to the unperturbed twin,
  - *containment*: it does not leak beyond the boundary,
  - *tolerance*: the largest perturbation still repaired.
  Frozen noise has none of these; chaotic goo spreads damage without repairing it; empty
  space repairs trivially but has S = 0.
- **I, individuality**: the region is a bounded unit with a persistent identity among others,
  also inside a colony (segmentation and tracking; ITI, §5.4). Separates goo from colonies.
- **C = S × M × I**. Motion is optional: a gliding Orbium and a stationary Aquarium ring can
  both score high. Observed dynamics (local TE) remains available as a separate component.

**Finding the individuals: a loop of observation and intervention** (Q13). Observational
segmentation (threshold and torus components now; later watershed splitting, tracking over
time with merge and split events, motion coherence, channel signatures) proposes candidates
in real time, but cannot tell touching colony members from one blob, or goo from a unit.
Interventions refine them: perturb a cell and record which cells' futures change (its
causal reach); an individual is a set of cells that influence each other strongly and the
outside weakly — a disturbance spreads, is repaired, and stops at the boundary. Touching
colony members separate; goo has no such closed sets. For differentiable systems (Lenia),
`jax.jvp` gives how one disturbance spreads everywhere and `jax.vjp` which past cells
influenced a region, each at about the cost of a few steps, batched over many probes;
large (collision-like) disturbances and discrete systems need finite twin runs. Confirmed,
tracked individuals are what M and the collision tests act on.

**Observation and intervention together** (Q14). Observation runs always and in real time:
detection, tracking, natural encounter statistics, S and observed dynamics. Interventions
run in the background or offline on the top candidates (hot regions): individuation, M,
tolerance curves. In busy scenes, natural collisions serve as free validation of what the
interventions predict. Perturbation protocols (kind, size, place) are fixed and documented,
since results depend on them.

**Outputs**: the C heatmap; **hot regions** (C above a threshold, connected in space and
time: position, extent, peak, integral; boxes in the UI, exportable); **run summaries**
(mean, maximum, 95th percentile over space and time) to compare conditions, batched over
seeds and parameters (M5 experiments).

**Validation**, with **collision experiments** as the acceptance test: head-on and glancing
collisions of two units, a unit into a patch of noise, a unit into an existing colony.
Survival = the same number of units afterwards, each within ±20 % of its mass, separable
again within a few hundred steps; identity tracked through the encounter. Continuous
Orbium, as agreed. A separate **Collider Lab** web app (Bert) may collect collision data
with human labels (individual, colony, goo): pattern pair and parameters, angle, offset,
speed, outcome (survived, merged, split, died, became goo), units and masses before and
after — the ground truth for whether high C matches what a person calls complex. Expected ranking, lowest to highest: empty,
noise; frozen noise; Life still lifes, the glider, Scutium (disintegrate on collision); goo
(no individuals); Orbium, the Aquarium unit, Aquarium colonies in static and moving modes
(survive collisions, members stay individuals).

### 8.1c Collision experiments (implemented 2026-11-01)

`complexity.collisions`, run by `scripts/collisions.py` (records in
`results/collisions/collisions.jsonl`, one JSON object per run; `scripts/relabel_collisions.py`
relabels them without rerunning). A sample for the goal measure, not a survey: the
full-fledged exploration is the Collider Lab, which can import this module (scene builder,
outcome labels) and share the record format.

- **Unit.** The pattern settles alone (300 steps), its object (the K ∗ A basin, §5.4) is
  cropped with a margin of R/2, and its heading and speed are measured over 64 steps.
- **Scenes.** Copies are rotated (by interpolation; Life in quarter turns, keeping its own
  heading) onto courses that meet at the grid's centre: `pair` (two copies, headings Δ
  apart: 180° head-on, 135°, 90°, 60°, 30° (Life 180°, 90°); impact offset 0, 0.25, 0.5
  unit sizes; the second copy mirrored or not), `noise` (the unit into an iid patch of its own size) and `alone`
  (the control). The grid is large enough that no course wraps around the torus (at
  least 256, 2 × start distance + 3 sizes); a 256 grid for the 73-cell Triscutium made
  the copies overlap through the edge before they met.
- **Run.** Until they meet (start distance / speed) and 400 steps more; objects detected
  every 2 steps; one `Tracker` per thrown unit (a takeover skips creatures another
  tracker holds); `contact` when the two were one object or their centres within a unit
  size.
- **Labels**, from the final objects, not the trackers (identity through a merge and a
  split is ambiguous): unit-sized = ±25 % of a unit's mass, unit-like = 0.4-2×. *missed*
  (no contact), *fused* (two units moving as one: objects with one velocity, within 1.5
  sizes, together 1.6-2.4 units' mass and 80 % of what is left), *lost one* (one
  unit-sized object left, nothing larger), *died* (under half the mass and no unit left;
  for noise, the unit under half its mass), *divided* (more unit-like objects than thrown, holding 70 % of the mass:
  replication), *goo* (more than twice the mass otherwise), *survived* (the thrown units
  unit-sized again, nothing else of half a unit's mass), *merged* (fewer unit-sized
  objects, a large one), *changed* (anything else). The unit-like range matters: a lone
  Aquarium unit settles at 264 cells, units among others at ~140.

| Unit | 30 collisions with a copy (Life 12) | Into noise | Alone |
|---|---|---|---|
| Orbium | survived 17, lost one 6, died 5, fused 2 | survived | survived |
| Triscutium valvatus laterale | goo 30: runaway growth, ~90 objects of ~3000 cells, 70× the mass | goo | survived |
| Aquarium unit (VT049W) | divided 30: replicates on contact into 4-30 units of ~145 cells | survived | survived |
| Life glider | died 6, divided 3 (still debris), lost one 2, merged 1 | divided (debris) | survived |

Orbium by heading difference: 180° survived 4, died 2; 135° lost one 3, survived 2, died
1; 90° survived 4, lost one 2; 60° fused 2, died 2, lost one 1, survived 1; 30° survived
6 (glancing, they part again).

**Synorbium** (Bert, 2026-11-01): two Orbia can fuse into Synorbium, two Orbia conjoined
and moving in one direction, which disrupted can split into two Orbia again. The
collisions find it at 60° without offset, in both chiralities: three potential basins
(its bumps) of 43-56 cells moving with one velocity, 147 cells in all, stable for at
least 800 steps after meeting. Before the near-parallel headings were added (60°, 30°)
none of the 18 geometries produced it, and earlier labels would have called it divided.
In a wider scan of near-parallel courses (20-60°, offsets to 0.75) one Orbium collision
also turned into goo (60°, offset 0.75, mirrored): Orbium's rule can grow goo too.

Readings: the controls survive, so the harness does not break units. The ranking expected
above holds in its order: the Aquarium unit survives every encounter by reproducing, and
each product is again an individual; Orbium survives most glancing collisions and noise,
loses one or both units at steeper angles, and sometimes fuses into a Synorbium; Triscutium (the Scutium-like unit) turns into goo at any touch, even
of noise; the glider dies or leaves debris. Caveats: an outcome depends on the phase in
which the units meet (Orbium at 90° survives or loses a unit with a few more settling
steps), so labels are statistics over geometries, not per-run truths; in Life, still
debris is unit-like by mass and counts as divided; "survived" does not yet check that the
units separate and keep moving. Still to do: static Aquarium modes.

**A unit into a colony** (`collisions.intrude`, `scripts/collisions.py --colony`, records
in `results/collisions/intrusions.jsonl`). A colony is 6 copies packed around the centre
(random turns, centres 1.1 sizes apart), settled 200 steps; a fresh copy is thrown at its
centre from 4 directions × 2 offsets (a discrete unit starts on the line of the heading
its quarter turns give it), for 2 layouts. Colonies change by themselves — the Aquarium's
chaotically, by replication — so each is also run without the intruder, as is and with 4
slight perturbations (`System.perturb`), and the run is labelled against that natural
range: *missed* (never within the colony's radius), *goo* / *collapsed* (mass over 2× /
under half the controls'), *divided* / *disrupted* (unit-like objects over 1.15 × the most
+ 2 / under 0.85 × the fewest in the controls), otherwise *joined* (the tracked intruder
still unit-like) or *absorbed* (not). Triscutium is left out: any contact makes it goo.

| Unit | Into 6 copies (16 runs) | Units at the end without intruder |
|---|---|---|
| Aquarium unit | joined 11, disrupted 2, absorbed 1, missed 2 | 28-54 (controls disagree: chaotic replication) |
| Orbium | joined 14, missed 2 | 3-4 (all controls agree) |
| Life glider | joined 16 | 1-12 |

Only the Aquarium forms a colony that stays together, and its intruders mostly become
members within the colony's natural range (joined): the individuality-in-a-colony of §8.1.
Orbia and gliders move on, so their "colony" is a crowd that scatters, and joined means
the intruder survived the crowd; Orbium's deterministic crowd shows it cleanly (one more
Orbium than the control, or one exchanged for a member). Two first versions were wrong and
are fixed: a single control could not tell an intruder's effect from the Aquarium's own
variation, and the glider, turning in quarter turns, flew past the colony while being
labelled joined (hence *missed*).

**A goo, found by collision.** Triscutium valvatus laterale's "goo" is a disc of
labyrinthine stripes that grows from the point of contact until it fills the grid:
self-organizing, of one level, with no individuals — the goo reference case of §8.1 that
the search near the Aquarium's parameters was meant to find. Lone Triscutium is stable;
any touch, even of noise, seeds it. `scripts/collisions.py --gifs` saves one GIF per unit
(the 90° collision without offset, and Orbium head-on) in `results/collisions/`.

### 8.1d A first prototype of C = S × M × I: 🧫 Autopoiesis (2026-11-01)

`complexity.goal.complexity_field(system, pattern, state)`; in the measure list as
**🧫 Autopoiesis (goal measure C)** (`autopoiesis`: the largest or mean C over objects, its
map the field C(x, y), computed in a background thread every 10 ticks; the simulation
does not wait). The name: autopoiesis (Maturana & Varela) is a
bounded unit that produces and maintains itself, which is what S × M × I scores.
`scripts/goal_ranking.py` ranks the reference scenes (`--figure` writes
[goal_field.png](assets/figures/goal_field.png): each scene beside its field). Per object
(the K ∗ A basins of §5.4), its value painted onto its cells:

- **S, structure**: 1 − bits(object) / bits(its cells shuffled within its outline), LZMA,
  mean over scales 1 and 2: the arrangement that matched noise (the same values) lacks.
- **M, self-maintenance** = ceiling × repair × containment × contact:
  - *repair*: twin runs with the object's living cells hit by Gaussian noise σ = 0.1-0.4
    (discrete: cells reset at random with probability 0.1-0.3), 60 steps on; the share of
    the object's similarity (aligned cross-correlation, so a shift is no damage) that the
    hit took and that comes back, comparing the object with itself found again in both
    runs; averaged over the strengths that changed it (a tolerance curve); a hit after
    which it grows past 1.5× counts as unrepaired;
  - *ceiling*: the similarity a negligible hit (σ = 0.02) ends at — a continuous object
    never returns in exactly the same phase, a chaotic colony drifts (1 for discrete);
    repair is measured up to it, and it counts as stability; a hit that dents the object
    less than that drift is repaired as far as it gets back to the ceiling;
  - *containment*: the share of the runs' difference near the object's path;
  - *contact*: a patch of full-strength noise of the object's size beside it (the "unit
    into noise" collision), judged after 180 steps: 0 if the run then holds more than
    twice the unperturbed mass plus the noise added (goo) or the object is gone, else how
    far it is still itself.
- **I, individuality**: the share of a ring around the object that lies outside every
  object (goo's basins tile the space), over 1 + area / (5 % of the grid), times
  1.5 / growth if it grows more than 1.5× in 60 steps.

Each hit is drawn 3 times (`trials`, in the ranking and the UI): where the noise lands
matters, and single draws moved Orbium's C between 0.009 and 0.059. All twin runs of all
objects are prepared first and run 16 at a time under `vmap` (`System.replace_cells`
swaps the cells into one shared state, so the rule's kernels are not copied per twin):
a field takes 0.4 s for Orbium, 1 s for the Aquarium soliton and 12 s for a 25-unit
colony with 3 draws, 7× faster than one twin after another (89 s); most of what is left
is detecting objects in each twin on the host. No autograd: every hit is a finite twin
run (strong hits and the contact patch are far from linear, and Life is discrete).

| Scene (256² unless noted) | C max | C mean | S | M | I |
|---|---|---|---|---|---|
| empty, iid noise, frozen noise (128²) | 0 | 0 | 0 | ≤ 0.03 | ≤ 0.05 |
| goo (Triscutium labyrinth) | 0 | 0 | 0.30 | 0.94 | 0 |
| Life glider (64²) | 0 | 0 | 0.03 | 0 | 0.90 |
| Life Gosper gun (128²) | 0.008 | 0.001 | 0 | 0.27 | 0.84 |
| Orbium | 0.018 | 0.018 | 0.14 | 0.15 | 0.87 |
| Aquarium soliton | 0.046 | 0.046 | 0.20 | 0.31 | 0.75 |
| Aquarium colony (8 units, step 600) | 0.087 | 0.059 | 0.19 | 0.52 | 0.66 |

(S, M, I are medians over objects.) The expected order holds, the colony on top. Each
component rules out a reference case: S the noise, M the glider and frozen noise (no
repair), I the goo (it maintains itself locally, M 0.94, but has no individuals), contact
the Triscutium unit.

**Against the collisions** (§8.1c), each unit alone, settled, on 256² (one draw):

| Unit | C | Collisions in which its units were kept or multiplied |
|---|---|---|
| Aquarium soliton | 0.067 | 100 % (divides) |
| Orbium | 0.059 | 63 % |
| Life glider | 0 | 25 % (debris) |
| Triscutium valvatus laterale | 0 | 0 % (goo) |

The same order. Before the contact test, a lone Triscutium scored 0.076, above Orbium:
it withstands noise inside but turns to goo on any contact, and only the contact test
sees that.

How it got there, each step a lesson: erasing a disk (radius 2 at Orbium's centre, 15 %
of its mass) kills Orbium, so holes probe only the strong end; one strength was either
too slight to read or lethal, hence the tolerance curve; repair against 1 failed for
every continuous creature, hence the ceiling; a window around the object let chaotic
neighbours in, hence the object against itself; isolation first counted the gaps between
goo's stripes as background, hence "outside every object"; colony members got M = 0
because every hit was skipped as slighter than the colony's chaotic drift, hence repair
up to the ceiling for slight hits; a ring of noise does not trigger Triscutium, a patch
of its size does, at once (13× its mass in 60 steps), but it takes 180 steps for the
patch to clear around units that survive it; and a plain dilation lost objects crossing
the grid's edge, so containment collapsed at random — the neighbourhoods now wrap.

Open: (1) S is weak (0.14-0.30) and only spatial; (2) no calibration between empty and
matched noise yet, as §8.1b proposes; (3) cost: per object 5 hits × `trials` twin runs of
60 steps and `trials` contact runs of 180, batched (1 s for a unit, 12 s for a 25-unit
colony), and the field describes the state it started from, so on the heatmap moving
creatures run ahead of their C patch by the time a field takes; (4) grids must be large
enough for the contact test (Orbium at 128² meets the noise's remains or itself: C = 0);
(5) on the GPU runs are not bit-reproducible, so values move by ~10 % between runs.

**Assessment, and M3b paused** (Bert, 2026-11-02, from watching it in the UI). Autopoiesis
does what was asked — it follows the requirements of §8.1b — but it is less useful than
hoped: slow on scenes of many creatures (the Aquarium with 8 units grows to dozens; each
needs its own twin runs and object matching), and complicated, because it rests on
object detection (§5.4) and on interventions run in real time (perturbations and a
collision-like contact test). The problem is in the idea, not the implementation: a
measure that must find individuals and experiment on each of them inherits both costs.
It stays in the measure list as the reference prototype.

**Next idea (deferred to a later session): a "physical" measure**, with no object
detection and no real-time experiments: a dense field read from the dynamics themselves,
closer to the autograd idea of §8.1b and Q13. For a differentiable system (Lenia),
`jax.jvp` / `jax.vjp` give the linear response of the next steps to a change of any cell,
at the cost of a few steps and batched over probes; from it, per cell or region: how far a
disturbance spreads (causal reach, its boundary), whether it decays (local Lyapunov /
stable part of the spectrum, Yevenko et al. 2025, §5.6) and whether it stays inside a
coherent region (containment as a property of the Jacobian, not of detected objects).
Self-maintenance would show as regions where disturbances decay and stay contained;
individuality as the boundaries of such regions, found rather than detected. Discrete
systems (Life) would need finite differences or another route.

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
- **Effective information and causal emergence** (Hoel 2013): interventional — set states
  uniformly at random and measure the information carried to the next step, across scales;
  relevant to the multi-scale part of the goal.
- **Causal information flow** (Ay & Polani 2008): the formal, interventional counterpart
  of transfer entropy; the basis for M's "internal flow".
- **Lyapunov spectrum and Kaplan–Yorke dimension** (Yevenko et al. 2025): the whole spectrum
  by `jvp` + QR, D_KY as a rule- or trajectory-level complexity; restricted to a tracked
  region, the stable part (how fast and how completely perturbations decay) and the
  covariant vectors (where they go) are an exact small-perturbation form of M's repair and
  containment.
- **Input-entropy variance** (Wuensche 1998): entropy of the rule-table lookup frequencies
  per step, and its variance over time; high for rules with gliders. Needs only the run and a
  lookup table (Life: 512 or 18 totalistic entries; Integer Lenia: potential values), so
  available without elementary CA.
- **Excess entropy and the complexity–entropy diagram** (Crutchfield & Feldman): E from the
  convergence of block entropies, cheap and structural; plotting E or C_μ against entropy
  density compares systems without their parameters.
- **Assembly for colonies** (Sharma et al. 2023): an Aquarium colony holds dozens of copies
  of one many-step unit — high copy number × index is AT's signature of selection; with
  tracked units as objects, assembly needs no molecular library.
- **ITI boundary search** (Krakauer et al. 2020): grow a candidate individual by
  environment cells while nC falls, with a cost on size since A* and A never decrease;
  an observational counterpart to individuation by intervention.

## 9. Questions and answers

Collected from the design conversation (2026-10-30 to 2026-10-31).

**Q1. Most measures rate iid noise highest, but effective complexity, logical depth and
transfer entropy rate it low, often near 0. Should those be inverted, to keep "higher =
more complex"?**
No. They measure structured complexity, and the others measure randomness (§2.1). The
principle is "higher = more complex *by the measure's definition*". Inverting a measure that
is low at both ends would make empty grids and noise score highest and structure lowest,
which fits neither meaning. Negation fixes only measures that run the wrong way along the
order-to-disorder axis. Instead, the UI tags each measure randomness or structural.

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
Each uses the core it was found with (since 2026-11-01). The hand-coded Lenium, Chan's
Lenia tutorial and Leniabreeder all build multi-channel kernels as Gaussian rings
`bell(D mod 1, 0.5, 0.15) · b`, so the Aquarium and Lenium's Orbiums are Gaussian. The
catalogue creatures were found on the bump core (`kn = 1`) and run on it (PLAN.md §4.2); until 2026-11-01 they ran on the Gaussian shell.

**Q11. Is the granularity right — spatial, temporal, and subject (pattern / rule /
system)?**
Re-derived from the literature on 2026-10-31, without the PDF or these notes (§2.3). Most
tags agreed. Fixed: FID is rule-level; block entropy is local (and system-level); IIT is not
local; the Lyapunov exponent is rule-level; functional information is ensemble-level. The
deeper finding is that information measures are properties of distributions: estimated from
one run they describe the run, not the rule, which one granularity tag cannot say — so each
measure now also states its subject and what it is estimated from.

**Q12. Should a still structure have low complexity, since nothing changes?**
No (Bert, 2026-10-31). A stable structure maintains itself: information flows inside its
boundary, it repairs damage and resists disruption, and it can join a colony while staying
an individual (Aquarium, Orbium); objects that break on collision (Scutium, the Life glider)
and goo without individuals are less complex. Observational measures cannot see this in a
structure that does not change, so the goal measure needs an interventional component —
perturb and follow the repair (§8.1b).

**Q13. To run collision and intervention experiments the objects must first be found. How —
or do interventions themselves reveal the individuals?**
Both, in a loop (§8.1b). Observational segmentation is cheap and real time but fails for
touching colony members and for goo. Interventions define the boundary causally: the set of
cells within which a disturbance spreads and is repaired, and beyond which it does not go.
Observation proposes candidates; interventions confirm, split or merge them in the
background; tracking keeps their identity. Autodiff makes small-perturbation responses
cheap in Lenia.

**Q14. Is observing a long enough simulation enough to find individuals and their
complexity, as a person would, or are interventions better?**
Observation can suffice when the scene produces its own natural experiments often enough
(busy colonies) and the observer brings priors, as people do. But a stable structure is
quiet — its repair shows only when disturbed; deterministic, tightly coupled parts confuse
inference from statistics alone; and observation never visits the limits of tolerance.
Interventions isolate cause and effect, probe any size and place, run in parallel, and are
far more efficient here because the simulator is ours, at the price of a fixed protocol and
of states that may not occur naturally. Hence a hybrid: observation always on,
interventions on the top candidates.

**Q15. Does any documented measure use an interventional mechanism like the goal
measure's, or is it new or specific to Lenia?**
The ingredients have precedents. In §3: the Lyapunov exponent uses the same twin-run
mechanism but rewards divergence (chaos), where M rewards repair and containment; IIT is
interventional and defines a boundary by integration, but needs the full transition table
and has no repair or tolerance; FID's uniform inputs are an intervention on the rule, and
FI's random patterns a comparison with other configurations, not perturbations of a
structure. Beyond §3: Beer's perturbation analysis of
the Life glider as an autopoietic unit (closest to the individuation loop); Hoel's
interventional effective information; Ay & Polani's causal information flow; Sensorimotor
Lenia (Hamon et al. 2024), which searched for Lenia creatures robust to obstacles and
perturbations as a sign of agency; ecological resilience and basins of attraction. What is
specific here: a spatial, real-time field rather than one number; the combination of
structure × self-maintenance (spread, repair, containment, tolerance) × individuality;
calibration against empty and matched noise; individuation by intervention in a loop with
tracking; and autodiff for exact small-perturbation responses. Not Lenia-specific in
concept — any simulator that can be perturbed and rerun works (Life with finite flips, as
Beer did); Lenia is special in being differentiable and in having self-repairing,
colony-forming individuals. Checked against the papers in `assets/` (2026-10-31): IIT is
interventional and its complexes are causally bounded regions (Albantakis et al. 2023);
Yevenko et al. 2025 tie Lenia's robustness to its negative Lyapunov exponents; Zenil's
algorithmic information dynamics perturbs parts and follows the change in complexity. Not
yet checked (papers not in `assets/`): Beer, Hoel, Ay & Polani, Hamon et al.

**Q16. Do the papers confirm the definitions and implementations here?**
Mostly (§2.4, checked 2026-10-31 against the 22 papers in `assets/`). One error: FID's
synergy was described as possibly negative; it cannot be, by construction, and Bohm et
al.'s Game of Life numbers are now a test. Two implementations were variants of the
paper's measure — transient length at one grid size instead of its growth class, and
functional information over rule parameters with a pass-or-fail function instead of over
patterns with a graded one — and were replaced by the papers' measures (2026-11-01, §5.7). Refinements: IIT's complexes are regions; AT's assembly is ensemble-level. New
leads for the goal measure: the Lyapunov spectrum (repair), IIT's complexes and ITI's
boundary search (individuation), assembly with copy number (colonies), input-entropy
variance and excess entropy.

**Q17. Among the structural measures, statistical complexity seems to follow total mass,
transfer entropy moving speed, and individuality (features) something else. Do the
definitions explain it?** (Bert, 2026-11-01)
Mostly yes. Measured on Orbium, Paraptera arcus pedes, Paraptera cavus furiosus,
Tetralapillium metamorpha and Florakronium arcus (128², 50 measurements over 500 steps):
- SC ranks the creatures exactly as mass does (log correlation 0.99). Local SC averages
  −log2 P(causal state) over all cells; the empty background is one common state that adds
  almost nothing, so the value is ≈ the creature's share of the grid × its cells' mean
  surprisal. On 256² Orbium's SC falls from 0.090 to 0.026.
- TE follows speed × mass, not speed alone: same rank and log correlation 1.00 with
  speed × mass, but only 0.4 rank correlation with speed (Orbium is fastest, yet second
  lowest). A cell's next state depends on its neighbour's when a pattern moves into it,
  and the cells changing per step grow with speed × size. Over time, curves move together
  clearly only for Tetralapillium (TE vs speed 0.90), whose speed varies by 76 %; the
  others keep mass and speed within 0.1–3 %.
- `iti_features` follows neither (vs speed: −0.7): with one creature its environment is
  empty and it measures how predictable the creature's mass is from step to step. Steady
  gliders score ~0 (Orbium 0.01, Paraptera pedes 0.09), creatures whose mass varies
  smoothly ~1–1.3. Interesting, but not individuality in Krakauer's sense; that needs
  other creatures (§5.4).
Hence the normalized versions (§5.10), which keep what is specific to each creature. TE's
divisor became the change rate rather than speed × mass, which spiked when a creature slowed
down while still changing shape.

**Q18. To find creatures, threshold the frame A and flood-fill, or use the neighbour sum
K ∗ A, which is how the Lenia model "sees" the world?** (Bert, 2026-11-01)
Use K ∗ A, but not by thresholding it. Thresholding A (the earlier method) splits large
creatures into fragments (Amoebium 7, Urium 4, Tetralapillium 2 patches). Thresholding
K ∗ A keeps creatures whole, but K ∗ A is A blurred over the kernel radius R, so colony
members closer than R share one hill: 12–16 patches where 30–80 Aquarium units are
visible. What separates them is that each unit is its own *peak*: flooding from the peaks
(catchment basins) splits the colony, and merging neighbouring basins whose saddle is high
relative to the lower peak rejoins the bumps of one creature. At a saddle ratio of 0.8:
Aquarium 24 of 24 units (step 500), 31 of 32, 52 of 68 (step 2000, touching units merged),
single creatures whole except Paraptera furiosus (2), Hydrogeminium (2), Urium (4) and
Amoebium (5). The two basins of Paraptera and Hydrogeminium are right, not over-splitting:
these creatures are composites of two units (Paraptera of two Orbium- or Scutium-like
units; Bert, 2026-11-01), and the basins find them — evidence that the basins of K ∗ A
pick out the model's own units. Whether Urium's and Amoebium's parts are units too is
open. 0.75 keeps all of them whole but merges more colony members and made the tracked
target flicker between one and two units. No single ratio does both: in the
model's own view, touching units do share one potential hill. Gating merges by the
connectivity of A was tried and made it worse. Figures (`scripts/detection_figures.py`):
[detection.png](assets/figures/detection.png) (A > 0.1, K ∗ A > 0.1 and K ∗ A basins, per
case) and [tracking.png](assets/figures/tracking.png) (a target unit followed through the
Aquarium scene, with S and E). In the UI, the heatmap list has the neighbour sum and the
detected objects (the target in its own colour) for every run.

**Q19. Is the goal measure useful as built (Autopoiesis, C = S × M × I)?** (Bert,
2026-11-02)
It meets the requirements and ranks the reference scenes and the collision outcomes in
the expected order (§8.1d), but in use it is slow for many creatures and complicated: it
needs object detection and real-time interventions on each object. That is a property of
the idea. M3b is paused; a more "physical" measure, a dense linear-response field from
autograd without objects or experiments, is the next idea (§8.1d), for a later session.

## 10. Open questions

- Multi-scale structural complexity: which base measure, which combination across scales
  (minimum, product, area under the profile), and how to set each scale's noise baseline.
- `iti_features`: richer features than mass (speed, shape) for S and E; detection that
  keeps touching colony members apart; are Urium's 4 and Amoebium's 5 basins units, as
  Paraptera's and Hydrogeminium's 2 are (Q18)? following a
  target through divisions and merges rather than taking the larger part.
- Does H.264 beat video LZMA for smoother (larger-R) creatures or longer windows?
- Transfer entropy for smooth fields: offset ≈ R/4, and history length k > 1.
- Validate statistical complexity against `emic` on 1-D series.
- Logical depth: does dividing time by compressed size make it reliably structural?
- Estimate causal states and probabilities from an ensemble of runs of the rule, then read
  local values on one run (§2.3), instead of from one 16-frame window.
- Functional information: a function with shape or motion (not only persistence), and
  enough samples to lift creatures off the ceiling (§5.7.3).
- Lyapunov spectrum by `jvp` + QR (§5.6): does the clipped Lenia give stable spectra, and do
  the stable exponents of a tracked creature separate Orbium and the Aquarium (repair)
  from Scutium (breaks)?
