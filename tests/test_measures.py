import jax
import jax.numpy as jnp
import numpy as np
import pytest

from complexity import registry
from complexity.measures.algorithmic import lz76, regular_part
from complexity.measures.base import Context, discretize
from complexity.measures.causal import causal_states
from complexity.measures.decomposition import fid, pid
from complexity.measures.geometry import box_counts
from complexity.measures.individuality import basins, decompose, detect, torus_label, track_features
from complexity.measures.information import block_entropy, transfer_entropy
from complexity.measures.rule import growth_class, transient
from complexity.measures.statistics import centers
from complexity.session import Run, RunSpec


def measured(system: str, pattern: str, steps: int = 20, size=(64, 64), **params) -> dict:
    """Every measure's value after `steps` steps (past every window)."""
    run = Run(RunSpec(system, pattern, size), list(registry.measures()), params)
    run.step(steps)
    return run.measure()[0]


@pytest.mark.parametrize("levels", ["binary", "13", "continuous"])
def test_orientation_noise_scores_at_least_empty(levels):
    """MEASURES.md §6: higher = more complex, so iid noise never scores below an empty grid."""
    empty, noise = measured("baseline", "empty"), measured("baseline", f"iid_noise_{levels}")
    for key, cls in registry.measures().items():
        if cls.peaks_at not in ("mixed", "none") and key in noise:  # skips measures for other systems
            assert noise[key] >= empty[key], key


def test_structure_peaked_measures_rate_a_glider_stream_above_noise_and_empty():
    stream = measured("life", "gosper_glider_gun", steps=100, size=(128, 128))
    noise = measured("baseline", "iid_noise_binary", size=(128, 128))
    empty = measured("baseline", "empty", size=(128, 128))
    # iti_features tracks one creature: in a glider stream a glider, whose mass is constant,
    # so A* = I(S'; S) <= H(S) = 0 by definition (MEASURES §5.4); tested on the Aquarium instead
    peaked = [k for k, cls in registry.measures().items()
              if cls.peaks_at == "structure" and k in stream and k in noise and k != "iti_features"]
    assert peaked
    for key in peaked:
        assert stream[key] > max(noise[key], empty[key]), key


def test_compression_orders_uniform_below_noise():
    rng = np.random.default_rng(0)
    uniform = [np.zeros((64, 64), np.float32)] * 16
    noise = [rng.random((64, 64), dtype=np.float32) for _ in range(16)]
    for key, cls in registry.measures().items():
        if cls.__module__ != "complexity.measures.compression":
            continue
        m = cls()
        assert m.compute(uniform[-m.window :]) < 0.5 < m.compute(noise[-m.window :]), key


@pytest.mark.parametrize("levels", ["binary", "13", "continuous"])
def test_transfer_entropy_on_baselines(levels):
    """MEASURES.md §6: 0 for empty and frozen noise, ~0 for iid noise after bias correction."""
    assert measured("baseline", "empty")["transfer_entropy"] == 0
    assert measured("baseline", f"frozen_noise_{levels}")["transfer_entropy"] < 1e-5  # float rounding
    assert measured("baseline", f"iid_noise_{levels}")["transfer_entropy"] < 1e-3


def test_transfer_entropy_bias_correction_removes_the_iid_bias():
    rng = np.random.default_rng(0)
    L = 13
    symbols = jnp.asarray(rng.integers(0, L, (16, 64, 64)), jnp.int32)
    raw = float(transfer_entropy(symbols, 16, L, 1, False))
    n = 15 * 64 * 64
    assert raw == pytest.approx((L - 1) ** 2 * L / (2 * n * np.log(2)), rel=0.1)
    assert float(transfer_entropy(symbols, 16, L, 1, True)) < 0.1 * raw


def test_transfer_entropy_detects_a_copied_neighbour():
    """A field that shifts one cell per step: each cell's next state is its neighbour's
    current one, so TE from that neighbour is H(X | Y) = 1 bit for binary noise; the other
    three directions add nothing."""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 2, (64, 64))
    symbols = jnp.asarray(np.stack([np.roll(frame, t, 1) for t in range(16)]), jnp.int32)
    assert float(transfer_entropy(symbols, 16, 2, 1)) == pytest.approx(1 / 4, abs=0.01)


def test_transfer_entropy_on_a_filling_window_ignores_the_padding():
    rng = np.random.default_rng(0)
    frames = jnp.asarray(rng.random((5, 32, 32)), jnp.float32)
    symbols = discretize(frames, 4)
    full = registry.make_measure("transfer_entropy", window=5).compute(symbols, Context(n_levels=4))
    padded = registry.make_measure("transfer_entropy").compute(symbols, Context(n_levels=4))
    assert padded == pytest.approx(full, abs=1e-6)


def test_measure_parameters():
    m = registry.make_measure("transfer_entropy", offset=3, n_levels=4)
    assert m.config() == {"window": 16, "offset": 3, "n_levels": 4, "correct_bias": True}
    assert m.levels(None) == 4 and registry.make_measure("transfer_entropy").levels(13) == 13
    with pytest.raises(ValueError):
        registry.make_measure("lzma", window=4)


def test_run_flags_measures_whose_window_is_filling():
    run = Run(RunSpec("baseline", "iid_noise_binary", (32, 32)), ["lzma", "transfer_entropy"])
    assert run.warming_up() == ["transfer_entropy"]
    run.step(15)
    assert run.warming_up() == []


def test_video_compression_rewards_repetition():
    rng = np.random.default_rng(0)
    still = [rng.random((64, 64), dtype=np.float32)] * 16
    fresh = [rng.random((64, 64), dtype=np.float32) for _ in range(16)]
    for key in ("video_lzma", "video_h264"):
        m = registry.make_measure(key)
        assert m.compute(still) < 0.5 * m.compute(fresh), key


def test_run_measures_every_registered_pair():
    for system, cls in registry.systems().items():
        run = Run(RunSpec(system, next(iter(cls.patterns)), (64, 64)), list(registry.measures()))
        run.step(3)
        values, _ = run.measure()
        assert set(values) | set(run.skipped) == set(registry.measures())
        assert all(np.isfinite(list(values.values())))


def test_novelty_separates_static_from_fresh_frames():
    rng = np.random.default_rng(0)
    m = registry.make_measure("novelty")
    still = [rng.random((64, 64), dtype=np.float32)] * 16
    fresh = [rng.random((64, 64), dtype=np.float32) for _ in range(16)]
    assert m.compute(still) < 0.2
    assert m.compute(fresh) > 0.8


def lz76_reference(s: bytes) -> int:
    """Kaspar & Schuster's parse, directly: grow each phrase while it occurs earlier."""
    n, i, c = len(s), 0, 0
    while i < n:
        l = 1
        while i + l <= n and s.find(s[i : i + l], 0, i + l - 1) != -1:
            l += 1
        c, i = c + 1, i + l
    return c


def test_lz76_matches_the_reference_parse():
    rng = np.random.default_rng(0)
    assert lz76(np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1])) == 6  # K&S 1987
    cases = [rng.integers(0, L, 3000) for L in (2, 8, 13)] + [
        np.zeros(5000, int),
        np.tile([0, 1, 1, 0, 2], 900),  # phrases longer than the hashed lengths
        np.r_[np.zeros(3000, int), rng.integers(0, 2, 500), np.zeros(2000, int)],
    ]
    for x in cases:
        assert lz76(x) == lz76_reference(x.astype(np.uint8).tobytes())


def test_lzc_is_near_0_for_uniform_and_near_1_for_binary_noise():
    rng = np.random.default_rng(0)
    m, ctx = registry.make_measure("lzc"), Context(n_levels=2)
    assert m.compute([np.zeros((128, 128), np.int32)], ctx) < 0.01
    assert m.compute([rng.integers(0, 2, (128, 128))], ctx) == pytest.approx(1, abs=0.05)


def test_block_entropy():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 2, (128, 128))
    assert block_entropy(noise, 2, 1) == pytest.approx(1, abs=0.01)
    assert block_entropy(noise, 2, 2) == pytest.approx(1, abs=0.01)
    assert block_entropy(np.zeros((32, 32), int), 2, 2) == 0
    stripes = np.tile([0, 1], (64, 32))  # 1 bit per cell alone, 1/4 bit per cell in 2x2 blocks
    assert block_entropy(stripes, 2, 1) == pytest.approx(1, abs=0.01)
    assert block_entropy(stripes, 2, 2) == pytest.approx(0.25, abs=0.01)


def test_fractal_dimension_of_a_point_a_line_and_a_plane():
    m = registry.make_measure("fractal_dimension")
    point, line, plane = (np.zeros((128, 128), np.float32) for _ in range(3))
    point[5, 5] = line[64, :] = plane[:] = 1
    assert m.compute([point]) == pytest.approx(0, abs=0.01)
    assert m.compute([line]) == pytest.approx(1, abs=0.01)
    assert m.compute([plane]) == pytest.approx(2, abs=0.01)
    assert m.compute([np.zeros((128, 128), np.float32)]) == 0
    sizes, counts = box_counts(line > 0)
    assert list(sizes) == [2, 4, 8, 16, 32] and list(counts) == [64, 32, 16, 8, 4]


@pytest.mark.parametrize("levels", ["binary", "13", "continuous"])
def test_effective_complexity_sees_noise_as_uniform(levels):
    frame = np.asarray(measured_frame("baseline", f"iid_noise_{levels}"))
    assert np.all(regular_part(frame, 2.0, 4.0) == 1)  # all in the noise band


def measured_frame(system: str, pattern: str):
    return Run(RunSpec(system, pattern, (64, 64)), []).host_frames()[-1]


def test_lyapunov_separates_chaos_from_order():
    lyap = lambda s, p: measured(s, p, steps=100, size=(128, 128))["lyapunov"]
    assert lyap("baseline", "frozen_noise_continuous") == pytest.approx(0, abs=1e-3)
    assert lyap("baseline", "iid_noise_continuous") > 0.5
    assert lyap("life", "random") > lyap("life", "glider")
    assert lyap("lenia", "random") > 0


@pytest.mark.parametrize("system", ["life", "integer_lenia", "baseline"])
def test_discrete_perturbations_change_one_cell_by_one_level(system):
    sys_ = registry.make_system(system)
    pattern = {"life": "glider", "integer_lenia": "min_orbium", "baseline": "iid_noise_13"}[system]
    state = sys_.init(pattern, (32, 32), jax.random.key(0))
    n = sys_.n_states(pattern) - 1
    a = np.asarray(sys_.observe(state)) * n
    b = np.asarray(sys_.observe(sys_.perturb(state, jax.random.key(1)))) * n
    diff = np.round(b - a).astype(int)
    assert np.count_nonzero(diff) == 1
    assert set(np.abs(diff[diff != 0])) <= {1, n}  # baseline wraps the top level to 0


def rule(key: str, system: str, pattern: str, **params) -> float:
    sys_ = registry.make_system(system)
    m = registry.make_measure(key, **params)
    return m.compute_rule(sys_, pattern, Context(sys_.n_states(pattern)))


def test_transient_is_the_start_of_the_first_cycle():
    seq = np.array([[5, 0], [6, 0], [7, 0], [8, 0], [6, 0], [7, 0]])  # 5, then cycle 6 7 8
    assert transient(seq) == 1
    assert transient(np.array([[1, 0], [1, 0]])) == 0
    assert transient(np.array([[1, 0], [2, 0], [3, 0]])) is None


def test_growth_class_of_synthetic_transients():
    n = np.array([16, 24, 32, 48, 64, 96])
    assert growth_class(n, 5 + 0.01 * n) == 0  # bounded
    assert growth_class(n, 40 * np.log(n)) == 1
    assert growth_class(n, 10 * n + 7) == 2
    assert growth_class(n, 0.5 * n ** 2.5) == 3
    assert growth_class(n, np.exp(n / 8)) == 4


def test_transient_length_classes():
    # linear, as in the paper; a narrower range of sizes can tip it to polynomial
    assert rule("transient_length", "life", "glider") == 2
    assert 1 < rule("transient_length", "life", "glider", report="exponent") < 2
    measure = registry.make_measure("transient_length")
    assert measure.accepts(registry.make_system("integer_lenia"))
    assert not measure.accepts(registry.make_system("lenia"))  # continuous: never repeats
    assert measure.sizes(reach=13)[0] == 28  # wider rules start above their neighbourhood


def test_langton_lambda_of_life():
    # born with 3 of 8 neighbours (56 of 256 tables), survive with 2 or 3 (84): 140 / 512
    assert rule("langton_lambda", "life", "glider", size=512) == pytest.approx(140 / 512, abs=0.005)
    assert not registry.make_measure("langton_lambda").accepts(registry.make_system("lenia"))


def test_functional_information_rates_soups_below_creatures():
    kw = {"samples": 31, "steps": 200, "size": 64}  # at most log2(32) = 5 bits
    assert rule("functional_information", "lenia", "random", **kw) < 1
    assert rule("functional_information", "lenia", "orbium", **kw) == pytest.approx(5)
    assert rule("functional_information", "life", "random", **kw) < 1


def test_rule_measures_are_constant_and_cached_per_rule():
    run = Run(RunSpec("life", "glider", (32, 32)), ["langton_lambda"],
              {"langton_lambda": {"size": 64}})
    first, _ = run.measure()
    run.step(5)
    again, ms = run.measure()
    assert first == again and ms["langton_lambda"] < 1  # cached: no recompute


def table(t, y, x, L=2) -> np.ndarray:
    """Counts of (t, y, x) for every (y, x) pattern once, t given by a function."""
    counts = np.zeros((L, L, L))
    for yv in range(L):
        for xv in range(L):
            counts[t(yv, xv), yv, xv] += 1
    return counts


def test_pid_on_textbook_gates():
    xor, copy = pid(table(lambda y, x: y ^ x, 0, 0)), pid(table(lambda y, x: y, 0, 0))
    assert xor["synergy"] == pytest.approx(1) and xor["redundancy"] == pytest.approx(0)
    assert copy["unique_self"] == pytest.approx(1) and copy["synergy"] == pytest.approx(0)
    same = np.zeros((2, 2, 2))
    same[0, 0, 0] = same[1, 1, 1] = 1  # both sources equal the target
    assert pid(same)["redundancy"] == pytest.approx(1)
    assert pid(same)["synergy"] == pytest.approx(0)


def test_fid_on_textbook_gates_and_unseen_inputs():
    xor, copy = fid(table(lambda y, x: y ^ x, 0, 0)), fid(table(lambda y, x: y, 0, 0))
    assert xor == pytest.approx({"self": 0, "neighbour": 0, "synergy": 1})
    assert copy == pytest.approx({"self": 1, "neighbour": 0, "synergy": 0})
    # OR seen on 3 of 4 inputs: the unseen (1, 1) gets a uniform output over the observed
    # targets, as in fid-tools
    seen = table(lambda y, x: y | x, 0, 0)
    seen[1, 1, 1] = 0
    full = fid(seen)
    p = np.array([[[1, 0], [0, 0.5]], [[0, 1], [1, 0.5]]]) / 4  # p(t, y, x) after completion
    h = lambda q: -np.sum(q[q > 0] * np.log2(q[q > 0]))
    i_y = h(p.sum((1, 2))) + h(p.sum((0, 2))) - h(p.sum(2))
    assert full["self"] == pytest.approx(i_y)


def test_fid_matches_bohm_game_of_life_and_is_never_negative():
    # Bohm et al. §3.4: Life's next state from its 3x3 at t; the 8 neighbours grouped into one
    # input leave the centre's independent information unchanged (their §2.3)
    counts = np.zeros((2, 2, 256))
    for code in range(256):
        n = code.bit_count()
        for c in (0, 1):
            counts[int(n == 3 or (c == 1 and n == 2)), c, code] += 1
    result = fid(counts)
    assert result["self"] == pytest.approx(0.01092, abs=1e-5)
    assert sum(result.values()) == pytest.approx(0.84635, abs=1e-5)  # total information
    # uniform inputs are independent, so synergy = I(Y; X | T) >= 0
    rng = np.random.default_rng(0)
    for _ in range(200):
        tables = rng.integers(0, 5, (3, 3, 3)) * (rng.random((3, 3, 3)) < 0.6)
        if tables.sum():
            assert fid(tables)["synergy"] >= -1e-12


def test_statistical_complexity():
    rng = np.random.default_rng(0)
    assert len(set(causal_states(np.zeros((4, 32, 32), int), 2, 1e-3)[0])) == 1
    sc = registry.make_measure("statistical_complexity")
    # iid noise: one causal state, up to a chance split of 512 pasts tested at alpha
    assert sc.compute(list(rng.integers(0, 2, (16, 64, 64))), Context(n_levels=2)) < 0.05
    frozen = rng.integers(0, 2, (64, 64))
    # frozen noise is predictable but needs its whole 5-cell cross remembered: 5 bits
    assert sc.compute([frozen] * 8, Context(n_levels=2)) == pytest.approx(5, abs=0.05)
    soup = measured("life", "random", steps=50, size=(64, 64))["statistical_complexity"]
    gun = measured("life", "gosper_glider_gun", steps=50, size=(64, 64))["statistical_complexity"]
    assert soup > gun > 0


@pytest.mark.parametrize("key", ["transfer_entropy", "block_entropy", "statistical_complexity"])
def test_local_maps_average_to_the_value(key):
    run = Run(RunSpec("life", "random", (64, 64)), [key])
    run.step(20)
    value = run.measure()[0][key]
    heat = run.map(key)
    assert heat.shape == (64, 64)
    tol = 0.02 if key == "block_entropy" else 1e-4  # block entropy's value is bias corrected
    assert heat.mean() == pytest.approx(value, abs=tol)


def test_tile_maps_are_excess_over_an_empty_tile():
    run = Run(RunSpec("life", "glider", (64, 64)), ["lzma", "lzc", "lyapunov"])
    run.step(4)
    heat = run.map("lzma")
    assert heat.shape == (8, 8)  # tiles of 8 cells
    assert np.count_nonzero(heat) in (1, 2, 3, 4) and heat.min() == 0  # only around the glider
    assert run.map("lyapunov").max() <= 1
    assert run.map("png") is None  # not running


def test_torus_label_joins_creatures_across_edges():
    mask = np.zeros((8, 8), bool)
    mask[3, 0] = mask[3, 7] = True  # one creature split by the left/right edge
    mask[0, 4] = mask[7, 5] = True  # one split diagonally by the top/bottom edge
    labels = torus_label(mask)
    assert labels[3, 0] == labels[3, 7] and labels[0, 4] == labels[7, 5]
    assert labels[3, 0] != labels[0, 4]


def test_basins_of_the_potential_separate_creatures_and_merge_bumps():
    y, x = np.mgrid[0:64, 0:64]

    def bump(cy, cx, r, height=1.0):  # on the torus
        dy, dx = (y - cy + 32) % 64 - 32, (x - cx + 32) % 64 - 32
        return height * np.exp(-(dy**2 + dx**2) / (2 * r**2))

    two = bump(20, 20, 4) + bump(20, 44, 4)  # far apart: a deep valley between them
    assert len(np.unique(detect(None, two, 0.05, 0.8, 1))) - 1 == 2
    lumpy = bump(32, 28, 4) + bump(32, 38, 4, 0.9)  # one hill with two tops
    assert len(np.unique(basins(lumpy, 0.05))) - 1 == 2
    assert len(np.unique(detect(None, lumpy, 0.05, 0.8, 1))) - 1 == 1
    edge = bump(32, 0, 4)  # across the left/right edge: one creature
    assert len(np.unique(detect(None, edge, 0.05, 0.8, 1))) - 1 == 1


def test_target_is_tracked_by_overlap():
    frames, labels = [], []
    for t in range(6):  # a unit moving right, and a heavier one far away
        f = np.zeros((32, 32)); lab = np.zeros((32, 32), int)
        f[5:9, 2 + t:6 + t] = 1; lab[5:9, 2 + t:6 + t] = 1
        f[20:26, 20:26] = 1; lab[20:26, 20:26] = 2
        frames.append(f); labels.append(lab)
    feats = track_features(frames, labels, "all", 3, "typical")
    assert np.all(feats == feats[0])  # the same unit throughout, with the rest as E
    largest = track_features(frames, labels, "all", 3, "largest")
    assert largest[0, 0] == 36 and largest[0, 1] == 16
    near = track_features(frames, labels, "near", 3, "largest")
    assert np.all(near[:, 1] == 0)  # the other unit is beyond reach


def test_iti_features_on_an_aquarium_scene():
    run = Run(RunSpec("lenia_multi", "aquarium_scene", (256, 256)), ["iti_features"],
              {"iti_features": {"window": 64}})
    run.step(300)
    pots = run.host_potentials()
    assert pots is not None and len(pots) == len(run.host_frames())
    assert run.measure()[0]["iti_features"] > 0.3


def test_iti_decomposition_of_self_and_environment_driven_systems():
    rng = np.random.default_rng(0)
    s, e = rng.integers(0, 2, 20000), rng.integers(0, 2, 20000)
    for nxt, expected in ((s, {"organismal": 1, "colonial": 1, "driven": 0}),
                          (e, {"organismal": 0, "colonial": 0, "driven": 1})):
        joint = np.zeros((2, 2, 2))
        np.add.at(joint, (nxt, s, e), 1)
        assert decompose(joint) == pytest.approx(expected, abs=0.01)


def test_individuality_measures():
    glider = measured("life", "glider", steps=20, size=(32, 32))
    assert glider["iti_local"] > 0.5  # survival is fixed by the in-creature neighbours
    noise = measured("baseline", "iid_noise_binary", size=(64, 64))
    assert noise["iti_local"] < 0.01 and noise["iti_features"] < 0.1
    run = Run(RunSpec("life", "glider", (32, 32)), ["iti_local"], {"iti_local": {"component": "driven"}})
    run.step(20)
    assert run.measure()[0]["iti_local"] == pytest.approx(0, abs=1e-9)  # nothing else around
    assert run.map("iti_local").shape == (32, 32)


def test_parameters_from_the_ui_are_typed_and_checked():
    m = registry.make_measure("transfer_entropy", window="8", offset=2.0, n_levels="", correct_bias="false")
    assert m.config() == {"window": 8, "offset": 2, "n_levels": None, "correct_bias": False}
    assert registry.make_measure("pid", component="redundancy").component == "redundancy"
    for key, bad in [("transfer_entropy", {"window": 0}), ("transfer_entropy", {"n_levels": 1}),
                     ("transfer_entropy", {"offset": "x"}), ("pid", {"component": "bogus"}),
                     ("lzma", {"window": 4}), ("iti_local", {"window": None})]:
        with pytest.raises(ValueError):
            registry.make_measure(key, **bad)


def test_center_of_a_blob_split_by_the_edge():
    frame = np.zeros((20, 20))
    frame[9:12, [18, 19, 0, 1, 2]] = 1  # 5 wide, centered on column 0 across the edge
    y, x = centers([frame])[0]
    assert y == pytest.approx(10) and abs((x + 10) % 20 - 10) < 1e-9  # column 0 (or 20)
    moved = np.roll(frame, 3, axis=1)
    c = centers([frame, moved])
    assert c[1] - c[0] == pytest.approx([0, 3])


def test_mass_and_speed():
    def stats(system, pattern, size, steps=200):
        run = Run(RunSpec(system, pattern, (size, size)), ["mass", "speed"])
        run.step(steps)
        return run.measure()[0]

    for size in (16, 128):  # on 16² the glider crosses the edge every ~16 steps
        glider = stats("life", "glider", size)
        assert glider["speed"] == pytest.approx(np.sqrt(2) / 4) and glider["mass"] == 5
    assert stats("life", "galaxy", 64)["speed"] == 0  # an oscillator stays put
    assert stats("baseline", "empty", 32) == {"mass": 0, "speed": 0}
    run = Run(RunSpec("life", "glider", (32, 32)), ["change_rate"])
    run.step(50)
    assert run.measure()[0]["change_rate"] == pytest.approx(4)  # 2 births and 2 deaths per step
    assert stats("lenia", "orbium", 64)["speed"] > 0.3


def test_normalized_versions():
    keys = ["statistical_complexity", "transfer_entropy", "mass", "speed"]

    def values(size, normalized):
        run = Run(RunSpec("lenia", "orbium", (size, size)), keys, normalized=normalized)
        run.step(300)
        return run.measure()[0]

    grid, small, large = values(128, False), values(128, True), values(256, True)
    assert small["mass"] == grid["mass"] and small["speed"] == grid["speed"]  # not ratios
    # SC x cells / mass, with the mass over SC's window (constant for Orbium to ~0.3 %)
    expect = grid["statistical_complexity"] * 128**2 / grid["mass"]
    assert small["statistical_complexity"] == pytest.approx(expect, rel=0.01)
    # grid averages shrink ~4x on a 4x larger grid; the ratios change little (MEASURES §5.10)
    for key in ("statistical_complexity", "transfer_entropy"):
        assert 1 < large[key] / small[key] < 1.3, key
    frozen = Run(RunSpec("baseline", "frozen_noise_binary", (32, 32)), ["transfer_entropy"],
                 normalized=True)
    frozen.step(20)
    assert frozen.measure()[0]["transfer_entropy"] == 0  # nothing changes: no ratio
    m = registry.make_measure("transfer_entropy")
    assert (m.ratio, m.ratio_unit, m.map_unit) == ("change", "bits/Δmass", "bits/cell/step")


def test_normalized_te_has_no_spikes_when_a_creature_slows_down():
    # Tetralapillium almost stops at times but keeps changing shape; TE / (speed × mass)
    # reached 39x its median there, TE / change rate stays within 2x (MEASURES §5.10)
    run = Run(RunSpec("lenia", "tetralapillium_metamorpha", (256, 256)),
              ["transfer_entropy", "pid", "speed", "change_rate"], normalized=True)
    run.step(200)
    te, pid, speed = [], [], []
    for _ in range(40):
        run.step(10)
        values = run.measure()[0]
        te.append(values["transfer_entropy"])
        pid.append(values["pid"])
        speed.append(values["speed"])
    assert min(speed) < 0.1 * max(speed)  # it does slow down
    assert max(te) < 2 * np.median(te) and max(pid) < 2 * np.median(pid)


def test_normalized_compression_counts_bits_above_an_empty_frame():
    # LZMA's fixed overhead is subtracted, so the creature's bits per unit of mass do not
    # depend on the grid size (MEASURES §5.10)
    def lzma(size):
        run = Run(RunSpec("lenia", "orbium", (size, size)), ["lzma", "effective_complexity"],
                  normalized=True)
        run.step(300)
        return run.measure()[0]

    small, large = lzma(128), lzma(256)
    for key in small:
        assert large[key] == pytest.approx(small[key], rel=0.03), key
    empty = Run(RunSpec("baseline", "empty", (64, 64)), ["lzma", "lzc"], normalized=True)
    assert empty.measure()[0] == {"lzma": 0, "lzc": 0}


def test_measures_are_listed_in_module_order():
    keys = list(registry.measures())
    assert keys[0] == "lzma" and keys[-3:] == ["mass", "speed", "change_rate"]
    assert all(cls.icon for cls in registry.measures().values())


def test_every_measure_has_an_equation_for_its_info_popup():
    for key, cls in registry.measures().items():
        assert cls.equation and cls.equation.count("{") == cls.equation.count("}"), key


def test_neighbour_sum_and_objects_maps():
    run = Run(RunSpec("lenia", "orbium", (64, 64)), [])
    run.step(50)
    potential, objects = run.map("potential"), run.map("objects")
    assert potential.shape == (64, 64) and potential.max() > 0.1
    assert set(np.unique(objects)) == {0, 1}  # one creature, the target
    noise = Run(RunSpec("baseline", "iid_noise_binary", (32, 32)), [])
    assert noise.map("potential") is None  # no potential: no map, but objects still
    assert noise.map("objects").shape == (32, 32)
