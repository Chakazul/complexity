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
from complexity.measures.individuality import decompose, torus_label
from complexity.measures.information import block_entropy, transfer_entropy
from complexity.measures.rule import transient
from complexity.session import Run, RunSpec


def measured(system: str, pattern: str, steps: int = 20, size=(64, 64), **params) -> dict:
    """Every measure's value after `steps` steps (past every window)."""
    run = Run(RunSpec(system, pattern, size), list(registry.measures()), params)
    run.step(steps)
    return run.measure()[0]


@pytest.mark.parametrize("levels", ["binary", "13", "continuous"])
def test_orientation_noise_scores_at_least_empty(levels):
    """PLAN.md §6: higher = more complex, so iid noise never scores below an empty grid."""
    empty, noise = measured("baseline", "empty"), measured("baseline", f"iid_noise_{levels}")
    for key, cls in registry.measures().items():
        if cls.peaks_at != "mixed" and key in noise:  # skips measures for other systems
            assert noise[key] >= empty[key], key


def test_structure_peaked_measures_rate_a_glider_stream_above_noise_and_empty():
    stream = measured("life", "gosper_glider_gun", steps=100, size=(128, 128))
    noise = measured("baseline", "iid_noise_binary", size=(128, 128))
    empty = measured("baseline", "empty", size=(128, 128))
    peaked = [k for k, cls in registry.measures().items()
              if cls.peaks_at == "structure" and k in stream and k in noise]
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
    """PLAN.md §6: 0 for empty and frozen noise, ~0 for iid noise after bias correction."""
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


def test_transient_length_on_baselines_and_life():
    kw = {"size": 32, "batch": 2, "max_steps": 256}
    assert rule("transient_length", "baseline", "frozen_noise_13", **kw) == 0
    assert rule("transient_length", "baseline", "iid_noise_binary", **kw) == np.log2(257)
    assert 0 < rule("transient_length", "life", "glider", **kw) < np.log2(257)


def test_langton_lambda_of_life():
    # born with 3 of 8 neighbours (56 of 256 tables), survive with 2 or 3 (84): 140 / 512
    assert rule("langton_lambda", "life", "glider", size=512) == pytest.approx(140 / 512, abs=0.005)
    assert not registry.make_measure("langton_lambda").accepts(registry.make_system("lenia"))


def test_functional_information_rates_soups_below_creatures():
    kw = {"samples": 16, "steps": 200, "size": 64}
    soup = rule("functional_information", "lenia", "random", **kw)
    assert soup < rule("functional_information", "lenia", "orbium", **kw) <= np.log2(32)


def test_rule_measures_are_constant_and_cached_per_rule():
    run = Run(RunSpec("baseline", "frozen_noise_binary", (32, 32)), ["transient_length"],
              {"transient_length": {"size": 32, "batch": 2, "max_steps": 64}})
    first, _ = run.measure()
    run.step(5)
    again, ms = run.measure()
    assert first == again and ms["transient_length"] < 1  # cached: no recompute


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
