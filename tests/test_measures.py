import jax.numpy as jnp
import numpy as np
import pytest

from complexity import registry
from complexity.measures.base import Context, discretize
from complexity.measures.information import transfer_entropy
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
    for key in registry.measures():
        assert noise[key] >= empty[key], key


def test_structure_peaked_measures_rate_a_glider_stream_above_noise_and_empty():
    stream = measured("life", "gosper_glider_gun", steps=100, size=(128, 128))
    noise = measured("baseline", "iid_noise_binary", size=(128, 128))
    empty = measured("baseline", "empty", size=(128, 128))
    peaked = [k for k, cls in registry.measures().items() if cls.peaks_at == "structure"]
    assert peaked
    for key in peaked:
        assert stream[key] > max(noise[key], empty[key]), key


def test_compression_orders_uniform_below_noise():
    rng = np.random.default_rng(0)
    uniform = [np.zeros((64, 64), np.float32)] * 16
    noise = [rng.random((64, 64), dtype=np.float32) for _ in range(16)]
    for key, cls in registry.measures().items():
        if cls.backend != "host" or cls.needs_discrete:
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
        assert set(values) == set(registry.measures()) and all(np.isfinite(list(values.values())))


def test_novelty_separates_static_from_fresh_frames():
    rng = np.random.default_rng(0)
    m = registry.make_measure("novelty")
    still = [rng.random((64, 64), dtype=np.float32)] * 16
    fresh = [rng.random((64, 64), dtype=np.float32) for _ in range(16)]
    assert m.compute(still) < 0.2
    assert m.compute(fresh) > 0.8
