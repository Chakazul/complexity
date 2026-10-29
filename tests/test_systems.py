"""System goldens and JAX parity with the M0 NumPy implementations (PLAN.md §6)."""

import dataclasses
import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import numpy_reference as ref
import pytest

from complexity import engine, registry
from complexity.vendor.lenium import integer

KEY = jax.random.key(0)


def run(system, state, steps):
    return engine.advance(system, state, steps)


def circular_com(frame) -> np.ndarray:
    """Center of mass on the torus, as angles per axis."""
    a = np.asarray(frame, float)
    return np.array([
        np.angle((a.sum(1 - ax) * np.exp(2j * np.pi * np.arange(n) / n)).sum())
        for ax, n in enumerate(a.shape)
    ])


def com_shift(a, b) -> float:
    """Distance in cells between the centers of mass of two frames on the same torus."""
    d = (circular_com(b) - circular_com(a) + np.pi) % (2 * np.pi) - np.pi
    return float(np.linalg.norm(d * np.array(a.shape) / (2 * np.pi)))


# ---- Game of Life ----

@pytest.fixture
def life():
    return registry.make_system("life")


def test_glider_translates_after_four_steps(life):
    start = life.init("glider", (32, 32), KEY)
    assert np.array_equal(jnp.roll(start, (1, 1), (0, 1)), run(life, start, 4))


def test_pulsar_has_period_three(life):
    start = life.init("pulsar", (32, 32), KEY)
    assert not np.array_equal(start, run(life, start, 1))
    assert np.array_equal(start, run(life, start, 3))


def test_gosper_gun_emits_gliders(life):
    start = life.init("gosper_glider_gun", (64, 96), KEY)
    assert start.sum() == 36
    assert run(life, start, 30).sum() == 36 + 5  # period 30: the gun plus one glider


def test_life_matches_numpy_for_200_steps_of_random_soup(life):
    grid = life.init("random", (64, 64), KEY)
    expected = np.asarray(grid)
    _, frames = engine.rollout(life, grid, 200)
    for t, frame in enumerate(np.asarray(frames)):
        expected = ref.life_step(expected)
        assert np.array_equal(frame, expected), t


# ---- Lenia ----

@pytest.fixture
def lenia():
    return registry.make_system("lenia")


def test_lenia_n_states_follow_the_pattern(lenia):
    assert lenia.n_states("min_orbium") == lenia.n_states("random") == 13
    assert lenia.n_states("orbium") is lenia.n_states("random_continuous") is None


def test_min_orbium_survives_moves_and_keeps_13_levels(lenia):
    state = lenia.init("min_orbium", (64, 64), KEY)
    mass0 = int(state.A.sum())
    _, frames = engine.rollout(lenia, state, 500, 50)
    frames = np.asarray(frames)
    for frame in frames:
        assert 0.9 * mass0 <= frame.sum() * 12 <= 1.1 * mass0
        assert np.array_equal(np.unique(frame), np.arange(13, dtype=np.float32) / np.float32(12))
    assert com_shift(lenia.observe(state), frames[0]) > 3


def test_min_orbium_trajectory_is_unchanged(lenia):
    """Golden from the port, which is bit-identical to the predecessor's IntegerLenia for
    1000 steps (checked against ~/github/lenia-ref when porting). Guards the switch from
    vendor/lenium to `import lenium`."""
    A = np.asarray(run(lenia, lenia.init("min_orbium", (64, 64), KEY), 500).A)
    assert A.sum() == 734
    assert hashlib.sha256(A.astype("<i4").tobytes()).hexdigest()[:16] == "758ee37c90b3e790"


def test_integer_potential_matches_world_size_kernel():
    """The cropped kernel gives the predecessor's world-size direct convolution."""
    R, P, N = 5, 12, 24
    K = integer.kernel(R, P)
    full = jnp.zeros((N, N), K.dtype).at[N // 2 - R : N // 2 + R + 1,
                                         N // 2 - R : N // 2 + R + 1].set(K)
    A = jax.random.randint(KEY, (N, N), 0, P + 1)
    pads = [((N - 1) // 2, N // 2)] * 2
    expected = jax.scipy.signal.convolve(jnp.pad(A, pads, mode="wrap"), full, "valid", "direct")
    assert np.array_equal(integer.potential(A, K), expected)


@pytest.mark.parametrize("size", [(64, 64), (128, 96), (75, 131)])
def test_integer_fft_potential_is_bit_identical_to_direct(size):
    """The FFT path is allowed only because it rounds to the direct convolution exactly
    (PLAN.md §4.2); its error before rounding must stay far below 0.5."""
    K = integer.kernel(12, 12)
    fK = integer.kernel_fft(K, size)
    grids = [jnp.full(size, 12), jax.random.randint(KEY, size, 0, 13)]
    lenia = registry.make_system("lenia")
    for pattern in ("min_orbium", "random"):
        state = lenia.init(pattern, size, KEY)
        _, frames = engine.rollout(lenia, state, 200, 20)
        grids += [jnp.round(f * 12).astype(jnp.int32) for f in frames]
    for A in grids:
        U = integer.potential(A, K)
        U_fft = jnp.fft.irfft2(jnp.fft.rfft2(A.astype(jnp.float32)) * fK, s=size)
        assert float(jnp.abs(U_fft - U).max()) < 0.05
        assert np.array_equal(integer.potential_fft(A, fK), U)


def test_classic_orbium_survives_and_moves(lenia):
    state = lenia.init("orbium", (128, 128), KEY)
    mass0 = float(state.A.sum())
    _, frames = engine.rollout(lenia, state, 200, 200)
    assert 0.8 * mass0 < frames[-1].sum() < 1.2 * mass0
    assert com_shift(lenia.observe(state), frames[-1]) > 3


@pytest.mark.parametrize("pattern", ["orbium", "random_continuous"])
def test_continuous_lenia_matches_numpy(lenia, pattern):
    """Each step matches to 1e-5. The NumPy reference runs its FFT in float64, so over a
    trajectory the float32 difference accumulates: about 4e-4 after 200 steps."""
    state = lenia.init(pattern, (64, 64), KEY)
    expected = np.asarray(state.A)
    fK = ref.lenia_kernel_fft((64, 64), 13, [1.0])
    for t in range(200):
        one_step = lenia.step(dataclasses.replace(state, A=jnp.asarray(expected))).A
        state = lenia.step(state)
        expected = ref.lenia_step(expected, fK, 10, 0.15, 0.015)
        np.testing.assert_allclose(one_step, expected, atol=1e-5, err_msg=f"step {t}")
    np.testing.assert_allclose(state.A, expected, atol=1e-3)
