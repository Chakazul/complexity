"""System goldens and JAX parity with the M0 NumPy implementations (PLAN.md §6)."""

import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import numpy_reference as ref
import pytest

from complexity import engine, registry
from complexity.systems.base import place_center
from complexity.systems.lenia import CATALOGUE
from complexity.vendor.lenium import model

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


def test_gosper_gun_emits_gliders(life):
    start = life.init("gosper_glider_gun", (64, 96), KEY)
    assert start.sum() == 36
    assert run(life, start, 30).sum() == 36 + 5  # period 30: the gun plus one glider


# (pattern, warm-up steps, period, shift per period); the catalogue spaceships start one
# cell short of the ship and grow into it within a few steps
PERIODIC = [
    ("glider", 0, 4, (1, 1)), ("lwss", 8, 4, (0, 2)), ("mwss", 8, 4, (0, 2)),
    ("hwss", 8, 4, (0, 2)), ("copperhead", 0, 10, (-1, 0)), ("weekender", 0, 7, (-2, 0)),
    ("lobster", 0, 7, (-1, 1)), ("sir_robin", 0, 6, (-2, -1)), ("galaxy", 0, 8, (0, 0)),
    ("star_gate", 0, 60, (0, 0)),
]


@pytest.mark.parametrize(("pattern", "warmup", "period", "shift"), PERIODIC)
def test_catalogue_spaceships_and_oscillators(life, pattern, warmup, period, shift):
    n = 256 if pattern == "star_gate" else 128
    start = run(life, life.init(pattern, (n, n), KEY), warmup)
    assert np.array_equal(jnp.roll(start, shift, (0, 1)), run(life, start, period))


@pytest.mark.parametrize("pattern", ["gosper_glider_gun", "sidecar_gun", "pufferfish",
                                     "switch_engine", "max"])
def test_guns_puffers_and_spacefillers_grow(life, pattern):
    n = 256 if pattern == "sidecar_gun" else 128
    start = life.init(pattern, (n, n), KEY)
    assert run(life, start, 100).sum() > start.sum()


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


@pytest.fixture
def integer_lenia():
    return registry.make_system("integer_lenia")


def test_lenia_n_states_follow_the_pattern(lenia, integer_lenia):
    assert integer_lenia.n_states("min_orbium") == integer_lenia.n_states("random") == 13
    assert lenia.n_states("orbium") is lenia.n_states("random") is None


def test_min_orbium_survives_moves_and_keeps_13_levels(integer_lenia):
    lenia = integer_lenia
    state = lenia.init("min_orbium", (64, 64), KEY)
    mass0 = int(state.A.sum())
    _, frames = engine.rollout(lenia, state, 500, 50)
    frames = np.asarray(frames)
    for frame in frames:
        assert 0.9 * mass0 <= frame.sum() * 12 <= 1.1 * mass0
        assert np.array_equal(np.unique(frame), np.arange(13, dtype=np.float32) / np.float32(12))
    assert com_shift(lenia.observe(state), frames[0]) > 3


def test_min_orbium_trajectory_is_unchanged(integer_lenia):
    lenia = integer_lenia
    """Golden from the port, which is bit-identical to the predecessor's IntegerLenia for
    1000 steps (checked against ~/github/lenium.old when porting). Guards the switch from
    vendor/lenium to `import lenium`."""
    A = np.asarray(run(lenia, lenia.init("min_orbium", (64, 64), KEY), 500).A)
    assert A.sum() == 734
    assert hashlib.sha256(A.astype("<i4").tobytes()).hexdigest()[:16] == "758ee37c90b3e790"


def test_integer_potential_matches_world_size_kernel():
    """The cropped kernel gives the predecessor's world-size direct convolution."""
    R, P, N = 5, 12, 24
    K = model.integer_kernel(R, P)
    full = jnp.zeros((N, N), K.dtype).at[N // 2 - R : N // 2 + R + 1,
                                         N // 2 - R : N // 2 + R + 1].set(K)
    A = jax.random.randint(KEY, (N, N), 0, P + 1)
    pads = [((N - 1) // 2, N // 2)] * 2
    expected = jax.scipy.signal.convolve(jnp.pad(A, pads, mode="wrap"), full, "valid", "direct")
    assert np.array_equal(model.potential(A, K), expected)


@pytest.mark.parametrize("size", [(64, 64), (128, 96), (75, 131)])
def test_integer_fft_potential_is_bit_identical_to_direct(size):
    """The FFT path is allowed only because it rounds to the direct convolution exactly
    (PLAN.md §4.2); its error before rounding must stay far below 0.5."""
    K = model.integer_kernel(12, 12)
    fK = model.integer_kernel_fft(K, size)
    grids = [jnp.full(size, 12), jax.random.randint(KEY, size, 0, 13)]
    lenia = registry.make_system("integer_lenia")
    for pattern in ("min_orbium", "random"):
        state = lenia.init(pattern, size, KEY)
        _, frames = engine.rollout(lenia, state, 200, 20)
        grids += [jnp.round(f * 12).astype(jnp.int32) for f in frames]
    for A in grids:
        U = model.potential(A, K)
        U_fft = jnp.fft.irfft2(jnp.fft.rfft2(A.astype(jnp.float32)) * fK, s=size)
        assert float(jnp.abs(U_fft - U).max()) < 0.05
        assert np.array_equal(model.potential_fft(A, fK), U)


def test_gaussian_orbium_survives_and_moves(lenia):
    state = lenia.init("orbium", (128, 128), KEY)
    mass0 = float(state.A.sum())
    _, frames = engine.rollout(lenia, state, 200, 200)
    assert 0.8 * mass0 < frames[-1].sum() < 1.2 * mass0
    assert com_shift(lenia.observe(state), frames[-1]) > 3
    # Golden mass from the port, which matches the hand-coded Lenium to ~1e-7 per step;
    # float rounding differs across devices, so not exact.
    assert abs(float(frames[-1].sum()) - 72.9739) < 1e-3


@pytest.mark.parametrize("pattern", list(CATALOGUE))
def test_catalogue_creatures_survive(lenia, pattern):
    n = 256 if pattern == "amoebium_dentalis" else 128
    state = lenia.init(pattern, (n, n), KEY)
    mass0 = float(state.A.sum())
    A = run(lenia, state, 500).A
    assert 0.8 * mass0 < float(A.sum()) < 1.2 * mass0
    assert float((A > 0.1).mean()) < 0.2  # still a creature, not a filled grid


# ---- every system ----

@pytest.mark.parametrize("system", ["life", "lenia", "integer_lenia"])
def test_every_pattern_inits_on_a_small_grid(system):
    sys_ = registry.make_system(system)
    for pattern in sys_.patterns:
        assert sys_.init(pattern, (32, 32), KEY) is not None, pattern


def test_a_pattern_larger_than_the_grid_is_cropped_to_its_center():
    pattern = np.arange(25).reshape(5, 5) + 1
    grid = place_center(jnp.zeros((3, 4), jnp.int32), pattern)
    assert np.array_equal(grid, pattern[1:4, 0:4])  # rows (5 - 3) // 2 = 1.., columns 0..


def _units(system, state) -> int:
    from complexity.measures.individuality import torus_label

    return len(np.unique(torus_label(np.asarray(system.observe(state)) > 0.1))) - 1


def test_multi_channel_kernels_and_maps():
    from complexity.systems.lenia_multi import CREATURES
    from complexity.vendor.lenium import model

    kernels = CREATURES["aquarium"]["kernels"]
    fK = model.multi_kernel_fft((64, 64), 12, kernels)
    assert fK.shape == (64, 33, 15)
    assert np.allclose(fK[0, 0], 1, atol=1e-5)  # every kernel sums to 1
    _, _, _, Mk, Mc = model.multi_params(kernels, 3)
    assert np.array_equal(Mk.sum(0), np.ones(15)) and np.array_equal(Mc.sum(1), np.ones(15))
    assert [int(np.argmax(Mk[:, i])) for i in range(15)] == [k["c0"] for k in kernels]


def test_vt049w_differs_from_lenium_in_two_kernels():
    from complexity.systems.lenia_multi import CREATURES

    a, b = CREATURES["aquarium"]["kernels"], CREATURES["vt049w"]["kernels"]
    assert [i for i, (x, y) in enumerate(zip(a, b)) if x != y] == [3, 13]


def test_aquarium_self_replicates_and_the_vt049w_unit_does_not():
    lm = registry.make_system("lenia_multi")
    aquarium = lm.init("aquarium", (256, 256), jax.random.key(0))
    assert _units(lm, engine.advance(lm, aquarium, 300)) >= 3
    unit = lm.init("aquarium_unit", (128, 128), jax.random.key(0))
    later = engine.advance(lm, unit, 500)
    assert _units(lm, later) == 1
    assert 0.8 < float(later.A.sum() / unit.A.sum()) < 3  # grows into its adult size, ~2.3x
