import jax
import numpy as np

from complexity import engine, registry
from complexity.measures.base import discretize
from complexity.session import Run, RunSpec


def test_rollout_observes_every_k_steps():
    life = registry.make_system("life")
    state = life.init("random", (32, 32), jax.random.key(0))
    final, frames = engine.rollout(life, state, 12, 4)
    assert frames.shape == (3, 32, 32) and frames.dtype == np.float32
    assert np.array_equal(frames[-1], life.observe(final))
    assert np.array_equal(frames[0], life.observe(engine.advance(life, state, 4)))


def test_rollout_batch_matches_single_rollouts():
    lenia = registry.make_system("integer_lenia")
    keys = jax.random.split(jax.random.key(0), 3)
    states = engine.init_batch(lenia, "random", (32, 32), keys)
    _, frames = engine.rollout_batch(lenia, states, 10, 5)
    assert frames.shape == (3, 2, 32, 32)
    for i, key in enumerate(keys):
        _, single = engine.rollout(lenia, lenia.init("random", (32, 32), key), 10, 5)
        assert np.array_equal(frames[i], single)
    assert not np.array_equal(frames[0], frames[1])  # seeds differ


def test_systems_compare_by_type_and_params():
    assert registry.make_system("life") == registry.make_system("life")
    assert hash(registry.make_system("lenia")) == hash(registry.make_system("lenia"))
    assert registry.make_system("life") != registry.make_system("lenia")


def test_run_buffers_the_window_and_skips_unobserved_steps():
    run = Run(RunSpec("life", "glider", (32, 32)), ["video_lzma"])  # window 16
    run.step(3)
    assert len(run.host_frames()) == 4  # initial frame + 3 steps
    run.step(40)
    frames = run.host_frames()
    assert run.t == 43 and len(frames) == 16
    reference = registry.make_system("life")
    state = reference.init("glider", (32, 32), jax.random.key(0))
    assert np.array_equal(frames[-1], reference.observe(engine.advance(reference, state, 43)))
    assert np.array_equal(frames[0], reference.observe(engine.advance(reference, state, 28)))


def test_discretize_is_exact_for_discrete_frames():
    for n in (2, 13):
        levels = np.arange(n)
        frames = (levels / np.float32(n - 1)).astype(np.float32)
        assert np.array_equal(discretize(frames, n), levels)
        assert np.array_equal(np.asarray(discretize(jax.numpy.asarray(frames), n)), levels)


def test_discretize_bins_continuous_frames_uniformly():
    frames = np.array([0.0, 0.12, 0.125, 0.5, 0.99, 1.0], np.float32)
    assert discretize(frames, 8).tolist() == [0, 0, 1, 4, 7, 7]
    assert discretize(frames, 8).dtype == np.int32
