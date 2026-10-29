import numpy as np

from complexity import registry
from complexity.session import Run, RunSpec


def test_compression_orders_uniform_below_noise():
    rng = np.random.default_rng(0)
    uniform = [np.zeros((64, 64), np.float32)] * 16
    noise = [rng.random((64, 64), dtype=np.float32) for _ in range(16)]
    for key, cls in registry.measures().items():
        m = cls()
        assert m.compute(uniform[-m.window :]) < 0.5 < m.compute(noise[-m.window :]), key


def test_video_compression_rewards_repetition():
    rng = np.random.default_rng(0)
    still = [rng.random((64, 64), dtype=np.float32)] * 16
    fresh = [rng.random((64, 64), dtype=np.float32) for _ in range(16)]
    for key in ("video_lzma", "video_h264", "video_min"):
        m = registry.make_measure(key)
        assert m.compute(still) < 0.5 * m.compute(fresh), key


def test_run_measures_every_registered_pair():
    for system, cls in registry.systems().items():
        run = Run(RunSpec(system, next(iter(cls.patterns)), (64, 64)), list(registry.measures()))
        run.step(3)
        values, _ = run.measure()
        assert set(values) == set(registry.measures()) and all(np.isfinite(list(values.values())))


def test_best_of_is_the_minimum_of_its_compressors():
    run = Run(RunSpec("lenia", "orbium", (64, 64)), ["png", "lzma", "image_min",
                                                      "video_lzma", "video_h264", "video_min"])
    run.step(20)
    v, _ = run.measure()
    assert v["image_min"] == min(v["png"], v["lzma"])
    assert v["video_min"] == min(v["video_lzma"], v["video_h264"])


def test_novelty_separates_static_from_fresh_frames():
    rng = np.random.default_rng(0)
    m = registry.make_measure("novelty")
    still = [rng.random((64, 64), dtype=np.float32)] * 16
    fresh = [rng.random((64, 64), dtype=np.float32) for _ in range(16)]
    assert m.compute(still) < 0.2
    assert m.compute(fresh) > 0.8
