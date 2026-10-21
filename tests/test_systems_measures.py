import numpy as np

from complexity import registry
from complexity.session import Run, RunSpec
from complexity.systems.life import GameOfLife


def test_glider_translates_after_four_steps():
    life = GameOfLife(size=(32, 32))
    life.reset("glider")
    start = life.grid.copy()
    for _ in range(4):
        life.step()
    assert np.array_equal(np.roll(np.roll(start, 1, 0), 1, 1), life.grid)


def test_pulsar_has_period_three():
    life = GameOfLife(size=(32, 32))
    life.reset("pulsar")
    start = life.grid.copy()
    life.step()
    assert not np.array_equal(start, life.grid)
    life.step()
    life.step()
    assert np.array_equal(start, life.grid)


def test_gosper_gun_emits_gliders():
    life = GameOfLife(size=(64, 96))
    life.reset("gosper_glider_gun")
    assert life.grid.sum() == 36
    for _ in range(30):
        life.step()
    assert life.grid.sum() == 36 + 5  # period 30: the gun plus one glider


def test_orbium_survives_and_moves():
    lenia = registry.make_system("lenia", size=(128, 128))
    lenia.reset("orbium")
    mass0 = lenia.A.sum()
    com0 = np.argwhere(lenia.A > 0.1).mean(0)
    for _ in range(200):
        lenia.step()
    assert 0.8 * mass0 < lenia.A.sum() < 1.2 * mass0
    assert np.linalg.norm(np.argwhere(lenia.A > 0.1).mean(0) - com0) > 3


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
    for key in ("video_lzma", "video_h264"):
        m = registry.make_measure(key)
        assert m.compute(still) < 0.5 * m.compute(fresh), key


def test_run_measures_every_registered_pair():
    for system, cls in registry.systems().items():
        run = Run(RunSpec(system, next(iter(cls.patterns)), (64, 64)), list(registry.measures()))
        run.step(3)
        values, _ = run.measure()
        assert set(values) == set(registry.measures()) and all(np.isfinite(list(values.values())))
