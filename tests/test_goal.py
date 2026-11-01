import jax
import numpy as np

from complexity import engine, registry
from complexity.goal import complexity_field, perturb


def scene(system_key, pattern, size, steps):
    system = registry.make_system(system_key)
    state = engine.advance(system, system.init(pattern, (size, size), jax.random.key(0)), steps)
    return system, pattern, state


def test_perturb_changes_only_the_chosen_cells():
    rng = np.random.default_rng(0)
    cells = np.full((8, 8), 0.5, np.float32)
    where = np.zeros((8, 8), bool)
    where[2:4, 2:4] = True
    out = perturb(cells, where, 0.3, None, rng)
    assert np.all(out[~where] == 0.5) and not np.all(out[where] == 0.5)
    life = perturb(np.ones((8, 8), np.uint8), where, 1.0, 2, rng)
    assert set(np.unique(life)) <= {0, 1} and np.all(life[~where] == 1)


def test_noise_and_empty_score_zero():
    for pattern in ("empty", "iid_noise_continuous", "frozen_noise_continuous"):
        field, _ = complexity_field(*scene("baseline", pattern, 64, 1))
        assert field.max() == 0, pattern


def test_a_self_maintaining_unit_scores_above_a_glider():
    # room for the contact test, and 3 draws per hit: one patch of noise can land where
    # it kills Orbium
    _, orbium = complexity_field(*scene("lenia", "orbium", 256, 200), trials=3)
    _, glider = complexity_field(*scene("life", "glider", 64, 20))
    assert orbium[0].C > 0 and orbium[0].M > 0  # repairs the weaker hits
    assert glider[0].C == 0  # a perturbed glider does not come back


def test_autopoiesis_measure_runs_in_the_background():
    import time

    from complexity.session import Run, RunSpec

    now = Run(RunSpec("lenia", "orbium", (256, 256)), ["autopoiesis"],
              {"autopoiesis": {"background": False}})
    now.step(200)
    now.measure()
    assert now.measures[0]._scores and now.map("autopoiesis").shape == (256, 256)
    later = Run(RunSpec("lenia", "orbium", (256, 256)), ["autopoiesis"])
    later.step(200)
    later.measure()
    assert not later.measures[0]._scores  # the first field is still being computed
    for _ in range(200):
        time.sleep(0.1)
        later.measure()
        if later.measures[0]._scores:
            break
    assert later.measures[0]._scores  # it arrived, without the run waiting for it
