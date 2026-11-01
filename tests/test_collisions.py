import numpy as np
import pytest

from complexity import collisions as co


@pytest.fixture(scope="module")
def orbium():
    return co.settle("lenia", "orbium", grid=128, steps=200)


def test_a_settled_unit_knows_its_heading_and_speed(orbium):
    assert orbium.speed == pytest.approx(0.60, abs=0.02)  # Orbium's speed, MEASURES §5.9
    assert orbium.mass == pytest.approx(72.9, abs=1)
    assert orbium.cells.ndim == 2 and orbium.size < 64


def test_rotation_turns_the_heading():
    glider = co.settle("life", "glider", grid=64, steps=40)
    assert glider.heading % 90 == pytest.approx(45)  # gliders move diagonally
    turned = co.rotate(glider.cells, 90, discrete=True)
    assert np.array_equal(np.rot90(turned, 1), glider.cells)


def test_collisions_are_labelled(orbium):
    head_on = co.collide(orbium, "pair", 180, 0.0, after=300)
    assert head_on.contact and head_on.label in co.LABELS
    # a collision's outcome depends on the phase the units meet in (the 90° one survives
    # or loses a unit with a few more settling steps), so only the controls are fixed
    alone = co.collide(orbium, "alone", after=300)
    assert alone.label == "survived" and alone.total_mass_ratio == pytest.approx(1, abs=0.1)
    noise = co.collide(orbium, "noise", after=300)
    assert noise.label == "survived"  # Orbium eats through a patch of noise of its size


def test_labels_come_from_the_final_objects():
    c = co.Collision("lenia", "orbium", "pair", 180, 0, False, 128, 100, 10.0, contact=True)
    for masses, ratio, expected in [([10, 9.5], 1.0, "survived"), ([19.8], 1.0, "merged"),
                                    ([10, 10, 10], 1.5, "divided"), ([], 0.0, "died"),
                                    ([30, 30], 2.5, "goo"), ([10, 6], 0.8, "changed"),
                                    ([5, 5, 5, 6], 1.05, "divided")]:  # half-size units: replication
        c.final_masses, c.total_mass_ratio = masses, ratio
        assert co.label(c, 20.0) == expected, masses
    c.contact = False
    c.final_masses, c.total_mass_ratio = [10, 10], 1.0
    assert co.label(c, 20.0) == "missed"


def test_intrusion_labels_compare_with_the_colonys_natural_range():
    r = co.Intrusion("lenia", "orbium", 6, 0, 0, 0.0, 256, 100, 10.0,
                     control_units=[20, 22, 25], control_mass=[200.0, 220.0, 250.0],
                     contact=True)
    for units, mass, intruder, expected in [(23, 230, 10, "joined"), (23, 230, 0, "absorbed"),
                                            (35, 300, 10, "divided"), (15, 180, 10, "disrupted"),
                                            (40, 600, 10, "goo"), (5, 80, 0, "collapsed")]:
        r.final_units, r.final_mass, r.intruder_mass = units, mass, intruder
        assert co.intrusion_label(r) == expected, (units, mass, intruder)
    r.contact = False
    assert co.intrusion_label(r) == "missed"


def test_a_unit_thrown_into_a_colony(orbium):
    (run,) = co.intrude(orbium, members=4, approaches=(0,), offsets=(0.0,), after=200,
                        controls=2)
    assert run.colony_units >= 1 and len(run.control_units) == 2
    assert run.contact  # thrown at the centre, it reaches the colony
    assert run.label in ("joined", "absorbed", "divided", "disrupted", "goo", "collapsed")


def test_two_units_moving_as_one_are_fused():
    c = co.Collision("lenia", "orbium", "pair", 60, 0, False, 256, 100, 73.0, 39, contact=True)
    # Synorbium as the detector sees it: three bumps, one velocity, two Orbia's mass
    c.final_objects = [(43, 100, 100, 0.26, 0.54), (55, 105, 110, 0.26, 0.53),
                       (49, 95, 108, 0.28, 0.54)]
    c.final_masses, c.total_mass_ratio = [55, 49, 43], 1.0
    assert co.fused(c) and co.label(c) == "fused"
    c.final_objects = [(73, 100, 100, 0.0, 0.6), (73, 200, 30, 0.5, -0.3)]  # apart, diverging
    c.final_masses = [73, 73]
    assert not co.fused(c) and co.label(c) == "survived"
    c.final_objects, c.final_masses, c.total_mass_ratio = [(73, 100, 100, 0.0, 0.6)], [73], 0.5
    assert co.label(c) == "lost one"
