"""The prototype goal measure C = S × M × I (complexity.goal) on the reference scenes of
MEASURES.md §8.1, lowest expected first: empty, noise, frozen noise, goo, the Life glider and
gun, Orbium, the Aquarium unit and an Aquarium colony. Prints per scene the objects, the
largest and mean C, and the median S, M and I.

Run: uv run python scripts/goal_ranking.py   (~6 min on GPU)
     uv run --with matplotlib python scripts/goal_ranking.py --figure   (also writes
     assets/figures/goal_field.png: each scene beside its C field)
"""
import sys
import time
from pathlib import Path

import jax
import numpy as np

from complexity import collisions as co
from complexity import engine, registry
from complexity.goal import complexity_field


def scene(system_key, pattern, size, steps):
    system = registry.make_system(system_key)
    return system, pattern, engine.advance(system, system.init(pattern, (size, size), jax.random.key(0)), steps)


def goo():
    """Triscutium thrown into a patch of noise grows into a labyrinth (MEASURES §8.1c)."""
    unit = co.settle("lenia", "triscutium_valvatus_laterale")
    system = registry.make_system("lenia")
    A = np.zeros((256, 256), unit.cells.dtype)
    co.place(A, unit.cells, (128, 70))
    co.place(A, np.random.default_rng(0).random((unit.size, unit.size)), (128, 160))
    state = engine.advance(system, system.with_cells("triscutium_valvatus_laterale", A), 400)
    return system, "triscutium_valvatus_laterale", state


TRIALS = 3  # draws per hit: where the noise lands matters

SCENES = [
    ("empty", lambda: scene("baseline", "empty", 128, 1)),
    ("iid noise", lambda: scene("baseline", "iid_noise_continuous", 128, 1)),
    ("frozen noise", lambda: scene("baseline", "frozen_noise_continuous", 128, 1)),
    ("goo (Triscutium labyrinth)", goo),
    ("Life glider", lambda: scene("life", "glider", 64, 20)),
    ("Life Gosper gun", lambda: scene("life", "gosper_glider_gun", 128, 200)),
    ("Orbium", lambda: scene("lenia", "orbium", 256, 200)),
    ("Aquarium unit", lambda: scene("lenia_multi", "aquarium_unit", 256, 300)),
    ("Aquarium colony", lambda: scene("lenia_multi", "aquarium_scene", 256, 600)),
]

if __name__ == "__main__":
    print(f"{'scene':28s} {'objects':>7s} {'C max':>7s} {'C mean':>7s} {'S':>6s} {'M':>6s} "
          f"{'I':>6s}  (medians)")
    for name, make in SCENES:
        t = time.perf_counter()
        system, pattern, state = make()
        _, scores = complexity_field(system, pattern, state, trials=TRIALS)
        if not scores:
            print(f"{name:28s} {0:7d} {0:7.3f} {0:7.3f}")
            continue
        c = np.array([s.C for s in scores])
        S, M, I = (float(np.median([getattr(s, k) for s in scores])) for k in "SMI")
        print(f"{name:28s} {len(scores):7d} {c.max():7.3f} {c.mean():7.3f} {S:6.2f} {M:6.2f} "
              f"{I:6.2f}  {time.perf_counter() - t:.0f}s", flush=True)


def figure(path) -> None:
    """Each scene's frame beside its C field (one shared scale), for assets/figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = [s for s in SCENES if s[0] not in ("empty",)]
    _, axes = plt.subplots(len(shown), 2, figsize=(7, 3.2 * len(shown)))
    fields = []
    for name, make in shown:
        system, pattern, state = make()
        field, _ = complexity_field(system, pattern, state, trials=TRIALS)
        frame = np.asarray(system.render(state) if system.render(state) is not None
                           else system.observe(state))
        fields.append((name, frame, field))
    top = max(float(f.max()) for *_, f in fields) or 1
    for (name, frame, field), (left, right) in zip(fields, axes):
        left.imshow(frame, cmap=None if frame.ndim == 3 else "gray")
        left.set_title(name, fontsize=10)
        right.imshow(field, cmap="magma", vmin=0, vmax=top)
        right.set_title(f"C: max {field.max():.3f}", fontsize=10)
        left.axis("off")
        right.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=60)


if __name__ == "__main__" and "--figure" in sys.argv:
    figure(Path(__file__).resolve().parent.parent / "assets" / "figures" / "goal_field.png")
