"""The rule-level measures over the rule space of elementary CA: Langton's λ, Wuensche's Z,
thermodynamic depth and the transient-length class (sides 20-200, as in Hudcová & Mikolov)
of the 88 rules that are the smallest of
their equivalence class (left-right mirror, 0-1 complement). Writes
results/eca_rule_space/rules.csv.

Run: uv run python scripts/eca_rule_space.py   (~6 min on CPU)
     uv run --with matplotlib python scripts/eca_rule_space.py --figure   (also writes
     assets/figures/eca_rule_space.png: rule space by Z, depth and the transient class,
     and the curated rules' space-time diagrams; --figure-only redraws it from the CSV)
"""
import csv
import sys
import time
from pathlib import Path

import jax
import numpy as np

from complexity import engine, registry
from complexity.measures.base import Context
from complexity.measures.rule import GROWTH_CLASSES, mirror
from complexity.systems.eca import CURATED, rule_table

OUT = Path("results/eca_rule_space")
MEASURES = ("langton_lambda", "wuensche_z", "thermodynamic_depth", "transient_length")
NOTABLE = (0, 18, 22, 30, 45, 54, 62, 73, 90, 105, 110, 150, 184, 204, 232)


def number(table: np.ndarray) -> int:
    return int(sum(int(b) << v for v, b in enumerate(table)))


def minimal_rules() -> list[int]:
    """The smallest rule number of each equivalence class: 88 rules."""
    def equivalents(r):
        t = rule_table(r)
        c = 1 - t[::-1]  # complement: outputs and inputs flipped
        return {number(t), number(mirror(t)), number(c), number(mirror(c))}

    return sorted({min(equivalents(r)) for r in range(256)})


# transient sides as in Hudcová & Mikolov's ECA classification
TRANSIENTS = {"min_size": 20, "max_size": 200, "n_sizes": 8}


def measure_all() -> list[dict]:
    eca = registry.make_system("eca")
    measures = {k: registry.make_measure(k) for k in MEASURES}
    measures["transient_length"] = registry.make_measure("transient_length", **TRANSIENTS)
    rows = []
    for r in minimal_rules():
        t = time.time()
        row = {"rule": r, **{k: m.compute_rule(eca, f"rule{r}", Context(2)) for k, m in measures.items()}}
        row["transient_class"] = GROWTH_CLASSES[int(row["transient_length"])]
        rows.append(row)
        print(f"rule {r:3d}: " + "  ".join(f"{k} {row[k]:.3f}" for k in MEASURES)
              + f"  ({row['transient_class']}, {time.time() - t:.1f} s)", flush=True)
    return rows


def figure(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    eca = registry.make_system("eca")
    fig = plt.figure(figsize=(15, 10), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=(1, 0.6))
    left, right = top.subplots(1, 2)
    jitter = np.random.default_rng(0).uniform(-1, 1, (len(rows), 2))
    z = np.array([row["wuensche_z"] for row in rows])
    depth = np.array([row["thermodynamic_depth"] for row in rows])
    cls = np.array([row["transient_length"] for row in rows])
    colours = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(GROWTH_CLASSES)))
    for c, name in enumerate(GROWTH_CLASSES):
        sel = cls == c
        left.scatter(z[sel] + 0.012 * jitter[sel, 0], depth[sel], s=30, color=colours[c],
                     label=f"transient {name}", alpha=0.8)
    points = right.scatter(cls + 0.15 * jitter[:, 0], depth, c=z, cmap="plasma", s=30, alpha=0.85)
    fig.colorbar(points, ax=right, label="Wuensche's Z")
    for ax, x in ((left, z + 0.012 * jitter[:, 0]), (right, cls + 0.15 * jitter[:, 0])):
        for i, row in enumerate(rows):
            if row["rule"] in NOTABLE:
                ax.annotate(str(row["rule"]), (x[i], depth[i]), fontsize=8,
                            xytext=(3, 3), textcoords="offset points")
        ax.set_ylabel("thermodynamic depth (bits/cell)")
    left.set_xlabel("Wuensche's Z")
    left.legend(fontsize=8)
    right.set_xticks(range(len(GROWTH_CLASSES)), GROWTH_CLASSES)
    right.set_xlabel("transient class (Hudcová & Mikolov, sides 20-200)")
    top.suptitle("The 88 elementary CA rules (one per equivalence class); points jittered sideways")
    by_rule = {row["rule"]: row for row in rows}
    panels = bottom.subplots(1, len(CURATED))
    for ax, r in zip(panels, CURATED):
        state = engine.advance(eca, eca.init(f"rule{r}", (96, 97), jax.random.key(0)), 95)
        ax.imshow(np.asarray(eca.observe(state)), cmap="gray_r", interpolation="nearest")
        row = by_rule[r]
        ax.set_title(f"rule {r}\nZ {row['wuensche_z']:.2f}  λ {row['langton_lambda']:.3g}\n"
                     f"depth {row['thermodynamic_depth']:.2f}\n{row['transient_class']}", fontsize=9)
        ax.set_axis_off()
    bottom.suptitle("Curated rules from a random row (97 cells, 96 steps, time downward)")
    fig.savefig(path, dpi=110)
    print(f"wrote {path}")


def read_rows() -> list[dict]:
    with open(OUT / "rules.csv") as f:
        return [{k: (v if k == "transient_class" else float(v) if "." in v else int(v))
                 for k, v in row.items()} for row in csv.DictReader(f)]


if __name__ == "__main__":
    if "--figure-only" in sys.argv:  # redraw from the last results
        figure(read_rows(), Path("assets/figures/eca_rule_space.png"))
        sys.exit()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = measure_all()
    with open(OUT / "rules.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT / 'rules.csv'}")
    if "--figure" in sys.argv:
        figure(rows, Path("assets/figures/eca_rule_space.png"))
