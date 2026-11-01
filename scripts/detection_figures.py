"""Figures for object detection in the potential K * A (MEASURES.md §5.4, Q18), saved to
assets/figures/: detection.png compares thresholding the frame A, thresholding K * A and
the basins of K * A's peaks; tracking.png follows a target unit in the Aquarium scene.

Run: uv run --with matplotlib python scripts/detection_figures.py   (~1 min)
"""
from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from complexity import engine, registry
from complexity.measures.individuality import (
    FeatureIndividuality,
    detect,
    follow_target,
    torus_label,
    track_features,
)
from complexity.session import Run, RunSpec

OUT = Path(__file__).resolve().parent.parent / "assets" / "figures"
SADDLE, THRESHOLD = FeatureIndividuality.saddle, FeatureIndividuality.threshold
CASES = [("lenia_multi", "aquarium_scene", 256, 500), ("lenia_multi", "aquarium_scene", 256, 2000),
         ("lenia_multi", "aquarium", 256, 1000), ("lenia", "orbium", 128, 300),
         ("lenia", "paraptera_cavus_furiosus", 128, 300), ("lenia", "hydrogeminium_natans", 128, 300),
         ("lenia", "amoebium_foliatus", 256, 300), ("life", "gosper_glider_gun", 128, 300)]


def colours(labels: np.ndarray) -> np.ndarray:
    ids, inverse = np.unique(labels, return_inverse=True)
    palette = np.random.default_rng(1).uniform(0.25, 1, (len(ids), 3))
    palette[ids == 0] = 0
    return palette[inverse.reshape(labels.shape)]


def count(labels: np.ndarray) -> int:
    return len(np.unique(labels[labels > 0]))


def detection_figure() -> None:
    fig, axes = plt.subplots(len(CASES), 5, figsize=(16, 3.3 * len(CASES)))
    for row, (system_key, pattern, n, steps) in zip(axes, CASES):
        system = registry.make_system(system_key)
        state = engine.advance(system, system.init(pattern, (n, n), jax.random.key(0)), steps)
        A = np.asarray(system.observe(state))
        U = np.asarray(jax.jit(system.potential)(state), np.float64)
        reach = system.reach(pattern)
        by_frame, by_level = torus_label(A > 0.1), torus_label(U > 0.1)
        by_basins = detect(A, U, THRESHOLD, SADDLE, np.pi * (reach / 4) ** 2)
        name = system.patterns[pattern].split(" (")[0]
        panels = [(A, "gray", f"{name}, step {steps}: A"), (U, "magma", "K ∗ A"),
                  (colours(by_frame), None, f"A > 0.1: {count(by_frame)}"),
                  (colours(by_level), None, f"K ∗ A > 0.1: {count(by_level)}"),
                  (colours(by_basins), None, f"K ∗ A basins (saddle {SADDLE}): {count(by_basins)}")]
        for ax, (image, cmap, title) in zip(row, panels):
            ax.imshow(image, cmap=cmap)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "detection.png", dpi=60)
    plt.close(fig)


def tracking_figure() -> None:
    run = Run(RunSpec("lenia_multi", "aquarium_scene", (256, 256)), ["iti_features"])
    run.step(600)
    frames, potentials = run.host_frames(), run.host_potentials()
    labels = [detect(f, u, THRESHOLD, SADDLE, np.pi * 9) for f, u in zip(frames, potentials)]
    path, target = [], None
    for target in follow_target(frames, labels, "typical"):
        if target is None:
            break
        ys, xs = np.nonzero(target)
        path.append((ys.mean(), xs.mean()))
    feats = track_features(frames, labels, "near", run.system.reach("aquarium_scene"))
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.5))
    left.imshow(frames[-1], cmap="gray")
    track = np.array(path)
    left.plot(track[:, 1], track[:, 0], ".", color="tab:red", ms=2)
    if target is not None:
        left.contour(target, levels=[0.5], colors="tab:red", linewidths=1)
    left.set_title("Aquarium scene, steps 473-600: the target unit's track (red)")
    left.axis("off")
    right.plot(feats[:, 0], label="S: the target's mass")
    right.plot(feats[:, 1], label="E: mass of creatures within reach R")
    right.set_xlabel("step in the 128-step window")
    right.legend()
    plt.tight_layout()
    plt.savefig(OUT / "tracking.png", dpi=60)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    detection_figure()
    tracking_figure()
    print(f"wrote {OUT / 'detection.png'} and {OUT / 'tracking.png'}")
