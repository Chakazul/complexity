"""Collision experiments (MEASURES.md §8.1b): each unit against a copy of itself at several
headings, offsets and chiralities, and against a patch of noise; outcomes labelled from the
objects detected in K ∗ A. Writes results/collisions/collisions.jsonl and prints a summary.

Run: uv run python scripts/collisions.py   (~6 min on GPU)
     uv run python scripts/collisions.py --gifs   (also an animated GIF per unit of the
     90° collision without offset, and Orbium's head-on one, in results/collisions/)
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

from complexity import collisions as co

UNITS = [("lenia", "orbium"), ("lenia", "triscutium_valvatus_laterale"),
         ("lenia_multi", "aquarium_unit"), ("life", "glider")]
OUT = Path(__file__).resolve().parent.parent / "results" / "collisions"

OUT.mkdir(parents=True, exist_ok=True)
records = []
for system, pattern in UNITS:
    t = time.perf_counter()
    unit = co.settle(system, pattern)
    deltas = (180, 90) if system == "life" else (180, 135, 90)  # Life turns in quarter turns
    runs = [("pair", d, b, m) for d in deltas for b in (0.0, 0.25, 0.5) for m in (False, True)]
    runs += [("noise", 180, 0.0, False), ("alone", 0, 0.0, False)]
    for kind, delta, offset, mirror in runs:
        gif = "--gifs" in sys.argv and kind == "pair" and offset == 0 and not mirror and (
            delta == 90 or (delta == 180 and pattern == "orbium"))
        frames = [] if gif else None
        result = co.collide(unit, kind, delta, offset, mirror, frames=frames)
        records.append(result.record())
        if gif:
            co.save_gif(frames, OUT / f"{pattern}_{delta}_{result.label}.gif")
    own = [r for r in records if r["pattern"] == pattern]
    pairs = Counter(r["label"] for r in own if r["kind"] == "pair")
    noise, alone = (next(r["label"] for r in own if r["kind"] == k) for k in ("noise", "alone"))
    print(f"{pattern:30s} pairs {dict(pairs)}  noise: {noise}  alone: {alone}  "
          f"{time.perf_counter() - t:.0f}s", flush=True)
with open(OUT / "collisions.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in records)
print(f"wrote {OUT / 'collisions.jsonl'} ({len(records)} runs)")
