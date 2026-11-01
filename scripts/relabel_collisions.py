"""Relabel results/collisions/collisions.jsonl with the current `collisions.label` (the
records hold everything it reads), without rerunning the experiments; prints the summary.

Run: uv run python scripts/relabel_collisions.py
"""
import json
from collections import Counter
from pathlib import Path

from complexity import collisions as co

PATH = Path(__file__).resolve().parent.parent / "results" / "collisions" / "collisions.jsonl"
records = [json.loads(line) for line in PATH.read_text().splitlines()]
for r in records:
    r["label"] = co.label(co.Collision(**r))
PATH.write_text("".join(json.dumps(r) + "\n" for r in records))
for pattern in dict.fromkeys(r["pattern"] for r in records):
    own = [r for r in records if r["pattern"] == pattern]
    pairs = Counter(r["label"] for r in own if r["kind"] == "pair")
    noise, alone = (next(r["label"] for r in own if r["kind"] == k) for k in ("noise", "alone"))
    print(f"{pattern:30s} pairs {dict(pairs)}  noise: {noise}  alone: {alone}")
