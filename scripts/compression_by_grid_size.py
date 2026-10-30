"""Compare the compression measures across grid sizes at step 100 (findings in docs/measures.md §5.1).

Run: uv run python scripts/compression_by_grid_size.py   (~1 min on CPU)
"""
import time

from complexity.session import Run, RunSpec

keys = ["png", "lzma", "video_lzma", "video_h264", "novelty"]
print(f"{'size':>5} {'run':<26}" + "".join(f"{k:>11}" for k in keys) + "   ms/tick")
for n in (128, 256, 512):
    for s, p in [("life", "gosper_glider_gun"), ("life", "random"), ("lenia", "orbium"), ("lenia", "random")]:
        r = Run(RunSpec(s, p, (n, n)), keys)
        r.step(100)
        t = time.perf_counter(); v, _ = r.measure(); ms = (time.perf_counter() - t) * 1000
        print(f"{n:>5} {s + ':' + p:<26}" + "".join(f"{v[k]:>11.3f}" for k in keys) + f"   {ms:7.0f}")
