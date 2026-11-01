"""Algorithmic-complexity measures beyond compressed size (MEASURES.md §5.2): Lempel-Ziv
complexity (AC), and proxies for effective complexity (EC) and logical depth (LD).

All run on the host. LZC reads discretized symbols; the other two read [0, 1] frames.
"""

from __future__ import annotations

import lzma
import statistics
import time

import numpy as np

from complexity.measures.base import NO_CONTEXT, Measure, to_uint8
from complexity.registry import register_measure

_LMAX = 48  # phrase lengths found by hashing; longer ones by bytes.find


def lz76(symbols: np.ndarray) -> int:
    """Lempel-Ziv (1976) complexity: the number of phrases in the exhaustive-history parse
    of the flattened symbols (Kaspar & Schuster 1987), each phrase the shortest substring
    not seen before (overlap allowed).

    A phrase starting at i extends while s[i:i+l] also starts at some p < i. For each l,
    one vectorized pass of polynomial rolling hashes marks those positions, so the phrase
    loop only does lookups: O(n · max phrase length) instead of a quadratic history scan.
    Phrases longer than `_LMAX` (long repeats, empty grids) fall back to bytes.find.
    """
    s = np.ascontiguousarray(np.ravel(symbols), np.uint8)
    n, raw = len(s), s.tobytes()
    base = np.uint64(0x9E3779B97F4A7C15)
    vals = s.astype(np.uint64) + np.uint64(1)
    h = np.zeros(n, np.uint64)
    seen = [None]  # seen[l][i]: s[i:i+l] also starts at some p < i

    def level(l: int) -> np.ndarray:
        nonlocal h
        while len(seen) <= l:
            m = len(seen)
            h = h[: n - m + 1] * base + vals[m - 1 :]  # wraps mod 2^64
            _, first, inverse = np.unique(h, return_index=True, return_inverse=True)
            seen.append(first[inverse] < np.arange(n - m + 1))
        return seen[l]

    def found(i: int, l: int) -> bool:
        return raw.find(raw[i : i + l], 0, i + l - 1) != -1

    i = c = 0
    with np.errstate(over="ignore"):
        while i < n:
            l = 0
            while l < n - i and l < _LMAX and level(l + 1)[i]:
                l += 1
            if l == _LMAX:  # a long phrase: exponential, then binary search
                lo, hi = l, 2 * l
                while i + hi <= n and found(i, hi):
                    lo, hi = hi, 2 * hi
                hi = min(hi, n - i + 1)
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    lo, hi = (mid, hi) if found(i, mid) else (lo, mid)
                l = lo
            c += 1
            i += l + 1
    return c


@register_measure
class LempelZiv(Measure):
    key = "lzc"
    icon = "🔤"
    equation = r"C_{LZ}(x) = c(x)\,\frac{\log_L n}{n}"
    reference = "Lempel & Ziv 1976; Kaspar & Schuster 1987"
    map = "tile"
    name = "Lempel-Ziv complexity"
    unit = "relative to noise"
    group = "Algorithmic complexity"
    ratio = "mass"
    ratio_unit = "symbols/mass"
    description = (
        "Lempel-Ziv (1976) phrase count c of the discretized frame read row by row, "
        "normalized as c · log_L(n) / n for n cells and L levels: ~0 for a uniform frame, "
        "~1 for iid noise (as used for Lenia by Adachi et al. 2024)."
    )
    needs_discrete = True
    params = ("n_levels",)

    def compute(self, frames, ctx=NO_CONTEXT) -> float:
        symbols = frames[-1]
        n, L = symbols.size, max(ctx.n_levels or 2, 2)
        return lz76(symbols) * np.log(n) / np.log(L) / n


def blur(frame: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur on the torus, by FFT."""
    h, w = frame.shape
    ky, kx = np.fft.fftfreq(h)[:, None], np.fft.rfftfreq(w)[None, :]
    gain = np.exp(-2 * (np.pi * sigma) ** 2 * (ky**2 + kx**2))
    return np.fft.irfft2(np.fft.rfft2(frame) * gain, s=frame.shape)


def regular_part(frame: np.ndarray, sigma: float, z: float) -> np.ndarray:
    """The frame coarse-grained to 3 levels: blurred values more than `z` standard
    deviations below or above the mean, or within. The deviation is that of iid cells with
    the frame's mean and variance after the same blur, std · sqrt(Σ w²) with Σ w² ≈
    1 / (4π sigma²), so noise of any kind lands in the middle level."""
    smooth = blur(frame, sigma)
    band = z * frame.std() / np.sqrt(4 * np.pi * sigma**2)
    mean = frame.mean()
    return ((smooth > mean + band).astype(np.uint8) + (smooth >= mean - band)).astype(np.uint8)


@register_measure
class EffectiveComplexity(Measure):
    key = "effective_complexity"
    icon = "💎"
    equation = r"\mathcal{E}(x) = K(\tilde E_x),\qquad \Sigma = \mathcal{E} + S(\tilde E_x) \;\to\; \min"
    reference = "Gell-Mann & Lloyd 1996"
    name = "Effective complexity (proxy)"
    unit = "bits/cell"
    group = "Algorithmic complexity"
    ratio = "mass"
    ratio_unit = "bits/mass"
    description = (
        "LZMA size of the frame's regular part (Gell-Mann & Lloyd): the frame blurred "
        "(Gaussian, `sigma` cells) and cut into 3 levels, below / within / above the band "
        "that iid noise with the same cell statistics would fill (`z` standard deviations). "
        "Noise and uniform frames collapse to one level (~0 bits); what remains is structure "
        "larger than the blur that noise cannot explain."
    )
    sigma = 2.0
    z = 4.0
    params = ("sigma", "z")
    peaks_at = "structure"

    def compute(self, frames, ctx=NO_CONTEXT) -> float:
        coarse = regular_part(frames[-1], self.sigma, self.z)
        return len(lzma.compress(coarse.tobytes())) * 8 / coarse.size


@register_measure
class LogicalDepth(Measure):
    key = "logical_depth"
    icon = "⏱️"
    equation = r"D_s(x) = \min\{\,T(p) : U(p) = x,\ |p| - |p^*| < s\,\}"
    reference = "Bennett 1988; proxy: Zenil, Delahaye & Gaucherel 2012"
    name = "Logical depth (proxy)"
    unit = "ns/cell"
    group = "Algorithmic complexity"
    description = (
        "Time to decompress the LZMA-compressed frame (8-bit gray), median of `repeats` "
        "runs, per cell: Zenil et al.'s proxy for Bennett's logical depth, the run time of "
        "a near-shortest program. Random frames are shallow too: LZMA stores "
        "incompressible data as is, which decompresses fastest. Timing is machine-dependent "
        "and noisy; compare runs measured in one session."
    )
    peaks_at = "mixed"
    repeats = 5
    params = ("repeats",)

    def compute(self, frames, ctx=NO_CONTEXT) -> float:
        frame = to_uint8(frames[-1])
        packed = lzma.compress(frame.tobytes())
        times = []
        for _ in range(self.repeats):
            t0 = time.perf_counter_ns()
            lzma.decompress(packed)
            times.append(time.perf_counter_ns() - t0)
        return statistics.median(times) / frame.size
