"""Lenia (Chan 2019): continuous cellular automaton, single channel, single kernel.

Polynomial kernel core and growth function with hard clipping, as in the original Lenia;
patterns use the same RLE and parameters as Lenium's creature library.
"""

from __future__ import annotations

import numpy as np

from complexity.registry import register_system
from complexity.systems.base import System, place_center

CREATURES = {
    "orbium": {
        "R": 13, "T": 10, "m": 0.15, "s": 0.015, "b": [1.0],
        "cells": (
            "7.MD6.qL$6.pKqEqFURpApBRAqQ$5.VqTrSsBrOpXpWpTpWpUpCrQ$4.CQrQsTsWsApITNPpGqGvL$"
            "3.IpIpWrOsGsBqXpJ4.LsFrL$A.DpKpSpJpDqOqUqSqE5.ExD$qL.pBpTT2.qCrGrVrWqM5.sTpP$"
            ".pGpWpD3.qUsMtItQtJ6.tL$.uFqGH3.pXtOuR2vFsK5.sM$.tUqL4.GuNwAwVxBwNpC4.qXpA$"
            "2.uH5.vBxGyEyMyHtW4.qIpL$2.wV5.tIyG3yOxQqW2.FqHpJ$2.tUS4.rM2yOyJyOyHtVpPMpFqNV$"
            "2.HsR4.pUxAyOxLxDxEuVrMqBqGqKJ$3.sLpE3.pEuNxHwRwGvUuLsHrCqTpR$"
            "3.TrMS2.pFsLvDvPvEuPtNsGrGqIP$4.pRqRpNpFpTrNtGtVtStGsMrNqNpF$"
            "5.pMqKqLqRrIsCsLsIrTrFqJpHE$6.RpSqJqPqVqWqRqKpRXE$8.OpBpIpJpFTK!"
        ),
    },
}


def _value(token: str) -> int:
    if token in ".b":
        return 0
    if token == "o":
        return 255
    if len(token) == 1:
        return ord(token) - ord("A") + 1
    return (ord(token[0]) - ord("p")) * 24 + (ord(token[1]) - ord("A") + 25)


def rle2arr(text: str) -> np.ndarray:
    """Decode a 2-D Lenia RLE string to a float32 array in [0, 1]."""
    rows, row, count, lead = [], [], "", ""
    for char in text.rstrip("!") + "$":
        if char.isdigit():
            count += char
        elif char in "pqrstuvwxy":
            lead = char
        else:
            n = int(count) if count else 1
            if char == "$":
                rows.append(row)
                rows.extend([[]] * (n - 1))
                row = []
            else:
                row.extend([_value(lead + char) / 255] * n)
            count, lead = "", ""
    width = max(len(r) for r in rows)
    return np.array([r + [0.0] * (width - len(r)) for r in rows], dtype=np.float32)


def kernel_fft(size: tuple[int, int], R: float, b: list[float]) -> np.ndarray:
    h, w = size
    y, x = np.ogrid[-(h // 2) : h - h // 2, -(w // 2) : w - w // 2]
    D = np.sqrt(x**2 + y**2) / R * len(b)
    ring = np.minimum(D.astype(int), len(b) - 1)
    r = D % 1
    K = (D < len(b)) * np.asarray(b)[ring] * (4 * r * (1 - r)) ** 4
    K /= K.sum()
    return np.fft.rfft2(np.fft.ifftshift(K))


@register_system
class Lenia(System):
    key = "lenia"
    name = "Lenia"
    state_type = "continuous"
    patterns = {
        "orbium": "Orbium (glider)",
        "random": "Random soup (Orbium params)",
    }

    def reset(self, pattern: str) -> None:
        creature = CREATURES["orbium" if pattern == "random" else pattern]
        self.R, self.T = creature["R"], creature["T"]
        self.m, self.s = creature["m"], creature["s"]
        self.fK = kernel_fft(self.size, self.R, creature["b"])
        self.A = np.zeros(self.size, dtype=np.float32)
        if pattern == "random":
            h, w = self.size
            place_center(self.A, self.rng.random((h * 3 // 4, w * 3 // 4), dtype=np.float32))
        else:
            place_center(self.A, rle2arr(creature["cells"]))

    def growth(self, U: np.ndarray) -> np.ndarray:
        return np.maximum(0, 1 - (U - self.m) ** 2 / (9 * self.s**2)) ** 4 * 2 - 1

    def step(self) -> None:
        U = np.fft.irfft2(self.fK * np.fft.rfft2(self.A), s=self.size)
        self.A = np.clip(self.A + self.growth(U) / self.T, 0, 1).astype(np.float32)

    def frame(self) -> np.ndarray:
        return self.A
