"""The M0 NumPy implementations of the system steps, kept as the reference for the JAX
parity tests (PLAN.md §6) after they were removed from the package."""

import numpy as np


def life_step(g: np.ndarray) -> np.ndarray:
    n = sum(
        np.roll(np.roll(g, dy, 0), dx, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if dy or dx
    )
    return ((n == 3) | ((g == 1) & (n == 2))).astype(np.uint8)


def lenia_kernel_fft(size, R, b):
    h, w = size
    y, x = np.ogrid[-(h // 2) : h - h // 2, -(w // 2) : w - w // 2]
    D = np.sqrt(x**2 + y**2) / R * len(b)
    ring = np.minimum(D.astype(int), len(b) - 1)
    r = D % 1
    K = (D < len(b)) * np.asarray(b)[ring] * (4 * r * (1 - r)) ** 4
    K /= K.sum()
    return np.fft.rfft2(np.fft.ifftshift(K))


def lenia_step(A, fK, T, m, s):
    U = np.fft.irfft2(fK * np.fft.rfft2(A), s=A.shape)
    G = np.maximum(0, 1 - (U - m) ** 2 / (9 * s**2)) ** 4 * 2 - 1
    return np.clip(A + G / T, 0, 1).astype(np.float32)
