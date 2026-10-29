# Ported to JAX from the predecessor Lenium library (~/github/lenia-ref/code/lenium, not
# under git): model.py `IntegerLenia`, func.py `bell` / `calc_kernels` / `calc_growth_lookup`
# / `discretize`, util.py `direct_convolve`. Single channel, single kernel. Lenium has no
# Integer Lenia engine yet; replace with `import lenium` once it has (PLAN.md §4.2).
"""Integer Lenia: integer states, kernel weights, potential and growth (PLAN.md §4.2).

Differences from the predecessor, none of which changes a result:
- The kernel is cropped to its support, (2⌈rR⌉+1)², instead of covering the whole world;
  the predecessor's direct convolution with a world-size kernel is O(N⁴).
- `potential` is the direct convolution as a float32 `lax.conv` at HIGHEST precision, then
  rounded: every product and partial sum is an integer below 2²⁴, so float32 is exact.
- `potential_fft` is a float32 FFT convolution, then rounded. It is much faster on CPU and
  its error before rounding is < 0.01, far from the 0.5 that would change a value; tests
  check it is bit-identical to `potential`.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def bell(x, m, s):
    return jnp.exp(-(((x - m) / s) ** 2) / 2.0)


def discretize(A, P=1):
    return jnp.round(A * P).astype(jnp.int32)


def kernel(R: int, P: int, b=(1.0,), r: float = 1.0) -> jax.Array:
    """Integer kernel weights 0..P, (2⌈rR⌉+1)², ring shape bell(D mod 1, 0.5, 0.15)."""
    half = math.ceil(r * R)
    I = jnp.mgrid[-half : half + 1, -half : half + 1]  # coordinates (d y x)
    D = jnp.linalg.norm(I, axis=0)  # distance from origin (y x)
    D = D * len(b) / R / jnp.float32(r)  # scaled distance
    K = bell(D % 1, 0.5, 0.15) * jnp.asarray(b, jnp.float32)[D.astype(int)] * (D < len(b))
    return discretize(K, P)


def growth_lookup(K: jax.Array, P: int, m: float, s: float, h: float = 1.0) -> jax.Array:
    """Integer growth g[u] in -P..P for every reachable potential u = 0..P·ΣK."""
    U_len = int(P * K.sum())
    U = jnp.pad(jnp.linspace(0, 1, U_len), [0, 1], constant_values=1)
    g = (bell(U, jnp.float32(m), jnp.float32(s)) * 2 - 1) * jnp.float32(h)
    return discretize(g, P)


def potential(A: jax.Array, K: jax.Array) -> jax.Array:
    """U = K ∗ A on a torus, by direct convolution; int32 (y x)."""
    pad = K.shape[0] // 2
    Ap = jnp.pad(A, pad, mode="wrap").astype(jnp.float32)
    U = jax.lax.conv(
        Ap[None, None], K.astype(jnp.float32)[None, None], (1, 1), "VALID",
        precision=jax.lax.Precision.HIGHEST,
    )[0, 0]
    return jnp.round(U).astype(jnp.int32)


def kernel_fft(K: jax.Array, size: tuple[int, int]) -> jax.Array:
    """rfft2 of the integer kernel centered on a (height, width) torus, for `potential_fft`."""
    h, w = size
    r = K.shape[0] // 2
    full = jnp.zeros(size, jnp.float32)
    full = full.at[h // 2 - r : h // 2 + r + 1, w // 2 - r : w // 2 + r + 1].set(K)
    return jnp.fft.rfft2(jnp.fft.ifftshift(full))


def potential_fft(A: jax.Array, fK: jax.Array) -> jax.Array:
    """`potential` by FFT, rounded back to the exact integers; int32 (y x)."""
    U = jnp.fft.irfft2(jnp.fft.rfft2(A.astype(jnp.float32)) * fK, s=A.shape)
    return jnp.round(U).astype(jnp.int32)


def update(A: jax.Array, U: jax.Array, g: jax.Array, T: int, P: int) -> jax.Array:
    """A ← clip(A + round(g[U] / T), 0, P) for potential U = K ∗ A, all integer."""
    G = discretize(g[U] / T)
    return jnp.clip(A + G, 0, P)
