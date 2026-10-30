# Ported to JAX functions from the hand-coded Lenium (~/github/lenium.old, identical to
# ~/github/lenia-ref/code/lenium) at commit 9acb864f82f1d4bea048a985ef3d3da864fb973c:
# model.py `Lenia` and `IntegerLenia`, func.py `bell` / `calc_kernels` / `calc_growth_lookup`
# / `discretize` / `get_params`, util.py `direct_convolve`. Single channel and kernel, and
# multi-channel, multi-kernel `Lenia` (2-D). Replace with `import lenium` once it is
# published (PLAN.md §4.2).
"""Lenia and Integer Lenia, with Gaussian (`bell`) kernel shells and growth.

Kernel: rings of height b over the scaled distance D = |x| · len(b) / R, each ring shaped
bell(D mod 1, 0.5, 0.15); zero for D ≥ len(b). Growth: 2 · bell(U, m, s) − 1.

Differences from the original, none of which changes a result:
- Functions over arrays instead of classes over a data dict.
- The integer kernel is cropped to its support, (2R+1)², instead of covering the whole
  world; the original direct convolution with a world-size kernel is O(N⁴).
- `potential` is the direct convolution as a float32 `lax.conv` at HIGHEST precision, then
  rounded: every product and partial sum is an integer below 2²⁴, so float32 is exact.
- `potential_fft` is a float32 FFT convolution, then rounded. It is much faster on CPU and
  its error before rounding is < 0.01, far from the 0.5 that would change a value; tests
  check it is bit-identical to `potential`.
- The continuous kernel grid is centered as -(N//2)..N-N//2-1, which equals the original's
  for even N and also works for odd N.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def bell(x, m, s):
    return jnp.exp(-(((x - m) / s) ** 2) / 2.0)


def discretize(A, P=1):
    return jnp.round(A * P).astype(jnp.int32)


def ring_kernel(dist: jax.Array, R: float, b, r: float = 1.0) -> jax.Array:
    """Unnormalized kernel from distances to the origin, rings of heights `b` out to R · r
    (divided in the original's order, so the floats match)."""
    D = dist * len(b) / jnp.float32(R) / jnp.float32(r)
    return bell(D % 1, 0.5, 0.15) * jnp.asarray(b, jnp.float32)[D.astype(int)] * (D < len(b))


# ---- continuous Lenia ----


def kernel_fft(size: tuple[int, int], R: float, b) -> jax.Array:
    """rfft2 of the normalized kernel on a (height, width) torus."""
    h, w = size
    y, x = jnp.ogrid[-(h // 2) : h - h // 2, -(w // 2) : w - w // 2]
    K = ring_kernel(jnp.sqrt((x**2 + y**2).astype(jnp.float32)), R, b)
    return jnp.fft.rfft2(jnp.fft.ifftshift(K / K.sum()))


def step(A: jax.Array, fK: jax.Array, m: float, s: float, T: float, h: float = 1.0) -> jax.Array:
    """A ← clip(A + (2 · bell(K ∗ A, m, s) − 1) · h / T, 0, 1)."""
    U = jnp.fft.irfft2(jnp.fft.rfft2(A) * fK, s=A.shape)
    G = (bell(U, jnp.float32(m), jnp.float32(s)) * 2 - 1) * jnp.float32(h)
    return jnp.clip(A + G / jnp.float32(T), 0, 1)


# ---- multi-channel, multi-kernel Lenia (Chan 2020), e.g. the Aquarium ----


def multi_kernel_fft(size: tuple[int, int], R: float, kernels) -> jax.Array:
    """rfft2 of the normalized kernels on a (height, width) torus, (y x' k). Kernel i has
    rings `b` out to radius R · r (`calc_kernels` with ring heights)."""
    h, w = size
    I = jnp.mgrid[-(h // 2) : h - h // 2, -(w // 2) : w - w // 2]  # as calc_kernels
    dist = jnp.linalg.norm(I, axis=0)
    K = jnp.stack([ring_kernel(dist, R, k["b"], k.get("r", 1)) for k in kernels], axis=-1)
    K = K / jnp.sum(K, axis=(0, 1), keepdims=True)
    return jnp.fft.rfft2(jnp.fft.ifftshift(K, axes=(0, 1)), axes=(0, 1))


def multi_params(kernels, channels: int):
    """Per-kernel growth m, s, h (k) and the maps channel -> kernel Mk (c k) and kernel ->
    channel Mc (k c) from source c0 and target c1 (`get_params`; absent values are 1, and
    absent channels 0)."""
    m, s, h = (jnp.array([k.get(key, 1) for k in kernels], jnp.float32) for key in "msh")
    Mk = jnp.zeros((channels, len(kernels)), jnp.float32)
    Mc = jnp.zeros((len(kernels), channels), jnp.float32)
    for i, k in enumerate(kernels):
        Mk, Mc = Mk.at[k.get("c0", 0), i].set(1.0), Mc.at[i, k.get("c1", 0)].set(1.0)
    return m, s, h, Mk, Mc


def multi_step(A, fK, m, s, h, Mk, Mc, T) -> jax.Array:
    """A (y x c) ← clip(A + Σ_k (2 · bell(K_k ∗ A_c0(k), m_k, s_k) − 1) · h_k → c1(k) / T, 0, 1)."""
    fA = jnp.fft.rfft2(A, axes=(0, 1))  # (y x' c)
    U = jnp.fft.irfft2(fA @ Mk * fK, s=A.shape[:2], axes=(0, 1))  # potentials (y x k)
    G = (bell(U, m, s) * 2 - 1) * h @ Mc  # growth (y x c)
    return jnp.clip(A + G / jnp.float32(T), 0, 1)


# ---- Integer Lenia ----


def integer_kernel(R: int, P: int, b=(1.0,)) -> jax.Array:
    """Integer kernel weights 0..P, cropped to (2R+1)²."""
    I = jnp.mgrid[-R : R + 1, -R : R + 1]  # coordinates (d y x)
    return discretize(ring_kernel(jnp.linalg.norm(I, axis=0), R, b), P)


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


def integer_kernel_fft(K: jax.Array, size: tuple[int, int]) -> jax.Array:
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


def integer_update(A: jax.Array, U: jax.Array, g: jax.Array, T: int, P: int) -> jax.Array:
    """A ← clip(A + round(g[U] / T), 0, P) for potential U = K ∗ A, all integer."""
    G = discretize(g[U] / T)
    return jnp.clip(A + G, 0, P)
