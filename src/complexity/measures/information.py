"""Information-theoretic measures on discretized frames: transfer entropy (TE, in JAX on
the device) and block entropy (SE, on the host).

All estimates are plug-in: probabilities are histogram frequencies over the cells (and time
steps) of the window, with the Miller-Madow bias correction. The bias grows with bins /
samples (docs/measures.md §5.3), so values are compared with the `baseline` system's iid noise of the
same kind.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from complexity.measures.base import NO_CONTEXT, Measure
from complexity.registry import register_measure


def _entropy_terms(index: jax.Array, weights: jax.Array, length: int):
    """(Σ c · log2 c over the bins, number of occupied bins) of the weighted counts c."""
    c = jnp.bincount(index.ravel(), weights.ravel(), length=length)
    return jnp.sum(jnp.where(c > 0, c * jnp.log2(jnp.where(c > 0, c, 1)), 0)), jnp.sum(c > 0)


@partial(jax.jit, static_argnums=(2, 3, 4))
def transfer_entropy(symbols: jax.Array, n_valid, n_levels: int, offset: int,
                     correct_bias: bool = True) -> jax.Array:
    """Apparent transfer entropy (history k = 1) from each cell's neighbour at distance
    `offset` to the cell, averaged over the 4 von Neumann directions; bits per cell per step.

    `symbols` is int32 (time, height, width) in 0..n_levels-1, on a torus; only its last
    `n_valid` frames count (traced, so a filling window does not recompile). With N samples
    and counts c, T = H(Y'|Y) - H(Y'|Y,X) = (Σ c(y'yx) log c(y'yx) - Σ c(yx) log c(yx)
    - Σ c(y'y) log c(y'y) + Σ c(y) log c(y)) / N. `correct_bias` applies the Miller-Madow
    correction to each entropy, (occupied bins - 1) / 2N nats, which removes the plug-in
    bias of iid noise: (L - 1)² L / 2N nats when all bins are occupied.
    """
    L = n_levels
    y_next, y = symbols[1:], symbols[:-1]
    t = jnp.arange(y.shape[0])[:, None, None]
    w = jnp.broadcast_to(t >= y.shape[0] - (n_valid - 1), y.shape).astype(jnp.float32)
    n = jnp.maximum(jnp.sum(w), 1)
    s_next, b_next = _entropy_terms(y_next * L + y, w, L * L)
    s_past, b_past = _entropy_terms(y, w, L)
    te = 0.0
    for shift in ((offset, 0), (-offset, 0), (0, offset), (0, -offset)):
        x = jnp.roll(y, shift, (1, 2))
        s_joint, b_joint = _entropy_terms((y_next * L + y) * L + x, w, L**3)
        s_pair, b_pair = _entropy_terms(y * L + x, w, L * L)
        te += (s_joint - s_pair - s_next + s_past) / n
        if correct_bias:
            te += (b_next - b_past - b_joint + b_pair) / (2 * n * jnp.log(2))
    return jnp.maximum(te / 4, 0.0)


@partial(jax.jit, static_argnums=(2, 3))
def local_transfer_entropy(symbols: jax.Array, n_valid, n_levels: int, offset: int) -> jax.Array:
    """Local transfer entropy (Lizier et al. 2008) per cell, averaged over the window's
    valid steps and the 4 directions: log2 p(y' | y, x) / p(y' | y) for each sample, with
    the probabilities of `transfer_entropy` (no bias correction); (height, width)."""
    L = n_levels
    y_next, y = symbols[1:], symbols[:-1]
    t = jnp.arange(y.shape[0])[:, None, None]
    w = jnp.broadcast_to(t >= y.shape[0] - (n_valid - 1), y.shape).astype(jnp.float32)
    nxt = y_next * L + y
    c_next = jnp.bincount(nxt.ravel(), w.ravel(), length=L * L)
    total = jnp.zeros(y.shape[1:])
    for shift in ((offset, 0), (-offset, 0), (0, offset), (0, -offset)):
        x = jnp.roll(y, shift, (1, 2))
        joint, pair = nxt * L + x, y * L + x
        c_joint = jnp.bincount(joint.ravel(), w.ravel(), length=L**3)
        c_pair = jnp.bincount(pair.ravel(), w.ravel(), length=L * L)
        c_past = jnp.bincount(y.ravel(), w.ravel(), length=L)
        ratio = c_joint[joint] * c_past[y] / (c_pair[pair] * c_next[nxt])
        total += jnp.sum(jnp.where(w > 0, jnp.log2(jnp.where(w > 0, ratio, 1)), 0), axis=0)
    return total / (4 * jnp.maximum(n_valid - 1, 1))


@register_measure
class TransferEntropy(Measure):
    key = "transfer_entropy"
    name = "Transfer entropy"
    unit = "bits/cell/step"
    group = "Information theory"
    description = (
        "Information a neighbour's state adds about a cell's next state beyond the cell's "
        "own state (Schreiber 2000; local TE for CA: Lizier et al. 2008), averaged over the "
        "4 neighbours at distance `offset`, all cells and the window's steps. 0 for empty, "
        "still and frozen patterns, ~0 for iid noise (nothing is predictable), highest for "
        "moving structure. History length 1, so it mixes in some information storage. "
        "Plug-in estimate with the Miller-Madow bias correction; clipped at 0."
    )
    window = 16
    backend = "jax"
    needs_discrete = True
    peaks_at = "structure"
    offset = 1
    correct_bias = True
    params = ("window", "offset", "n_levels", "correct_bias")
    scope = "local"
    temporal = "dynamic"

    map = "local"

    def _padded(self, frames):
        """A filling window padded to the full shape (no recompile), and its valid length."""
        n = frames.shape[0]
        if n < self.window:  # count only the n real frames
            frames = jnp.concatenate([jnp.zeros_like(frames[:1]).repeat(self.window - n, 0), frames])
        return frames, n

    def compute(self, frames, ctx=NO_CONTEXT) -> float:
        if frames.shape[0] < 2:
            return 0.0
        frames, n = self._padded(frames)
        return float(transfer_entropy(frames, n, ctx.n_levels, self.offset, self.correct_bias))

    def compute_map(self, frames, ctx=NO_CONTEXT) -> np.ndarray:
        if frames.shape[0] < 2:
            return np.zeros(frames.shape[1:], np.float32)
        frames, n = self._padded(frames)
        return np.asarray(local_transfer_entropy(frames, n, ctx.n_levels, self.offset))


def block_codes(symbols: np.ndarray, n_levels: int, block: int) -> np.ndarray:
    """The `block` x `block` patch at every cell (on the torus) as one integer."""
    codes = np.zeros(symbols.shape, np.int64)
    for dy in range(block):
        for dx in range(block):
            codes = codes * n_levels + np.roll(symbols, (-dy, -dx), (0, 1))
    return codes


def block_entropy(symbols: np.ndarray, n_levels: int, block: int) -> float:
    """Entropy of the `block` x `block` blocks at every cell (overlapping, on the torus),
    with the Miller-Madow correction, in bits per cell."""
    codes = block_codes(symbols, n_levels, block)
    _, counts = np.unique(codes, return_counts=True)
    n = codes.size
    h = np.log2(n) - np.sum(counts * np.log2(counts)) / n + (len(counts) - 1) / (2 * n * np.log(2))
    return float(h / block**2)


@register_measure
class BlockEntropy(Measure):
    key = "block_entropy"
    name = "Block entropy"
    unit = "bits/cell"
    group = "Information theory"
    description = (
        "Shannon entropy of the discretized frame's `block` x `block` patches (every cell's "
        "patch, on the torus), per cell. `block = 1` is the entropy of single cell states; "
        "larger blocks also count spatial correlations. 0 for a uniform frame, log2(levels) "
        "for iid noise (less for many levels: there are fewer cells than possible blocks)."
    )
    needs_discrete = True
    block = 2
    params = ("block", "n_levels")

    map = "local"

    def compute(self, frames, ctx=NO_CONTEXT) -> float:
        return block_entropy(frames[-1], ctx.n_levels, self.block)

    def compute_map(self, frames, ctx=NO_CONTEXT) -> np.ndarray:
        """Local surprisal of each cell's block, -log2 p(block) / block², whose mean is
        the plug-in block entropy."""
        codes = block_codes(frames[-1], ctx.n_levels, self.block)
        _, inverse, counts = np.unique(codes, return_inverse=True, return_counts=True)
        p = counts[inverse.reshape(codes.shape)] / codes.size
        return (-np.log2(p) / self.block**2).astype(np.float32)
