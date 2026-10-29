# Vendored from Lenium (~/github/lenium), src/lenium/core/profiles.py at commit 139492ce0227ec2556f6455901f681676cfaace7.
# Copied verbatim; replace with `import lenium` once Lenium is published (PLAN.md §4.2).
"""Orthogonal 1-D bump-profile registry shared by the kernel and growth cores (§4.7).

`(r, w, b)` is to the kernel what `(m, s, h)` is to growth (§4.1): both triples
parameterize the *same* family of 1-D bumps — centre, width, height — one over
kernel radius, the other over potential. So one registry serves both axes, and a
new profile is one function plus one `register()` call rather than an edit to
every variant.

Coordinate convention
---------------------
Every profile is a pure function of the normalized coordinate

    t = (x - centre) / width

and returns a value in `[0, 1]`. Compact profiles have support exactly
`|t| <= 1`; the Gaussian uses the width as its sigma and has unbounded support.

The reference implementations state the same convention on a `[0, 1]` domain:
`LeniaF.py`'s `kernel_core` is evaluated at `u = (x - r) / (2w) + 1/2`, which is
`u = (t + 1) / 2`. The formulas below are those cores rewritten in `t`.

Per-axis width scale
--------------------
The two axes disagree on how a stored width maps onto `t`, and the disagreement
is per profile, not per axis — so each registered profile carries its own
`kernel_scale` / `growth_scale`, and the thin `kernel_core` / `growth_core`
adapters apply it:

    kernel_core(distance, r, w) = profile((distance - r) / (w * kernel_scale))
    growth_core(potential, m, s) = profile((potential - m) / (s * growth_scale))

The scales are read off the references, not chosen: `LeniaF.py`'s step kernel
keeps `u in [1/4, 3/4]`, i.e. half the nominal width (`kernel_scale = 0.5`),
while its step growth keeps `|potential - m| <= s` (`growth_scale = 1`); its
polynomial growth divides by `9 s**2` under a 4th power, i.e. `growth_scale = 3`,
while its polynomial kernel spans the full ring (`kernel_scale = 1`). A profile
whose scale is `None` on an axis is not valid on that axis.
"""

from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp

__all__ = [
    "Profile",
    "register",
    "get",
    "available",
    "kernel_core",
    "growth_core",
]

Array = jnp.ndarray
ProfileFn = Callable[[Array], Array]

_AXES = ("kernel", "growth")


class Profile(NamedTuple):
    """A registered 1-D bump profile.

    Attributes:
        fn:           `(t) -> [0, 1]`, the profile on the normalized coordinate.
        kernel_scale: multiplier applied to `w` on the kernel axis, or None if
                      the profile is not valid as a kernel core.
        growth_scale: multiplier applied to `s` on the growth axis, or None if
                      the profile is not valid as a growth core.
    """

    fn: ProfileFn
    kernel_scale: float | None
    growth_scale: float | None

    def scale(self, axis: str) -> float:
        scale = self.kernel_scale if axis == "kernel" else self.growth_scale
        if scale is None:
            raise ValueError(f"profile is not valid on the {axis!r} axis")
        return scale


_REGISTRY: dict[str, Profile] = {}


def register(
    name: str,
    fn: ProfileFn,
    kernel_scale: float | None = 1.0,
    growth_scale: float | None = 1.0,
    overwrite: bool = False,
) -> None:
    """Add a profile to the open registry.

    Args:
        name:         registry key, e.g. `"gaussian"`.
        fn:           `(t) -> [0, 1]` on the normalized coordinate `t`.
        kernel_scale: width multiplier on the kernel axis; None marks the
                      profile invalid there (as `leaky_gaussian` is on growth).
        growth_scale: width multiplier on the growth axis; None as above.
        overwrite:    allow replacing an existing entry.
    """
    if name in _REGISTRY and not overwrite:
        raise ValueError(f"profile {name!r} is already registered; pass overwrite=True")
    if kernel_scale is None and growth_scale is None:
        raise ValueError(f"profile {name!r} would be valid on no axis")
    _REGISTRY[name] = Profile(fn, kernel_scale, growth_scale)


def get(name: str) -> Profile:
    """Look up a registered profile, with a listing in the error message."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown profile {name!r}; available: {available()}") from None


def available(axis: str | None = None) -> tuple[str, ...]:
    """Registered profile names, optionally restricted to one axis (§4.7)."""
    if axis is None:
        return tuple(sorted(_REGISTRY))
    if axis not in _AXES:
        raise ValueError(f"axis must be one of {_AXES}, got {axis!r}")
    field = "kernel_scale" if axis == "kernel" else "growth_scale"
    return tuple(sorted(n for n, p in _REGISTRY.items() if getattr(p, field) is not None))


# ---- axis adapters -------------------------------------------------------


def kernel_core(name: str, distance: Array, r: Array, w: Array) -> Array:
    """Evaluate a profile as a KERNEL core — a ring at radius `r` (§4.7).

    Args:
        distance: (xyz ...), distance from the kernel origin, already divided
                  by the space resolution `R` so a ring radius is a fraction.
        r:        `r`, broadcastable, ring radius — centre of the bump.
        w:        `w`, broadcastable, ring width — half-width of the bump.

    Returns:
        (xyz ...), the ring profile in `[0, 1]`, before the ring height `b`.
    """
    profile = get(name)
    return profile.fn((distance - r) / (w * profile.scale("kernel")))


def growth_core(name: str, potential: Array, m: Array, s: Array) -> Array:
    """Evaluate a profile as a GROWTH core — UNSIGNED, in `[0, 1]` (§4.9).

    The signed form `2 * G - 1` is applied by `growth/bell.py`, not here.

    Args:
        potential: `U`, (xyz k), per-kernel neighborhood potential.
        m:         `m`, (k,), growth mean — centre of the bump.
        s:         `s`, (k,), growth width — half-width / std of the bump.

    Returns:
        `G`, (xyz k), unsigned growth in `[0, 1]`.
    """
    profile = get(name)
    return profile.fn((potential - m) / (s * profile.scale("growth")))


# ---- built-in profiles ---------------------------------------------------
# Formulas ported from `LeniaF.py`'s `Automaton.kernel_core` / `growth_func`,
# rewritten on `t = 2u - 1`; `gaussian` is the predecessor's `bell`.


def _gaussian(t: Array) -> Array:
    """exp(-t^2 / 2) — unbounded support, sigma = width."""
    return jnp.exp(-(t**2) / 2.0)


def _polynomial(t: Array) -> Array:
    """(1 - t^2)^4 on |t| <= 1 — LeniaF `kn=1` / `gn=1` (quad4)."""
    return jnp.maximum(0.0, 1.0 - t**2) ** 4


def _step(t: Array) -> Array:
    """Rectangle on |t| <= 1 — LeniaF `kn=3` / `gn=3` (stpz)."""
    return (jnp.abs(t) <= 1.0).astype(jnp.result_type(t, jnp.float32))


def _trapezoid(t: Array) -> Array:
    """Flat on |t| <= 1/2, linear down to 0 at |t| = 1."""
    return jnp.clip(2.0 - 2.0 * jnp.abs(t), 0.0, 1.0)


def _leaky_gaussian(t: Array) -> Array:
    """Gaussian of internal sigma 0.3 truncated to |t| < 1 — LeniaF `kn=4`.

    The 0.3 is not a free width: it is what fits a bell inside a ring so that
    +/- 3.3 sigma spans the ring exactly (§4.1). Kernel axis only.
    """
    return jnp.where(jnp.abs(t) < 1.0, jnp.exp(-((t / 0.3) ** 2) / 2.0), 0.0)


register("gaussian", _gaussian, kernel_scale=1.0, growth_scale=1.0)
register("polynomial", _polynomial, kernel_scale=1.0, growth_scale=3.0)
register("step", _step, kernel_scale=0.5, growth_scale=1.0)
register("trapezoid", _trapezoid, kernel_scale=1.0, growth_scale=1.0)
register("leaky_gaussian", _leaky_gaussian, kernel_scale=1.0, growth_scale=None)
