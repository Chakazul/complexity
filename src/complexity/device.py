"""Choose JAX's platform before JAX loads, so a missing GPU falls back to the CPU quietly.

With a CUDA build of JAX installed (`jax[cuda13]`), JAX initializes its CUDA plugin on
first use. When no GPU is usable — on this laptop the NVIDIA dGPU is off on battery — the
plugin fails with a long traceback before JAX falls back to the CPU. Asking the CUDA driver
first (`cuInit`, ~10 ms) avoids that: without a device, JAX is told to use the CPU.
`JAX_PLATFORMS`, when set, always wins. Works when JAX was imported first too, as long
as it has not created its backends yet.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import sys

CUDA_PLUGINS = ("jax_plugins.xla_cuda13", "jax_plugins.xla_cuda12")

# Why the GPU was skipped, for `complexity info`; None if it was not.
gpu_skipped: str | None = None


def cuda_devices() -> tuple[int, str]:
    """(number of CUDA devices, reason if none), from the CUDA driver."""
    try:
        lib = ctypes.CDLL("libcuda.so.1" if os.name != "nt" else "nvcuda.dll")
    except OSError:
        return 0, "no CUDA driver"
    if (err := lib.cuInit(0)) != 0:
        return 0, f"CUDA driver error {err}" + (" (no device)" if err == 100 else "")
    n = ctypes.c_int(0)
    if lib.cuDeviceGetCount(ctypes.byref(n)) != 0 or n.value == 0:
        return 0, "no CUDA device"
    return n.value, ""


def select_platform() -> None:
    """Set JAX_PLATFORMS=cpu if a CUDA plugin is installed but no CUDA device is usable."""
    global gpu_skipped
    if "JAX_PLATFORMS" in os.environ:
        return
    try:
        if not any(importlib.util.find_spec(m) for m in CUDA_PLUGINS):
            return
    except ModuleNotFoundError:  # no jax_plugins package
        return
    n, reason = cuda_devices()
    if n == 0:
        os.environ["JAX_PLATFORMS"] = "cpu"
        if "jax" in sys.modules:  # imported before us: it has read the environment already
            sys.modules["jax"].config.update("jax_platforms", "cpu")
        gpu_skipped = reason
