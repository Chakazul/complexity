"""Registries of systems and measures, filled in by the `@register_*` decorators.

Adding a system or a measure means writing one class and decorating it; the UI and the
CLI discover it from here. Modules listed in `_BUILTIN_MODULES` are imported on first use.
"""

from __future__ import annotations

import importlib

SYSTEMS: dict[str, type] = {}
MEASURES: dict[str, type] = {}

_BUILTIN_MODULES = (
    "complexity.systems.life",
    "complexity.systems.lenia",
    "complexity.measures.compression",
)
_loaded = False


def _load_builtins() -> None:
    global _loaded
    if not _loaded:
        _loaded = True
        for module in _BUILTIN_MODULES:
            importlib.import_module(module)


def register_system(cls: type) -> type:
    if cls.key in SYSTEMS:
        raise ValueError(f"system {cls.key!r} registered twice")
    SYSTEMS[cls.key] = cls
    return cls


def register_measure(cls: type) -> type:
    if cls.key in MEASURES:
        raise ValueError(f"measure {cls.key!r} registered twice")
    MEASURES[cls.key] = cls
    return cls


def systems() -> dict[str, type]:
    _load_builtins()
    return SYSTEMS


def measures() -> dict[str, type]:
    _load_builtins()
    return MEASURES


def make_system(key: str, **kwargs):
    return systems()[key](**kwargs)


def make_measure(key: str, **kwargs):
    return measures()[key](**kwargs)
