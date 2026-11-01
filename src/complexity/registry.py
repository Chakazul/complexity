"""Registries of systems and measures, filled in by the `@register_*` decorators.

Adding a system or a measure means writing one class and decorating it; the UI and the
CLI discover it from here. Modules listed in `_BUILTIN_MODULES` are imported on first use.
"""

from __future__ import annotations

import importlib

SYSTEMS: dict[str, type] = {}
MEASURES: dict[str, type] = {}

_BUILTIN_MODULES = (
    "complexity.systems.lenia",
    "complexity.systems.lenia_multi",
    "complexity.systems.life",
    "complexity.systems.baseline",
    "complexity.measures.compression",  # measure modules in the UI's group order
    "complexity.measures.algorithmic",
    "complexity.measures.causal",
    "complexity.measures.information",
    "complexity.measures.decomposition",
    "complexity.measures.individuality",
    "complexity.measures.geometry",
    "complexity.measures.dynamics",
    "complexity.measures.autopoiesis",
    "complexity.measures.rule",
    "complexity.measures.statistics",
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


def _in_module_order(registered: dict[str, type]) -> dict[str, type]:
    """Built-in modules in `_BUILTIN_MODULES` order (the UI's group order), whatever order
    they were imported in; others after them; registration order within a module."""
    rank = {m: i for i, m in enumerate(_BUILTIN_MODULES)}
    return dict(sorted(registered.items(),
                       key=lambda item: rank.get(item[1].__module__, len(rank))))


def systems() -> dict[str, type]:
    _load_builtins()
    return _in_module_order(SYSTEMS)


def measures() -> dict[str, type]:
    _load_builtins()
    return _in_module_order(MEASURES)


def make_system(key: str, **kwargs):
    return systems()[key](**kwargs)


def make_measure(key: str, **kwargs):
    return measures()[key](**kwargs)
