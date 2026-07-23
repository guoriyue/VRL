"""Cosmos Predict2 Anima family."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vrl.models.families.cosmos.anima.config import (
        CosmosAnimaModelSection as CosmosAnimaModelSection,
    )
    from vrl.models.families.cosmos.anima.model import AnimaModel as AnimaModel
    from vrl.models.families.cosmos.anima.runtime import (
        build_anima_replay_runtime_bundle as build_anima_replay_runtime_bundle,
    )


# Public lazy-import boundary: config discovery must not import the model runtime.
_PUBLIC_EXPORTS = {
    "AnimaModel": ("vrl.models.families.cosmos.anima.model", "AnimaModel"),
    "CosmosAnimaModelSection": (
        "vrl.models.families.cosmos.anima.config",
        "CosmosAnimaModelSection",
    ),
    "build_anima_replay_runtime_bundle": (
        "vrl.models.families.cosmos.anima.runtime",
        "build_anima_replay_runtime_bundle",
    ),
}

__all__ = list(_PUBLIC_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load a public Cosmos Anima symbol only when requested."""

    try:
        module_name, symbol_name = _PUBLIC_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), symbol_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
