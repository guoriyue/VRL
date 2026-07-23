"""Cosmos Predict2.5 family."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vrl.models.families.cosmos.predict2_5.config import (
        CosmosPredict25ModelSection as CosmosPredict25ModelSection,
    )
    from vrl.models.families.cosmos.predict2_5.model import (
        CosmosPredict25Model as CosmosPredict25Model,
    )
    from vrl.models.families.cosmos.predict2_5.runtime import (
        CosmosPredict25ChunkExecutor as CosmosPredict25ChunkExecutor,
    )


# Public lazy-import boundary: config discovery must not import the model runtime.
_PUBLIC_EXPORTS = {
    "CosmosPredict25ChunkExecutor": (
        "vrl.models.families.cosmos.predict2_5.runtime",
        "CosmosPredict25ChunkExecutor",
    ),
    "CosmosPredict25Model": (
        "vrl.models.families.cosmos.predict2_5.model",
        "CosmosPredict25Model",
    ),
    "CosmosPredict25ModelSection": (
        "vrl.models.families.cosmos.predict2_5.config",
        "CosmosPredict25ModelSection",
    ),
}

__all__ = list(_PUBLIC_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load a public Cosmos Predict2.5 symbol only when requested."""

    try:
        module_name, symbol_name = _PUBLIC_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), symbol_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
