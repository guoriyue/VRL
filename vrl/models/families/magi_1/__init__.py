"""MAGI-1 family integration."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vrl.models.families.magi_1.config import (
        Magi1ModelSection as Magi1ModelSection,
    )
    from vrl.models.families.magi_1.model import Magi1Model as Magi1Model
    from vrl.models.families.magi_1.model import (
        Magi1SubprocessConfig as Magi1SubprocessConfig,
    )
    from vrl.models.families.magi_1.model import (
        Magi1SubprocessModel as Magi1SubprocessModel,
    )


# Public lazy-import boundary: config discovery must not import the model runtime.
_PUBLIC_EXPORTS = {
    "Magi1Model": ("vrl.models.families.magi_1.model", "Magi1Model"),
    "Magi1ModelSection": ("vrl.models.families.magi_1.config", "Magi1ModelSection"),
    "Magi1SubprocessConfig": (
        "vrl.models.families.magi_1.model",
        "Magi1SubprocessConfig",
    ),
    "Magi1SubprocessModel": (
        "vrl.models.families.magi_1.model",
        "Magi1SubprocessModel",
    ),
}

__all__ = list(_PUBLIC_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load a public MAGI-1 symbol only when requested."""

    try:
        module_name, symbol_name = _PUBLIC_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), symbol_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
