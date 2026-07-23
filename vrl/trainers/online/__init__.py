"""Online training loop public facade."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vrl.trainers.online.config import TrainerConfig as TrainerConfig
    from vrl.trainers.online.trainer import OnlineTrainer as OnlineTrainer


_PUBLIC_EXPORTS = {
    "OnlineTrainer": ("vrl.trainers.online.trainer", "OnlineTrainer"),
    "TrainerConfig": ("vrl.trainers.online.config", "TrainerConfig"),
}

__all__ = list(_PUBLIC_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load a public online trainer symbol only when it is requested."""

    try:
        module_name, symbol_name = _PUBLIC_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), symbol_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
