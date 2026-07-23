"""JoyAI-Echo video model family (single-shot flow-matching RL policy)."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vrl.models.families.echo.config import EchoModelSection as EchoModelSection
    from vrl.models.families.echo.model import EchoModel as EchoModel
    from vrl.models.families.echo.model import EchoReplayModel as EchoReplayModel
    from vrl.models.families.echo.model import EchoSamplingState as EchoSamplingState


# Public lazy-import boundary: config discovery must not import the model runtime.
_PUBLIC_EXPORTS = {
    "EchoModel": ("vrl.models.families.echo.model", "EchoModel"),
    "EchoModelSection": ("vrl.models.families.echo.config", "EchoModelSection"),
    "EchoReplayModel": ("vrl.models.families.echo.model", "EchoReplayModel"),
    "EchoSamplingState": ("vrl.models.families.echo.model", "EchoSamplingState"),
}

__all__ = list(_PUBLIC_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load a public Echo symbol only when requested."""

    try:
        module_name, symbol_name = _PUBLIC_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), symbol_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
