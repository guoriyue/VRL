"""CausVid causal-chunk video model family."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vrl.models.families.causvid.config import (
        CausVidModelSection as CausVidModelSection,
    )
    from vrl.models.families.causvid.model import CausVidModel as CausVidModel
    from vrl.models.families.causvid.model import CausVidReplayModel as CausVidReplayModel
    from vrl.models.families.causvid.runner import (
        CausVidCausalRunner as CausVidCausalRunner,
    )
    from vrl.models.families.causvid.runner import CausVidGeometry as CausVidGeometry
    from vrl.models.families.causvid.runner import CausVidRunResult as CausVidRunResult
    from vrl.models.families.causvid.runner import CausVidSchedule as CausVidSchedule


# Public lazy-import boundary: config discovery must not import the model runtime.
_PUBLIC_EXPORTS = {
    "CausVidCausalRunner": (
        "vrl.models.families.causvid.runner",
        "CausVidCausalRunner",
    ),
    "CausVidGeometry": ("vrl.models.families.causvid.runner", "CausVidGeometry"),
    "CausVidModel": ("vrl.models.families.causvid.model", "CausVidModel"),
    "CausVidModelSection": (
        "vrl.models.families.causvid.config",
        "CausVidModelSection",
    ),
    "CausVidReplayModel": (
        "vrl.models.families.causvid.model",
        "CausVidReplayModel",
    ),
    "CausVidRunResult": ("vrl.models.families.causvid.runner", "CausVidRunResult"),
    "CausVidSchedule": ("vrl.models.families.causvid.runner", "CausVidSchedule"),
}

__all__ = list(_PUBLIC_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load a public CausVid symbol only when requested."""

    try:
        module_name, symbol_name = _PUBLIC_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), symbol_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
