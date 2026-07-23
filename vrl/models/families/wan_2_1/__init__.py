"""Wan 2.1 diffusion family."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vrl.models.families.wan_2_1.config import WanModelSection as WanModelSection
    from vrl.models.families.wan_2_1.model import (
        WanI2VDiffusersModel as WanI2VDiffusersModel,
    )
    from vrl.models.families.wan_2_1.model import WanI2VReplayModel as WanI2VReplayModel
    from vrl.models.families.wan_2_1.model import (
        WanI2VSamplingState as WanI2VSamplingState,
    )
    from vrl.models.families.wan_2_1.model import (
        WanT2VDiffusersModel as WanT2VDiffusersModel,
    )
    from vrl.models.families.wan_2_1.model import WanT2VReplayModel as WanT2VReplayModel
    from vrl.models.families.wan_2_1.model import (
        WanT2VSamplingState as WanT2VSamplingState,
    )
    from vrl.models.families.wan_2_1.runtime import (
        Wan_2_1I2VChunkExecutor as Wan_2_1I2VChunkExecutor,
    )


# Public lazy-import boundary: config discovery must not import the model runtime.
_PUBLIC_EXPORTS = {
    "WanI2VDiffusersModel": (
        "vrl.models.families.wan_2_1.model",
        "WanI2VDiffusersModel",
    ),
    "WanI2VReplayModel": ("vrl.models.families.wan_2_1.model", "WanI2VReplayModel"),
    "WanI2VSamplingState": (
        "vrl.models.families.wan_2_1.model",
        "WanI2VSamplingState",
    ),
    "WanModelSection": ("vrl.models.families.wan_2_1.config", "WanModelSection"),
    "WanT2VDiffusersModel": (
        "vrl.models.families.wan_2_1.model",
        "WanT2VDiffusersModel",
    ),
    "WanT2VReplayModel": ("vrl.models.families.wan_2_1.model", "WanT2VReplayModel"),
    "WanT2VSamplingState": (
        "vrl.models.families.wan_2_1.model",
        "WanT2VSamplingState",
    ),
    "Wan_2_1I2VChunkExecutor": (
        "vrl.models.families.wan_2_1.runtime",
        "Wan_2_1I2VChunkExecutor",
    ),
}

__all__ = list(_PUBLIC_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load a public Wan symbol only when it is requested."""

    try:
        module_name, symbol_name = _PUBLIC_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), symbol_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
