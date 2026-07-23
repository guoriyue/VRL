"""Emu3 family — BAAI's autoregressive multimodal model (HF-native).

Currently supports the HF-format ``BAAI/Emu3-Gen-hf`` checkpoint for
text-to-image generation under the visual-rl GRPO pipeline. Unlike janus_pro
(which needs the out-of-tree ``janus`` package), Emu3 ships inside
``transformers`` (>= 4.48), so no extra dependency is required.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vrl.models.families.emu3.config import Emu3Config as Emu3Config
    from vrl.models.families.emu3.model import Emu3Model as Emu3Model
    from vrl.models.families.emu3.model import Emu3ReplayModel as Emu3ReplayModel
    from vrl.models.families.emu3.model import (
        emu3_allowed_token_mask as emu3_allowed_token_mask,
    )
    from vrl.models.families.emu3.model import (
        emu3_forced_token_schedule as emu3_forced_token_schedule,
    )
    from vrl.models.families.emu3.model import emu3_grid_token_num as emu3_grid_token_num
    from vrl.models.families.emu3.runtime import (
        Emu3ChunkExecutor as Emu3ChunkExecutor,
    )


# Public lazy-import boundary: config discovery must not import the model runtime.
_PUBLIC_EXPORTS = {
    "Emu3ChunkExecutor": ("vrl.models.families.emu3.runtime", "Emu3ChunkExecutor"),
    "Emu3Config": ("vrl.models.families.emu3.config", "Emu3Config"),
    "Emu3Model": ("vrl.models.families.emu3.model", "Emu3Model"),
    "Emu3ReplayModel": ("vrl.models.families.emu3.model", "Emu3ReplayModel"),
    "emu3_allowed_token_mask": (
        "vrl.models.families.emu3.model",
        "emu3_allowed_token_mask",
    ),
    "emu3_forced_token_schedule": (
        "vrl.models.families.emu3.model",
        "emu3_forced_token_schedule",
    ),
    "emu3_grid_token_num": (
        "vrl.models.families.emu3.model",
        "emu3_grid_token_num",
    ),
}

__all__ = list(_PUBLIC_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load a public Emu3 symbol only when requested."""

    try:
        module_name, symbol_name = _PUBLIC_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), symbol_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
