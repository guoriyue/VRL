"""GLM-Image family — ZhipuAI's hybrid AR + diffusion t2i model.

Supports the ``zai-org/GLM-Image`` checkpoint for text-to-image generation
under the visual-rl GRPO pipeline. The 9B AR section
(``GlmImageForConditionalGeneration``, transformers >= 5.13) is the trainable
policy; the 7B DiT decoder (``GlmImagePipeline``, diffusers >= 0.37) is a
frozen postprocess.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vrl.models.families.glm_image.config import GlmImageConfig as GlmImageConfig
    from vrl.models.families.glm_image.model import GlmImageModel as GlmImageModel
    from vrl.models.families.glm_image.model import (
        GlmImageReplayModel as GlmImageReplayModel,
    )
    from vrl.models.families.glm_image.model import (
        glm_image_decode_position_schedule as glm_image_decode_position_schedule,
    )
    from vrl.models.families.glm_image.model import (
        glm_image_grid_dims as glm_image_grid_dims,
    )
    from vrl.models.families.glm_image.model import (
        glm_image_prefill_position_ids as glm_image_prefill_position_ids,
    )
    from vrl.models.families.glm_image.model import (
        glm_image_token_num as glm_image_token_num,
    )
    from vrl.models.families.glm_image.runtime import (
        GlmImageChunkExecutor as GlmImageChunkExecutor,
    )


# Public lazy-import boundary: config discovery must not import the model runtime.
_PUBLIC_EXPORTS = {
    "GlmImageChunkExecutor": (
        "vrl.models.families.glm_image.runtime",
        "GlmImageChunkExecutor",
    ),
    "GlmImageConfig": ("vrl.models.families.glm_image.config", "GlmImageConfig"),
    "GlmImageModel": ("vrl.models.families.glm_image.model", "GlmImageModel"),
    "GlmImageReplayModel": (
        "vrl.models.families.glm_image.model",
        "GlmImageReplayModel",
    ),
    "glm_image_decode_position_schedule": (
        "vrl.models.families.glm_image.model",
        "glm_image_decode_position_schedule",
    ),
    "glm_image_grid_dims": (
        "vrl.models.families.glm_image.model",
        "glm_image_grid_dims",
    ),
    "glm_image_prefill_position_ids": (
        "vrl.models.families.glm_image.model",
        "glm_image_prefill_position_ids",
    ),
    "glm_image_token_num": (
        "vrl.models.families.glm_image.model",
        "glm_image_token_num",
    ),
}

__all__ = list(_PUBLIC_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load a public GLM-Image symbol only when requested."""

    try:
        module_name, symbol_name = _PUBLIC_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), symbol_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
