"""GLM-Image family — ZhipuAI's hybrid AR + diffusion t2i model.

Supports the ``zai-org/GLM-Image`` checkpoint for text-to-image generation
under the visual-rl GRPO pipeline. The 9B AR section
(``GlmImageForConditionalGeneration``, transformers >= 5.13) is the trainable
policy; the 7B DiT decoder (``GlmImagePipeline``, diffusers >= 0.37) is a
frozen postprocess.
"""

from __future__ import annotations

from vrl.models.ar.glm_image.model import (
    GlmImageConfig,
    GlmImageModel,
    GlmImageReplayModel,
    glm_image_decode_position_schedule,
    glm_image_grid_dims,
    glm_image_prefill_position_ids,
    glm_image_token_num,
)
from vrl.models.ar.glm_image.runtime import GlmImageChunkExecutor

__all__ = [
    "GlmImageChunkExecutor",
    "GlmImageConfig",
    "GlmImageModel",
    "GlmImageReplayModel",
    "glm_image_decode_position_schedule",
    "glm_image_grid_dims",
    "glm_image_prefill_position_ids",
    "glm_image_token_num",
]
