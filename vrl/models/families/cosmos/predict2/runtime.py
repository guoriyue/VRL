"""Cosmos Predict2 family runtime.

Backend imports live inside the model's ``from_build`` so the shared runtime
does not import diffusers or cosmos-library backends eagerly.
"""

from __future__ import annotations

from typing import Any

from vrl.generation.bindings.full_sequence_denoise import (
    DiffusionBatchExecutorBase,
    DiffusionSamplingParams,
    ReferenceConditionedBatches,
)
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.types import GenerationRequest, VideoGenerationRequest
from vrl.utils.logging import init_logger

logger = init_logger(__name__)


class CosmosBatchExecutor(ReferenceConditionedBatches, DiffusionBatchExecutorBase):
    """Diffusion executor for Cosmos Predict2 Video2World rollouts."""

    family: str = "cosmos-predict2"
    task: str = "v2w"
    default_num_frames: int = 93
    default_fps: int | None = 16
    default_samples_per_generation_batch: int = 8

    def build_batch_encoded(
        self,
        *,
        encoded: dict[str, Any],
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        params: DiffusionSamplingParams,
        batch: GenerationSampleBatch,
    ) -> dict[str, Any]:
        """Repeat Cosmos text embeds and pass reference image through unchanged."""

        del video_request, params
        chunk_g = batch.sample_count
        reference_image = self._reference_image_for_chunk(generation_request, batch)
        batch_encoded: dict[str, Any] = {
            "prompt_embeds": self.layout.repeat_batch(
                encoded["prompt_embeds"],
                chunk_g,
            ),
            "reference_image": encoded.get("reference_image", reference_image),
        }
        neg = encoded.get("negative_prompt_embeds")
        if neg is not None:
            batch_encoded["negative_prompt_embeds"] = self.layout.repeat_batch(
                neg,
                chunk_g,
            )
        else:
            batch_encoded["negative_prompt_embeds"] = None
        return batch_encoded


__all__ = [
    "CosmosBatchExecutor",
]
