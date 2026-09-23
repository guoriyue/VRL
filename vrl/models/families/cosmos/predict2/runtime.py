"""Cosmos Predict2 family runtime.

Backend imports live inside the model's ``from_build`` so the shared runtime
does not import diffusers or cosmos-library backends eagerly.
"""

from __future__ import annotations

from typing import Any

from vrl.generation.bindings.full_sequence import (
    DenoiseBatchExecutorBase,
    ReferenceConditionedBatches,
)
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.types import GenerationRequest
from vrl.utils.logging import init_logger

logger = init_logger(__name__)


class CosmosBatchExecutor(ReferenceConditionedBatches, DenoiseBatchExecutorBase):
    """Diffusion executor for Cosmos Predict2 Video2World rollouts."""

    family: str = "cosmos-predict2"
    task: str = "v2w"
    default_num_frames: int = 93
    default_fps: int | None = 16

    def expand_conditioning_to_batch(
        self,
        *,
        encoded: dict[str, Any],
        generation_request: GenerationRequest,
        batch: GenerationSampleBatch,
    ) -> dict[str, Any]:
        """Repeat Cosmos text embeds and pass reference image through unchanged."""

        batch_encoded = super().expand_conditioning_to_batch(
            encoded={
                "prompt_embeds": encoded["prompt_embeds"],
                "negative_prompt_embeds": encoded.get("negative_prompt_embeds"),
            },
            generation_request=generation_request,
            batch=batch,
        )
        batch_encoded["reference_image"] = (
            encoded["reference_image"]
            if "reference_image" in encoded
            else self._reference_image_for_batch(generation_request, batch)
        )
        return batch_encoded


__all__ = [
    "CosmosBatchExecutor",
]
