"""Cosmos Predict2 family runtime.

Backend imports live inside the model's ``from_build`` so the shared runtime
does not import diffusers or cosmos-library backends eagerly.
"""

from __future__ import annotations

from typing import Any

from vrl.generation.bindings.joint_denoise import (
    DiffusionChunkExecutorBase,
    DiffusionSamplingParams,
)
from vrl.generation.bindings.joint_denoise.executor import ReferenceConditionedChunks
from vrl.generation.execution.chunks import SampleChunk
from vrl.generation.types import GenerationRequest, VideoGenerationRequest
from vrl.utils.logging import init_logger

logger = init_logger(__name__)


class CosmosChunkExecutor(ReferenceConditionedChunks, DiffusionChunkExecutorBase):
    """Diffusion executor for Cosmos Predict2 Video2World rollouts."""

    family: str = "cosmos-predict2"
    task: str = "v2w"
    default_num_frames: int = 93
    default_fps: int | None = 16
    default_max_sequence_length: int = 512
    include_max_sequence_length_extra: bool = False

    def __init__(
        self,
        model: Any,  # CosmosPredict2Model
        *,
        samples_per_chunk: int = 8,
    ) -> None:
        self.model = model
        self.default_samples_per_chunk = max(1, int(samples_per_chunk))

    def build_chunk_encoded(
        self,
        *,
        encoded: dict[str, Any],
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        params: DiffusionSamplingParams,
        chunk: SampleChunk,
    ) -> dict[str, Any]:
        """Repeat Cosmos text embeds and pass reference image through unchanged."""

        del video_request, params
        chunk_g = chunk.sample_count
        reference_image = self._reference_image_for_chunk(generation_request, chunk)
        chunk_encoded: dict[str, Any] = {
            "prompt_embeds": self.layout.repeat_batch(
                encoded["prompt_embeds"],
                chunk_g,
            ),
            "reference_image": encoded.get("reference_image", reference_image),
        }
        neg = encoded.get("negative_prompt_embeds")
        if neg is not None:
            chunk_encoded["negative_prompt_embeds"] = self.layout.repeat_batch(
                neg,
                chunk_g,
            )
        else:
            chunk_encoded["negative_prompt_embeds"] = None
        return chunk_encoded


__all__ = [
    "CosmosChunkExecutor",
]
