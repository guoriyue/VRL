"""SD 3.5 family runtime.

Registry-descriptor family: no builder functions live here — the generic
functions in ``vrl.models.diffusion.build`` construct the bundles from the
``DiffusionFamilyBuild`` recipe on the sd3_5 registry entry.
"""

from __future__ import annotations

from typing import Any

from vrl.generation.diffusion import (
    DiffusionChunkExecutorBase,
    DiffusionSamplingParams,
)
from vrl.generation.diffusion.layout import VideoGenerationRequest
from vrl.generation.execution.chunks import SampleChunk
from vrl.generation.types import GenerationRequest
from vrl.models.diffusion.capabilities import diffusion_family_capability

SD3_5_FAMILY_CAPABILITY = diffusion_family_capability("sd3_5", "t2i")


class SD3_5ChunkExecutor(DiffusionChunkExecutorBase):
    """Diffusion executor for SD3.5-M text-to-image rollouts."""

    family: str = "sd3_5"
    task: str = "t2i"
    family_capability = SD3_5_FAMILY_CAPABILITY
    default_num_frames: int = 1
    default_max_sequence_length: int = 128

    def __init__(
        self,
        model: Any,  # SD3_5Model
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
        """Repeat SD3 prompt and pooled embeds across the chunk batch."""

        del generation_request, video_request, params
        chunk_g = chunk.sample_count
        chunk_encoded: dict[str, Any] = {
            "prompt_embeds": self.layout.repeat_batch(
                encoded["prompt_embeds"],
                chunk_g,
            ),
            "pooled_prompt_embeds": self.layout.repeat_batch(
                encoded["pooled_prompt_embeds"],
                chunk_g,
            ),
        }
        neg = encoded.get("negative_prompt_embeds")
        neg_pool = encoded.get("negative_pooled_prompt_embeds")
        if neg is not None:
            chunk_encoded["negative_prompt_embeds"] = self.layout.repeat_batch(
                neg,
                chunk_g,
            )
        if neg_pool is not None:
            chunk_encoded["negative_pooled_prompt_embeds"] = self.layout.repeat_batch(
                neg_pool,
                chunk_g,
            )
        return chunk_encoded


__all__ = [
    "SD3_5ChunkExecutor",
    "SD3_5_FAMILY_CAPABILITY",
]
