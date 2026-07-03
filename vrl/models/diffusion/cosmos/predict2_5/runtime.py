"""Cosmos Predict2.5 family runtime.

Registry-descriptor family: no builder functions live here — the generic
functions in ``vrl.models.diffusion.build`` construct the bundles from the
``DiffusionFamilyBuild`` recipe on this family's registry entry (UniPC replay
scheduler, LoRA-only DiffusionNFT guard).
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

COSMOS_PREDICT25_FAMILY_CAPABILITY = diffusion_family_capability(
    "cosmos-predict2.5",
    "t2w",
)


class CosmosPredict25ChunkExecutor(DiffusionChunkExecutorBase):
    """Diffusion executor for Cosmos Predict2.5 text-to-world rollouts."""

    family: str = "cosmos-predict2.5"
    task: str = "t2w"
    family_capability = COSMOS_PREDICT25_FAMILY_CAPABILITY
    default_num_frames: int = 93
    default_fps: int | None = 16
    default_max_sequence_length: int = 512

    def __init__(
        self,
        model: Any,
        *,
        samples_per_chunk: int = 1,
    ) -> None:
        self.model = model
        self.default_samples_per_chunk = max(1, int(samples_per_chunk))

    def encode_prompt_for_chunk(
        self,
        *,
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        params: DiffusionSamplingParams,
        chunk: SampleChunk,
    ) -> dict[str, Any]:
        del generation_request
        return self.model.encode_prompt(
            chunk.prompt,
            video_request.negative_prompt or None,
            max_sequence_length=params.base.max_sequence_length,
            guidance_scale=params.base.guidance_scale,
        )

__all__ = [
    "COSMOS_PREDICT25_FAMILY_CAPABILITY",
    "CosmosPredict25ChunkExecutor",
]
