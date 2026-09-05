"""Cosmos Predict2.5 family runtime.

Registry-descriptor family: no builder functions live here — the generic
functions in ``vrl.models.steps.denoise.build`` construct the bundles from the
``DenoiseFamilyBuild`` recipe on this family's registry entry (UniPC replay
scheduler, LoRA-only DiffusionNFT guard).
"""

from __future__ import annotations

from typing import Any

from vrl.generation.bindings.full_sequence_denoise import (
    DiffusionBatchExecutorBase,
    DiffusionSamplingParams,
)
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.types import DenoiseRequest, GenerationRequest


class CosmosPredict25BatchExecutor(DiffusionBatchExecutorBase):
    """Diffusion executor for Cosmos Predict2.5 text-to-world rollouts."""

    family: str = "cosmos-predict2.5"
    task: str = "t2w"
    default_num_frames: int = 93
    default_fps: int | None = 16
    default_max_sequence_length: int = 512

    def encode_prompt_for_batch(
        self,
        *,
        generation_request: GenerationRequest,
        video_request: DenoiseRequest,
        params: DiffusionSamplingParams,
        batch: GenerationSampleBatch,
    ) -> dict[str, Any]:
        return self.model.encode_prompt(
            generation_request.inputs[batch.prompt_index].prompt,
            video_request.negative_prompt or None,
            **params.text_encode_kwargs(),
        )


__all__ = [
    "CosmosPredict25BatchExecutor",
]
