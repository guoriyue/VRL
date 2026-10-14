"""Cosmos3 Omni vision generator family runtime.

NOTE — no ``runner.py``: predict2's runner massages the SD3-shaped backbone
contract + applies EDM ``/sigma`` finalize. Cosmos3's transformer takes a ~10-kwarg
interleaved joint sequence and returns a 3-tuple of lists, which does not fit
``DenoiseBackboneInput``; CFG + velocity-masking run inline in
``Cosmos3Model.forward_step`` (see model.py). A runner would be a thin no-op layer
with no protocol boundary, so it is intentionally omitted.
"""

from __future__ import annotations

from typing import Any

from vrl.generation.bindings.full_sequence import (
    DenoiseBatchExecutorBase,
    DenoiseSamplingParams,
)
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.types import GenerationRequest


class Cosmos3BatchExecutor(DenoiseBatchExecutorBase):
    """Diffusion executor for Cosmos3 Omni text-to-video rollouts.

    Strictly ``samples_per_generation_batch=1``: the Cosmos3OmniPipeline packs one sample at a
    time (no native B>1 path), so the batch executor pins a single sample.
    """

    family: str = "cosmos3"
    task: str = "t2v"
    default_num_frames: int = 93
    default_fps: int | None = 24

    def encode_prompt_for_batch(
        self,
        *,
        generation_request: GenerationRequest,
        params: DenoiseSamplingParams,
        batch: GenerationSampleBatch,
    ) -> dict[str, Any]:
        return self.model.encode_prompt(
            generation_request.inputs[batch.prompt_index].prompt,
            params.model_request.negative_prompt or None,
            guidance_scale=params.model_request.guidance_scale,
            num_frames=params.model_request.frame_count,
            height=params.model_request.height,
            width=params.model_request.width,
            fps=params.model_request.fps or 24,
        )


__all__ = [
    "Cosmos3BatchExecutor",
]
