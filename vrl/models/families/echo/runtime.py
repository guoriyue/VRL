"""JoyAI-Echo family runtime (single-shot video flow-matching policy).

The rollout bundle routes through the shared diffusion builder (2026-07-01),
which wires rollout quantization and torch.compile — both conditional no-ops
while the config leaves them off, and both still UNVALIDATED on the LTX
transformer (built against diffusers-shaped modules). Before enabling either
knob on Echo, run a real-rollout parity check on an 80GB card.
"""

from __future__ import annotations

from typing import Any

from vrl.generation.bindings.full_sequence import (
    DenoiseBatchExecutorBase,
    DenoiseSamplingParams,
)
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.types import GenerationRequest


class EchoBatchExecutor(DenoiseBatchExecutorBase):
    """Diffusion executor for Echo text-to-video rollouts."""

    family: str = "echo"
    task: str = "t2v"
    # Frame count and FPS remain request behavior. The spatial grid is instead
    # fixed when the Echo wrapper is constructed and checked below.
    default_num_frames: int = 25
    default_fps: int | None = 24

    def parse_sampling_params(self, request: GenerationRequest) -> DenoiseSamplingParams:
        """Require requests to use the wrapper's fixed latent-grid dimensions."""

        params = super().parse_sampling_params(request)
        dimensions = (params.model_request.height, params.model_request.width)
        expected = (self.model.video_height, self.model.video_width)
        if dimensions != expected:
            raise ValueError(
                f"Echo sampling dimensions {dimensions[0]}x{dimensions[1]} must equal "
                f"the model construction dimensions {expected[0]}x{expected[1]}",
            )
        return params

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
        )


__all__ = [
    "EchoBatchExecutor",
]
