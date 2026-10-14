"""MiniMax-H3 family runtime: the batch-one executor.

NOTE — no ``runner.py``: the H3 transformer takes a packed-sequence layout
(nine kwargs) and returns a ``(video, audio)`` pair, which does not fit
``DenoiseBackboneInput``; the call and the audio side-stream update run
inline in ``MiniMaxH3Model.forward_step``.
"""

from __future__ import annotations

from typing import Any

from vrl.generation.bindings.full_sequence import (
    DenoiseBatchExecutorBase,
    DenoiseSamplingParams,
)
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.types import GenerationRequest

# 5 s at 24 fps is 120 frames; the video VAE decodes ``17n + 5``, so 124.
DEFAULT_NUM_FRAMES = 124
DEFAULT_FPS = 24
DEFAULT_MAX_SEQUENCE_LENGTH = 512


class MiniMaxH3BatchExecutor(DenoiseBatchExecutorBase):
    """Diffusion executor for MiniMax-H3 text-to-video rollouts.

    Strictly ``samples_per_generation_batch=1``: the transformer's batch axis is
    a replication axis over ONE packed layout, and the layout is a function of
    the prompt length, so one prompt per generation batch is the native shape.
    """

    family: str = "minimax_h3"
    task: str = "t2v"
    default_num_frames: int = DEFAULT_NUM_FRAMES
    default_fps: int | None = DEFAULT_FPS
    default_max_sequence_length: int | None = DEFAULT_MAX_SEQUENCE_LENGTH

    def encode_prompt_for_batch(
        self,
        *,
        generation_request: GenerationRequest,
        params: DenoiseSamplingParams,
        batch: GenerationSampleBatch,
    ) -> dict[str, Any]:
        return self.model.encode_prompt(
            generation_request.inputs[batch.prompt_index].prompt,
            None,
            **params.text_encode_kwargs(),
        )


__all__ = [
    "DEFAULT_FPS",
    "DEFAULT_MAX_SEQUENCE_LENGTH",
    "DEFAULT_NUM_FRAMES",
    "MiniMaxH3BatchExecutor",
]
