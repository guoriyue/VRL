"""Cosmos3 Omni vision generator family runtime.

NOTE — no ``runner.py``: predict2's runner massages the SD3-shaped backbone
contract + applies EDM ``/sigma`` finalize. Cosmos3's transformer takes a ~10-kwarg
interleaved joint sequence and returns a 3-tuple of lists, which does not fit
``DiffusionBackboneInput``; CFG + velocity-masking run inline in
``Cosmos3Model.forward_step`` (see model.py). A runner would be a thin no-op layer
with no protocol boundary, so it is intentionally omitted.

NOTE — replay bundle loads a FULL pipeline, not a transformer-only minimal model:
the replay model needs the pipeline's ``_prepare_text_segment`` /
``_prepare_vision_segment`` to rebuild the step-invariant packed_static. A
transformer-only "option B" (persist+replay the packed-static index tensors)
is the memory optimization follow-up; correctness comes first.
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
from vrl.models.interfaces.runtime import RuntimeBuildSpec, RuntimeBundle
from vrl.utils.logging import init_logger

logger = init_logger(__name__)
def build_cosmos3_replay_runtime_bundle(spec: RuntimeBuildSpec) -> RuntimeBundle:
    from vrl.models.diffusion.cosmos.cosmos3.model import Cosmos3Model, Cosmos3ReplayModel

    logger.info("Building cosmos3 replay runtime bundle from %s", spec.model_name_or_path)
    # Reuse from_spec's pipeline loader, then wrap pipeline-shell in the replay model
    # (it needs the segment builders to rebuild packed_static at recompute time).
    driver = Cosmos3Model.from_spec(spec)
    model = Cosmos3ReplayModel(
        pipeline_shell=driver.pipeline,
        scheduler=driver.scheduler,
        device=spec.device,
    )
    from vrl.models.diffusion.build import assemble_replay_bundle

    return assemble_replay_bundle(model, spec, family="cosmos3")


class Cosmos3ChunkExecutor(DiffusionChunkExecutorBase):
    """Diffusion executor for Cosmos3 Omni text-to-video rollouts.

    Strictly ``samples_per_chunk=1``: the Cosmos3OmniPipeline packs one sample at a
    time (no native B>1 path), so the chunk executor pins a single sample.
    """

    family: str = "cosmos3"
    task: str = "t2v"
    default_num_frames: int = 93
    default_fps: int | None = 24
    default_max_sequence_length: int = 512

    def __init__(self, model: Any, *, samples_per_chunk: int = 1) -> None:
        del samples_per_chunk  # cosmos3 is strictly batch=1 (pipeline constraint)
        self.model = model
        self.default_samples_per_chunk = 1

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
            guidance_scale=params.base.guidance_scale,
            num_frames=video_request.frame_count,
            height=video_request.height,
            width=video_request.width,
            fps=video_request.fps or 24,
        )

    def build_chunk_encoded(
        self,
        *,
        encoded: dict[str, Any],
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        params: DiffusionSamplingParams,
        chunk: SampleChunk,
    ) -> dict[str, Any]:
        # batch=1: input_ids are python lists (not tensors); pass through unchanged.
        del generation_request, video_request, params, chunk
        return dict(encoded)


__all__ = [
    "Cosmos3ChunkExecutor",
    "build_cosmos3_replay_runtime_bundle",
]
