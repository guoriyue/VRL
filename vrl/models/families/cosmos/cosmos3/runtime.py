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

from vrl.generation.bindings.full_sequence_denoise import (
    DiffusionBatchExecutorBase,
    DiffusionSamplingParams,
)
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.protocols import GenerationBatchGatherer
from vrl.generation.types import GenerationRequest, VideoGenerationRequest
from vrl.models.interfaces.runtime import ModelBuild, RuntimeBundle
from vrl.utils.logging import init_logger

logger = init_logger(__name__)


def build_cosmos3_replay_runtime_bundle(build: ModelBuild) -> RuntimeBundle:
    from vrl.models.families.cosmos.cosmos3.model import Cosmos3Model, Cosmos3ReplayModel

    logger.info("Building cosmos3 replay runtime bundle from %s", build.model_name_or_path)
    # Reuse from_build's pipeline loader, then wrap pipeline-shell in the replay model
    # (it needs the segment builders to rebuild packed_static at recompute time).
    driver = Cosmos3Model.from_build(build)
    model = Cosmos3ReplayModel(
        pipeline_shell=driver.pipeline,
        scheduler=driver.scheduler,
        device=build.device,
    )
    from vrl.models.steps.denoise.build import assemble_replay_bundle

    return assemble_replay_bundle(model, build)


class Cosmos3BatchExecutor(DiffusionBatchExecutorBase):
    """Diffusion executor for Cosmos3 Omni text-to-video rollouts.

    Strictly ``samples_per_generation_batch=1``: the Cosmos3OmniPipeline packs one sample at a
    time (no native B>1 path), so the batch executor pins a single sample.
    """

    family: str = "cosmos3"
    task: str = "t2v"
    default_num_frames: int = 93
    default_fps: int | None = 24

    def __init__(
        self,
        model: Any,
        *,
        gatherer: GenerationBatchGatherer | None = None,
        samples_per_generation_batch: int | None = None,
    ) -> None:
        del samples_per_generation_batch  # cosmos3 is strictly batch=1 (pipeline constraint)
        super().__init__(model, gatherer=gatherer)

    def encode_prompt_for_batch(
        self,
        *,
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        params: DiffusionSamplingParams,
        batch: GenerationSampleBatch,
    ) -> dict[str, Any]:
        return self.model.encode_prompt(
            generation_request.inputs[batch.prompt_index].prompt,
            video_request.negative_prompt or None,
            guidance_scale=params.model_request.guidance_scale,
            num_frames=video_request.frame_count,
            height=video_request.height,
            width=video_request.width,
            fps=video_request.fps or 24,
        )

    def build_batch_encoded(
        self,
        *,
        encoded: dict[str, Any],
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        params: DiffusionSamplingParams,
        batch: GenerationSampleBatch,
    ) -> dict[str, Any]:
        # batch=1: input_ids are python lists (not tensors); pass through unchanged.
        del generation_request, video_request, params, batch
        return dict(encoded)


__all__ = [
    "Cosmos3BatchExecutor",
    "build_cosmos3_replay_runtime_bundle",
]
