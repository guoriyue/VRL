"""Shared executor scaffolding for diffusion generation families."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from vrl.engine.core.capabilities import FamilyCapability
from vrl.engine.core.protocols import (
    ChunkedFamilyPipelineExecutor,
)
from vrl.engine.core.types import (
    GenerationRequest,
    GenerationSampleSpec,
    OutputBatch,
    WorkloadSignature,
)
from vrl.engine.diffusion.denoise import (
    DiffusionChunkResult,
    DiffusionDenoiseConfig,
    run_diffusion_denoise_chunk,
)
from vrl.engine.diffusion.gather import gather_diffusion_chunks
from vrl.engine.diffusion.layout import DiffusionRequestLayout
from vrl.engine.diffusion.request import VideoGenerationRequest
from vrl.engine.diffusion.spec import DiffusionGenerationSpec
from vrl.engine.execution.batching import forward_batch_by_merging_prompts
from vrl.engine.execution.microbatching import (
    MicroBatchPlan,
    run_microbatches_with_oom_retry,
)
from vrl.engine.execution.planner import attach_engine_plan, build_engine_plan


class DiffusionPipelineExecutorBase(
    ChunkedFamilyPipelineExecutor,
):
    """Common GenerationRequest -> diffusion OutputBatch execution path."""

    family: str
    task: str
    model: Any
    default_sample_batch_size: int = 1
    default_num_frames: int = 1
    default_fps: int | None = None
    default_max_sequence_length: int = 512
    respect_cfg_flag: bool = True
    sde_type: str = "sde"
    include_max_sequence_length_extra: bool = True
    family_capability: FamilyCapability | None = None

    # -- protocol ------------------------------------------------------

    @property
    def layout(self) -> DiffusionRequestLayout:
        return DiffusionRequestLayout(
            default_sample_batch_size=self.default_sample_batch_size,
            default_num_frames=self.default_num_frames,
            default_fps=self.default_fps,
            default_max_sequence_length=self.default_max_sequence_length,
            sde_type=self.sde_type,
        )

    def workload_signature(self, request: GenerationRequest) -> WorkloadSignature:
        return WorkloadSignature.from_request_and_capability(request, self.capability())

    def capability(self) -> FamilyCapability:
        if self.family_capability is None:
            raise RuntimeError(
                f"{type(self).__name__} must declare family_capability explicitly"
            )
        return self.family_capability

    def plan(
        self,
        request: GenerationRequest,
        sample_specs: list[GenerationSampleSpec],
    ) -> Any:
        spec = self.parse_spec(request)
        return build_engine_plan(
            request,
            sample_specs,
            capability=self.capability(),
            max_samples_per_microbatch=spec.base.sample_batch_size,
        )

    def parse_spec(self, request: GenerationRequest) -> DiffusionGenerationSpec:
        return self.layout.parse_spec(request)

    def build_video_request(
        self,
        prompt: str,
        spec: DiffusionGenerationSpec,
    ) -> VideoGenerationRequest:
        """Build the backend-agnostic model request for one prompt chunk."""

        base = spec.base
        req_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "num_steps": base.num_steps,
            "guidance_scale": base.guidance_scale,
            "height": base.height,
            "width": base.width,
            "frame_count": base.num_frames,
        }
        if base.fps is not None:
            req_kwargs["fps"] = base.fps
        if base.negative_prompt is not None:
            req_kwargs["negative_prompt"] = base.negative_prompt
        if base.seed is not None:
            req_kwargs["seed"] = base.seed

        extra = self.build_video_request_extra(spec)
        if extra:
            req_kwargs["extra"] = extra
        return VideoGenerationRequest(**req_kwargs)

    def build_video_request_extra(
        self,
        spec: DiffusionGenerationSpec,
    ) -> dict[str, Any]:
        """Return family-neutral request.extra payload."""

        if not self.include_max_sequence_length_extra:
            return {}
        return {"max_sequence_length": spec.base.max_sequence_length}

    def build_denoise_config(
        self,
        spec: DiffusionGenerationSpec,
        chunk: MicroBatchPlan,
    ) -> DiffusionDenoiseConfig:
        """Build the SDE denoise config for one micro-batch."""

        if spec.sde is None:
            raise NotImplementedError(
                f"{type(self).__name__} must override denoise for non-SDE diffusion",
            )
        return self.layout.build_denoise_config(spec, chunk)

    def forward_plan(
        self,
        request: GenerationRequest,
        sample_specs: list[GenerationSampleSpec],
        plan: Any,
    ) -> OutputBatch:
        chunks = run_microbatches_with_oom_retry(
            plan.micro_batches,
            lambda micro_batch: self.forward_chunk_plan(
                request,
                micro_batch,
                plan.chunk_unit_for(micro_batch),
                plan.summary(),
            ),
        )
        return attach_engine_plan(self.gather_chunks(request, sample_specs, chunks), plan)

    def forward_chunk_plan(
        self,
        request: GenerationRequest,
        chunk: MicroBatchPlan,
        execution_unit: Any,
        plan_summary: Mapping[str, object],
    ) -> DiffusionChunkResult:
        from vrl.trainers.profiling import record_function

        del execution_unit, plan_summary
        spec = self.parse_spec(request)
        video_request = self.build_video_request(chunk.prompt, spec)
        with record_function("engine.prefill"):
            encoded = self.encode_prompt_for_chunk(
                generation_request=request,
                video_request=video_request,
                spec=spec,
                chunk=chunk,
            )
        chunk_encoded = self.build_chunk_encoded(
            encoded=encoded,
            generation_request=request,
            video_request=video_request,
            spec=spec,
            chunk=chunk,
        )
        return run_diffusion_denoise_chunk(
            model=self.model,
            request=video_request,
            encoded=chunk_encoded,
            config=self.build_denoise_config(spec, chunk),
            prepare_kwargs=self.build_prepare_kwargs(
                encoded=encoded,
                generation_request=request,
                video_request=video_request,
                spec=spec,
                chunk=chunk,
            ),
        )

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_specs: Sequence[GenerationSampleSpec],
        chunks: Sequence[DiffusionChunkResult],
    ) -> OutputBatch:
        return gather_diffusion_chunks(
            request,
            sample_specs,
            chunks,
            model_family=self.family,
            respect_cfg_flag=self.respect_cfg_flag,
        )

    def forward_batch_plan(
        self,
        requests: list[GenerationRequest],
        sample_specs_by_request: dict[str, list[GenerationSampleSpec]],
        engine_plans_by_request: dict[str, Any],
    ) -> dict[str, OutputBatch]:
        return forward_batch_by_merging_prompts(
            self,
            requests,
            sample_specs_by_request,
            engine_plans_by_request=engine_plans_by_request,
        )

    # -- family hooks --------------------------------------------------

    def encode_prompt_for_chunk(
        self,
        *,
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        spec: DiffusionGenerationSpec,
        chunk: MicroBatchPlan,
    ) -> dict[str, Any]:
        """Encode prompt conditioning for a single prompt chunk."""

        del generation_request
        return self.model.encode_prompt(
            chunk.prompt,
            video_request.negative_prompt or None,
            max_sequence_length=spec.base.max_sequence_length,
            guidance_scale=spec.base.guidance_scale,
            request=video_request,
        )

    def build_chunk_encoded(
        self,
        *,
        encoded: dict[str, Any],
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        spec: DiffusionGenerationSpec,
        chunk: MicroBatchPlan,
    ) -> dict[str, Any]:
        """Build per-sample encoded tensors for one micro-batch."""

        del generation_request, video_request, spec
        return self.layout.repeat_encoded_batch(encoded, chunk.sample_count)

    def build_prepare_kwargs(
        self,
        *,
        encoded: dict[str, Any],
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        spec: DiffusionGenerationSpec,
        chunk: MicroBatchPlan,
    ) -> dict[str, Any] | None:
        """Return additional family kwargs for model.prepare_sampling."""

        del encoded, generation_request, video_request, spec, chunk
        return None

__all__ = ["DiffusionPipelineExecutorBase", "DiffusionRequestLayout"]
