"""Shared executor scaffolding for full-sequence denoise generation families."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from vrl.generation.bindings.full_sequence_denoise.gather import DiffusionChunkGatherer
from vrl.generation.bindings.full_sequence_denoise.layout import (
    DiffusionRequestLayout,
    DiffusionSamplingParams,
)
from vrl.generation.execution.chunks import (
    SampleChunk,
    run_sample_chunks_with_oom_retry,
)
from vrl.generation.execution.planner import build_engine_plan
from vrl.generation.protocols import (
    GenerationChunkExecutor,
)
from vrl.generation.steps.denoise.config import DenoiseLoopConfig
from vrl.generation.steps.denoise.loop import (
    DenoiseLoopResult,
    cuda_phase_peak_bytes,
    reset_cuda_peak,
    run_denoise_loop,
)
from vrl.generation.types import (
    GenerationOutput,
    GenerationRequest,
    GenerationSampleRow,
    VideoGenerationRequest,
)
from vrl.trajectory.storage import (
    TrajectoryStoragePolicy,
    apply_value_storage_policy,
    trajectory_storage_policy_from_cfg,
    trajectory_tensor_bytes,
)
from vrl.utils.media import load_reference_image, to_uint8


@dataclass(slots=True)
class DiffusionPromptStageInput:
    """Prompt-encode stage input for one diffusion sample chunk."""

    generation_request: GenerationRequest
    chunk: SampleChunk
    params: DiffusionSamplingParams
    video_request: VideoGenerationRequest


@dataclass(slots=True)
class DiffusionPromptStageOutput:
    """Prompt-encode stage output for downstream prepare/denoise stages."""

    generation_request: GenerationRequest
    chunk: SampleChunk
    params: DiffusionSamplingParams
    video_request: VideoGenerationRequest
    encoded: dict[str, Any]


@dataclass(slots=True)
class DiffusionPreparedStageOutput:
    """Prepared latent state and denoise config for the denoise stage."""

    chunk_encoded: dict[str, Any]
    config: DenoiseLoopConfig
    state: Any


@dataclass(slots=True)
class DiffusionDenoisedStageOutput:
    """Denoise stage output before VAE decode."""

    config: DenoiseLoopConfig
    denoise_result: DenoiseLoopResult
    stage_durations: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class DiffusionChunkResult:
    """Output of one fused diffusion sample chunk."""

    prompt_index: int
    sample_start: int
    sample_count: int
    observations: Any
    actions: Any
    log_probs: Any
    timesteps: Any
    kl: Any
    video: Any
    replay_tensors: dict[str, Any]
    context: dict[str, Any]
    # Display/provenance-only: emitted through per-chunk runtime debug metrics.
    peak_memory_mb: float | None = None
    # Completed ChunkMemoryReading fields for byte-admission shadow telemetry
    # (see vrl/generation/execution/chunk_placement.py). None off-CUDA.
    memory: dict[str, int] | None = None
    # Display/provenance-only: emitted through per-chunk runtime debug metrics.
    stage_durations: dict[str, float] = field(default_factory=dict)
    # Display/provenance-only: emitted through per-chunk runtime debug metrics.
    engine_counters: dict[str, Any] = field(default_factory=dict)


class ReferenceConditionedChunks:
    """Reference-image threading for per-chunk encode/prepare.

    Cosmos Predict2 Video2World and Wan 2.1 I2V condition every chunk on a
    reference image carried by its ``GenerationInput``. The two executors had
    copy-pasted these
    hooks; ``build_chunk_encoded`` stays family-specific because the encoded
    payloads genuinely differ (Wan carries ``image_embeds``).
    """

    model: Any

    def encode_prompt_for_chunk(
        self,
        *,
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        params: Any,
        chunk: SampleChunk,
    ) -> dict[str, Any]:
        """Encode text plus the active reference-image conditioning for one chunk."""

        reference_image = self._reference_image_for_chunk(generation_request, chunk)
        return self.model.encode_prompt(
            chunk.prompt,
            video_request.negative_prompt or None,
            max_sequence_length=params.base.max_sequence_length,
            guidance_scale=params.base.guidance_scale,
            reference_image=reference_image,
        )

    def build_prepare_kwargs(
        self,
        *,
        encoded: dict[str, Any],
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        params: Any,
        chunk: SampleChunk,
    ) -> dict[str, Any]:
        """Thread the active reference image into family prepare_sampling."""

        del encoded, video_request, params
        return {
            "reference_image": self._reference_image_for_chunk(
                generation_request,
                chunk,
            ),
        }

    def _reference_image_for_chunk(
        self,
        request: GenerationRequest,
        chunk: SampleChunk,
    ) -> Any:
        ref = request.inputs[chunk.prompt_index].reference_image
        if ref is None:
            raise ValueError(
                f"{request.family} requires reference_image for prompt index {chunk.prompt_index}",
            )
        return load_reference_image(ref)


class DiffusionChunkExecutorBase(
    GenerationChunkExecutor,
):
    """Common GenerationRequest -> diffusion GenerationOutput execution path."""

    family: str
    task: str
    model: Any
    default_samples_per_chunk: int = 1
    default_num_frames: int = 1
    default_fps: int | None = None
    default_max_sequence_length: int = 512
    sde_type: str = "flow_grpo"
    include_max_sequence_length_extra: bool = True

    # -- protocol ------------------------------------------------------

    @property
    def layout(self) -> DiffusionRequestLayout:
        return DiffusionRequestLayout(
            default_samples_per_chunk=self.default_samples_per_chunk,
            default_num_frames=self.default_num_frames,
            default_fps=self.default_fps,
            default_max_sequence_length=self.default_max_sequence_length,
            sde_type=self.sde_type,
        )

    def plan(
        self,
        request: GenerationRequest,
        sample_rows: list[GenerationSampleRow],
    ) -> Any:
        params = self.parse_sampling_params(request)
        return build_engine_plan(
            request,
            max_samples_per_chunk=params.base.samples_per_chunk,
        )

    def parse_sampling_params(self, request: GenerationRequest) -> DiffusionSamplingParams:
        return self.layout.parse_sampling_params(request)

    def build_video_request(
        self,
        prompt: str,
        params: DiffusionSamplingParams,
    ) -> VideoGenerationRequest:
        """Build the backend-agnostic model request for one prompt chunk."""

        base = params.base
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

        if self.include_max_sequence_length_extra:
            req_kwargs["extra"] = {
                "max_sequence_length": params.base.max_sequence_length,
            }
        return VideoGenerationRequest(**req_kwargs)

    def build_denoise_config(
        self,
        params: DiffusionSamplingParams,
        chunk: SampleChunk,
    ) -> DenoiseLoopConfig:
        """Build the SDE denoise config for one sample chunk."""

        if params.sde is None:
            raise NotImplementedError(
                f"{type(self).__name__} must override denoise for non-SDE diffusion",
            )
        layout = self.layout
        return DenoiseLoopConfig(
            prompt_index=chunk.prompt_index,
            sample_start=chunk.sample_start,
            sample_count=chunk.sample_count,
            seed=params.base.seed,
            sde=params.sde,
            sde_window=layout.select_sde_window(
                params.sde.sde_window_size,
                params.sde.sde_window_range,
            ),
            denoise_mode=params.denoise_mode,
            teacache=params.teacache,
        )

    def forward_plan(
        self,
        request: GenerationRequest,
        sample_rows: list[GenerationSampleRow],
        plan: Any,
    ) -> GenerationOutput:
        chunks = run_sample_chunks_with_oom_retry(
            plan.chunks,
            lambda chunk: self.forward_chunk_plan(request, chunk),
        )
        return self.gather_chunks(request, sample_rows, chunks)

    def forward_plan_pipelined(
        self,
        request: GenerationRequest,
        sample_rows: list[GenerationSampleRow],
        plan: Any,
    ) -> GenerationOutput:
        """In-process software-pipelined variant of forward_plan: chunk N+1's
        produce (encode->prepare->denoise->decode, GPU compute on the default
        stream) overlaps chunk N's teardown (the GPU->CPU result copy + host
        packing, on a copy stream), hiding the per-chunk copy+CPU boundary behind
        the next chunk's denoise. BIT-EXACT to forward_plan: same per-chunk stage
        methods (via forward_chunk_plan), value-preserving side-stream copy,
        and the SAME order-preserving gather_chunks — so the gathered output is
        identical; only the wall-clock changes.

        This is the executor-level entry for the single-GPU stage-overlap lever; the
        Ray worker calls it per-request (all of a request's chunks on one worker)
        instead of dispatching one monolithic forward_chunk_plan per chunk.
        """

        from vrl.generation.execution.pipeline import forward_chunks_pipelined

        chunks = forward_chunks_pipelined(self, request, plan.chunks)
        return self.gather_chunks(request, sample_rows, chunks)

    def forward_chunk_plan(
        self,
        request: GenerationRequest,
        chunk: SampleChunk,
    ) -> DiffusionChunkResult:
        from vrl.utils.profiling import record_function

        stage_durations: dict[str, float] = {}
        prompt_input = self.build_prompt_stage_input(request, chunk)
        prompt_output = self.run_prompt_encode_stage(
            prompt_input,
            stage_durations=stage_durations,
            record_function=record_function,
        )
        prepared = self.run_prepare_stage(
            prompt_output,
            stage_durations=stage_durations,
        )
        denoised = self.run_denoise_stage(
            prepared,
            stage_durations=stage_durations,
        )
        return self.apply_wire_storage_policy(
            request,
            self.run_decode_stage(denoised),
        )

    def apply_wire_storage_policy(
        self,
        request: GenerationRequest,
        chunk_result: DiffusionChunkResult,
    ) -> DiffusionChunkResult:
        """Apply rollout.trajectory_storage BEFORE tensors cross the wire.

        The same policy is re-applied driver-side when the trajectory batch is
        built (idempotent there); applying it here is what turns a dtype
        downcast into actual worker->driver transfer savings. The default
        preserve/preserve policy is a no-op, keeping the GRPO baseline
        bit-for-bit.
        """

        policy = trajectory_storage_policy_from_cfg(
            request.sampling.get("trajectory_storage"),
        )
        if policy == TrajectoryStoragePolicy():
            return chunk_result
        chunk_result.observations = apply_value_storage_policy(
            chunk_result.observations,
            policy,
        )
        chunk_result.actions = apply_value_storage_policy(chunk_result.actions, policy)
        chunk_result.log_probs = apply_value_storage_policy(chunk_result.log_probs, policy)
        chunk_result.timesteps = apply_value_storage_policy(chunk_result.timesteps, policy)
        chunk_result.kl = apply_value_storage_policy(chunk_result.kl, policy)
        chunk_result.replay_tensors = apply_value_storage_policy(
            chunk_result.replay_tensors,
            policy,
        )
        return chunk_result

    def build_prompt_stage_input(
        self,
        request: GenerationRequest,
        chunk: SampleChunk,
    ) -> DiffusionPromptStageInput:
        """Build the prompt-encode payload for one chunk."""

        params = self.parse_sampling_params(request)
        video_request = self.build_video_request(chunk.prompt, params)
        return DiffusionPromptStageInput(
            generation_request=request,
            chunk=chunk,
            params=params,
            video_request=video_request,
        )

    def run_prompt_encode_stage(
        self,
        payload: DiffusionPromptStageInput,
        *,
        stage_durations: dict[str, float],
        record_function: Any,
    ) -> DiffusionPromptStageOutput:
        """Run prompt encoding and return the typed stage payload."""

        started = time.perf_counter()
        with record_function("generation.prompt_encode"):
            encoded = self.encode_prompt_for_chunk(
                generation_request=payload.generation_request,
                video_request=payload.video_request,
                params=payload.params,
                chunk=payload.chunk,
            )
        stage_durations["encode"] = time.perf_counter() - started
        return DiffusionPromptStageOutput(
            generation_request=payload.generation_request,
            chunk=payload.chunk,
            params=payload.params,
            video_request=payload.video_request,
            encoded=encoded,
        )

    def run_prepare_stage(
        self,
        payload: DiffusionPromptStageOutput,
        *,
        stage_durations: dict[str, float],
    ) -> DiffusionPreparedStageOutput:
        """Prepare latent state and denoise config for one chunk."""

        started = time.perf_counter()
        chunk_encoded = self.build_chunk_encoded(
            encoded=payload.encoded,
            generation_request=payload.generation_request,
            video_request=payload.video_request,
            params=payload.params,
            chunk=payload.chunk,
        )
        prepare_kwargs = self.build_prepare_kwargs(
            encoded=payload.encoded,
            generation_request=payload.generation_request,
            video_request=payload.video_request,
            params=payload.params,
            chunk=payload.chunk,
        )
        config = self.build_denoise_config(payload.params, payload.chunk)
        state = self.prepare_denoise_state(
            request=payload.video_request,
            encoded=chunk_encoded,
            config=config,
            prepare_kwargs=prepare_kwargs,
        )
        stage_durations["prepare_latent"] = time.perf_counter() - started
        return DiffusionPreparedStageOutput(
            chunk_encoded=chunk_encoded,
            config=config,
            state=state,
        )

    def run_denoise_stage(
        self,
        payload: DiffusionPreparedStageOutput,
        *,
        stage_durations: dict[str, float],
    ) -> DiffusionDenoisedStageOutput:
        """Run the trainable denoise stage for one chunk."""

        started = time.perf_counter()
        denoise_result = self.run_denoise_steps(
            state=payload.state,
            encoded=payload.chunk_encoded,
            config=payload.config,
        )
        stage_durations["denoise"] = time.perf_counter() - started
        return DiffusionDenoisedStageOutput(
            config=payload.config,
            denoise_result=denoise_result,
            stage_durations=dict(stage_durations),
        )

    def run_decode_stage(
        self,
        payload: DiffusionDenoisedStageOutput,
    ) -> DiffusionChunkResult:
        """Run VAE decode and pack the final chunk result."""

        started = time.perf_counter()
        chunk_result = self.decode_denoise_result(
            config=payload.config,
            denoise_result=payload.denoise_result,
            stage_durations=payload.stage_durations,
        )
        chunk_result.stage_durations["decode"] = time.perf_counter() - started
        return chunk_result

    def prepare_denoise_state(
        self,
        *,
        request: VideoGenerationRequest,
        encoded: dict[str, Any],
        config: DenoiseLoopConfig,
        prepare_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Prepare latent state for one diffusion sample chunk."""

        from vrl.utils.profiling import record_function

        model = self.model
        with record_function("generation.prepare_sampling"):
            state = model.prepare_sampling(request, encoded, **(prepare_kwargs or {}))
        chunk_batch = state.latents.shape[0]
        if int(chunk_batch) != config.sample_count:
            raise ValueError(
                "Diffusion denoise chunk produced "
                f"{chunk_batch} rows, expected {config.sample_count}",
            )
        return state

    def run_denoise_steps(
        self,
        *,
        state: Any,
        encoded: dict[str, Any],
        config: DenoiseLoopConfig,
    ) -> DenoiseLoopResult:
        """Adapt the full-sequence denoise binding to the step-owned loop."""

        del encoded
        return run_denoise_loop(
            model=self.model,
            state=state,
            config=config,
        )

    def decode_denoise_result(
        self,
        *,
        config: DenoiseLoopConfig,
        denoise_result: DenoiseLoopResult,
        stage_durations: dict[str, float] | None = None,
    ) -> DiffusionChunkResult:
        """Decode the final latents and pack one diffusion chunk result."""

        from vrl.utils.profiling import record_function

        model = self.model
        state = denoise_result.state
        # Byte-admission shadow: the decode + pack spike is the second memory
        # phase, measured separately from the denoise plateau.
        reset_cuda_peak()
        with record_function("generation.decode_latents"):
            video = model.decode_latents(state.latents)
        # Pack decoded video as uint8 before it crosses the worker->driver
        # wire: every downstream consumer (reward models, mp4 artifacts)
        # quantizes to uint8 anyway, so fp32 here is 4x wasted transfer
        # (~474MB/group). The driver reconstructs [0, 1] floats as k/255,
        # which round-trips bit-exactly through the same to_uint8 formula.
        # Training tensors (latents/log_probs/replay) are NOT touched.
        if isinstance(video, torch.Tensor) and video.is_floating_point():
            video = to_uint8(video)
        replay_tensors = model.export_replay_tensors(state)
        # Carry the rollout proposal mean alongside the model's replay tensors so
        # it concatenates + lands under the denoise segment like old_log_prob,
        # readable at replay via replay_tensor_dict("denoise"). Only present when
        # a trust-region recipe opted in (return_prev_sample_mean).
        if denoise_result.prev_sample_means is not None:
            replay_tensors = {
                **replay_tensors,
                "old_prev_sample_mean": denoise_result.prev_sample_means,
            }
        # Frozen reference noise_pred per step (cache_ref_noise_pred). Lands under
        # the denoise segment like old_log_prob; the SDE evaluator reads it at
        # replay via replay_tensor_dict("denoise") to skip the ref forward.
        if denoise_result.ref_noise_preds is not None:
            replay_tensors = {
                **replay_tensors,
                "ref_noise_pred": denoise_result.ref_noise_preds,
            }
        context = dict(model.export_batch_context(state))

        decode_peak_bytes = cuda_phase_peak_bytes()
        memory = None
        if denoise_result.memory is not None and decode_peak_bytes is not None:
            memory = {**denoise_result.memory, "decode_peak_bytes": decode_peak_bytes}
        decode_peak_mb = None if decode_peak_bytes is None else decode_peak_bytes / (1024 * 1024)
        phase_peaks = [
            peak for peak in (denoise_result.peak_memory_mb, decode_peak_mb) if peak is not None
        ]

        return DiffusionChunkResult(
            prompt_index=config.prompt_index,
            sample_start=config.sample_start,
            sample_count=config.sample_count,
            observations=denoise_result.observations,
            actions=denoise_result.actions,
            log_probs=denoise_result.log_probs,
            timesteps=denoise_result.timesteps,
            kl=denoise_result.kl,
            video=video,
            replay_tensors=replay_tensors,
            context=context,
            peak_memory_mb=max(phase_peaks) if phase_peaks else None,
            memory=memory,
            stage_durations=dict(stage_durations or {}),
            engine_counters={
                **denoise_result.engine_counters,
                "diffusion_replay_tensor_bytes": trajectory_tensor_bytes(replay_tensors),
                "diffusion_video_bytes": trajectory_tensor_bytes(video),
            },
        )

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        chunks: Sequence[DiffusionChunkResult],
    ) -> GenerationOutput:
        return DiffusionChunkGatherer().gather_chunks(
            request,
            sample_rows,
            chunks,
        )

    # -- family hooks --------------------------------------------------

    def encode_prompt_for_chunk(
        self,
        *,
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        params: DiffusionSamplingParams,
        chunk: SampleChunk,
    ) -> dict[str, Any]:
        """Encode prompt conditioning for a single prompt chunk."""

        del generation_request
        return self.model.encode_prompt(
            chunk.prompt,
            video_request.negative_prompt or None,
            max_sequence_length=params.base.max_sequence_length,
            guidance_scale=params.base.guidance_scale,
            request=video_request,
        )

    # Encoded keys copied through UNREPEATED by the default build_chunk_encoded.
    # For batch-shared tensors whose leading dim is not a batch axis (FLUX's
    # ``text_ids`` is ``[seq, 3]``), the generic repeat would corrupt the shape,
    # so the family lists them here instead of overriding the whole method.
    # Non-tensor values (PIL reference images, python lists) already pass
    # through ``repeat_batch`` untouched and need no listing.
    chunk_passthrough_keys: tuple[str, ...] = ()

    def build_chunk_encoded(
        self,
        *,
        encoded: dict[str, Any],
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        params: DiffusionSamplingParams,
        chunk: SampleChunk,
    ) -> dict[str, Any]:
        """Build per-sample encoded tensors for one sample chunk."""

        del generation_request, video_request, params
        if not self.chunk_passthrough_keys:
            return self.layout.repeat_encoded_batch(encoded, chunk.sample_count)
        passthrough = set(self.chunk_passthrough_keys)
        return {
            key: (
                value
                if key in passthrough
                else self.layout.repeat_batch(value, chunk.sample_count)
            )
            for key, value in encoded.items()
        }

    def build_prepare_kwargs(
        self,
        *,
        encoded: dict[str, Any],
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        params: DiffusionSamplingParams,
        chunk: SampleChunk,
    ) -> dict[str, Any] | None:
        """Return additional family kwargs for model.prepare_sampling."""

        del encoded, generation_request, video_request, params, chunk
        return None


__all__ = [
    "DiffusionChunkExecutor",
    "DiffusionChunkExecutorBase",
    "DiffusionChunkResult",
    "DiffusionDenoisedStageOutput",
    "DiffusionPreparedStageOutput",
    "DiffusionPromptStageInput",
    "DiffusionPromptStageOutput",
    "DiffusionRequestLayout",
    "DiffusionSamplingParams",
    "ReferenceConditionedChunks",
]


class DiffusionChunkExecutor(DiffusionChunkExecutorBase):
    """Generic chunk executor for pure-data diffusion families.

    A family whose executor overrides no method (no ``build_chunk_encoded`` /
    ``encode_prompt_for_chunk``) is pure configuration: ``family`` / ``task``
    plus a few ``default_*`` values. Rather than ship a boilerplate subclass,
    it declares a ``model.executor`` block in its model config yaml and
    dispatches here; the launcher reads that block wholesale into these
    constructor kwargs (family/task come from the registry entry, the worker
    injects family/task from the launch contract). Families with real
    per-chunk tensor logic (cosmos predict2/2.5, cosmos3, echo, wan i2v) keep
    their own subclass.
    """

    def __init__(
        self,
        model: Any,
        *,
        family: str,
        task: str,
        num_frames: int = 1,
        max_sequence_length: int = 512,
        fps: int | None = None,
        chunk_passthrough_keys: tuple[str, ...] = (),
        samples_per_chunk: int = 8,
    ) -> None:
        self.model = model
        self.family = family
        self.task = task
        self.default_num_frames = int(num_frames)
        self.default_max_sequence_length = int(max_sequence_length)
        self.default_fps = None if fps is None else int(fps)
        self.chunk_passthrough_keys = tuple(chunk_passthrough_keys)
        self.default_samples_per_chunk = max(1, int(samples_per_chunk))
