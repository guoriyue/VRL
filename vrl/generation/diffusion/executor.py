"""Shared executor scaffolding for diffusion generation families."""

from __future__ import annotations

import time
from collections.abc import Sequence
from contextlib import nullcontext, suppress
from dataclasses import dataclass, field
from typing import Any

import torch

from vrl.generation.capabilities import FamilyCapability
from vrl.generation.diffusion.gather import DiffusionChunkGatherer
from vrl.generation.diffusion.layout import (
    DiffusionRequestLayout,
    DiffusionSamplingParams,
    DiffusionSDEParams,
    VideoGenerationRequest,
)
from vrl.generation.diffusion.teacache import (
    TeaCacheConfig,
    TeaCacheState,
    teacache_signal,
)
from vrl.generation.execution.chunks import (
    SampleChunk,
    run_sample_chunks_with_oom_retry,
)
from vrl.generation.execution.planner import attach_engine_plan, build_engine_plan
from vrl.generation.protocols import (
    GenerationChunkExecutor,
)
from vrl.generation.types import (
    GenerationOutput,
    GenerationRequest,
    GenerationSampleRow,
)
from vrl.math.diffusion.flow_matching import sde_step_with_logprob
from vrl.trajectory.storage import (
    TrajectoryStoragePolicy,
    apply_value_storage_policy,
    trajectory_storage_policy_from_cfg,
    trajectory_tensor_bytes,
)
from vrl.utils.media import load_reference_image, to_uint8


@dataclass(frozen=True, slots=True)
class DiffusionDenoiseConfig:
    """Runtime knobs for one diffusion sample chunk denoise loop.

    Chunk identity (prompt_index/sample_start/sample_count/seed), the per-chunk
    resolved ``sde_window``, and cross-group knobs (denoise_mode, teacache). The
    SDE pass-through knobs are NOT re-declared here — they live on ``sde`` (the
    parsed DiffusionSDEParams) and are read via ``config.sde.<knob>``. One source
    of truth: a new SDE knob is added in DiffusionSDEParams alone and cannot rot
    into a silent no-op by being forgotten in a field-by-field copy.
    """

    prompt_index: int
    sample_start: int
    sample_count: int
    seed: int | None
    # Parsed SDE knobs (noise_level / sde_type / same_latent / return_kl /
    # return_prev_sample_mean / cache_ref_noise_pred). Read via config.sde.<knob>.
    sde: DiffusionSDEParams
    # Per-chunk resolved denoise window: select_sde_window picks it from
    # sde.sde_window_size/range and may be random per chunk, so it is resolved
    # here rather than carried on the request-scoped sde params.
    sde_window: tuple[int, int] | None
    denoise_mode: str = "sde"
    # Opt-in rollout-only TeaCache (skip transformer forwards on low-change steps).
    # None = off = the unchanged full-forward path, so the GRPO baseline is exact.
    teacache: TeaCacheConfig | None = None
    # Startup chunk-size probe only (SPRINT_chunk_size_probe): execute just this
    # many denoise steps while STILL preallocating full-size trajectory buffers,
    # so the memory peak is real but a probe trial costs seconds, not minutes.
    # None = run every step (the unchanged training path).
    execute_steps: int | None = None


@dataclass(slots=True)
class DiffusionDenoiseResult:
    """Trainable denoise-stage output before VAE decode and artifact packing."""

    state: Any
    observations: Any
    actions: Any
    log_probs: Any
    timesteps: Any
    kl: Any
    prev_sample_means: Any | None = None
    ref_noise_preds: Any | None = None
    peak_memory_mb: float | None = None
    # Partial ChunkMemoryReading fields (denoise phase); decode_denoise_result
    # completes it with the decode-phase peak. None off-CUDA.
    memory: dict[str, int] | None = None
    engine_counters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DiffusionDenoiseBuffers:
    """Preallocated replay tensors written once per denoise step."""

    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    timesteps: torch.Tensor
    kl: torch.Tensor
    # Rollout proposal mean per step (full latent shape); None unless the denoise
    # config opts in via return_prev_sample_mean.
    prev_sample_means: torch.Tensor | None = None
    # Frozen reference noise_pred per step (full latent shape); None unless the
    # denoise config opts in via cache_ref_noise_pred.
    ref_noise_preds: torch.Tensor | None = None


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

    generation_request: GenerationRequest
    chunk: SampleChunk
    params: DiffusionSamplingParams
    video_request: VideoGenerationRequest
    encoded: dict[str, Any]
    chunk_encoded: dict[str, Any]
    prepare_kwargs: dict[str, Any]
    config: DiffusionDenoiseConfig
    state: Any


@dataclass(slots=True)
class DiffusionDenoisedStageOutput:
    """Denoise stage output before VAE decode."""

    request: VideoGenerationRequest
    config: DiffusionDenoiseConfig
    denoise_result: DiffusionDenoiseResult
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
    peak_memory_mb: float | None = None
    # Completed ChunkMemoryReading fields for byte-admission shadow telemetry
    # (see vrl/generation/execution/chunk_placement.py). None off-CUDA.
    memory: dict[str, int] | None = None
    stage_durations: dict[str, float] = field(default_factory=dict)
    engine_counters: dict[str, Any] = field(default_factory=dict)


def _reset_cuda_peak() -> None:
    """Rezero the per-process CUDA peak counters at a chunk phase boundary.

    Without this, ``max_memory_allocated()`` is a process-lifetime high-water
    mark: chunk 2's "peak" silently includes chunk 1's decode spike, poisoning
    the byte-admission calibration data. Per-phase resets make the denoise
    plateau and decode spike separately attributable (the two coefficients the
    L2 cost model needs).
    """

    if torch.cuda.is_available():
        with suppress(Exception):
            torch.cuda.reset_peak_memory_stats()


def _cuda_phase_peak_bytes() -> int | None:
    if not torch.cuda.is_available():
        return None
    try:
        return int(torch.cuda.max_memory_allocated())
    except Exception:
        return None


def _cuda_occupancy_snapshot() -> dict[str, int] | None:
    """Allocated/reserved/free/total bytes at a phase boundary (None off-CUDA)."""

    if not torch.cuda.is_available():
        return None
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "baseline_allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_start_bytes": int(torch.cuda.memory_reserved()),
            "free_start_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
        }
    except Exception:
        return None


def preallocate_denoise_buffers(
    *,
    state: Any,
    config: DiffusionDenoiseConfig,
) -> DiffusionDenoiseBuffers:
    """Allocate final denoise replay tensors before entering the step loop."""

    latents = state.latents
    if not isinstance(latents, torch.Tensor):
        raise TypeError("diffusion denoise state.latents must be a torch.Tensor")
    chunk_batch = int(latents.shape[0])
    if chunk_batch != int(config.sample_count):
        raise ValueError(
            "Diffusion denoise chunk produced "
            f"{chunk_batch} rows, expected {config.sample_count}",
        )
    num_steps = len(state.timesteps)
    latent_shape = tuple(latents.shape[1:])
    device = latents.device
    timestep_dtype = _timestep_dtype(state.timesteps)

    return DiffusionDenoiseBuffers(
        observations=torch.empty(
            (chunk_batch, num_steps, *latent_shape),
            dtype=latents.dtype,
            device=device,
        ),
        actions=torch.empty(
            (chunk_batch, num_steps, *latent_shape),
            dtype=latents.dtype,
            device=device,
        ),
        log_probs=torch.empty((chunk_batch, num_steps), dtype=torch.float32, device=device),
        timesteps=torch.empty(
            (chunk_batch, num_steps),
            dtype=timestep_dtype,
            device=device,
        ),
        kl=torch.empty((chunk_batch, num_steps), dtype=torch.float32, device=device),
        prev_sample_means=(
            torch.empty(
                (chunk_batch, num_steps, *latent_shape),
                dtype=latents.dtype,
                device=device,
            )
            if config.sde.return_prev_sample_mean
            else None
        ),
        ref_noise_preds=(
            torch.empty(
                (chunk_batch, num_steps, *latent_shape),
                dtype=latents.dtype,
                device=device,
            )
            if config.sde.cache_ref_noise_pred
            else None
        ),
    )


def _timestep_dtype(timesteps: Any) -> torch.dtype:
    if isinstance(timesteps, torch.Tensor):
        return timesteps.dtype
    try:
        first = timesteps[0]
    except Exception:
        return torch.float32
    if isinstance(first, torch.Tensor):
        return first.dtype
    return torch.float32


def _dtype_label(dtype: Any) -> str | None:
    if dtype is None:
        return None
    return str(dtype).removeprefix("torch.")


def _expand_timestep_for_buffer(
    timestep: Any,
    *,
    chunk_batch: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(timestep, torch.Tensor):
        timestep = torch.as_tensor(timestep)
    timestep = timestep.to(device=device, dtype=dtype)
    if timestep.ndim == 0:
        return timestep.expand(chunk_batch)
    if tuple(timestep.shape) == (chunk_batch,):
        return timestep
    if timestep.numel() == 1:
        return timestep.reshape(()).expand(chunk_batch)
    try:
        return timestep.reshape(chunk_batch)
    except RuntimeError as exc:
        raise ValueError(
            "diffusion timestep cannot be expanded to chunk batch "
            f"{chunk_batch}: shape={tuple(timestep.shape)}",
        ) from exc


class ReferenceConditionedChunks:
    """Reference-image threading for per-chunk encode/prepare.

    Cosmos Predict2 Video2World and Wan 2.1 I2V condition every chunk on a
    reference image that can arrive globally (executor attribute) or
    per-sample (request metadata). The two executors had copy-pasted these
    hooks; ``build_chunk_encoded`` stays family-specific because the encoded
    payloads genuinely differ (Wan carries ``image_embeds``).
    """

    model: Any
    reference_image: Any

    def encode_prompt_for_chunk(
        self,
        *,
        generation_request: GenerationRequest,
        video_request: VideoGenerationRequest,
        params: Any,
        chunk: SampleChunk,
    ) -> dict[str, Any]:
        """Encode text plus the active reference-image conditioning for one chunk."""

        reference_image = self._reference_image_for_request(generation_request)
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

        del encoded, video_request, params, chunk
        return {
            "reference_image": self._reference_image_for_request(
                generation_request,
            ),
        }

    def _reference_image_for_request(self, request: GenerationRequest) -> Any:
        ref = request.metadata.get("reference_image", self.reference_image)
        if isinstance(ref, str) and ref:
            # The manifest stores reference_image as a data-root-relative path
            # ("video_world/references/x.png"). Resolve it against the data root the
            # SAME way the reward resolves target_video (target_dino_similarity._resolve),
            # else the rollout opens it relative to CWD and FileNotFoundErrors. The
            # intended resolver (resolve_prompt_example_artifacts) is never wired into
            # prompt loading, so per-sample reference paths arrive here unresolved.
            from vrl.utils.artifacts import default_data_root, resolve_artifact_path

            ref = str(resolve_artifact_path(ref, data_root=default_data_root(), allow_absolute=True))
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
    family_capability: FamilyCapability | None = None

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

    def capability(self) -> FamilyCapability:
        if self.family_capability is None:
            raise RuntimeError(
                f"{type(self).__name__} must declare family_capability explicitly"
            )
        return self.family_capability

    def plan(
        self,
        request: GenerationRequest,
        sample_rows: list[GenerationSampleRow],
    ) -> Any:
        params = self.parse_sampling_params(request)
        return build_engine_plan(
            request,
            sample_rows,
            capability=self.capability(),
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

        extra = self.build_video_request_extra(params)
        if extra:
            req_kwargs["extra"] = extra
        return VideoGenerationRequest(**req_kwargs)

    def build_video_request_extra(
        self,
        params: DiffusionSamplingParams,
    ) -> dict[str, Any]:
        """Return family-neutral request.extra payload."""

        if not self.include_max_sequence_length_extra:
            return {}
        return {"max_sequence_length": params.base.max_sequence_length}

    def build_denoise_config(
        self,
        params: DiffusionSamplingParams,
        chunk: SampleChunk,
    ) -> DiffusionDenoiseConfig:
        """Build the SDE denoise config for one sample chunk."""

        if params.sde is None:
            raise NotImplementedError(
                f"{type(self).__name__} must override denoise for non-SDE diffusion",
            )
        layout = self.layout
        return DiffusionDenoiseConfig(
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
            lambda chunk: self.forward_chunk_plan(
                request,
                chunk,
                plan.chunk_stage_for(chunk),
            ),
        )
        return attach_engine_plan(self.gather_chunks(request, sample_rows, chunks), plan)

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
        methods (via run_chunk_through_pipeline), value-preserving side-stream copy,
        and the SAME order-preserving gather_chunks — so the gathered output is
        identical; only the wall-clock changes.

        This is the executor-level entry for the single-GPU stage-overlap lever; the
        Ray worker calls it per-request (all of a request's chunks on one worker)
        instead of dispatching one monolithic forward_chunk_plan per chunk.
        """

        from vrl.generation.diffusion.pipeline import forward_chunks_pipelined

        chunks = forward_chunks_pipelined(self, request, plan.chunks)
        return attach_engine_plan(self.gather_chunks(request, sample_rows, chunks), plan)

    def forward_chunk_plan(
        self,
        request: GenerationRequest,
        chunk: SampleChunk,
        execution_stage: Any,
    ) -> DiffusionChunkResult:
        del execution_stage

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
            chunk_result.observations, policy,
        )
        chunk_result.actions = apply_value_storage_policy(chunk_result.actions, policy)
        chunk_result.log_probs = apply_value_storage_policy(chunk_result.log_probs, policy)
        chunk_result.timesteps = apply_value_storage_policy(chunk_result.timesteps, policy)
        chunk_result.kl = apply_value_storage_policy(chunk_result.kl, policy)
        chunk_result.replay_tensors = apply_value_storage_policy(
            chunk_result.replay_tensors, policy,
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
            generation_request=payload.generation_request,
            chunk=payload.chunk,
            params=payload.params,
            video_request=payload.video_request,
            encoded=payload.encoded,
            chunk_encoded=chunk_encoded,
            prepare_kwargs=prepare_kwargs,
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
            request=payload.video_request,
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
            request=payload.request,
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
        config: DiffusionDenoiseConfig,
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
        config: DiffusionDenoiseConfig,
    ) -> DiffusionDenoiseResult:
        """Run the trainable denoise stage and collect replay tensors."""

        from vrl.utils.profiling import record_function

        model = self.model
        chunk_batch = state.latents.shape[0]
        device = state.latents.device
        # Byte-admission shadow: baseline = weights + encoded + initial latents
        # resident before the denoise workspace (trajectory buffers, CFG batch,
        # fp32 SDE copies) grows on top of it.
        _reset_cuda_peak()
        occupancy = _cuda_occupancy_snapshot()
        latent_bytes = int(state.latents.numel() * state.latents.element_size())
        if config.seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(config.seed + config.sample_start)
        elif config.sde.same_latent:
            raise ValueError("same_latent=True requires an explicit sampling.seed")
        else:
            generator = None

        buffers = preallocate_denoise_buffers(state=state, config=config)

        # Rollout-only TeaCache: skip the transformer forward on low-change steps
        # and reuse the cached noise_pred. None when disabled (baseline unchanged).
        # The introduced rollout-vs-replay drift rides the same drift-guard/TIS
        # correction as fp8 — see vrl/generation/diffusion/teacache.py.
        teacache = (
            TeaCacheState(config.teacache, len(state.timesteps))
            if config.teacache is not None and config.teacache.enabled
            else None
        )

        prompt_embeds = encoded.get("prompt_embeds")
        transformer_dtype = (
            prompt_embeds.dtype
            if isinstance(prompt_embeds, torch.Tensor)
            else state.latents.dtype
        )
        if getattr(state.latents.device, "type", None) == "cuda" and transformer_dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            autocast_ctx = torch.amp.autocast("cuda", dtype=transformer_dtype)
            rollout_autocast_enabled = True
        else:
            autocast_ctx = nullcontext()
            rollout_autocast_enabled = False
        num_steps_to_run = len(state.timesteps)
        if config.execute_steps is not None:
            num_steps_to_run = max(1, min(num_steps_to_run, int(config.execute_steps)))
        with autocast_ctx, torch.no_grad():
            for step_idx in range(num_steps_to_run):
                with record_function("generation.denoise_step"):
                    with record_function("generation.latent_snapshot"):
                        latents_ori = state.latents.clone()
                        timestep = state.timesteps[step_idx]

                    if teacache is not None and not teacache.should_run(
                        teacache_signal(latents_ori, config.teacache.signal),
                        step_idx,
                    ):
                        noise_pred = teacache.cached_noise_pred
                    else:
                        with record_function("generation.denoise_forward"):
                            step_output = model.forward_step(state, step_idx)
                        noise_pred = step_output["noise_pred"]
                        if teacache is not None:
                            teacache.cache_noise_pred(noise_pred)

                    ref_noise_pred = None
                    if buffers.ref_noise_preds is not None:
                        # Frozen reference forward (LoRA disabled) at collect time.
                        # state.latents is still x_t here — the scheduler step below
                        # overwrites it — so the ref sees the exact input the KL
                        # replay would. Computed once per step; replay reads it back
                        # instead of rerunning this forward every ppo_epoch.
                        # Already inside the loop's torch.no_grad(); just drop the
                        # adapter so the forward is the frozen base model.
                        with record_function(
                            "generation.ref_denoise_forward",
                        ), model.disable_adapter():
                            ref_step_output = model.forward_step(state, step_idx)
                        ref_noise_pred = ref_step_output["noise_pred"]

                    if config.denoise_mode == "native":
                        with record_function("generation.scheduler_step"):
                            prev_latents = state.scheduler.step(
                                noise_pred,
                                timestep,
                                state.latents,
                                return_dict=False,
                            )[0]
                        sde_result = sde_step_with_logprob(
                            state.scheduler,
                            noise_pred.float(),
                            timestep.unsqueeze(0),
                            state.latents.float(),
                            prev_sample=prev_latents.float(),
                            return_dt=config.sde.return_kl,
                            noise_level=config.sde.noise_level,
                            sde_type=config.sde.sde_type,
                            step_index=step_idx,
                        )
                    else:
                        in_sde_window = config.sde_window is None or (
                            config.sde_window[0] <= step_idx < config.sde_window[1]
                        )
                        with record_function("generation.scheduler_step"):
                            sde_result = sde_step_with_logprob(
                                state.scheduler,
                                noise_pred.float(),
                                timestep.unsqueeze(0),
                                state.latents.float(),
                                generator=generator if in_sde_window else None,
                                deterministic=not in_sde_window,
                                return_dt=config.sde.return_kl,
                                noise_level=config.sde.noise_level,
                                sde_type=config.sde.sde_type,
                                step_index=step_idx,
                            )
                        prev_latents = sde_result.prev_sample
                    with record_function("generation.latent_write"):
                        state.latents = prev_latents

                with record_function("generation.trajectory_buffer_write"):
                    buffers.observations[:, step_idx].copy_(latents_ori.detach())
                    buffers.actions[:, step_idx].copy_(
                        prev_latents.detach().to(dtype=buffers.actions.dtype),
                    )
                    buffers.log_probs[:, step_idx].copy_(
                        sde_result.log_prob.detach().to(dtype=buffers.log_probs.dtype),
                    )
                    buffers.timesteps[:, step_idx].copy_(
                        _expand_timestep_for_buffer(
                            timestep.detach(),
                            chunk_batch=chunk_batch,
                            dtype=buffers.timesteps.dtype,
                            device=device,
                        ),
                    )
                    if config.sde.return_kl:
                        buffers.kl[:, step_idx].copy_(
                            sde_result.log_prob.detach()
                            .abs()
                            .to(dtype=buffers.kl.dtype),
                        )
                    else:
                        buffers.kl[:, step_idx].zero_()
                    if buffers.prev_sample_means is not None:
                        buffers.prev_sample_means[:, step_idx].copy_(
                            sde_result.prev_sample_mean.detach().to(
                                dtype=buffers.prev_sample_means.dtype,
                            ),
                        )
                    if buffers.ref_noise_preds is not None:
                        buffers.ref_noise_preds[:, step_idx].copy_(
                            ref_noise_pred.detach().to(
                                dtype=buffers.ref_noise_preds.dtype,
                            ),
                        )
        denoise_peak_bytes = _cuda_phase_peak_bytes()
        peak_memory_mb = (
            None if denoise_peak_bytes is None else denoise_peak_bytes / (1024 * 1024)
        )
        memory = None
        if occupancy is not None and denoise_peak_bytes is not None:
            memory = {
                "sample_count": int(chunk_batch),
                "latent_bytes": latent_bytes,
                **occupancy,
                "denoise_peak_bytes": denoise_peak_bytes,
            }

        return DiffusionDenoiseResult(
            state=state,
            observations=buffers.observations,
            actions=buffers.actions,
            log_probs=buffers.log_probs,
            timesteps=buffers.timesteps,
            kl=buffers.kl,
            prev_sample_means=buffers.prev_sample_means,
            ref_noise_preds=buffers.ref_noise_preds,
            peak_memory_mb=peak_memory_mb,
            memory=memory,
            engine_counters={
                "diffusion_num_denoise_steps": int(buffers.timesteps.shape[1]),
                "diffusion_samples_per_chunk": int(chunk_batch),
                "diffusion_observation_bytes": trajectory_tensor_bytes(
                    buffers.observations,
                ),
                "diffusion_action_bytes": trajectory_tensor_bytes(buffers.actions),
                "diffusion_old_logprob_bytes": trajectory_tensor_bytes(
                    buffers.log_probs,
                ),
                "diffusion_timestep_bytes": trajectory_tensor_bytes(buffers.timesteps),
                "diffusion_kl_bytes": trajectory_tensor_bytes(buffers.kl),
                "diffusion_ref_noise_pred_bytes": (
                    trajectory_tensor_bytes(buffers.ref_noise_preds)
                    if buffers.ref_noise_preds is not None
                    else 0
                ),
                "diffusion_denoise_mode": config.denoise_mode,
                "diffusion_rollout_transformer_dtype": _dtype_label(transformer_dtype),
                "diffusion_rollout_autocast_enabled": rollout_autocast_enabled,
                **(teacache.counters() if teacache is not None else {}),
            },
        )

    def decode_denoise_result(
        self,
        *,
        request: VideoGenerationRequest,
        config: DiffusionDenoiseConfig,
        denoise_result: DiffusionDenoiseResult,
        stage_durations: dict[str, float] | None = None,
    ) -> DiffusionChunkResult:
        """Decode the final latents and pack one diffusion chunk result."""

        from vrl.utils.profiling import record_function

        model = self.model
        state = denoise_result.state
        # Byte-admission shadow: the decode + pack spike is the second memory
        # phase, measured separately from the denoise plateau.
        _reset_cuda_peak()
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
        context.setdefault("denoise_mode", config.denoise_mode)
        context.setdefault(
            "rollout_transformer_dtype",
            denoise_result.engine_counters.get("diffusion_rollout_transformer_dtype"),
        )
        context.setdefault(
            "rollout_autocast_enabled",
            denoise_result.engine_counters.get("diffusion_rollout_autocast_enabled"),
        )

        decode_peak_bytes = _cuda_phase_peak_bytes()
        memory = None
        if denoise_result.memory is not None and decode_peak_bytes is not None:
            memory = {**denoise_result.memory, "decode_peak_bytes": decode_peak_bytes}
        decode_peak_mb = (
            None if decode_peak_bytes is None else decode_peak_bytes / (1024 * 1024)
        )
        phase_peaks = [
            peak
            for peak in (denoise_result.peak_memory_mb, decode_peak_mb)
            if peak is not None
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
    "DiffusionChunkExecutorBase",
    "DiffusionChunkResult",
    "DiffusionDenoiseBuffers",
    "DiffusionDenoiseConfig",
    "DiffusionDenoiseResult",
    "DiffusionDenoisedStageOutput",
    "DiffusionPreparedStageOutput",
    "DiffusionPromptStageInput",
    "DiffusionPromptStageOutput",
    "DiffusionRequestLayout",
    "DiffusionSamplingParams",
    "ReferenceConditionedChunks",
    "preallocate_denoise_buffers",
]
