"""Shared executor scaffolding for diffusion generation families."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch

from vrl.generation.capabilities import FamilyCapability
from vrl.generation.diffusion.gather import DiffusionChunkGatherer
from vrl.generation.diffusion.layout import (
    DiffusionRequestLayout,
    DiffusionSamplingParams,
    VideoGenerationRequest,
)
from vrl.generation.execution.chunks import (
    SampleChunk,
    run_sample_chunks_with_oom_retry,
)
from vrl.generation.execution.planner import attach_engine_plan, build_engine_plan
from vrl.generation.execution.request_batch import RequestBatch
from vrl.generation.protocols import (
    PipelineExecutor,
)
from vrl.generation.types import (
    GenerationOutput,
    GenerationRequest,
    GenerationSampleRow,
    WorkloadSignature,
)
from vrl.math.diffusion.flow_matching import sde_step_with_logprob
from vrl.trajectory.storage import trajectory_tensor_bytes


@dataclass(frozen=True, slots=True)
class DiffusionDenoiseConfig:
    """Runtime knobs for one diffusion sample chunk denoise loop."""

    prompt_index: int
    sample_start: int
    sample_count: int
    seed: int | None
    same_latent: bool
    sde_window: tuple[int, int] | None
    return_kl: bool
    noise_level: float = 1.0
    sde_type: str = "sde"
    denoise_mode: str = "sde"


@dataclass(slots=True)
class DiffusionDenoiseResult:
    """Trainable denoise-stage output before VAE decode and artifact packing."""

    state: Any
    observations: Any
    actions: Any
    log_probs: Any
    timesteps: Any
    kl: Any
    peak_memory_mb: float | None = None
    engine_counters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DiffusionDenoiseBuffers:
    """Preallocated replay tensors written once per denoise step."""

    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    timesteps: torch.Tensor
    kl: torch.Tensor


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
    stage_durations: dict[str, float] = field(default_factory=dict)
    engine_counters: dict[str, Any] = field(default_factory=dict)


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


class DiffusionPipelineExecutorBase(
    PipelineExecutor,
):
    """Common GenerationRequest -> diffusion GenerationOutput execution path."""

    family: str
    task: str
    model: Any
    default_sample_batch_size: int = 1
    default_num_frames: int = 1
    default_fps: int | None = None
    default_max_sequence_length: int = 512
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
        sample_rows: list[GenerationSampleRow],
    ) -> Any:
        params = self.parse_sampling_params(request)
        return build_engine_plan(
            request,
            sample_rows,
            capability=self.capability(),
            max_samples_per_chunk=params.base.sample_batch_size,
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
            same_latent=params.sde.same_latent,
            sde_window=layout.select_sde_window(
                params.sde.sde_window_size,
                params.sde.sde_window_range,
            ),
            return_kl=params.sde.return_kl,
            noise_level=params.sde.noise_level,
            sde_type=params.sde.sde_type,
            denoise_mode=params.denoise_mode,
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
                plan.summary(),
            ),
        )
        return attach_engine_plan(self.gather_chunks(request, sample_rows, chunks), plan)

    def forward_chunk_plan(
        self,
        request: GenerationRequest,
        chunk: SampleChunk,
        execution_stage: Any,
        plan_summary: Mapping[str, object],
    ) -> DiffusionChunkResult:
        del execution_stage, plan_summary

        from vrl.utils.profiling import record_function

        params = self.parse_sampling_params(request)
        video_request = self.build_video_request(chunk.prompt, params)
        stage_durations: dict[str, float] = {}

        started = time.perf_counter()
        with record_function("generation.prompt_encode"):
            encoded = self.encode_prompt_for_chunk(
                generation_request=request,
                video_request=video_request,
                params=params,
                chunk=chunk,
            )
        stage_durations["encode"] = time.perf_counter() - started

        started = time.perf_counter()
        chunk_encoded = self.build_chunk_encoded(
            encoded=encoded,
            generation_request=request,
            video_request=video_request,
            params=params,
            chunk=chunk,
        )
        prepare_kwargs = self.build_prepare_kwargs(
            encoded=encoded,
            generation_request=request,
            video_request=video_request,
            params=params,
            chunk=chunk,
        )
        config = self.build_denoise_config(params, chunk)
        state = self.prepare_denoise_state(
            request=video_request,
            encoded=chunk_encoded,
            config=config,
            prepare_kwargs=prepare_kwargs,
        )
        stage_durations["prepare_latent"] = time.perf_counter() - started

        started = time.perf_counter()
        denoise_result = self.run_denoise_steps(
            state=state,
            encoded=chunk_encoded,
            config=config,
        )
        stage_durations["denoise"] = time.perf_counter() - started

        started = time.perf_counter()
        chunk_result = self.decode_denoise_result(
            request=video_request,
            config=config,
            denoise_result=denoise_result,
            stage_durations=stage_durations,
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
        if config.seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(config.seed + config.sample_start)
        elif config.same_latent:
            raise ValueError("same_latent=True requires an explicit sampling.seed")
        else:
            generator = None

        buffers = preallocate_denoise_buffers(state=state, config=config)

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
        with autocast_ctx, torch.no_grad():
            for step_idx in range(len(state.timesteps)):
                with record_function("generation.denoise_step"):
                    with record_function("generation.latent_snapshot"):
                        latents_ori = state.latents.clone()
                        timestep = state.timesteps[step_idx]

                    with record_function("generation.denoise_forward"):
                        step_output = model.forward_step(state, step_idx)
                    noise_pred = step_output["noise_pred"]

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
                            return_dt=config.return_kl,
                            noise_level=config.noise_level,
                            sde_type=config.sde_type,
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
                                return_dt=config.return_kl,
                                noise_level=config.noise_level,
                                sde_type=config.sde_type,
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
                    if config.return_kl:
                        buffers.kl[:, step_idx].copy_(
                            sde_result.log_prob.detach()
                            .abs()
                            .to(dtype=buffers.kl.dtype),
                        )
                    else:
                        buffers.kl[:, step_idx].zero_()
        peak_memory_mb = None
        if torch.cuda.is_available():
            try:
                peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            except Exception:
                peak_memory_mb = None

        return DiffusionDenoiseResult(
            state=state,
            observations=buffers.observations,
            actions=buffers.actions,
            log_probs=buffers.log_probs,
            timesteps=buffers.timesteps,
            kl=buffers.kl,
            peak_memory_mb=peak_memory_mb,
            engine_counters={
                "diffusion_num_denoise_steps": int(buffers.timesteps.shape[1]),
                "diffusion_sample_batch_size": int(chunk_batch),
                "diffusion_observation_bytes": trajectory_tensor_bytes(
                    buffers.observations,
                ),
                "diffusion_action_bytes": trajectory_tensor_bytes(buffers.actions),
                "diffusion_old_logprob_bytes": trajectory_tensor_bytes(
                    buffers.log_probs,
                ),
                "diffusion_timestep_bytes": trajectory_tensor_bytes(buffers.timesteps),
                "diffusion_kl_bytes": trajectory_tensor_bytes(buffers.kl),
                "diffusion_denoise_mode": config.denoise_mode,
                "diffusion_rollout_transformer_dtype": _dtype_label(transformer_dtype),
                "diffusion_rollout_autocast_enabled": rollout_autocast_enabled,
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
        with record_function("generation.decode_latents"):
            video = model.decode_latents(state.latents)
        replay_tensors = model.export_replay_tensors(state)
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
            peak_memory_mb=denoise_result.peak_memory_mb,
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

    def forward_batch_plan(
        self,
        requests: list[GenerationRequest],
        sample_rows_by_request: dict[str, list[GenerationSampleRow]],
        engine_plans_by_request: dict[str, Any],
    ) -> dict[str, GenerationOutput]:
        def forward(
            request: GenerationRequest,
            sample_rows: list[GenerationSampleRow],
        ) -> GenerationOutput:
            plan = engine_plans_by_request.get(request.request_id)
            if plan is None:
                plan = self.plan(request, sample_rows)
            output = self.forward_plan(request, sample_rows, plan)
            execution_extra = output.extra.setdefault("engine_execution", {})
            if isinstance(execution_extra, dict):
                execution_extra["plan_aware_forward"] = True
                execution_extra["forward_plan_id"] = plan.request_id
            return output

        outputs = RequestBatch(
            requests=requests,
            sample_rows_by_request=sample_rows_by_request,
        ).run(forward)
        return {
            request_id: attach_engine_plan(output, engine_plans_by_request[request_id])
            for request_id, output in outputs.items()
        }

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
        return self.layout.repeat_encoded_batch(encoded, chunk.sample_count)

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
    "DiffusionChunkResult",
    "DiffusionDenoiseConfig",
    "DiffusionDenoiseResult",
    "DiffusionPipelineExecutorBase",
    "DiffusionRequestLayout",
    "DiffusionSamplingParams",
]
