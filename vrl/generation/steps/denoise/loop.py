"""Continuous denoise loop independent of output temporal organization."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import torch

from vrl.generation.steps.denoise.config import DenoiseLoopConfig
from vrl.generation.steps.denoise.teacache import (
    TeaCacheState,
    teacache_signal,
)
from vrl.math.denoise.flow_matching import sde_step_with_logprob
from vrl.trajectory.storage import trajectory_tensor_bytes
from vrl.utils.profiling import record_function


@dataclass(slots=True)
class DenoiseLoopResult:
    """Denoise-loop output before decode and artifact packing."""

    state: Any
    observations: Any
    actions: Any
    log_probs: Any
    timesteps: Any
    kl: Any
    prev_sample_means: Any | None = None
    ref_noise_preds: Any | None = None
    peak_memory_mb: float | None = None
    memory: dict[str, int] | None = None
    engine_counters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DenoiseTrajectoryBuffers:
    """Preallocated replay tensors written once per denoise step."""

    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    timesteps: torch.Tensor
    kl: torch.Tensor
    prev_sample_means: torch.Tensor | None = None
    ref_noise_preds: torch.Tensor | None = None


def reset_cuda_peak() -> None:
    """Reset process CUDA peak counters at a generation phase boundary."""

    if torch.cuda.is_available():
        with suppress(Exception):
            torch.cuda.reset_peak_memory_stats()


def cuda_phase_peak_bytes() -> int | None:
    """Return the current phase's CUDA allocation peak when available."""

    if not torch.cuda.is_available():
        return None
    try:
        return int(torch.cuda.max_memory_allocated())
    except Exception:
        return None


def _cuda_occupancy_snapshot() -> dict[str, int] | None:
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
    config: DenoiseLoopConfig,
) -> DenoiseTrajectoryBuffers:
    """Allocate final replay tensors before entering the denoise loop."""

    latents = state.latents
    if not isinstance(latents, torch.Tensor):
        raise TypeError("denoise state.latents must be a torch.Tensor")
    chunk_batch = int(latents.shape[0])
    if chunk_batch != int(config.sample_count):
        raise ValueError(
            f"denoise chunk produced {chunk_batch} rows, expected {config.sample_count}",
        )
    num_steps = len(state.timesteps)
    latent_shape = tuple(latents.shape[1:])
    device = latents.device
    timestep_dtype = _timestep_dtype(state.timesteps)

    return DenoiseTrajectoryBuffers(
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


def run_denoise_loop(
    *,
    model: Any,
    state: Any,
    config: DenoiseLoopConfig,
) -> DenoiseLoopResult:
    """Run continuous denoise steps and collect replay tensors."""

    chunk_batch = state.latents.shape[0]
    device = state.latents.device
    reset_cuda_peak()
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
    teacache = (
        TeaCacheState(config.teacache, len(state.timesteps))
        if config.teacache is not None and config.teacache.enabled
        else None
    )

    num_steps_to_run = len(state.timesteps)
    if config.execute_steps is not None:
        num_steps_to_run = max(1, min(num_steps_to_run, int(config.execute_steps)))
    with torch.no_grad():
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
                    with (
                        record_function("generation.ref_denoise_forward"),
                        model.disable_adapter(),
                    ):
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
                        sde_result.log_prob.detach().abs().to(dtype=buffers.kl.dtype),
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
                        ref_noise_pred.detach().to(dtype=buffers.ref_noise_preds.dtype),
                    )

    denoise_peak_bytes = cuda_phase_peak_bytes()
    peak_memory_mb = None if denoise_peak_bytes is None else denoise_peak_bytes / (1024 * 1024)
    memory = None
    if occupancy is not None and denoise_peak_bytes is not None:
        memory = {
            "sample_count": int(chunk_batch),
            "latent_bytes": latent_bytes,
            **occupancy,
            "denoise_peak_bytes": denoise_peak_bytes,
        }

    return DenoiseLoopResult(
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
            "diffusion_observation_bytes": trajectory_tensor_bytes(buffers.observations),
            "diffusion_action_bytes": trajectory_tensor_bytes(buffers.actions),
            "diffusion_old_logprob_bytes": trajectory_tensor_bytes(buffers.log_probs),
            "diffusion_timestep_bytes": trajectory_tensor_bytes(buffers.timesteps),
            "diffusion_kl_bytes": trajectory_tensor_bytes(buffers.kl),
            "diffusion_ref_noise_pred_bytes": (
                trajectory_tensor_bytes(buffers.ref_noise_preds)
                if buffers.ref_noise_preds is not None
                else 0
            ),
            "diffusion_denoise_mode": config.denoise_mode,
            **(teacache.counters() if teacache is not None else {}),
        },
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
            "denoise timestep cannot be expanded to chunk batch "
            f"{chunk_batch}: shape={tuple(timestep.shape)}",
        ) from exc


__all__ = [
    "DenoiseLoopResult",
    "DenoiseTrajectoryBuffers",
    "cuda_phase_peak_bytes",
    "preallocate_denoise_buffers",
    "reset_cuda_peak",
    "run_denoise_loop",
]
