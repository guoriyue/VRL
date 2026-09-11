"""Continuous denoise loop independent of output generation regime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from vrl.generation.execution.batch_memory import cuda_occupancy_snapshot
from vrl.generation.steps.denoise.config import DenoiseLoopConfig
from vrl.generation.steps.denoise.teacache import TeaCacheState
from vrl.math.denoise.flow_matching import SDEStepResult, sde_step_with_logprob
from vrl.trajectory.storage import trajectory_tensor_bytes
from vrl.utils.cuda_memory import (
    cuda_peak_allocated_bytes,
    cuda_peak_allocated_mb,
    reset_cuda_peak,
)
from vrl.utils.profiling import profile_range


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
    # display/provenance-only: measured peak forwarded to optional runtime-debug
    # telemetry after decode completes.
    peak_memory_mb: float | None = None
    memory: dict[str, int] | None = None
    # display/provenance-only: denoise-engine counters forwarded to optional
    # runtime-debug telemetry.
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

    @classmethod
    def allocate(
        cls,
        *,
        state: Any,
        config: DenoiseLoopConfig,
    ) -> DenoiseTrajectoryBuffers:
        """Allocate final replay tensors before entering the denoise loop."""

        latents = state.latents
        if not isinstance(latents, torch.Tensor):
            raise TypeError("denoise state.latents must be a torch.Tensor")
        batch_rows = int(latents.shape[0])
        if batch_rows != int(config.sample_count):
            raise ValueError(
                f"denoise batch produced {batch_rows} rows, expected {config.sample_count}",
            )
        latent_shape = tuple(latents.shape[1:])
        device = latents.device
        if not isinstance(state.timesteps, torch.Tensor):
            raise TypeError("denoise state.timesteps must be a torch.Tensor")
        num_steps = len(state.timesteps)
        timestep_dtype = state.timesteps.dtype

        return cls(
            observations=torch.empty(
                (batch_rows, num_steps, *latent_shape),
                dtype=latents.dtype,
                device=device,
            ),
            actions=torch.empty(
                (batch_rows, num_steps, *latent_shape),
                dtype=latents.dtype,
                device=device,
            ),
            log_probs=torch.empty((batch_rows, num_steps), dtype=torch.float32, device=device),
            timesteps=torch.empty(
                (batch_rows, num_steps),
                dtype=timestep_dtype,
                device=device,
            ),
            kl=torch.empty((batch_rows, num_steps), dtype=torch.float32, device=device),
            prev_sample_means=(
                torch.empty(
                    (batch_rows, num_steps, *latent_shape),
                    dtype=latents.dtype,
                    device=device,
                )
                if config.sde.return_prev_sample_mean
                else None
            ),
            ref_noise_preds=(
                torch.empty(
                    (batch_rows, num_steps, *latent_shape),
                    dtype=latents.dtype,
                    device=device,
                )
                if config.sde.cache_ref_noise_pred
                else None
            ),
        )

    def record_step(
        self,
        step_idx: int,
        *,
        observation: torch.Tensor,
        action: torch.Tensor,
        timestep: torch.Tensor,
        sde_result: SDEStepResult,
        return_kl: bool,
        ref_noise_pred: torch.Tensor | None = None,
    ) -> None:
        """Write one transition into the preallocated replay tensors."""
        self.observations[:, step_idx].copy_(observation.detach())
        self.actions[:, step_idx].copy_(
            action.detach().to(dtype=self.actions.dtype),
        )
        self.log_probs[:, step_idx].copy_(
            sde_result.log_prob.detach().to(dtype=self.log_probs.dtype),
        )
        self.timesteps[:, step_idx].copy_(
            self._expand_timestep(timestep.detach()),
        )
        if return_kl:
            self.kl[:, step_idx].copy_(
                sde_result.log_prob.detach().abs().to(dtype=self.kl.dtype),
            )
        else:
            self.kl[:, step_idx].zero_()
        if self.prev_sample_means is not None:
            self.prev_sample_means[:, step_idx].copy_(
                sde_result.prev_sample_mean.detach().to(
                    dtype=self.prev_sample_means.dtype,
                ),
            )
        if self.ref_noise_preds is not None:
            self.ref_noise_preds[:, step_idx].copy_(
                ref_noise_pred.detach().to(dtype=self.ref_noise_preds.dtype),
            )

    def _expand_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        batch_rows = self.timesteps.shape[0]
        dtype = self.timesteps.dtype
        device = self.timesteps.device
        timestep = timestep.to(device=device, dtype=dtype)
        if timestep.ndim == 0:
            return timestep.expand(batch_rows)
        if tuple(timestep.shape) == (batch_rows,):
            return timestep
        if timestep.numel() == 1:
            return timestep.reshape(()).expand(batch_rows)
        try:
            return timestep.reshape(batch_rows)
        except RuntimeError as exc:
            raise ValueError(
                "denoise timestep cannot be expanded to batch "
                f"{batch_rows}: shape={tuple(timestep.shape)}",
            ) from exc


def run_denoise_loop(
    *,
    model: Any,
    state: Any,
    config: DenoiseLoopConfig,
) -> DenoiseLoopResult:
    """Run continuous denoise steps and collect replay tensors."""

    batch_rows = state.latents.shape[0]
    device = state.latents.device
    reset_cuda_peak()
    occupancy = cuda_occupancy_snapshot()
    if config.seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(config.seed + config.sample_start)
    else:
        generator = None

    buffers = DenoiseTrajectoryBuffers.allocate(state=state, config=config)
    teacache = (
        TeaCacheState(config.teacache, len(state.timesteps))
        if config.teacache is not None
        else None
    )

    num_steps_to_run = len(state.timesteps)
    if config.execute_steps is not None:
        num_steps_to_run = min(num_steps_to_run, config.execute_steps)
    with torch.no_grad():
        for step_idx in range(num_steps_to_run):
            with profile_range("generation.denoise_step"):
                with profile_range("generation.latent_snapshot"):
                    latents_before_step = state.latents.clone()
                    timestep = state.timesteps[step_idx]

                if teacache is not None and not teacache.should_run(
                    latents_before_step,
                    step_idx,
                ):
                    noise_pred = teacache.cached_noise_pred
                else:
                    with profile_range("generation.denoise_forward"):
                        step_output = model.forward_step(state, step_idx)
                    noise_pred = step_output["noise_pred"]
                    if teacache is not None:
                        teacache.cache_noise_pred(noise_pred)

                ref_noise_pred = None
                if buffers.ref_noise_preds is not None:
                    with (
                        profile_range("generation.ref_denoise_forward"),
                        model.disable_adapter(),
                    ):
                        ref_step_output = model.forward_step(state, step_idx)
                    ref_noise_pred = ref_step_output["noise_pred"]

                if config.denoise_mode == "native":
                    with profile_range("generation.scheduler_step"):
                        next_latents = state.scheduler.step(
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
                        prev_sample=next_latents.float(),
                        return_dt=config.sde.return_kl,
                        noise_level=config.sde.noise_level,
                        sde_type=config.sde.sde_type,
                        step_index=step_idx,
                    )
                else:
                    in_sde_window = config.sde_window is None or (
                        config.sde_window[0] <= step_idx < config.sde_window[1]
                    )
                    with profile_range("generation.scheduler_step"):
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
                    next_latents = sde_result.prev_sample
                with profile_range("generation.latent_write"):
                    state.latents = next_latents

            with profile_range("generation.trajectory_buffer_write"):
                buffers.record_step(
                    step_idx,
                    observation=latents_before_step,
                    action=next_latents,
                    timestep=timestep,
                    sde_result=sde_result,
                    return_kl=config.sde.return_kl,
                    ref_noise_pred=ref_noise_pred,
                )

    denoise_peak_bytes = cuda_peak_allocated_bytes()
    peak_memory_mb = cuda_peak_allocated_mb()
    memory = None
    if occupancy is not None and denoise_peak_bytes is not None:
        memory = {
            "sample_count": int(batch_rows),
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
            "diffusion_samples_per_generation_batch": int(batch_rows),
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


__all__ = [
    "DenoiseLoopResult",
    "DenoiseTrajectoryBuffers",
    "run_denoise_loop",
]
