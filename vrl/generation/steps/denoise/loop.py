"""Continuous denoise loop independent of output generation regime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from vrl.generation.execution.types import BatchMemoryReading
from vrl.generation.steps.denoise.config import DenoiseLoopConfig
from vrl.generation.steps.denoise.teacache import TeaCacheState
from vrl.math.denoise.flow_matching import SDEStepResult, sde_step_with_logprob
from vrl.trajectory.storage import trajectory_tensor_bytes
from vrl.utils.cuda_memory import (
    cuda_peak_allocated_bytes,
    reset_cuda_peak,
)
from vrl.utils.profiling import profile_range


@dataclass(slots=True)
class DenoiseLoopResult:
    """Denoise-loop output before decode and artifact packing."""

    state: Any
    # The whole denoise path, ``(batch, num_steps + 1, *latent)``; see
    # ``DenoiseTrajectoryBuffers.latents``. Carried as one storage so the
    # observation/action pair is never materialized twice downstream.
    latents: Any
    log_probs: Any
    timesteps: Any
    kl: Any
    prev_sample_means: Any | None = None
    # Batch-memory reading (occupancy at loop start plus the denoise peak),
    # None off CUDA. The executor adds the decode peak and derives the
    # runtime-debug peak display from it.
    memory: dict[str, int] | None = None
    # display/provenance-only: denoise-engine counters forwarded to optional
    # runtime-debug telemetry.
    engine_counters: dict[str, Any] = field(default_factory=dict)

    @property
    def observations(self) -> Any:
        return self.latents[:, :-1]

    @property
    def actions(self) -> Any:
        return self.latents[:, 1:]


@dataclass(slots=True)
class DenoiseTrajectoryBuffers:
    """Preallocated replay tensors written once per denoise step.

    ``latents`` holds the whole denoise path, ``(batch, num_steps + 1, *latent)``:
    row ``t`` is the sample the step-``t`` forward consumed and row ``t + 1`` the
    sample the step produced. ``observations`` and ``actions`` are the two
    step-aligned views ``latents[:, :-1]`` and ``latents[:, 1:]``. The
    observation of step ``t + 1`` is the action of step ``t`` by construction
    (the loop assigns, never mutates, ``state.latents``), so keeping two buffers
    would hold every intermediate latent twice on the device and copy it twice
    per step.
    """

    latents: torch.Tensor
    log_probs: torch.Tensor
    timesteps: torch.Tensor
    kl: torch.Tensor
    prev_sample_means: torch.Tensor | None = None

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
        if batch_rows != config.sample_count:
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
            latents=torch.empty(
                (batch_rows, num_steps + 1, *latent_shape),
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
        )

    @property
    def observations(self) -> torch.Tensor:
        return self.latents[:, :-1]

    @property
    def actions(self) -> torch.Tensor:
        return self.latents[:, 1:]

    def record_initial_latents(self, latents: torch.Tensor) -> None:
        """Write the observation of step 0; every later observation is a recorded action."""
        self.latents[:, 0].copy_(latents.detach())

    def record_step(
        self,
        step_idx: int,
        *,
        action: torch.Tensor,
        timestep: torch.Tensor,
        sde_result: SDEStepResult,
        return_kl: bool,
    ) -> None:
        """Write detached values; copy_ casts into each buffer's allocated dtype."""
        self.latents[:, step_idx + 1].copy_(action.detach())
        self.log_probs[:, step_idx].copy_(
            sde_result.log_prob.detach(),
        )
        self.timesteps[:, step_idx].copy_(timestep.detach())
        if return_kl:
            self.kl[:, step_idx].copy_(
                sde_result.log_prob.detach().abs(),
            )
        else:
            self.kl[:, step_idx].zero_()
        if self.prev_sample_means is not None:
            self.prev_sample_means[:, step_idx].copy_(
                sde_result.prev_sample_mean.detach(),
            )


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
    occupancy = BatchMemoryReading.cuda_occupancy_snapshot()
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
        with profile_range("generation.latent_snapshot"):
            buffers.record_initial_latents(state.latents)
        for step_idx in range(num_steps_to_run):
            with profile_range("generation.denoise_step"):
                timestep = state.timesteps[step_idx]

                if teacache is not None and not teacache.should_run(
                    state.latents,
                    step_idx,
                ):
                    noise_pred = teacache.cached_noise_pred
                else:
                    with profile_range("generation.denoise_forward"):
                        step_output = model.forward_step(state, step_idx)
                    noise_pred = step_output["noise_pred"]
                    if teacache is not None:
                        teacache.cache_noise_pred(noise_pred)

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
                    action=next_latents,
                    timestep=timestep,
                    sde_result=sde_result,
                    return_kl=config.sde.return_kl,
                )

    denoise_peak_bytes = cuda_peak_allocated_bytes()
    memory = None
    if occupancy is not None and denoise_peak_bytes is not None:
        memory = {
            "sample_count": int(batch_rows),
            **occupancy,
            "denoise_peak_bytes": denoise_peak_bytes,
        }

    return DenoiseLoopResult(
        state=state,
        latents=buffers.latents,
        log_probs=buffers.log_probs,
        timesteps=buffers.timesteps,
        kl=buffers.kl,
        prev_sample_means=buffers.prev_sample_means,
        memory=memory,
        engine_counters={
            "diffusion_num_denoise_steps": num_steps_to_run,
            "diffusion_samples_per_generation_batch": int(batch_rows),
            "diffusion_observation_bytes": trajectory_tensor_bytes(buffers.observations),
            "diffusion_action_bytes": trajectory_tensor_bytes(buffers.actions),
            "diffusion_old_logprob_bytes": trajectory_tensor_bytes(buffers.log_probs),
            "diffusion_timestep_bytes": trajectory_tensor_bytes(buffers.timesteps),
            "diffusion_kl_bytes": trajectory_tensor_bytes(buffers.kl),
            "diffusion_denoise_mode": config.denoise_mode,
            **(teacache.counters() if teacache is not None else {}),
        },
    )


__all__ = [
    "DenoiseLoopResult",
    "DenoiseTrajectoryBuffers",
    "run_denoise_loop",
]
