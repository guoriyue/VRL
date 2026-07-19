"""Exact temporal-chunk policy loop for the released CausVid checkpoint.

The released sampler performs three causal transformer forwards per temporal
chunk.  Only the first two forwards parameterize stochastic actions: each
predicts ``x0`` and re-noises it at the next schedule sigma.  The third forward
produces the terminal clean chunk, followed by a separate timestep-zero forward
whose output is discarded and whose only purpose is to commit that clean chunk
to the K/V cache.

Keeping that ordering here, behind a small backend protocol, makes it testable
without importing upstream CausVid (whose causal-model import eagerly compiles
FlexAttention).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

import torch

from vrl.math.denoise.renoise import renoise_step_with_logprob


@dataclass(frozen=True, slots=True)
class CausVidGeometry:
    """Latent and decoded-video geometry owned by one CausVid release."""

    latent_frames: int
    latent_channels: int
    latent_height: int
    latent_width: int
    frames_per_chunk: int
    pixel_frames: int
    pixel_height: int
    pixel_width: int
    fps: int = 16

    def __post_init__(self) -> None:
        values = (
            self.latent_frames,
            self.latent_channels,
            self.latent_height,
            self.latent_width,
            self.frames_per_chunk,
            self.pixel_frames,
            self.pixel_height,
            self.pixel_width,
            self.fps,
        )
        if any(value < 1 for value in values):
            raise ValueError("CausVid geometry dimensions must all be positive")
        if self.latent_frames % self.frames_per_chunk:
            raise ValueError(
                "latent_frames must be divisible by frames_per_chunk: "
                f"{self.latent_frames} % {self.frames_per_chunk} != 0",
            )

    @property
    def temporal_chunk_count(self) -> int:
        return self.latent_frames // self.frames_per_chunk

    @property
    def latent_shape(self) -> tuple[int, int, int, int]:
        return (
            self.latent_frames,
            self.latent_channels,
            self.latent_height,
            self.latent_width,
        )


@dataclass(frozen=True, slots=True)
class CausVidSchedule:
    """Released three-forward/two-transition denoise recipe."""

    prediction_timesteps: tuple[int, int, int]
    transition_sigmas: tuple[float, float]
    cache_timestep: int = 0

    def __post_init__(self) -> None:
        if len(self.prediction_timesteps) != 3:
            raise ValueError("CausVid requires exactly three prediction timesteps")
        if len(self.transition_sigmas) != 2:
            raise ValueError("CausVid requires exactly two Gaussian transitions")
        if any(sigma <= 0 or sigma > 1 for sigma in self.transition_sigmas):
            raise ValueError("CausVid transition sigmas must be in (0, 1]")
        if self.cache_timestep != 0:
            raise ValueError("CausVid cache finalization must run at timestep zero")

    @property
    def transition_count(self) -> int:
        return len(self.transition_sigmas)


# These values are an architecture/checkpoint boundary, not workflow taxonomy.
# FlowMatchScheduler(shift=8, sigma_min=0, extra_one_step=True) maps the released
# integer timestep requests 757 and 522 to these exact nearest table entries.
OFFICIAL_CAUSVID_GEOMETRY = CausVidGeometry(
    latent_frames=21,
    latent_channels=16,
    latent_height=60,
    latent_width=104,
    frames_per_chunk=3,
    pixel_frames=81,
    pixel_height=480,
    pixel_width=832,
    fps=16,
)
OFFICIAL_CAUSVID_SCHEDULE = CausVidSchedule(
    prediction_timesteps=(1000, 757, 522),
    transition_sigmas=(0.7567567229270935, 0.52173912525177),
    cache_timestep=0,
)


class CausVidCausalBackend(Protocol):
    """Family backend calls needed by rollout and differentiable replay."""

    @property
    def device(self) -> Any: ...

    @property
    def dtype(self) -> torch.dtype: ...

    def new_causal_cache(
        self,
        *,
        batch_size: int,
        geometry: CausVidGeometry,
    ) -> Any:
        """Allocate a fresh request-local self/cross-attention cache."""
        ...

    def predict_x0_cached(
        self,
        noisy_chunk: torch.Tensor,
        *,
        prompt_embeds: torch.Tensor,
        timestep: int,
        cache: Any,
        chunk_index: int,
        finalize_cache: bool,
        geometry: CausVidGeometry,
    ) -> torch.Tensor:
        """Predict one chunk while reading/writing the request-local cache."""
        ...

    def predict_x0_full_prefix(
        self,
        prefix_and_chunk: torch.Tensor,
        *,
        prompt_embeds: torch.Tensor,
        per_frame_timesteps: torch.Tensor,
        geometry: CausVidGeometry,
    ) -> torch.Tensor:
        """Predict a full block-causal prefix with ``kv_cache=None``."""
        ...


class CausVidTrajectoryMapping(TypedDict):
    """Typed projection into ``ChunkAutoregressiveDenoiseResult`` fields."""

    temporal_chunk_count: int
    denoise_transition_count: int
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_prob: torch.Tensor
    mask: torch.Tensor
    timesteps: torch.Tensor
    finalized_chunk_latents: torch.Tensor
    replay_tensors: dict[str, Any]
    context: dict[str, Any]


@dataclass(slots=True)
class CausVidRunResult:
    """One sample batch produced by the released causal loop."""

    latents: torch.Tensor
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_prob: torch.Tensor
    timesteps: torch.Tensor
    next_sigmas: torch.Tensor
    finalized_chunk_latents: torch.Tensor
    prompt_embeds: torch.Tensor
    geometry: CausVidGeometry
    schedule: CausVidSchedule

    def trajectory_mapping(self, *, context: dict[str, Any]) -> CausVidTrajectoryMapping:
        """Map family-owned facts to the shared chunk-denoise wire contract."""

        return {
            "temporal_chunk_count": self.geometry.temporal_chunk_count,
            "denoise_transition_count": self.schedule.transition_count,
            "observations": self.observations,
            "actions": self.actions,
            "old_log_prob": self.old_log_prob,
            "mask": torch.ones_like(self.old_log_prob),
            "timesteps": self.timesteps,
            "finalized_chunk_latents": self.finalized_chunk_latents,
            "replay_tensors": {
                "prompt_embeds": self.prompt_embeds,
                "next_sigmas": self.next_sigmas,
            },
            "context": dict(context),
        }


class CausVidCausalRunner:
    """Run the exact released causal schedule with request-local caches."""

    def __init__(
        self,
        backend: CausVidCausalBackend,
        *,
        geometry: CausVidGeometry = OFFICIAL_CAUSVID_GEOMETRY,
        schedule: CausVidSchedule = OFFICIAL_CAUSVID_SCHEDULE,
        math_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.backend = backend
        self.geometry = geometry
        self.schedule = schedule
        self.math_dtype = math_dtype

    def run(
        self,
        initial_noise: torch.Tensor,
        *,
        prompt_embeds: torch.Tensor,
        generators: tuple[torch.Generator, ...] | None = None,
    ) -> CausVidRunResult:
        """Generate all temporal chunks and record the two policy actions each."""

        self._validate_inputs(initial_noise, prompt_embeds, generators)
        batch_size = int(initial_noise.shape[0])
        cache = self.backend.new_causal_cache(
            batch_size=batch_size,
            geometry=self.geometry,
        )

        chunk_observations: list[torch.Tensor] = []
        chunk_actions: list[torch.Tensor] = []
        chunk_log_probs: list[torch.Tensor] = []
        finalized_chunks: list[torch.Tensor] = []
        frames_per_chunk = self.geometry.frames_per_chunk

        # Rollout must not retain the transformer's large causal activation graph.
        with torch.no_grad():
            for chunk_index in range(self.geometry.temporal_chunk_count):
                start = chunk_index * frames_per_chunk
                end = start + frames_per_chunk
                noisy_chunk = initial_noise[:, start:end]
                observations: list[torch.Tensor] = []
                actions: list[torch.Tensor] = []
                log_probs: list[torch.Tensor] = []

                for transition_index, next_sigma in enumerate(
                    self.schedule.transition_sigmas,
                ):
                    observations.append(noisy_chunk)
                    pred_x0 = self.backend.predict_x0_cached(
                        noisy_chunk,
                        prompt_embeds=prompt_embeds,
                        timestep=self.schedule.prediction_timesteps[transition_index],
                        cache=cache,
                        chunk_index=chunk_index,
                        finalize_cache=False,
                        geometry=self.geometry,
                    )
                    transition_noise = self._transition_noise(
                        pred_x0,
                        generators=generators,
                    )
                    transition = renoise_step_with_logprob(
                        pred_x0,
                        next_sigma,
                        noise=transition_noise,
                        math_dtype=self.math_dtype,
                    )
                    noisy_chunk = transition.next_sample
                    actions.append(noisy_chunk)
                    log_probs.append(transition.log_prob)

                terminal_chunk = self.backend.predict_x0_cached(
                    noisy_chunk,
                    prompt_embeds=prompt_embeds,
                    timestep=self.schedule.prediction_timesteps[-1],
                    cache=cache,
                    chunk_index=chunk_index,
                    finalize_cache=False,
                    geometry=self.geometry,
                )
                # The fourth call is not an RL action. Its output is discarded;
                # it deterministically overwrites this temporal cache slot with
                # the clean terminal chunk at timestep zero.
                self.backend.predict_x0_cached(
                    terminal_chunk,
                    prompt_embeds=prompt_embeds,
                    timestep=self.schedule.cache_timestep,
                    cache=cache,
                    chunk_index=chunk_index,
                    finalize_cache=True,
                    geometry=self.geometry,
                )

                chunk_observations.append(torch.stack(observations, dim=1))
                chunk_actions.append(torch.stack(actions, dim=1))
                chunk_log_probs.append(torch.stack(log_probs, dim=1))
                finalized_chunks.append(terminal_chunk)

        observations_tensor = torch.stack(chunk_observations, dim=1)
        actions_tensor = torch.stack(chunk_actions, dim=1)
        log_probs_tensor = torch.stack(chunk_log_probs, dim=1)
        finalized_tensor = torch.stack(finalized_chunks, dim=1)
        timesteps = (
            torch.tensor(
                self.schedule.prediction_timesteps[: self.schedule.transition_count],
                device=initial_noise.device,
                dtype=torch.long,
            )
            .view(1, 1, -1)
            .expand(batch_size, self.geometry.temporal_chunk_count, -1)
            .clone()
        )
        next_sigmas = (
            torch.tensor(
                self.schedule.transition_sigmas,
                device=initial_noise.device,
                dtype=self.math_dtype,
            )
            .view(1, 1, -1)
            .expand(batch_size, self.geometry.temporal_chunk_count, -1)
            .clone()
        )
        return CausVidRunResult(
            latents=torch.cat(finalized_chunks, dim=1),
            observations=observations_tensor,
            actions=actions_tensor,
            old_log_prob=log_probs_tensor,
            timesteps=timesteps,
            next_sigmas=next_sigmas,
            finalized_chunk_latents=finalized_tensor,
            prompt_embeds=prompt_embeds,
            geometry=self.geometry,
            schedule=self.schedule,
        )

    def _validate_inputs(
        self,
        initial_noise: torch.Tensor,
        prompt_embeds: torch.Tensor,
        generators: tuple[torch.Generator, ...] | None,
    ) -> None:
        if not isinstance(initial_noise, torch.Tensor) or initial_noise.ndim != 5:
            raise ValueError("initial_noise must be a [B,F,C,H,W] tensor")
        expected = self.geometry.latent_shape
        actual = tuple(int(length) for length in initial_noise.shape[1:])
        if actual != expected:
            raise ValueError(f"CausVid latent shape must be {expected}, got {actual}")
        if not isinstance(prompt_embeds, torch.Tensor) or prompt_embeds.ndim < 2:
            raise ValueError("prompt_embeds must be a sample-aligned tensor")
        if int(prompt_embeds.shape[0]) != int(initial_noise.shape[0]):
            raise ValueError("prompt_embeds batch must match initial_noise batch")
        if generators is not None and len(generators) != int(initial_noise.shape[0]):
            raise ValueError("generators must contain one torch.Generator per sample")

    @staticmethod
    def _transition_noise(
        pred_x0: torch.Tensor,
        *,
        generators: tuple[torch.Generator, ...] | None,
    ) -> torch.Tensor | None:
        if generators is None:
            return None
        rows = [
            torch.randn(
                pred_x0[index].shape,
                generator=generator,
                device=pred_x0.device,
                # Upstream uses torch.randn_like(denoised_pred): RNG is drawn
                # directly in checkpoint dtype before fp32 transition math.
                dtype=pred_x0.dtype,
            )
            for index, generator in enumerate(generators)
        ]
        return torch.stack(rows, dim=0)


__all__ = [
    "OFFICIAL_CAUSVID_GEOMETRY",
    "OFFICIAL_CAUSVID_SCHEDULE",
    "CausVidCausalBackend",
    "CausVidCausalRunner",
    "CausVidGeometry",
    "CausVidRunResult",
    "CausVidSchedule",
    "CausVidTrajectoryMapping",
]
