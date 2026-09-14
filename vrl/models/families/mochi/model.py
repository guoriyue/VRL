"""Mochi-1 t2v diffusers-backed model.

Diffusion implementation for Genmo Mochi-1 (10B AsymmDiT).
The generation helper flow mirrors every diffusion family:

    encode_prompt -> prepare_sampling -> forward_step xN -> decode_latents

Mochi specifics vs Wan (the reference video family):
- Single T5-XXL encoder (~9.5 GB bf16, parked on CPU) returning sequence
  embeds + a BOOL attention mask; true batched CFG with uncond rows first.
- The shipped scheduler config sets ``invert_sigmas: true``: Mochi's native
  time axis is REVERSED (t=0 noise -> t=1000 image, ascending timesteps,
  positive dt) — which the shared flow-matching SDE math cannot consume
  (its noise scale is sqrt(-dt)). We therefore run the STANDARD descending
  sigma domain end to end and translate at the model boundary, exactly the
  Lumina2 treatment: the scheduler is rebuilt with ``invert_sigmas=False`` on
  Genmo's linear-quadratic sigma ladder, ``forward_step`` feeds the
  transformer ``num_train_timesteps - t`` (its native ascending clock), and
  ``postprocess_branch`` negates the prediction (a velocity in the reversed
  axis is the negative standard velocity). ``sde_step_with_logprob`` and
  ``scheduler.step`` then see a completely ordinary flow-match family.
- The reference pipeline runs CFG combine + scheduler.step in fp32; our SDE
  step already upcasts model_output/sample to fp32 (math_dtype default).
- VAE decode denormalizes with per-channel latents_mean/std (Wan-style),
  12-channel latents, 6x temporal / 8x spatial compression.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from vrl.generation.types import DenoiseRequest
from vrl.models.interfaces.runtime import ModelBuild
from vrl.models.steps.denoise import (
    DiffusersPipelineModelBase,
    DiffusersReplayModelBase,
)
from vrl.models.steps.denoise.common import (
    ChunkedLatentDecoder,
    DiffusionBackboneInput,
    DiffusionBranch,
    EncoderAttentionMaskRunnerBase,
    LatentDecodePlan,
    MaskedPromptModelMixin,
    TrainTimestepMaskedPromptSamplingState,
)
from vrl.models.steps.denoise.common.lora import LoraModelMixin


def standard_mochi_scheduler(scheduler_config: Any, num_steps: int, device: Any) -> Any:
    """Build a standard (descending-sigma) scheduler on Mochi's sigma ladder.

    One construction site for both the rollout and replay paths: Genmo's
    linear-quadratic ladder, ``invert_sigmas`` forced off so timesteps descend
    and dt is negative — the domain every shared flow-match consumer assumes.
    """
    import numpy as np
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.mochi.pipeline_mochi import linear_quadratic_schedule

    scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        dict(scheduler_config),
        invert_sigmas=False,
    )
    sigmas = np.array(linear_quadratic_schedule(num_steps, 0.025))
    scheduler.set_timesteps(sigmas=sigmas, device=device)
    return scheduler


@dataclass
class MochiSamplingState(TrainTimestepMaskedPromptSamplingState):
    """Private Mochi sampling state. Engine MUST NOT introspect."""


class MochiModel(
    MaskedPromptModelMixin,
    LoraModelMixin,
    DiffusersPipelineModelBase,
    EncoderAttentionMaskRunnerBase,
):
    """Diffusers-backed Mochi-1 t2v model."""

    cfg_mode = "batched_cfg"
    cfg_base = "uncond"
    sampling_state_cls = MochiSamplingState
    _default_max_sequence_length = 256
    _default_guidance_scale = 4.5
    _pipeline_encode_kwargs: ClassVar[Mapping[str, Any]] = {"num_videos_per_prompt": 1}

    def _sampling_scheduler(self, request: DenoiseRequest) -> Any:
        return standard_mochi_scheduler(
            self.pipeline.scheduler.config,
            request.num_steps,
            self.device,
        )

    def _latent_shape_args(self, request: DenoiseRequest) -> tuple[Any, ...]:
        return (request.frame_count,)

    def _backbone_timestep(
        self,
        timestep: torch.Tensor,
        state: MochiSamplingState,
    ) -> torch.Tensor:
        # Standard clock t descends 1000->0; Mochi's native clock ascends, so
        # the transformer sees num_train_timesteps - t.
        return float(state.num_train_timesteps) - timestep.to(self._transformer_dtype())

    # -- backend ownership (called by runtime, not by collectors) -------
    _pipeline_classname = "MochiPipeline"
    _frozen_encoder_names = ("text_encoder",)
    # T5-XXL (~9.5 GB bf16) + the 20 GB 10B transformer exceed a 32 GB card;
    # park the encoder on CPU (Qwen-Image discipline).
    _prompt_encoder_on_cpu = True

    def postprocess_branch(
        self,
        request: DiffusionBackboneInput,
        branch: DiffusionBranch,
        raw_output: torch.Tensor,
    ) -> torch.Tensor:
        """Negate: a velocity in Mochi's reversed time axis is -v_standard."""
        del request, branch
        return -raw_output

    # -- decode_latents ------------------------------------------------

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode 5D latents -> video [B, C, T, H, W] via the Mochi VAE."""
        pipe = self.pipeline
        vae = pipe.vae

        def _transform(batch: torch.Tensor) -> torch.Tensor:
            x = batch.to(vae.dtype)
            mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(x.device, x.dtype)
            std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(x.device, x.dtype)
            return x * std / vae.config.scaling_factor + mean

        decoder = ChunkedLatentDecoder(
            LatentDecodePlan(
                prepare_latents=_transform,
                vae_decode=lambda batch: vae.decode(batch, return_dict=False)[0],
                postprocess=lambda video: pipe.video_processor.postprocess_video(
                    video,
                    output_type="pt",
                ),
                output_layout="video_btchw",
                decode_batch_size=getattr(pipe, "decode_batch_size", None),
            ),
        )
        return decoder(latents)


class MochiReplayModel(DiffusersReplayModelBase, MochiModel):
    """Replay-only Mochi model that owns no prompt encoder, VAE, or pipeline."""

    def prepare_replay(self, build: ModelBuild) -> None:
        """Standardize the replay scheduler onto Mochi's descending-sigma domain.

        The generic replay loader builds the scheduler from the shipped config
        (``invert_sigmas: true`` — ascending time), but the trajectory buffers
        and the SDE log-prob math live in the standard descending domain; the
        replay scheduler must carry the same table the rollout used.
        """
        num_steps = build.num_steps
        if num_steps is not None:
            self._scheduler = standard_mochi_scheduler(
                self._scheduler.config,
                num_steps,
                build.device,
            )


__all__ = [
    "MochiModel",
    "MochiReplayModel",
    "MochiSamplingState",
    "standard_mochi_scheduler",
]
