"""Lumina-Image-2.0 t2i diffusers-backed model.

Diffusion implementation for Alpha-VLLM Lumina-Image-2.0 (2.6B Next-DiT).
The generation helper flow mirrors every diffusion family:

    encode_prompt -> prepare_sampling -> forward_step xN -> decode_latents

Lumina2 specifics vs SD3 (the reference family):
- Single Gemma-2 text encoder returning sequence embeds + attention mask (no
  pooled). The encoder prepends a system prompt ("... <Prompt Start> ...");
  VRL always passes ``system_prompt=None``, keeping the pipeline's default
  template, so RL and diffusers-parity runs agree without extra plumbing.
- The transformer runs on REVERSED normalized time: Lumina uses t=0 as noise
  and t=1 as the image, so ``forward_step`` feeds
  ``1 - t / num_train_timesteps`` (the scheduler itself still steps on raw t).
- The transformer predicts the NEGATIVE flow velocity: diffusers negates the
  prediction right before ``scheduler.step``. We negate per branch in
  ``postprocess_branch`` — equivalent (CFG combine is linear, the norm rescale
  is magnitude-based) and it keeps the exported cond/uncond branches
  sign-consistent with what the SDE step consumes.
- TRUE classifier-free guidance run as SEPARATE branches (mirrors the
  reference pipeline), with Lumina's norm-preserving rescale reproduced by the
  shared ``cfg_normalization`` combine (the pipeline argument of that name).
- ``cfg_trunc_ratio`` (late-step CFG truncation) is NOT supported: a
  step-index-dependent CFG rule cannot be recomputed on the replay path
  (the eval forward sees one packed step, not the original index).
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from typing import Any

import torch

from vrl.generation.types import VideoGenerationRequest
from vrl.models.steps.denoise import (
    DiffusersPipelineModelBase,
    DiffusersReplayModelBase,
)
from vrl.models.steps.denoise.common import (
    DiffusionBackboneCaller,
    DiffusionBackboneInput,
    DiffusionBranch,
    EncoderAttentionMaskRunnerBase,
    MaskedPromptCollectorMixin,
    TrainTimestepMaskedPromptSamplingState,
    VaeDecodeMixin,
    expand_batch_timestep,
)
from vrl.models.steps.denoise.common.lora import LoraModelMixin


@dataclass
class Lumina2SamplingState(TrainTimestepMaskedPromptSamplingState):
    """Private Lumina2 sampling state. Engine MUST NOT introspect."""


class Lumina2Model(
    VaeDecodeMixin,
    MaskedPromptCollectorMixin,
    LoraModelMixin,
    DiffusersPipelineModelBase,
    EncoderAttentionMaskRunnerBase,
):
    """Diffusers-backed Lumina-Image-2.0 t2i model."""

    cfg_mode = "separate_cfg"
    cfg_base = "uncond"
    # Lumina's reference pipeline rescales the combined prediction back to the
    # conditional branch's norm on every CFG step.
    cfg_normalization = True
    sampling_state_cls = Lumina2SamplingState

    # -- backend ownership (called by runtime, not by collectors) -------
    _pipeline_classname = "Lumina2Pipeline"
    _frozen_encoder_names = ("text_encoder",)
    # Gemma-2-2B co-resides with the 2.6B DiT; keep it on-device.
    _prompt_encoder_on_cpu = False

    def postprocess_branch(
        self,
        request: DiffusionBackboneInput,
        branch: DiffusionBranch,
        raw_output: torch.Tensor,
    ) -> torch.Tensor:
        """Negate the raw prediction: Lumina predicts the reversed-time velocity.

        diffusers negates AFTER the CFG combine; negating per branch is
        equivalent (linear combine, magnitude-based rescale) and keeps every
        exported branch tensor in the sign the flow-match SDE step expects.
        """
        del request, branch
        return -raw_output

    # -- encode_prompt -------------------------------------------------

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode prompt via Gemma-2 (sequence embeds + attention mask, no pooled).

        VRL always passes ``system_prompt=None``, which lets the pipeline apply
        its own default template ("<system> <Prompt Start> <prompt>").
        """
        max_seq = kwargs.get("max_sequence_length", 256)
        guidance_scale = kwargs.get("guidance_scale", 4.0)
        do_cfg = guidance_scale > 1.0
        neg = negative_prompt if negative_prompt is not None else ""

        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=neg,
            num_images_per_prompt=1,
            device=self.device,
            # VRL always uses the pipeline's default system-prompt template.
            system_prompt=None,
            max_sequence_length=max_seq,
        )

        td = self.transformer.dtype
        result: dict[str, Any] = {
            "prompt_embeds": prompt_embeds.to(td),
            "prompt_attention_mask": (
                None if prompt_attention_mask is None else prompt_attention_mask.to(self.device)
            ),
        }
        if do_cfg and negative_prompt_embeds is not None:
            result["negative_prompt_embeds"] = negative_prompt_embeds.to(td)
            result["negative_prompt_attention_mask"] = (
                None
                if negative_prompt_attention_mask is None
                else negative_prompt_attention_mask.to(self.device)
            )
        return result

    # -- prepare_sampling ----------------------------------------------

    def prepare_sampling(
        self,
        request: VideoGenerationRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> Lumina2SamplingState:
        """Build the per-request SamplingState for a denoise loop."""
        pipe = self.pipeline
        device = self.device

        prompt_embeds = encoded["prompt_embeds"]
        prompt_attention_mask = encoded.get("prompt_attention_mask")
        negative_prompt_embeds = encoded.get("negative_prompt_embeds")
        negative_prompt_attention_mask = encoded.get("negative_prompt_attention_mask")

        pipe.scheduler.set_timesteps(request.num_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        seed = request.seed if request.seed is not None else random.randint(0, sys.maxsize)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        num_channels_latents = pipe.transformer.config.in_channels
        batch_size = prompt_embeds.shape[0]
        latents = pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            request.height,
            request.width,
            torch.float32,
            device,
            generator,
            None,
        )

        do_cfg = request.guidance_scale > 1.0 and negative_prompt_embeds is not None

        return Lumina2SamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=pipe.scheduler,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            guidance_scale=request.guidance_scale,
            do_cfg=do_cfg,
            num_train_timesteps=int(pipe.scheduler.config.num_train_timesteps),
        )

    # -- forward_step --------------------------------------------------

    def forward_step(
        self,
        state: Lumina2SamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        """Lumina2 transformer forward on reversed normalized time."""
        t = state.timesteps[step_idx]
        bsz = state.latents.shape[0]
        td = self._transformer_dtype()

        latent_input = state.latents.to(td)
        # Lumina uses t=0 as noise and t=1 as the image; the transformer sees
        # the reversed normalized time while the scheduler steps on raw t.
        timestep_batch = 1.0 - expand_batch_timestep(t, bsz).to(
            device=latent_input.device, dtype=td
        ) / float(state.num_train_timesteps)
        negative_embeds = (
            None if state.negative_prompt_embeds is None else state.negative_prompt_embeds.to(td)
        )
        output = DiffusionBackboneCaller(
            self.transformer,
            self,
        )(
            DiffusionBackboneInput(
                hidden_states=latent_input,
                timestep=timestep_batch,
                prompt_embeds=state.prompt_embeds.to(td),
                negative_prompt_embeds=negative_embeds,
                guidance_scale=state.guidance_scale,
                do_cfg=state.do_cfg,
                output_dtype=td,
                extra={
                    "encoder_attention_mask": state.prompt_attention_mask,
                    "negative_encoder_attention_mask": state.negative_prompt_attention_mask,
                },
            ),
        )
        return output.as_dict()


class Lumina2ReplayModel(DiffusersReplayModelBase, Lumina2Model):
    """Replay-only Lumina2 model that owns no prompt encoder, VAE, or pipeline."""


__all__ = ["Lumina2Model", "Lumina2ReplayModel", "Lumina2SamplingState"]
