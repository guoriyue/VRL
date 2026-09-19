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

from vrl.generation.types import DenoiseRequest
from vrl.models.steps.denoise import (
    DiffusersPipelineModelBase,
    DiffusersReplayModelBase,
    GuidedDenoiseSamplingStateBase,
)
from vrl.models.steps.denoise.common import (
    DenoiseBackboneCaller,
    DenoiseBackboneInput,
    DenoiseBackboneRunnerBase,
    DenoiseBranch,
    VaeDecodeMixin,
    expand_batch_timestep,
    pack_eval_timestep,
)


@dataclass
class Lumina2SamplingState(GuidedDenoiseSamplingStateBase):
    """Private Lumina2 sampling state. Engine MUST NOT introspect.

    ``latents`` / ``timesteps`` / ``scheduler`` are the only fields the batch
    executor may touch. ``num_train_timesteps`` is the training-clock length
    the transformer's reversed time is derived from: replay rebuilds one
    packed step without a scheduler, so the constant travels in the batch
    context instead of being read back off ``scheduler.config``.
    """

    prompt_embeds: torch.Tensor
    prompt_attention_mask: torch.Tensor | None
    negative_prompt_embeds: torch.Tensor | None
    negative_prompt_attention_mask: torch.Tensor | None
    do_cfg: bool
    num_train_timesteps: int


class Lumina2Model(
    VaeDecodeMixin,
    DiffusersPipelineModelBase,
    DenoiseBackboneRunnerBase,
):
    """Diffusers-backed Lumina-Image-2.0 t2i model."""

    cfg_mode = "separate_cfg"
    cfg_base = "uncond"
    # Lumina's reference pipeline rescales the combined prediction back to the
    # conditional branch's norm on every CFG step.
    cfg_normalization = True

    def build_branch(
        self,
        request: DenoiseBackboneInput,
        branch: str,
    ) -> DenoiseBranch:
        """Map the branch's Gemma-2 embeds and attention mask into a branch call."""
        if branch == "cond":
            embeds = request.prompt_embeds
            mask = request.extra.get("encoder_attention_mask")
        else:
            embeds = request.negative_prompt_embeds
            mask = request.extra.get("negative_encoder_attention_mask")
        return DenoiseBranch(
            hidden_states=request.hidden_states,
            timestep=request.timestep,
            encoder_hidden_states=embeds,
            extra_kwargs={"encoder_attention_mask": mask},
        )

    # -- encode_prompt -------------------------------------------------

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode the prompt via Gemma-2 to sequence embeds + padding mask (no pooled)."""
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
            negative_prompt=neg,
            do_classifier_free_guidance=do_cfg,
            num_images_per_prompt=1,
            device=self._encoder_device(),
            max_sequence_length=max_seq,
            # VRL always uses the pipeline's default system-prompt template
            # ("<system> <Prompt Start> <prompt>").
            system_prompt=None,
        )

        td = self.transformer.dtype
        result: dict[str, Any] = {
            "prompt_embeds": prompt_embeds.to(self.device, dtype=td),
            "prompt_attention_mask": (
                None if prompt_attention_mask is None else prompt_attention_mask.to(self.device)
            ),
        }
        if do_cfg and negative_prompt_embeds is not None:
            result["negative_prompt_embeds"] = negative_prompt_embeds.to(self.device, dtype=td)
            result["negative_prompt_attention_mask"] = (
                None
                if negative_prompt_attention_mask is None
                else negative_prompt_attention_mask.to(self.device)
            )
        return result

    # -- prepare_sampling ----------------------------------------------

    def prepare_sampling(
        self,
        request: DenoiseRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> Lumina2SamplingState:
        """Build the per-request SamplingState for a denoise loop."""
        del kwargs
        pipe = self.pipeline
        device = self.device

        prompt_embeds = encoded["prompt_embeds"]
        negative_prompt_embeds = encoded.get("negative_prompt_embeds")

        scheduler = pipe.scheduler
        scheduler.set_timesteps(request.num_steps, device=device)

        seed = request.seed if request.seed is not None else random.randint(0, sys.maxsize)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        latents = pipe.prepare_latents(
            prompt_embeds.shape[0],
            pipe.transformer.config.in_channels,
            request.height,
            request.width,
            torch.float32,
            device,
            generator,
            None,
        )
        # No-op for FlowMatchEuler (init_noise_sigma == 1.0); kept for pipeline parity.
        latents = latents * scheduler.init_noise_sigma

        return Lumina2SamplingState(
            latents=latents,
            timesteps=scheduler.timesteps,
            scheduler=scheduler,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=encoded.get("prompt_attention_mask"),
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=encoded.get("negative_prompt_attention_mask"),
            guidance_scale=request.guidance_scale,
            do_cfg=request.guidance_scale > 1.0 and negative_prompt_embeds is not None,
            num_train_timesteps=int(scheduler.config.num_train_timesteps),
        )

    # -- forward_step --------------------------------------------------

    def forward_step(
        self,
        state: Lumina2SamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        """Transformer forward + separate-branch CFG."""
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
        output = DenoiseBackboneCaller(
            self.transformer,
            self,
        )(
            DenoiseBackboneInput(
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

    # -- trajectory boundary -------------------------------------------
    # Masks and negative embeds are exported only when present, so the no-CFG
    # path stays tensor-free and restore reads them back with ``.get``.

    def export_batch_context(self, state: Lumina2SamplingState) -> dict[str, Any]:
        """Project sampling state into shared trajectory context."""
        return {
            "guidance_scale": state.guidance_scale,
            "cfg": state.do_cfg,
            "num_train_timesteps": state.num_train_timesteps,
        }

    def export_replay_tensors(self, state: Lumina2SamplingState) -> dict[str, Any]:
        """Project sampling state into per-sample trajectory tensors."""
        tensors: dict[str, Any] = {"prompt_embeds": state.prompt_embeds}
        for name in (
            "prompt_attention_mask",
            "negative_prompt_embeds",
            "negative_prompt_attention_mask",
        ):
            value = getattr(state, name)
            if value is not None:
                tensors[name] = value
        return tensors

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> Lumina2SamplingState:
        """Rebuild the sampling state from a batch slice for eval forward."""
        negative_prompt_embeds = replay_tensors.get("negative_prompt_embeds")
        return Lumina2SamplingState(
            latents=latents,
            timesteps=pack_eval_timestep(replay_tensors["timesteps"], step_idx),
            # forward_step never calls scheduler.step, so replay needs none.
            scheduler=None,
            prompt_embeds=replay_tensors["prompt_embeds"],
            prompt_attention_mask=replay_tensors.get("prompt_attention_mask"),
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=replay_tensors.get("negative_prompt_attention_mask"),
            guidance_scale=batch_context["guidance_scale"],
            do_cfg=batch_context["cfg"] and negative_prompt_embeds is not None,
            num_train_timesteps=int(batch_context["num_train_timesteps"]),
        )

    def postprocess_branch(
        self,
        request: DenoiseBackboneInput,
        branch: DenoiseBranch,
        raw_output: torch.Tensor,
    ) -> torch.Tensor:
        """Negate the raw prediction: Lumina predicts the reversed-time velocity.

        diffusers negates AFTER the CFG combine; negating per branch is
        equivalent (linear combine, magnitude-based rescale) and keeps every
        exported branch tensor in the sign the flow-match SDE step expects.
        """
        del request, branch
        return -raw_output


class Lumina2ReplayModel(DiffusersReplayModelBase, Lumina2Model):
    """Replay-only Lumina2 model that owns no prompt encoder, VAE, or pipeline."""


__all__ = ["Lumina2Model", "Lumina2ReplayModel", "Lumina2SamplingState"]
