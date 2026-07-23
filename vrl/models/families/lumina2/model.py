"""Lumina-Image-2.0 t2i diffusers-backed model.

Diffusion implementation for Alpha-VLLM Lumina-Image-2.0 (2.6B Next-DiT).
The generation helper flow mirrors every diffusion family:

    encode_prompt -> prepare_sampling -> forward_step xN -> decode_latents

Lumina2 specifics vs SD3 (the reference family):
- Single Gemma-2 text encoder returning sequence embeds + attention mask (no
  pooled). The encoder prepends a system prompt ("... <Prompt Start> ...");
  passing ``system_prompt=None`` keeps the pipeline's default template, so RL
  and diffusers-parity runs agree without extra plumbing.
- The transformer runs on REVERSED normalized time: Lumina uses t=0 as noise
  and t=1 as the image, so ``forward_step`` feeds
  ``1 - t / num_train_timesteps`` (the scheduler itself still steps on raw t).
- The transformer predicts the NEGATIVE flow velocity: diffusers negates the
  prediction right before ``scheduler.step``. We negate per branch in
  ``postprocess_branch`` — equivalent (CFG combine is linear, the norm rescale
  is magnitude-based) and it keeps the exported cond/uncond branches
  sign-consistent with what the SDE step consumes.
- TRUE classifier-free guidance run as SEPARATE branches (mirrors the
  reference pipeline), with Lumina's norm-preserving rescale
  (``cfg_normalization``) reproduced in ``finalize_noise_pred``.
- ``cfg_trunc_ratio`` (late-step CFG truncation) is NOT supported: a
  step-index-dependent CFG rule cannot be recomputed on the replay path
  (the eval forward sees one packed step, not the original index), so
  ``prepare_sampling`` fails loud on non-default values.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from typing import Any

import torch

from vrl.generation.types import VideoGenerationRequest
from vrl.models.interfaces.runtime import ModelBuild
from vrl.models.steps.denoise import (
    DiffusersPipelineModelBase,
    DiffusersReplayModelBase,
    GuidedDiffusionSamplingStateBase,
    diffusers_pipeline_dtypes,
)
from vrl.models.steps.denoise.common import (
    ChunkedLatentDecoder,
    DiffusionBackboneCaller,
    DiffusionBackboneInput,
    DiffusionBackboneRunnerBase,
    DiffusionBranch,
    LatentDecodePlan,
    expand_batch_timestep,
    pack_eval_timestep,
)
from vrl.models.steps.denoise.common.lora import LoraModelMixin
from vrl.models.steps.denoise.common.tensors import require_tensor


@dataclass
class Lumina2SamplingState(GuidedDiffusionSamplingStateBase):
    """Private Lumina2 sampling state. Engine MUST NOT introspect."""

    prompt_embeds: torch.Tensor
    prompt_attention_mask: torch.Tensor | None
    negative_prompt_embeds: torch.Tensor | None
    negative_prompt_attention_mask: torch.Tensor | None
    do_cfg: bool
    num_train_timesteps: int


class Lumina2Model(LoraModelMixin, DiffusersPipelineModelBase, DiffusionBackboneRunnerBase):
    """Diffusers-backed Lumina-Image-2.0 t2i model."""

    cfg_mode = "separate_cfg"
    cfg_base = "uncond"

    def build_branch(
        self,
        request: DiffusionBackboneInput,
        branch: str,
    ) -> DiffusionBranch:
        """Map Lumina2 transformer kwargs into the shared backbone contract."""
        if branch == "cond":
            embeds = request.prompt_embeds
            mask = request.extra.get("encoder_attention_mask")
        else:
            embeds = require_tensor(
                request.negative_prompt_embeds,
                "negative_prompt_embeds",
            )
            mask = request.extra.get("negative_encoder_attention_mask")
        return DiffusionBranch(
            hidden_states=request.hidden_states,
            timestep=request.timestep,
            encoder_hidden_states=embeds,
            extra_kwargs={"encoder_attention_mask": mask},
        )

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

    def finalize_noise_pred(
        self,
        request: DiffusionBackboneInput,
        combined: torch.Tensor,
        cond: torch.Tensor,
        uncond: torch.Tensor,
    ) -> torch.Tensor:
        """Lumina2 norm-preserving CFG: rescale the combined pred to the cond norm."""
        del uncond
        if not request.do_cfg:
            return combined
        cond_norm = torch.norm(cond, dim=-1, keepdim=True)
        comb_norm = torch.norm(combined, dim=-1, keepdim=True)
        return combined * (cond_norm / comb_norm)

    # -- backend ownership (called by runtime, not by collectors) -------

    @classmethod
    def from_build(cls, build: ModelBuild) -> Lumina2Model:
        """Load the diffusers Lumina2 pipeline + freeze non-trainable modules."""
        from diffusers import Lumina2Pipeline

        model_dtype = build.parameter_dtype
        prompt_encoder_dtype, load_kwargs = diffusers_pipeline_dtypes(build, model_dtype)
        pipeline = Lumina2Pipeline.from_pretrained(
            build.model_name_or_path,
            **load_kwargs,
        )
        pipeline.vae.requires_grad_(False)
        text_encoder = getattr(pipeline, "text_encoder", None)
        if text_encoder is not None:
            # Gemma-2-2B co-resides with the 2.6B DiT; keep it on-device.
            text_encoder.requires_grad_(False)
            text_encoder.to(build.device, dtype=prompt_encoder_dtype)
        pipeline.vae.to(build.device, dtype=torch.float32)
        return cls(
            pipeline=pipeline,
            device=build.device,
        )

    # -- encode_prompt -------------------------------------------------

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode prompt via Gemma-2 (sequence embeds + attention mask, no pooled).

        ``system_prompt`` defaults to None, which lets the pipeline apply its
        own default template ("<system> <Prompt Start> <prompt>").
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
            system_prompt=kwargs.get("system_prompt"),
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
        cfg_trunc_ratio = float(kwargs.get("cfg_trunc_ratio", 1.0))
        if cfg_trunc_ratio != 1.0:
            raise ValueError(
                "Lumina2 cfg_trunc_ratio is not supported in RL rollouts: a "
                "step-dependent CFG rule cannot be recomputed on the replay "
                "path. Leave it at 1.0 (CFG on every step).",
            )
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

    # -- collector boundary --------------------------------------------

    def export_batch_context(self, state: Lumina2SamplingState) -> dict[str, Any]:
        """Project Lumina2 sampling state into trajectory context."""
        return {
            "guidance_scale": state.guidance_scale,
            "cfg": state.do_cfg,
            "num_train_timesteps": state.num_train_timesteps,
        }

    def export_replay_tensors(self, state: Lumina2SamplingState) -> dict[str, Any]:
        """Project Lumina2 sampling state into per-sample trajectory tensors."""
        tensors: dict[str, Any] = {"prompt_embeds": state.prompt_embeds}
        if state.prompt_attention_mask is not None:
            tensors["prompt_attention_mask"] = state.prompt_attention_mask
        if state.negative_prompt_embeds is not None:
            tensors["negative_prompt_embeds"] = state.negative_prompt_embeds
        if state.negative_prompt_attention_mask is not None:
            tensors["negative_prompt_attention_mask"] = state.negative_prompt_attention_mask
        return tensors

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> Lumina2SamplingState:
        """Rebuild Lumina2SamplingState from a batch slice for the eval forward path."""
        ts = replay_tensors["timesteps"]
        timesteps = pack_eval_timestep(ts, step_idx)
        return Lumina2SamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=None,  # not needed for forward_step (no scheduler.step here)
            prompt_embeds=replay_tensors["prompt_embeds"],
            prompt_attention_mask=replay_tensors.get("prompt_attention_mask"),
            negative_prompt_embeds=replay_tensors.get("negative_prompt_embeds"),
            negative_prompt_attention_mask=replay_tensors.get(
                "negative_prompt_attention_mask",
            ),
            guidance_scale=batch_context["guidance_scale"],
            do_cfg=batch_context["cfg"]
            and replay_tensors.get("negative_prompt_embeds") is not None,
            num_train_timesteps=int(batch_context["num_train_timesteps"]),
        )

    # -- decode_latents ------------------------------------------------

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents -> image via the KL VAE (scaling + shift, 4D)."""
        pipe = self.pipeline
        vae = pipe.vae
        scaling_factor = vae.config.scaling_factor
        shift_factor = getattr(vae.config, "shift_factor", 0.0) or 0.0
        decoder = ChunkedLatentDecoder(
            LatentDecodePlan(
                prepare_latents=lambda chunk: chunk.to(vae.dtype) / scaling_factor + shift_factor,
                vae_decode=lambda chunk: vae.decode(chunk, return_dict=False)[0],
                postprocess=lambda image: pipe.image_processor.postprocess(
                    image,
                    output_type="pt",
                ),
                output_layout="image_bchw",
                decode_batch_size=getattr(pipe, "decode_batch_size", None),
            ),
        )
        return decoder(latents)


class Lumina2ReplayModel(DiffusersReplayModelBase, Lumina2Model):
    """Replay-only Lumina2 model that owns no prompt encoder, VAE, or pipeline."""


__all__ = ["Lumina2Model", "Lumina2ReplayModel", "Lumina2SamplingState"]
