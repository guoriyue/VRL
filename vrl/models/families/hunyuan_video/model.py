"""HunyuanVideo t2v diffusers-backed model.

Diffusion implementation for Tencent HunyuanVideo (13B dual/single-stream DiT).
The generation helper flow mirrors every diffusion family:

    encode_prompt -> prepare_sampling -> forward_step xN -> decode_latents

HunyuanVideo specifics vs Wan (the reference video family):
- GUIDANCE-DISTILLED, like FLUX: a single transformer forward conditioned on a
  ``guidance`` embedding (``guidance_scale * 1000``), no unconditional branch
  (``cfg_mode = "single_branch"``, ``do_cfg=False`` always).
- Two text encoders: LLaVA-LLaMA-3-8B produces sequence embeds + attention
  mask through a chat template the pipeline crops back out (crop_start=95),
  CLIP-L produces the pooled embed. The 15 GB LLaMA is parked on CPU
  (Qwen-Image discipline); CLIP stays on-device.
- 5D latents ``[B, 16, (F-1)//4+1, H//8, W//8]`` (4x temporal, 8x spatial
  compression), no packing; the transformer computes RoPE internally.
- The reference pipeline passes an EXPLICIT sigma ladder
  ``np.linspace(1.0, 0.0, steps+1)[:-1]`` into set_timesteps (shift=7.0 is
  applied by the scheduler); ``prepare_sampling`` mirrors that exactly instead
  of relying on the scheduler's default ladder.
- VAE decode is a plain ``latents / scaling_factor`` (no shift, no mean/std),
  video output ``[B, C, T, H, W]``.
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
    GuidedDiffusionSamplingStateBase,
)
from vrl.models.steps.denoise.common import (
    DiffusionBackboneCaller,
    DiffusionBackboneInput,
    DiffusionBackboneRunnerBase,
    DiffusionBranch,
    VaeDecodeMixin,
    expand_batch_timestep,
    pack_eval_timestep,
)
from vrl.models.steps.denoise.common.lora import LoraModelMixin


@dataclass
class HunyuanVideoSamplingState(GuidedDiffusionSamplingStateBase):
    """Private HunyuanVideo sampling state. Engine MUST NOT introspect."""

    prompt_embeds: torch.Tensor
    prompt_attention_mask: torch.Tensor | None
    pooled_prompt_embeds: torch.Tensor


class HunyuanVideoModel(
    VaeDecodeMixin,
    LoraModelMixin,
    DiffusersPipelineModelBase,
    DiffusionBackboneRunnerBase,
):
    """Diffusers-backed HunyuanVideo t2v model (guidance-distilled)."""

    cfg_mode = "single_branch"
    cfg_base = "cond"
    # The causal video VAE decodes 5D latents; the shared decoder permutes the
    # frame axis back to [B, C, T, H, W].
    _decode_output_layout = "video_btchw"

    # -- backend ownership (called by runtime, not by collectors) -------
    _pipeline_classname = "HunyuanVideoPipeline"
    _frozen_encoder_names = ("text_encoder", "text_encoder_2")
    # The LLaVA-LLaMA-3-8B encoder is ~15 GB bf16; with the 25 GB 13B
    # transformer it cannot co-reside on a 32 GB card. It feeds only the
    # one-shot encode_prompt, so park it on CPU (Qwen-Image discipline). The
    # pipeline's encode_prompt drives BOTH encoders on one device, so the tiny
    # CLIP-L pooled encoder parks with it rather than mixing devices in one call.
    _prompt_encoder_on_cpu = True

    def build_branch(
        self,
        request: DiffusionBackboneInput,
        branch: str,
    ) -> DiffusionBranch:
        """Map HunyuanVideo transformer kwargs into the shared backbone contract."""
        if branch != "cond":
            raise ValueError("HunyuanVideo is guidance-distilled and has no uncond branch")
        return DiffusionBranch(
            hidden_states=request.hidden_states,
            timestep=request.timestep,
            encoder_hidden_states=request.prompt_embeds,
            extra_kwargs={
                "encoder_attention_mask": request.extra["encoder_attention_mask"],
                "pooled_projections": request.extra["pooled_projections"],
                "guidance": request.extra["guidance"],
            },
        )

    # -- encode_prompt -------------------------------------------------

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode prompt via LLaMA (sequence + mask) and CLIP-L (pooled).

        HunyuanVideo is guidance-distilled: there is no unconditional branch,
        so negative prompts are unsupported (the guidance scalar carries the
        strength).
        """
        self._reject_unsupported_negative_prompt(negative_prompt)
        max_seq = kwargs.get("max_sequence_length", 256)
        pipe = self.pipeline
        enc_device = self._encoder_device()
        td = self.transformer.dtype

        prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = pipe.encode_prompt(
            prompt=prompt,
            num_videos_per_prompt=1,
            device=enc_device,
            max_sequence_length=max_seq,
        )
        return {
            "prompt_embeds": prompt_embeds.to(self.device, dtype=td),
            "pooled_prompt_embeds": pooled_prompt_embeds.to(self.device, dtype=td),
            "prompt_attention_mask": (
                None if prompt_attention_mask is None else prompt_attention_mask.to(self.device)
            ),
        }

    # -- prepare_sampling ----------------------------------------------

    def prepare_sampling(
        self,
        request: DenoiseRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> HunyuanVideoSamplingState:
        """Build the per-request 5D-latent SamplingState for a denoise loop."""
        del kwargs
        import numpy as np

        pipe = self.pipeline
        device = self.device

        prompt_embeds = encoded["prompt_embeds"]
        pooled_prompt_embeds = encoded["pooled_prompt_embeds"]
        prompt_attention_mask = encoded.get("prompt_attention_mask")

        # Mirror the reference pipeline's explicit sigma ladder (the scheduler
        # applies shift=7.0 on top).
        sigmas = np.linspace(1.0, 0.0, request.num_steps + 1)[:-1]
        pipe.scheduler.set_timesteps(sigmas=sigmas, device=device)
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
            request.frame_count,
            torch.float32,
            device,
            generator,
            None,
        )

        return HunyuanVideoSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=pipe.scheduler,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            pooled_prompt_embeds=pooled_prompt_embeds,
            guidance_scale=request.guidance_scale,
        )

    # -- forward_step --------------------------------------------------

    def forward_step(
        self,
        state: HunyuanVideoSamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        """HunyuanVideo transformer forward (distilled guidance, single branch)."""
        t = state.timesteps[step_idx]
        bsz = state.latents.shape[0]
        td = self._transformer_dtype()

        latent_input = state.latents.to(td)
        timestep_batch = expand_batch_timestep(t, bsz).to(
            device=latent_input.device,
            dtype=td,
        )
        guidance = torch.full(
            (bsz,),
            float(state.guidance_scale) * 1000.0,
            device=latent_input.device,
            dtype=td,
        )
        output = DiffusionBackboneCaller(
            self.transformer,
            self,
        )(
            DiffusionBackboneInput(
                hidden_states=latent_input,
                timestep=timestep_batch,
                prompt_embeds=state.prompt_embeds.to(td),
                guidance_scale=state.guidance_scale,
                do_cfg=False,
                output_dtype=td,
                extra={
                    "encoder_attention_mask": state.prompt_attention_mask,
                    "pooled_projections": state.pooled_prompt_embeds.to(td),
                    "guidance": guidance,
                },
            ),
        )
        return output.as_dict()

    # -- collector boundary --------------------------------------------

    def export_batch_context(self, state: HunyuanVideoSamplingState) -> dict[str, Any]:
        """Project HunyuanVideo sampling state into trajectory context."""
        return {
            "guidance_scale": state.guidance_scale,
            "cfg": False,
        }

    def export_replay_tensors(self, state: HunyuanVideoSamplingState) -> dict[str, Any]:
        """Project HunyuanVideo sampling state into per-sample trajectory tensors."""
        tensors: dict[str, Any] = {
            "prompt_embeds": state.prompt_embeds,
            "pooled_prompt_embeds": state.pooled_prompt_embeds,
        }
        if state.prompt_attention_mask is not None:
            tensors["prompt_attention_mask"] = state.prompt_attention_mask
        return tensors

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> HunyuanVideoSamplingState:
        """Rebuild HunyuanVideoSamplingState from a batch slice for the eval path."""
        ts = replay_tensors["timesteps"]
        timesteps = pack_eval_timestep(ts, step_idx)
        return HunyuanVideoSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=None,  # not needed for forward_step (no scheduler.step here)
            prompt_embeds=replay_tensors["prompt_embeds"],
            prompt_attention_mask=replay_tensors.get("prompt_attention_mask"),
            pooled_prompt_embeds=replay_tensors["pooled_prompt_embeds"],
            guidance_scale=batch_context["guidance_scale"],
        )


class HunyuanVideoReplayModel(DiffusersReplayModelBase, HunyuanVideoModel):
    """Replay-only HunyuanVideo model owning no prompt encoders, VAE, or pipeline."""


__all__ = [
    "HunyuanVideoModel",
    "HunyuanVideoReplayModel",
    "HunyuanVideoSamplingState",
]
