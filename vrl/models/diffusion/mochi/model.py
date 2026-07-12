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

import random
import sys
from dataclasses import dataclass
from typing import Any

import torch

from vrl.generation.diffusion.layout import VideoGenerationRequest
from vrl.models.diffusion import (
    DiffusersPipelineModelBase,
    DiffusersReplayModelBase,
    DiffusionSamplingStateBase,
    diffusers_pipeline_dtypes,
)
from vrl.models.diffusion.common import (
    ChunkedLatentDecoder,
    DiffusionBackboneCaller,
    DiffusionBackboneInput,
    DiffusionBackboneRunnerBase,
    DiffusionBranch,
    LatentDecodePlan,
    expand_batch_timestep,
    pack_eval_timestep,
)
from vrl.models.diffusion.common.lora import LoraModelMixin
from vrl.models.diffusion.common.tensors import require_tensor
from vrl.models.interfaces.runtime import ModelBuild


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
class MochiSamplingState(DiffusionSamplingStateBase):
    """Private Mochi sampling state. Engine MUST NOT introspect."""

    prompt_embeds: torch.Tensor
    prompt_attention_mask: torch.Tensor | None
    negative_prompt_embeds: torch.Tensor | None
    negative_prompt_attention_mask: torch.Tensor | None
    do_cfg: bool
    num_train_timesteps: int


class MochiModel(LoraModelMixin, DiffusersPipelineModelBase, DiffusionBackboneRunnerBase):
    """Diffusers-backed Mochi-1 t2v model."""

    cfg_mode = "batched_cfg"
    cfg_base = "uncond"

    def build_branch(
        self,
        request: DiffusionBackboneInput,
        branch: str,
    ) -> DiffusionBranch:
        """Map Mochi transformer kwargs into the shared backbone contract."""
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
        """Negate: a velocity in Mochi's reversed time axis is -v_standard."""
        del request, branch
        return -raw_output

    # -- backend ownership (called by runtime, not by collectors) -------

    @classmethod
    def from_build(cls, build: ModelBuild) -> MochiModel:
        """Load the diffusers Mochi pipeline + freeze non-trainable modules."""
        from diffusers import MochiPipeline

        model_dtype = build.parameter_dtype
        prompt_encoder_dtype, load_kwargs = diffusers_pipeline_dtypes(build, model_dtype)
        pipeline = MochiPipeline.from_pretrained(
            build.model_name_or_path,
            **load_kwargs,
        )
        pipeline.vae.requires_grad_(False)
        # T5-XXL (~9.5 GB bf16) + the 20 GB 10B transformer exceed a 32 GB
        # card; park the encoder on CPU (Qwen-Image discipline).
        text_encoder = getattr(pipeline, "text_encoder", None)
        if text_encoder is not None:
            text_encoder.requires_grad_(False)
            text_encoder.to("cpu", dtype=prompt_encoder_dtype)
        pipeline.vae.to(build.device, dtype=torch.float32)
        return cls(
            pipeline=pipeline,
            device=build.device,
        )

    def _encoder_device(self) -> Any:
        """Device the frozen T5 encoder lives on (CPU when offloaded)."""
        enc = getattr(self.pipeline, "text_encoder", None)
        if enc is not None:
            try:
                return next(enc.parameters()).device
            except StopIteration:
                pass
        return self.device

    # -- encode_prompt -------------------------------------------------

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode prompt via T5-XXL (sequence embeds + bool mask, no pooled)."""
        max_seq = kwargs.get("max_sequence_length", 256)
        guidance_scale = kwargs.get("guidance_scale", 4.5)
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
            num_videos_per_prompt=1,
            max_sequence_length=max_seq,
            device=self._encoder_device(),
        )

        td = self.transformer.dtype
        result: dict[str, Any] = {
            "prompt_embeds": prompt_embeds.to(self.device, dtype=td),
            "prompt_attention_mask": (
                None if prompt_attention_mask is None else prompt_attention_mask.to(self.device)
            ),
        }
        if do_cfg and negative_prompt_embeds is not None:
            result["negative_prompt_embeds"] = negative_prompt_embeds.to(
                self.device,
                dtype=td,
            )
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
    ) -> MochiSamplingState:
        """Build the per-request 5D-latent SamplingState for a denoise loop."""
        del kwargs
        pipe = self.pipeline
        device = self.device

        prompt_embeds = encoded["prompt_embeds"]
        prompt_attention_mask = encoded.get("prompt_attention_mask")
        negative_prompt_embeds = encoded.get("negative_prompt_embeds")
        negative_prompt_attention_mask = encoded.get("negative_prompt_attention_mask")

        scheduler = standard_mochi_scheduler(
            pipe.scheduler.config,
            request.num_steps,
            device,
        )
        timesteps = scheduler.timesteps

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

        do_cfg = request.guidance_scale > 1.0 and negative_prompt_embeds is not None

        return MochiSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=scheduler,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            guidance_scale=request.guidance_scale,
            do_cfg=do_cfg,
            num_train_timesteps=int(scheduler.config.num_train_timesteps),
        )

    # -- forward_step --------------------------------------------------

    def forward_step(
        self,
        state: MochiSamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        """Mochi transformer forward on its native (reversed) clock."""
        t = state.timesteps[step_idx]
        bsz = state.latents.shape[0]
        td = self._transformer_dtype()

        latent_input = state.latents.to(td)
        # Standard clock t descends 1000->0; Mochi's native clock ascends, so
        # the transformer sees num_train_timesteps - t.
        timestep_batch = float(state.num_train_timesteps) - expand_batch_timestep(t, bsz).to(
            device=latent_input.device, dtype=td
        )
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

    def export_batch_context(self, state: MochiSamplingState) -> dict[str, Any]:
        """Project Mochi sampling state into trajectory context."""
        return {
            "guidance_scale": state.guidance_scale,
            "cfg": state.do_cfg,
            "num_train_timesteps": state.num_train_timesteps,
        }

    def export_replay_tensors(self, state: MochiSamplingState) -> dict[str, Any]:
        """Project Mochi sampling state into per-sample trajectory tensors."""
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
    ) -> MochiSamplingState:
        """Rebuild MochiSamplingState from a batch slice for the eval forward path."""
        ts = replay_tensors["timesteps"]
        timesteps = pack_eval_timestep(ts, step_idx)
        return MochiSamplingState(
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
        """Decode 5D latents -> video [B, C, T, H, W] via the Mochi VAE."""
        pipe = self.pipeline
        vae = pipe.vae

        def _transform(chunk: torch.Tensor) -> torch.Tensor:
            x = chunk.to(vae.dtype)
            mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(x.device, x.dtype)
            std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(x.device, x.dtype)
            return x * std / vae.config.scaling_factor + mean

        decoder = ChunkedLatentDecoder(
            LatentDecodePlan(
                prepare_latents=_transform,
                vae_decode=lambda chunk: vae.decode(chunk, return_dict=False)[0],
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
        # A scheduler without .config is a hand-injected test double — only
        # real diffusers schedulers carry the shipped (inverted) config that
        # needs standardizing.
        config = getattr(self._scheduler, "config", None)
        if num_steps is not None and config is not None:
            self._scheduler = standard_mochi_scheduler(
                config,
                int(num_steps),
                build.device,
            )


__all__ = [
    "MochiModel",
    "MochiReplayModel",
    "MochiSamplingState",
    "standard_mochi_scheduler",
]
