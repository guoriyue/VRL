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

from vrl.generation.types import DenoiseRequest
from vrl.models.interfaces.runtime import ModelBuild
from vrl.models.steps.denoise import (
    DiffusersPipelineModelBase,
    DiffusersReplayModelBase,
    GuidedDenoiseSamplingStateBase,
)
from vrl.models.steps.denoise.common import (
    ChunkedLatentDecoder,
    DenoiseBackboneCaller,
    DenoiseBackboneInput,
    DenoiseBackboneRunnerBase,
    DenoiseBranch,
    LatentDecodePlan,
    expand_batch_timestep,
    pack_eval_timestep,
)


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
class MochiSamplingState(GuidedDenoiseSamplingStateBase):
    """Private Mochi sampling state. Engine MUST NOT introspect.

    ``latents`` / ``timesteps`` / ``scheduler`` are the only fields the batch
    executor may touch. ``num_train_timesteps`` is the training-clock length
    the transformer's native ascending time is derived from: replay rebuilds
    one packed step without a scheduler, so the constant travels in the batch
    context instead of being read back off ``scheduler.config``.
    """

    prompt_embeds: torch.Tensor
    prompt_attention_mask: torch.Tensor | None
    negative_prompt_embeds: torch.Tensor | None
    negative_prompt_attention_mask: torch.Tensor | None
    do_cfg: bool
    num_train_timesteps: int


class MochiModel(
    DiffusersPipelineModelBase,
    DenoiseBackboneRunnerBase,
):
    """Diffusers-backed Mochi-1 t2v model.

    Implements the backbone-runner protocol itself. Both CFG branches pad to
    the same sequence length (T5 encode at ``max_sequence_length``), so they
    pack into one batched transformer call; the attention masks ride the
    batch as ``encoder_attention_mask``.
    """

    cfg_mode = "batched_cfg"
    cfg_base = "uncond"

    def build_branch(
        self,
        request: DenoiseBackboneInput,
        branch: str,
    ) -> DenoiseBranch:
        """Map the branch's T5 embeds and attention mask into a branch call."""
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
        """Encode the prompt via T5-XXL to sequence embeds + bool padding mask (no pooled)."""
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
            device=self._encoder_device(),
            max_sequence_length=max_seq,
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
        *,
        initial_latents: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> MochiSamplingState:
        """Build the per-request SamplingState for a denoise loop."""
        del kwargs
        pipe = self.pipeline
        device = self.device

        prompt_embeds = encoded["prompt_embeds"]
        negative_prompt_embeds = encoded.get("negative_prompt_embeds")

        # The shipped inverted-sigma scheduler is NOT used: the rollout steps on
        # the standard descending domain (see :func:`standard_mochi_scheduler`).
        scheduler = standard_mochi_scheduler(
            pipe.scheduler.config,
            request.num_steps,
            device,
        )

        seed = request.seed if request.seed is not None else random.randint(0, sys.maxsize)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        latents = pipe.prepare_latents(
            prompt_embeds.shape[0],
            pipe.transformer.config.in_channels,
            request.height,
            request.width,
            request.frame_count,
            torch.float32,
            device,
            generator,
            initial_latents,
        )

        return MochiSamplingState(
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
        state: MochiSamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        """Transformer forward + batched CFG."""
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

    def export_batch_context(self, state: MochiSamplingState) -> dict[str, Any]:
        """Project sampling state into shared trajectory context."""
        return {
            "guidance_scale": state.guidance_scale,
            "cfg": state.do_cfg,
            "num_train_timesteps": state.num_train_timesteps,
        }

    def export_replay_tensors(self, state: MochiSamplingState) -> dict[str, Any]:
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
    ) -> MochiSamplingState:
        """Rebuild the sampling state from a batch slice for eval forward."""
        negative_prompt_embeds = replay_tensors.get("negative_prompt_embeds")
        return MochiSamplingState(
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
