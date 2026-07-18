"""SANA t2i diffusers-backed model.

Diffusion implementation for NVIDIA SANA (linear-attention DiT + DC-AE).
The generation helper flow mirrors every diffusion family:

    encode_prompt -> prepare_sampling -> forward_step xN -> decode_latents

SANA specifics vs SD3 (the reference family):
- Single Gemma-2 text encoder: ``encode_prompt`` returns a SEQUENCE embed plus
  an attention mask and NO pooled vector; the mask threads through the
  transformer as ``encoder_attention_mask`` on both CFG branches.
- Latents are UNPACKED ``[B, 32, H/32, W/32]`` (DC-AE, 32x compression — vs
  the 8x KL-VAE of SD3/FLUX). No packing anywhere; ``decode_latents`` is a
  plain ``latents / scaling_factor`` decode (DC-AE has no shift_factor).
- TRUE classifier-free guidance with both branches padded to the same
  ``max_sequence_length`` (300), so the branches batch into one forward
  (``batched_cfg``, like SD3 — unlike Qwen-Image's separate branches).
- The transformer multiplies the timestep by ``config.timestep_scale``
  (1.0 on current checkpoints; respected for parity with SanaPipeline).
- ``complex_human_instruction`` (SANA's CHI prompt template) is exposed as a
  sampling kwarg and defaults to OFF: RL datasets control their prompts.
  diffusers' ``SanaPipeline.__call__`` defaults it ON, so GPU parity runs must
  pass the same CHI value on both sides.
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


@dataclass
class SanaSamplingState(DiffusionSamplingStateBase):
    """Private SANA sampling state. Engine MUST NOT introspect."""

    prompt_embeds: torch.Tensor
    prompt_attention_mask: torch.Tensor | None
    negative_prompt_embeds: torch.Tensor | None
    negative_prompt_attention_mask: torch.Tensor | None
    do_cfg: bool


class SanaModel(LoraModelMixin, DiffusersPipelineModelBase, DiffusionBackboneRunnerBase):
    """Diffusers-backed SANA t2i model.

    Implements the backbone-runner protocol itself. Both CFG branches pad to
    the same sequence length (Gemma-2 encode at ``max_sequence_length``), so
    they pack into one batched transformer call; the attention masks ride the
    batch as ``encoder_attention_mask``.
    """

    cfg_mode = "batched_cfg"
    cfg_base = "uncond"

    def build_branch(
        self,
        request: DiffusionBackboneInput,
        branch: str,
    ) -> DiffusionBranch:
        """Map SANA transformer kwargs into the shared backbone contract."""
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

    # -- backend ownership (called by runtime, not by collectors) -------

    @classmethod
    def from_build(cls, build: ModelBuild) -> SanaModel:
        """Load the diffusers SANA pipeline + freeze non-trainable modules."""
        from diffusers import SanaPipeline

        # The family capability rejects non-native transformer dtypes before
        # this loader runs. Rollout and replay both receive the validated FP16
        # role dtype from the unified precision policy.
        model_dtype = build.parameter_dtype
        prompt_encoder_dtype, load_kwargs = diffusers_pipeline_dtypes(build, model_dtype)
        pipeline = SanaPipeline.from_pretrained(
            build.model_name_or_path,
            **load_kwargs,
        )
        pipeline.vae.requires_grad_(False)
        text_encoder = getattr(pipeline, "text_encoder", None)
        if text_encoder is not None:
            # Gemma-2-2B is small enough to co-reside with the 1.6B DiT; keep
            # it on-device (no CPU offload dance like Qwen-Image's 15 GB VL).
            text_encoder.requires_grad_(False)
            text_encoder.to(build.device, dtype=prompt_encoder_dtype)
        # DC-AE decodes in fp32 for output fidelity regardless of denoiser dtype.
        pipeline.vae.to(build.device, dtype=torch.float32)
        # SANA is rectified-flow native; diffusers ships DPMSolverMultistep for
        # fast inference, but flow-matching GRPO's per-step SDE log-prob needs a
        # FlowMatchEuler scheduler on BOTH sides. The replay bundle already loads
        # FlowMatchEuler (build.py, no scheduler_classname); the rollout was still
        # on DPMSolver, so rollout timesteps never matched replay's and
        # index_for_timestep(t) returned empty at the first-step parity check.
        # Swap rollout to FlowMatchEuler for per-step log-prob. SANA's shipped
        # DPM config calls this value ``flow_shift``; FlowMatch calls it ``shift``.
        # Passing it explicitly preserves the checkpoint's shift=3 instead of
        # silently accepting FlowMatch's default shift=1 (the color-block bug).
        from diffusers import FlowMatchEulerDiscreteScheduler

        scheduler_config = dict(pipeline.scheduler.config)
        pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            scheduler_config,
            shift=float(scheduler_config.get("flow_shift", 1.0)),
        )
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

        Returns the conditional embeds/mask and, when CFG is active, the
        unconditional embeds/mask (SANA's uncond default is the empty string).
        """
        max_seq = kwargs.get("max_sequence_length", 300)
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
            do_classifier_free_guidance=do_cfg,
            negative_prompt=neg,
            num_images_per_prompt=1,
            device=self.device,
            max_sequence_length=max_seq,
            complex_human_instruction=kwargs.get("complex_human_instruction"),
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
    ) -> SanaSamplingState:
        """Build the per-request SamplingState for a denoise loop."""
        del kwargs
        pipe = self.pipeline
        device = self.device

        prompt_embeds = encoded["prompt_embeds"]
        prompt_attention_mask = encoded.get("prompt_attention_mask")
        negative_prompt_embeds = encoded.get("negative_prompt_embeds")
        negative_prompt_attention_mask = encoded.get("negative_prompt_attention_mask")

        # Static flow-shift schedule (SANA has no dynamic shifting).
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

        return SanaSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=pipe.scheduler,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            guidance_scale=request.guidance_scale,
            do_cfg=do_cfg,
        )

    # -- forward_step --------------------------------------------------

    def forward_step(
        self,
        state: SanaSamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        """SANA transformer forward + optional batched CFG."""
        t = state.timesteps[step_idx]
        bsz = state.latents.shape[0]
        td = self._transformer_dtype()

        latent_input = state.latents.to(td)
        # SanaPipeline multiplies the raw timestep by config.timestep_scale.
        timestep_scale = float(
            getattr(self.transformer.config, "timestep_scale", 1.0),
        )
        # Keep the scheduler timestep in fp32, exactly as SanaPipeline does.
        # Transformer inputs/weights remain at the native FP16 role dtype;
        # the time embedding owns its internal conversion.
        timestep_batch = expand_batch_timestep(t, bsz).to(latent_input.device) * timestep_scale
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
                # SanaPipeline promotes each transformer branch before CFG.
                # Keeping this fp32 also feeds the protected scheduler/log-prob
                # path without a lossy fp16 round trip.
                output_dtype=torch.float32,
                extra={
                    "encoder_attention_mask": state.prompt_attention_mask,
                    "negative_encoder_attention_mask": state.negative_prompt_attention_mask,
                },
            ),
        )
        return output.as_dict()

    # -- collector boundary --------------------------------------------

    def export_batch_context(self, state: SanaSamplingState) -> dict[str, Any]:
        """Project SANA sampling state into trajectory context."""
        return {
            "guidance_scale": state.guidance_scale,
            "cfg": state.do_cfg,
        }

    def export_replay_tensors(self, state: SanaSamplingState) -> dict[str, Any]:
        """Project SANA sampling state into per-sample trajectory tensors.

        Masks / negative embeds are only stored when present, so ``restore``
        reads them with ``.get`` and the no-CFG path stays tensor-free.
        """
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
    ) -> SanaSamplingState:
        """Rebuild SanaSamplingState from a batch slice for the eval forward path."""
        ts = replay_tensors["timesteps"]
        timesteps = pack_eval_timestep(ts, step_idx)
        return SanaSamplingState(
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
        )

    # -- decode_latents ------------------------------------------------

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents -> image via DC-AE (4D, no shift_factor, no unpack)."""
        pipe = self.pipeline
        vae = pipe.vae
        scaling_factor = vae.config.scaling_factor
        decoder = ChunkedLatentDecoder(
            LatentDecodePlan(
                prepare_latents=lambda chunk: chunk.to(vae.dtype) / scaling_factor,
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


class SanaReplayModel(DiffusersReplayModelBase, SanaModel):
    """Replay-only SANA model that owns no prompt encoder, VAE, or pipeline."""


__all__ = ["SanaModel", "SanaReplayModel", "SanaSamplingState"]
