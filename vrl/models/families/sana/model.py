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
- ``complex_human_instruction`` (SANA's CHI prompt template) is always
  disabled: VRL fixes it to None so RL datasets control their prompts.
  diffusers' ``SanaPipeline.__call__`` defaults it ON, so GPU parity runs must
  pass the same (disabled) CHI value on both sides.
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
    DenoiseBackboneCaller,
    DenoiseBackboneInput,
    DenoiseBackboneRunnerBase,
    DenoiseBranch,
    VaeDecodeMixin,
    expand_batch_timestep,
    pack_eval_timestep,
)


@dataclass
class SanaSamplingState(GuidedDenoiseSamplingStateBase):
    """Private SANA sampling state. Engine MUST NOT introspect.

    ``latents`` / ``timesteps`` / ``scheduler`` are the only fields the batch
    executor may touch.
    """

    prompt_embeds: torch.Tensor
    prompt_attention_mask: torch.Tensor | None
    negative_prompt_embeds: torch.Tensor | None
    negative_prompt_attention_mask: torch.Tensor | None
    do_cfg: bool


class SanaModel(
    VaeDecodeMixin,
    DiffusersPipelineModelBase,
    DenoiseBackboneRunnerBase,
):
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
            negative_prompt=neg,
            do_classifier_free_guidance=do_cfg,
            num_images_per_prompt=1,
            device=self._encoder_device(),
            max_sequence_length=max_seq,
            # VRL intentionally disables SANA's CHI template so RL datasets own
            # their prompts. Pin to None so an upstream default flip cannot
            # silently re-enable it.
            complex_human_instruction=None,
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
    ) -> SanaSamplingState:
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
            initial_latents,
        )

        return SanaSamplingState(
            latents=latents,
            timesteps=scheduler.timesteps,
            scheduler=scheduler,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=encoded.get("prompt_attention_mask"),
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=encoded.get("negative_prompt_attention_mask"),
            guidance_scale=request.guidance_scale,
            do_cfg=request.guidance_scale > 1.0 and negative_prompt_embeds is not None,
        )

    # -- forward_step --------------------------------------------------

    def forward_step(
        self,
        state: SanaSamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        """Transformer forward + batched CFG."""
        t = state.timesteps[step_idx]
        bsz = state.latents.shape[0]
        td = self._transformer_dtype()

        latent_input = state.latents.to(td)
        # SanaPipeline multiplies the raw timestep by config.timestep_scale and
        # keeps it in fp32; the time embedding owns its internal conversion.
        timestep_batch = expand_batch_timestep(t, bsz).to(device=latent_input.device) * float(
            getattr(self.transformer.config, "timestep_scale", 1.0)
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

    # -- trajectory boundary -------------------------------------------
    # Masks and negative embeds are exported only when present, so the no-CFG
    # path stays tensor-free and restore reads them back with ``.get``.

    def export_batch_context(self, state: SanaSamplingState) -> dict[str, Any]:
        """Project sampling state into shared trajectory context."""
        return {"guidance_scale": state.guidance_scale, "cfg": state.do_cfg}

    def export_replay_tensors(self, state: SanaSamplingState) -> dict[str, Any]:
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
    ) -> SanaSamplingState:
        """Rebuild the sampling state from a batch slice for eval forward."""
        negative_prompt_embeds = replay_tensors.get("negative_prompt_embeds")
        return SanaSamplingState(
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
        )

    # -- backend ownership (called by runtime, not by collectors) -------

    @staticmethod
    def _apply_fp16_saturation_clamp(transformer: Any) -> None:
        """Reapply SANA's fp16 attention saturation on a non-fp16 transformer.

        The published fp16 checkpoint was calibrated WITH diffusers'
        ``SanaLinearAttnProcessor2_0`` output clip to the fp16 range, but that
        clip is dtype-conditional (``if original_dtype == torch.float16``).
        Running the same weights in fp32/bf16 skips it, and the un-saturated
        linear-attention outputs corrupt every image. It must be reapplied at
        each linear-attention layer (not the transformer's final output), so it
        rides ``set_attn_processor`` rather than ``forward_step``. Verified
        2026-07-18: the official pipeline in fp32 reproduces the corruption, and
        fp32 plus this clamp matches fp16 quality
        (outputs/quality_preflight/sana_fp32_probe).
        """

        from diffusers.models.attention_processor import SanaLinearAttnProcessor2_0

        class _SaturatedLinearAttnProcessor(SanaLinearAttnProcessor2_0):
            def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
                return super().__call__(*args, **kwargs).clip(-65504, 65504)

        transformer.set_attn_processor(
            {
                name: (
                    _SaturatedLinearAttnProcessor()
                    if type(processor) is SanaLinearAttnProcessor2_0
                    else processor
                )
                for name, processor in transformer.attn_processors.items()
            },
        )

    @classmethod
    def from_build(cls, build: ModelBuild) -> SanaModel:
        """Shared pipeline load, then SANA's two deviations: flow-match scheduler + fp16 clamp.

        The freeze / placement sequence (frozen encoder at the rollout prompt
        dtype, DC-AE in fp32 for decode fidelity) is the shared loader's; only
        what follows is SANA-specific.
        """
        model = super().from_build(build)
        pipeline = model.pipeline
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
        if build.parameter_dtype != torch.float16:
            cls._apply_fp16_saturation_clamp(pipeline.transformer)
        return model

    def prepare_replay(self, build: ModelBuild) -> None:
        """Replay forwards need the same non-fp16 saturation clamp as rollout."""
        if build.parameter_dtype != torch.float16:
            self._apply_fp16_saturation_clamp(self.transformer)


class SanaReplayModel(DiffusersReplayModelBase, SanaModel):
    """Replay-only SANA model that owns no prompt encoder, VAE, or pipeline."""


__all__ = ["SanaModel", "SanaReplayModel", "SanaSamplingState"]
