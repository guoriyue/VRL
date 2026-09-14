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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from vrl.models.interfaces.runtime import ModelBuild
from vrl.models.steps.denoise import (
    DiffusersReplayModelBase,
)
from vrl.models.steps.denoise.common import (
    MaskedPromptDenoiseModel,
    MaskedPromptSamplingState,
    VaeDecodeMixin,
)


@dataclass
class SanaSamplingState(MaskedPromptSamplingState):
    """Private SANA sampling state. Engine MUST NOT introspect."""


class SanaModel(
    VaeDecodeMixin,
    MaskedPromptDenoiseModel,
):
    """Diffusers-backed SANA t2i model.

    Implements the backbone-runner protocol itself. Both CFG branches pad to
    the same sequence length (Gemma-2 encode at ``max_sequence_length``), so
    they pack into one batched transformer call; the attention masks ride the
    batch as ``encoder_attention_mask``.
    """

    cfg_mode = "batched_cfg"
    cfg_base = "uncond"
    sampling_state_cls = SanaSamplingState
    _default_max_sequence_length = 300
    _default_guidance_scale = 4.5
    _pipeline_encode_kwargs: ClassVar[Mapping[str, Any]] = {
        "num_images_per_prompt": 1,
        # VRL intentionally disables SANA's CHI template so RL datasets own
        # their prompts. Pin to None so an upstream default flip cannot
        # silently re-enable it.
        "complex_human_instruction": None,
    }
    # SanaPipeline promotes each transformer branch before CFG. Keeping this
    # fp32 also feeds the protected scheduler/log-prob path without a lossy
    # fp16 round trip.
    _backbone_output_dtype = torch.float32

    def _backbone_timestep(
        self,
        timestep: torch.Tensor,
        state: SanaSamplingState,
    ) -> torch.Tensor:
        # SanaPipeline multiplies the raw timestep by config.timestep_scale and
        # keeps it in fp32; the time embedding owns its internal conversion.
        del state
        return timestep * float(getattr(self.transformer.config, "timestep_scale", 1.0))

    _pipeline_classname = "SanaPipeline"
    _frozen_encoder_names = ("text_encoder",)
    # Gemma-2-2B is small enough to co-reside with the 1.6B DiT; keep it
    # on-device (no CPU offload dance like Qwen-Image's 15 GB VL).
    _prompt_encoder_on_cpu = False

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
