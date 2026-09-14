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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from vrl.models.steps.denoise import (
    DiffusersReplayModelBase,
)
from vrl.models.steps.denoise.common import (
    DenoiseBackboneInput,
    DenoiseBranch,
    MaskedPromptDenoiseModel,
    TrainTimestepMaskedPromptSamplingState,
    VaeDecodeMixin,
)


@dataclass
class Lumina2SamplingState(TrainTimestepMaskedPromptSamplingState):
    """Private Lumina2 sampling state. Engine MUST NOT introspect."""


class Lumina2Model(
    VaeDecodeMixin,
    MaskedPromptDenoiseModel,
):
    """Diffusers-backed Lumina-Image-2.0 t2i model."""

    cfg_mode = "separate_cfg"
    cfg_base = "uncond"
    # Lumina's reference pipeline rescales the combined prediction back to the
    # conditional branch's norm on every CFG step.
    cfg_normalization = True
    sampling_state_cls = Lumina2SamplingState
    _default_max_sequence_length = 256
    _default_guidance_scale = 4.0
    _pipeline_encode_kwargs: ClassVar[Mapping[str, Any]] = {
        "num_images_per_prompt": 1,
        # VRL always uses the pipeline's default system-prompt template
        # ("<system> <Prompt Start> <prompt>").
        "system_prompt": None,
    }

    def _backbone_timestep(
        self,
        timestep: torch.Tensor,
        state: Lumina2SamplingState,
    ) -> torch.Tensor:
        # Lumina uses t=0 as noise and t=1 as the image; the transformer sees
        # the reversed normalized time while the scheduler steps on raw t.
        return 1.0 - timestep.to(self._transformer_dtype()) / float(state.num_train_timesteps)

    # -- backend ownership (called by runtime, not by collectors) -------
    _pipeline_classname = "Lumina2Pipeline"
    _frozen_encoder_names = ("text_encoder",)
    # Gemma-2-2B co-resides with the 2.6B DiT; keep it on-device.
    _prompt_encoder_on_cpu = False

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
