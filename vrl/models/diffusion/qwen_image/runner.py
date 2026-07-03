"""Qwen-Image runner for shared diffusion backbone call execution.

Qwen-Image does TRUE classifier-free guidance: the conditional and
unconditional prompts tokenize to DIFFERENT sequence lengths (their
``encoder_hidden_states_mask`` differ), so the two branches cannot be packed
into one batched call (``pack_batched_cfg`` asserts matching shapes after the
batch dim). The runner therefore uses ``separate_cfg`` — two independent
transformer forwards — instead of sd3_5's ``batched_cfg``. It also reproduces
Qwen-Image's norm-preserving CFG combine in ``finalize_noise_pred``.
"""

from __future__ import annotations

from typing import Literal

import torch

from vrl.models.diffusion.common import (
    DiffusionBackboneInput,
    DiffusionBackboneRunnerBase,
    DiffusionBranch,
)
from vrl.models.diffusion.common.tensors import require_tensor


class QwenImageDiffusionBackboneRunner(DiffusionBackboneRunnerBase):
    """Map Qwen-Image transformer kwargs into the shared backbone contract."""

    cfg_mode = "separate_cfg"
    cfg_base = "uncond"

    def build_branch(
        self,
        request: DiffusionBackboneInput,
        branch: Literal["cond", "uncond"],
    ) -> DiffusionBranch:
        if branch == "cond":
            embeds = request.prompt_embeds
            mask = request.extra.get("encoder_hidden_states_mask")
        else:
            embeds = require_tensor(
                request.negative_prompt_embeds, "negative_prompt_embeds",
            )
            mask = request.extra.get("negative_encoder_hidden_states_mask")
        extra_kwargs = {
            "encoder_hidden_states_mask": mask,
            "img_shapes": request.extra["img_shapes"],
        }
        if request.extra.get("guidance") is not None:
            extra_kwargs["guidance"] = request.extra["guidance"]
        return DiffusionBranch(
            hidden_states=request.hidden_states,
            timestep=request.timestep,
            encoder_hidden_states=embeds,
            extra_kwargs=extra_kwargs,
        )

    def finalize_noise_pred(
        self,
        request: DiffusionBackboneInput,
        combined: torch.Tensor,
        cond: torch.Tensor,
        uncond: torch.Tensor,
    ) -> torch.Tensor:
        """Qwen-Image norm-preserving CFG: rescale the combined pred to the cond norm.

        Mirrors diffusers QwenImagePipeline:
            comb = uncond + s * (cond - uncond)   # done by combine_cfg upstream
            noise = comb * (||cond|| / ||comb||)
        """
        del uncond
        if not request.do_cfg:
            return combined
        cond_norm = torch.norm(cond, dim=-1, keepdim=True)
        comb_norm = torch.norm(combined, dim=-1, keepdim=True)
        return combined * (cond_norm / comb_norm)
