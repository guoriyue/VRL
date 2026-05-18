"""Wan 2.1 runner for shared diffusion backbone call execution."""

from __future__ import annotations

from typing import Literal

import torch

from vrl.models.diffusion.common import (
    DiffusionBackboneInput,
    DiffusionBranch,
)


class WanDiffusionBackboneRunner:
    """Map Wan transformer kwargs into the shared backbone contract."""

    cfg_mode = "batched_cfg"
    cfg_base = "uncond"

    def build_branch(
        self,
        request: DiffusionBackboneInput,
        branch: Literal["cond", "uncond"],
    ) -> DiffusionBranch:
        embeds = request.prompt_embeds
        if branch == "uncond":
            embeds = _require_tensor(request.negative_prompt_embeds)
        return DiffusionBranch(
            name=branch,
            hidden_states=request.hidden_states,
            timestep=request.timestep,
            encoder_hidden_states=embeds,
        )

    def postprocess_branch(
        self,
        request: DiffusionBackboneInput,
        branch: DiffusionBranch,
        raw_output: torch.Tensor,
    ) -> torch.Tensor:
        del request, branch
        return raw_output

    def finalize_noise_pred(
        self,
        request: DiffusionBackboneInput,
        combined: torch.Tensor,
        cond: torch.Tensor,
        uncond: torch.Tensor,
    ) -> torch.Tensor:
        del request, cond, uncond
        return combined


def _require_tensor(value: torch.Tensor | None) -> torch.Tensor:
    if value is None:
        raise ValueError("Wan CFG branch requires negative_prompt_embeds")
    return value
