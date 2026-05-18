"""Cosmos Predict2.5 runner for shared diffusion backbone call execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from vrl.models.diffusion.common import (
    DiffusionBackboneInput,
    DiffusionBranch,
)


@dataclass(slots=True)
class CosmosPredict25DiffusionBackboneRunner:
    """Map Cosmos Predict2.5 transformer kwargs into the shared backbone contract."""

    cfg_mode = "separate_cfg"
    cfg_base = "cond"
    sigma: torch.Tensor
    conditional_frame_timestep: float = 0.1

    def build_branch(
        self,
        request: DiffusionBackboneInput,
        branch: Literal["cond", "uncond"],
    ) -> DiffusionBranch:
        extra = request.extra
        embeds = request.prompt_embeds
        if branch == "uncond":
            embeds = _require_tensor(request.negative_prompt_embeds)
        hidden_states, timestep, gt_velocity = self._prepare_branch(
            latents=request.hidden_states,
            cond_latent=extra["cond_latent"],
            cond_mask=extra["cond_mask"],
            cond_indicator=extra["cond_indicator"],
        )
        return DiffusionBranch(
            name=branch,
            hidden_states=hidden_states.to(extra["transformer_dtype"]),
            timestep=timestep.to(extra["transformer_dtype"]),
            encoder_hidden_states=embeds,
            extra_kwargs={
                "condition_mask": extra["cond_mask"].to(extra["transformer_dtype"]),
                "padding_mask": extra["padding_mask"],
            },
            metadata={"gt_velocity": gt_velocity},
        )

    def postprocess_branch(
        self,
        request: DiffusionBackboneInput,
        branch: DiffusionBranch,
        raw_output: torch.Tensor,
    ) -> torch.Tensor:
        return branch.metadata["gt_velocity"] + raw_output * (
            1 - request.extra["cond_mask"]
        )

    def finalize_noise_pred(
        self,
        request: DiffusionBackboneInput,
        combined: torch.Tensor,
        cond: torch.Tensor,
        uncond: torch.Tensor,
    ) -> torch.Tensor:
        del request, cond, uncond
        return combined

    def _prepare_branch(
        self,
        *,
        latents: torch.Tensor,
        cond_latent: torch.Tensor,
        cond_mask: torch.Tensor,
        cond_indicator: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        condition_timestep = torch.ones_like(cond_indicator) * float(
            self.conditional_frame_timestep,
        )
        hidden_states = cond_mask * cond_latent + (1 - cond_mask) * latents
        timestep = cond_indicator * condition_timestep + (
            1 - cond_indicator
        ) * self.sigma
        gt_velocity = (latents - cond_latent) * cond_mask
        return hidden_states, timestep, gt_velocity


def _require_tensor(value: torch.Tensor | None) -> torch.Tensor:
    if value is None:
        raise ValueError("Cosmos Predict2.5 CFG branch requires negative_prompt_embeds")
    return value
