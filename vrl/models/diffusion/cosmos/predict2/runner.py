"""Cosmos Predict2 runner for shared diffusion backbone call execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from vrl.models.diffusion.common import (
    DiffusionBackboneInput,
    DiffusionBranch,
)
from vrl.models.diffusion.common.tensors import require_tensor


@dataclass(slots=True)
class CosmosPredict2DiffusionBackboneRunner:
    """Map Cosmos Predict2 transformer kwargs into the shared backbone contract."""

    cfg_mode = "separate_cfg"
    cfg_base = "cond"
    current_sigma: torch.Tensor
    sigma_conditioning: float = 0.0001

    @property
    def current_t(self) -> torch.Tensor:
        return self.current_sigma / (self.current_sigma + 1)

    def build_branch(
        self,
        request: DiffusionBackboneInput,
        branch: Literal["cond", "uncond"],
    ) -> DiffusionBranch:
        extra = request.extra
        if branch == "cond":
            embeds = request.prompt_embeds
            indicator = extra["cond_indicator"]
            condition_mask = extra["cond_mask"]
        else:
            embeds = require_tensor(request.negative_prompt_embeds, "Cosmos Predict2 negative_prompt_embeds")
            indicator = require_tensor(extra.get("uncond_indicator"), "Cosmos Predict2 uncond_indicator")
            condition_mask = extra.get("uncond_mask")
        hidden_states, timestep = self._prepare_branch(
            latents=request.hidden_states,
            init_latents=extra["init_latents"],
            indicator=indicator,
            timestep=request.timestep,
        )
        return DiffusionBranch(
            name=branch,
            hidden_states=hidden_states.to(extra["transformer_dtype"]),
            timestep=timestep.to(extra["transformer_dtype"]),
            encoder_hidden_states=embeds,
            extra_kwargs={
                "fps": extra["fps"],
                "condition_mask": condition_mask,
                "padding_mask": extra["padding_mask"],
            },
            metadata={"indicator": indicator},
        )

    def postprocess_branch(
        self,
        request: DiffusionBackboneInput,
        branch: DiffusionBranch,
        raw_output: torch.Tensor,
    ) -> torch.Tensor:
        return self._postprocess_branch(
            raw=raw_output,
            latents=request.hidden_states,
            init_latents=request.extra["init_latents"],
            indicator=branch.metadata["indicator"],
            dtype=request.output_dtype or raw_output.dtype,
        )

    def finalize_noise_pred(
        self,
        request: DiffusionBackboneInput,
        combined: torch.Tensor,
        cond: torch.Tensor,
        uncond: torch.Tensor,
    ) -> torch.Tensor:
        del cond, uncond
        return (request.hidden_states - combined) / self.current_sigma

    def _prepare_branch(
        self,
        *,
        latents: torch.Tensor,
        init_latents: torch.Tensor,
        indicator: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c_in = 1 - self.current_t
        hidden_states = indicator * init_latents + (1 - indicator) * (latents * c_in)
        condition_timestep = torch.full_like(timestep, float(self.sigma_conditioning))
        branch_timestep = indicator * condition_timestep + (1 - indicator) * timestep
        return hidden_states, branch_timestep

    def _postprocess_branch(
        self,
        *,
        raw: torch.Tensor,
        latents: torch.Tensor,
        init_latents: torch.Tensor,
        indicator: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        c_skip = 1 - self.current_t
        c_out = -self.current_t
        branch = (c_skip * latents + c_out * raw.float()).to(dtype)
        return indicator * init_latents + (1 - indicator) * branch


