"""Diffusion transformer branch orchestration shared by family models."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import torch

from vrl.models.diffusion.common.cfg import (
    DiffusionBranch,
    DiffusionCFGBase,
    combine_cfg,
    pack_batched_cfg,
    split_batched_cfg_output,
)

DiffusionCFGMode = Literal["batched_cfg", "separate_cfg", "single_branch"]


@dataclass(slots=True)
class DiffusionBackboneInput:
    """Inputs for one denoise transformer call."""

    hidden_states: torch.Tensor
    timestep: torch.Tensor
    prompt_embeds: torch.Tensor
    guidance_scale: float
    do_cfg: bool
    negative_prompt_embeds: torch.Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    output_dtype: torch.dtype | None = None


@dataclass(slots=True)
class DiffusionBackboneOutput:
    """Canonical diffusion branch output contract."""

    noise_pred: torch.Tensor
    noise_pred_cond: torch.Tensor
    noise_pred_uncond: torch.Tensor
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "noise_pred": self.noise_pred,
            "noise_pred_cond": self.noise_pred_cond,
            "noise_pred_uncond": self.noise_pred_uncond,
        }


class DiffusionBackboneRunner(Protocol):
    """Family-owned backbone call runner."""

    cfg_mode: DiffusionCFGMode
    cfg_base: DiffusionCFGBase

    def build_branch(
        self,
        request: DiffusionBackboneInput,
        branch: Literal["cond", "uncond"],
    ) -> DiffusionBranch:
        ...

    def postprocess_branch(
        self,
        request: DiffusionBackboneInput,
        branch: DiffusionBranch,
        raw_output: torch.Tensor,
    ) -> torch.Tensor:
        ...

    def finalize_noise_pred(
        self,
        request: DiffusionBackboneInput,
        combined: torch.Tensor,
        cond: torch.Tensor,
        uncond: torch.Tensor,
    ) -> torch.Tensor:
        ...


class DiffusionBackboneCaller:
    """Run one diffusion transformer step with shared CFG orchestration."""

    def __init__(self, transformer: Any, runner: DiffusionBackboneRunner) -> None:
        self.transformer = transformer
        self.runner = runner

    def __call__(self, request: DiffusionBackboneInput) -> DiffusionBackboneOutput:
        started = time.perf_counter()
        cond_branch = self.runner.build_branch(request, "cond")
        branch_calls = 1

        if request.do_cfg:
            uncond_branch = self.runner.build_branch(request, "uncond")
            if self.runner.cfg_mode == "batched_cfg":
                batched = pack_batched_cfg(cond=cond_branch, uncond=uncond_branch)
                raw = self._call_transformer(batched.as_transformer_kwargs())
                raw_uncond, raw_cond = split_batched_cfg_output(raw)
                raw_calls = 1
            elif self.runner.cfg_mode == "separate_cfg":
                raw_cond = self._call_transformer(cond_branch.as_transformer_kwargs())
                raw_uncond = self._call_transformer(uncond_branch.as_transformer_kwargs())
                raw_calls = 2
            else:
                raise ValueError("single_branch runner cannot run CFG")
            branch_calls = 2
            noise_pred_uncond = self.runner.postprocess_branch(
                request,
                uncond_branch,
                raw_uncond,
            )
        else:
            raw_cond = self._call_transformer(cond_branch.as_transformer_kwargs())
            raw_calls = 1
            noise_pred_uncond = None

        noise_pred_cond = self.runner.postprocess_branch(request, cond_branch, raw_cond)
        output_dtype = request.output_dtype or noise_pred_cond.dtype
        noise_pred_cond = noise_pred_cond.to(output_dtype)
        if noise_pred_uncond is None:
            noise_pred_uncond = torch.zeros_like(noise_pred_cond)
        else:
            noise_pred_uncond = noise_pred_uncond.to(output_dtype)

        combined = combine_cfg(
            noise_pred_cond,
            noise_pred_uncond,
            guidance_scale=request.guidance_scale,
            do_cfg=request.do_cfg,
            base=getattr(self.runner, "cfg_base", "uncond"),
        )
        noise_pred = self.runner.finalize_noise_pred(
            request,
            combined,
            noise_pred_cond,
            noise_pred_uncond,
        ).to(output_dtype)
        return DiffusionBackboneOutput(
            noise_pred=noise_pred,
            noise_pred_cond=noise_pred_cond,
            noise_pred_uncond=noise_pred_uncond,
            metrics={
                "branch_calls": branch_calls,
                "transformer_calls": raw_calls,
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            },
        )

    def _call_transformer(self, kwargs: dict[str, Any]) -> torch.Tensor:
        output = self.transformer(**kwargs)
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output[0]
