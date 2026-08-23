"""Diffusion transformer branch orchestration shared by family models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Protocol

import torch

from vrl.models.steps.denoise.common.cfg import (
    DiffusionBranch,
    DiffusionCFGBase,
    combine_cfg,
    pack_batched_cfg,
    split_batched_cfg_output,
)
from vrl.models.steps.denoise.common.tensors import require_tensor

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
    cfg_normalization: bool

    def build_branch(
        self,
        request: DiffusionBackboneInput,
        branch: Literal["cond", "uncond"],
    ) -> DiffusionBranch: ...

    def postprocess_branch(
        self,
        request: DiffusionBackboneInput,
        branch: DiffusionBranch,
        raw_output: torch.Tensor,
    ) -> torch.Tensor: ...

    def finalize_noise_pred(
        self,
        request: DiffusionBackboneInput,
        combined: torch.Tensor,
        cond: torch.Tensor,
        uncond: torch.Tensor,
    ) -> torch.Tensor: ...


class DiffusionBackboneRunnerBase:
    """No-op defaults for the two optional runner hooks.

    Every family runner must map kwargs in ``build_branch``, but most have
    nothing to do after the transformer call: ``postprocess_branch`` and
    ``finalize_noise_pred`` were byte-identical identity methods across
    sd3_5/flux/wan. They live here once; a runner overrides only when it does
    real math (cosmos predict2 converts the combined prediction back to the
    noise domain in ``finalize_noise_pred``).
    """

    # Norm-preserving CFG: rescale the combined prediction back to the
    # conditional branch's norm. Their reference pipelines make this a
    # family-level fact, not a per-step decision, so it selects a branch of
    # ``combine_cfg`` instead of an overridden method. lumina2 and qwen_image
    # turn it on; every other family does the plain linear combine.
    cfg_normalization: bool = False

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


class EncoderAttentionMaskRunnerBase(DiffusionBackboneRunnerBase):
    """``build_branch`` for families conditioned on embeds + an attention mask.

    sana, lumina2, mochi and pixart_sigma mapped their branches identically:
    the branch's sequence embeds as ``encoder_hidden_states`` and its padding
    mask as ``encoder_attention_mask``, with pixart_sigma's constant
    micro-conditioning dict the only addition.

    This is a SIBLING opt-in, not a default on ``DiffusionBackboneRunnerBase``:
    a family that forgets to map its own transformer kwargs must fail loud, so
    the base deliberately declares no ``build_branch``.
    """

    # Constant kwargs every branch of the family needs (pixart_sigma's
    # ``added_cond_kwargs``). Both branches must carry the SAME value — the
    # batched-CFG kwarg packer rejects branch-specific non-tensors.
    branch_extra_kwargs: ClassVar[Mapping[str, Any]] = {}

    def build_branch(
        self,
        request: DiffusionBackboneInput,
        branch: str,
    ) -> DiffusionBranch:
        """Map the branch's prompt embeds and attention mask into a branch call."""

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
            extra_kwargs={"encoder_attention_mask": mask, **self.branch_extra_kwargs},
        )


class DiffusionBackboneCaller:
    """Run one diffusion transformer step with shared CFG orchestration."""

    def __init__(self, transformer: Any, runner: DiffusionBackboneRunner) -> None:
        self.transformer = transformer
        self.runner = runner

    def __call__(self, request: DiffusionBackboneInput) -> DiffusionBackboneOutput:
        cond_branch = self.runner.build_branch(request, "cond")

        if request.do_cfg:
            uncond_branch = self.runner.build_branch(request, "uncond")
            if self.runner.cfg_mode == "batched_cfg":
                batched = pack_batched_cfg(cond=cond_branch, uncond=uncond_branch)
                raw = self._call_transformer(batched.as_transformer_kwargs())
                raw_uncond, raw_cond = split_batched_cfg_output(raw)
            elif self.runner.cfg_mode == "separate_cfg":
                raw_cond = self._call_transformer(cond_branch.as_transformer_kwargs())
                raw_uncond = self._call_transformer(uncond_branch.as_transformer_kwargs())
            else:
                raise ValueError("single_branch runner cannot run CFG")
            noise_pred_uncond = self.runner.postprocess_branch(
                request,
                uncond_branch,
                raw_uncond,
            )
        else:
            raw_cond = self._call_transformer(cond_branch.as_transformer_kwargs())
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
            base=self.runner.cfg_base,
            normalize=self.runner.cfg_normalization,
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
