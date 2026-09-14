"""Diffusion transformer branch orchestration shared by family models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import torch

from vrl.models.steps.denoise.common.cfg import (
    DenoiseBranch,
    DenoiseCFGBase,
    combine_cfg,
    pack_batched_cfg,
    split_batched_cfg_output,
)

DiffusionCFGMode = Literal["batched_cfg", "separate_cfg", "single_branch"]


@dataclass(slots=True)
class DenoiseBackboneInput:
    """Inputs for one denoise transformer call."""

    hidden_states: torch.Tensor
    timestep: torch.Tensor
    prompt_embeds: torch.Tensor
    guidance_scale: float
    do_cfg: bool
    # Present exactly when ``do_cfg`` is set — proven in __post_init__ below, so
    # no uncond branch re-checks it.
    negative_prompt_embeds: torch.Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    output_dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        """One home for the CFG conditioning invariant.

        ``DenoiseBackboneCaller`` builds the uncond branch if and only if
        ``do_cfg``, so a CFG request without negative conditioning is
        unusable. It used to be re-proven per family: every uncond branch
        raised on its own while four producers silently dropped the negatives
        they had just been asked to use. Rejecting the pair here means the
        state cannot be built, so no consumer re-checks it.
        """

        if self.do_cfg and self.negative_prompt_embeds is None:
            raise ValueError(
                "CFG requires negative_prompt_embeds; do_cfg=True was paired with None",
            )


@dataclass(slots=True)
class DenoiseBackboneOutput:
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


class DenoiseBackboneRunner(Protocol):
    """Family-owned backbone call runner."""

    cfg_mode: DiffusionCFGMode
    cfg_base: DenoiseCFGBase
    cfg_normalization: bool

    def build_branch(
        self,
        request: DenoiseBackboneInput,
        branch: Literal["cond", "uncond"],
    ) -> DenoiseBranch: ...

    def postprocess_branch(
        self,
        request: DenoiseBackboneInput,
        branch: DenoiseBranch,
        raw_output: torch.Tensor,
    ) -> torch.Tensor: ...

    def finalize_noise_pred(
        self,
        request: DenoiseBackboneInput,
        combined: torch.Tensor,
        cond: torch.Tensor,
        uncond: torch.Tensor,
    ) -> torch.Tensor: ...


class DenoiseBackboneRunnerBase:
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
        request: DenoiseBackboneInput,
        branch: DenoiseBranch,
        raw_output: torch.Tensor,
    ) -> torch.Tensor:
        del request, branch
        return raw_output

    def finalize_noise_pred(
        self,
        request: DenoiseBackboneInput,
        combined: torch.Tensor,
        cond: torch.Tensor,
        uncond: torch.Tensor,
    ) -> torch.Tensor:
        del request, cond, uncond
        return combined


class DenoiseBackboneCaller:
    """Run one diffusion transformer step with shared CFG orchestration."""

    def __init__(self, transformer: Any, runner: DenoiseBackboneRunner) -> None:
        self.transformer = transformer
        self.runner = runner

    def __call__(self, request: DenoiseBackboneInput) -> DenoiseBackboneOutput:
        cond_branch = self.runner.build_branch(request, "cond")

        if request.do_cfg:
            uncond_branch = self.runner.build_branch(request, "uncond")
            if self.runner.cfg_mode == "batched_cfg":
                batched = pack_batched_cfg(cond=cond_branch, uncond=uncond_branch)
                raw = self._forward_branch(batched)
                raw_uncond, raw_cond = split_batched_cfg_output(raw)
            elif self.runner.cfg_mode == "separate_cfg":
                raw_cond = self._forward_branch(cond_branch)
                raw_uncond = self._forward_branch(uncond_branch)
            else:
                raise ValueError("single_branch runner cannot run CFG")
            noise_pred_uncond = self.runner.postprocess_branch(
                request,
                uncond_branch,
                raw_uncond,
            )
        else:
            raw_cond = self._forward_branch(cond_branch)
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
        return DenoiseBackboneOutput(
            noise_pred=noise_pred,
            noise_pred_cond=noise_pred_cond,
            noise_pred_uncond=noise_pred_uncond,
        )

    def _forward_branch(self, branch: DenoiseBranch) -> torch.Tensor:
        """Invoke one prepared branch and extract its prediction tensor."""

        output = self.transformer(**branch.as_transformer_kwargs())
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output[0]
