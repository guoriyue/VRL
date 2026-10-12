"""Diffusion transformer branch orchestration shared by family models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from vrl.models.steps.denoise.common.cfg import (
    DenoiseBranch,
    DenoiseCFGBase,
    combine_cfg,
    pack_batched_cfg,
    split_batched_cfg_output,
)

DenoiseCFGMode = Literal["batched_cfg", "separate_cfg", "single_branch"]


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

        ``DenoiseBackboneRunnerBase.run_backbone`` builds the uncond branch if and only if
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


class DenoiseBackboneRunnerBase:
    """One diffusion transformer step with the shared CFG orchestration.

    A family runner declares ``cfg_mode`` and ``cfg_base`` and maps its
    conditioning in ``build_branch``. Most have nothing to do after the
    transformer call, so ``postprocess_branch`` and ``finalize_noise_pred``
    default to identities; a runner overrides one only when it does real math
    (cosmos predict2 converts the combined prediction back to the noise domain
    in ``finalize_noise_pred``).
    """

    cfg_mode: DenoiseCFGMode
    cfg_base: DenoiseCFGBase

    # Norm-preserving CFG: rescale the combined prediction back to the
    # conditional branch's norm. Their reference pipelines make this a
    # family-level fact, not a per-step decision, so it selects a branch of
    # ``combine_cfg`` instead of an overridden method. lumina2 and qwen_image
    # turn it on; every other family does the plain linear combine.
    cfg_normalization: bool = False

    def build_branch(
        self,
        request: DenoiseBackboneInput,
        branch: Literal["cond", "uncond"],
    ) -> DenoiseBranch:
        """Map the request onto the transformer kwargs of one CFG branch."""

        raise NotImplementedError(f"{type(self).__name__}.build_branch")

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

    def run_backbone(self, transformer: Any, request: DenoiseBackboneInput) -> dict[str, Any]:
        """``noise_pred`` plus the conditional and unconditional branch predictions."""

        def forward(branch: DenoiseBranch) -> torch.Tensor:
            output = transformer(**branch.as_transformer_kwargs())
            if isinstance(output, torch.Tensor):
                return output
            if isinstance(output, tuple):
                return output[0]
            if hasattr(output, "sample"):
                return output.sample
            return output[0]

        cond_branch = self.build_branch(request, "cond")

        if request.do_cfg:
            uncond_branch = self.build_branch(request, "uncond")
            if self.cfg_mode == "batched_cfg":
                batched = pack_batched_cfg(cond=cond_branch, uncond=uncond_branch)
                raw = forward(batched)
                raw_uncond, raw_cond = split_batched_cfg_output(raw)
            elif self.cfg_mode == "separate_cfg":
                raw_cond = forward(cond_branch)
                raw_uncond = forward(uncond_branch)
            else:
                raise ValueError("single_branch runner cannot run CFG")
            noise_pred_uncond = self.postprocess_branch(
                request,
                uncond_branch,
                raw_uncond,
            )
        else:
            raw_cond = forward(cond_branch)
            noise_pred_uncond = None

        noise_pred_cond = self.postprocess_branch(request, cond_branch, raw_cond)
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
            base=self.cfg_base,
            normalize=self.cfg_normalization,
        )
        noise_pred = self.finalize_noise_pred(
            request,
            combined,
            noise_pred_cond,
            noise_pred_uncond,
        ).to(output_dtype)
        return {
            "noise_pred": noise_pred,
            "noise_pred_cond": noise_pred_cond,
            "noise_pred_uncond": noise_pred_uncond,
        }
