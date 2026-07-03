"""FLUX runner for shared diffusion backbone call execution.

FLUX.1-dev is guidance-distilled: a single transformer forward conditioned on a
``guidance`` embedding, NOT classifier-free guidance with an unconditional
branch. So the runner is ``single_branch`` and ``forward_step`` always sets
``do_cfg=False``; the shared caller then runs one ``cond`` branch and the
``guidance`` scalar rides through ``extra`` as an ordinary conditioning input.
"""

from __future__ import annotations

from typing import Literal


from vrl.models.diffusion.common import (
    DiffusionBackboneInput,
    DiffusionBackboneRunnerBase,
    DiffusionBranch,
)


class FluxDiffusionBackboneRunner(DiffusionBackboneRunnerBase):
    """Map FLUX transformer kwargs into the shared backbone contract.

    FLUX's MMDiT takes packed image latents plus rotary position ids
    (``img_ids`` / ``txt_ids``), a CLIP ``pooled_projections`` vector, and an
    optional ``guidance`` embedding. All of these ride through
    ``request.extra`` and are spread into the transformer call.
    """

    cfg_mode = "single_branch"
    cfg_base = "cond"

    def build_branch(
        self,
        request: DiffusionBackboneInput,
        branch: Literal["cond", "uncond"],
    ) -> DiffusionBranch:
        if branch != "cond":
            raise ValueError("FLUX is guidance-distilled and has no uncond branch")
        extra_kwargs = {
            "pooled_projections": request.extra["pooled_projections"],
            "img_ids": request.extra["img_ids"],
            "txt_ids": request.extra["txt_ids"],
        }
        # ``guidance`` is None for non-distilled checkpoints (config.guidance_embeds
        # off); pass it through either way so the transformer's own None-check fires.
        extra_kwargs["guidance"] = request.extra.get("guidance")
        return DiffusionBranch(
            hidden_states=request.hidden_states,
            timestep=request.timestep,
            encoder_hidden_states=request.prompt_embeds,
            extra_kwargs=extra_kwargs,
        )
