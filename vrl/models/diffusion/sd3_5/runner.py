"""SD3.5 runner for shared diffusion backbone call execution."""

from __future__ import annotations

from typing import Literal


from vrl.models.diffusion.common import (
    DiffusionBackboneInput,
    DiffusionBackboneRunnerBase,
    DiffusionBranch,
)
from vrl.models.diffusion.common.tensors import require_tensor
from vrl.nn.layers.attention.joint import SD3JointAttentionProcessor


def install_sd3_joint_attention_processor(transformer: object) -> bool:
    """Install VRL SD3 attention processor when the transformer supports it."""

    for candidate in _candidate_transformers(transformer):
        set_attn_processor = getattr(candidate, "set_attn_processor", None)
        if callable(set_attn_processor):
            set_attn_processor(SD3JointAttentionProcessor())
            return True
    return False


def _candidate_transformers(transformer: object) -> tuple[object, ...]:
    candidates: list[object] = []
    stack = [transformer]
    seen: set[int] = set()
    while stack:
        item = stack.pop(0)
        marker = id(item)
        if marker in seen:
            continue
        seen.add(marker)
        candidates.append(item)
        for attr in ("module", "base_model", "model"):
            child = getattr(item, attr, None)
            if child is not None:
                stack.append(child)
    return tuple(candidates)


class SD3DiffusionBackboneRunner(DiffusionBackboneRunnerBase):
    """Map SD3 transformer kwargs into the shared backbone contract."""

    cfg_mode = "batched_cfg"
    cfg_base = "uncond"

    def build_branch(
        self,
        request: DiffusionBackboneInput,
        branch: Literal["cond", "uncond"],
    ) -> DiffusionBranch:
        if branch == "cond":
            embeds = request.prompt_embeds
            pooled = request.extra["pooled_prompt_embeds"]
        else:
            embeds = require_tensor(request.negative_prompt_embeds, "negative_prompt_embeds")
            pooled = require_tensor(
                request.extra.get("negative_pooled_prompt_embeds"),
                "negative_pooled_prompt_embeds",
            )
        return DiffusionBranch(
            hidden_states=request.hidden_states,
            timestep=request.timestep,
            encoder_hidden_states=embeds,
            extra_kwargs={"pooled_projections": pooled},
        )
