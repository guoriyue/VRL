"""Wan 2.1 runner for shared diffusion backbone call execution.

Two runners live here:

- :class:`WanDiffusionBackboneRunner` — T2V runner.
- :class:`WanI2VDiffusionBackboneRunner` — Image-to-Video runner that adds
  ``encoder_hidden_states_image`` (CLIP visual features) and channel-wise
  ``[noise_latents, condition]`` concat hidden_states. The conditioning
  tensor + image embeds are passed via :class:`DiffusionBackboneInput.extra`
  so the shared backbone caller can keep its uniform signature.
"""

from __future__ import annotations

from typing import Literal

import torch

from vrl.models.diffusion.common import (
    DiffusionBackboneInput,
    DiffusionBranch,
)
from vrl.models.diffusion.common.tensors import require_tensor


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
            embeds = require_tensor(request.negative_prompt_embeds, "Wan negative_prompt_embeds")
        return DiffusionBranch(
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


class WanI2VDiffusionBackboneRunner:
    """Map Wan 2.1 I2V transformer kwargs into the shared backbone contract.

    Differences from T2V runner:

    - Hidden states are ``cat([noise_latents, condition], dim=1)`` so the
      transformer sees both the noisy frames and the (mask + first-frame
      latent) conditioning channel-wise concatenated. The condition tensor
      is produced once per request by ``pipe.prepare_latents`` and stored
      in ``request.extra``.
    - ``encoder_hidden_states_image`` is the CLIP visual feature produced by
      ``pipe.encode_image``; cond/uncond share the same tensor object so
      the batched-CFG packer concatenates it along dim 0 alongside
      ``prompt_embeds``.

    Note on Wan 2.2 expand_timesteps mode: the runner supports the A14B
    dual-stage path, where routing selects ``transformer`` or
    ``transformer_2`` outside the runner. Wan 2.2 5B with expand_timesteps
    blends ``(1 - first_frame_mask) * condition + first_frame_mask * latents``
    into hidden_states; that branch remains a separate future upgrade path.
    """

    cfg_mode = "batched_cfg"
    cfg_base = "uncond"

    def build_branch(
        self,
        request: DiffusionBackboneInput,
        branch: Literal["cond", "uncond"],
    ) -> DiffusionBranch:
        embeds = request.prompt_embeds
        if branch == "uncond":
            embeds = require_tensor(request.negative_prompt_embeds, "Wan negative_prompt_embeds")

        extra = request.extra
        condition = _batch_align_tensor(
            require_tensor(extra.get("condition"), "Wan condition"),
            request.hidden_states.shape[0],
        )
        # Channel-wise concat (mimics WanImageToVideoPipeline `__call__`:
        # `latent_model_input = torch.cat([latents, condition], dim=1)`).
        hidden_states = torch.cat(
            [request.hidden_states, condition.to(request.hidden_states.dtype)],
            dim=1,
        )

        extra_kwargs: dict[str, torch.Tensor] = {}
        image_embeds = extra.get("image_embeds")
        if image_embeds is not None:
            extra_kwargs["encoder_hidden_states_image"] = _batch_align_tensor(
                require_tensor(image_embeds, "Wan image_embeds"),
                request.hidden_states.shape[0],
            )

        return DiffusionBranch(
            hidden_states=hidden_states,
            timestep=request.timestep,
            encoder_hidden_states=embeds,
            extra_kwargs=extra_kwargs,
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


def _batch_align_tensor(value: torch.Tensor, batch_size: int) -> torch.Tensor:
    if value.shape[0] == batch_size:
        return value
    if value.shape[0] != 1:
        raise ValueError(
            "Wan I2V conditioning tensor has incompatible batch dimension: "
            f"{tuple(value.shape)} for batch_size={batch_size}",
        )
    return value.expand(batch_size, *value.shape[1:])
