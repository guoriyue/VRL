"""Sampling state and collector boundary for masked-prompt diffusion families.

sana, lumina2, mochi and pixart_sigma all condition on ONE sequence embedding
plus its padding mask (no pooled vector), and carry the cond/uncond pair of
both through the trajectory. Their sampling-state fields and all three
collector-boundary methods were byte-identical; the only genuine difference is
whether the family also needs the scheduler's ``num_train_timesteps`` to
rebuild the transformer clock on the replay path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from vrl.models.steps.denoise.base import GuidedDiffusionSamplingStateBase
from vrl.models.steps.denoise.common.timestep import pack_eval_timestep


@dataclass
class MaskedPromptSamplingState(GuidedDiffusionSamplingStateBase):
    """Private state for (sequence embeds, attention mask) conditioning.

    Engine MUST NOT introspect: ``latents`` / ``timesteps`` / ``scheduler`` are
    the only fields the batch executor may touch.
    """

    prompt_embeds: torch.Tensor
    prompt_attention_mask: torch.Tensor | None
    negative_prompt_embeds: torch.Tensor | None
    negative_prompt_attention_mask: torch.Tensor | None
    do_cfg: bool


@dataclass
class TrainTimestepMaskedPromptSamplingState(MaskedPromptSamplingState):
    """Adds the training-clock length the transformer's timestep is derived from.

    lumina2 feeds the transformer ``1 - t / num_train_timesteps`` and mochi
    feeds ``num_train_timesteps - t``. Replay rebuilds one packed step without
    a scheduler, so the constant has to travel in the batch context instead of
    being read back off ``scheduler.config``.
    """

    num_train_timesteps: int


class MaskedPromptCollectorMixin:
    """Trajectory projection shared by the masked-prompt families.

    ``sampling_state_cls`` is the only per-family knob: the mixin exports the
    tensors the state declares and rebuilds that same class on the replay path.
    Masks and negative embeds are exported only when present, so the no-CFG
    path stays tensor-free and restore reads them back with ``.get``.
    """

    sampling_state_cls: ClassVar[type[MaskedPromptSamplingState]]

    def export_batch_context(self, state: MaskedPromptSamplingState) -> dict[str, Any]:
        """Project sampling state into shared trajectory context."""

        context: dict[str, Any] = {
            "guidance_scale": state.guidance_scale,
            "cfg": state.do_cfg,
        }
        if isinstance(state, TrainTimestepMaskedPromptSamplingState):
            context["num_train_timesteps"] = state.num_train_timesteps
        return context

    def export_replay_tensors(self, state: MaskedPromptSamplingState) -> dict[str, Any]:
        """Project sampling state into per-sample trajectory tensors."""

        tensors: dict[str, Any] = {"prompt_embeds": state.prompt_embeds}
        for name in (
            "prompt_attention_mask",
            "negative_prompt_embeds",
            "negative_prompt_attention_mask",
        ):
            value = getattr(state, name)
            if value is not None:
                tensors[name] = value
        return tensors

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> MaskedPromptSamplingState:
        """Rebuild the family sampling state from a batch slice for eval forward."""

        state_cls = self.sampling_state_cls
        state_kwargs: dict[str, Any] = {
            "latents": latents,
            "timesteps": pack_eval_timestep(replay_tensors["timesteps"], step_idx),
            # forward_step never calls scheduler.step, so replay needs none.
            "scheduler": None,
            "prompt_embeds": replay_tensors["prompt_embeds"],
            "prompt_attention_mask": replay_tensors.get("prompt_attention_mask"),
            "negative_prompt_embeds": replay_tensors.get("negative_prompt_embeds"),
            "negative_prompt_attention_mask": replay_tensors.get(
                "negative_prompt_attention_mask",
            ),
            "guidance_scale": batch_context["guidance_scale"],
            "do_cfg": batch_context["cfg"]
            and replay_tensors.get("negative_prompt_embeds") is not None,
        }
        if issubclass(state_cls, TrainTimestepMaskedPromptSamplingState):
            state_kwargs["num_train_timesteps"] = int(batch_context["num_train_timesteps"])
        return state_cls(**state_kwargs)


__all__ = [
    "MaskedPromptCollectorMixin",
    "MaskedPromptSamplingState",
    "TrainTimestepMaskedPromptSamplingState",
]
