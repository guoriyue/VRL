"""Sampling state, collector boundary and rollout steps for masked-prompt families.

sana, lumina2, mochi and pixart_sigma all condition on ONE sequence embedding
plus its padding mask (no pooled vector), and carry the cond/uncond pair of
both through the trajectory. Their sampling-state fields, the three
collector-boundary methods, and the encode/prepare/forward rollout steps were
the same code; what genuinely differs per family is declared on the class
(pipeline encode kwargs, the transformer clock, the backbone output dtype,
which scheduler the rollout standardizes onto) and lives in a short override.
"""

from __future__ import annotations

import random
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from vrl.generation.types import DenoiseRequest
from vrl.models.steps.denoise.base import GuidedDenoiseSamplingStateBase
from vrl.models.steps.denoise.common.backbone import (
    DenoiseBackboneCaller,
    DenoiseBackboneInput,
)
from vrl.models.steps.denoise.common.timestep import expand_batch_timestep, pack_eval_timestep


@dataclass
class MaskedPromptSamplingState(GuidedDenoiseSamplingStateBase):
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


class MaskedPromptModelMixin(MaskedPromptCollectorMixin):
    """encode_prompt / prepare_sampling / forward_step for the masked-prompt families.

    The pipeline encodes cond and (under CFG) uncond sequence embeds plus
    masks; sampling draws the initial latents through ``pipe.prepare_latents``
    on a seeded generator; the forward runs the transformer through
    :class:`DenoiseBackboneCaller` with the masks as extra kwargs. A family
    declares the pipeline-specific encode kwargs and defaults on the class and
    overrides the small hooks below where its checkpoint differs.
    """

    # ``encode_prompt`` defaults when the caller passes no sampling values.
    _default_max_sequence_length: ClassVar[int]
    _default_guidance_scale: ClassVar[float]
    # Extra keyword arguments for ``pipe.encode_prompt`` (batch-count spelling,
    # template switches VRL pins off, ...).
    _pipeline_encode_kwargs: ClassVar[Mapping[str, Any]] = {}
    # Dtype the backbone output is promoted to before CFG; None keeps the
    # transformer dtype.
    _backbone_output_dtype: ClassVar[torch.dtype | None] = None

    # -- encode_prompt -------------------------------------------------

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode the prompt to sequence embeds + padding mask (no pooled vector)."""

        max_seq = kwargs.get("max_sequence_length", self._default_max_sequence_length)
        guidance_scale = kwargs.get("guidance_scale", self._default_guidance_scale)
        do_cfg = guidance_scale > 1.0
        neg = negative_prompt if negative_prompt is not None else ""

        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            negative_prompt=neg,
            do_classifier_free_guidance=do_cfg,
            device=self._encoder_device(),
            max_sequence_length=max_seq,
            **self._pipeline_encode_kwargs,
        )

        td = self.transformer.dtype
        result: dict[str, Any] = {
            "prompt_embeds": prompt_embeds.to(self.device, dtype=td),
            "prompt_attention_mask": (
                None if prompt_attention_mask is None else prompt_attention_mask.to(self.device)
            ),
        }
        if do_cfg and negative_prompt_embeds is not None:
            result["negative_prompt_embeds"] = negative_prompt_embeds.to(self.device, dtype=td)
            result["negative_prompt_attention_mask"] = (
                None
                if negative_prompt_attention_mask is None
                else negative_prompt_attention_mask.to(self.device)
            )
        return result

    # -- prepare_sampling ----------------------------------------------

    def _sampling_scheduler(self, request: DenoiseRequest) -> Any:
        """The scheduler the rollout steps on, with ``request.num_steps`` timesteps set.

        Default: the pipeline's own scheduler. Mochi and PixArt-Sigma
        standardize onto a rebuilt scheduler instead.
        """

        scheduler = self.pipeline.scheduler
        scheduler.set_timesteps(request.num_steps, device=self.device)
        return scheduler

    def _latent_shape_args(self, request: DenoiseRequest) -> tuple[Any, ...]:
        """Positional shape arguments ``pipe.prepare_latents`` takes after height/width."""

        del request
        return ()

    def prepare_sampling(
        self,
        request: DenoiseRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> MaskedPromptSamplingState:
        """Build the per-request SamplingState for a denoise loop."""

        del kwargs
        pipe = self.pipeline
        device = self.device

        prompt_embeds = encoded["prompt_embeds"]
        negative_prompt_embeds = encoded.get("negative_prompt_embeds")

        scheduler = self._sampling_scheduler(request)

        seed = request.seed if request.seed is not None else random.randint(0, sys.maxsize)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        latents = pipe.prepare_latents(
            prompt_embeds.shape[0],
            pipe.transformer.config.in_channels,
            request.height,
            request.width,
            *self._latent_shape_args(request),
            torch.float32,
            device,
            generator,
            None,
        )
        # No-op for every scheduler these families use (init_noise_sigma == 1.0);
        # kept for pipeline parity.
        latents = latents * scheduler.init_noise_sigma

        state_kwargs: dict[str, Any] = {
            "latents": latents,
            "timesteps": scheduler.timesteps,
            "scheduler": scheduler,
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": encoded.get("prompt_attention_mask"),
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_attention_mask": encoded.get("negative_prompt_attention_mask"),
            "guidance_scale": request.guidance_scale,
            "do_cfg": request.guidance_scale > 1.0 and negative_prompt_embeds is not None,
        }
        if issubclass(self.sampling_state_cls, TrainTimestepMaskedPromptSamplingState):
            state_kwargs["num_train_timesteps"] = int(scheduler.config.num_train_timesteps)
        return self.sampling_state_cls(**state_kwargs)

    # -- forward_step --------------------------------------------------

    def _backbone_timestep(
        self,
        timestep: torch.Tensor,
        state: MaskedPromptSamplingState,
    ) -> torch.Tensor:
        """Map the scheduler timestep (already batched, on the latent device) to the
        transformer's clock. Default: pass the raw timestep through."""

        del state
        return timestep

    def forward_step(
        self,
        state: MaskedPromptSamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        """Transformer forward + optional batched CFG."""

        t = state.timesteps[step_idx]
        bsz = state.latents.shape[0]
        td = self._transformer_dtype()

        latent_input = state.latents.to(td)
        timestep_batch = self._backbone_timestep(
            expand_batch_timestep(t, bsz).to(device=latent_input.device),
            state,
        )
        negative_embeds = (
            None if state.negative_prompt_embeds is None else state.negative_prompt_embeds.to(td)
        )
        output = DenoiseBackboneCaller(
            self.transformer,
            self,
        )(
            DenoiseBackboneInput(
                hidden_states=latent_input,
                timestep=timestep_batch,
                prompt_embeds=state.prompt_embeds.to(td),
                negative_prompt_embeds=negative_embeds,
                guidance_scale=state.guidance_scale,
                do_cfg=state.do_cfg,
                output_dtype=td
                if self._backbone_output_dtype is None
                else self._backbone_output_dtype,
                extra={
                    "encoder_attention_mask": state.prompt_attention_mask,
                    "negative_encoder_attention_mask": state.negative_prompt_attention_mask,
                },
            ),
        )
        return output.as_dict()


__all__ = [
    "MaskedPromptCollectorMixin",
    "MaskedPromptModelMixin",
    "MaskedPromptSamplingState",
    "TrainTimestepMaskedPromptSamplingState",
]
