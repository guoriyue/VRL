"""Emu3 family runtime for Ray rollout workers."""

from __future__ import annotations

from typing import Any

import torch

from vrl.generation.bindings.token_autoregressive import ARBatchInputs, ARDiscreteBatchExecutorBase
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.types import GenerationRequest
from vrl.models.families.emu3.model import (
    emu3_forced_token_schedule,
)
from vrl.models.families.emu3.runner import Emu3TokenRunner
from vrl.models.interfaces.runtime import ModelBuild
from vrl.models.steps.token.build import token_model_config_base


def emu3_config_from_build(build: ModelBuild) -> dict[str, Any]:
    sampling_config = build.sampling_config or {}
    config = token_model_config_base(build)

    for key in ("guidance_scale", "temperature", "image_area", "ratio"):
        if key in sampling_config:
            config[key] = sampling_config[key]

    return config


class Emu3BatchExecutor(ARDiscreteBatchExecutorBase):
    """AR executor for Emu3 text-to-image rollouts.

    The collector constructs a ``GenerationRequest`` whose ``sampling``
    dict holds:

    - ``guidance_scale``: float — classifier-free guidance scale.
    - ``temperature``: float — sampling temperature.
    - ``image_area``: int — target pixel area; the Emu3 processor derives
      the latent grid ``(height, width)`` from it (262144 -> 64x64 at 1:1).
    - ``ratio``: str — aspect ratio, e.g. ``"1:1"``.
    - ``max_text_length``: int — pad prompts to this length so ``L_text``
      is constant across multi-prompt requests (REQUIRED).
    - ``seed``: int | None — when set, ``torch.manual_seed`` is applied
      per batch for parity tests.

    Emu3 deliberately does NOT read ``image_token_num``/``image_size`` from
    sampling: the token count is fully determined by the latent grid
    (``h*(w+1) + 3`` including the forced EOL/EOF/EOI/EOS structural tokens),
    so a user-set knob would be dead or contradictory.

    The trajectory carries the sampled generation-vocab token ids, per-token
    conditional log-probs (GRPO's ``old_log_prob``), a token mask that zeroes
    the forced structural positions, prompt-side replay inputs, and a context
    with ``image_height``/``image_width`` (replay needs them to rebuild the
    structural mask).

    ``model`` is an ``Emu3Model``, or any stub exposing the same interface:
    ``processor``, ``device``, ``language_model``,
    ``encode_generation_prompts``, runner-step primitives, and
    ``decode_image_tokens``.
    """

    family: str = "emu3"
    _runner_cls = Emu3TokenRunner
    _runner_attention_family = "emu3"
    task: str = "ar_t2i"

    # -- protocol ------------------------------------------------------

    def prepare_batch_inputs(
        self,
        request: GenerationRequest,
        batch: GenerationSampleBatch,
    ) -> ARBatchInputs:
        """Encode cond+uncond prompts against the grid-derived token budget."""

        sampling = request.sampling

        guidance_scale = float(
            sampling.get("guidance_scale", self.model.config.guidance_scale),
        )
        temperature = float(
            sampling.get("temperature", self.model.config.temperature),
        )
        if "max_text_length" not in sampling:
            raise ValueError("request.sampling.max_text_length is required")
        max_text_length = int(sampling["max_text_length"])
        image_area = sampling.get("image_area")
        ratio = sampling.get("ratio")

        repeated_prompts = [request.inputs[batch.prompt_index].prompt] * batch.sample_count
        prompt_ids, prompt_mask, (height, width) = self.model.encode_generation_prompts(
            repeated_prompts,
            max_text_length=max_text_length,
            image_area=image_area,
            ratio=ratio,
        )
        uncond_ids, uncond_mask, uncond_grid = self.model.encode_generation_prompts(
            [""] * batch.sample_count,
            max_text_length=max_text_length,
            image_area=image_area,
            ratio=ratio,
        )
        if uncond_grid != (height, width):
            raise RuntimeError(
                f"Emu3 uncond grid {uncond_grid} != cond grid {(height, width)}",
            )
        pad_id = getattr(self.model.processor.tokenizer, "pad_token_id", None) or 0
        prompt_ids, prompt_mask, uncond_ids, uncond_mask = self.layout.align_pair(
            prompt_ids,
            prompt_mask,
            uncond_ids,
            uncond_mask,
            pad_id=pad_id,
        )

        cond_embeds = self._embed(prompt_ids)
        uncond_embeds = self._embed(uncond_ids)

        return ARBatchInputs(
            init_args=(cond_embeds, uncond_embeds, prompt_mask, uncond_mask),
            init_kwargs={
                "guidance_scale": guidance_scale,
                "temperature": temperature,
                "height": height,
                "width": width,
            },
            image_decode_kwargs={"height": height, "width": width},
            prompt_input_ids=prompt_ids,
            prompt_attention_mask=prompt_mask,
            uncond_input_ids=uncond_ids,
            uncond_attention_mask=uncond_mask,
            context={
                "temperature": temperature,
                "image_height": height,
                "image_width": width,
                # Display/provenance-only: OnlineTrainer writes the sampling
                # policy into its first-step ``rollout_context`` debug record.
                "guidance_scale": guidance_scale,
            },
        )

    def chunk_token_mask(
        self,
        inputs: ARBatchInputs,
        token_ids: torch.Tensor,
        token_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Mask Emu3's forced structural positions out of the trainable set.

        Forced positions (EOL/EOF/EOI/EOS) always renormalize to a single
        legal token (lp == 0 for old AND new policy), so mask them out
        instead of shipping constant terms.
        """

        forced = emu3_forced_token_schedule(
            int(inputs.context["image_height"]),
            int(inputs.context["image_width"]),
            self.model.image_vocab_size,
        ).to(token_log_probs.device)
        return (
            (forced < 0)
            .to(token_log_probs.dtype)
            .unsqueeze(0)
            .expand(token_ids.shape[0], -1)
            .contiguous()
        )


__all__ = [
    "Emu3BatchExecutor",
    "emu3_config_from_build",
]
