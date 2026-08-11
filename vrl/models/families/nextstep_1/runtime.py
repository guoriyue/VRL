"""NextStep-1 family runtime for Ray rollout workers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vrl.generation.bindings.token_autoregressive import (
    ARBatchExecutorBase,
    ARRequestLayout,
    ARSamplingParams,
)
from vrl.generation.composition.token_autoregressive.token_loop import TokenAutoregressiveLoop
from vrl.generation.execution.sample_batches import (
    GenerationSampleBatch,
    require_matching_batch_context,
)
from vrl.generation.types import (
    GenerationOutput,
    GenerationRequest,
    GenerationSampleRow,
)
from vrl.models.families.nextstep_1.runner import NextStep1ARModelRunner
from vrl.models.interfaces.runtime import ModelBuild
from vrl.models.steps.token.build import token_model_config_base
from vrl.trajectory import build_ar_continuous_trajectory
from vrl.utils.cuda_memory import cuda_peak_allocated_mb


def nextstep_config_from_build(build: ModelBuild) -> dict[str, Any]:
    model_config = build.model_config or {}
    sampling_config = build.sampling_config or {}
    config = token_model_config_base(build)

    for key in (
        "guidance_scale",
        "num_steps",
        "image_token_num",
    ):
        if key in sampling_config:
            config[key] = sampling_config[key]

    for key in ("vae_path", "vae_revision", "freeze_vae", "gradient_checkpointing"):
        value = model_config.get(key)
        if value is not None:
            config[key] = value

    return config


@dataclass(slots=True)
class NextStep1ARBatchResult:
    """Output of one prompt/sample NextStep-1 AR batch."""

    batch: GenerationSampleBatch
    output: torch.Tensor
    tokens: torch.Tensor
    saved_noise: torch.Tensor
    log_probs: torch.Tensor
    prompt_input_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor
    uncond_input_ids: torch.Tensor
    uncond_attention_mask: torch.Tensor
    context: dict[str, Any]
    # Display/provenance-only: emitted through per-batch runtime debug metrics.
    peak_memory_mb: float | None = None


class NextStep1BatchExecutor(ARBatchExecutorBase):
    """Continuous-token AR executor for NextStep-1 text-to-image rollouts.

    The collector constructs a ``GenerationRequest`` whose ``sampling``
    dict holds:

    - ``guidance_scale``: float
    - ``num_steps``: int
    - ``noise_level``: float
    - ``image_token_num``: int (L_img — number of continuous image tokens)
    - ``image_size``: int (passed to ``decode_image_tokens``)
    - ``max_text_length``: int
    - ``seed``: int | None

    And whose ``metadata`` may carry ``rollout_metadata`` (target_text,
    references, etc.) for the collector's reward layer.

    The executor returns an ``GenerationOutput`` whose first-class trajectory carries
    tokens, saved noise, log-probs, decoded reward images, and prompt-side replay
    context.
    """

    family: str = "nextstep_1"
    _runner_cls = NextStep1ARModelRunner
    _runner_attention_family = "nextstep_1"
    task: str = "ar_t2i"
    default_image_token_num: int | None = None
    default_image_size: int | None = None

    # -- protocol ------------------------------------------------------

    def forward_batch(
        self,
        request: GenerationRequest,
        batch: GenerationSampleBatch,
    ) -> NextStep1ARBatchResult:
        """Run one prompt-major AR batch through the black-box sampling path."""

        self.require_native_ar_engine(request)
        self.layout.validate_chunk(request, batch)
        scheduler_batch_size = self.resolve_scheduler_batch_size(
            request,
            row_count=batch.sample_count,
        )
        sampling = request.sampling
        params: ARSamplingParams = self.layout.parse_sampling_params(request)

        guidance_scale = float(sampling["guidance_scale"])
        num_steps = int(sampling["num_steps"])
        noise_level = float(sampling["noise_level"])

        repeated_prompts = [request.inputs[batch.prompt_index].prompt] * batch.sample_count
        prompt_ids, prompt_mask = self._tokenize_prompts(
            repeated_prompts,
            max_text_length=params.max_text_length,
        )
        uncond_ids, uncond_mask = self._tokenize_prompts(
            [""] * batch.sample_count,
            max_text_length=params.max_text_length,
        )
        pad_id = getattr(self.model.processor, "pad_token_id", None) or 0
        prompt_ids, prompt_mask, uncond_ids, uncond_mask = self.layout.align_pair(
            prompt_ids,
            prompt_mask,
            uncond_ids,
            uncond_mask,
            pad_id=pad_id,
        )

        cond_embeds = self._embed(prompt_ids)
        uncond_embeds = self._embed(uncond_ids)

        generator: torch.Generator | None = None
        if params.seed is not None:
            generator = torch.Generator(device=self.model.device)
            generator.manual_seed(
                params.seed + self.layout.chunk_seed_offset(request, batch),
            )

        sample_kwargs: dict[str, Any] = {
            "guidance_scale": guidance_scale,
            "num_steps": num_steps,
            "noise_level": noise_level,
            "image_token_num": params.image_token_num,
        }
        if generator is not None:
            sample_kwargs["generator"] = generator

        tokens, saved_noise, old_logprobs = TokenAutoregressiveLoop(
            runner=self._ar_runner(request),
            scheduler_batch_size=scheduler_batch_size,
            init_args=(cond_embeds, uncond_embeds, prompt_mask, uncond_mask),
            init_kwargs=sample_kwargs,
        ).run()

        images = self.model.decode_image_tokens(tokens, image_size=params.image_size)

        return NextStep1ARBatchResult(
            batch=batch,
            output=images,
            tokens=tokens,
            saved_noise=saved_noise,
            log_probs=old_logprobs,
            prompt_input_ids=prompt_ids,
            prompt_attention_mask=prompt_mask,
            uncond_input_ids=uncond_ids,
            uncond_attention_mask=uncond_mask,
            context={
                "guidance_scale": guidance_scale,
                "num_steps": num_steps,
                "noise_level": noise_level,
            },
            peak_memory_mb=cuda_peak_allocated_mb(),
        )

    # -- internals -----------------------------------------------------

    def _tokenize_prompts(
        self,
        prompts: list[str],
        *,
        max_text_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenise via the upstream NextStep tokenizer.

        Mirrors ``NextStep1Collector._tokenize_prompts`` exactly so the
        old direct path and the engine path produce bitwise-identical
        ``input_ids``/``attention_mask`` pairs.
        """
        tok = self.model.processor
        device = self.model.device

        enc = tok(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_text_length,
        )
        ids = enc["input_ids"]
        mask = enc["attention_mask"]
        ids, mask = self._align_tokenizer_output(
            ids,
            mask,
            max_text_length=max_text_length,
            pad_id=getattr(tok, "pad_token_id", None) or 0,
        )
        return ids.to(device), mask.to(device)


class NextStep1GenerationBatchGatherer:
    """Pure driver-side gatherer for NextStep-1 AR batch payloads."""

    layout = ARRequestLayout()

    def gather_batches(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        batches: Sequence[NextStep1ARBatchResult],
    ) -> GenerationOutput:
        """Pack prompt/sample AR batches back into the canonical GenerationOutput."""

        fields = (
            "output",
            "tokens",
            "saved_noise",
            "log_probs",
            "prompt_input_ids",
            "prompt_attention_mask",
            "uncond_input_ids",
            "uncond_attention_mask",
        )
        ordered_ar_chunks = self.layout.ordered_batches(
            request,
            sample_rows,
            batches,
            row_fields=fields,
        )
        cat = self.layout.cat_batch_fields(ordered_ar_chunks, fields)
        trajectory_context = require_matching_batch_context(
            [batch.context for batch in ordered_ar_chunks],
        )
        trajectory = build_ar_continuous_trajectory(
            request=request,
            sample_rows=list(sample_rows),
            tokens=cat["tokens"],
            saved_noise=cat["saved_noise"],
            token_log_probs=cat["log_probs"],
            token_mask=torch.ones_like(cat["log_probs"]),
            prompt_input_ids=cat["prompt_input_ids"],
            prompt_attention_mask=cat["prompt_attention_mask"],
            uncond_input_ids=cat["uncond_input_ids"],
            uncond_attention_mask=cat["uncond_attention_mask"],
            context=trajectory_context,
        )

        return GenerationOutput(
            output=cat["output"],
            trajectory=trajectory,
        )


__all__ = [
    "NextStep1ARBatchResult",
    "NextStep1BatchExecutor",
    "NextStep1GenerationBatchGatherer",
    "nextstep_config_from_build",
]
