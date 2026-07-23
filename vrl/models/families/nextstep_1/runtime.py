"""NextStep-1 family runtime for Ray rollout workers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vrl.generation.bindings.token_autoregressive import (
    ARChunkExecutorBase,
    ARRequestLayout,
    ARSamplingParams,
)
from vrl.generation.composition.token_autoregressive.token_loop import TokenAutoregressiveLoop
from vrl.generation.execution.chunks import SampleChunk
from vrl.generation.types import (
    GenerationOutput,
    GenerationRequest,
    GenerationSampleRow,
)
from vrl.models.families.nextstep_1.runner import NextStep1ARModelRunner
from vrl.models.interfaces.runtime import ModelBuild
from vrl.models.steps.token.build import token_model_config_base
from vrl.trajectory import build_ar_continuous_trajectory

# NextStep LoRA defaults mirror the upstream recipe; applied at read time so the
# carried ``model.lora`` block only needs the values it overrides.
_NEXTSTEP_LORA_DEFAULTS: dict[str, Any] = {
    "rank": 32,
    "alpha": 64,
    "target_modules": ("q_proj", "v_proj"),
    "dropout": 0.0,
    "init": "gaussian",
}


def nextstep_config_from_build(build: ModelBuild) -> dict[str, Any]:
    model_config = build.model_config or {}
    sampling_config = build.sampling_config or {}
    config = token_model_config_base(build, _NEXTSTEP_LORA_DEFAULTS)

    for key in (
        "guidance_scale",
        "num_steps",
        "noise_level",
        "image_token_num",
        "token_dim",
    ):
        if key in sampling_config:
            config[key] = sampling_config[key]

    for key in ("vae_path", "vae_revision", "freeze_vae", "gradient_checkpointing"):
        value = model_config.get(key)
        if value is not None:
            config[key] = value

    return config


@dataclass(slots=True)
class NextStep1ARChunkResult:
    """Output of one prompt/sample NextStep-1 AR chunk."""

    prompt_index: int
    sample_start: int
    sample_count: int
    output: torch.Tensor
    tokens: torch.Tensor
    saved_noise: torch.Tensor
    log_probs: torch.Tensor
    prompt_input_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor
    uncond_input_ids: torch.Tensor
    uncond_attention_mask: torch.Tensor
    context: dict[str, Any]
    # Display/provenance-only: emitted through per-chunk runtime debug metrics.
    peak_memory_mb: float | None = None


class NextStep1ChunkExecutor(ARChunkExecutorBase):
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

    def __init__(
        self,
        model: Any,  # NextStep1Model
    ) -> None:
        self.model = model

    # -- protocol ------------------------------------------------------

    def forward_chunk_plan(
        self,
        request: GenerationRequest,
        chunk: SampleChunk,
    ) -> NextStep1ARChunkResult:
        """Run one prompt-major AR chunk through the black-box sampling path."""

        self.require_native_ar_engine(request)
        self.layout.validate_chunk(request, chunk)
        sampling = request.sampling
        params: ARSamplingParams = self.layout.parse_sampling_params(request)

        guidance_scale = float(sampling["guidance_scale"])
        num_steps = int(sampling["num_steps"])
        noise_level = float(sampling["noise_level"])

        repeated_prompts = [chunk.prompt] * chunk.sample_count
        prompt_ids, prompt_mask = self._tokenize_prompts(
            repeated_prompts,
            max_text_length=params.max_text_length,
        )
        uncond_ids, uncond_mask = self._tokenize_prompts(
            [""] * chunk.sample_count,
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
            generator.manual_seed(params.seed + self.layout.chunk_seed_offset(request, chunk))

        sample_kwargs: dict[str, Any] = {
            "guidance_scale": guidance_scale,
            "num_steps": num_steps,
            "noise_level": noise_level,
            "image_token_num": params.image_token_num,
        }
        if generator is not None:
            sample_kwargs["generator"] = generator

        decode_result = TokenAutoregressiveLoop(
            request=request,
            sample_rows=self.layout.chunk_sample_rows(request, chunk),
            runner=self._ar_runner(request),
            max_new_tokens=params.image_token_num,
            tokenizer_key="nextstep_1",
            dtype=str(cond_embeds.dtype),
            scheduler_batch_size=chunk.sample_count,
            init_args=(cond_embeds, uncond_embeds, prompt_mask, uncond_mask),
            init_kwargs=sample_kwargs,
            step_kwargs=sample_kwargs,
        ).run()
        tokens, saved_noise, old_logprobs = decode_result.finalized

        images = self.model.decode_image_tokens(tokens, image_size=params.image_size)
        peak_mem_mb = self.layout.peak_memory_mb()

        return NextStep1ARChunkResult(
            prompt_index=chunk.prompt_index,
            sample_start=chunk.sample_start,
            sample_count=chunk.sample_count,
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
                "image_token_num": params.image_token_num,
                "image_size": params.image_size,
            },
            peak_memory_mb=peak_mem_mb,
        )

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        chunks: Sequence[NextStep1ARChunkResult],
    ) -> GenerationOutput:
        return NextStep1ChunkGatherer().gather_chunks(request, sample_rows, chunks)

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


class NextStep1ChunkGatherer:
    """Pure driver-side gatherer for NextStep-1 AR chunk payloads."""

    layout = ARRequestLayout()

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        chunks: Sequence[NextStep1ARChunkResult],
    ) -> GenerationOutput:
        """Pack prompt/sample AR chunks back into the canonical GenerationOutput."""

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
        ordered_ar_chunks = self.layout.ordered_chunks(
            request,
            sample_rows,
            chunks,
            row_fields=fields,
        )
        cat = self.layout.cat_chunk_fields(ordered_ar_chunks, fields)
        trajectory_context = dict(ordered_ar_chunks[0].context)
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
            images_for_reward=cat["output"],
            context=trajectory_context,
        )

        return GenerationOutput(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            sample_rows=list(sample_rows),
            output=cat["output"],
            trajectory=trajectory,
            extra={},
        )


__all__ = [
    "NextStep1ARChunkResult",
    "NextStep1ChunkExecutor",
    "NextStep1ChunkGatherer",
    "nextstep_config_from_build",
]
