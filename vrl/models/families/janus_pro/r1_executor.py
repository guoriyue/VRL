"""Janus-Pro-R1 AR text-to-image pipeline executor.

This executor owns generation only. Reward computation, advantage
normalization, and rollout packing stay outside the model family layer.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vrl.engine.ar import ARGenerationSpec, ordered_chunks
from vrl.engine.core.protocols import PipelineChunkResult
from vrl.engine.core.types import (
    GenerationMetrics,
    GenerationRequest,
    GenerationSampleSpec,
    OutputBatch,
)
from vrl.engine.microbatching import MicroBatchPlan
from vrl.models.families.janus_pro.executor import JanusProPipelineExecutor
from vrl.models.families.janus_pro.r1_types import JanusR1Segment

logger = logging.getLogger(__name__)

R1_SEGMENT_NAMES = ("initial_image", "selfcheck_text", "final_image")


@dataclass(slots=True)
class JanusProR1ChunkResult(PipelineChunkResult):
    """Output of one prompt/sample Janus-Pro-R1 chunk."""

    prompt_index: int
    sample_start: int
    sample_count: int
    output: torch.Tensor
    initial_image: torch.Tensor
    final_image: torch.Tensor
    selfcheck: torch.Tensor
    segments: dict[str, JanusR1Segment]
    context: dict[str, Any]
    peak_memory_mb: float | None = None


class JanusProR1PipelineExecutor(JanusProPipelineExecutor):
    """R1-style Janus-Pro executor for three-stage AR T2I generation."""

    family: str = "janus_pro_r1"
    task: str = "ar_t2i_r1"

    def forward(
        self,
        request: GenerationRequest,
        sample_specs: list[GenerationSampleSpec],
    ) -> OutputBatch:
        sampling = request.sampling
        spec: ARGenerationSpec = self.parse_spec(request)
        prompts = list(request.prompts)

        if spec.seed is not None:
            torch.manual_seed(spec.seed)

        repeated_prompts = self.expand_prompts(request)
        prompt_ids, prompt_mask, uncond_ids, uncond_mask = self._tokenize_r1_prompts(
            repeated_prompts,
            max_text_length=spec.max_text_length,
        )

        result = self.model.generate_with_refine(
            prompt_ids,
            prompt_mask,
            cfg_weight=float(sampling.get("cfg_weight", 5.0)),
            temperature=float(sampling.get("temperature", 1.0)),
            image_token_num=spec.image_token_num,
            max_reflect_len=int(sampling.get("max_reflect_len", 80)),
            task_stages=_parse_task_stages(sampling.get("task_stages")),
            uncond_input_ids=uncond_ids,
            uncond_attention_mask=uncond_mask,
            image_size=spec.image_size,
            refine_mode=_resolve_refine_mode(sampling, self.model),
        )

        peak_mem_mb = self.peak_memory_mb()
        segment_extra = _segments_to_extra(result.segments)
        metrics = GenerationMetrics(
            num_prompts=len(prompts),
            num_samples=len(sample_specs),
            num_steps=_segment_token_steps(segment_extra),
            micro_batches=1,
            peak_memory_mb=peak_mem_mb,
        )

        return OutputBatch(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=prompts,
            sample_specs=sample_specs,
            output=result.final_image,
            rollout_trajectory_data=None,
            extra={
                "initial_image": result.initial_image,
                "final_image": result.final_image,
                "selfcheck": result.selfcheck,
                "selfcheck_text": segment_extra["selfcheck_text"]["token_ids"],
                "segments": segment_extra,
                "context": result.context,
            },
            metrics=metrics,
            peak_memory_mb=peak_mem_mb or 0.0,
        )

    def forward_chunk(
        self,
        request: GenerationRequest,
        chunk: MicroBatchPlan,
    ) -> JanusProR1ChunkResult:
        self.validate_chunk(request, chunk)
        sampling = request.sampling
        spec: ARGenerationSpec = self.parse_spec(request)

        if spec.seed is not None:
            torch.manual_seed(spec.seed + self.chunk_seed_offset(request, chunk))

        repeated_prompts = [chunk.prompt] * chunk.sample_count
        prompt_ids, prompt_mask, uncond_ids, uncond_mask = self._tokenize_r1_prompts(
            repeated_prompts,
            max_text_length=spec.max_text_length,
        )

        result = self.model.generate_with_refine(
            prompt_ids,
            prompt_mask,
            cfg_weight=float(sampling.get("cfg_weight", 5.0)),
            temperature=float(sampling.get("temperature", 1.0)),
            image_token_num=spec.image_token_num,
            max_reflect_len=int(sampling.get("max_reflect_len", 80)),
            task_stages=_parse_task_stages(sampling.get("task_stages")),
            uncond_input_ids=uncond_ids,
            uncond_attention_mask=uncond_mask,
            image_size=spec.image_size,
            refine_mode=_resolve_refine_mode(sampling, self.model),
        )

        return JanusProR1ChunkResult(
            prompt_index=chunk.prompt_index,
            sample_start=chunk.sample_start,
            sample_count=chunk.sample_count,
            output=result.final_image,
            initial_image=result.initial_image,
            final_image=result.final_image,
            selfcheck=result.selfcheck,
            segments=result.segments,
            context=result.context,
            peak_memory_mb=self.peak_memory_mb(),
        )

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_specs: Sequence[GenerationSampleSpec],
        chunks: Sequence[JanusProR1ChunkResult],
    ) -> OutputBatch:
        return JanusProR1ChunkGatherer().gather_chunks(request, sample_specs, chunks)

    def _tokenize_r1_prompts(
        self,
        prompts: list[str],
        *,
        max_text_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_ids, prompt_mask = self._tokenize_prompts(
            prompts,
            max_text_length=max_text_length,
        )
        uncond_ids, uncond_mask = self._tokenize_prompts(
            [""] * len(prompts),
            max_text_length=max_text_length,
        )
        pad_id = getattr(self.model.processor.tokenizer, "pad_token_id", None) or 0
        return self.align_pair(
            prompt_ids,
            prompt_mask,
            uncond_ids,
            uncond_mask,
            pad_id=pad_id,
        )


class JanusProR1ChunkGatherer:
    """Driver-side gatherer for Janus-Pro-R1 chunk payloads."""

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_specs: Sequence[GenerationSampleSpec],
        chunks: Sequence[JanusProR1ChunkResult],
    ) -> OutputBatch:
        ordered = ordered_chunks(
            request,
            sample_specs,
            chunks,
            row_fields=(
                "output",
                "initial_image",
                "final_image",
                "selfcheck",
            ),
        )
        output = torch.cat([chunk.output for chunk in ordered], dim=0)
        initial_image = torch.cat([chunk.initial_image for chunk in ordered], dim=0)
        final_image = torch.cat([chunk.final_image for chunk in ordered], dim=0)
        selfcheck = torch.cat([chunk.selfcheck for chunk in ordered], dim=0)
        segment_extra = _cat_segment_extra(ordered)
        peak_mem_mb = self._max_peak_memory_mb(ordered)
        metrics = GenerationMetrics(
            num_prompts=len(request.prompts),
            num_samples=len(sample_specs),
            num_steps=_segment_token_steps(segment_extra),
            micro_batches=len(ordered),
            peak_memory_mb=peak_mem_mb,
        )

        return OutputBatch(
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=list(request.prompts),
            sample_specs=list(sample_specs),
            output=output,
            rollout_trajectory_data=None,
            extra={
                "initial_image": initial_image,
                "final_image": final_image,
                "selfcheck": selfcheck,
                "selfcheck_text": segment_extra["selfcheck_text"]["token_ids"],
                "segments": segment_extra,
                "context": dict(ordered[0].context),
            },
            metrics=metrics,
            peak_memory_mb=peak_mem_mb or 0.0,
        )

    @staticmethod
    def _max_peak_memory_mb(
        chunks: Sequence[JanusProR1ChunkResult],
    ) -> float | None:
        peaks = [chunk.peak_memory_mb for chunk in chunks if chunk.peak_memory_mb is not None]
        return max(peaks) if peaks else None


def _parse_task_stages(value: Any) -> tuple[str, ...]:
    if value is None:
        return R1_SEGMENT_NAMES
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(part) for part in value)


def _resolve_refine_mode(sampling: dict[str, Any], model: Any) -> str:
    policy = sampling.get("final_image_policy")
    if policy == "always_generate":
        return "always"
    if policy == "use_selfcheck":
        return "selfcheck"
    return str(
        sampling.get(
            "refine_mode",
            getattr(getattr(model, "config", None), "r1_refine_mode", "selfcheck"),
        )
    )


def _segment_to_extra(segment: JanusR1Segment) -> dict[str, Any]:
    return {
        "name": segment.name,
        "token_ids": segment.token_ids,
        "token_log_probs": segment.token_log_probs,
        "token_mask": segment.token_mask,
        "prompt_embeds": segment.prompt_embeds,
        "attention_mask": segment.attention_mask,
        "prompt_attention_mask": segment.attention_mask,
        "visual": segment.visual,
        "cfg": segment.cfg,
    }


def _segments_to_extra(
    segments: dict[str, JanusR1Segment],
) -> dict[str, dict[str, Any]]:
    return {name: _segment_to_extra(segment) for name, segment in segments.items()}


def _cat_segment_extra(
    chunks: Sequence[JanusProR1ChunkResult],
) -> dict[str, dict[str, Any]]:
    names = tuple(chunks[0].segments)
    if set(names) != set(R1_SEGMENT_NAMES):
        logger.warning("Unexpected Janus-Pro-R1 segment names: %s", names)

    out: dict[str, dict[str, Any]] = {}
    for name in names:
        first = chunks[0].segments[name]
        token_log_probs = None
        if first.token_log_probs is not None:
            token_log_probs = torch.cat(
                [chunk.segments[name].token_log_probs for chunk in chunks],
                dim=0,
            )
        out[name] = {
            "name": name,
            "token_ids": torch.cat(
                [chunk.segments[name].token_ids for chunk in chunks],
                dim=0,
            ),
            "token_log_probs": token_log_probs,
            "token_mask": torch.cat(
                [chunk.segments[name].token_mask for chunk in chunks],
                dim=0,
            ),
            "prompt_embeds": torch.cat(
                [chunk.segments[name].prompt_embeds for chunk in chunks],
                dim=0,
            ),
            "attention_mask": torch.cat(
                [chunk.segments[name].attention_mask for chunk in chunks],
                dim=0,
            ),
            "prompt_attention_mask": torch.cat(
                [chunk.segments[name].attention_mask for chunk in chunks],
                dim=0,
            ),
            "visual": first.visual,
            "cfg": first.cfg,
        }
    return out


def _segment_token_steps(segments: dict[str, dict[str, Any]]) -> int:
    return sum(int(segment["token_ids"].shape[1]) for segment in segments.values())


__all__ = [
    "JanusProR1ChunkGatherer",
    "JanusProR1ChunkResult",
    "JanusProR1PipelineExecutor",
]
