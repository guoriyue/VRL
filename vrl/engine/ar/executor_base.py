"""Shared scaffolding for autoregressive generation executors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

import torch

from vrl.engine.ar.spec import ARGenerationSpec
from vrl.engine.core.protocols import (
    ChunkedFamilyPipelineExecutor,
    PipelineChunkResult,
)
from vrl.engine.core.types import (
    GenerationRequest,
    GenerationSampleSpec,
    OutputBatch,
)
from vrl.engine.execution.batching import forward_batch_by_merging_prompts
from vrl.engine.execution.microbatching import MicroBatchPlan


class ARChunkResult(PipelineChunkResult, Protocol):
    """Common metadata every prompt-major AR chunk result carries."""

    prompt_index: int
    sample_start: int
    sample_count: int
    peak_memory_mb: float | None


TChunk = TypeVar("TChunk", bound=ARChunkResult)


@dataclass(frozen=True, slots=True)
class ARRequestLayout:
    """Prompt-major request layout shared by AR executors and gatherers."""

    default_image_token_num: int | None = 576
    default_image_size: int | None = 384
    default_max_text_length: int | None = 256

    def parse_spec(self, request: GenerationRequest) -> ARGenerationSpec:
        """Parse family-neutral AR fields from ``GenerationRequest.sampling``."""

        sampling = request.sampling
        return ARGenerationSpec(
            image_token_num=self._sampling_int(
                sampling,
                "image_token_num",
                self.default_image_token_num,
            ),
            image_size=self._sampling_int(
                sampling,
                "image_size",
                self.default_image_size,
            ),
            max_text_length=self._sampling_int(
                sampling,
                "max_text_length",
                self.default_max_text_length,
            ),
            seed=self._optional_int(sampling.get("seed")),
            use_ar_scheduler=bool(sampling.get("use_ar_scheduler", False)),
            ar_scheduler_batch_size=self._optional_int(
                sampling.get("ar_scheduler_batch_size"),
            ),
        )

    def expand_prompts(self, request: GenerationRequest) -> list[str]:
        """Repeat prompts in the same prompt-major order as sample specs."""

        samples_per_prompt = int(request.samples_per_prompt)
        return [prompt for prompt in request.prompts for _ in range(samples_per_prompt)]

    def validate_chunk(self, request: GenerationRequest, chunk: MicroBatchPlan) -> None:
        """Validate one prompt/sample AR chunk against its request."""

        if chunk.prompt_index >= len(request.prompts):
            raise ValueError(
                f"chunk.prompt_index={chunk.prompt_index} is out of range",
            )
        if chunk.prompt != request.prompts[chunk.prompt_index]:
            raise ValueError(
                "chunk.prompt does not match request.prompts[chunk.prompt_index]",
            )
        if chunk.sample_end > request.samples_per_prompt:
            raise ValueError(
                "chunk sample range exceeds request.samples_per_prompt: "
                f"{chunk.sample_start}:{chunk.sample_end} > "
                f"{request.samples_per_prompt}",
            )

    def ordered_chunks(
        self,
        request: GenerationRequest,
        sample_specs: Sequence[GenerationSampleSpec],
        chunks: Sequence[TChunk],
        *,
        row_fields: Sequence[str] = (),
    ) -> list[TChunk]:
        """Sort AR chunks and ensure they exactly cover prompt-major samples."""

        if not chunks:
            raise ValueError("chunks must be non-empty")

        ordered = sorted(
            chunks,
            key=lambda chunk: (
                int(chunk.prompt_index),
                int(chunk.sample_start),
            ),
        )
        expected = [(spec.prompt_index, spec.sample_index) for spec in sample_specs]
        actual: list[tuple[int, int]] = []
        for chunk in ordered:
            prompt_index = int(chunk.prompt_index)
            sample_start = int(chunk.sample_start)
            sample_count = int(chunk.sample_count)
            self._validate_chunk_range(
                request,
                prompt_index=prompt_index,
                sample_start=sample_start,
                sample_count=sample_count,
            )
            for field_name in row_fields:
                self.require_rows(field_name, getattr(chunk, field_name), sample_count)
            actual.extend(
                (prompt_index, sample_index)
                for sample_index in range(sample_start, sample_start + sample_count)
            )
        if actual != expected:
            raise ValueError(
                "AR chunks do not cover sample_specs in prompt-major sample order",
            )
        return ordered

    def require_rows(self, name: str, value: Any, count: int) -> None:
        """Require a chunk tensor-like payload to have ``count`` batch rows."""

        shape = getattr(value, "shape", None)
        if shape is None or len(shape) < 1:
            raise ValueError(f"chunk {name} must have a leading batch dimension")
        if int(shape[0]) != count:
            raise ValueError(
                f"chunk {name} has {shape[0]} rows, expected {count}",
            )

    def chunk_seed_offset(self, request: GenerationRequest, chunk: MicroBatchPlan) -> int:
        """Return the prompt-major sample offset for deterministic chunk seeding."""

        return chunk.prompt_index * int(request.samples_per_prompt) + chunk.sample_start

    def chunk_sample_specs(
        self,
        request: GenerationRequest,
        chunk: MicroBatchPlan,
    ) -> list[GenerationSampleSpec]:
        """Build prompt-major sample specs for an AR microbatch chunk."""

        return [
            GenerationSampleSpec(
                prompt_index=chunk.prompt_index,
                sample_index=sample_index,
                prompt=chunk.prompt,
                prompt_id=f"{request.request_id}:prompt:{chunk.prompt_index}",
                group_id=f"{request.request_id}:group:{chunk.prompt_index}",
                sample_id=(
                    f"{request.request_id}:sample:"
                    f"{chunk.prompt_index}:{sample_index}"
                ),
                trajectory_id=(
                    f"{request.request_id}:trajectory:"
                    f"{chunk.prompt_index}:{sample_index}"
                ),
                seed=self._optional_int(request.sampling.get("seed")),
                metadata={},
            )
            for sample_index in range(chunk.sample_start, chunk.sample_end)
        ]

    def max_peak_memory_mb(self, chunks: Sequence[ARChunkResult]) -> float | None:
        """Return the maximum non-null peak memory metric across chunk results."""

        peaks = [
            chunk.peak_memory_mb
            for chunk in chunks
            if chunk.peak_memory_mb is not None
        ]
        return max(peaks) if peaks else None

    def align_pair(
        self,
        a_ids: torch.Tensor,
        a_mask: torch.Tensor,
        b_ids: torch.Tensor,
        b_mask: torch.Tensor,
        pad_id: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Right-pad two ``[B, L]`` token/mask pairs to a common length."""

        target_length = max(a_ids.shape[1], b_ids.shape[1])

        def _pad(
            ids: torch.Tensor,
            mask: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            current_length = ids.shape[1]
            if current_length == target_length:
                return ids, mask
            extra_len = target_length - current_length
            pad_ids = torch.full(
                (ids.shape[0], extra_len),
                pad_id,
                dtype=ids.dtype,
                device=ids.device,
            )
            pad_mask = torch.zeros(
                (mask.shape[0], extra_len),
                dtype=mask.dtype,
                device=mask.device,
            )
            return (
                torch.cat([ids, pad_ids], dim=1),
                torch.cat([mask, pad_mask], dim=1),
            )

        a_ids, a_mask = _pad(a_ids, a_mask)
        b_ids, b_mask = _pad(b_ids, b_mask)
        return a_ids, a_mask, b_ids, b_mask

    def peak_memory_mb(self) -> float | None:
        """Read CUDA peak allocated memory in MiB, when available."""

        if not torch.cuda.is_available():
            return None
        try:
            peak_bytes = torch.cuda.max_memory_allocated()
        except Exception:
            return None
        return peak_bytes / (1024 * 1024)

    @staticmethod
    def _sampling_int(
        sampling: dict[str, Any],
        key: str,
        default: int | None,
    ) -> int:
        if key in sampling:
            return int(sampling[key])
        if default is None:
            raise KeyError(key)
        return int(default)

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None
        return int(value)

    @staticmethod
    def _validate_chunk_range(
        request: GenerationRequest,
        *,
        prompt_index: int,
        sample_start: int,
        sample_count: int,
    ) -> None:
        if prompt_index < 0 or prompt_index >= len(request.prompts):
            raise ValueError(f"chunk.prompt_index={prompt_index} is out of range")
        sample_end = sample_start + sample_count
        if sample_start < 0 or sample_count < 1:
            raise ValueError(
                "chunk sample range must have non-negative start and positive count",
            )
        if sample_end > request.samples_per_prompt:
            raise ValueError(
                "chunk sample range exceeds request.samples_per_prompt: "
                f"{sample_start}:{sample_end} > {request.samples_per_prompt}",
            )


class ARPipelineExecutorBase(
    ChunkedFamilyPipelineExecutor,
):
    """Base helpers for AR family executors.

    Subclasses still own tokenization details, sampling math, decoding, and
    family-specific output packing.
    """

    family: str
    task: str
    default_image_token_num: int | None = 576
    default_image_size: int | None = 384
    default_max_text_length: int | None = 256

    @property
    def layout(self) -> ARRequestLayout:
        return ARRequestLayout(
            default_image_token_num=self.default_image_token_num,
            default_image_size=self.default_image_size,
            default_max_text_length=self.default_max_text_length,
        )

    def parse_spec(self, request: GenerationRequest) -> ARGenerationSpec:
        return self.layout.parse_spec(request)

    def expand_prompts(self, request: GenerationRequest) -> list[str]:
        return self.layout.expand_prompts(request)

    def forward_batch_plan(
        self,
        requests: list[GenerationRequest],
        sample_specs_by_request: dict[str, list[GenerationSampleSpec]],
        engine_plans_by_request: dict[str, Any],
    ) -> dict[str, OutputBatch]:
        return forward_batch_by_merging_prompts(
            self,
            requests,
            sample_specs_by_request,
            engine_plans_by_request=engine_plans_by_request,
        )

    def validate_chunk(self, request: GenerationRequest, chunk: MicroBatchPlan) -> None:
        self.layout.validate_chunk(request, chunk)

    def ordered_chunks(
        self,
        request: GenerationRequest,
        sample_specs: Sequence[GenerationSampleSpec],
        chunks: Sequence[TChunk],
        *,
        row_fields: Sequence[str] = (),
    ) -> list[TChunk]:
        return self.layout.ordered_chunks(
            request,
            sample_specs,
            chunks,
            row_fields=row_fields,
        )

    def require_rows(self, name: str, value: Any, count: int) -> None:
        self.layout.require_rows(name, value, count)

    def chunk_seed_offset(
        self,
        request: GenerationRequest,
        chunk: MicroBatchPlan,
    ) -> int:
        return self.layout.chunk_seed_offset(request, chunk)

    def chunk_sample_specs(
        self,
        request: GenerationRequest,
        chunk: MicroBatchPlan,
    ) -> list[GenerationSampleSpec]:
        return self.layout.chunk_sample_specs(request, chunk)

    def max_peak_memory_mb(
        self,
        chunks: Sequence[ARChunkResult],
    ) -> float | None:
        return self.layout.max_peak_memory_mb(chunks)

    def align_pair(
        self,
        a_ids: torch.Tensor,
        a_mask: torch.Tensor,
        b_ids: torch.Tensor,
        b_mask: torch.Tensor,
        pad_id: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.layout.align_pair(a_ids, a_mask, b_ids, b_mask, pad_id=pad_id)

    def peak_memory_mb(self) -> float | None:
        return self.layout.peak_memory_mb()


__all__ = [
    "ARChunkResult",
    "ARPipelineExecutorBase",
    "ARRequestLayout",
]
