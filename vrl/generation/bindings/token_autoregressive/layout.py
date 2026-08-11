"""Request layout helpers shared by token-autoregressive executors and gatherers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

import torch

from vrl.generation.execution.sample_batches import (
    GenerationSampleBatch,
    ordered_covering_batches,
    validate_batch_range,
)
from vrl.generation.types import GenerationRequest, GenerationSampleRow


class ARBatchPayload(Protocol):
    """Common metadata every prompt-major AR batch result carries."""

    batch: GenerationSampleBatch
    peak_memory_mb: float | None


TBatch = TypeVar("TBatch", bound=ARBatchPayload)


@dataclass(frozen=True, slots=True)
class ARSamplingParams:
    """Parsed AR sampling fields from ``GenerationRequest.sampling``."""

    image_token_num: int
    image_size: int
    max_text_length: int
    seed: int | None


def right_pad(
    ids: torch.Tensor,
    mask: torch.Tensor,
    *,
    target_length: int,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a ``[B, L]`` ids/mask pair up to ``target_length``.

    Pads only when the input is shorter (``>= target_length`` is a no-op), so an
    over-length input from a misbehaving stub tokenizer passes through unchanged
    instead of triggering a negative-width ``torch.full``.
    """

    current_length = ids.shape[1]
    if current_length >= target_length:
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


class ARRequestLayout:
    """Prompt-major request layout shared by AR executors and gatherers."""

    __slots__ = (
        "default_image_size",
        "default_image_token_num",
        "default_max_text_length",
    )

    def __init__(
        self,
        *,
        default_image_token_num: int | None = None,
        default_image_size: int | None = None,
        default_max_text_length: int | None = None,
    ) -> None:
        self.default_image_token_num = default_image_token_num
        self.default_image_size = default_image_size
        self.default_max_text_length = default_max_text_length

    def parse_sampling_params(self, request: GenerationRequest) -> ARSamplingParams:
        """Parse family-neutral AR fields from ``GenerationRequest.sampling``."""

        sampling = request.sampling
        return ARSamplingParams(
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
            seed=None if sampling.get("seed") is None else int(sampling.get("seed")),
        )

    def resolve_scheduler_batch_size(self, request: GenerationRequest) -> int | None:
        """Resolve the request-local token-step row bound without coercion."""

        value = request.sampling.get("ar_scheduler_batch_size")
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                "request.sampling.ar_scheduler_batch_size must be a positive integer or null",
            )
        return value

    def validate_chunk(self, request: GenerationRequest, batch: GenerationSampleBatch) -> None:
        """Validate one prompt/sample AR batch against its request."""

        validate_batch_range(
            request,
            prompt_index=batch.prompt_index,
            sample_start=batch.sample_start,
            sample_count=batch.sample_count,
        )

    def ordered_batches(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        batches: Sequence[TBatch],
        *,
        row_fields: Sequence[str] = (),
    ) -> list[TBatch]:
        """Sort AR batches and ensure they exactly cover prompt-major samples."""

        return ordered_covering_batches(
            request,
            sample_rows,
            batches,
            row_fields=row_fields,
        )

    def chunk_seed_offset(self, request: GenerationRequest, batch: GenerationSampleBatch) -> int:
        """Return the prompt-major sample offset for deterministic batch seeding."""

        return batch.prompt_index * int(request.samples_per_prompt) + batch.sample_start

    def cat_batch_fields(
        self,
        batches: Sequence[Any],
        fields: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        """Concatenate ordered batch payload tensors along the batch dim.

        The gatherers' cat step is pure data (a field-name list over already
        ``ordered_batches``-validated payloads), so it lives here once instead
        of each family writing one ``torch.cat`` block per field.
        """

        return {
            field: torch.cat([getattr(batch, field) for batch in batches], dim=0)
            for field in fields
        }

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
        a_ids, a_mask = right_pad(a_ids, a_mask, target_length=target_length, pad_id=pad_id)
        b_ids, b_mask = right_pad(b_ids, b_mask, target_length=target_length, pad_id=pad_id)
        return a_ids, a_mask, b_ids, b_mask

    @staticmethod
    def _sampling_int(
        sampling: dict[str, Any],
        key: str,
        default: int | None,
    ) -> int:
        if key in sampling:
            return int(sampling[key])
        if default is None:
            raise ValueError(f"request.sampling.{key} is required")
        return int(default)


__all__ = ["ARBatchPayload", "ARRequestLayout", "ARSamplingParams", "right_pad"]
