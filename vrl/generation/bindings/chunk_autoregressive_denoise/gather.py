"""Pure gatherer for chunk-autoregressive denoise sample payloads."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from vrl.generation.execution.chunks import (
    concatenate_sample_values,
    gather_replay_tensors,
    ordered_covering_chunks,
    require_matching_chunk_context,
)
from vrl.generation.protocols import ChunkResult
from vrl.generation.types import (
    GenerationOutput,
    GenerationRequest,
    GenerationSampleRow,
)
from vrl.trajectory import (
    build_chunk_autoregressive_denoise_trajectory,
    build_chunk_autoregressive_generation_trajectory,
)

if TYPE_CHECKING:
    from vrl.generation.bindings.chunk_autoregressive_denoise.executor import (
        ChunkAutoregressiveDenoiseResult,
    )


@dataclass(frozen=True, slots=True)
class ChunkAutoregressiveDenoiseGatherer:
    """Order and concatenate prompt/sample chunks without owning a model."""

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        chunks: Sequence[ChunkResult],
    ) -> GenerationOutput:
        ordered = _ordered_chunks(
            request,
            sample_rows,
            cast("Sequence[ChunkAutoregressiveDenoiseResult]", chunks),
        )
        output = concatenate_sample_values([chunk.output for chunk in ordered], name="output")
        rows = list(sample_rows)
        context = require_matching_chunk_context([chunk.context for chunk in ordered])

        if ordered[0].has_trainable_trajectory:
            trajectory = build_chunk_autoregressive_denoise_trajectory(
                request=request,
                sample_rows=rows,
                observations=_cat_field(ordered, "observations"),
                actions=_cat_field(ordered, "actions"),
                old_log_prob=_cat_field(ordered, "old_log_prob"),
                mask=_cat_field(ordered, "mask"),
                timesteps=_cat_field(ordered, "timesteps"),
                kl=_cat_optional_field(ordered, "kl"),
                finalized_chunk_latents=_cat_field(
                    ordered,
                    "finalized_chunk_latents",
                ),
                replay_tensors=gather_replay_tensors(
                    [chunk.replay_tensors for chunk in ordered],
                    sample_counts=[chunk.sample_count for chunk in ordered],
                ),
                context=context,
            )
        else:
            trajectory = build_chunk_autoregressive_generation_trajectory(
                request=request,
                sample_rows=rows,
                output=output,
                temporal_chunk_count=ordered[0].temporal_chunk_count,
                context=context,
            )

        return GenerationOutput(
            request_id=request.request_id,
            sample_rows=rows,
            output=output,
            trajectory=trajectory,
            extra={},
        )


def _ordered_chunks(
    request: GenerationRequest,
    sample_rows: Sequence[GenerationSampleRow],
    chunks: Sequence[ChunkAutoregressiveDenoiseResult],
) -> list[ChunkAutoregressiveDenoiseResult]:
    ordered = ordered_covering_chunks(
        request,
        sample_rows,
        chunks,
        row_fields=("output",),
    )
    first = ordered[0]
    for chunk in ordered:
        if chunk.temporal_chunk_count != first.temporal_chunk_count:
            raise ValueError("all results must have the same temporal_chunk_count")
        if chunk.has_trainable_trajectory != first.has_trainable_trajectory:
            raise ValueError("cannot gather mixed trainable and generation-only results")
        if chunk.has_trainable_trajectory:
            if chunk.denoise_transition_count != first.denoise_transition_count:
                raise ValueError(
                    "all trainable results must have the same denoise_transition_count",
                )
            _validate_trainable_chunk(chunk)
    return ordered


def _validate_trainable_chunk(chunk: ChunkAutoregressiveDenoiseResult) -> None:
    transition_count = chunk.denoise_transition_count
    if transition_count is None:
        raise ValueError("trainable result is missing denoise_transition_count")
    transition_prefix = (
        chunk.sample_count,
        chunk.temporal_chunk_count,
        transition_count,
    )
    for field_name in ("observations", "actions", "old_log_prob", "mask", "timesteps"):
        _require_shape_prefix(field_name, getattr(chunk, field_name), transition_prefix)
    if chunk.kl is not None:
        _require_shape_prefix("kl", chunk.kl, transition_prefix)
    _require_shape_prefix(
        "finalized_chunk_latents",
        chunk.finalized_chunk_latents,
        (chunk.sample_count, chunk.temporal_chunk_count),
    )


def _require_shape_prefix(name: str, value: Any, expected: tuple[int, ...]) -> None:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < len(expected):
        raise ValueError(f"chunk {name} must have leading dimensions {expected}")
    actual = tuple(int(length) for length in shape[: len(expected)])
    if actual != expected:
        raise ValueError(
            f"chunk {name} has leading dimensions {actual}, expected {expected}",
        )


def _cat_field(chunks: Sequence[Any], field_name: str) -> Any:
    values = [getattr(chunk, field_name) for chunk in chunks]
    if any(value is None for value in values):
        raise ValueError(f"trainable chunk field {field_name!r} must be present")
    return concatenate_sample_values(values, name=field_name)


def _cat_optional_field(chunks: Sequence[Any], field_name: str) -> Any | None:
    values = [getattr(chunk, field_name) for chunk in chunks]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"optional chunk field {field_name!r} must be present on all results")
    return concatenate_sample_values(values, name=field_name)


__all__ = ["ChunkAutoregressiveDenoiseGatherer"]
