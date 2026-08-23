"""Pure gatherer for chunk-autoregressive denoise sample payloads."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from vrl.generation.execution.sample_batches import (
    concatenate_sample_values,
    gather_replay_tensors,
    ordered_covering_batches,
    require_matching_batch_context,
)
from vrl.generation.protocols import BatchPayload
from vrl.generation.types import (
    GenerationOutput,
    GenerationRequest,
    GenerationSampleRow,
)
from vrl.trajectory import (
    build_chunk_autoregressive_denoise_trajectory,
    build_chunk_autoregressive_generation_trajectory,
)
from vrl.trajectory.validation import require_shape_prefix

if TYPE_CHECKING:
    from vrl.generation.bindings.chunk_autoregressive_denoise.executor import (
        ChunkAutoregressiveDenoiseResult,
    )


class ChunkAutoregressiveDenoiseGatherer:
    """Order and concatenate prompt/sample batches without owning a model."""

    def gather_batches(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        batches: Sequence[BatchPayload],
    ) -> GenerationOutput:
        ordered = _ordered_batches(
            request,
            sample_rows,
            cast("Sequence[ChunkAutoregressiveDenoiseResult]", batches),
        )
        output = concatenate_sample_values([batch.output for batch in ordered], name="output")
        rows = list(sample_rows)
        context = require_matching_batch_context([batch.context for batch in ordered])

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
                    [batch.replay_tensors for batch in ordered],
                    sample_counts=[batch.batch.sample_count for batch in ordered],
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
            output=output,
            trajectory=trajectory,
        )


def _ordered_batches(
    request: GenerationRequest,
    sample_rows: Sequence[GenerationSampleRow],
    batches: Sequence[ChunkAutoregressiveDenoiseResult],
) -> list[ChunkAutoregressiveDenoiseResult]:
    ordered = ordered_covering_batches(
        request,
        sample_rows,
        batches,
        row_fields=("output",),
    )
    first = ordered[0]
    for batch in ordered:
        if batch.temporal_chunk_count != first.temporal_chunk_count:
            raise ValueError("all results must have the same temporal_chunk_count")
        if batch.has_trainable_trajectory != first.has_trainable_trajectory:
            raise ValueError("cannot gather mixed trainable and generation-only results")
        if batch.has_trainable_trajectory:
            if batch.denoise_transition_count != first.denoise_transition_count:
                raise ValueError(
                    "all trainable results must have the same denoise_transition_count",
                )
            _validate_trainable_chunk(batch)
    return ordered


def _validate_trainable_chunk(batch: ChunkAutoregressiveDenoiseResult) -> None:
    transition_count = batch.denoise_transition_count
    if transition_count is None:
        raise ValueError("trainable result is missing denoise_transition_count")
    transition_prefix = (
        batch.batch.sample_count,
        batch.temporal_chunk_count,
        transition_count,
    )
    for field_name in ("observations", "actions", "old_log_prob", "mask", "timesteps"):
        require_shape_prefix(f"batch {field_name}", getattr(batch, field_name), transition_prefix)
    if batch.kl is not None:
        require_shape_prefix("batch kl", batch.kl, transition_prefix)
    require_shape_prefix(
        "batch finalized_chunk_latents",
        batch.finalized_chunk_latents,
        (batch.batch.sample_count, batch.temporal_chunk_count),
    )


def _cat_field(batches: Sequence[Any], field_name: str) -> Any:
    values = [getattr(batch, field_name) for batch in batches]
    if any(value is None for value in values):
        raise ValueError(f"trainable batch field {field_name!r} must be present")
    return concatenate_sample_values(values, name=field_name)


def _cat_optional_field(batches: Sequence[Any], field_name: str) -> Any | None:
    values = [getattr(batch, field_name) for batch in batches]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"optional batch field {field_name!r} must be present on all results")
    return concatenate_sample_values(values, name=field_name)


__all__ = ["ChunkAutoregressiveDenoiseGatherer"]
