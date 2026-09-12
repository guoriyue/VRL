"""Pure gatherer for chunk-autoregressive denoise sample payloads."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from vrl.generation.execution.sample_batches import (
    concatenate_sample_values,
    gather_batch_context,
    gather_replay_tensors,
    sort_and_validate_batch_coverage,
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

if TYPE_CHECKING:
    from vrl.generation.bindings.chunk_autoregressive_denoise.executor import (
        ChunkAutoregressiveDenoiseResult,
    )


class ChunkAutoregressiveDenoiseGatherer:
    """Order and concatenate prompt/sample batches without owning a model."""

    def merge_generation_batches(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        batches: Sequence[BatchPayload],
    ) -> GenerationOutput:
        ordered = self._order_and_validate_batches(
            request,
            sample_rows,
            cast("Sequence[ChunkAutoregressiveDenoiseResult]", batches),
        )
        output = concatenate_sample_values([batch.output for batch in ordered], name="output")
        rows = list(sample_rows)
        context = gather_batch_context([batch.context for batch in ordered])

        if ordered[0].has_trainable_trajectory:
            kl_values = [batch.kl for batch in ordered]
            if all(value is None for value in kl_values):
                kl = None
            elif any(value is None for value in kl_values):
                raise ValueError("optional batch field 'kl' must be present on all results")
            else:
                kl = concatenate_sample_values(kl_values, name="kl")
            trajectory = build_chunk_autoregressive_denoise_trajectory(
                request=request,
                sample_rows=rows,
                observations=concatenate_sample_values(
                    [batch.observations for batch in ordered], name="observations"
                ),
                actions=concatenate_sample_values(
                    [batch.actions for batch in ordered], name="actions"
                ),
                old_log_prob=concatenate_sample_values(
                    [batch.old_log_prob for batch in ordered], name="old_log_prob"
                ),
                mask=concatenate_sample_values([batch.mask for batch in ordered], name="mask"),
                timesteps=concatenate_sample_values(
                    [batch.timesteps for batch in ordered], name="timesteps"
                ),
                kl=kl,
                finalized_chunk_latents=concatenate_sample_values(
                    [batch.finalized_chunk_latents for batch in ordered],
                    name="finalized_chunk_latents",
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

    @staticmethod
    def _order_and_validate_batches(
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        batches: Sequence[ChunkAutoregressiveDenoiseResult],
    ) -> list[ChunkAutoregressiveDenoiseResult]:
        ordered = sort_and_validate_batch_coverage(
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
                batch.validate_trainable_trajectory()
        return ordered


__all__ = ["ChunkAutoregressiveDenoiseGatherer"]
