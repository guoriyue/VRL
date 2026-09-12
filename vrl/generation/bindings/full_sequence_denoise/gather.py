"""Pure gatherer for full-sequence denoise batch payloads.

Split from ``executor.py`` because reassembly runs driver-side, where no model
is loaded: the gatherer crosses the Ray launch contract as a serializable
object (see ``GenerationBatchGatherer`` in ``vrl/generation/protocols.py``) and must not
drag executor/model imports with it — hence the TYPE_CHECKING-only import of
``DiffusionBatchResult``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from vrl.generation.execution.sample_batches import (
    concatenate_sample_values,
    gather_batch_context,
    gather_replay_tensors,
    ordered_covering_batches,
)
from vrl.generation.protocols import BatchPayload
from vrl.generation.types import (
    GenerationOutput,
    GenerationRequest,
    GenerationSampleRow,
)
from vrl.trajectory import build_diffusion_trajectory

if TYPE_CHECKING:
    from vrl.generation.bindings.full_sequence_denoise.executor import DiffusionBatchResult


class DiffusionBatchGatherer:
    """Pure gatherer for shared diffusion batch payloads."""

    def merge_generation_batches(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        batches: Sequence[BatchPayload],
    ) -> GenerationOutput:
        ordered_batches = ordered_covering_batches(
            request,
            sample_rows,
            cast("Sequence[DiffusionBatchResult]", batches),
            row_fields=("observations", "actions", "log_probs", "timesteps", "kl", "video"),
        )

        observations = concatenate_sample_values(
            [batch.observations for batch in ordered_batches], name="observations"
        )
        actions = concatenate_sample_values(
            [batch.actions for batch in ordered_batches], name="actions"
        )
        log_probs = concatenate_sample_values(
            [batch.log_probs for batch in ordered_batches], name="log_probs"
        )
        timesteps_tensor = concatenate_sample_values(
            [batch.timesteps for batch in ordered_batches], name="timesteps"
        )
        kl_tensor = concatenate_sample_values([batch.kl for batch in ordered_batches], name="kl")
        video = concatenate_sample_values([batch.video for batch in ordered_batches], name="video")
        replay_tensors = gather_replay_tensors(
            [batch.replay_tensors for batch in ordered_batches],
            sample_counts=[batch.batch.sample_count for batch in ordered_batches],
        )
        rollout_context = gather_batch_context(
            [batch.context for batch in ordered_batches],
        )
        if not rollout_context:
            raise ValueError("DiffusionBatchResult.context must be non-empty")

        rows = list(sample_rows)
        trajectory = build_diffusion_trajectory(
            request=request,
            sample_rows=rows,
            observations=observations,
            actions=actions,
            old_log_prob=log_probs,
            timesteps=timesteps_tensor,
            kl=kl_tensor,
            replay_tensors=replay_tensors,
            context=rollout_context,
        )

        return GenerationOutput(
            output=video,
            trajectory=trajectory,
        )


__all__ = [
    "DiffusionBatchGatherer",
]
