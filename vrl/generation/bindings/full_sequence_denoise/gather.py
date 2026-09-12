"""Pure gatherer for full-sequence denoise batch payloads.

Reassembly runs driver-side without a model instance. The gatherer crosses the
Ray launch contract as a serializable object (see ``GenerationBatchGatherer``
in ``vrl/generation/protocols.py``). Its result annotation uses a TYPE_CHECKING
import; the parent package's public exports still import the executor module.
"""

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
from vrl.trajectory.builders import build_diffusion_trajectory

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
        ordered_batches = sort_and_validate_batch_coverage(
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
