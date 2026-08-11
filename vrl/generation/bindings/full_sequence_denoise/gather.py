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

import torch

from vrl.generation.bindings.full_sequence_denoise.layout import DiffusionRequestLayout
from vrl.generation.execution.sample_batches import (
    gather_replay_tensors,
    require_matching_batch_context,
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

    def gather_batches(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        batches: Sequence[BatchPayload],
    ) -> GenerationOutput:
        ordered_batches = DiffusionRequestLayout.ordered_batches(
            request,
            sample_rows,
            cast("Sequence[DiffusionBatchResult]", batches),
        )

        observations = torch.cat([batch.observations for batch in ordered_batches], dim=0)
        actions = torch.cat([batch.actions for batch in ordered_batches], dim=0)
        log_probs = torch.cat([batch.log_probs for batch in ordered_batches], dim=0)
        timesteps_tensor = torch.cat([batch.timesteps for batch in ordered_batches], dim=0)
        kl_tensor = torch.cat([batch.kl for batch in ordered_batches], dim=0)
        video = torch.cat([batch.video for batch in ordered_batches], dim=0)
        replay_tensors = gather_replay_tensors(
            [batch.replay_tensors for batch in ordered_batches],
            sample_counts=[batch.batch.sample_count for batch in ordered_batches],
        )
        rollout_context = require_matching_batch_context(
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
