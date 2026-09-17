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
    gather_batch_media,
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
            # Decoded media or boxed references are validated separately below.
            row_fields=("latents", "log_probs", "timesteps", "kl"),
        )

        # One storage for the denoise path: concatenate it once and hand the
        # trajectory its two step-aligned views. Splitting per batch would
        # materialize every intermediate latent twice on the driver.
        latents = concatenate_sample_values(
            [batch.latents for batch in ordered_batches], name="latents"
        )
        log_probs = concatenate_sample_values(
            [batch.log_probs for batch in ordered_batches], name="log_probs"
        )
        if latents.shape[1] != log_probs.shape[1] + 1:
            raise ValueError(
                f"latents path has {latents.shape[1]} rows per sample, expected "
                f"{log_probs.shape[1] + 1} (one more than the log_probs steps)",
            )
        observations = latents[:, :-1]
        actions = latents[:, 1:]
        timesteps_tensor = concatenate_sample_values(
            [batch.timesteps for batch in ordered_batches], name="timesteps"
        )
        kl_tensor = concatenate_sample_values([batch.kl for batch in ordered_batches], name="kl")
        video = gather_batch_media(ordered_batches)
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
