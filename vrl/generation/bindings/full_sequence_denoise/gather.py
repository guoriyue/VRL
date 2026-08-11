"""Pure gatherer for full-sequence denoise chunk payloads."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch

from vrl.generation.bindings.full_sequence_denoise.layout import DiffusionRequestLayout
from vrl.generation.execution.chunks import (
    gather_replay_tensors,
    require_matching_chunk_context,
)
from vrl.generation.protocols import ChunkResult
from vrl.generation.types import (
    GenerationOutput,
    GenerationRequest,
    GenerationSampleRow,
)
from vrl.trajectory import build_diffusion_trajectory

if TYPE_CHECKING:
    from vrl.generation.bindings.full_sequence_denoise.executor import DiffusionChunkResult


class DiffusionChunkGatherer:
    """Pure gatherer for shared diffusion chunk payloads."""

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_rows: Sequence[GenerationSampleRow],
        chunks: Sequence[ChunkResult],
    ) -> GenerationOutput:
        ordered_chunks = DiffusionRequestLayout.ordered_chunks(
            request,
            sample_rows,
            cast("Sequence[DiffusionChunkResult]", chunks),
        )

        observations = torch.cat([chunk.observations for chunk in ordered_chunks], dim=0)
        actions = torch.cat([chunk.actions for chunk in ordered_chunks], dim=0)
        log_probs = torch.cat([chunk.log_probs for chunk in ordered_chunks], dim=0)
        timesteps_tensor = torch.cat([chunk.timesteps for chunk in ordered_chunks], dim=0)
        kl_tensor = torch.cat([chunk.kl for chunk in ordered_chunks], dim=0)
        video = torch.cat([chunk.video for chunk in ordered_chunks], dim=0)
        replay_tensors = gather_replay_tensors(
            [chunk.replay_tensors for chunk in ordered_chunks],
            sample_counts=[chunk.chunk.sample_count for chunk in ordered_chunks],
        )
        rollout_context = require_matching_chunk_context(
            [chunk.context for chunk in ordered_chunks],
        )
        if not rollout_context:
            raise ValueError("DiffusionChunkResult.context must be non-empty")

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
    "DiffusionChunkGatherer",
]
