"""Pure gatherer for full-sequence denoise chunk payloads."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch

from vrl.generation.bindings.full_sequence_denoise.layout import DiffusionRequestLayout
from vrl.generation.protocols import ChunkResult
from vrl.generation.types import (
    GenerationOutput,
    GenerationRequest,
    GenerationSampleRow,
)
from vrl.trajectory import build_diffusion_trajectory

if TYPE_CHECKING:
    from vrl.generation.bindings.full_sequence_denoise.executor import DiffusionChunkResult


@dataclass(frozen=True, slots=True)
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
        replay_tensors: dict[str, Any] = {}
        for key in ordered_chunks[0].replay_tensors:
            vals = [chunk.replay_tensors[key] for chunk in ordered_chunks]
            if any(value is None for value in vals):
                replay_tensors[key] = None
            elif all(isinstance(value, torch.Tensor) for value in vals):
                replay_tensors[key] = torch.cat(vals, dim=0)
            else:
                replay_tensors[key] = vals[0]
        rollout_context = ordered_chunks[0].context
        if not rollout_context:
            raise ValueError("DiffusionChunkResult.context must be non-empty")

        rows = list(sample_rows)
        prompts = list(request.prompts)
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
            request_id=request.request_id,
            family=request.family,
            task=request.task,
            prompts=prompts,
            sample_rows=rows,
            output=video,
            trajectory=trajectory,
            extra={},
        )


__all__ = [
    "DiffusionChunkGatherer",
]
