"""Pure gatherer for diffusion chunk payloads."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch

from vrl.engine.core.protocols import PipelineChunkResult
from vrl.engine.core.types import (
    GenerationMetrics,
    GenerationRequest,
    GenerationSampleSpec,
    OutputBatch,
)
from vrl.engine.diffusion.denoise import DiffusionChunkResult
from vrl.engine.diffusion.layout import DiffusionRequestLayout
from vrl.engine.trajectory import build_diffusion_trajectory


def gather_diffusion_chunks(
    request: GenerationRequest,
    sample_specs: Sequence[GenerationSampleSpec],
    chunks: Sequence[PipelineChunkResult],
    *,
    model_family: str,
    respect_cfg_flag: bool = True,
) -> OutputBatch:
    """Gather diffusion chunks using only request metadata and CPU payloads."""

    del model_family, respect_cfg_flag
    sampling = request.sampling
    layout = DiffusionRequestLayout()
    ordered_chunks = layout.ordered_chunks(
        request,
        sample_specs,
        cast("Sequence[DiffusionChunkResult]", chunks),
    )
    return build_diffusion_output_batch(
        request=request,
        sample_specs=list(sample_specs),
        prompts=list(request.prompts),
        chunks=ordered_chunks,
        num_steps=int(sampling["num_steps"]),
    )


@dataclass(frozen=True, slots=True)
class DiffusionChunkGatherer:
    """Pure gatherer for shared diffusion chunk payloads."""

    model_family: str
    respect_cfg_flag: bool = True

    def gather_chunks(
        self,
        request: GenerationRequest,
        sample_specs: Sequence[GenerationSampleSpec],
        chunks: Sequence[PipelineChunkResult],
    ) -> OutputBatch:
        return gather_diffusion_chunks(
            request,
            sample_specs,
            chunks,
            model_family=self.model_family,
            respect_cfg_flag=self.respect_cfg_flag,
        )


def build_diffusion_output_batch(
    *,
    request: GenerationRequest,
    sample_specs: list[GenerationSampleSpec],
    prompts: list[str],
    chunks: list[DiffusionChunkResult],
    num_steps: int,
) -> OutputBatch:
    """Pack ordered diffusion chunks into the canonical engine OutputBatch."""

    if not chunks:
        raise ValueError("chunks must be non-empty")

    observations = torch.cat([chunk.observations for chunk in chunks], dim=0)
    actions = torch.cat([chunk.actions for chunk in chunks], dim=0)
    log_probs = torch.cat([chunk.log_probs for chunk in chunks], dim=0)
    timesteps_tensor = torch.cat([chunk.timesteps for chunk in chunks], dim=0)
    kl_tensor = torch.cat([chunk.kl for chunk in chunks], dim=0)
    video = torch.cat([chunk.video for chunk in chunks], dim=0)
    replay_tensors = _concat_replay_tensors([chunk.replay_tensors for chunk in chunks])
    rollout_context = chunks[0].context
    if not rollout_context:
        raise ValueError("DiffusionChunkResult.context must be non-empty")

    layout = DiffusionRequestLayout()
    peak_mem_mb = layout.max_peak_memory_mb(chunks)
    metrics = GenerationMetrics(
        num_prompts=len(prompts),
        num_samples=len(sample_specs),
        num_steps=num_steps,
        micro_batches=len(chunks),
        peak_memory_mb=peak_mem_mb,
    )
    trajectory = build_diffusion_trajectory(
        request=request,
        sample_specs=sample_specs,
        observations=observations,
        actions=actions,
        old_log_prob=log_probs,
        timesteps=timesteps_tensor,
        kl=kl_tensor,
        replay_tensors=replay_tensors,
        context=rollout_context,
    )

    return OutputBatch(
        request_id=request.request_id,
        family=request.family,
        task=request.task,
        prompts=prompts,
        sample_specs=sample_specs,
        output=video,
        trajectory=trajectory,
        extra={},
        metrics=metrics,
        peak_memory_mb=peak_mem_mb or 0.0,
    )


def _concat_replay_tensors(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    replay_tensors: dict[str, Any] = {}
    if not chunks:
        return replay_tensors
    for key in chunks[0]:
        vals = [chunk[key] for chunk in chunks]
        if any(value is None for value in vals):
            replay_tensors[key] = None
        elif all(isinstance(value, torch.Tensor) for value in vals):
            replay_tensors[key] = torch.cat(vals, dim=0)
        else:
            replay_tensors[key] = vals[0]
    return replay_tensors


__all__ = [
    "DiffusionChunkGatherer",
    "build_diffusion_output_batch",
    "gather_diffusion_chunks",
]
