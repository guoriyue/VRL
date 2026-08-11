"""Execution boundary for temporal-batch autoregressive denoise families."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vrl.generation.execution.executor_base import BatchExecutorBase
from vrl.generation.execution.sample_batches import (
    GenerationSampleBatch,
    validate_batch_range,
)
from vrl.generation.protocols import GenerationBatchGatherer
from vrl.generation.types import GenerationRequest


@dataclass(slots=True)
class ChunkAutoregressiveDenoiseResult:
    """Wire payload for one prompt/sample batch.

    The outer ``GenerationSampleBatch`` is only a transport batching decision.  The
    ``temporal_chunk_count`` and optional ``[B, C, S]`` tensors describe the
    model's actual generation organization inside each sample.
    """

    batch: GenerationSampleBatch
    output: Any
    temporal_chunk_count: int
    denoise_transition_count: int | None = None
    observations: Any | None = None
    actions: Any | None = None
    old_log_prob: Any | None = None
    mask: Any | None = None
    timesteps: Any | None = None
    kl: Any | None = None
    finalized_chunk_latents: Any | None = None
    replay_tensors: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.temporal_chunk_count < 1:
            raise ValueError("temporal_chunk_count must be >= 1")
        if self.denoise_transition_count is not None and self.denoise_transition_count < 0:
            raise ValueError("denoise_transition_count must be >= 0 when set")

        transition_values = (
            self.observations,
            self.actions,
            self.old_log_prob,
            self.mask,
            self.timesteps,
        )
        present = tuple(value is not None for value in transition_values)
        if any(present) and not all(present):
            raise ValueError(
                "trainable batch-denoise results require observations, actions, "
                "old_log_prob, mask, and timesteps together",
            )
        if all(present):
            if not self.denoise_transition_count:
                raise ValueError(
                    "trainable batch-denoise results require a positive denoise_transition_count",
                )
            if self.finalized_chunk_latents is None:
                raise ValueError(
                    "trainable batch-denoise results require finalized_chunk_latents",
                )

    @property
    def has_trainable_trajectory(self) -> bool:
        """Whether this result carries exact stochastic replay facts."""

        return self.old_log_prob is not None


class ChunkAutoregressiveDenoiseExecutorBase(BatchExecutorBase):
    """Shared request transport around family-owned batch generation.

    Cache allocation, per-temporal-batch scheduling, denoise math, and decode
    stay in the family model's ``generate_chunk_autoregressive`` method.  This
    base only owns prompt/sample batching and the typed gather boundary.
    """

    family: str
    task: str
    model: Any

    def __init__(
        self,
        model: Any,
        *,
        gatherer: GenerationBatchGatherer | None = None,
    ) -> None:
        super().__init__(gatherer=gatherer)
        self.model = model

    def forward_batch(
        self,
        request: GenerationRequest,
        batch: GenerationSampleBatch,
    ) -> ChunkAutoregressiveDenoiseResult:
        validate_batch_range(
            request,
            prompt_index=batch.prompt_index,
            sample_start=batch.sample_start,
            sample_count=batch.sample_count,
        )
        result = self.model.generate_chunk_autoregressive(
            request=request,
            batch=batch,
        )
        if not isinstance(result, ChunkAutoregressiveDenoiseResult):
            raise TypeError(
                "model.generate_chunk_autoregressive must return ChunkAutoregressiveDenoiseResult",
            )
        if result.batch != batch:
            raise ValueError(
                "model.generate_chunk_autoregressive returned a result for a "
                "different prompt/sample batch",
            )
        return result


__all__ = [
    "ChunkAutoregressiveDenoiseExecutorBase",
    "ChunkAutoregressiveDenoiseResult",
]
