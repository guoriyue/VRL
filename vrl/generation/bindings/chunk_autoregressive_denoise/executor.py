"""Execution boundary for temporal-chunk autoregressive denoise families."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vrl.generation.execution.chunks import (
    SampleChunk,
    validate_chunk_range,
)
from vrl.generation.execution.executor_base import ChunkExecutorBase
from vrl.generation.execution.planner import build_engine_plan
from vrl.generation.types import GenerationRequest


@dataclass(slots=True)
class ChunkAutoregressiveDenoiseResult:
    """Wire payload for one prompt/sample chunk.

    The outer ``SampleChunk`` is only a transport batching decision.  The
    ``temporal_chunk_count`` and optional ``[B, C, S]`` tensors describe the
    model's actual generation organization inside each sample.
    """

    prompt_index: int
    sample_start: int
    sample_count: int
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
    peak_memory_mb: float | None = None

    def __post_init__(self) -> None:
        if self.prompt_index < 0:
            raise ValueError("prompt_index must be >= 0")
        if self.sample_start < 0:
            raise ValueError("sample_start must be >= 0")
        if self.sample_count < 1:
            raise ValueError("sample_count must be >= 1")
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
                "trainable chunk-denoise results require observations, actions, "
                "old_log_prob, mask, and timesteps together",
            )
        if all(present):
            if not self.denoise_transition_count:
                raise ValueError(
                    "trainable chunk-denoise results require a positive denoise_transition_count",
                )
            if self.finalized_chunk_latents is None:
                raise ValueError(
                    "trainable chunk-denoise results require finalized_chunk_latents",
                )

    @property
    def has_trainable_trajectory(self) -> bool:
        """Whether this result carries exact stochastic replay facts."""

        return self.old_log_prob is not None


class ChunkAutoregressiveDenoiseExecutorBase(ChunkExecutorBase):
    """Shared request transport around family-owned chunk generation.

    Cache allocation, per-temporal-chunk scheduling, denoise math, and decode
    stay in the family model's ``generate_chunk_autoregressive`` method.  This
    base only owns prompt/sample batching and the typed gather boundary.
    """

    family: str
    task: str
    model: Any
    default_samples_per_chunk: int = 1

    def __init__(self, model: Any, *, samples_per_chunk: int | None = None) -> None:
        # Keyword-only and optional: the worker constructs executors by dotted
        # string (``executor_cls(model, **executor_kwargs)``) and only injects
        # samples_per_chunk for families whose registry entry declares
        # ``accepts_samples_per_chunk``. Unset keeps the class default.
        self.model = model
        if samples_per_chunk is not None:
            self.default_samples_per_chunk = max(1, int(samples_per_chunk))

    def plan(
        self,
        request: GenerationRequest,
    ) -> Any:
        return build_engine_plan(
            request,
            max_samples_per_chunk=self.default_samples_per_chunk,
        )

    def forward_chunk_plan(
        self,
        request: GenerationRequest,
        chunk: SampleChunk,
    ) -> ChunkAutoregressiveDenoiseResult:
        validate_chunk_range(
            request,
            prompt_index=chunk.prompt_index,
            sample_start=chunk.sample_start,
            sample_count=chunk.sample_count,
        )
        if chunk.prompt != request.prompts[chunk.prompt_index]:
            raise ValueError(
                "chunk.prompt does not match request.prompts[chunk.prompt_index]",
            )
        result = self.model.generate_chunk_autoregressive(
            request=request,
            chunk=chunk,
        )
        if not isinstance(result, ChunkAutoregressiveDenoiseResult):
            raise TypeError(
                "model.generate_chunk_autoregressive must return ChunkAutoregressiveDenoiseResult",
            )
        if (
            result.prompt_index != chunk.prompt_index
            or result.sample_start != chunk.sample_start
            or result.sample_count != chunk.sample_count
        ):
            raise ValueError(
                "model.generate_chunk_autoregressive returned a result for a "
                "different prompt/sample chunk",
            )
        return result


__all__ = [
    "ChunkAutoregressiveDenoiseExecutorBase",
    "ChunkAutoregressiveDenoiseResult",
]
