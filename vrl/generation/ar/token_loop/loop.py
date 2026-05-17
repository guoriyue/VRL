"""Shared autoregressive decode loop."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from vrl.generation.ar.token_loop.scheduler import TokenScheduler
from vrl.generation.ar.token_loop.sequence import ActiveSequence
from vrl.generation.ar.token_loop.state import (
    ARStepOutput,
    ARTokenLoopEnvelope,
    ARTokenLoopInit,
)
from vrl.generation.types import GenerationRequest, GenerationSampleRow


@dataclass(slots=True)
class ARDecodeResult:
    """Result of one engine-level AR decode loop."""

    finalized: Any
    scheduler_batches: int
    engine_counters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ARDecodeLoop:
    """Family-neutral AR decode loop driver."""

    request: GenerationRequest
    sample_rows: Sequence[GenerationSampleRow]
    runner: Any
    max_new_tokens: int
    tokenizer_key: str
    dtype: str
    scheduler_batch_size: int | None = None
    init_args: Sequence[Any] = ()
    init_kwargs: Mapping[str, Any] | None = None
    step_kwargs: Mapping[str, Any] | None = None

    def run(self) -> ARDecodeResult:
        rows = list(self.sample_rows)
        if not rows:
            raise ValueError("sample_rows must be non-empty")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")

        self._require_hooks(("init_ar", "step_ar", "finalize_ar"))

        init_state = self._call_with_supported_kwargs(
            self.runner.init_ar,
            *self.init_args,
            **dict(self.init_kwargs or {}),
        )
        if not isinstance(init_state, ARTokenLoopInit):
            raise TypeError(
                "AR model runner init_ar must return ARTokenLoopInit; "
                f"got {type(init_state).__name__}",
            )
        envelope = ARTokenLoopEnvelope.from_init(init_state, batch_size=len(rows))
        state = envelope.state
        sequences = self._build_sequences(rows)
        scheduler = TokenScheduler(max_batch_size=self._max_batch_size(sequences))
        scheduler.add_many(sequences)

        scheduler_batches = 0
        decode_tokens = 0
        engine_counters: dict[str, Any] = {
            "ar_decode_loop_enabled": True,
            "ar_scheduler_batch_size": self.scheduler_batch_size,
        }
        call_step_kwargs = dict(self.step_kwargs or {})

        while True:
            batch = scheduler.pop_batch()
            if batch is None:
                break
            scheduler_batches += 1
            decode_tokens += len(batch.sequences)
            step_batch = envelope.build_step_batch(batch.sequences)
            step_output = self._call_with_supported_kwargs(
                self.runner.step_ar,
                state,
                step_batch,
                **call_step_kwargs,
            )
            if not isinstance(step_output, ARStepOutput):
                raise TypeError(
                    "AR model runner step_ar must return ARStepOutput; "
                    f"got {type(step_output).__name__}",
                )
            envelope.apply_step_output(step_batch, step_output)
            result = step_output.result
            debug_counters = getattr(result, "debug_counters", None)
            if debug_counters:
                engine_counters.update(dict(debug_counters))
            for sequence in batch.sequences:
                sequence.advance()
            scheduler.push_back_unfinished(batch)

        finalized = self.runner.finalize_ar(state)
        engine_counters.setdefault("ar_decode_tokens", decode_tokens)
        engine_counters["ar_scheduler_batches"] = scheduler_batches
        return ARDecodeResult(
            finalized=finalized,
            scheduler_batches=scheduler_batches,
            engine_counters=engine_counters,
        )

    def _max_batch_size(self, sequences: Sequence[ActiveSequence]) -> int:
        max_batch_size = int(self.scheduler_batch_size or len(sequences))
        return max(1, max_batch_size)

    def _require_hooks(self, names: Sequence[str]) -> None:
        missing = [name for name in names if not hasattr(self.runner, name)]
        if missing:
            raise TypeError(
                "AR decode loop requires AR model runner hooks: "
                + ", ".join(missing),
            )

    def _build_sequences(
        self,
        sample_rows: Sequence[GenerationSampleRow],
    ) -> list[ActiveSequence]:
        return [
            ActiveSequence(
                request_id=self.request.request_id,
                sample_id=row.sample_id,
                family=self.request.family,
                task=self.request.task,
                tokenizer_key=self.tokenizer_key,
                dtype=self.dtype,
                max_new_tokens=self.max_new_tokens,
                metadata={
                    **dict(row.metadata),
                    "row_index": row_index,
                    "prompt_index": row.prompt_index,
                    "sample_index": row.sample_index,
                },
            )
            for row_index, row in enumerate(sample_rows)
        ]

    def _call_with_supported_kwargs(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            return fn(*args, **kwargs)
        parameters = signature.parameters
        if any(
            param.kind is inspect.Parameter.VAR_KEYWORD
            for param in parameters.values()
        ):
            return fn(*args, **kwargs)
        supported = {key: value for key, value in kwargs.items() if key in parameters}
        return fn(*args, **supported)


__all__ = ["ARDecodeLoop", "ARDecodeResult"]
