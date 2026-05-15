"""Shared KV-cache decode loop for autoregressive generation."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from vrl.engine.ar.sequence import ActiveSequence
from vrl.engine.ar.token_scheduler import TokenScheduler
from vrl.engine.core.types import GenerationRequest, GenerationSampleSpec


@dataclass(slots=True)
class KVDecodeResult:
    """Result of one engine-level AR KV decode loop."""

    finalized: Any
    scheduler_batches: int
    engine_counters: dict[str, Any] = field(default_factory=dict)


def run_kv_decode(
    *,
    request: GenerationRequest,
    sample_specs: Sequence[GenerationSampleSpec],
    model: Any,
    init_args: Sequence[Any] = (),
    init_kwargs: Mapping[str, Any] | None = None,
    max_new_tokens: int,
    tokenizer_key: str,
    dtype: str,
    scheduler_batch_size: int | None,
    step_kwargs: Mapping[str, Any] | None = None,
) -> KVDecodeResult:
    """Run a family-neutral AR KV decode loop.

    The engine owns active-row scheduling and loop progression. The family
    model owns prefill, one-token forward, sampling math, and final tensor
    packing through ``init_ar`` / ``step_ar`` / ``finalize_ar`` hooks.
    """

    specs = list(sample_specs)
    if not specs:
        raise ValueError("sample_specs must be non-empty")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")

    _require_hooks(model, ("init_ar", "step_ar", "finalize_ar"))

    state = _call_with_supported_kwargs(
        model.init_ar,
        *init_args,
        **dict(init_kwargs or {}),
    )
    sequences = _build_sequences(
        request=request,
        sample_specs=specs,
        tokenizer_key=tokenizer_key,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
    )
    max_batch_size = int(scheduler_batch_size or len(sequences))
    scheduler = TokenScheduler(max_batch_size=max(1, max_batch_size))
    scheduler.add_many(sequences)

    scheduler_batches = 0
    decode_tokens = 0
    engine_counters: dict[str, Any] = {
        "ar_kv_cache_enabled": True,
        "ar_scheduler_batch_size": scheduler_batch_size,
    }
    call_step_kwargs = dict(step_kwargs or {})

    while True:
        batch = scheduler.pop_batch()
        if batch is None:
            break
        scheduler_batches += 1
        decode_tokens += len(batch.sequences)
        result = _call_with_supported_kwargs(
            model.step_ar,
            state,
            batch.sequences,
            **call_step_kwargs,
        )
        debug_counters = getattr(result, "debug_counters", None)
        if debug_counters:
            engine_counters.update(dict(debug_counters))
        for sequence in batch.sequences:
            sequence.advance()
        scheduler.push_back_unfinished(batch)

    finalized = model.finalize_ar(state)
    engine_counters.setdefault("ar_decode_tokens", decode_tokens)
    engine_counters["ar_scheduler_batches"] = scheduler_batches
    return KVDecodeResult(
        finalized=finalized,
        scheduler_batches=scheduler_batches,
        engine_counters=engine_counters,
    )


def _require_hooks(model: Any, names: Sequence[str]) -> None:
    missing = [name for name in names if not hasattr(model, name)]
    if missing:
        raise TypeError(
            "AR KV decode requires model hooks: " + ", ".join(missing),
        )


def _build_sequences(
    *,
    request: GenerationRequest,
    sample_specs: Sequence[GenerationSampleSpec],
    tokenizer_key: str,
    dtype: str,
    max_new_tokens: int,
) -> list[ActiveSequence]:
    return [
        ActiveSequence(
            request_id=request.request_id,
            sample_id=spec.sample_id,
            family=request.family,
            task=request.task,
            tokenizer_key=tokenizer_key,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            metadata={
                **dict(spec.metadata),
                "row_index": row_index,
                "prompt_index": spec.prompt_index,
                "sample_index": spec.sample_index,
            },
        )
        for row_index, spec in enumerate(sample_specs)
    ]


def _call_with_supported_kwargs(fn: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    parameters = signature.parameters
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return fn(*args, **kwargs)
    supported = {key: value for key, value in kwargs.items() if key in parameters}
    return fn(*args, **supported)


__all__ = ["KVDecodeResult", "run_kv_decode"]
