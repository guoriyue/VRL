"""Autoregressive generation helpers."""

from vrl.engine.ar.cache import ar_concat_rows, ar_split_rows
from vrl.engine.ar.decode_loop import KVDecodeResult, run_kv_decode
from vrl.engine.ar.executor_base import (
    ARChunkResult,
    ARPipelineExecutorBase,
    align_pair,
    chunk_sample_specs,
    chunk_seed_offset,
    expand_prompt_major_prompts,
    max_peak_memory_mb,
    ordered_chunks,
    parse_ar_generation_spec,
    peak_memory_mb,
    require_rows,
    validate_chunk,
)
from vrl.engine.ar.sequence import ActiveSequence, ARSequenceKey
from vrl.engine.ar.spec import ARGenerationSpec
from vrl.engine.ar.token_scheduler import TokenBatch, TokenScheduler
from vrl.engine.ar.types import ARStepResult

__all__ = [
    "ARChunkResult",
    "ARGenerationSpec",
    "ARPipelineExecutorBase",
    "ARSequenceKey",
    "ARStepResult",
    "ActiveSequence",
    "KVDecodeResult",
    "TokenBatch",
    "TokenScheduler",
    "align_pair",
    "ar_concat_rows",
    "ar_split_rows",
    "chunk_sample_specs",
    "chunk_seed_offset",
    "expand_prompt_major_prompts",
    "max_peak_memory_mb",
    "ordered_chunks",
    "parse_ar_generation_spec",
    "peak_memory_mb",
    "require_rows",
    "run_kv_decode",
    "validate_chunk",
]
