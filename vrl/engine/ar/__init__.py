"""Autoregressive generation helpers."""

from vrl.engine.ar.cache import ARCacheRows, ar_concat_rows, ar_split_rows
from vrl.engine.ar.decode_loop import KVDecodeResult, run_kv_decode
from vrl.engine.ar.executor_base import (
    ARChunkResult,
    ARPipelineExecutorBase,
    ARRequestLayout,
)
from vrl.engine.ar.sequence import ActiveSequence, ARSequenceKey
from vrl.engine.ar.spec import ARGenerationSpec
from vrl.engine.ar.token_scheduler import TokenBatch, TokenScheduler
from vrl.engine.ar.types import ARStepResult

__all__ = [
    "ARCacheRows",
    "ARChunkResult",
    "ARGenerationSpec",
    "ARPipelineExecutorBase",
    "ARRequestLayout",
    "ARSequenceKey",
    "ARStepResult",
    "ActiveSequence",
    "KVDecodeResult",
    "TokenBatch",
    "TokenScheduler",
    "ar_concat_rows",
    "ar_split_rows",
    "run_kv_decode",
]
