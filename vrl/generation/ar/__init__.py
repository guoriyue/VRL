"""Autoregressive generation helpers."""

from vrl.generation.ar.executor import ARPipelineExecutorBase
from vrl.generation.ar.layout import (
    ARChunkResult,
    ARRequestLayout,
    ARSamplingParams,
)
from vrl.generation.ar.token_loop import (
    ARStepBatch,
    ARStepOutput,
    ARTokenLoopEnvelope,
    ARTokenLoopInit,
)
from vrl.generation.ar.token_loop.loop import ARDecodeLoop, ARDecodeResult
from vrl.generation.ar.token_loop.row_cache import ARCacheRows, ar_concat_rows, ar_split_rows
from vrl.generation.ar.token_loop.scheduler import TokenBatch, TokenScheduler
from vrl.generation.ar.token_loop.sequence import ActiveSequence, ARSequenceKey
from vrl.generation.ar.token_loop.state import ARStepResult

__all__ = [
    "ARCacheRows",
    "ARChunkResult",
    "ARDecodeLoop",
    "ARDecodeResult",
    "ARPipelineExecutorBase",
    "ARRequestLayout",
    "ARSamplingParams",
    "ARSequenceKey",
    "ARStepBatch",
    "ARStepOutput",
    "ARStepResult",
    "ARTokenLoopEnvelope",
    "ARTokenLoopInit",
    "ActiveSequence",
    "TokenBatch",
    "TokenScheduler",
    "ar_concat_rows",
    "ar_split_rows",
]
