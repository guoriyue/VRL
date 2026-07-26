"""Reusable attention layer contracts."""

from vrl.nn.layers.attention.cache_rows import ARCacheRows, ar_concat_rows, ar_split_rows
from vrl.nn.layers.attention.joint import SD3JointAttentionProcessor
from vrl.nn.layers.attention.paged import (
    ARAttentionBackend,
    ARAttentionConfig,
    ARAttentionPrefillInput,
    ARAttentionPrefillOutput,
    ARAttentionStepInput,
    ARAttentionStepOutput,
    ARAttentionUnavailable,
    VllmPagedAttentionConfig,
)

__all__ = [
    "ARAttentionBackend",
    "ARAttentionConfig",
    "ARAttentionPrefillInput",
    "ARAttentionPrefillOutput",
    "ARAttentionStepInput",
    "ARAttentionStepOutput",
    "ARAttentionUnavailable",
    "ARCacheRows",
    "SD3JointAttentionProcessor",
    "VllmPagedAttentionConfig",
    "ar_concat_rows",
    "ar_split_rows",
]
