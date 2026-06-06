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
    ARPrefixCacheKey,
    ARPrefixCachePolicy,
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
    "ARPrefixCacheKey",
    "ARPrefixCachePolicy",
    "SD3JointAttentionProcessor",
    "ar_concat_rows",
    "ar_split_rows",
]
