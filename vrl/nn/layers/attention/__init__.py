"""Reusable attention layer contracts."""

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
    "ARPrefixCacheKey",
    "ARPrefixCachePolicy",
    "SD3JointAttentionProcessor",
]
