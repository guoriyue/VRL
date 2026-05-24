"""Reusable attention layer contracts."""

from vrl.nn.layers.attention.base import AttentionCacheView
from vrl.nn.layers.attention.joint import SD3JointAttentionProcessor
from vrl.nn.layers.attention.paged import (
    ARPagedAttentionBackend,
    ARPagedAttentionConfig,
    ARPagedAttentionPrefillInput,
    ARPagedAttentionPrefillOutput,
    ARPagedAttentionStepInput,
    ARPagedAttentionStepOutput,
    ARPagedAttentionUnavailable,
    ARPrefixCacheKey,
    ARPrefixCachePolicy,
)

__all__ = [
    "ARPagedAttentionBackend",
    "ARPagedAttentionConfig",
    "ARPagedAttentionPrefillInput",
    "ARPagedAttentionPrefillOutput",
    "ARPagedAttentionStepInput",
    "ARPagedAttentionStepOutput",
    "ARPagedAttentionUnavailable",
    "ARPrefixCacheKey",
    "ARPrefixCachePolicy",
    "AttentionCacheView",
    "SD3JointAttentionProcessor",
]
