"""Shared AR paged-attention layer contract for model-family runners.

This module intentionally does not import vLLM. Concrete vLLM internal API
calls live in ``vrl.nn.kernels.attention.vllm_paged``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


class ARAttentionUnavailable(RuntimeError):
    """Raised when the vLLM paged-attention pieces cannot be initialized."""


@dataclass(frozen=True, slots=True)
class ARAttentionConfig:
    """Runtime identity shared by AR attention backends."""

    family: str


@dataclass(frozen=True, slots=True)
class VllmPagedAttentionConfig(ARAttentionConfig):
    """Runtime configuration owned by the vLLM paged-attention backend."""

    block_size: int = 16
    cache_dtype: str = "auto"


@dataclass(frozen=True, slots=True)
class ARAttentionPrefillInput:
    """One branch prefill payload for a model-family runner."""

    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    branch: str
    max_new_tokens: int = 1

    def __post_init__(self) -> None:
        _require_embed_mask_batch(self.inputs_embeds, self.attention_mask)
        if self.max_new_tokens < 1:
            raise ValueError("ARAttentionPrefillInput.max_new_tokens must be >= 1")


@dataclass(frozen=True, slots=True)
class ARAttentionPrefillOutput:
    """Prefill output consumed by a model-family runner."""

    last_hidden: torch.Tensor
    sequence_states: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class ARAttentionStepInput:
    """One image-token decode step payload."""

    input_embeds: torch.Tensor
    attention_mask: torch.Tensor
    sequence_states: tuple[Any, ...]

    def __post_init__(self) -> None:
        _require_embed_mask_batch(self.input_embeds, self.attention_mask)
        batch = self.input_embeds.shape[0]
        if len(self.sequence_states) != batch:
            raise ValueError("ARAttentionStepInput.sequence_states must match batch")


@dataclass(frozen=True, slots=True)
class ARAttentionStepOutput:
    """One image-token decode step result."""

    last_hidden: torch.Tensor
    sequence_states: tuple[Any, ...]


class ARAttentionBackend:
    """Minimal backend contract needed by AR model-family runners."""

    def __init__(self, config: ARAttentionConfig) -> None:
        self.config = config

    def prefill(
        self,
        request: ARAttentionPrefillInput,
    ) -> ARAttentionPrefillOutput:
        raise NotImplementedError

    def step(self, request: ARAttentionStepInput) -> ARAttentionStepOutput:
        raise NotImplementedError


def _require_embed_mask_batch(inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> None:
    if inputs_embeds.ndim != 3:
        raise ValueError("inputs_embeds must have shape [B, T, H]")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [B, T]")
    if inputs_embeds.shape[0] != attention_mask.shape[0]:
        raise ValueError("inputs_embeds and attention_mask batch sizes must match")


__all__ = [
    "ARAttentionBackend",
    "ARAttentionConfig",
    "ARAttentionPrefillInput",
    "ARAttentionPrefillOutput",
    "ARAttentionStepInput",
    "ARAttentionStepOutput",
    "ARAttentionUnavailable",
    "VllmPagedAttentionConfig",
]
