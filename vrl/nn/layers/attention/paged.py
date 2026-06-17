"""Shared AR paged-attention layer contract for model-family runners.

This module intentionally does not import vLLM. Concrete vLLM internal API
calls live in ``vrl.nn.kernels.attention.vllm_paged``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

import torch


class ARAttentionUnavailable(RuntimeError):
    """Raised when the vLLM paged-attention pieces cannot be initialized."""


@dataclass(frozen=True, slots=True)
class ARAttentionConfig:
    """Runtime identity for a paged-attention backend instance."""

    family: str
    model_key: str
    block_size: int = 16
    cache_layout_version: str = "vllm.v1"
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.family:
            raise ValueError("ARAttentionConfig.family must be non-empty")
        if not self.model_key:
            raise ValueError("ARAttentionConfig.model_key must be non-empty")
        if self.block_size < 1:
            raise ValueError("ARAttentionConfig.block_size must be >= 1")


@dataclass(frozen=True, slots=True)
class ARAttentionPrefillInput:
    """One branch prefill payload for a model-family runner."""

    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    branch: str
    sequence_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_embed_mask_batch(self.inputs_embeds, self.attention_mask)
        if not self.branch:
            raise ValueError("ARAttentionPrefillInput.branch must be non-empty")
        if self.sequence_ids and len(self.sequence_ids) != self.inputs_embeds.shape[0]:
            raise ValueError(
                "ARAttentionPrefillInput.sequence_ids must match batch size",
            )


@dataclass(frozen=True, slots=True)
class ARAttentionPrefillOutput:
    """Prefill output consumed by a model-family runner."""

    last_hidden: torch.Tensor
    sequence_states: tuple[Any, ...]
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_last_hidden_batch(self.last_hidden, len(self.sequence_states))


@dataclass(frozen=True, slots=True)
class ARAttentionStepInput:
    """One image-token decode step payload."""

    input_embeds: torch.Tensor
    attention_mask: torch.Tensor
    sequence_states: tuple[Any, ...]
    branch_names: tuple[str, ...]
    position: int
    row_indices: tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_embed_mask_batch(self.input_embeds, self.attention_mask)
        batch = self.input_embeds.shape[0]
        if len(self.sequence_states) != batch:
            raise ValueError("ARAttentionStepInput.sequence_states must match batch")
        if len(self.branch_names) != batch:
            raise ValueError("ARAttentionStepInput.branch_names must match batch")
        if self.position < 0:
            raise ValueError("ARAttentionStepInput.position must be >= 0")


@dataclass(frozen=True, slots=True)
class ARAttentionStepOutput:
    """One image-token decode step result."""

    last_hidden: torch.Tensor
    sequence_states: tuple[Any, ...]
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_last_hidden_batch(self.last_hidden, len(self.sequence_states))


@dataclass(frozen=True, slots=True)
class ARPrefixCacheKey:
    """Policy-safe identity for reusable AR prompt KV pages."""

    family: str
    task: str
    policy_version: int | str
    tokenizer_key: str
    prompt_hash: str
    model_dtype: str
    cache_dtype: str
    cfg_branch_kind: str
    cache_layout_version: str
    paged_attention_config_hash: str

    @classmethod
    def from_prompt_tokens(
        cls,
        *,
        family: str,
        task: str,
        policy_version: int | str,
        tokenizer_key: str,
        prompt_token_ids: Sequence[int],
        model_dtype: str,
        cache_dtype: str,
        cfg_branch_kind: str,
        cache_layout_version: str,
        paged_attention_config_hash: str,
    ) -> ARPrefixCacheKey:
        prompt_hash = _stable_hash([int(token) for token in prompt_token_ids])
        return cls(
            family=family,
            task=task,
            policy_version=policy_version,
            tokenizer_key=tokenizer_key,
            prompt_hash=prompt_hash,
            model_dtype=model_dtype,
            cache_dtype=cache_dtype,
            cfg_branch_kind=cfg_branch_kind,
            cache_layout_version=cache_layout_version,
            paged_attention_config_hash=paged_attention_config_hash,
        )

    def __post_init__(self) -> None:
        required = {
            "family": self.family,
            "task": self.task,
            "tokenizer_key": self.tokenizer_key,
            "prompt_hash": self.prompt_hash,
            "model_dtype": self.model_dtype,
            "cache_dtype": self.cache_dtype,
            "cfg_branch_kind": self.cfg_branch_kind,
            "cache_layout_version": self.cache_layout_version,
            "paged_attention_config_hash": self.paged_attention_config_hash,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"ARPrefixCacheKey missing fields: {', '.join(missing)}")


@dataclass(slots=True)
class ARPrefixCachePolicy:
    """Small policy object for prefix-cache correctness checks."""

    enabled: bool = True

    def can_reuse(
        self,
        cached_key: ARPrefixCacheKey,
        requested_key: ARPrefixCacheKey,
    ) -> bool:
        if not self.enabled:
            return False
        return cached_key == requested_key


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

    def free(self, sequence_states: Sequence[Any]) -> None:
        del sequence_states

    def debug_info(self) -> Mapping[str, Any]:
        return {
            "family": self.config.family,
            "model_key": self.config.model_key,
            "cache_layout_version": self.config.cache_layout_version,
        }


def _require_embed_mask_batch(inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> None:
    if inputs_embeds.ndim != 3:
        raise ValueError("inputs_embeds must have shape [B, T, H]")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [B, T]")
    if inputs_embeds.shape[0] != attention_mask.shape[0]:
        raise ValueError("inputs_embeds and attention_mask batch sizes must match")


def _require_last_hidden_batch(last_hidden: torch.Tensor, batch_size: int) -> None:
    if batch_size < 1:
        raise ValueError("sequence_states must be non-empty")
    if last_hidden.ndim not in (2, 3):
        raise ValueError("last_hidden must have shape [B, H] or [B, 1, H]")
    if last_hidden.shape[0] != batch_size:
        raise ValueError("last_hidden batch size must match sequence_states")


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256(payload).hexdigest()


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
]
