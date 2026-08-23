"""Shared AR attention backend selection and builders.

Model families should expose `_lm_trunk()`. Backend names stay global, while
the runtime still passes `family` so config and metrics preserve model-family
identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.nn.layers.attention.paged import (
    ARAttentionConfig,
    VllmPagedAttentionConfig,
)
from vrl.nn.modules.ar_decoder import VllmDecoderPagedAttentionBackend
from vrl.nn.modules.torch_attention import TorchNativeDecoderAttentionBackend

_ATTENTION_BACKENDS = ("torch_native", "vllm_paged")


def attention_backend_name(sampling: Mapping[str, Any]) -> str:
    """Resolve the configured SGLang-style attention backend name."""

    return str(sampling.get("attention_backend", "vllm_paged"))


def resolve_attention_backend(
    family: str,
    name: str,
    model: Any,
    *,
    block_size: int = 16,
    cache_dtype: str = "auto",
) -> VllmDecoderPagedAttentionBackend | TorchNativeDecoderAttentionBackend:
    """Build one of the two supported AR attention backends."""

    if name == "vllm_paged":
        return build_vllm_attention_backend(
            model,
            family=family,
            block_size=block_size,
            cache_dtype=cache_dtype,
        )
    if name == "torch_native":
        return build_torch_native_backend(model, family=family)
    raise ValueError(
        f"unknown attention backend {name!r}; registered={_ATTENTION_BACKENDS}",
    )


def build_vllm_attention_backend(
    model: Any,
    *,
    family: str,
    block_size: int = 16,
    cache_dtype: str = "auto",
) -> VllmDecoderPagedAttentionBackend:
    """Build the shared vLLM paged-attention backend for an AR family model."""

    config = VllmPagedAttentionConfig(
        family=family,
        block_size=block_size,
        cache_dtype=cache_dtype,
    )
    return VllmDecoderPagedAttentionBackend(
        trunk=_lm_trunk(model),
        config=config,
    )


def build_torch_native_backend(
    model: Any,
    *,
    family: str,
) -> TorchNativeDecoderAttentionBackend:
    """Build the shared HF-cache fallback backend for an AR family model."""

    return TorchNativeDecoderAttentionBackend(
        trunk=_lm_trunk(model),
        config=ARAttentionConfig(family=family),
    )


def _lm_trunk(model: Any) -> Any:
    hook = getattr(model, "_lm_trunk", None)
    if not callable(hook):
        raise AttributeError("AR attention backend requires model._lm_trunk()")
    return hook()


__all__ = [
    "attention_backend_name",
    "build_torch_native_backend",
    "build_vllm_attention_backend",
    "resolve_attention_backend",
]
