"""Shared AR attention backend selection and builders.

Model families should expose `_lm_trunk()`. Backend names stay global, while
the runtime still passes `family` so config and metrics preserve model-family
identity.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

from vrl.nn.layers.attention.paged import ARAttentionConfig
from vrl.nn.modules.ar_decoder import VllmDecoderPagedAttentionBackend
from vrl.nn.modules.torch_attention import TorchNativeDecoderAttentionBackend

_backend_builders = {
    "vllm_paged": "build_vllm_attention_backend",
    "torch_native": "build_torch_native_backend",
}


def attention_backend_name(sampling: Mapping[str, Any]) -> str:
    """Resolve the configured SGLang-style attention backend name."""

    return str(sampling.get("attention_backend", "vllm_paged"))


def available_attention_backends() -> tuple[str, ...]:
    """Return supported AR attention backend names."""

    return tuple(sorted(_backend_builders))


def resolve_attention_backend(
    family: str,
    name: str,
    model: Any,
    **kwargs: Any,
) -> VllmDecoderPagedAttentionBackend | TorchNativeDecoderAttentionBackend:
    """Build the selected AR attention backend, forwarding supported kwargs."""

    try:
        builder_name = _backend_builders[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown attention backend {name!r}; registered={available_attention_backends()}",
        ) from exc
    builder = globals()[builder_name]
    parameters = inspect.signature(builder).parameters
    accepts_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    supported = (
        dict(kwargs)
        if accepts_var_kwargs
        else {key: value for key, value in kwargs.items() if key in parameters}
    )
    if "family" in parameters or accepts_var_kwargs:
        supported["family"] = family
    return builder(model, **supported)


def build_vllm_attention_backend(
    model: Any,
    *,
    family: str,
    block_size: int = 16,
    cache_dtype: str = "auto",
) -> VllmDecoderPagedAttentionBackend:
    """Build the shared vLLM paged-attention backend for an AR family model."""

    config = _ar_config(
        family=family,
        block_size=block_size,
        backend_label=f"{family}_vllm_paged_attention",
        extra={"cache_dtype": cache_dtype},
    )
    return VllmDecoderPagedAttentionBackend(
        trunk=_lm_trunk(model),
        config=config,
    )


def build_torch_native_backend(
    model: Any,
    *,
    family: str,
    **_ignored: Any,
) -> TorchNativeDecoderAttentionBackend:
    """Build the shared HF-cache fallback backend for an AR family model."""

    return TorchNativeDecoderAttentionBackend(
        trunk=_lm_trunk(model),
        config=_ar_config(
            family=family,
            backend_label=f"{family}_torch_native_attention",
        ),
    )


def _ar_config(
    *,
    family: str,
    backend_label: str,
    block_size: int = 16,
    extra: dict[str, Any] | None = None,
) -> ARAttentionConfig:
    extras = dict(extra or {})
    extras["backend_label"] = backend_label
    return ARAttentionConfig(
        family=family,
        block_size=block_size,
        extra=extras,
    )


def _lm_trunk(model: Any) -> Any:
    hook = getattr(model, "_lm_trunk", None)
    if not callable(hook):
        raise AttributeError("AR attention backend requires model._lm_trunk()")
    return hook()


__all__ = [
    "attention_backend_name",
    "available_attention_backends",
    "build_torch_native_backend",
    "build_vllm_attention_backend",
    "resolve_attention_backend",
]
