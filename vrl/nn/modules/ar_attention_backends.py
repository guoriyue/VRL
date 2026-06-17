"""Shared AR attention backend selection and builders.

Model families should expose `_lm_trunk()`. Backend names stay global, while
the runtime still passes `family` so config and metrics preserve model-family
identity.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

import torch

from vrl.nn.layers.attention.paged import ARAttentionConfig
from vrl.nn.modules.ar_decoder import VllmDecoderPagedAttentionBackend
from vrl.nn.modules.torch_attention import (
    PrefillFn,
    StepFn,
    TorchNativeDecoderAttentionBackend,
    require_past_key_values,
)

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
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    supported = dict(kwargs) if accepts_var_kwargs else {
        key: value for key, value in kwargs.items() if key in parameters
    }
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
        model,
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

    config = _ar_config(
        model,
        family=family,
        backend_label=f"{family}_torch_native_attention",
    )
    prefill_fn, step_fn = _native_decode_fns(
        model,
        backend_label=str(config.extra["backend_label"]),
    )
    return TorchNativeDecoderAttentionBackend(
        config=config,
        prefill_fn=prefill_fn,
        step_fn=step_fn,
    )


def _ar_config(
    model: Any,
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
        model_key=str(getattr(getattr(model, "config", None), "model_path", family)),
        block_size=block_size,
        extra=extras,
    )


def _native_decode_fns(model: Any, *, backend_label: str) -> tuple[PrefillFn, StepFn]:
    trunk = _lm_trunk(model)

    def _prefill(embeds: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, Any]:
        outputs = trunk(
            inputs_embeds=embeds,
            attention_mask=mask,
            use_cache=True,
            output_hidden_states=True,
        )
        return _last_token_hidden(outputs), require_past_key_values(outputs, backend_label)

    def _step(embeds: torch.Tensor, mask: torch.Tensor, kv: Any) -> tuple[torch.Tensor, Any]:
        outputs = trunk(
            inputs_embeds=embeds,
            attention_mask=mask,
            past_key_values=kv,
            use_cache=True,
            output_hidden_states=True,
        )
        return _last_token_hidden(outputs), require_past_key_values(outputs, backend_label)

    return _prefill, _step


def _last_token_hidden(outputs: Any) -> torch.Tensor:
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError(
                "language model output must expose last_hidden_state or hidden_states",
            )
        hidden = hidden_states[-1]
    if hidden.ndim == 3:
        return hidden[:, -1, :]
    if hidden.ndim == 2:
        return hidden
    raise RuntimeError(f"language model hidden state must be rank 2 or 3, got {hidden.ndim}")


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
