"""Reusable NN modules backed by model-executor layers and kernels."""

from vrl.nn.modules.ar_attention_backends import (
    attention_backend_name,
    available_attention_backends,
    build_torch_native_backend,
    build_vllm_attention_backend,
    resolve_attention_backend,
)
from vrl.nn.modules.ar_decoder import (
    VllmDecoderPagedAttentionBackend,
    VllmDecoderPagedSequenceState,
)
from vrl.nn.modules.torch_attention import (
    TorchNativeDecoderAttentionBackend,
)

__all__ = [
    "TorchNativeDecoderAttentionBackend",
    "VllmDecoderPagedAttentionBackend",
    "VllmDecoderPagedSequenceState",
    "attention_backend_name",
    "available_attention_backends",
    "build_torch_native_backend",
    "build_vllm_attention_backend",
    "resolve_attention_backend",
]
