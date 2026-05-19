"""Tests for the AR decoder NN module contract."""

from __future__ import annotations

from vrl.nn.layers.attention.paged import ARPagedAttentionBackend
from vrl.nn.modules.ar_decoder import VllmDecoderPagedAttentionBackend


def test_vllm_decoder_backend_uses_paged_attention_contract() -> None:
    assert issubclass(VllmDecoderPagedAttentionBackend, ARPagedAttentionBackend)
