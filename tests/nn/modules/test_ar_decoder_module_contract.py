"""Tests for the AR decoder NN module contract."""

from __future__ import annotations

from vrl.nn.layers.attention.paged import ARPagedAttentionBackend
from vrl.nn.modules.ar_decoder import ARDecoderModule, VllmDecoderPagedAttentionBackend


def test_ar_decoder_module_uses_paged_attention_contract() -> None:
    assert issubclass(ARDecoderModule, ARPagedAttentionBackend)
    assert issubclass(VllmDecoderPagedAttentionBackend, ARDecoderModule)
