from __future__ import annotations

import torch

from vrl.nn.layers.attention.joint import SD3JointAttentionProcessor


def test_sd3_joint_attention_processor_matches_diffusers_reference() -> None:
    from diffusers.models.attention_processor import Attention, JointAttnProcessor2_0

    torch.manual_seed(0)
    attn = Attention(
        query_dim=8,
        cross_attention_dim=None,
        added_kv_proj_dim=8,
        dim_head=4,
        heads=2,
        out_dim=8,
        context_pre_only=False,
        bias=True,
        processor=JointAttnProcessor2_0(),
        qk_norm=None,
        eps=1e-6,
    )
    hidden_states = torch.randn(2, 5, 8)
    encoder_hidden_states = torch.randn(2, 3, 8)

    expected = JointAttnProcessor2_0()(
        attn,
        hidden_states,
        encoder_hidden_states,
    )
    actual = SD3JointAttentionProcessor()(
        attn,
        hidden_states,
        encoder_hidden_states,
    )

    assert isinstance(actual, tuple)
    assert isinstance(expected, tuple)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
