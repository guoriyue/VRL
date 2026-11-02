"""Fused Qwen-Image-2.1 blocks compute the stock blocks: same outputs, same LoRA gradients."""

from __future__ import annotations

import copy

import pytest
import torch
from peft import LoraConfig, get_peft_model

from tests.models.steps.denoise.fixtures import build_tiny_transformer
from vrl.models.families.qwen_image_21.fused_block import (
    FusedQwenImage21TransformerBlock,
    install_fused_blocks,
)

_TEXT, _IMAGE = 4, 6


def _lora_transformer() -> torch.nn.Module:
    transformer = build_tiny_transformer("qwen_image_21", num_layers=2)
    transformer.requires_grad_(False)
    model = get_peft_model(
        transformer,
        LoraConfig(r=4, lora_alpha=8, target_modules=["to_q", "to_k", "to_v", "to_out.0"]),
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.1)
    return model


def _run_blocks(model: torch.nn.Module, *, causal_condition: bool) -> torch.Tensor:
    """Two samples over ``[text | target image]``, the segmented prefill when conditioned."""

    transformer = model.base_model.model
    dim = transformer.inner_dim
    head_dim = dim // transformer.config.num_attention_heads
    generator = torch.Generator().manual_seed(0)
    seq = _TEXT + _IMAGE
    hidden_states = torch.randn(2, seq, dim, generator=generator)
    rotary_emb = torch.polar(
        torch.ones(seq, head_dim // 2), 6 * torch.rand(seq, head_dim // 2, generator=generator)
    )
    kwargs = {}
    rows = 2
    if causal_condition:
        rows = 3
        kwargs = {
            "target_token_mask": torch.arange(seq) >= _TEXT,
            "segments": [(0, _TEXT, True)],
        }
    modulation = 0.3 * torch.randn(rows, 4 * dim, generator=generator)
    for block in transformer.transformer_blocks:
        hidden_states = block(hidden_states, modulation, rotary_emb=rotary_emb, **kwargs)
    return hidden_states


@pytest.mark.parametrize(
    "causal_condition", [False, True], ids=["shared-timestep", "causal-condition"]
)
def test_fused_blocks_match_the_stock_ones_with_their_lora_gradients(
    causal_condition: bool,
) -> None:
    stock = _lora_transformer()
    fused = copy.deepcopy(stock)

    assert install_fused_blocks(fused) == 2
    assert all(
        type(block) is FusedQwenImage21TransformerBlock
        for block in fused.base_model.model.transformer_blocks
    )
    assert fused.state_dict().keys() == stock.state_dict().keys()

    expected = _run_blocks(stock, causal_condition=causal_condition)
    actual = _run_blocks(fused, causal_condition=causal_condition)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    expected.square().sum().backward()
    actual.square().sum().backward()
    stock_grads = {n: p.grad for n, p in stock.named_parameters() if p.requires_grad}
    fused_grads = {n: p.grad for n, p in fused.named_parameters() if p.requires_grad}
    assert stock_grads.keys() == fused_grads.keys() and stock_grads
    for name, grad in stock_grads.items():
        torch.testing.assert_close(fused_grads[name], grad, rtol=1e-4, atol=1e-6, msg=name)
