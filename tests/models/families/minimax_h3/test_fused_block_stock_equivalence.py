"""Fused MiniMax-H3 blocks compute the stock blocks: same outputs, same LoRA gradients."""

from __future__ import annotations

import copy

import pytest
import torch
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3AttnProcessor
from peft import LoraConfig, get_peft_model

from tests.models.steps.denoise.fixtures import build_tiny_minimax_h3_transformer
from vrl.models.families.minimax_h3.fused_block import (
    FusedMiniMaxH3AttnProcessor,
    FusedMiniMaxH3TransformerBlock,
    install_fused_blocks,
)

_SEQ = 12
_TIMESTEPS = 2


class _HybridAttention(torch.nn.Module):
    """A non-stock attention in the block's ``attn`` slot, as VDN-H3 grafts one."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, hidden_states, rotary_emb=None, attention_mask=None):
        return 0.5 * self.inner(hidden_states, rotary_emb, attention_mask)


def _lora_transformer(*, hybrid: bool = False) -> torch.nn.Module:
    transformer = build_tiny_minimax_h3_transformer(num_layers=2)
    if hybrid:
        for block in transformer.transformer_blocks:
            block.attn = _HybridAttention(block.attn)
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


def _run_blocks(model: torch.nn.Module) -> torch.Tensor:
    transformer = model.base_model.model
    generator = torch.Generator().manual_seed(0)
    hidden_states = torch.randn(2, _SEQ, transformer.config.hidden_size, generator=generator)
    temb = torch.randn(_TIMESTEPS, transformer.config.time_embed_dim, generator=generator)
    adaln_indices = torch.randint(0, 3 * _TIMESTEPS, (_SEQ,), generator=generator)
    position_ids = torch.randint(0, 5, (_SEQ, 3), generator=generator)
    rotary_emb = transformer.rope(position_ids)
    for block in transformer.transformer_blocks:
        hidden_states = block(hidden_states, temb, adaln_indices, rotary_emb)
    return hidden_states


@pytest.mark.parametrize("hybrid", [False, True], ids=["stock-attention", "grafted-attention"])
def test_fused_blocks_match_the_stock_ones_with_their_lora_gradients(hybrid: bool) -> None:
    stock = _lora_transformer(hybrid=hybrid)
    fused = copy.deepcopy(stock)

    assert install_fused_blocks(fused) == 2
    blocks = fused.base_model.model.transformer_blocks
    assert all(type(block) is FusedMiniMaxH3TransformerBlock for block in blocks)
    processors = [type(getattr(block.attn, "inner", block.attn).processor) for block in blocks]
    # A grafted attention keeps its own stock processor inside it.
    expected_processor = MiniMaxH3AttnProcessor if hybrid else FusedMiniMaxH3AttnProcessor
    assert all(processor is expected_processor for processor in processors)
    assert fused.state_dict().keys() == stock.state_dict().keys()

    expected, actual = _run_blocks(stock), _run_blocks(fused)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    expected.square().sum().backward()
    actual.square().sum().backward()
    stock_grads = {n: p.grad for n, p in stock.named_parameters() if p.requires_grad}
    fused_grads = {n: p.grad for n, p in fused.named_parameters() if p.requires_grad}
    assert stock_grads.keys() == fused_grads.keys() and stock_grads
    for name, grad in stock_grads.items():
        torch.testing.assert_close(fused_grads[name], grad, rtol=1e-4, atol=1e-6, msg=name)
