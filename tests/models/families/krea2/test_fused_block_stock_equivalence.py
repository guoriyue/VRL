"""Fused Krea 2 blocks compute the stock blocks: same outputs, same LoRA gradients."""

from __future__ import annotations

import copy

import torch
from peft import LoraConfig, get_peft_model

from tests.models.steps.denoise.fixtures import (
    TINY_KREA2_IN_CHANNELS,
    TINY_KREA2_TEXT_DIM,
    TINY_KREA2_TEXT_LAYERS,
    build_tiny_transformer,
)
from vrl.models.families.krea2.fused_block import FusedKrea2TransformerBlock, install_fused_blocks
from vrl.models.families.krea2.model import Krea2Model

_TARGETS = ["to_q", "to_k", "to_v", "to_gate", "to_out.0", "gate", "up", "down"]
_TEXT, _SIDE = 4, 4


def _lora_transformer() -> torch.nn.Module:
    transformer = build_tiny_transformer("krea2", num_layers=2)
    transformer.requires_grad_(False)
    model = get_peft_model(transformer, LoraConfig(r=4, lora_alpha=8, target_modules=_TARGETS))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            # The norms' zero-centred scales too, so that they are not trivial.
            if "lora_B" in name or "norm" in name:
                parameter.normal_(std=0.1)
    return model


def _forward(model: torch.nn.Module) -> torch.Tensor:
    transformer = model.base_model.model
    generator = torch.Generator().manual_seed(0)
    mask = torch.ones(2, _TEXT, dtype=torch.bool)
    mask[1, 3:] = False  # the second prompt is padded
    return transformer(
        hidden_states=torch.randn(2, _SIDE * _SIDE, TINY_KREA2_IN_CHANNELS, generator=generator),
        encoder_hidden_states=torch.randn(
            2, _TEXT, TINY_KREA2_TEXT_LAYERS, TINY_KREA2_TEXT_DIM, generator=generator
        ),
        timestep=torch.tensor([0.3, 0.8]),
        # 16 pixels per packed token: a _SIDE x _SIDE token grid.
        position_ids=Krea2Model._position_ids(_TEXT, 16 * _SIDE, 16 * _SIDE, "cpu"),
        encoder_attention_mask=mask,
        return_dict=False,
    )[0]


def test_fused_blocks_match_the_stock_ones_with_their_lora_gradients() -> None:
    stock = _lora_transformer()
    fused = copy.deepcopy(stock)

    assert install_fused_blocks(fused) == 2
    assert all(
        type(block) is FusedKrea2TransformerBlock
        for block in fused.base_model.model.transformer_blocks
    )
    assert fused.state_dict().keys() == stock.state_dict().keys()

    expected = _forward(stock)
    actual = _forward(fused)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    expected.square().sum().backward()
    actual.square().sum().backward()
    stock_grads = {n: p.grad for n, p in stock.named_parameters() if p.requires_grad}
    fused_grads = {n: p.grad for n, p in fused.named_parameters() if p.requires_grad}
    assert stock_grads.keys() == fused_grads.keys() and stock_grads
    for name, grad in stock_grads.items():
        torch.testing.assert_close(fused_grads[name], grad, rtol=1e-4, atol=1e-6, msg=name)
