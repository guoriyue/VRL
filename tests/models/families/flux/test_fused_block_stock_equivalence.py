"""Fused FLUX blocks compute the stock blocks: same outputs, same LoRA gradients."""

from __future__ import annotations

import copy

import torch
from peft import LoraConfig, get_peft_model

from tests.models.steps.denoise.fixtures import (
    TINY_FLUX_IN_CHANNELS,
    TINY_FLUX_JOINT_DIM,
    TINY_FLUX_POOLED_DIM,
    build_tiny_transformer,
)
from vrl.models.families.flux.fused_block import (
    FusedFluxSingleTransformerBlock,
    FusedFluxTransformerBlock,
    install_fused_blocks,
)

_TARGETS = [
    "to_q", "to_k", "to_v", "to_out.0", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
    "ff.net.0.proj", "ff.net.2", "proj_mlp", "proj_out",
]  # fmt: skip
_TEXT, _SIDE = 3, 4


def _lora_transformer() -> torch.nn.Module:
    transformer = build_tiny_transformer("flux", num_layers=2, num_single_layers=2)
    transformer.requires_grad_(False)
    model = get_peft_model(transformer, LoraConfig(r=4, lora_alpha=8, target_modules=_TARGETS))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.1)
    return model


def _forward(model: torch.nn.Module) -> torch.Tensor:
    transformer = model.base_model.model
    generator = torch.Generator().manual_seed(0)
    rows, cols = torch.meshgrid(torch.arange(_SIDE), torch.arange(_SIDE), indexing="ij")
    image_ids = torch.stack([torch.zeros_like(rows), rows, cols], dim=-1).flatten(0, 1).float()
    return transformer(
        hidden_states=torch.randn(2, _SIDE * _SIDE, TINY_FLUX_IN_CHANNELS, generator=generator),
        encoder_hidden_states=torch.randn(2, _TEXT, TINY_FLUX_JOINT_DIM, generator=generator),
        pooled_projections=torch.randn(2, TINY_FLUX_POOLED_DIM, generator=generator),
        timestep=torch.tensor([0.3, 0.8]),
        guidance=torch.tensor([3.5, 3.5]),
        img_ids=image_ids,
        txt_ids=torch.zeros(_TEXT, 3),
        return_dict=False,
    )[0]


def test_fused_blocks_match_the_stock_ones_with_their_lora_gradients() -> None:
    stock = _lora_transformer()
    fused = copy.deepcopy(stock)

    assert install_fused_blocks(fused) == 4
    transformer = fused.base_model.model
    assert all(type(b) is FusedFluxTransformerBlock for b in transformer.transformer_blocks)
    assert all(
        type(b) is FusedFluxSingleTransformerBlock for b in transformer.single_transformer_blocks
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
