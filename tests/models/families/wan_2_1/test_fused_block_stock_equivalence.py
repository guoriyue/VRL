"""Fused Wan blocks compute the stock blocks: same outputs, same LoRA gradients."""

from __future__ import annotations

import copy

import pytest
import torch
from peft import LoraConfig, get_peft_model

from tests.models.steps.denoise.fixtures import build_tiny_transformer
from vrl.models.families.wan_2_1.fused_block import FusedWanTransformerBlock, install_fused_blocks

_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "ffn.net.0.proj", "ffn.net.2"]


def _lora_transformer(name: str) -> torch.nn.Module:
    transformer = build_tiny_transformer(name, num_layers=2)
    transformer.requires_grad_(False)
    model = get_peft_model(transformer, LoraConfig(r=4, lora_alpha=8, target_modules=_TARGETS))
    with torch.no_grad():
        for parameter_name, parameter in model.named_parameters():
            if "lora_B" in parameter_name:
                parameter.normal_(std=0.1)
    return model


def _forward(model: torch.nn.Module, *, i2v: bool, per_token_timesteps: bool) -> torch.Tensor:
    transformer = model.base_model.model
    generator = torch.Generator().manual_seed(0)
    channels = transformer.config.in_channels
    latents = torch.randn(2, channels, 2, 4, 4, generator=generator)
    text = torch.randn(2, 3, transformer.config.text_dim, generator=generator)
    # Two frames of 2x2 patches: 8 tokens per sample.
    timestep = (
        torch.randint(0, 1000, (2, 8), generator=generator)
        if per_token_timesteps
        else torch.tensor([300, 700])
    )
    kwargs = {}
    if i2v:
        kwargs["encoder_hidden_states_image"] = torch.randn(
            2, 4, transformer.config.image_dim, generator=generator
        )
    return transformer(latents, timestep, text, return_dict=False, **kwargs)[0]


@pytest.mark.parametrize(
    ("name", "per_token_timesteps"),
    [("wan", False), ("wan", True), ("wan_i2v", False)],
    ids=["t2v", "ti2v-per-token-timesteps", "i2v-image-context"],
)
def test_fused_blocks_match_the_stock_ones_with_their_lora_gradients(
    name: str, per_token_timesteps: bool
) -> None:
    stock = _lora_transformer(name)
    fused = copy.deepcopy(stock)

    assert install_fused_blocks(fused) == 2
    assert all(type(block) is FusedWanTransformerBlock for block in fused.base_model.model.blocks)
    assert fused.state_dict().keys() == stock.state_dict().keys()

    kwargs = {"i2v": name == "wan_i2v", "per_token_timesteps": per_token_timesteps}
    expected = _forward(stock, **kwargs)
    actual = _forward(fused, **kwargs)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    expected.square().sum().backward()
    actual.square().sum().backward()
    stock_grads = {n: p.grad for n, p in stock.named_parameters() if p.requires_grad}
    fused_grads = {n: p.grad for n, p in fused.named_parameters() if p.requires_grad}
    assert stock_grads.keys() == fused_grads.keys() and stock_grads
    for parameter_name, grad in stock_grads.items():
        torch.testing.assert_close(
            fused_grads[parameter_name], grad, rtol=1e-4, atol=1e-6, msg=parameter_name
        )
