"""Fused Cosmos blocks compute the stock blocks: same outputs, same LoRA gradients."""

from __future__ import annotations

import copy

import pytest
import torch
from peft import LoraConfig, get_peft_model

from tests.models.steps.denoise.fixtures import (
    TINY_COSMOS_LATENT_SHAPE,
    TINY_COSMOS_TEXT_DIM,
    build_tiny_transformer,
)
from vrl.models.families.cosmos.fused_block import (
    FusedCosmosTransformer3DModel,
    FusedCosmosTransformerBlock,
    install_fused_blocks,
)

_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]
_FRAMES = 2


def _lora_transformer() -> torch.nn.Module:
    transformer = build_tiny_transformer("cosmos", num_layers=2)
    transformer.requires_grad_(False)
    model = get_peft_model(transformer, LoraConfig(r=4, lora_alpha=8, target_modules=_TARGETS))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.1)
    return model


def _forward(model: torch.nn.Module, *, per_frame_timesteps: bool) -> torch.Tensor:
    transformer = model.base_model.model
    generator = torch.Generator().manual_seed(0)
    _, channels, _, height, width = TINY_COSMOS_LATENT_SHAPE
    latents = torch.randn(2, channels, _FRAMES, height, width, generator=generator)
    text = torch.randn(2, 3, TINY_COSMOS_TEXT_DIM, generator=generator)
    # Video-to-world conditions the first frame at its own timestep.
    timestep = (
        torch.tensor([[0.0, 0.7], [0.0, 0.4]]).view(2, 1, _FRAMES, 1, 1)
        if per_frame_timesteps
        else torch.tensor([0.7, 0.4])
    )
    return transformer(
        latents,
        timestep,
        text,
        condition_mask=torch.zeros(2, 1, _FRAMES, height, width),
        padding_mask=torch.zeros(1, 1, height, width),
        return_dict=False,
    )[0]


@pytest.mark.parametrize(
    "per_frame_timesteps", [False, True], ids=["shared-timestep", "per-frame-timesteps"]
)
def test_fused_blocks_match_the_stock_ones_with_their_lora_gradients(
    per_frame_timesteps: bool,
) -> None:
    stock = _lora_transformer()
    fused = copy.deepcopy(stock)

    assert install_fused_blocks(fused) == 2
    # The model too: it hands the blocks one timestep row per frame.
    assert type(fused.base_model.model) is FusedCosmosTransformer3DModel
    assert all(
        type(block) is FusedCosmosTransformerBlock
        for block in fused.base_model.model.transformer_blocks
    )
    assert fused.state_dict().keys() == stock.state_dict().keys()

    expected = _forward(stock, per_frame_timesteps=per_frame_timesteps)
    actual = _forward(fused, per_frame_timesteps=per_frame_timesteps)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    expected.square().sum().backward()
    actual.square().sum().backward()
    stock_grads = {n: p.grad for n, p in stock.named_parameters() if p.requires_grad}
    fused_grads = {n: p.grad for n, p in fused.named_parameters() if p.requires_grad}
    assert stock_grads.keys() == fused_grads.keys() and stock_grads
    for name, grad in stock_grads.items():
        torch.testing.assert_close(fused_grads[name], grad, rtol=1e-4, atol=1e-6, msg=name)
