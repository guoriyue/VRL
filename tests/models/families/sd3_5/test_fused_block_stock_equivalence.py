"""Fused SD3 blocks compute the stock blocks: same outputs, same LoRA gradients."""

from __future__ import annotations

import copy

import pytest
import torch
from peft import LoraConfig, get_peft_model

from tests.models.steps.denoise.fixtures import (
    TINY_SD3_JOINT_DIM,
    TINY_SD3_LATENT_SHAPE,
    TINY_SD3_POOLED_DIM,
    build_tiny_transformer,
)
from vrl.models.families.sd3_5.fused_block import (
    FusedJointAttnProcessor,
    FusedJointTransformerBlock,
    install_fused_blocks,
)

_TARGETS = [
    "to_q", "to_k", "to_v", "to_out.0", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
    "ff.net.0.proj", "ff.net.2",
]  # fmt: skip


def _lora_transformer(**overrides: object) -> torch.nn.Module:
    # Two blocks: the last one is context_pre_only, with the continuous context norm.
    transformer = build_tiny_transformer("sd3", num_layers=2, **overrides)
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
    return transformer(
        hidden_states=torch.randn(TINY_SD3_LATENT_SHAPE, generator=generator),
        encoder_hidden_states=torch.randn(2, 5, TINY_SD3_JOINT_DIM, generator=generator),
        pooled_projections=torch.randn(2, TINY_SD3_POOLED_DIM, generator=generator),
        timestep=torch.tensor([300.0, 700.0]),
        return_dict=False,
    )[0]


@pytest.mark.parametrize(
    "overrides",
    [{}, {"qk_norm": "rms_norm", "dual_attention_layers": (0,)}],
    ids=["sd3-no-qk-norm", "sd3.5-medium-qk-norm-dual-attention"],
)
def test_fused_blocks_match_the_stock_ones_with_their_lora_gradients(overrides: dict) -> None:
    stock = _lora_transformer(**overrides)
    fused = copy.deepcopy(stock)

    assert install_fused_blocks(fused) == 2
    blocks = fused.base_model.model.transformer_blocks
    assert all(type(block) is FusedJointTransformerBlock for block in blocks)
    fused_processors = [
        type(attn.processor) is FusedJointAttnProcessor
        for block in blocks
        for attn in (block.attn, block.attn2)
        if attn is not None
    ]
    # Only q/k RMSNorms have a fused attention; the norms around it always fuse.
    assert set(fused_processors) == {"qk_norm" in overrides}
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
