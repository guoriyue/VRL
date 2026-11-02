"""Fused Qwen-Image blocks compute the stock blocks: same outputs, same gradients."""

from __future__ import annotations

import copy

import pytest
import torch
from peft import LoraConfig, get_peft_model

from tests.models.steps.denoise.fixtures import (
    TINY_QWEN_IN_CHANNELS,
    TINY_QWEN_JOINT_DIM,
    build_tiny_transformer,
)
from vrl.models.families.qwen_image.fused_block import (
    FusedQwenImageTransformerBlock,
    install_fused_blocks,
)

_LORA_TARGETS = ["to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"]


def _lora_transformer() -> torch.nn.Module:
    transformer = build_tiny_transformer("qwen_image", num_layers=2)
    transformer.requires_grad_(False)
    model = get_peft_model(
        transformer, LoraConfig(r=4, lora_alpha=8, target_modules=_LORA_TARGETS)
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.1)
    return model


def _call(model: torch.nn.Module, mask: torch.Tensor | None) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return model(
        hidden_states=torch.randn(2, 16, TINY_QWEN_IN_CHANNELS, generator=generator),
        encoder_hidden_states=torch.randn(2, 5, TINY_QWEN_JOINT_DIM, generator=generator),
        encoder_hidden_states_mask=mask,
        timestep=torch.tensor([0.3, 0.7]),
        img_shapes=[[(1, 4, 4)]] * 2,
        return_dict=False,
    )[0]


@pytest.mark.parametrize(
    "mask",
    [None, torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])],
    ids=["unmasked", "padded-text"],
)
def test_fused_transformer_matches_the_stock_one_with_its_lora_gradients(
    mask: torch.Tensor | None,
) -> None:
    stock = _lora_transformer()
    fused = copy.deepcopy(stock)

    assert install_fused_blocks(fused, keep_lora_input_cast=True) == 2
    assert all(
        type(block) is FusedQwenImageTransformerBlock
        for block in fused.base_model.model.transformer_blocks
    )
    assert fused.state_dict().keys() == stock.state_dict().keys()

    expected, actual = _call(stock, mask), _call(fused, mask)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    expected.square().sum().backward()
    actual.square().sum().backward()
    stock_grads = {n: p.grad for n, p in stock.named_parameters() if p.requires_grad}
    fused_grads = {n: p.grad for n, p in fused.named_parameters() if p.requires_grad}
    assert stock_grads.keys() == fused_grads.keys() and stock_grads
    for name, grad in stock_grads.items():
        torch.testing.assert_close(fused_grads[name], grad, rtol=1e-4, atol=1e-6, msg=name)


def test_fused_blocks_install_once_and_before_anything_subclasses_them() -> None:
    model = _lora_transformer()
    assert install_fused_blocks(model, keep_lora_input_cast=False) == 2
    assert install_fused_blocks(model, keep_lora_input_cast=False) == 2
    lora_layers = [m for n, m in model.named_modules() if n.endswith("to_q")]
    assert lora_layers and all(not m.cast_input_dtype_enabled for m in lora_layers)

    other = _lora_transformer()
    block = other.base_model.model.transformer_blocks[0]
    block.__class__ = type("Wrapped", (type(block),), {})
    with pytest.raises(TypeError, match="install fused blocks before"):
        install_fused_blocks(other, keep_lora_input_cast=True)
