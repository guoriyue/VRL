"""Memoized MiniMax-H3 AdaLN projections reproduce the stock blocks, one stored row per timestep."""

from __future__ import annotations

import copy

import pytest
import torch
from peft import LoraConfig, get_peft_model

from tests.models.steps.denoise.fixtures import build_tiny_minimax_h3_transformer
from vrl.models.families.minimax_h3.adaln_memo import memoize_adaln_projections

_SEQ = 12


def _run_blocks(transformer: torch.nn.Module, temb: torch.Tensor) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    hidden_states = torch.randn(1, _SEQ, transformer.config.hidden_size, generator=generator)
    adaln_indices = torch.randint(0, 3 * temb.shape[0], (_SEQ,), generator=generator)
    rotary_emb = transformer.rope(torch.randint(0, 5, (_SEQ, 3), generator=generator))
    for block in transformer.transformer_blocks:
        hidden_states = block(hidden_states, temb, adaln_indices, rotary_emb)
    return hidden_states


def test_memoized_blocks_match_the_stock_ones_and_store_each_timestep_once() -> None:
    stock = build_tiny_minimax_h3_transformer(num_layers=2).requires_grad_(False)
    memoized = copy.deepcopy(stock)
    assert memoize_adaln_projections(memoized) == 2
    # The projection weights leave the module tree: nothing moves or saves them.
    assert not any("adaln_proj" in name for name in memoized.state_dict())

    temb = torch.randn(4, stock.config.time_embed_dim, generator=torch.Generator().manual_seed(1))
    schedule = [temb[:2], temb[1:3], temb[:2].clone(), temb[3:]]
    for step in schedule:
        torch.testing.assert_close(
            _run_blocks(memoized, step), _run_blocks(stock, step), rtol=1e-5, atol=1e-6
        )
    tables = [block.adaln_proj.table for block in memoized.transformer_blocks]
    assert [table.shape[0] for table in tables] == [4, 4]

    # A row is its own GEMM: the timestep that arrived with a neighbour above
    # arrives alone here and stores the same row.
    alone = copy.deepcopy(stock)
    memoize_adaln_projections(alone)
    _run_blocks(alone, temb[1:2])
    assert torch.equal(alone.transformer_blocks[0].adaln_proj.table[0], tables[0][1])


def test_memoized_blocks_carry_the_lora_gradient_of_the_stock_ones() -> None:
    transformer = build_tiny_minimax_h3_transformer(num_layers=2).requires_grad_(False)
    stock = get_peft_model(
        transformer, LoraConfig(r=4, lora_alpha=8, target_modules=["to_q", "to_v"])
    )
    with torch.no_grad():
        for name, parameter in stock.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.1)
    memoized = copy.deepcopy(stock)
    memoize_adaln_projections(memoized)

    temb = torch.randn(2, transformer.config.time_embed_dim)
    for model in (stock, memoized):
        _run_blocks(model.base_model.model, temb).square().sum().backward()
    grads = {name: p.grad for name, p in stock.named_parameters() if p.requires_grad}
    assert grads
    for name, parameter in memoized.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, grads[name], rtol=1e-4, atol=1e-6, msg=name)


def test_an_adapter_on_the_projection_is_refused() -> None:
    transformer = build_tiny_minimax_h3_transformer().requires_grad_(False)
    adapted = get_peft_model(transformer, LoraConfig(r=2, target_modules=["adaln_proj.linear"]))
    with pytest.raises(ValueError, match="cannot be memoized"):
        memoize_adaln_projections(adapted)
