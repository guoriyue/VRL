"""``repeated_block_lists`` finds the outermost homogeneous block lists."""

from __future__ import annotations

from torch import nn

from vrl.nn.optimization.regional_compile import repeated_block_lists


class _Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.proj = nn.Linear(width, width)


class _Dit(nn.Module):
    """Stem, repeated blocks, head — the shape every diffusion transformer has."""

    def __init__(self, width: int = 8, depth: int = 3) -> None:
        super().__init__()
        self.stem = nn.Linear(width, width)
        self.blocks = nn.ModuleList([_Block(width) for _ in range(depth)])
        self.head = nn.Linear(width, width)


def test_repeated_block_lists_selects_outermost_homogeneous_lists() -> None:
    root = nn.Module()
    # A LoRA-style wrapper nests the transformer two levels down.
    root.base_model = nn.Module()
    root.base_model.model = _Dit(depth=3)
    # A block owning its own homogeneous list is covered by the outer block.
    root.base_model.model.blocks[0].experts = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
    root.mixed = nn.ModuleList([nn.Linear(8, 8), nn.LayerNorm(8)])
    root.single = nn.ModuleList([nn.Linear(8, 8)])

    assert list(repeated_block_lists(root)) == ["base_model.model.blocks"]
