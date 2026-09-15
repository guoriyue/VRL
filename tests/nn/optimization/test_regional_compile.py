"""Regional compile: the repeated blocks compile in place, the root stays itself.

The theorem: ``model.torch_compile.regional`` compiles every block of every
repeated block list, changes no math and no ``state_dict`` namespace, and the
compile pass observes that coverage on the modules rather than trusting the
call.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vrl.nn.optimization import apply_rollout_optimizations
from vrl.nn.optimization.regional_compile import (
    compile_repeated_blocks,
    compiled_block_count,
    repeated_block_lists,
)


class _Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.proj = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj(self.norm(x))


class _Dit(nn.Module):
    """Stem, repeated blocks, head — the shape every diffusion transformer has."""

    def __init__(self, width: int = 8, depth: int = 3) -> None:
        super().__init__()
        self.stem = nn.Linear(width, width)
        self.blocks = nn.ModuleList([_Block(width) for _ in range(depth)])
        self.head = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


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


def test_compile_repeated_blocks_keeps_math_and_namespace() -> None:
    torch.manual_seed(0)
    model = _Dit().eval()
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        eager = model(x)
    keys = list(model.state_dict())

    assert compile_repeated_blocks(model, mode="default") == 3

    assert compiled_block_count(model) == 3
    assert not hasattr(model, "_orig_mod")
    assert list(model.state_dict()) == keys
    with torch.no_grad():
        torch.testing.assert_close(model(x), eager)


def test_compile_repeated_blocks_refuses_a_policy_without_blocks() -> None:
    with pytest.raises(ValueError, match="no repeated block list"):
        compile_repeated_blocks(nn.Linear(8, 8), mode="default")


def _build(*, regional: bool) -> SimpleNamespace:
    from vrl.config.precision import RolePrecision

    return SimpleNamespace(
        device="cpu",
        family="test",
        torch_compile={"enable": True, "mode": "default", "regional": regional},
        precision=RolePrecision("bf16", "tf32", None),
        rollout=SimpleNamespace(base_weight_sync=True),
    )


class _Policy:
    quantization_exclude: tuple[str, ...] = ()

    def __init__(self) -> None:
        self._cores: dict[str, nn.Module] = {"transformer": _Dit(), "transformer_2": _Dit()}

    @property
    def policy_cores(self) -> dict[str, nn.Module]:
        return self._cores

    def torch_compile_transformer(self, mode: str, *, regional: bool = False) -> None:
        assert regional is True
        for core in self._cores.values():
            compile_repeated_blocks(core, mode=mode)


def test_compile_pass_observes_regional_coverage_on_the_blocks() -> None:
    model = _Policy()

    apply_rollout_optimizations(model, _build(regional=True))

    assert all(compiled_block_count(core) == 3 for core in model.policy_cores.values())
    assert not any(hasattr(core, "_orig_mod") for core in model.policy_cores.values())


def test_compile_pass_catches_a_core_left_without_compiled_blocks() -> None:
    class _HalfCompiling(_Policy):
        def torch_compile_transformer(self, mode: str, *, regional: bool = False) -> None:
            compile_repeated_blocks(self._cores["transformer"], mode=mode)

    with pytest.raises(RuntimeError, match=r"left transformer_2 uncompiled"):
        apply_rollout_optimizations(_HalfCompiling(), _build(regional=True))
