"""Host-streamed frozen blocks: same math, same names, gradients only where they belong."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from vrl.models.host_streaming import is_host_streamed, stream_frozen_blocks


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen = nn.Linear(4, 4)
        self.frozen.requires_grad_(False)
        self.lora = nn.Linear(4, 4, bias=False)
        self.register_buffer("gain", torch.tensor(2.0))

    def forward(self, x: torch.Tensor, *, residual: bool = True) -> torch.Tensor:
        out = self.lora(self.frozen(x)) * self.gain
        return x + out if residual else out


class _Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(4, 4)
        self.blocks = nn.ModuleList([_Block() for _ in range(3)])
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x, residual=True)
        return self.head(x)


def test_streamed_policy_matches_the_plain_one_and_keeps_its_namespace() -> None:
    torch.manual_seed(0)
    plain = _Policy()
    streamed = copy.deepcopy(plain)
    names = [name for name, _ in plain.named_parameters()]
    x = torch.randn(2, 4)

    count = stream_frozen_blocks(streamed, device="cpu")

    assert count == 3 and is_host_streamed(streamed) and not is_host_streamed(plain)
    assert [name for name, _ in streamed.named_parameters()] == names
    assert streamed.state_dict().keys() == plain.state_dict().keys()
    torch.testing.assert_close(streamed(x), plain(x))

    streamed(x).sum().backward()
    plain(x).sum().backward()
    for (name, parameter), (_, reference) in zip(
        streamed.named_parameters(), plain.named_parameters(), strict=True
    ):
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, reference.grad, msg=name)
        else:
            assert parameter.grad is None


def test_streamed_blocks_recompute_under_activation_checkpointing() -> None:
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(1)
    policy = _Policy()
    stream_frozen_blocks(policy, device="cpu")
    x = torch.randn(2, 4, requires_grad=True)

    out = x
    for block in policy.blocks:
        out = checkpoint(block.__call__, out, use_reentrant=False)
    out.sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(block.lora.weight.grad is not None for block in policy.blocks)


def test_a_policy_without_repeated_blocks_cannot_stream() -> None:
    with pytest.raises(ValueError, match="no repeated block list"):
        stream_frozen_blocks(nn.Linear(2, 2), device="cpu")
