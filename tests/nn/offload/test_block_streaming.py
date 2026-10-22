"""Host-streamed frozen blocks: same math, same names, gradients only where they belong."""

from __future__ import annotations

import copy
import os
import shutil
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from vrl.nn.offload import HostStreaming


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


def _checkpointed_blocks(policy: _Policy, x: torch.Tensor) -> torch.Tensor:
    out = x
    for block in policy.blocks:
        out = checkpoint(block.__call__, out, use_reentrant=False)
    return out


def test_streamed_policy_matches_the_plain_one_and_keeps_its_namespace() -> None:
    torch.manual_seed(0)
    plain = _Policy()
    streamed = copy.deepcopy(plain)
    names = [name for name, _ in plain.named_parameters()]
    x = torch.randn(2, 4)

    streaming = HostStreaming.install(streamed, device="cpu")

    assert len(streaming.blocks) == 3
    assert HostStreaming.of(streamed) is streaming and HostStreaming.of(plain) is None
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
    torch.manual_seed(1)
    policy = _Policy()
    HostStreaming.install(policy, device="cpu")
    x = torch.randn(2, 4, requires_grad=True)

    _checkpointed_blocks(policy, x).sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(block.lora.weight.grad is not None for block in policy.blocks)


def test_prefetch_follows_the_forward_pass_and_the_recompute_in_backward() -> None:
    """Forward 0,1,2 then recompute 2,1,0: only the first block of each pass misses."""

    policy = _Policy()
    streaming = HostStreaming.install(policy, device="cpu")
    x = torch.randn(2, 4, requires_grad=True)

    _checkpointed_blocks(policy, x).sum().backward()

    assert streaming.counters == {"prefetch_hits": 4, "prefetch_misses": 2}


def test_in_place_compile_shares_one_trace_across_streamed_blocks() -> None:
    """Compile before or after streaming: the compiled frame is the block's own forward.

    The streaming wrapper sits on ``__call__``, outside ``_call_impl`` that
    ``nn.Module.compile`` wraps, so dynamo traces one frame for the three blocks
    and never recompiles for a different block instance.
    """

    import torch._dynamo
    from torch._dynamo.utils import counters

    torch.manual_seed(2)
    plain = _Policy()
    compiled_first = copy.deepcopy(plain)
    streamed_first = copy.deepcopy(plain)
    x = torch.randn(2, 4)

    for block in compiled_first.blocks:
        block.compile(backend="eager")
    HostStreaming.install(compiled_first, device="cpu")
    HostStreaming.install(streamed_first, device="cpu")
    for block in streamed_first.blocks:
        block.compile(backend="eager")

    for policy in (compiled_first, streamed_first):
        torch._dynamo.reset()
        counters.clear()
        torch.testing.assert_close(policy(x), plain(x))
        assert counters["frames"]["ok"] == 1, dict(counters["frames"])
        assert not counters["graph_break"], dict(counters["graph_break"])
        with torch._dynamo.config.patch(error_on_recompile=True):
            torch.testing.assert_close(policy(x), plain(x))


def test_a_policy_without_repeated_blocks_cannot_stream() -> None:
    with pytest.raises(ValueError, match="no repeated block list"):
        HostStreaming.install(nn.Linear(2, 2), device="cpu")


@pytest.mark.gpu
@pytest.mark.parametrize("pin_host", [True, False], ids=["pinned", "pageable"])
def test_cuda_prefetch_matches_the_resident_policy(pin_host: bool, tmp_path: Path) -> None:
    """Every upload path matches the resident policy: pinned anonymous memory, the
    shared tmpfs copy of a file-backed weight, and plain pageable copies."""

    torch.manual_seed(3)
    plain = _Policy().cuda()
    streamed = copy.deepcopy(plain).cpu()
    # One buffer already on the card: it must pass through, not round-trip.
    streamed.blocks[1].gain = streamed.blocks[1].gain.cuda()
    # One weight memory-mapped from a file, as a checkpoint loader leaves it.
    weight = streamed.blocks[2].frozen.weight
    (tmp_path / "w.bin").write_bytes(
        weight.detach().contiguous().view(torch.uint8).numpy().tobytes()
    )
    weight.data = torch.from_file(
        str(tmp_path / "w.bin"), shared=False, size=weight.numel(), dtype=weight.dtype
    ).view(weight.shape)
    # The shared copy must live on tmpfs: the driver refuses to pin pages of a disk file.
    share = Path("/dev/shm") / f"vrl-host-streaming-test-{os.getpid()}"
    try:
        streaming = HostStreaming.install(
            streamed, device="cuda", pin_host=pin_host, share_directory=share
        )
        # Registration pins whole pages, and a 4x4 weight shares its page with
        # whatever the allocator put beside it (possibly a page the other
        # parametrization pinned), so only the file-mapped weight, alone on
        # its pages, can be asserted unpinned.
        if pin_host:
            assert streamed.blocks[0].frozen.weight.is_pinned()
        assert streamed.blocks[2].frozen.weight.is_pinned() is pin_host
        assert bool(list(share.glob("*.bin"))) is pin_host

        x = torch.randn(2, 4, device="cuda", requires_grad=True)
        x_plain = x.detach().clone().requires_grad_(True)
        for _ in range(3):
            torch.testing.assert_close(streamed(x), plain(x))
        _checkpointed_blocks(streamed, x).sum().backward()
        _checkpointed_blocks(plain, x_plain).sum().backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(x.grad, x_plain.grad)
        for (name, parameter), (_, reference) in zip(
            streamed.named_parameters(), plain.named_parameters(), strict=True
        ):
            if parameter.requires_grad:
                torch.testing.assert_close(parameter.grad, reference.grad, msg=name)
                assert parameter.device.type == "cuda"
            else:
                assert parameter.device.type == "cpu"
        assert streaming.counters["prefetch_hits"] > 0
    finally:
        shutil.rmtree(share, ignore_errors=True)
