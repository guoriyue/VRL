"""Root placement under the ``offload`` section: streamed, or left where a loader dispatched it."""

from __future__ import annotations

import torch
from torch import nn

from vrl.nn.offload import HostStreaming, Offload


class _Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(2)])
        for block in self.blocks:
            block.requires_grad_(False)
        self.head = nn.Linear(2, 2)


def test_the_section_streams_each_host_root_s_frozen_blocks() -> None:
    policy = _Policy()

    streamed = Offload().place({"transformer": policy}, device="cpu")

    assert streamed == 2 and HostStreaming.of(policy) is not None


def test_a_root_a_loader_already_dispatched_is_left_alone() -> None:
    """Placement acts on host-resident roots only; a meta-device root counts as dispatched."""

    with torch.device("meta"):
        dispatched = _Policy()

    streamed = Offload().place({"transformer": dispatched}, device="cpu")

    assert streamed == 0 and HostStreaming.of(dispatched) is None
    assert next(dispatched.parameters()).device.type == "meta"


def test_the_parsed_section_and_its_absence_have_runtime_forms() -> None:
    assert Offload.from_section(None) is None
    assert Offload.from_section({}) == Offload()
