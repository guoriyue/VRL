"""The previous / reference policies under FSDP2: snapshots of sharded parameters.

``PolicySnapshot`` clones each rank's shard, swaps it in place for a forward and
blends it with a fused lerp, so it must work on the DTensors ``fully_shard``
leaves behind — across ranks, not just on one. Two gloo ranks shard a toy
transformer on CPU on every run; the same body runs on one nccl rank when a
CUDA device is present (a single card cannot host a two-rank nccl group).
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from tests.trainers._strategy_policies import free_port
from vrl.trainers.distributed import DistributedTrainingContext


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 8)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.proj(inputs))


class _Transformer(nn.Module):
    _no_split_modules = ("_Block",)

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(3)])
        self.head = nn.Linear(8, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            inputs = block(inputs)
        return self.head(inputs)


def _run_snapshot_round_trip(
    rank: int, world_size: int, port: int, queue: mp.Queue, cuda: bool
) -> None:
    from torch.distributed.tensor import DTensor

    from vrl.models.policy_snapshot import PolicySnapshot
    from vrl.trainers.fsdp import apply_fsdp, build_fsdp_mesh, mixed_precision_policy

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    device = torch.device("cuda", rank) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if cuda else "gloo", rank=rank, world_size=world_size)
    try:
        context = DistributedTrainingContext(
            strategy="fsdp", rank=rank, world_size=world_size, device=device
        )
        torch.manual_seed(3)
        model = _Transformer().to(device)
        apply_fsdp(
            model,
            mesh=build_fsdp_mesh(context, ["dp_shard"]),
            mp_policy=mixed_precision_policy("none"),
        )
        trainable = [p for p in model.parameters() if p.requires_grad]
        inputs = torch.arange(16, dtype=torch.float32, device=device).reshape(2, 8) / 16

        snapshot = PolicySnapshot(trainable)
        sharded = all(isinstance(p, DTensor) for p in trainable) and all(
            isinstance(shadow, DTensor) for shadow in snapshot.shadows
        )
        start = [p.detach().full_tensor().clone() for p in trainable]
        with torch.no_grad():
            reference_output = model(inputs).clone()
            for p in trainable:
                p.add_(0.25)
            moved_output = model(inputs).clone()
            with snapshot.active():
                swapped_back = torch.equal(model(inputs), reference_output)
                shards_restored = all(
                    torch.equal(p.detach().full_tensor(), before)
                    for p, before in zip(trainable, start, strict=True)
                )
            live_kept = torch.equal(model(inputs), moved_output)
            snapshot.update(0.5)
            with snapshot.active():
                blended = all(
                    torch.allclose(p.detach().full_tensor(), before + 0.125)
                    for p, before in zip(trainable, start, strict=True)
                )
            snapshot.update(0.0)
            with snapshot.active():
                copied = torch.equal(model(inputs), moved_output)
        queue.put(
            (
                rank,
                sharded,
                swapped_back,
                shards_restored,
                live_kept,
                blended,
                copied,
                not snapshot.state_dict(),
            )
        )
    finally:
        dist.destroy_process_group()


def test_two_rank_snapshot_swaps_blends_and_copies_shards() -> None:
    queue = mp.get_context("spawn").Queue()
    mp.spawn(_run_snapshot_round_trip, args=(2, free_port(), queue, False), nprocs=2, join=True)
    results = {queue.get(timeout=10) for _ in range(2)}
    assert results == {(rank, True, True, True, True, True, True, True) for rank in (0, 1)}


@pytest.mark.gpu
def test_cuda_rank_snapshot_swaps_blends_and_copies_shards() -> None:
    queue = mp.get_context("spawn").Queue()
    mp.spawn(_run_snapshot_round_trip, args=(1, free_port(), queue, True), nprocs=1, join=True)
    assert queue.get(timeout=10) == (0, True, True, True, True, True, True, True)
