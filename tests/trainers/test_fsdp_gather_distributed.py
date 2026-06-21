"""gather_full_state_dict must materialize the full state on EVERY rank.

Regression for the rank0-only trap: ``get_model_state_dict(full_state_dict=True,
cpu_offload=True)`` returns the full dict only on rank0 and an EMPTY dict on every
other rank. In the symmetric colocated model each rank pushes its gathered weights
to its own colocated rollout, so a non-rank0 empty dict makes select_trainable_state
report every (LoRA) param "missing" — which is exactly what a real 2x1 NCCL run hit
and the world_size=1 CPU test could never catch (the gather is a no-op there).

This spawns a real gloo 2-rank group, shards a PEFT-LoRA toy transformer with FSDP2,
and asserts BOTH ranks gather the full set of trainable keys as plain CPU tensors.
"""

from __future__ import annotations

import os
import socket
from typing import ClassVar

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

peft = pytest.importorskip("peft")

from vrl.trainers.distributed import DistributedTrainingContext  # noqa: E402
from vrl.trainers.fsdp import (  # noqa: E402
    apply_fsdp,
    build_fsdp_mesh,
    gather_full_state_dict,
    mixed_precision_policy,
)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = nn.Linear(16, 16)
        self.to_k = nn.Linear(16, 16)
        self.to_v = nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_q(x) + self.to_k(x) + self.to_v(x)


class _ToyTransformer(nn.Module):
    _no_split_modules: ClassVar[list[str]] = ["_Block"]

    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block() for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.transformer_blocks:
            x = block(x)
        return x


def _free_port() -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def _run_rank(rank: int, world_size: int, port: int, q: mp.Queue) -> None:
    from peft import LoraConfig, get_peft_model

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        model = get_peft_model(
            _ToyTransformer(),
            LoraConfig(r=4, target_modules=["to_q", "to_k", "to_v"], lora_alpha=8),
        )
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        ctx = DistributedTrainingContext(
            strategy="fsdp", distributed=True, rank=rank, local_rank=0,
            world_size=world_size, is_primary=(rank == 0), device=torch.device("cpu"),
        )
        apply_fsdp(
            model,
            mesh=build_fsdp_mesh(ctx, ["dp_shard"]),
            mp_policy=mixed_precision_policy("none"),
            reshard_after_forward=True,
        )
        gathered = gather_full_state_dict(model)
        missing = sorted(trainable - set(gathered))
        all_cpu = all(
            isinstance(v, torch.Tensor) and v.device.type == "cpu" for v in gathered.values()
        )
        q.put((rank, len(gathered), missing, all_cpu))
    finally:
        dist.destroy_process_group()


def test_gather_full_state_dict_is_full_on_every_rank() -> None:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_run_rank, args=(r, 2, port, q)) for r in range(2)]
    for p in procs:
        p.start()
    results = {}
    for _ in range(2):
        rank, n, missing, all_cpu = q.get(timeout=60)
        results[rank] = (n, missing, all_cpu)
    for p in procs:
        p.join(timeout=10)
        assert p.exitcode == 0

    # BOTH ranks must gather the full trainable key set as plain CPU tensors —
    # non-rank0 must NOT be empty.
    for rank in (0, 1):
        n, missing, all_cpu = results[rank]
        assert not missing, f"rank{rank} missing trainable keys: {missing[:5]}"
        assert n > 0, f"rank{rank} gathered an empty state dict"
        assert all_cpu, f"rank{rank} gathered non-CPU tensors"
