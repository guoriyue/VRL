"""Selective FSDP state must materialize trainable tensors on EVERY rank.

Regression for the rank0-only trap: ``get_model_state_dict(full_state_dict=True,
cpu_offload=True)`` returns the full dict only on rank0 and an EMPTY dict on every
other rank. In the symmetric colocated model each rank pushes its gathered weights
to its own colocated rollout, so a non-rank0 empty dict makes select_trainable_state
report every (LoRA) param "missing" — which is exactly what a real 2x1 NCCL run hit
and the world_size=1 CPU test could never catch (the gather is a no-op there).

This spawns a real gloo 2-rank group, shards a PEFT-LoRA toy transformer with FSDP2,
and asserts BOTH ranks gather only the LoRA tensors, then scatter them back on load.
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
    gather_trainable_state_dict,
    load_trainable_state_dict,
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
        base = _ToyTransformer()
        base.register_buffer("frozen_cache", torch.ones(8))
        model = get_peft_model(
            base,
            LoraConfig(r=4, target_modules=["to_q", "to_k", "to_v"], lora_alpha=8),
        )
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        ctx = DistributedTrainingContext(
            strategy="fsdp",
            distributed=True,
            rank=rank,
            local_rank=0,
            world_size=world_size,
            is_primary=(rank == 0),
            device=torch.device("cpu"),
        )
        apply_fsdp(
            model,
            mesh=build_fsdp_mesh(ctx, ["dp_shard"]),
            mp_policy=mixed_precision_policy("none"),
            reshard_after_forward=True,
        )
        frozen_before = {
            name: value
            for name, value in gather_full_state_dict(model).items()
            if name not in trainable
        }

        from torch.distributed.tensor import DTensor

        full_tensor_calls = 0
        original_full_tensor = DTensor.full_tensor

        def _record_full_tensor(self, *args, **kwargs):
            nonlocal full_tensor_calls
            full_tensor_calls += 1
            return original_full_tensor(self, *args, **kwargs)

        DTensor.full_tensor = _record_full_tensor
        try:
            gathered = gather_trainable_state_dict(model)
        finally:
            DTensor.full_tensor = original_full_tensor

        all_cpu = all(
            isinstance(v, torch.Tensor) and v.device.type == "cpu" for v in gathered.values()
        )
        selective = set(gathered) == trainable and full_tensor_calls == len(trainable)

        replacement = {name: torch.full_like(value, 9.0) for name, value in gathered.items()}
        load_trainable_state_dict(model, replacement, strict=True)
        restored = gather_trainable_state_dict(model)
        trainable_restored = all(
            torch.equal(value, torch.full_like(value, 9.0)) for value in restored.values()
        )
        frozen_after = {
            name: value
            for name, value in gather_full_state_dict(model).items()
            if name not in trainable
        }
        frozen_unchanged = frozen_before.keys() == frozen_after.keys() and all(
            torch.equal(value, frozen_after[name]) for name, value in frozen_before.items()
        )
        q.put((rank, selective, all_cpu, trainable_restored, frozen_unchanged))
    finally:
        dist.destroy_process_group()


def test_trainable_state_is_selective_and_round_trips_on_every_rank() -> None:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_run_rank, args=(r, 2, port, q)) for r in range(2)]
    for p in procs:
        p.start()
    results = {}
    for _ in range(2):
        rank, *flags = q.get(timeout=60)
        results[rank] = flags
    for p in procs:
        p.join(timeout=10)
        assert p.exitcode == 0

    for rank in (0, 1):
        selective, all_cpu, trainable_restored, frozen_unchanged = results[rank]
        assert selective, f"rank{rank} gathered a frozen tensor or skipped a LoRA tensor"
        assert all_cpu, f"rank{rank} gathered non-CPU tensors"
        assert trainable_restored, f"rank{rank} did not restore the full LoRA tensors"
        assert frozen_unchanged, f"rank{rank} changed frozen base state during LoRA load"


def _run_optim_ema_rank(rank: int, world_size: int, port: int, q: mp.Queue) -> None:
    """Real 2-rank shard semantics for optimizer resume + EMA (ws=1 gathers are no-ops)."""

    from torch.distributed.tensor import DTensor

    from vrl.trainers.fsdp import (
        gather_full_optimizer_state_dict,
        load_full_optimizer_state_dict,
    )
    from vrl.trainers.online.ema import EMAModuleWrapper

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(0)  # identical init on both ranks pre-shard
        model = _ToyTransformer()
        global_shapes = {n: tuple(p.shape) for n, p in model.named_parameters()}
        ctx = DistributedTrainingContext(
            strategy="fsdp",
            distributed=True,
            rank=rank,
            local_rank=0,
            world_size=world_size,
            is_primary=(rank == 0),
            device=torch.device("cpu"),
        )
        apply_fsdp(
            model,
            mesh=build_fsdp_mesh(ctx, ["dp_shard"]),
            mp_policy=mixed_precision_policy("none"),
            reshard_after_forward=True,
        )
        params = list(model.parameters())

        # One real training step so Adam moments exist and are rank-sharded.
        optimizer = torch.optim.AdamW(params, lr=1e-2)
        torch.manual_seed(7)  # identical batch on both ranks
        model(torch.randn(2, 16)).sum().backward()
        optimizer.step()

        exported = gather_full_optimizer_state_dict(model, optimizer)
        moments_full = all(
            isinstance(entry.get("exp_avg"), torch.Tensor)
            and not isinstance(entry["exp_avg"], DTensor)
            and tuple(entry["exp_avg"].shape) == global_shapes[fqn]
            for fqn, entry in exported["state"].items()
        )
        # Round trip into a fresh optimizer (resume precondition: no grads).
        for p in params:
            p.grad = None
        fresh = torch.optim.AdamW(params, lr=1e-2)
        load_full_optimizer_state_dict(model, fresh, exported)
        reexported = gather_full_optimizer_state_dict(model, fresh)
        optim_round_trip = all(
            torch.equal(entry["exp_avg"], reexported["state"][fqn]["exp_avg"])
            and torch.equal(entry["exp_avg_sq"], reexported["state"][fqn]["exp_avg_sq"])
            for fqn, entry in exported["state"].items()
        )

        # EMA over sharded params: step + checkpoint gather + reshard on load.
        ema = EMAModuleWrapper(params, decay=0.5, update_step_interval=1)
        with torch.no_grad():
            for p in params:
                p.add_(1.0)
        ema.step(params, optimization_step=0)
        state = ema.state_dict()
        ema_full = all(
            isinstance(t, torch.Tensor)
            and not isinstance(t, DTensor)
            and tuple(t.shape) == tuple(shadow.shape)  # DTensor.shape is global
            for t, shadow in zip(state["ema_parameters"], ema.ema_parameters, strict=True)
        )
        restored = EMAModuleWrapper(params, decay=0.5, update_step_interval=1)
        restored.load_state_dict(state)
        ema_round_trip = all(
            isinstance(got, DTensor) and torch.equal(got.full_tensor(), want.full_tensor())
            for got, want in zip(restored.ema_parameters, ema.ema_parameters, strict=True)
        )

        q.put((rank, moments_full, optim_round_trip, ema_full, ema_round_trip))
    finally:
        dist.destroy_process_group()


def test_optimizer_and_ema_full_state_on_every_rank() -> None:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_run_optim_ema_rank, args=(r, 2, port, q)) for r in range(2)]
    for p in procs:
        p.start()
    results = {}
    for _ in range(2):
        rank, *flags = q.get(timeout=120)
        results[rank] = flags
    for p in procs:
        p.join(timeout=10)
        assert p.exitcode == 0

    for rank in (0, 1):
        moments_full, optim_round_trip, ema_full, ema_round_trip = results[rank]
        assert moments_full, f"rank{rank}: optimizer moments not full plain tensors"
        assert optim_round_trip, f"rank{rank}: optimizer state round trip diverged"
        assert ema_full, f"rank{rank}: EMA checkpoint state not full plain tensors"
        assert ema_round_trip, f"rank{rank}: EMA reshard-on-load diverged"
