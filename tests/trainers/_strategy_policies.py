"""Toy policies and the one-rank gloo group shared by the strategy tests.

``test_fsdp.py`` and ``test_ddp.py`` carried byte-identical copies of the toy
DiT, the policy shapes the strategies wrap, and the module-scoped gloo fixture;
five distributed tests each re-wrote the free-port probe, and the spawned
multi-rank tests each re-wrote the spawn/queue/join scaffolding. One owner here.
``test_fsdp_gather_distributed.py`` and ``test_fsdp_fp32_master.py`` keep their
own toys on purpose: the diffusers-named ``transformer_blocks`` and the bf16
blocks are what those tests are about.

The module-scoped ``cpu_process_group`` fixture lives in ``tests/trainers/conftest.py``
(a fixture imported by name trips ruff's F811 at every test that requests it);
pytest builds one group per requesting module, exactly as the inlined copies did.
"""

from __future__ import annotations

import socket
from collections.abc import Callable
from typing import Any, ClassVar

import torch
import torch.multiprocessing as mp
from torch import nn


class ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


class ToyTransformer(nn.Module):
    """Stands in for a diffusers DiT: per-layer blocks named in ``_no_split_modules``.

    DDP never reads ``_no_split_modules`` (only ``vrl/trainers/fsdp.py`` does), so
    the shared class carries it for FSDP and it is inert for DDP.
    """

    _no_split_modules: ClassVar[list[str]] = ["ToyBlock"]

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([ToyBlock() for _ in range(2)])
        self.head = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class FakePolicy:
    """Diffusion-policy shape the strategies need: trainable_modules + writer."""

    def __init__(self, transformer: nn.Module) -> None:
        self.transformer = transformer
        self.set_calls = 0

    def set_module_root(self, name: str, module: nn.Module) -> None:
        if name != "transformer":
            raise ValueError(f"unknown trainable root: {name!r}")
        self.transformer = module
        self.set_calls += 1

    @property
    def trainable_modules(self) -> dict[str, nn.Module]:
        return {"transformer": self.transformer}


class DualStagePolicy(FakePolicy):
    """Wan-style policy with two independently writable trainable roots.

    ONE name-keyed writer serves both roots -- the strategy never needs a
    per-root method to exist under a derived name.
    """

    def __init__(self, transformer: nn.Module) -> None:
        super().__init__(transformer)
        self.transformer_2 = ToyTransformer()
        self.set_2_calls = 0

    def set_module_root(self, name: str, module: nn.Module) -> None:
        if name == "transformer_2":
            self.transformer_2 = module
            self.set_2_calls += 1
            return
        super().set_module_root(name, module)

    @property
    def trainable_modules(self) -> dict[str, nn.Module]:
        return {"transformer": self.transformer, "transformer_2": self.transformer_2}


class Bundle:
    def __init__(self, module: nn.Module) -> None:
        self.trainable_modules = {"transformer": module}


def free_port() -> int:
    """An ephemeral localhost port for a fresh process group (no fixed-port collisions)."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def run_ranks(
    target: Callable[..., None],
    world_size: int,
    *args: Any,
    timeout: float = 120.0,
) -> dict[int, tuple[Any, ...]]:
    """Run ``target(rank, world_size, port, queue, *args)`` on spawned ranks.

    Each rank puts one ``(rank, *values)`` tuple on ``queue``; the result maps
    every rank to its ``values``. A rank that does not report within ``timeout``
    or exits non-zero fails the call (the message carries what the ranks did
    report), and no child outlives it: multiprocessing joins live children at
    interpreter exit, so a wedged rank would otherwise hang the whole session.
    """

    context = mp.get_context("spawn")
    queue = context.Queue()
    port = free_port()
    processes = [
        context.Process(target=target, args=(rank, world_size, port, queue, *args))
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    results: dict[int, tuple[Any, ...]] = {}
    try:
        for _ in processes:
            rank, *values = queue.get(timeout=timeout)
            results[rank] = tuple(values)
        for process in processes:
            process.join(timeout=30)
        exit_codes = [process.exitcode for process in processes]
        assert exit_codes == [0] * world_size, (exit_codes, results)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
    return results
