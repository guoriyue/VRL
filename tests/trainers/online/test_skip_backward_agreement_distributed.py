"""The skip-backward decision must be unanimous across training ranks.

A backward pass fires cross-rank collectives — FSDP2 per-layer all-gather +
reduce-scatter, or DDP's gradient all-reduce. If one rank skips an all-filtered
(zero-advantage) microbatch while another rank runs it, those collectives
mismatch and the job DEADLOCKS (an unrecoverable NCCL hang). ``_all_ranks_have_work``
all-reduces the local ``has_work`` flag with MIN so every rank takes the SAME
branch: the microbatch runs only when ALL ranks have work. This spawns a real
gloo 2-rank group and asserts the agreed result is the logical AND of the ranks'
local flags.
"""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vrl.trainers.online.trainer import _all_ranks_have_work

# (rank0_has_work, rank1_has_work) -> the agreed result both ranks must return.
_CASES = {
    "both_have_work": ([True, True], True),
    "one_rank_empty": ([True, False], False),
    "both_empty": ([False, False], False),
}


def _free_port() -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def _run_rank(rank: int, world_size: int, port: int, local_flags: list[bool], q: mp.Queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        agreed = _all_ranks_have_work(local_flags[rank], torch.device("cpu"))
        q.put((rank, agreed))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("local_flags", "expected"),
    list(_CASES.values()),
    ids=list(_CASES),
)
def test_skip_backward_decision_is_unanimous(local_flags: list[bool], expected: bool) -> None:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_run_rank, args=(r, 2, port, local_flags, q)) for r in range(2)]
    for p in procs:
        p.start()
    results = {}
    for _ in range(2):
        rank, agreed = q.get(timeout=50)
        results[rank] = agreed
    for p in procs:
        p.join(timeout=10)
        assert p.exitcode == 0
    # Both ranks must agree, and on the AND of the local flags.
    assert results[0] is expected
    assert results[1] is expected


def test_falls_back_to_local_without_process_group() -> None:
    assert _all_ranks_have_work(True, torch.device("cpu")) is True
    assert _all_ranks_have_work(False, torch.device("cpu")) is False
