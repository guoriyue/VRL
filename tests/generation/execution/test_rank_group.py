"""Process-group smoke for multi-rank engines — CPU + gloo, no GPU needed.

Two real OS processes rendezvous through the same RankGroupSpec the launcher
stamps, exchange tensors through a collective, and leave the group cleanly. This pins the P3 runtime (init /
collective / destroy) on a single-GPU dev box; the nccl path differs only by
backend string and device placement.
"""

from __future__ import annotations

import multiprocessing
import socket

import pytest

from vrl.generation.execution.rank_group import (
    RankGroupSpec,
    destroy_rank_process_group,
    init_rank_process_group,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _rank_main(rank: int, world: int, port: int, queue: multiprocessing.Queue) -> None:
    try:
        import torch
        import torch.distributed as dist

        spec = RankGroupSpec(
            master_addr="127.0.0.1",
            master_port=port,
            group_rank=rank,
            group_world_size=world,
            backend="gloo",
        )
        init_rank_process_group(spec)
        try:
            # gloo lacks all_to_all (an nccl primitive); all_gather moves the
            # same bytes, and the CPU numeric tests for sequence parallelism
            # compose all-to-all from gather + slice on exactly this path.
            send = torch.full((2,), float(rank))
            recv = [torch.empty(2) for _ in range(world)]
            dist.all_gather(recv, send)
            received = [int(chunk[0].item()) for chunk in recv]
        finally:
            destroy_rank_process_group()
        queue.put((rank, received))
    except BaseException as error:  # pragma: no cover - transported to parent
        queue.put((rank, f"error: {error!r}"))


@pytest.mark.slow_test
def test_two_ranks_rendezvous_all_to_all_and_leave() -> None:
    context = multiprocessing.get_context("spawn")
    queue: multiprocessing.Queue = context.Queue()
    port = _free_port()
    world = 2
    procs = [
        context.Process(target=_rank_main, args=(rank, world, port, queue))
        for rank in range(world)
    ]
    for proc in procs:
        proc.start()
    results = {}
    try:
        for _ in procs:
            rank, payload = queue.get(timeout=120)
            results[rank] = payload
    finally:
        for proc in procs:
            proc.join(timeout=30)
            if proc.is_alive():
                proc.kill()

    # Every rank gathered every peer's row in rank order.
    assert results == {0: [0, 1], 1: [0, 1]}


def test_spec_rejects_degenerate_and_out_of_range_shapes() -> None:
    with pytest.raises(ValueError, match="multi-rank"):
        RankGroupSpec("127.0.0.1", 29500, 0, 1)
    with pytest.raises(ValueError, match="outside world"):
        RankGroupSpec("127.0.0.1", 29500, 2, 2)
    with pytest.raises(ValueError, match="backend"):
        RankGroupSpec("127.0.0.1", 29500, 0, 2, backend="mpi")
    with pytest.raises(ValueError, match="master_port"):
        RankGroupSpec("127.0.0.1", 0, 0, 2)
