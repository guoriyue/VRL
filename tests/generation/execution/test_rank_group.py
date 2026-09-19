"""Process-group smoke for multi-rank engines — CPU + gloo, no GPU needed.

Two real OS processes rendezvous through the same RankGroupSpec the launcher
stamps, exchange tensors through all_gather, and leave the group cleanly.
This exercises CPU rendezvous and collective lifecycle; it does not verify
NCCL collectives or GPU placement.
"""

from __future__ import annotations

import multiprocessing
import socket
from types import SimpleNamespace

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
        process_group = init_rank_process_group(spec)
        assert process_group is dist.group.WORLD
        try:
            # gloo lacks all_to_all (an nccl primitive); all_gather moves the
            # same bytes, and the CPU numeric tests for sequence parallelism
            # compose all-to-all from gather + slice on exactly this path.
            send = torch.full((2,), float(rank))
            recv = [torch.empty(2) for _ in range(world)]
            dist.all_gather(recv, send)
            received = [int(chunk[0].item()) for chunk in recv]

            # Exercise the real worker entrypoint with a tiny CPU producer.
            # Rank-local RNGs deliberately differ before every request.
            import random

            from vrl.generation.execution.worker import GenerationWorkerCore
            from vrl.generation.types import GenerationRequest

            core = object.__new__(GenerationWorkerCore)
            core.rank_group_spec = spec
            core._memory_parking = SimpleNamespace(require_active=lambda *args, **kwargs: None)
            core.load_policy = lambda: None
            core._uses_versioned_slots = False
            core._policy_version = None
            core.executor = SimpleNamespace(
                forward_plan_pipelined=lambda *args, **kwargs: torch.cat(
                    [torch.rand(4), torch.tensor([random.random()])]
                ),
            )
            for iteration in range(2):
                torch.manual_seed(rank + iteration * 100)
                random.seed(rank + iteration * 100)
                output = core.execute_request_batches(
                    GenerationRequest("r", "sd3_5", "t2i", ["p"], 1),
                    None,
                    [],
                    completion_callback=None,
                )
                peer_outputs = [torch.empty_like(output) for _ in range(world)]
                dist.all_gather(peer_outputs, output)
                assert all(torch.equal(output, peer) for peer in peer_outputs)
        finally:
            destroy_rank_process_group(process_group)
            assert not dist.is_initialized()
        queue.put((rank, received))
    except BaseException as error:  # pragma: no cover - transported to parent
        queue.put((rank, f"error: {error!r}"))


@pytest.mark.slow_test
def test_two_ranks_rendezvous_all_gather_and_leave() -> None:
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


@pytest.mark.parametrize("field", ["master_port", "group_rank", "group_world_size"])
@pytest.mark.parametrize("value", [True, False, 2.5, "2", float("nan"), float("inf")])
def test_spec_requires_integer_rendezvous_fields(field, value) -> None:
    settings = dict(master_addr="127.0.0.1", master_port=29500, group_rank=0, group_world_size=2)
    settings[field] = value
    with pytest.raises(ValueError, match=field + " must be an integer"):
        RankGroupSpec(**settings)


@pytest.mark.parametrize("address", [123, True, b"127.0.0.1", ["127.0.0.1"]])
def test_spec_rejects_nonstring_rendezvous_address(address) -> None:
    with pytest.raises(ValueError, match="master_addr"):
        RankGroupSpec(address, 29500, 0, 2)
