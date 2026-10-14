"""Process-group smoke for multi-rank engines — CPU + gloo, no GPU needed.

Two real OS processes rendezvous through the same ProcessGroupConfig the launcher
stamps, exchange tensors through all_gather, and leave the group cleanly.
This exercises CPU rendezvous and collective lifecycle; it does not verify
NCCL collectives or GPU placement.
"""

from __future__ import annotations

import multiprocessing
import socket
from types import SimpleNamespace

import pytest

from vrl.generation.execution.rank_group import EngineProcessGroup, ProcessGroupConfig


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _rank_main(rank: int, world: int, port: int, queue: multiprocessing.Queue) -> None:
    try:
        import torch
        import torch.distributed as dist

        config = ProcessGroupConfig(
            master_addr="127.0.0.1",
            master_port=port,
            group_rank=rank,
            group_world_size=world,
            backend="gloo",
        )
        process_group = EngineProcessGroup.open(config)
        assert process_group.world is dist.group.WORLD
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

            from tests.generation.execution._helpers import launch_config
            from vrl.generation.execution.worker import GenerationWorkerCore
            from vrl.generation.types import GenerationRequest

            core = GenerationWorkerCore("rollout-0", launch_config(), config)
            core.executor = SimpleNamespace(
                execute_request_batches=lambda request, batches, **kwargs: torch.cat(
                    [torch.rand(4), torch.tensor([random.random()])]
                ),
            )
            for iteration in range(2):
                torch.manual_seed(rank + iteration * 100)
                random.seed(rank + iteration * 100)
                output = core.execute_request_batches(
                    GenerationRequest("r", "sd3_5", "t2i", ["p"], 1),
                    (),
                )
                peer_outputs = [torch.empty_like(output) for _ in range(world)]
                dist.all_gather(peer_outputs, output)
                assert all(torch.equal(output, peer) for peer in peer_outputs)
        finally:
            process_group.shutdown()
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
        ProcessGroupConfig("127.0.0.1", 29500, 0, 1)
    with pytest.raises(ValueError, match="outside world"):
        ProcessGroupConfig("127.0.0.1", 29500, 2, 2)
    with pytest.raises(ValueError, match="backend"):
        ProcessGroupConfig("127.0.0.1", 29500, 0, 2, backend="mpi")
    with pytest.raises(ValueError, match="master_port"):
        ProcessGroupConfig("127.0.0.1", 0, 0, 2)


@pytest.mark.parametrize("field", ["master_port", "group_rank", "group_world_size"])
def test_spec_requires_integer_rendezvous_fields(field) -> None:
    settings = dict(master_addr="127.0.0.1", master_port=29500, group_rank=0, group_world_size=2)
    settings[field] = True
    with pytest.raises(ValueError, match=field + " must be an integer"):
        ProcessGroupConfig(**settings)


def test_spec_rejects_nonstring_rendezvous_address() -> None:
    with pytest.raises(ValueError, match="master_addr"):
        ProcessGroupConfig(b"127.0.0.1", 29500, 0, 2)
