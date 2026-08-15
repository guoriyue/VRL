"""Numeric equivalence of Ulysses sequence parallelism — CPU + gloo, no GPU.

A tiny randomly-initialized SD3 transformer runs the same input twice: once
plain (the reference), once as two spawned ranks with the sequence-parallel
install. Every rank's gathered output must match the reference elementwise —
this is the correctness proof that the shard/exchange/gather choreography and
the joint-attention head bookkeeping change nothing but the compute layout.
Covers both processor branches: the interior block and the context_pre_only
final block.
"""

from __future__ import annotations

import multiprocessing
import socket
from pathlib import Path

import pytest

TINY_SD3_CONFIG = {
    "sample_size": 16,
    "patch_size": 2,
    "in_channels": 4,
    "num_layers": 2,
    "attention_head_dim": 8,
    "num_attention_heads": 4,
    "joint_attention_dim": 16,
    "caption_projection_dim": 32,
    "pooled_projection_dim": 8,
    "out_channels": 4,
}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _build_inputs():
    import torch

    torch.manual_seed(20260814)
    return {
        "hidden_states": torch.randn(2, 4, 16, 16),
        "encoder_hidden_states": torch.randn(2, 6, 16),
        "pooled_projections": torch.randn(2, 8),
        "timestep": torch.tensor([3.0, 7.0]),
    }


def _rank_main(
    rank: int,
    world: int,
    port: int,
    weights_path: str,
    queue: multiprocessing.Queue,
) -> None:
    try:
        import torch
        import torch.distributed as dist
        from diffusers import SD3Transformer2DModel

        from vrl.generation.execution.ulysses import install_sd3_sequence_parallel

        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world,
        )
        try:
            model = SD3Transformer2DModel(**TINY_SD3_CONFIG)
            model.load_state_dict(torch.load(weights_path, weights_only=True))
            model.eval()
            install_sd3_sequence_parallel(model, dist.group.WORLD)
            with torch.no_grad():
                output = model(**_build_inputs()).sample
        finally:
            dist.destroy_process_group()
        queue.put((rank, output))
    except BaseException as error:  # pragma: no cover - transported to parent
        queue.put((rank, f"error: {error!r}"))


@pytest.mark.slow_test
def test_two_rank_forward_matches_the_single_rank_reference(tmp_path: Path) -> None:
    import torch
    from diffusers import SD3Transformer2DModel

    torch.manual_seed(20260814)
    model = SD3Transformer2DModel(**TINY_SD3_CONFIG)
    model.eval()
    weights_path = tmp_path / "tiny_sd3.pt"
    torch.save(model.state_dict(), weights_path)
    with torch.no_grad():
        reference = model(**_build_inputs()).sample

    context = multiprocessing.get_context("spawn")
    queue: multiprocessing.Queue = context.Queue()
    port = _free_port()
    world = 2
    procs = [
        context.Process(
            target=_rank_main,
            args=(rank, world, port, str(weights_path), queue),
        )
        for rank in range(world)
    ]
    for proc in procs:
        proc.start()
    results = {}
    try:
        for _ in procs:
            rank, payload = queue.get(timeout=300)
            results[rank] = payload
    finally:
        for proc in procs:
            proc.join(timeout=60)
            if proc.is_alive():
                proc.kill()

    errors = {rank: payload for rank, payload in results.items() if isinstance(payload, str)}
    assert not errors, errors
    for rank in range(world):
        torch.testing.assert_close(results[rank], reference, rtol=1e-5, atol=1e-5)


def test_install_requires_a_multi_rank_group() -> None:
    """A one-rank group is the degenerate case the plain path already serves."""

    import torch.distributed as dist

    from vrl.generation.execution.rank_group import destroy_rank_process_group
    from vrl.generation.execution.ulysses import install_sd3_sequence_parallel

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{_free_port()}",
        rank=0,
        world_size=1,
    )
    try:
        with pytest.raises(ValueError, match="rank group of >= 2"):
            install_sd3_sequence_parallel(object(), dist.group.WORLD)
    finally:
        destroy_rank_process_group()
