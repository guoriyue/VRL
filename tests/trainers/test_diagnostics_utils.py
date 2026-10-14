"""Low-overhead model diagnostic contracts."""

from __future__ import annotations

import hashlib
import os

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn

from tests.trainers._strategy_policies import run_ranks
from vrl.trainers.diagnostics import trainable_state_digest


def test_trainable_state_digest_preserves_plain_tensor_digest() -> None:
    model = nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -2.0, 3.5]]))

    result = trainable_state_digest(model)

    expected = hashlib.sha256()
    expected.update(b"weight")
    expected.update(b"(1, 3)")
    expected.update(b"torch.float32")
    expected.update(model.weight.detach().float().cpu().contiguous().numpy().tobytes())
    assert result == {
        "sha256": expected.hexdigest(),
        "tensor_count": 1,
        "numel": 3,
        "l2_norm": pytest.approx(4.153311931459037),
        "mean_abs": pytest.approx(2.1666666666666665),
        "max_abs": 3.5,
        "dtypes": {"torch.float32": 1},
        "devices": {"cpu": 1},
        "scope": "full_tensors",
    }


def test_trainable_digest_does_not_fall_back_to_frozen_weights():
    model = nn.Linear(3, 1, bias=False).requires_grad_(False)
    before = trainable_state_digest(model)
    with torch.no_grad():
        model.weight.add_(1)
    after = trainable_state_digest(model)
    assert before == after
    assert before["tensor_count"] == 0
    assert before["numel"] == 0


def _run_uneven_shard_digest_rank(
    rank: int,
    world_size: int,
    port: int,
    queue: mp.Queue,
) -> None:
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard, distribute_tensor

    torch.set_num_threads(1)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp_shard",))
        value = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        parameter = nn.Parameter(distribute_tensor(value, mesh, [Shard(0)]))
        model = nn.Module()
        model.register_parameter("weight", parameter)

        def _forbid_full_gather(self, *args, **kwargs):
            del self, args, kwargs
            raise AssertionError("diagnostic digest must not gather a full DTensor")

        original_full_tensor = DTensor.full_tensor
        DTensor.full_tensor = _forbid_full_gather
        try:
            before = trainable_state_digest(model)
            local_shape = tuple(parameter.to_local().shape)
            with torch.no_grad():
                parameter.to_local().add_(rank + 1.0)
            after = trainable_state_digest(model)
        finally:
            DTensor.full_tensor = original_full_tensor

        queue.put(
            (
                rank,
                local_shape,
                before["numel"],
                before["tensor_count"],
                before["scope"],
                before["sha256"],
                after["sha256"],
            ),
        )
    finally:
        dist.destroy_process_group()


def test_trainable_state_digest_hashes_real_uneven_dtensor_shards() -> None:
    import torch.distributed as dist

    if not dist.is_gloo_available():
        pytest.skip("PyTorch was built without gloo")

    # A rank-local digest that joined a collective would hang past the timeout.
    results = run_ranks(_run_uneven_shard_digest_rank, 2, timeout=30)

    assert [results[rank][0] for rank in (0, 1)] == [(3, 3), (2, 3)]
    assert [results[rank][1] for rank in (0, 1)] == [9, 6]
    for rank, (_, _, tensor_count, scope, before_sha, after_sha) in results.items():
        assert tensor_count == 1, f"rank {rank} skipped its trainable DTensor"
        assert scope == "rank_local_dtensor_values"
        assert before_sha != after_sha, f"rank {rank} did not observe its local update"
    assert results[0][4] != results[1][4]


@pytest.fixture
def cpu_device_mesh(cpu_process_group):
    from torch.distributed.device_mesh import init_device_mesh

    return init_device_mesh("cpu", (1,), mesh_dim_names=("dp_shard",))


def test_trainable_state_digest_supports_replicated_dtensor(cpu_device_mesh) -> None:
    from torch.distributed.tensor import Replicate, distribute_tensor

    value = torch.tensor([[1.0, -2.0, 3.5]])
    parameter = nn.Parameter(distribute_tensor(value, cpu_device_mesh, [Replicate()]))
    distributed_model = nn.Module()
    distributed_model.register_parameter("weight", parameter)
    plain_model = nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        plain_model.weight.copy_(value)

    distributed = trainable_state_digest(distributed_model)
    plain = trainable_state_digest(plain_model)

    assert distributed["scope"] == "rank_local_dtensor_values"
    assert distributed["sha256"] == plain["sha256"]
    assert distributed["numel"] == plain["numel"]


def test_trainable_state_digest_rejects_partial_dtensor(cpu_device_mesh) -> None:
    from torch.distributed.tensor import DTensor, Partial

    partial = DTensor.from_local(
        torch.ones(2, 3),
        cpu_device_mesh,
        [Partial()],
        run_check=False,
    )
    model = nn.Module()
    model.register_parameter("weight", nn.Parameter(partial))

    with pytest.raises(
        ValueError,
        match=r"Partial placement.*not a stable parameter value",
    ):
        trainable_state_digest(model)
