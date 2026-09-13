"""Real CUDA residency transitions for native-precision adapter-only FSDP."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.trainers._strategy_policies import free_port
from tests.trainers.online.test_fsdp_streaming_equivalence import _Policy
from vrl.trainers.distributed import DistributedTrainingContext
from vrl.trainers.strategy import FSDPStrategy, TrainingMemoryState


def _worker(rank, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    strategy = FSDPStrategy(
        DistributedTrainingContext("fsdp", rank, 4, device),
        mesh_dims=["dp_shard"],
        precision_policy="none",
        reshard_after_forward=True,
        cpu_offload=False,
        shard_trainable_only=True,
    )
    try:
        model = strategy.prepare_model(_Policy())
        assert all(getattr(p, "_local_tensor", p).device == device for p in model.parameters())
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=0.01)
        ema = SimpleNamespace(
            ema_parameters=[p.detach().clone() for p in parameters],
            temp_stored_parameters=[],
            device=device,
        )
        memory = TrainingMemoryState(
            model=model,
            ref_model=None,
            optimizer=optimizer,
            ema=ema,
            grad_scaler=None,
            device=device,
        )
        for step in range(3):
            identities = [id(p) for p in model.parameters()]
            snapshots = [
                getattr(p, "_local_tensor", p).detach().cpu().clone() for p in model.parameters()
            ]
            strategy.park_training_state(memory)
            for parameter, expected in zip(model.parameters(), snapshots, strict=True):
                local = getattr(parameter, "_local_tensor", parameter)
                assert local.device.type == "cpu"
                torch.testing.assert_close(local, expected)
                if parameter.grad is not None:
                    assert parameter.grad._local_tensor.device.type == "cpu"
            assert all(p._local_tensor.device.type == "cpu" for p in ema.ema_parameters)
            strategy.restore_training_state(memory)
            assert identities == [id(p) for p in model.parameters()]
            for parameter, expected in zip(model.parameters(), snapshots, strict=True):
                local = getattr(parameter, "_local_tensor", parameter)
                assert local.device == device
                torch.testing.assert_close(local.cpu(), expected)
            assert model.transformer.block.base.weight.dtype == torch.bfloat16
            assert model.transformer.block.adapter.weight.dtype == torch.float32
            if step:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            model(torch.tensor([1.0, 2.0], device=device)).square().mean().backward()
        assert all(torch.isfinite(p._local_tensor).all() for p in parameters)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four CUDA devices")
def test_four_gpu_adapter_parking():
    mp.spawn(_worker, args=(free_port(),), nprocs=4, join=True)
