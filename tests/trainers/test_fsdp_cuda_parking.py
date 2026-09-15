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
from vrl.trainers.online.ema import EMAWeights
from vrl.trainers.strategy import FSDPStrategy, TrainingMemoryState


def _owned_tensors(memory):
    for role, model in (("model", memory.model), ("reference", memory.ref_model)):
        for name, parameter in model.named_parameters():
            yield f"{role}.{name}", parameter
            if parameter.grad is not None:
                yield f"{role}.{name}.grad", parameter.grad
    for index, parameter in enumerate(memory.optimizer.param_groups[0]["params"]):
        for name, tensor in memory.optimizer.state.get(parameter, {}).items():
            if isinstance(tensor, torch.Tensor):
                yield f"adam.{index}.{name}", tensor
    for index, tensor in enumerate(memory.ema.ema_parameters):
        yield f"ema.{index}", tensor
    for index, tensor in enumerate(memory.ema.temp_stored_parameters or []):
        yield f"ema_temp.{index}", tensor


def _snapshot(memory):
    return {
        name: getattr(tensor, "_local_tensor", tensor).detach().cpu().clone()
        for name, tensor in _owned_tensors(memory)
    }


def _assert_exact(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        assert actual[name].dtype == expected[name].dtype, name
        assert torch.equal(actual[name], expected[name]), name


def _equivalence_worker(rank, port):
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
        states = []
        for _ in range(2):
            model = strategy.prepare_model(_Policy())
            parameters = [p for p in model.parameters() if p.requires_grad]
            states.append(
                TrainingMemoryState(
                    model=model,
                    ref_model=_Policy().to(device).requires_grad_(False),
                    optimizer=torch.optim.AdamW(parameters, lr=0.01),
                    ema=EMAWeights(parameters, decay=0.9, device=device),
                    grad_scaler=None,
                    device=device,
                )
            )
        control, parked = states
        for step in range(3):
            for memory in states:
                memory.optimizer.zero_grad(set_to_none=True)
                values = torch.tensor([rank + 1.0, step + 2.0], device=device)
                memory.model(values).square().mean().backward()
                memory.ema.copy_ema_to(memory.optimizer.param_groups[0]["params"])
            _assert_exact(_snapshot(parked), _snapshot(control))
            before = _snapshot(parked)
            devices = {
                name: getattr(tensor, "_local_tensor", tensor).device
                for name, tensor in _owned_tensors(parked)
            }
            identities = [id(p) for p in parked.model.parameters()]
            strategy.park_training_state(parked)
            strategy.park_training_state(parked)
            assert all(
                getattr(t, "_local_tensor", t).device.type == "cpu"
                for _, t in _owned_tensors(parked)
            )
            assert parked.ema.device == torch.device("cpu")
            _assert_exact(_snapshot(parked), before)
            strategy.restore_training_state(parked)
            strategy.restore_training_state(parked)
            assert [id(p) for p in parked.model.parameters()] == identities
            assert parked.ema.device == device
            assert {
                name: getattr(t, "_local_tensor", t).device for name, t in _owned_tensors(parked)
            } == devices
            _assert_exact(_snapshot(parked), before)
            for memory in states:
                parameters = memory.optimizer.param_groups[0]["params"]
                memory.ema.copy_temp_to(parameters)
                memory.optimizer.step()
                memory.ema.step(parameters, step)
                assert memory.ema.num_updates == step + 1
            _assert_exact(_snapshot(parked), _snapshot(control))
            assert any(name.startswith("adam.") for name in _snapshot(parked))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


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


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four CUDA devices")
def test_four_gpu_adapter_parking():
    mp.spawn(_worker, args=(free_port(),), nprocs=4, join=True)


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four CUDA devices")
def test_four_gpu_parking_preserves_next_updates():
    mp.spawn(_equivalence_worker, args=(free_port(),), nprocs=4, join=True)
