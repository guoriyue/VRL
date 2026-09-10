"""Actual disk I/O and numerical acceptance for streamed AdamW moments."""

import copy
from types import SimpleNamespace

import pytest
import torch

from vrl.trainers.core.types import OptimConfig
from vrl.trainers.disk_optimizer import DiskStreamingAdamW
from vrl.trainers.optimizer import FP32MasterWeightOptimizer, build_optimizer


def disk(parameters, directory, bucket_bytes=36):
    return DiskStreamingAdamW(
        parameters,
        directory=directory,
        bucket_bytes=bucket_bytes,
        lr=0.003,
        betas=(0.8, 0.95),
        eps=1e-7,
        weight_decay=0.02,
    )


def gradients(parameters, step):
    for index, parameter in enumerate(parameters):
        parameter.grad = (
            None
            if (step + index) % 4 == 0
            else torch.full_like(parameter, (step + 1) * (index + 1) / 7)
        )


def assert_state_equal(actual, expected):
    assert actual["state"].keys() == expected["state"].keys()
    for index, values in actual["state"].items():
        for key, value in values.items():
            torch.testing.assert_close(
                value.cpu(), expected["state"][index][key].cpu(), rtol=0, atol=0
            )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_multigroup_multistep_portable_resume_and_immutable_export(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    parameters = [
        torch.nn.Parameter(torch.arange(n, dtype=torch.float32, device=device)) for n in (4, 2, 3)
    ]
    reference = [torch.nn.Parameter(p.detach().clone()) for p in parameters]

    def groups(ps):
        return [{"params": ps[:2]}, {"params": ps[2:], "lr": 0.02}]

    streamed = disk(groups(parameters), tmp_path / "scratch")
    resident = torch.optim.AdamW(
        groups(reference),
        lr=0.003,
        betas=(0.8, 0.95),
        eps=1e-7,
        weight_decay=0.02,
        foreach=False,
        fused=False,
    )
    for step in range(8):
        gradients(parameters, step)
        gradients(reference, step)
        streamed.step()
        resident.step()
        assert not streamed.state
        for actual, expected in zip(parameters, reference, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert_state_equal(streamed.state_dict(), resident.state_dict())
    saved = streamed.state_dict()
    frozen = copy.deepcopy(saved)
    # Later steps unlink the old generation, but exported mmap storage must stay valid.
    gradients(parameters, 8)
    streamed.step()
    assert_state_equal(saved, frozen)
    checkpoint = tmp_path / "optimizer.pt"
    torch.save(saved, checkpoint)
    restored_parameters = [torch.nn.Parameter(p.detach().clone()) for p in reference]
    restored = disk(groups(restored_parameters), tmp_path / "other-scratch")
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    gradients(restored_parameters, 8)
    gradients(reference, 8)
    restored.step()
    resident.step()
    for actual, expected in zip(restored_parameters, reference, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert_state_equal(restored.state_dict(), resident.state_dict())


def test_fp32_master_residuals_resume(tmp_path):
    source = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    reference = torch.nn.Parameter(source.detach().float())
    wrapped = FP32MasterWeightOptimizer([source], lambda ps: disk(ps, tmp_path))
    resident = torch.optim.AdamW(
        [reference],
        lr=0.003,
        betas=(0.8, 0.95),
        eps=1e-7,
        weight_decay=0.02,
        foreach=False,
        fused=False,
    )
    for _ in range(5):
        source.grad = torch.ones_like(source)
        reference.grad = torch.ones_like(reference)
        wrapped.prepare_gradients()
        wrapped.step()
        resident.step()
    checkpoint = tmp_path / "masters.pt"
    torch.save(wrapped.state_dict(), checkpoint)
    new_source = torch.nn.Parameter(torch.zeros_like(source))
    restored = FP32MasterWeightOptimizer([new_source], lambda ps: disk(ps, tmp_path))
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    new_source.grad = torch.ones_like(new_source)
    restored.prepare_gradients()
    restored.step()
    resident.step()
    torch.testing.assert_close(next(restored.parameters()), reference, rtol=0, atol=0)
    torch.testing.assert_close(new_source, reference.bfloat16(), rtol=0, atol=0)


@pytest.mark.parametrize("failure", ["full", "short", "cancel", "missing", "corrupt"])
def test_failed_update_is_terminal(tmp_path, monkeypatch, failure):
    parameters = [torch.nn.Parameter(torch.ones(4)) for _ in range(2)]
    optimizer = disk(parameters, tmp_path)
    gradients(parameters, 1)
    optimizer.step()
    if failure == "missing":
        (optimizer._current / "bucket-1.pt").unlink()
    elif failure == "corrupt":
        (optimizer._current / "bucket-1.pt").write_bytes(b"corrupt")
    else:
        original = optimizer._write_bucket

        def fail(directory, index, state):
            if index == 1:
                if failure == "short":
                    (directory / "bucket-1.pt").write_bytes(b"partial")
                if failure == "cancel":
                    raise KeyboardInterrupt()
                raise OSError("injected disk write failure")
            return original(directory, index, state)

        monkeypatch.setattr(optimizer, "_write_bucket", fail)
    gradients(parameters, 2)
    with pytest.raises((OSError, ValueError, KeyboardInterrupt)):
        optimizer.step()
    assert not optimizer.state
    assert optimizer._inner.param_groups is optimizer.param_groups
    for operation in (optimizer.step, optimizer.state_dict):
        with pytest.raises(RuntimeError, match="restore a checkpoint"):
            operation()
    assert not list(tmp_path.glob("adamw-*/pending-*"))


@pytest.mark.parametrize("damage", ["missing_state", "layout", "amsgrad", "fractional_step"])
def test_checkpoint_validation_precedes_publish(tmp_path, damage):
    parameter = torch.nn.Parameter(torch.ones(4))
    optimizer = disk([parameter], tmp_path)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    state = copy.deepcopy(optimizer.state_dict())
    if damage == "missing_state":
        del state["state"][0]
    elif damage == "layout":
        state["disk_streaming"]["layout"]["parameters"][0]["shape"] = [2, 2]
    elif damage == "amsgrad":
        state["param_groups"][0]["amsgrad"] = True
    else:
        state["state"][0]["step"].fill_(1.5)
    before = optimizer._current
    with pytest.raises(ValueError):
        optimizer.load_state_dict(state)
    assert optimizer._current == before
    optimizer.step()


def test_factory_opt_in_and_bounds(tmp_path):
    config = SimpleNamespace(
        optim=OptimConfig(lr=0.01, disk_state_directory=str(tmp_path), disk_state_bucket_bytes=36)
    )
    parameter = torch.nn.Parameter(torch.ones(4))
    assert isinstance(build_optimizer([parameter], config), DiskStreamingAdamW)
    with pytest.raises(ValueError, match="increase bucket_bytes"):
        disk([parameter], tmp_path, bucket_bytes=35)
    with pytest.raises(NotImplementedError, match="plain FP32"):
        disk([torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))], tmp_path)
    with pytest.raises(ValueError, match="incompatible"):
        OptimConfig(lr=0.01, disk_state_directory=str(tmp_path), optim_8bit=True)


def test_only_current_bucket_moments_are_live(tmp_path, monkeypatch):
    parameters = [torch.nn.Parameter(torch.ones(4)) for _ in range(6)]
    optimizer = disk(parameters, tmp_path)
    original = optimizer._inner.step
    observed = []

    def observe():
        original()
        observed.append(
            sum(
                t.numel() * t.element_size()
                for state in optimizer.state.values()
                for t in state.values()
            )
        )
        assert len(optimizer.state) == 1
        assert observed[-1] <= 36

    monkeypatch.setattr(optimizer._inner, "step", observe)
    for _ in range(3):
        for p in parameters:
            p.grad = torch.ones_like(p)
        optimizer.step()
        assert not optimizer.state
    assert len(observed) == 18


def test_restore_io_failure_and_group_reordering_are_rejected(tmp_path, monkeypatch):
    parameters = [torch.nn.Parameter(torch.ones(4)) for _ in range(2)]
    optimizer = disk(parameters, tmp_path)
    gradients(parameters, 1)
    optimizer.step()
    checkpoint = optimizer.state_dict()
    target = disk([torch.nn.Parameter(p.detach().clone()) for p in parameters], tmp_path)

    def fail(*args):
        raise OSError("injected restore failure")

    monkeypatch.setattr(target, "_write_bucket", fail)
    with pytest.raises(OSError):
        target.load_state_dict(checkpoint)
    with pytest.raises(RuntimeError, match="restore a checkpoint"):
        target.step()
    optimizer.param_groups[0]["params"].reverse()
    with pytest.raises(RuntimeError, match="layout changed"):
        optimizer.step()
