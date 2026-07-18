"""CPU acceptance tests for FP32 master-weight optimization."""

from __future__ import annotations

import io

import pytest
import torch
from torch import nn

from vrl.trainers.optimizer import FP32MasterWeightOptimizer


def _sgd(
    source: nn.Parameter,
    *,
    lr: float,
    momentum: float = 0.0,
) -> FP32MasterWeightOptimizer:
    return FP32MasterWeightOptimizer(
        [source],
        lambda parameters: torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
        ),
    )


def _constant_gradient_step(
    optimizer: FP32MasterWeightOptimizer,
    source: nn.Parameter,
) -> None:
    source.grad = torch.ones_like(source)
    optimizer.prepare_gradients()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def test_reuses_fp32_sources_and_creates_one_master_for_low_precision() -> None:
    fp16 = nn.Parameter(torch.tensor([1.0], dtype=torch.float16))
    fp32 = nn.Parameter(torch.tensor([2.0], dtype=torch.float32))
    received: tuple[nn.Parameter, ...] | None = None

    def factory(parameters):
        nonlocal received
        received = tuple(parameters)
        return torch.optim.SGD(received, lr=0.1)

    optimizer = FP32MasterWeightOptimizer([fp16, fp32], factory)
    masters = tuple(optimizer.parameters())

    assert received == masters
    assert masters[0] is not fp16
    assert masters[0].dtype == torch.float32
    assert masters[0].shape == fp16.shape
    assert masters[1] is fp32
    assert optimizer.param_groups[0]["params"] == list(masters)


def test_bitsandbytes_factory_receives_fp32_master() -> None:
    bitsandbytes = pytest.importorskip("bitsandbytes")
    source = nn.Parameter(torch.ones(4, dtype=torch.float16))
    created = None

    def factory(parameters):
        nonlocal created
        created = bitsandbytes.optim.AdamW8bit(parameters, lr=1e-3)
        return created

    optimizer = FP32MasterWeightOptimizer([source], factory)

    assert isinstance(created, bitsandbytes.optim.AdamW8bit)
    assert next(optimizer.parameters()).dtype == torch.float32
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_sub_ulp_updates_accumulate_in_master_until_source_can_represent_them() -> None:
    source = nn.Parameter(torch.tensor([1.0], dtype=torch.float16))
    optimizer = _sgd(source, lr=1e-4)
    master = next(optimizer.parameters())

    _constant_gradient_step(optimizer, source)

    assert source.item() == 1.0
    assert master.item() == pytest.approx(0.9999, abs=1e-7)

    for _ in range(2):
        _constant_gradient_step(optimizer, source)

    assert master.item() == pytest.approx(0.9997, abs=1e-7)
    assert source.item() == 0.99951171875


def test_cpu_grad_scaler_unscales_master_before_step_and_clip() -> None:
    source = nn.Parameter(torch.tensor([1.0], dtype=torch.float16))
    optimizer = _sgd(source, lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=128.0)

    scaler.scale(source.sum()).backward()
    assert source.grad is not None
    assert source.grad.item() == 128.0
    optimizer.prepare_gradients()
    assert source.grad is None
    scaler.unscale_(optimizer)

    master = next(optimizer.parameters())
    assert master.grad is not None
    assert master.grad.dtype == torch.float32
    assert master.grad.item() == 1.0
    norm = nn.utils.clip_grad_norm_(optimizer.parameters(), max_norm=0.5)
    assert norm.item() == pytest.approx(1.0)

    scaler.step(optimizer)
    scaler.update()

    assert master.item() == pytest.approx(0.95)
    assert source.item() == pytest.approx(0.9501953125)


def test_cpu_grad_scaler_skips_inf_master_update() -> None:
    source = nn.Parameter(torch.tensor([1.0], dtype=torch.float16))
    optimizer = _sgd(source, lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=128.0)
    master = next(optimizer.parameters())

    scaler.scale(source.sum()).backward()
    assert source.grad is not None
    source.grad.fill_(float("inf"))
    optimizer.prepare_gradients()
    scale_before = scaler.get_scale()

    scaler.unscale_(optimizer)
    scaler.step(optimizer)
    scaler.update()

    assert master.item() == 1.0
    assert source.item() == 1.0
    assert scaler.get_scale() < scale_before
    optimizer.zero_grad(set_to_none=True)
    assert source.grad is None
    assert master.grad is None


@pytest.mark.parametrize("set_to_none", [True, False])
def test_zero_grad_clears_source_and_master(set_to_none: bool) -> None:
    source = nn.Parameter(torch.tensor([1.0], dtype=torch.float16))
    optimizer = _sgd(source, lr=0.1)
    source.grad = torch.ones_like(source)
    optimizer.prepare_gradients()
    master = next(optimizer.parameters())

    optimizer.zero_grad(set_to_none=set_to_none)

    assert source.grad is None
    if set_to_none:
        assert master.grad is None
    else:
        assert master.grad is not None
        assert master.grad.item() == 0.0


def test_checkpoint_round_trip_preserves_invisible_master_residual_and_momentum() -> None:
    source = nn.Parameter(torch.tensor([1.0], dtype=torch.float16))
    uninterrupted = _sgd(source, lr=1e-5, momentum=0.9)
    for _ in range(2):
        _constant_gradient_step(uninterrupted, source)

    master_before = next(uninterrupted.parameters()).detach().clone()
    assert source.item() == 1.0
    assert master_before.item() != source.float().item()
    serialized = io.BytesIO()
    torch.save(uninterrupted.state_dict(), serialized)
    serialized.seek(0)
    checkpoint = torch.load(serialized, weights_only=True)

    restored_source = nn.Parameter(source.detach().clone())
    restored = _sgd(restored_source, lr=1e-5, momentum=0.9)
    restored.load_state_dict(checkpoint)

    assert torch.equal(next(restored.parameters()), master_before)
    assert restored_source.item() == source.item()

    for _ in range(6):
        _constant_gradient_step(uninterrupted, source)
        _constant_gradient_step(restored, restored_source)

    assert torch.equal(next(restored.parameters()), next(uninterrupted.parameters()))
    assert torch.equal(restored_source, source)
    restored_momentum = restored.state[next(restored.parameters())]["momentum_buffer"]
    uninterrupted_momentum = uninterrupted.state[next(uninterrupted.parameters())][
        "momentum_buffer"
    ]
    assert torch.equal(restored_momentum, uninterrupted_momentum)


@pytest.mark.gpu
def test_cuda_adamw8bit_checkpoint_round_trip_preserves_next_update() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the AdamW8bit checkpoint gate")
    bitsandbytes = pytest.importorskip("bitsandbytes")
    device = torch.device("cuda:0")

    def build(source: nn.Parameter) -> FP32MasterWeightOptimizer:
        return FP32MasterWeightOptimizer(
            [source],
            lambda parameters: bitsandbytes.optim.AdamW8bit(
                parameters,
                lr=1e-5,
                weight_decay=0.0,
                min_8bit_size=4096,
            ),
        )

    def update(optimizer: FP32MasterWeightOptimizer, source: nn.Parameter) -> None:
        source.grad = torch.linspace(
            0.5,
            1.5,
            source.numel(),
            device=device,
            dtype=source.dtype,
        ).reshape_as(source)
        optimizer.prepare_gradients()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    source = nn.Parameter(torch.ones(4096, device=device, dtype=torch.float16))
    uninterrupted = build(source)
    update(uninterrupted, source)
    master = next(uninterrupted.parameters())

    assert master.dtype == torch.float32
    assert not torch.equal(master, source.float())
    assert any(
        isinstance(value, torch.Tensor) and value.dtype == torch.uint8
        for value in uninterrupted.state[master].values()
    )

    serialized = io.BytesIO()
    torch.save(uninterrupted.state_dict(), serialized)
    serialized.seek(0)
    checkpoint = torch.load(serialized, map_location="cpu", weights_only=True)
    restored_source = nn.Parameter(source.detach().clone())
    restored = build(restored_source)
    restored.load_state_dict(checkpoint)

    assert torch.equal(next(restored.parameters()), master)
    assert torch.equal(restored_source, source)
    _assert_state_tree_equal(restored.state_dict(), uninterrupted.state_dict())

    update(uninterrupted, source)
    update(restored, restored_source)

    assert torch.equal(next(restored.parameters()), next(uninterrupted.parameters()))
    assert torch.equal(restored_source, source)
    _assert_state_tree_equal(restored.state_dict(), uninterrupted.state_dict())


def test_load_rejects_checkpoint_without_master_residual() -> None:
    source = nn.Parameter(torch.tensor([1.0], dtype=torch.float16))
    optimizer = _sgd(source, lr=0.1)
    checkpoint = optimizer.state_dict()
    del checkpoint["fp32_master_weights"]

    with pytest.raises(ValueError, match="missing fp32_master_weights"):
        optimizer.load_state_dict(checkpoint)


def test_load_rejects_different_wrapped_optimizer_type() -> None:
    source = nn.Parameter(torch.tensor([1.0], dtype=torch.float16))
    checkpoint = _sgd(source, lr=0.1).state_dict()
    restored_source = nn.Parameter(source.detach().clone())
    restored = FP32MasterWeightOptimizer(
        [restored_source],
        lambda parameters: torch.optim.AdamW(parameters, lr=0.1),
    )

    with pytest.raises(ValueError, match="optimizer type does not match"):
        restored.load_state_dict(checkpoint)


def test_step_requires_explicit_gradient_preparation() -> None:
    source = nn.Parameter(torch.tensor([1.0], dtype=torch.float16))
    optimizer = _sgd(source, lr=0.1)
    source.grad = torch.ones_like(source)

    with pytest.raises(RuntimeError, match=r"prepare_gradients\(\)"):
        optimizer.step()

    assert source.item() == 1.0


def _assert_state_tree_equal(actual, expected) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual.detach().cpu(), expected.detach().cpu())
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_state_tree_equal(actual[key], expected[key])
        return
    if isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for got, want in zip(actual, expected, strict=True):
            _assert_state_tree_equal(got, want)
        return
    assert actual == expected
