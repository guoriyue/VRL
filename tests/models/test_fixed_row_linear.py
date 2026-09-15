"""Private Linear padding must not alter shapes, parameters or gradients."""

import copy

import pytest
import torch

from vrl.models.precision import fixed_row_linear_compute


@pytest.mark.parametrize("count", [0, 1, 64, 65, 144])
def test_fixed_row_linear_forward_backward_and_restore(count):
    torch.manual_seed(19)
    reference = torch.nn.Linear(5, 7).double()
    candidate = copy.deepcopy(reference)
    expected_input = torch.randn(2, 5, count, dtype=torch.float64).transpose(1, 2).requires_grad_()
    actual_input = expected_input.detach().clone().requires_grad_()
    keys = list(candidate.state_dict())
    parameters = list(candidate.parameters())
    expected = reference(expected_input)
    with fixed_row_linear_compute(candidate):
        actual = candidate(actual_input)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        actual.square().sum().backward()
        with (
            pytest.raises(ValueError, match="already installed"),
            fixed_row_linear_compute(candidate),
        ):
            pass
    expected.square().sum().backward()
    torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=1e-12, atol=1e-12)
    for original, parameter in zip(reference.parameters(), candidate.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, original.grad, rtol=1e-12, atol=1e-12)
    assert list(candidate.state_dict()) == keys
    assert all(a is b for a, b in zip(parameters, candidate.parameters(), strict=True))
    assert "forward" not in candidate.__dict__


def test_fixed_row_fp32_overrides_outer_autocast_and_preserves_custom_forward():
    model = torch.nn.Linear(5, 7)
    observed = []
    original = model.forward

    def record(input):
        observed.append((input.shape[0], input.dtype, torch.is_autocast_enabled("cpu")))
        return original(input)

    model.forward = record
    with (
        torch.autocast("cpu", dtype=torch.bfloat16),
        fixed_row_linear_compute(model, fp32_modules=[model]),
    ):
        output = model(torch.randn(65, 5).bfloat16())
        assert output.dtype == torch.float32
        output.sum().backward()
    assert observed == [(64, torch.float32, False), (64, torch.float32, False)]
    assert model.forward is record


def test_fixed_row_validation_and_exception_cleanup():
    model = torch.nn.Linear(5, 7)
    for rows in (0, -1, True, 1.5):
        with (
            pytest.raises(ValueError, match="positive integer"),
            fixed_row_linear_compute(model, rows=rows),
        ):
            pass
    with (
        pytest.raises(ValueError, match="belong"),
        fixed_row_linear_compute(model, fp32_modules=[torch.nn.Linear(5, 7)]),
    ):
        pass
    with pytest.raises(RuntimeError, match="abort"), fixed_row_linear_compute(model):
        raise RuntimeError("abort")
    assert "forward" not in model.__dict__
    model.bfloat16()
    with (
        pytest.raises(ValueError, match="FP32 parameters"),
        fixed_row_linear_compute(model, fp32_modules=[model]),
    ):
        pass


def test_fixed_row_real_peft_lora_under_bfloat16_autocast():
    from peft import LoraConfig, get_peft_model
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(481)
    model = get_peft_model(
        torch.nn.Sequential(torch.nn.Linear(5, 7)),
        LoraConfig(r=2, lora_alpha=2, target_modules=["0"]),
    ).bfloat16()
    layer = model.base_model.model[0]
    branches = [layer.lora_A["default"].float(), layer.lora_B["default"].float()]
    with torch.no_grad():
        branches[1].weight.normal_(std=0.01)
    events = []
    handles = [
        branch.register_forward_hook(
            lambda module, args, output: events.append((args[0].dtype, output.dtype))
        )
        for branch in branches
    ]
    try:
        with (
            torch.autocast("cpu", dtype=torch.bfloat16),
            fixed_row_linear_compute(model, fp32_modules=branches),
        ):
            output = checkpoint(model, torch.randn(65, 5).bfloat16(), use_reentrant=False)
            assert output.dtype == torch.bfloat16
            output.float().square().mean().backward()
        assert events
        assert all(output_dtype == torch.float32 for _, output_dtype in events)
        for branch in branches:
            assert branch.weight.grad is not None
            assert torch.isfinite(branch.weight.grad).all()
            assert torch.count_nonzero(branch.weight.grad) > 0
        assert layer.base_layer.weight.grad is None
        assert layer.base_layer.weight.dtype == torch.bfloat16
    finally:
        for handle in handles:
            handle.remove()
