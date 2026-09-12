"""Reference passes must not retain autograd graphs."""

from contextlib import contextmanager

import pytest
import torch

from vrl.rollouts.evaluators.token.ref_pass import reference_model_context


class _Model:
    def __init__(self):
        self.weight = torch.tensor(2.0, requires_grad=True)
        self.adapter_disabled = False

    @contextmanager
    def disable_adapter(self):
        self.adapter_disabled = True
        try:
            yield
        finally:
            self.adapter_disabled = False


@pytest.mark.parametrize("use_reference", [False, True])
def test_reference_pass_disables_gradients_and_restores_caller_state(use_reference):
    model = _Model()
    reference = _Model() if use_reference else None

    def forward(selected):
        assert selected is (reference if use_reference else model)
        assert selected.adapter_disabled is (not use_reference)
        assert not torch.is_grad_enabled()
        return selected.weight.square()

    with torch.enable_grad():
        with reference_model_context(model, reference) as selected:
            result = forward(selected)
        assert torch.is_grad_enabled()
        assert not result.requires_grad
    assert not model.adapter_disabled


@pytest.mark.parametrize("use_reference", [False, True])
def test_reference_failure_restores_grad_and_adapter_state(use_reference):
    model = _Model()
    reference = _Model() if use_reference else None
    failure = RuntimeError("reference forward failed")

    def forward(selected):
        assert not torch.is_grad_enabled()
        raise failure

    with torch.enable_grad():
        with (
            pytest.raises(RuntimeError) as caught,
            reference_model_context(model, reference) as selected,
        ):
            forward(selected)
        assert caught.value is failure
        assert torch.is_grad_enabled()
    assert not model.adapter_disabled
