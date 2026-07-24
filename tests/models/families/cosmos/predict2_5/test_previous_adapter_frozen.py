"""NFT's ``previous`` adapter must be frozen so DDP's reducer ignores it.

The previous-policy adapter is forward-only (no_grad replay, refreshed by weight
copy, never optimized). If its PEFT params keep ``requires_grad=True`` the DDP
reducer expects a gradient they never get, and the first backward dies unless
``find_unused_parameters=true`` is forced. The shared freeze/register primitive
freezes exactly the named adapter via PEFT's ``.{name}.`` marker, so
``find_unused_parameters=false`` stays correct while checkpoint resume keeps
the mutable frozen snapshot.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vrl.models.interfaces.runtime import checkpoint_owned_state_names
from vrl.models.steps.denoise.common.lora import (
    freeze_checkpoint_owned_adapter_params,
)


class _PeftLikeModule(nn.Module):
    """Mimics PEFT's param naming: ``...lora_{A,B}.{adapter}.weight``."""

    def __init__(self) -> None:
        super().__init__()
        # one trainable "default" adapter + one "previous" snapshot, both LoRA-shaped
        self.lora_A_default = nn.Parameter(torch.randn(4, 4))
        self.lora_B_default = nn.Parameter(torch.randn(4, 4))
        self.lora_A_previous = nn.Parameter(torch.randn(4, 4))
        self.lora_B_previous = nn.Parameter(torch.randn(4, 4))

    def named_parameters(self, *args, **kwargs):
        # rename to PEFT's dotted convention so the ".{name}." marker matches
        rename = {
            "lora_A_default": "base.lora_A.default.weight",
            "lora_B_default": "base.lora_B.default.weight",
            "lora_A_previous": "base.lora_A.previous.weight",
            "lora_B_previous": "base.lora_B.previous.weight",
        }
        for name, param in super().named_parameters(*args, **kwargs):
            yield rename[name], param


def test_freezes_only_the_named_adapter() -> None:
    m = _PeftLikeModule()
    # all start trainable (PEFT default)
    assert all(p.requires_grad for _, p in m.named_parameters())

    freeze_checkpoint_owned_adapter_params(m, "previous")

    frozen = {n for n, p in m.named_parameters() if not p.requires_grad}
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert frozen == {
        "base.lora_A.previous.weight",
        "base.lora_B.previous.weight",
    }
    # the optimized "default" adapter must stay trainable
    assert trainable == {
        "base.lora_A.default.weight",
        "base.lora_B.default.weight",
    }
    assert checkpoint_owned_state_names(m) == frozen | trainable


def test_raises_if_adapter_absent() -> None:
    m = _PeftLikeModule()
    with pytest.raises(RuntimeError, match="no parameters found for adapter"):
        freeze_checkpoint_owned_adapter_params(m, "nonexistent")

    assert all(parameter.requires_grad for _, parameter in m.named_parameters())
    assert checkpoint_owned_state_names(m) == {name for name, _ in m.named_parameters()}


def test_registration_failure_rolls_back_requires_grad(monkeypatch) -> None:
    m = _PeftLikeModule()

    def reject_registration(*_args, **_kwargs) -> None:
        raise RuntimeError("registration failed")

    monkeypatch.setattr(
        "vrl.models.steps.denoise.common.lora.register_checkpoint_owned_state",
        reject_registration,
    )

    with pytest.raises(RuntimeError, match="registration failed"):
        freeze_checkpoint_owned_adapter_params(m, "previous")

    assert all(parameter.requires_grad for _, parameter in m.named_parameters())
