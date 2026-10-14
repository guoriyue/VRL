"""Tests for runtime model contracts."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import pytest
import torch
import torch.nn as nn

from vrl.config.precision import RolePrecision
from vrl.models.families.registry import FAMILY_REGISTRY
from vrl.models.interfaces import (
    RuntimeBundle,
    RuntimeModel,
)
from vrl.models.steps.denoise import DenoiseModelBase
from vrl.utils.config import import_from_path

# RuntimeModel's required surface. Derived from the protocol's
# ``__protocol_attrs__``, so a method add/rename auto-widens the contract check.
_RUNTIME_MODEL_METHODS = tuple(sorted(RuntimeModel.__protocol_attrs__))


class _MinimalRuntimeModel:
    def replay_forward(self, batch: Any, timestep_idx: int = 0) -> dict[str, Any]:
        del batch, timestep_idx
        return {}

    def reference_policy(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def load_trainable_state(self, state_dict: dict[str, Any]) -> None:
        del state_dict


class _DiffusionModelBaseStub(DenoiseModelBase):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Linear(2, 1, bias=True)
        self.transformer.bias.requires_grad_(False)

    def forward_step(self, state, step_idx):
        del state, step_idx
        return {}

    @classmethod
    def from_build(cls, build):
        del build
        return cls()

    def apply_lora(self, build):
        del build

    def apply_full_finetune(self, build):
        del build
        return None

    def torch_compile_transformer(self, mode: str, *, regional: bool = False):
        del mode, regional

    def set_num_steps(self, n: int):
        del n

    @property
    def trainable_modules(self) -> dict[str, Any]:
        return {"transformer": self.transformer}

    @property
    def scheduler(self) -> Any:
        return None

    @contextlib.contextmanager
    def reference_policy(self) -> Iterator[None]:
        yield


def _runtime_contract_classes() -> dict[str, type]:
    """``DenoiseModelBase`` plus every registered runtime class that does not inherit it.

    A ``DenoiseModelBase`` subclass inherits the whole protocol surface, so
    checking the base once covers those families; the registry walk keeps a
    newly registered family outside the base from skipping the contract.
    """

    classes: dict[str, type] = {"DenoiseModelBase": DenoiseModelBase}
    for family, entry in FAMILY_REGISTRY.items():
        runtime_cls = import_from_path(entry.family_build.model_cls)
        if not issubclass(runtime_cls, DenoiseModelBase):
            classes[family] = runtime_cls
    return classes


@pytest.mark.parametrize("owner", sorted(_runtime_contract_classes()))
def test_registered_family_runtime_model_satisfies_contract(owner: str) -> None:
    """Every registered family's runtime-model class satisfies RuntimeModel.

    The check is class-level — ``callable(getattr(cls, m))`` like the runtime
    protocol guard — because instantiating a real family model needs weights/GPU.
    """
    runtime_cls = _runtime_contract_classes()[owner]
    missing = [m for m in _RUNTIME_MODEL_METHODS if not callable(getattr(runtime_cls, m, None))]
    assert not missing, f"{owner}: {runtime_cls.__name__} missing RuntimeModel methods {missing}"


def test_runtime_bundle_exposes_model_contract() -> None:
    """``RuntimeBundle`` hands the very ``RolePrecision`` object it holds to the model, so bundle
    precision and ``model.precision`` cannot drift apart.
    """
    model = _MinimalRuntimeModel()
    bundle = RuntimeBundle(
        model=model,
        trainable_modules={},
        scheduler=None,
        precision=RolePrecision("fp32", "ieee", outer_autocast=False),
    )

    assert bundle.model is model
    assert bundle.precision == RolePrecision("fp32", "ieee", outer_autocast=False)
    assert model.precision is bundle.precision


def test_diffusion_load_trainable_state_accepts_trainable_only_payload() -> None:
    """A payload holding exactly the trainable keys loads in place: the transformer weight takes
    the new values.
    """
    model = _DiffusionModelBaseStub()
    new_weight = torch.full_like(model.transformer.weight, 3.0)

    model.load_trainable_state({"transformer.weight": new_weight})

    assert torch.equal(model.transformer.weight, new_weight)


def test_diffusion_load_trainable_state_rejects_frozen_payload() -> None:
    model = _DiffusionModelBaseStub()

    with pytest.raises(ValueError, match="exactly trainable"):
        model.load_trainable_state(
            {
                "transformer.weight": torch.ones_like(model.transformer.weight),
                "transformer.bias": torch.ones_like(model.transformer.bias),
            },
        )
