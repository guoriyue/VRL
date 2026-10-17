"""Tests for runtime model contracts."""

from __future__ import annotations

import contextlib
from typing import Any

import pytest

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
