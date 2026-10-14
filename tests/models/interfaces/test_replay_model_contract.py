"""Tests for trainer replay model contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vrl.models.families.registry import FAMILY_REGISTRY
from vrl.models.interfaces import (
    ReplayModel,
    RuntimeModel,
    require_replay_model,
    require_runtime_model,
)
from vrl.models.steps.denoise import DenoiseModelBase
from vrl.utils.config import import_from_path

# ReplayModel's required surface. Derived from the protocol's
# ``__protocol_attrs__``, so a method add/rename auto-widens the contract check.
_REPLAY_MODEL_METHODS = tuple(sorted(ReplayModel.__protocol_attrs__))


def test_every_registered_replay_model_satisfies_contract() -> None:
    """Every registered replay class satisfies ReplayModel.

    Replay recipes are ``DenoiseFamilyBuild`` classes, which inherit the whole
    protocol surface from ``DenoiseModelBase``; the registry walk keeps a newly
    registered replay class outside the base from skipping the contract. The
    check is class-level — ``callable(getattr(cls, m))`` like the runtime
    protocol guard — because instantiating a real family model needs weights/GPU.
    """
    owners: dict[str, type] = {"DenoiseModelBase": DenoiseModelBase}
    for family, entry in FAMILY_REGISTRY.items():
        if not entry.supports_policy_replay:
            continue
        replay_cls = import_from_path(entry.family_build.replay_cls)
        if not issubclass(replay_cls, DenoiseModelBase):
            owners[family] = replay_cls
    for owner, replay_cls in owners.items():
        missing = [m for m in _REPLAY_MODEL_METHODS if not callable(getattr(replay_cls, m, None))]
        assert not missing, f"{owner}: {replay_cls.__name__} missing ReplayModel methods {missing}"


@pytest.mark.parametrize(
    "protocol,guard", [(ReplayModel, require_replay_model), (RuntimeModel, require_runtime_model)]
)
def test_protocol_guard_rejects_noncallable_members(protocol, guard):
    for name in protocol.__protocol_attrs__:
        methods = {method: lambda: None for method in protocol.__protocol_attrs__}
        methods[name] = 42
        with pytest.raises(TypeError, match=rf"missing:.*{name}"):
            guard(SimpleNamespace(**methods))


@pytest.mark.parametrize(
    "protocol,guard", [(ReplayModel, require_replay_model), (RuntimeModel, require_runtime_model)]
)
def test_protocol_guard_returns_callable_implementation(protocol, guard):
    model = SimpleNamespace(**{name: lambda: None for name in protocol.__protocol_attrs__})
    assert guard(model) is model
