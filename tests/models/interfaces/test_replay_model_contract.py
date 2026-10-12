"""Tests for trainer replay model contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.models.interfaces import registered_replay_model_classes
from vrl.models.interfaces import (
    ReplayModel,
    RuntimeModel,
    require_replay_model,
    require_runtime_model,
)
from vrl.models.steps.denoise.base import DenoiseModelBase

# ReplayModel's required surface. Derived from the protocol's
# ``__protocol_attrs__``, so a method add/rename auto-widens the contract check.
_REPLAY_MODEL_METHODS = tuple(sorted(ReplayModel.__protocol_attrs__))


class _WholeTrajectoryReplay(DenoiseModelBase):
    """Replays the whole trajectory at once: it has no timestep axis to select."""

    replay_indexes_timesteps = False

    def forward_step(self, state, step_idx):
        raise NotImplementedError


def test_replay_timestep_guard_rejects_nonzero_index() -> None:
    model = _WholeTrajectoryReplay()
    model.reject_replay_timestep_selection(0)
    with pytest.raises(ValueError, match="timestep_idx must be 0"):
        model.reject_replay_timestep_selection(1)


@pytest.mark.parametrize(
    "family",
    sorted(registered_replay_model_classes()),
)
def test_registered_family_replay_model_satisfies_contract(family: str) -> None:
    """Every registered family's replay-model class satisfies ReplayModel.

    Runs over the family registry (not a hand-written list) so a newly
    registered family cannot silently skip the contract. The check is
    class-level — ``callable(getattr(cls, m))`` like the runtime protocol guard —
    because instantiating a real family model needs weights/GPU.
    """
    replay_cls = registered_replay_model_classes()[family]
    missing = [m for m in _REPLAY_MODEL_METHODS if not callable(getattr(replay_cls, m, None))]
    assert not missing, f"{family}: {replay_cls.__name__} missing ReplayModel methods {missing}"


@pytest.mark.parametrize(
    "protocol,guard", [(ReplayModel, require_replay_model), (RuntimeModel, require_runtime_model)]
)
@pytest.mark.parametrize("bad_method", [None, 42])
def test_protocol_guard_rejects_noncallable_members(protocol, guard, bad_method):
    for name in protocol.__protocol_attrs__:
        methods = {method: lambda: None for method in protocol.__protocol_attrs__}
        methods[name] = bad_method
        with pytest.raises(TypeError, match=rf"missing:.*{name}"):
            guard(SimpleNamespace(**methods))


@pytest.mark.parametrize(
    "protocol,guard", [(ReplayModel, require_replay_model), (RuntimeModel, require_runtime_model)]
)
def test_protocol_guard_returns_callable_implementation(protocol, guard):
    model = SimpleNamespace(**{name: lambda: None for name in protocol.__protocol_attrs__})
    assert guard(model) is model
