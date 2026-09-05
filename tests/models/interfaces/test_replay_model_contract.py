"""Tests for trainer replay model contracts."""

from __future__ import annotations

import pytest

from tests.models.interfaces import registered_replay_model_classes
from vrl.models.interfaces import (
    ReplayModel,
    ReplayRequest,
    ReplayResult,
    ReplaySegmentResult,
    require_replay_segments,
    require_zero_replay_timestep,
)

# ReplayModel's required surface. Derived from the protocol's
# ``__protocol_attrs__``, so a method add/rename auto-widens the contract check.
_REPLAY_MODEL_METHODS = tuple(sorted(ReplayModel.__protocol_attrs__))


def test_replay_result_requires_non_empty_segments() -> None:
    """Checks replay result requires non empty segments."""
    with pytest.raises(ValueError, match="segments must be non-empty"):
        ReplayResult(segments={})


def test_replay_result_requires_matching_segment_key() -> None:
    """Checks replay result requires matching segment key."""
    with pytest.raises(ValueError, match="must match"):
        ReplayResult(
            segments={
                "wrong": ReplaySegmentResult(
                    segment="image_tokens",
                    values={},
                ),
            },
        )


def test_replay_request_requires_non_empty_segment_names() -> None:
    """Checks replay request requires non empty segment names."""
    with pytest.raises(ValueError, match="segment_names"):
        ReplayRequest(segment_names=("",))


def test_replay_segment_guard_rejects_unsupported_selection() -> None:
    with pytest.raises(ValueError, match="supports segments"):
        require_replay_segments(
            ReplayRequest(segment_names=("unsupported",)),
            ("denoise",),
            owner="test",
        )


def test_replay_timestep_guard_rejects_nonzero_index() -> None:
    require_zero_replay_timestep(0, owner="test")
    with pytest.raises(ValueError, match="timestep_idx must be 0"):
        require_zero_replay_timestep(1, owner="test")


@pytest.mark.parametrize(
    "family",
    sorted(registered_replay_model_classes()),
)
def test_registered_family_replay_model_satisfies_contract(family: str) -> None:
    """Every registered family's replay-model class satisfies ReplayModel.

    Runs over the family registry (not a hand-written list) so a newly
    registered family cannot silently skip the contract. The check is
    class-level — ``callable(getattr(cls, m))`` like ``_missing_callables`` —
    because instantiating a real family model needs weights/GPU.
    """
    replay_cls = registered_replay_model_classes()[family]
    missing = [m for m in _REPLAY_MODEL_METHODS if not callable(getattr(replay_cls, m, None))]
    assert not missing, f"{family}: {replay_cls.__name__} missing ReplayModel methods {missing}"
