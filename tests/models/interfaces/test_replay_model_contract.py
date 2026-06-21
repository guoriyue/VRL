"""Tests for trainer replay model contracts."""

from __future__ import annotations

import contextlib
from typing import Any

import pytest

from tests.models.interfaces import registered_family_model_classes
from vrl.models.interfaces import (
    ReplayModel,
    ReplayRequest,
    ReplayResult,
    ReplaySegmentResult,
    require_replay_model,
)

# ReplayModel's required surface. Derived from the protocol's
# ``__protocol_attrs__``, so a method add/rename auto-widens the contract check.
_REPLAY_MODEL_METHODS = tuple(sorted(ReplayModel.__protocol_attrs__))


class _MinimalReplayModel:
    def replay_forward(
        self,
        batch: Any,
        timestep_idx: int = 0,
        *,
        request: ReplayRequest | None = None,
    ) -> ReplayResult:
        del batch, timestep_idx, request
        return ReplayResult(
            segments={
                "image_tokens": ReplaySegmentResult(
                    segment="image_tokens",
                    values={"logits": object()},
                ),
            },
        )

    def disable_adapter(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()


class _IncompleteReplayModel:
    def replay_forward(
        self,
        batch: Any,
        timestep_idx: int = 0,
        *,
        request: ReplayRequest | None = None,
    ) -> ReplayResult:
        del batch, timestep_idx
        return ReplayResult(
            segments={
                "image_tokens": ReplaySegmentResult(
                    segment="image_tokens",
                    values={},
                ),
            },
        )


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


def test_replay_segment_result_requires_dict_values() -> None:
    """Checks replay segment result requires dict values."""
    with pytest.raises(TypeError, match="values must be a dict"):
        ReplaySegmentResult(
            segment="image_tokens",
            values=[],  # type: ignore[arg-type]
        )


def test_replay_result_has_no_primary_segment_field() -> None:
    """Checks replay result has no primary segment field."""
    result = ReplayResult(
        segments={
            "image_tokens": ReplaySegmentResult(
                segment="image_tokens",
                values={},
            ),
        },
    )

    assert not hasattr(result, "primary_segment")


def test_replay_request_requires_non_empty_segment_names() -> None:
    """Checks replay request requires non empty segment names."""
    with pytest.raises(ValueError, match="segment_names"):
        ReplayRequest(segment_names=("",))


def test_replay_model_protocol_accepts_minimal_shape() -> None:
    """Checks replay model protocol accepts minimal shape."""
    assert isinstance(_MinimalReplayModel(), ReplayModel)


@pytest.mark.parametrize(
    "family",
    sorted(registered_family_model_classes()),
)
def test_registered_family_replay_model_satisfies_contract(family: str) -> None:
    """Every registered family's replay-model class satisfies ReplayModel.

    Runs over the family registry (not a hand-written list) so a newly
    registered family cannot silently skip the contract. The check is
    class-level — ``callable(getattr(cls, m))`` like ``_missing_callables`` —
    because instantiating a real family model needs weights/GPU.
    """
    _runtime_cls, replay_cls = registered_family_model_classes()[family]
    missing = [m for m in _REPLAY_MODEL_METHODS if not callable(getattr(replay_cls, m, None))]
    assert not missing, f"{family}: {replay_cls.__name__} missing ReplayModel methods {missing}"


def test_require_replay_model_reports_missing_methods() -> None:
    """Checks require replay model reports missing methods."""
    with pytest.raises(TypeError, match="missing: disable_adapter"):
        require_replay_model(_IncompleteReplayModel(), owner="test.model")
