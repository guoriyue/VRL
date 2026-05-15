"""Tests for trainer replay model contracts."""

from __future__ import annotations

import contextlib
from typing import Any

import pytest

from vrl.models.interfaces import (
    ReplayModel,
    ReplayRequest,
    ReplayResult,
    ReplaySegmentResult,
    require_replay_model,
)


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
    with pytest.raises(ValueError, match="segments must be non-empty"):
        ReplayResult(segments={})


def test_replay_result_requires_matching_segment_key() -> None:
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
    with pytest.raises(TypeError, match="values must be a dict"):
        ReplaySegmentResult(
            segment="image_tokens",
            values=[],  # type: ignore[arg-type]
        )


def test_replay_result_has_no_primary_segment_field() -> None:
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
    with pytest.raises(ValueError, match="segment_names"):
        ReplayRequest(segment_names=("",))


def test_replay_model_protocol_accepts_minimal_shape() -> None:
    assert isinstance(_MinimalReplayModel(), ReplayModel)


def test_require_replay_model_reports_missing_methods() -> None:
    with pytest.raises(TypeError, match="missing: disable_adapter"):
        require_replay_model(_IncompleteReplayModel(), owner="test.model")
