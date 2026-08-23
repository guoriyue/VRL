"""Tests for trainer replay model contracts."""

from __future__ import annotations

import contextlib
from typing import Any

import pytest

from tests.models.interfaces import registered_replay_model_classes
from vrl.models.interfaces import (
    ReplayModel,
    ReplayRequest,
    ReplayResult,
    ReplaySegmentResult,
    require_replay_model,
    require_replay_segments,
    require_zero_replay_timestep,
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


def test_replay_segment_guard_accepts_supported_selection() -> None:
    require_replay_segments(None, ("denoise",), owner="test")
    require_replay_segments(ReplayRequest(), ("denoise",), owner="test")
    require_replay_segments(
        ReplayRequest(segment_names=("denoise",)),
        ("denoise",),
        owner="test",
    )


@pytest.mark.parametrize("segment_names", [(), ("unsupported",)])
def test_replay_segment_guard_rejects_unsupported_selection(
    segment_names: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="supports segments"):
        require_replay_segments(
            ReplayRequest(segment_names=segment_names),
            ("denoise",),
            owner="test",
        )


def test_replay_timestep_guard_rejects_nonzero_index() -> None:
    require_zero_replay_timestep(0, owner="test")
    with pytest.raises(ValueError, match="timestep_idx must be 0"):
        require_zero_replay_timestep(1, owner="test")


@pytest.mark.parametrize("family", ["emu3", "glm_image", "llamagen", "nextstep_1"])
def test_single_segment_ar_replay_rejects_unsupported_protocol_values(
    family: str,
) -> None:
    replay_cls = registered_replay_model_classes()[family]
    # ``__new__`` without ``__init__``: the guards must fire before any model
    # state is touched, so an uninitialized instance is enough to prove it.
    model = replay_cls.__new__(replay_cls)
    with pytest.raises(ValueError, match="timestep_idx must be 0"):
        replay_cls.replay_forward(model, object(), timestep_idx=1)
    with pytest.raises(ValueError, match="supports segments"):
        replay_cls.replay_forward(
            model,
            object(),
            timestep_idx=0,
            request=ReplayRequest(segment_names=("unsupported",)),
        )


@pytest.mark.parametrize(
    "family",
    [
        "cogvideox",
        "cosmos-predict2",
        "cosmos-predict2-anima",
        "cosmos-predict2.5",
        "cosmos3",
        "echo",
        "flux",
        "hunyuan_image",
        "hunyuan_video",
        "lumina2",
        "mochi",
        "pixart_sigma",
        "qwen_image",
        "sana",
        "sd3_5",
        "wan_2_1",
        "wan_2_1_i2v",
    ],
)
def test_denoise_replay_rejects_unsupported_segment_selection(
    family: str,
) -> None:
    replay_cls = registered_replay_model_classes()[family]
    with pytest.raises(ValueError, match="supports segments"):
        replay_cls.replay_forward(
            object.__new__(replay_cls),
            object(),
            timestep_idx=0,
            request=ReplayRequest(segment_names=("unsupported",)),
        )


@pytest.mark.parametrize("family", ["causvid", "janus_pro", "janus_pro_r1"])
def test_grouped_or_multisegment_replay_rejects_nonzero_timestep(
    family: str,
) -> None:
    replay_cls = registered_replay_model_classes()[family]
    with pytest.raises(ValueError, match="timestep_idx must be 0"):
        replay_cls.replay_forward(object.__new__(replay_cls), object(), timestep_idx=1)


@pytest.mark.parametrize("family", ["causvid", "janus_pro", "janus_pro_r1"])
def test_grouped_or_multisegment_replay_rejects_unsupported_segment(
    family: str,
) -> None:
    replay_cls = registered_replay_model_classes()[family]
    with pytest.raises(ValueError, match="supports segments"):
        replay_cls.replay_forward(
            object.__new__(replay_cls),
            object(),
            request=ReplayRequest(segment_names=("unsupported",)),
        )


def test_replay_model_protocol_accepts_minimal_shape() -> None:
    """Checks replay model protocol accepts minimal shape."""
    assert isinstance(_MinimalReplayModel(), ReplayModel)


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


def test_require_replay_model_reports_missing_methods() -> None:
    """Checks require replay model reports missing methods."""
    with pytest.raises(TypeError, match="missing: disable_adapter"):
        require_replay_model(_IncompleteReplayModel(), owner="test.model")
