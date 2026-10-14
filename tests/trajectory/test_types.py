"""Trajectory declarations reject ambiguous dimensions and unknown schema values."""

import pytest

from vrl.trajectory.types import TrajectoryAxis


@pytest.mark.parametrize("length", [True, -1])
def test_axis_rejects_non_integer_or_negative_length(length) -> None:
    with pytest.raises(ValueError, match=r"TrajectoryAxis\.length"):
        TrajectoryAxis("denoise", "denoise_step", length)


def test_axis_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match=r"TrajectoryAxis\.kind"):
        TrajectoryAxis("denoise", "denosie_step", 2)


@pytest.mark.parametrize("name", ["", None])
def test_axis_requires_a_string_name(name) -> None:
    with pytest.raises(ValueError, match=r"TrajectoryAxis\.name"):
        TrajectoryAxis(name, "denoise_step", 2)


@pytest.mark.parametrize("length", [None, 0])
def test_axis_preserves_unknown_empty_and_positive_lengths(length) -> None:
    axis = TrajectoryAxis("iteration", "denoise_step", length)
    assert axis.length == length


def test_tensor_rejects_unknown_role() -> None:
    from vrl.trajectory.types import TrajectoryTensor

    with pytest.raises(ValueError, match=r"TrajectoryTensor.role"):
        TrajectoryTensor("value", [1], ("sample",), "old_log_probs")


@pytest.mark.parametrize(
    "field, value",
    [
        ("modality", "vedio"),
        ("distribution", "guassian"),
    ],
)
def test_segment_rejects_unknown_schema_values(field, value) -> None:
    from vrl.trajectory.types import TrajectorySegment

    kwargs = dict(
        name="output", modality="latent", distribution="gaussian", trainable=False, tensors={}
    )
    kwargs[field] = value
    with pytest.raises(ValueError, match=rf"TrajectorySegment.{field}"):
        TrajectorySegment(**kwargs)
