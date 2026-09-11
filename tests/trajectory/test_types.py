"""Trajectory axis declarations reject ambiguous dimensions and kinds."""

import pytest

from vrl.trajectory import TrajectoryAxis


@pytest.mark.parametrize("length", [True, False, 2.0, 2.5, "2", -1])
def test_axis_rejects_non_integer_or_negative_length(length) -> None:
    with pytest.raises(ValueError, match=r"TrajectoryAxis\.length"):
        TrajectoryAxis("denoise", "denoise_step", length)


@pytest.mark.parametrize("kind", ["denosie_step", "", None])
def test_axis_rejects_unknown_kind(kind) -> None:
    with pytest.raises(ValueError, match=r"TrajectoryAxis\.kind"):
        TrajectoryAxis("denoise", kind, 2)


@pytest.mark.parametrize("name", ["", 1, True, None])
def test_axis_requires_a_string_name(name) -> None:
    with pytest.raises(ValueError, match=r"TrajectoryAxis\.name"):
        TrajectoryAxis(name, "denoise_step", 2)


@pytest.mark.parametrize("length", [None, 0, 2])
def test_axis_preserves_unknown_empty_and_positive_lengths(length) -> None:
    axis = TrajectoryAxis("iteration", "denoise_step", length)
    assert axis.length == length
