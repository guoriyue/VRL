"""Constructor-boundary tests for the trajectory validation index."""

from __future__ import annotations

import pytest

from vrl.trajectory.validation import TrajectoryValidator


def test_validator_builds_its_tensor_index_internally() -> None:
    validator = TrajectoryValidator(batch=object())
    assert validator.tensor_refs == {}


def test_validator_rejects_an_injected_tensor_index() -> None:
    with pytest.raises(TypeError, match="tensor_refs"):
        TrajectoryValidator(batch=object(), tensor_refs={})
