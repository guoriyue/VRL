"""Tests for the behavioral generation capability wire contract."""

from __future__ import annotations

import pytest

from vrl.generation.capabilities import FamilyCapability


def test_family_capability_round_trip_preserves_runtime_decisions() -> None:
    capability = FamilyCapability(
        family="unit",
        task="i2v",
        trajectory_kind="diffusion",
        supports_reference_conditioning=True,
        supports_torch_compile=True,
    )

    assert FamilyCapability.from_value(capability.to_dict()) == capability


def test_family_capability_rejects_unknown_trajectory_kind() -> None:
    with pytest.raises(ValueError, match="unsupported trajectory_kind"):
        FamilyCapability.from_value(
            {
                "family": "unit",
                "task": "t2i",
                "trajectory_kind": "not-a-trajectory",
            },
        )
