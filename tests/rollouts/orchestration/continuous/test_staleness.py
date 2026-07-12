"""Tests for the continuous rollout staleness policy."""

from __future__ import annotations

import pytest

from vrl.rollouts.orchestration.continuous.staleness import StalenessPolicy


def test_staleness_difference_and_none() -> None:
    """Checks staleness difference and none."""
    policy = StalenessPolicy(max_stale_policy_versions=1)
    assert policy.staleness(3, 5) == 2
    assert policy.staleness(None, 5) is None
    assert policy.staleness(3, None) is None


def test_admit_within_bound() -> None:
    """Checks admit within bound."""
    policy = StalenessPolicy(max_stale_policy_versions=1)
    assert policy.admit(5, 5) is True  # fresh
    assert policy.admit(4, 5) is True  # one behind, within bound
    assert policy.admit(3, 5) is False  # two behind, beyond bound


def test_admit_when_versions_absent() -> None:
    """Checks admit when versions absent."""
    # Mechanism-only boundary: production continuous config requires >= 1.
    policy = StalenessPolicy(max_stale_policy_versions=0)
    assert policy.admit(None, None) is True
    assert policy.admit(None, 7) is True


def test_too_stale_and_future() -> None:
    """Checks too stale and future."""
    # Mechanism-only boundary: production continuous config requires >= 1.
    policy = StalenessPolicy(max_stale_policy_versions=0)
    assert policy.too_stale(4, 5) is True
    assert policy.too_stale(5, 5) is False
    assert policy.is_future(6, 5) is True
    assert policy.is_future(5, 5) is False


def test_negative_bound_rejected() -> None:
    """Checks negative bound rejected."""
    with pytest.raises(ValueError, match="max_stale_policy_versions"):
        StalenessPolicy(max_stale_policy_versions=-1)
