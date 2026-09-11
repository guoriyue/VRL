"""Tests for the continuous rollout staleness policy."""

from __future__ import annotations

import pytest

from vrl.rollouts.orchestration.continuous.staleness import StalenessPolicy


def test_staleness_difference_and_none() -> None:
    """``staleness`` is ``current - produced`` and ``None`` whenever either version is unknown."""
    policy = StalenessPolicy(max_stale_policy_versions=1)
    assert policy.staleness(3, 5) == 2
    assert policy.staleness(None, 5) is None
    assert policy.staleness(3, None) is None


def test_too_stale_and_future() -> None:
    # Mechanism-only boundary: production continuous config requires >= 1.
    policy = StalenessPolicy(max_stale_policy_versions=0)
    assert policy.too_stale(4, 5) is True
    assert policy.too_stale(5, 5) is False
    assert policy.is_future(6, 5) is True
    assert policy.is_future(5, 5) is False


def test_negative_bound_rejected() -> None:
    with pytest.raises(ValueError, match="max_stale_policy_versions"):
        StalenessPolicy(max_stale_policy_versions=-1)


@pytest.mark.parametrize("value", [-0.5, 1.5, "1", True])
def test_window_is_not_coerced(value) -> None:
    with pytest.raises(ValueError, match="max_stale_policy_versions"):
        StalenessPolicy(max_stale_policy_versions=value)


@pytest.mark.parametrize("value", [-1, 3.5, "3", True])
@pytest.mark.parametrize("other", [None, 3])
def test_versions_are_not_coerced(value, other) -> None:
    policy = StalenessPolicy(max_stale_policy_versions=1)
    with pytest.raises(ValueError, match="item_policy_version"):
        policy.too_stale(value, other)
    with pytest.raises(ValueError, match="current_policy_version"):
        policy.is_future(other, value)


@pytest.mark.parametrize("window", [1.5, "1", True])
def test_continuous_config_does_not_coerce_policy_window(window) -> None:
    from vrl.trainers.core.types import ContinuousRolloutConfig

    with pytest.raises(ValueError, match=r"continuous\.max_stale_policy_versions"):
        ContinuousRolloutConfig(max_stale_policy_versions=window)
