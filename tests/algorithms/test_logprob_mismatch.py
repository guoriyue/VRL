"""Unit tests for rollout-vs-replay logprob mismatch stats."""

from __future__ import annotations

import math

import pytest
import torch

from vrl.algorithms.logprob_mismatch import (
    LogprobMismatchStats,
    compute_logprob_mismatch_stats,
)


def test_metrics_are_zero_when_fresh_equals_old() -> None:
    stats = compute_logprob_mismatch_stats(torch.zeros(4), torch.zeros(4))
    assert stats == LogprobMismatchStats(finite=True)


def test_known_constant_drift_matches_closed_form() -> None:
    delta = 0.1
    fresh = torch.full((3,), delta)
    old = torch.zeros(3)

    stats = compute_logprob_mismatch_stats(fresh, old)

    ratio = math.exp(delta)
    assert stats.logprob_abs_diff_mean == pytest.approx(delta, abs=1e-6)
    assert stats.logprob_abs_diff_max == pytest.approx(delta, abs=1e-6)
    assert stats.ratio_abs_dev_mean == pytest.approx(ratio - 1.0, abs=1e-6)
    assert stats.ratio_abs_dev_max == pytest.approx(ratio - 1.0, abs=1e-6)
    assert stats.mismatch_kl == pytest.approx(-delta, abs=1e-6)  # mean(old - fresh)
    assert stats.mismatch_k3_kl == pytest.approx(ratio - delta - 1.0, abs=1e-6)
    assert stats.finite is True


def test_max_differs_from_mean_on_uneven_drift() -> None:
    fresh = torch.tensor([0.0, 0.2])
    old = torch.zeros(2)

    stats = compute_logprob_mismatch_stats(fresh, old)

    assert stats.logprob_abs_diff_max == pytest.approx(0.2, abs=1e-6)
    assert stats.logprob_abs_diff_mean == pytest.approx(0.1, abs=1e-6)


def test_reductions_run_in_fp32_for_bf16_inputs() -> None:
    # bf16 inputs must not crash and the stats themselves are fp32 floats.
    fresh = torch.full((4,), 0.05, dtype=torch.bfloat16)
    old = torch.zeros(4, dtype=torch.bfloat16)

    stats = compute_logprob_mismatch_stats(fresh, old)

    assert stats.finite is True
    assert stats.logprob_abs_diff_mean > 0.0


def test_empty_input_returns_defaults() -> None:
    stats = compute_logprob_mismatch_stats(torch.empty(0), torch.empty(0))
    assert stats == LogprobMismatchStats()


def test_nonfinite_drift_is_flagged() -> None:
    fresh = torch.tensor([0.0, float("inf")])
    old = torch.zeros(2)

    stats = compute_logprob_mismatch_stats(fresh, old)

    assert stats.finite is False
