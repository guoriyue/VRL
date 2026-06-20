"""Unit tests for rollout-vs-replay logprob mismatch stats."""

from __future__ import annotations

import math

import pytest
import torch

from vrl.algorithms.logprob_mismatch import (
    LogprobMismatchStats,
    PrecisionCorrectionConfig,
    apply_rejection_sample_mask,
    combine_keep_masks,
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


# ---------------------------------------------------------------------------
# Reject-sampling (RS) config validation
# ---------------------------------------------------------------------------

LN_HALF = math.log(0.5)
LN_TWO = math.log(2.0)


class TestRejectSampleConfig:
    """RS knobs sit on the same PrecisionCorrectionConfig as TIS (orthogonal)."""

    def test_defaults_are_off_and_symmetric_band(self) -> None:
        pc = PrecisionCorrectionConfig()
        assert pc.rs_mode == "off"
        assert pc.recompute_old_logprob == "off"
        assert pc.rs_log_ratio_low == pytest.approx(LN_HALF)
        assert pc.rs_log_ratio_high == pytest.approx(LN_TWO)

    def test_token_rs_mode_rejected_with_pointer_to_seq(self) -> None:
        with pytest.raises(ValueError, match=r"seq_mean_k1.*seq_max_k1|seq_max_k1"):
            PrecisionCorrectionConfig(rs_mode="token_k1")

    def test_unknown_rs_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="rs_mode"):
            PrecisionCorrectionConfig(rs_mode="seq_median_k1")

    def test_inverted_band_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"rs_log_ratio_low.*<.*rs_log_ratio_high"):
            PrecisionCorrectionConfig(rs_mode="seq_mean_k1", rs_log_ratio_low=1.0, rs_log_ratio_high=0.0)

    def test_recompute_on_is_not_implemented_not_silent_noop(self) -> None:
        with pytest.raises(NotImplementedError, match="recompute_old_logprob"):
            PrecisionCorrectionConfig(recompute_old_logprob="on")

    def test_invalid_recompute_value_rejected(self) -> None:
        with pytest.raises(ValueError, match="recompute_old_logprob"):
            PrecisionCorrectionConfig(recompute_old_logprob="yes")

    def test_rs_and_tis_coexist(self) -> None:
        pc = PrecisionCorrectionConfig(tis_mode="mask", rs_mode="seq_mean_k1")
        assert pc.tis_mode == "mask"
        assert pc.rs_mode == "seq_mean_k1"


# ---------------------------------------------------------------------------
# Reject-sampling (RS) mask
# ---------------------------------------------------------------------------


class TestRejectSampleMask:
    """apply_rejection_sample_mask judges the LOG-RATIO (not the IS weight)."""

    @staticmethod
    def _cfg(mode: str) -> PrecisionCorrectionConfig:
        return PrecisionCorrectionConfig(
            rs_mode=mode, rs_log_ratio_low=LN_HALF, rs_log_ratio_high=LN_TWO,
        )

    def test_off_returns_none(self) -> None:
        out = apply_rejection_sample_mask(torch.tensor([0.0, 5.0]), self._cfg("off"))
        assert out is None

    def test_per_sample_in_band_kept_out_of_band_rejected(self) -> None:
        # log-ratio 0 (ratio 1) in band; ln(3) (ratio 3) above high=ln2 → rejected.
        log_ratio = torch.tensor([0.0, math.log(3.0)])
        keep = apply_rejection_sample_mask(log_ratio, self._cfg("seq_mean_k1"))
        assert keep.tolist() == [1.0, 0.0]

    def test_per_sample_modes_coincide(self) -> None:
        # No sequence axis: seq_mean and seq_max give the identical per-sample mask.
        log_ratio = torch.tensor([0.0, math.log(3.0), LN_HALF])
        mean = apply_rejection_sample_mask(log_ratio, self._cfg("seq_mean_k1"))
        mx = apply_rejection_sample_mask(log_ratio, self._cfg("seq_max_k1"))
        assert torch.equal(mean, mx)

    def test_seq_mean_uses_mean_over_sequence(self) -> None:
        # seq0 mean = (0 + 0)/2 = 0 (kept); seq1 mean = (ln3 + ln3)/2 = ln3 (rejected).
        log_ratio = torch.tensor([[0.0, 0.0], [math.log(3.0), math.log(3.0)]])
        keep = apply_rejection_sample_mask(log_ratio, self._cfg("seq_mean_k1"))
        assert keep.shape == (2, 1)
        assert keep.squeeze(-1).tolist() == [1.0, 0.0]

    def test_seq_mean_tolerates_offsetting_steps(self) -> None:
        # One step above band, one below, but the MEAN is in band → kept.
        log_ratio = torch.tensor([[math.log(3.0), math.log(1 / 3)]])  # mean 0
        keep = apply_rejection_sample_mask(log_ratio, self._cfg("seq_mean_k1"))
        assert keep.squeeze(-1).item() == 1.0

    def test_seq_max_single_outlier_rejects_whole_sequence(self) -> None:
        # Same offsetting steps as above: seq_max rejects because ONE step is out.
        log_ratio = torch.tensor([[math.log(3.0), math.log(1 / 3)]])
        keep = apply_rejection_sample_mask(log_ratio, self._cfg("seq_max_k1"))
        assert keep.squeeze(-1).item() == 0.0

    def test_seq_max_all_in_band_kept(self) -> None:
        log_ratio = torch.tensor([[0.1, -0.2, 0.3]])
        keep = apply_rejection_sample_mask(log_ratio, self._cfg("seq_max_k1"))
        assert keep.squeeze(-1).item() == 1.0

    def test_mask_excludes_padding_from_reduction(self) -> None:
        # seq mean over valid tokens only: the padded ln(50) step must not count
        # (unmasked mean = ln50/3 ~= 1.30 > ln2 → would reject).
        log_ratio = torch.tensor([[0.0, 0.0, math.log(50.0)]])
        valid = torch.tensor([[1.0, 1.0, 0.0]])
        keep_masked = apply_rejection_sample_mask(log_ratio, self._cfg("seq_mean_k1"), mask=valid)
        keep_unmasked = apply_rejection_sample_mask(log_ratio, self._cfg("seq_mean_k1"))
        assert keep_masked.squeeze(-1).item() == 1.0   # padding ignored → mean 0
        assert keep_unmasked.squeeze(-1).item() == 0.0  # padding counted → rejected

    def test_seq_max_padding_does_not_trigger_rejection(self) -> None:
        log_ratio = torch.tensor([[0.1, math.log(9.0)]])  # 2nd step huge but padded
        valid = torch.tensor([[1.0, 0.0]])
        keep = apply_rejection_sample_mask(log_ratio, self._cfg("seq_max_k1"), mask=valid)
        assert keep.squeeze(-1).item() == 1.0


class TestCombineKeepMasks:
    """combine_keep_masks intersects optional keep masks by multiplication."""

    def test_all_none_returns_none(self) -> None:
        assert combine_keep_masks(None, None) is None

    def test_single_passthrough(self) -> None:
        m = torch.tensor([1.0, 0.0])
        assert torch.equal(combine_keep_masks(None, m, None), m)

    def test_intersection_broadcasts(self) -> None:
        token = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])  # (B, L)
        rs = torch.tensor([[1.0], [0.0]])                         # (B, 1)
        out = combine_keep_masks(token, rs)
        assert out.tolist() == [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
