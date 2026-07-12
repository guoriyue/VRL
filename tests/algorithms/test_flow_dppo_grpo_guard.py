"""Loss-correctness tests for Flow-DPPO and GRPO-Guard.

Both are FlowGRPO-family trust-region variants that read the rollout proposal
mean (old_prev_sample_mean). These pin the §2.2 / §2.3 acceptance from
SPRINT_diffusion_algorithm_parity using synthetic SDE signals (no model).
"""

from __future__ import annotations

import pytest
import torch

from vrl.algorithms.grpo.continuous import (
    GRPO,
    FlowDPPO,
    FlowDPPOConfig,
    GRPOGuard,
    GRPOGuardConfig,
)
from vrl.algorithms.logprob_mismatch import PrecisionCorrectionConfig
from vrl.algorithms.trajectory import AlgorithmInput
from vrl.rollouts.evaluators.types import SegmentSignal, TrajectorySignalBatch


def _signals(
    *,
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    prev_sample_mean: torch.Tensor,
    old_prev_sample_mean: torch.Tensor | None,
    std_dev_t: torch.Tensor,
    dt: torch.Tensor,
) -> TrajectorySignalBatch:
    return TrajectorySignalBatch(
        segments={
            "denoise": SegmentSignal(
                name="denoise",
                distribution="flow_matching",
                log_prob=log_prob,
                old_log_prob=old_log_prob,
                mask=torch.ones_like(log_prob),
                prev_sample_mean=prev_sample_mean,
                old_prev_sample_mean=old_prev_sample_mean,
                std_dev_t=std_dev_t,
                dt=dt,
            ),
        },
        group_ids=torch.arange(log_prob.shape[0]),
        primary_segment="denoise",
    )


def _input(signals: TrajectorySignalBatch, advantages: torch.Tensor) -> AlgorithmInput:
    return AlgorithmInput(rewards=None, group_ids=None, advantages=advantages, signals=signals)


def _on_policy_signals(n: int):
    """Signals at the on-policy point (ratio==1, zero current-vs-rollout drift).

    log_prob is a leaf so we can read the gradient the loss puts on it; mean ==
    old mean so Flow-DPPO masks nothing and GRPO-Guard's bias is zero — the loss
    reduces to vanilla policy gradient and the gradient direction is unambiguous.
    """
    old_lp = torch.zeros(n)
    log_prob = old_lp.clone().requires_grad_(True)  # leaf; ratio = exp(0) = 1
    mean = torch.zeros(n, 1, 2, 2)
    sig = _signals(
        log_prob=log_prob,
        old_log_prob=old_lp,
        prev_sample_mean=mean,
        old_prev_sample_mean=mean.clone(),
        std_dev_t=torch.ones(n, 1, 1, 1),
        dt=torch.full((n, 1, 1, 1), 0.1),
    )
    return log_prob, sig


# --------------------------------------- gradient direction (all 3 new algos)
# dance_grpo dispatches to the GRPO algorithm verbatim (the factory builds GRPO
# for kind=dance_grpo; its only delta is the trainer's random timestep
# selection, which never touches per-step gradient direction), so it is covered
# by constructing GRPO here under the dance_grpo id.


@pytest.mark.parametrize(
    "make_algo",
    [
        pytest.param(GRPO, id="grpo"),
        pytest.param(GRPO, id="dance_grpo"),  # == GRPO loss (alias)
        pytest.param(lambda: FlowDPPO(FlowDPPOConfig(kl_mask_threshold=10.0)), id="flow_dppo"),
        pytest.param(GRPOGuard, id="grpo_guard"),
    ],
)
def test_gradient_points_to_raise_high_advantage_and_lower_low(make_algo) -> None:
    # Policy-gradient invariant: minimizing the loss must INCREASE the log-prob of
    # a positive-advantage sample and DECREASE it for a negative-advantage one.
    # Gradient descent moves opposite the gradient, so grad sign must be opposite
    # the advantage sign: grad<0 for adv>0, grad>0 for adv<0.
    log_prob, sig = _on_policy_signals(2)
    adv = torch.tensor([1.0, -1.0])
    loss, _ = make_algo().compute_loss(_input(sig, adv))
    loss.backward()
    assert log_prob.grad is not None
    assert log_prob.grad[0].item() < 0  # adv>0 -> push log_prob up
    assert log_prob.grad[1].item() > 0  # adv<0 -> push log_prob down


def test_gradient_magnitude_scales_with_advantage() -> None:
    # Sanity: a 2x larger advantage gives a 2x larger gradient on the same loss
    # surface (linear in advantage at the on-policy point), confirming the
    # direction signal is the advantage, not an artifact.
    log_prob, sig = _on_policy_signals(2)
    loss, _ = GRPO().compute_loss(_input(sig, torch.tensor([1.0, 2.0])))
    loss.backward()
    assert log_prob.grad[1].item() == pytest.approx(2.0 * log_prob.grad[0].item())


# ============================================================ Flow-DPPO (§2.2)


def test_flow_dppo_infinite_threshold_is_vanilla_policy_gradient() -> None:
    # kl_mask_threshold = +inf -> keep_mask all 1 -> loss == mean(-adv * ratio).
    n = 4
    log_prob = torch.tensor([0.1, -0.2, 0.3, -0.1])
    old_log_prob = torch.zeros(n)
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    sig = _signals(
        log_prob=log_prob,
        old_log_prob=old_log_prob,
        prev_sample_mean=torch.zeros(n, 1, 1, 1),
        old_prev_sample_mean=torch.ones(n, 1, 1, 1),  # nonzero drift, but masked off
        std_dev_t=torch.ones(n, 1, 1, 1),
        dt=torch.full((n, 1, 1, 1), 0.1),
    )
    algo = FlowDPPO(FlowDPPOConfig(kl_mask_threshold=float("inf")))
    loss, _ = algo.compute_loss(_input(sig, adv))
    expected = (-adv * torch.exp(log_prob - old_log_prob)).mean()
    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)


def test_flow_dppo_masks_only_gap_widening_updates() -> None:
    # Construct: high KL on all (huge drift, tiny threshold). Sample 0: adv>0,
    # ratio>1 -> masked (pos_rm). Sample 1: adv<0, ratio<1 -> masked (neg_rm).
    # Sample 2: adv>0, ratio<1 -> KEPT (pulls back). Sample 3: adv<0, ratio>1 -> KEPT.
    log_prob = torch.tensor([0.5, -0.5, -0.5, 0.5])  # ratios >1,<1,<1,>1
    old_log_prob = torch.zeros(4)
    adv = torch.tensor([1.0, -1.0, 1.0, -1.0])
    sig = _signals(
        log_prob=log_prob,
        old_log_prob=old_log_prob,
        prev_sample_mean=torch.zeros(4, 1, 1, 1),
        old_prev_sample_mean=torch.full((4, 1, 1, 1), 10.0),  # huge drift -> high KL
        std_dev_t=torch.ones(4, 1, 1, 1),
        dt=torch.full((4, 1, 1, 1), 0.1),
    )
    algo = FlowDPPO(FlowDPPOConfig(kl_mask_threshold=0.0, add_kl_coefficient=False))
    _loss, metrics = algo.compute_loss(_input(sig, adv))
    # Exactly the two gap-widening samples (0 and 1) are dropped.
    assert metrics.update.clip_fraction == pytest.approx(0.5)


def test_flow_dppo_zero_drift_keeps_everything() -> None:
    # old == current proposal mean -> KL 0 -> never above threshold -> all kept,
    # regardless of how small the threshold is.
    n = 3
    log_prob = torch.tensor([0.4, -0.4, 0.2])
    mean = torch.randn(n, 1, 2, 2)
    # std_dev_t / dt are scalar-per-sample (the SDE noise level), so [n,1,1,1].
    sig = _signals(
        log_prob=log_prob,
        old_log_prob=torch.zeros(n),
        prev_sample_mean=mean,
        old_prev_sample_mean=mean.clone(),
        std_dev_t=torch.ones(n, 1, 1, 1),
        dt=torch.full((n, 1, 1, 1), 0.1),
    )
    algo = FlowDPPO(FlowDPPOConfig(kl_mask_threshold=1e-9))
    _loss, metrics = algo.compute_loss(_input(sig, torch.tensor([1.0, -1.0, 1.0])))
    assert metrics.update.clip_fraction == pytest.approx(0.0)


def test_flow_dppo_unit_variance_branch_has_no_std_dev_t_denominator() -> None:
    # add_kl_coefficient=False -> KL = mean_diff_sq / 2, with NO std_dev_t in the
    # denominator. mean_diff_sq = 2**2 = 4 -> KL = 2.0 >= threshold 1.0, so the one
    # gap-widening sample (ratio>1, adv>0) is masked. If std_dev_t (=3) wrongly
    # entered the denominator, KL would be 4/(2*9) ~= 0.22 < 1.0 and nothing masks.
    sig = _signals(
        log_prob=torch.tensor([0.5]),  # ratio = e^0.5 > 1
        old_log_prob=torch.zeros(1),
        prev_sample_mean=torch.zeros(1, 1, 1, 1),
        old_prev_sample_mean=torch.full((1, 1, 1, 1), 2.0),  # mean_diff_sq = 4
        std_dev_t=torch.full((1, 1, 1, 1), 3.0),  # must NOT affect the false branch
        dt=torch.full((1, 1, 1, 1), 0.1),
    )
    algo = FlowDPPO(FlowDPPOConfig(kl_mask_threshold=1.0, add_kl_coefficient=False))
    _loss, metrics = algo.compute_loss(_input(sig, torch.tensor([1.0])))
    assert metrics.update.clip_fraction == pytest.approx(1.0)  # masked


def test_flow_dppo_requires_old_prev_sample_mean() -> None:
    n = 2
    sig = _signals(
        log_prob=torch.zeros(n),
        old_log_prob=torch.zeros(n),
        prev_sample_mean=torch.zeros(n, 1, 1, 1),
        old_prev_sample_mean=None,  # recipe forgot return_prev_sample_mean
        std_dev_t=torch.ones(n, 1, 1, 1),
        dt=torch.ones(n, 1, 1, 1),
    )
    with pytest.raises(RuntimeError, match="return_prev_sample_mean"):
        FlowDPPO().compute_loss(_input(sig, torch.ones(n)))


def test_flow_dppo_truncates_precision_weight_into_loss() -> None:
    # Under fp8/bf16-split rollout, TIS-truncate must cap the rollout->replay ratio
    # that enters the trust-region loss; without it a single inflated weight (e^3)
    # dominates the gradient. Trust region keeps the sample (inf threshold), so the
    # only thing acting is the precision correction.
    sig = _signals(
        log_prob=torch.tensor([3.0]),  # raw ratio = e^3 ~ 20
        old_log_prob=torch.zeros(1),
        prev_sample_mean=torch.zeros(1, 1, 1, 1),
        old_prev_sample_mean=torch.zeros(1, 1, 1, 1),
        std_dev_t=torch.ones(1, 1, 1, 1),
        dt=torch.full((1, 1, 1, 1), 0.1),
    )
    algo = FlowDPPO(FlowDPPOConfig(kl_mask_threshold=float("inf")))
    algo.precision_correction = PrecisionCorrectionConfig(
        tis_mode="truncate",
        tis_imp_weight_cap=1.5,
    )
    loss, _ = algo.compute_loss(_input(sig, torch.tensor([1.0])))
    assert loss.item() == pytest.approx(-1.5, rel=1e-5)  # capped, not ~-20


def test_flow_dppo_rs_rejects_out_of_band_precision_drift() -> None:
    # RS drops a whole sample whose rollout->replay log-ratio is out of band, even
    # though the trust region (inf threshold) keeps everything. Sample 2's drift
    # (log-ratio 5) is far outside [-1, 1] -> rejected.
    n = 4
    sig = _signals(
        log_prob=torch.tensor([0.0, 0.0, 5.0, 0.0]),
        old_log_prob=torch.zeros(n),
        prev_sample_mean=torch.zeros(n, 1, 1, 1),
        old_prev_sample_mean=torch.zeros(n, 1, 1, 1),
        std_dev_t=torch.ones(n, 1, 1, 1),
        dt=torch.full((n, 1, 1, 1), 0.1),
    )
    algo = FlowDPPO(FlowDPPOConfig(kl_mask_threshold=float("inf")))
    algo.precision_correction = PrecisionCorrectionConfig(
        rs_mode="seq_mean_k1",
        rs_log_ratio_low=-1.0,
        rs_log_ratio_high=1.0,
    )
    _loss, metrics = algo.compute_loss(_input(sig, torch.ones(n)))
    assert metrics.update.rs_seq_masked_fraction == pytest.approx(0.25)


def test_flow_dppo_default_precision_correction_is_noop() -> None:
    # The default (tis/rs off) must leave the loss exactly at the vanilla value —
    # the fix must not perturb non-fp8 (same-dtype) runs.
    n = 3
    log_prob = torch.tensor([0.3, -0.2, 0.1])
    sig = _signals(
        log_prob=log_prob,
        old_log_prob=torch.zeros(n),
        prev_sample_mean=torch.zeros(n, 1, 1, 1),
        old_prev_sample_mean=torch.zeros(n, 1, 1, 1),
        std_dev_t=torch.ones(n, 1, 1, 1),
        dt=torch.full((n, 1, 1, 1), 0.1),
    )
    adv = torch.tensor([1.0, -1.0, 0.5])
    loss, m = FlowDPPO(FlowDPPOConfig(kl_mask_threshold=float("inf"))).compute_loss(
        _input(sig, adv),
    )
    assert loss.item() == pytest.approx((-adv * torch.exp(log_prob)).mean().item(), rel=1e-6)
    assert m.update.tis_clip_fraction == 0.0
    assert m.update.rs_seq_masked_fraction == 0.0


def test_grpo_guard_rs_rejects_out_of_band_precision_drift() -> None:
    # GRPO-Guard keeps every sample by design; RS still drops a whole sample whose
    # raw rollout->replay drift is out of band (the precision guard, orthogonal to
    # the soft ratio-mean-bias correction).
    n = 4
    mean = torch.zeros(n, 1, 2, 2)
    sig = _signals(
        log_prob=torch.tensor([0.0, 0.0, 5.0, 0.0]),
        old_log_prob=torch.zeros(n),
        prev_sample_mean=mean,
        old_prev_sample_mean=mean.clone(),
        std_dev_t=torch.ones(n, 1, 2, 2),
        dt=torch.ones(n, 1, 2, 2),
    )
    algo = GRPOGuard()
    algo.precision_correction = PrecisionCorrectionConfig(
        rs_mode="seq_mean_k1",
        rs_log_ratio_low=-1.0,
        rs_log_ratio_high=1.0,
    )
    _loss, metrics = algo.compute_loss(_input(sig, torch.ones(n)))
    assert metrics.update.rs_seq_masked_fraction == pytest.approx(0.25)


def test_trust_region_losses_fail_fast_without_dt() -> None:
    # dt is a hard contract: missing it must raise, not silently drop the
    # diffusion coefficient (Flow-DPPO) or collapse the step-scale to 1 (Guard).
    n = 2
    for algo in (FlowDPPO(), GRPOGuard()):
        sig = _signals(
            log_prob=torch.zeros(n),
            old_log_prob=torch.zeros(n),
            prev_sample_mean=torch.zeros(n, 1, 1, 1),
            old_prev_sample_mean=torch.zeros(n, 1, 1, 1),
            std_dev_t=torch.ones(n, 1, 1, 1),
            dt=None,
        )
        with pytest.raises(RuntimeError, match="dt"):
            algo.compute_loss(_input(sig, torch.ones(n)))


# ============================================================ GRPO-Guard (§2.3)


def test_grpo_guard_no_drift_unit_scale_approximates_flow_grpo() -> None:
    # old == current mean -> ratio_mean_bias 0; std_dev_t=1, dt=1 -> scale 1 and
    # the 1/sqrt_dt**2 norm is 1, so loss ~ mean(max(-adv*ratio, -adv*clip)).
    n = 4
    log_prob = torch.tensor([0.1, -0.2, 0.05, -0.1])
    mean = torch.randn(n, 1, 2, 2)
    sig = _signals(
        log_prob=log_prob,
        old_log_prob=torch.zeros(n),
        prev_sample_mean=mean,
        old_prev_sample_mean=mean.clone(),
        std_dev_t=torch.ones(n, 1, 2, 2),
        dt=torch.ones(n, 1, 2, 2),
    )
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    algo = GRPOGuard(GRPOGuardConfig(clip_ratio=0.2))
    loss, _ = algo.compute_loss(_input(sig, adv))
    ratio = torch.exp(log_prob)
    clipped = torch.clamp(ratio, 0.8, 1.2)
    expected = torch.maximum(-adv * ratio, -adv * clipped).mean()
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)


def test_grpo_guard_ratio_mean_increases_with_drift() -> None:
    # More current-vs-rollout drift -> larger ratio_mean_bias -> larger reported
    # bias metric (kl_penalty carries ratio_mean_bias.mean()).
    n = 3
    base = dict(
        log_prob=torch.zeros(n),
        old_log_prob=torch.zeros(n),
        std_dev_t=torch.ones(n, 1, 2, 2),
        dt=torch.full((n, 1, 2, 2), 0.5),
    )
    mean = torch.zeros(n, 1, 2, 2)
    small = _signals(prev_sample_mean=mean, old_prev_sample_mean=mean + 0.1, **base)
    large = _signals(prev_sample_mean=mean, old_prev_sample_mean=mean + 1.0, **base)
    algo = GRPOGuard()
    adv = torch.tensor([1.0, -1.0, 0.5])
    _, m_small = algo.compute_loss(_input(small, adv))
    _, m_large = algo.compute_loss(_input(large, adv))
    assert m_large.kl_penalty > m_small.kl_penalty


def test_grpo_guard_requires_old_prev_sample_mean() -> None:
    n = 2
    sig = _signals(
        log_prob=torch.zeros(n),
        old_log_prob=torch.zeros(n),
        prev_sample_mean=torch.zeros(n, 1, 1, 1),
        old_prev_sample_mean=None,
        std_dev_t=torch.ones(n, 1, 1, 1),
        dt=torch.ones(n, 1, 1, 1),
    )
    with pytest.raises(RuntimeError, match="return_prev_sample_mean"):
        GRPOGuard().compute_loss(_input(sig, torch.ones(n)))


# ============================================================ kind dispatch


@pytest.mark.parametrize(
    "kind,expected",
    [("flow_dppo", FlowDPPOConfig), ("grpo_guard", GRPOGuardConfig)],
)
def test_kind_builds_the_right_config(kind: str, expected: type) -> None:
    from omegaconf import OmegaConf

    from vrl.config.builders import build_algorithm_config

    cfg = OmegaConf.create({"algorithm": {"kind": kind}})
    assert isinstance(build_algorithm_config(cfg), expected)


def test_trust_region_algorithms_request_kl_intermediates() -> None:
    assert FlowDPPO.needs_kl_intermediates is True
    assert GRPOGuard.needs_kl_intermediates is True
