"""Flash-GRPO: temporal gradient rectification + the sde_window training mode.

Flash-GRPO (arXiv:2605.15980) is three mechanisms; each has a test anchor here:

1. Temporal gradient rectification — the loss weight is the reciprocal of the
   log-prob gradient scale c(t) = sqrt(-dt)/std + std*sqrt(-dt)*(1-sigma)/(2*sigma),
   batch-normalized to mean 1. The reference implementation hardcodes the
   resulting per-timestep table for the Wan 20-step shift-3 schedule
   (train_wan2_1_flash_1node.py's value_dict); the parity test below reproduces
   that table THROUGH vrl's own sde_step_with_logprob, so the two codebases are
   pinned to the same math, not the same source.
2. Single stochastic step — rollout.sde.window_size=1 (existing window
   machinery) + actor.timestep_selection="sde_window", which trains exactly the
   recorded stochastic step. Validation tests live here; the recording test
   lives with the trajectory fixture.
3. Iso-temporal grouping — the window is drawn once per generation request
   (seeded), covered in the layout tests.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from vrl.algorithms.grpo.continuous import FlashGRPO, FlashGRPOConfig
from vrl.algorithms.trajectory import AlgorithmInput
from vrl.math.denoise.flow_matching import sde_step_with_logprob
from vrl.rollouts.evaluators.types import SegmentSignal, TrajectorySignalBatch

# The reference implementation's hardcoded rectification table: coe(t) for the
# first 10 steps of the Wan2.1 20-step shift-3 flow-match schedule, where
# coe = 1/(sqrt(-dt)/std + std*sqrt(-dt)*(1-sigma)/(2*sigma)) is the value its
# wan pipeline returns (wan2_1_pipeline_with_logprob2.py:81) and its trainer
# multiplies into the loss (train_wan2_1_flash_1node.py's value_dict).
_REFERENCE_COE = [7.4770, 7.0414, 6.6112, 6.1867, 5.7682, 5.3559, 4.9502, 4.5513, 4.1596, 3.7754]


def _wan20_sigmas() -> torch.Tensor:
    """Wan2.1 T2V 20-step shift-3 rectified-flow sigma table (+ final zero)."""

    def shifted(u: float, shift: float = 3.0) -> float:
        return shift * u / (1 + (shift - 1) * u)

    sigmas = [1.0] + [shifted(1 - i / 20) for i in range(1, 20)] + [0.0]
    return torch.tensor(sigmas, dtype=torch.float64)


def _sde_intermediates(steps: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(std_dev_t, sqrt_neg_dt, sigma) per step, via the REAL SDE step math."""

    scheduler = SimpleNamespace(sigmas=_wan20_sigmas())
    stds, dts, sigmas = [], [], []
    for step in steps:
        result = sde_step_with_logprob(
            scheduler,
            torch.zeros(1, 4),
            torch.tensor([float(scheduler.sigmas[step] * 1000)]),
            torch.zeros(1, 4),
            generator=torch.Generator().manual_seed(0),
            return_dt=True,
            sde_type="flow_grpo",
            step_index=step,
        )
        stds.append(result.std_dev_t.reshape(-1))
        dts.append(result.sqrt_neg_dt.reshape(-1))
        sigmas.append(result.sigma.reshape(-1))
    return torch.cat(stds), torch.cat(dts), torch.cat(sigmas)


def _signals(
    n: int,
    *,
    std_dev_t: torch.Tensor | None = None,
    dt: torch.Tensor | None = None,
    sigma: torch.Tensor | None = None,
) -> TrajectorySignalBatch:
    segment = SegmentSignal(
        name="denoise",
        distribution="flow_matching",
        log_prob=torch.zeros(n),
        old_log_prob=torch.zeros(n),
        mask=torch.ones(n),
        std_dev_t=std_dev_t,
        dt=dt,
        sigma=sigma,
    )
    return TrajectorySignalBatch(
        segments={"denoise": segment},
        group_ids=torch.arange(n),
        primary_segment="denoise",
    )


def _input(signals: TrajectorySignalBatch, advantages: torch.Tensor) -> AlgorithmInput:
    return AlgorithmInput(rewards=None, group_ids=None, advantages=advantages, signals=signals)


# ------------------------------------------------------- rectification parity


def test_rectification_weight_matches_reference_table() -> None:
    """vrl's weight == the reference's hardcoded table, through the real SDE.

    Comparison happens after batch normalization (both sides divided by their
    mean), exactly the quantity that reaches the loss — this also cancels the
    residual approximation in reconstructing the schedule.
    """

    std, dt, sigma = _sde_intermediates(list(range(10)))
    weight = FlashGRPO()._loss_weight(
        _signals(10, std_dev_t=std, dt=dt, sigma=sigma).primary,
    )

    table = torch.tensor(_REFERENCE_COE)
    expected = table / table.mean()
    assert torch.allclose(weight, expected.float(), rtol=2e-3), (
        f"weight {weight.tolist()} != normalized reference {expected.tolist()}"
    )


def test_earlier_noisier_steps_get_larger_weight() -> None:
    # coe decreases monotonically over the window: early high-noise steps have
    # the SMALLEST log-prob gradients, so rectification weights them UP.
    std, dt, sigma = _sde_intermediates(list(range(10)))
    weight = FlashGRPO()._loss_weight(
        _signals(10, std_dev_t=std, dt=dt, sigma=sigma).primary,
    )
    assert torch.all(weight[:-1] > weight[1:])
    assert weight[0] > 1.0 > weight[-1]


def test_weight_is_mean_one_so_effective_lr_is_unchanged() -> None:
    std, dt, sigma = _sde_intermediates([0, 3, 6, 9])
    weight = FlashGRPO()._loss_weight(
        _signals(4, std_dev_t=std, dt=dt, sigma=sigma).primary,
    )
    assert weight.mean().item() == pytest.approx(1.0, rel=1e-6)


# ------------------------------------------------------------- loss behavior


def test_on_policy_unit_advantages_give_loss_minus_one() -> None:
    # ratio == 1 on-policy, A == 1: loss = -mean(w) = -1 for ANY sigma profile —
    # rectification redistributes gradient across timesteps without changing
    # the loss scale.
    std, dt, sigma = _sde_intermediates([0, 3, 6, 9])
    loss, metrics = FlashGRPO().compute_loss(
        _input(_signals(4, std_dev_t=std, dt=dt, sigma=sigma), torch.ones(4)),
    )
    assert float(loss) == pytest.approx(-1.0, rel=1e-6)
    assert metrics.policy_loss == pytest.approx(-1.0, rel=1e-6)


def test_one_hot_advantage_exposes_the_per_step_weight() -> None:
    # Loss with a one-hot advantage isolates w_i/n: the early step's magnitude
    # must exceed the late step's by the table ratio (~1.98x over the window).
    std, dt, sigma = _sde_intermediates(list(range(10)))
    sig = _signals(10, std_dev_t=std, dt=dt, sigma=sigma)

    def loss_with_advantage_at(index: int) -> float:
        advantages = torch.zeros(10)
        advantages[index] = 1.0
        loss, _ = FlashGRPO().compute_loss(_input(sig, advantages))
        return float(loss)

    ratio = loss_with_advantage_at(0) / loss_with_advantage_at(9)
    assert ratio == pytest.approx(_REFERENCE_COE[0] / _REFERENCE_COE[9], rel=5e-3)


def test_missing_sigma_fails_fast() -> None:
    std, dt, _ = _sde_intermediates([0, 1])
    with pytest.raises(RuntimeError, match="sigma"):
        FlashGRPO().compute_loss(
            _input(_signals(2, std_dev_t=std, dt=dt, sigma=None), torch.ones(2)),
        )


# ------------------------------------------------------------ kind dispatch


def test_flash_grpo_kind_dispatches_to_its_config() -> None:
    from vrl.config.algorithm import algorithm_config_class

    cls = algorithm_config_class("flash_grpo")
    assert cls is FlashGRPOConfig
    # The paper's tight clip is the algorithm's default identity, not a preset.
    assert cls().clip_ratio == pytest.approx(1e-3)
    assert FlashGRPO().needs_kl_intermediates is True


# --------------------------------------------- sde_window selection contract


def _trainer_config(**overrides):
    from vrl.trainers.core.types import DebugConfig, EMAConfig, OptimConfig
    from vrl.trainers.online.config import OnlineBatchPlan, TrainerConfig

    base = dict(
        batch_plan=OnlineBatchPlan(prompts_per_batch=1, n_samples_per_prompt=2),
        timestep_fraction=1.0,
        drop_zero_advantage=False,
        output_dir="outputs/",
        optim=OptimConfig(lr=0.01),
        ema=EMAConfig(),
        debug=DebugConfig(),
    )
    base.update(overrides)
    return TrainerConfig(**base)


def test_sde_window_selection_refuses_partial_timestep_fraction() -> None:
    # fraction would be silently ignored — refuse instead of quietly training
    # a different objective than configured.
    with pytest.raises(ValueError, match="timestep_fraction"):
        _trainer_config(timestep_selection="sde_window", timestep_fraction=0.5)


def _denoise_batch(replay_tensors: dict) -> object:
    from vrl.generation import GenerationRequest, GenerationSampleRow
    from vrl.rollouts.batch import RolloutBatch
    from vrl.trajectory import build_diffusion_trajectory

    request = GenerationRequest(
        request_id="req",
        family="wan_2_1",
        task="t2v",
        inputs=["p0", "p1"],
        samples_per_prompt=1,
    )
    sample_rows = [
        GenerationSampleRow(
            prompt_index=index,
            sample_index=0,
            prompt=f"p{index}",
            sample_id=f"s{index}",
        )
        for index in range(2)
    ]
    trajectory = build_diffusion_trajectory(
        request=request,
        sample_rows=sample_rows,
        observations=torch.ones(2, 4, 3),
        actions=torch.ones(2, 4, 3) * 0.5,
        old_log_prob=torch.zeros(2, 4),
        timesteps=torch.zeros(2, 4),
        kl=torch.zeros(2, 4),
        replay_tensors=replay_tensors,
        context={"guidance_scale": 1.0, "cfg": False, "model_family": "wan_2_1"},
    )
    return RolloutBatch(
        rewards=torch.ones(2),
        group_ids=torch.arange(2),
        trajectory=trajectory,
    )


def test_sde_window_indices_read_the_recorded_window() -> None:
    from vrl.trainers.online import OnlineTrainer

    batch = _denoise_batch({"sde_window": torch.tensor([[2, 3], [2, 3]])})
    assert OnlineTrainer._sde_window_indices(batch) == [2]


def test_sde_window_indices_require_the_recording() -> None:
    from vrl.trainers.online import OnlineTrainer

    with pytest.raises(ValueError, match="sde_window"):
        OnlineTrainer._sde_window_indices(_denoise_batch({}))


def test_sde_window_indices_reject_mixed_windows() -> None:
    from vrl.trainers.online import OnlineTrainer

    batch = _denoise_batch({"sde_window": torch.tensor([[2, 3], [1, 2]])})
    with pytest.raises(ValueError, match="differs across rows"):
        OnlineTrainer._sde_window_indices(batch)


# ------------------------------------------------------- math cross-check


def test_grad_scale_formula_is_the_logprob_gradient_magnitude() -> None:
    """c(t) really is |d log_prob / d model_output| — checked by autograd.

    This is the derivation the rectification stands on: under the flow SDE the
    log-prob gradient w.r.t. the velocity has magnitude c(t) per unit noise,
    so weighting by 1/c(t) equalizes gradient magnitude across timesteps.
    """

    scheduler = SimpleNamespace(sigmas=_wan20_sigmas())
    step = 5
    model_output = torch.zeros(1, 1, dtype=torch.float64, requires_grad=True)
    sample = torch.randn(1, 1, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    result = sde_step_with_logprob(
        scheduler,
        model_output,
        torch.tensor([float(scheduler.sigmas[step] * 1000)]),
        sample,
        generator=torch.Generator().manual_seed(2),
        return_dt=True,
        sde_type="flow_grpo",
        math_dtype=torch.float64,
        step_index=step,
    )
    result.log_prob.sum().backward()

    std = result.std_dev_t.reshape(())
    sqrt_neg_dt = result.sqrt_neg_dt.reshape(())
    sigma = result.sigma.reshape(())
    c = sqrt_neg_dt / std + std * sqrt_neg_dt * (1 - sigma) / (2 * sigma)
    # The sampled noise for this step: (prev_sample - mean) / (std * sqrt(-dt)).
    noise = (result.prev_sample - result.prev_sample_mean) / (std * sqrt_neg_dt)
    # Magnitude identity: |grad| = |noise| * c. The SIGN is -noise * c because
    # d(mean)/d(model_output) carries dt < 0; rectification corrects magnitude
    # only, so the identity under test is the absolute one.
    expected_magnitude = (noise.reshape(()).detach() * c).abs()
    assert abs(float(model_output.grad.reshape(()))) == pytest.approx(
        float(expected_magnitude),
        rel=1e-9,
    )


def test_reference_table_is_reproduced_from_first_principles() -> None:
    """The hardcoded table is 1/c(t) on the wan schedule — no vrl code involved.

    Guards the table constant itself: if someone edits _REFERENCE_COE, this
    fails against plain math.
    """

    sigmas = _wan20_sigmas().tolist()
    sigma_max, sigma_min = sigmas[1], sigmas[-1]
    for step, reference in enumerate(_REFERENCE_COE):
        sigma, sigma_prev = sigmas[step], sigmas[step + 1]
        sqrt_neg_dt = math.sqrt(sigma - sigma_prev)
        std = sigma_min + (sigma_max - sigma_min) * sigma
        c = sqrt_neg_dt / std + std * sqrt_neg_dt * (1 - sigma) / (2 * sigma)
        assert 1 / c == pytest.approx(reference, rel=2e-3)
