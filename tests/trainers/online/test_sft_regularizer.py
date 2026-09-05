"""GRPO diffusion-loss regularizer (algorithm.sft_weight) trainer term."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from diffusers import FlowMatchEulerDiscreteScheduler

from tests.trainers.online._collector_control import CollectorControlFake
from tests.trainers.online._helpers import (
    _diffusion_rollout_batch,
    _stamp_model_precision,
)
from vrl.algorithms.grpo.continuous import GRPO, GRPOConfig
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.base import Evaluator
from vrl.trainers.core.types import EMAConfig, OptimConfig
from vrl.trainers.online import OnlineTrainer
from vrl.trainers.online.config import OnlineBatchPlan, TrainerConfig
from vrl.trainers.online.trainer import _ReplayMetrics

_B, _T = 2, 4
_LATENT = (3, 2, 2)
_TARGET_VIDEO = "targets/training.mp4"
_TARGET_IMAGE = "targets/training.png"


class _Collector(CollectorControlFake):
    async def score_rollouts(self, pendings):
        return list(pendings)


class _Evaluator(Evaluator):
    def __init__(self) -> None:
        self.scheduler = FlowMatchEulerDiscreteScheduler()
        self.scheduler.set_timesteps(_T)

    def evaluate(self, model, batch, timestep_idx, **kw):  # pragma: no cover
        raise AssertionError("unit test never replays signals")


class _Policy(nn.Module):
    """Stub family policy: the 'prediction' is weight * noisy latents."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.forward_calls: list[tuple[int, torch.Tensor]] = []

    def replay_forward_with_latents(self, batch, timestep_idx, latents):
        self.forward_calls.append((timestep_idx, latents))
        return {
            "noise_pred": -100.0 * self.weight * latents,
            "noise_pred_cond": self.weight * latents,
        }

    def diffusion_pretraining_prediction(self, values):
        return values["noise_pred_cond"]


def _batch(scheduler) -> RolloutBatch:
    observations = torch.randn(_B, _T, *_LATENT)
    timesteps = scheduler.timesteps[:_T].unsqueeze(0).expand(_B, _T).clone()
    return _diffusion_rollout_batch(
        rewards=torch.zeros(_B),
        group_ids=torch.arange(_B),
        observations=observations,
        actions=torch.randn(_B, _T, *_LATENT),
        timesteps=timesteps,
        context={
            "reward_metadata": {"target_video": _TARGET_VIDEO},
        },
    )


def _trainer(tmp_path, *, sft_weight: float, sft_latents) -> OnlineTrainer:
    model = _Policy()
    _stamp_model_precision(model)
    return OnlineTrainer(
        algorithm=GRPO(GRPOConfig(sft_weight=sft_weight)),
        collector=_Collector(),
        evaluator=_Evaluator(),
        model=model,
        config=TrainerConfig(
            batch_plan=OnlineBatchPlan(prompts_per_batch=1, n_samples_per_prompt=1),
            timestep_fraction=1.0,
            drop_zero_advantage=False,
            optim=OptimConfig(lr=0.01),
            ema=EMAConfig(),
            train_precision="no",
            output_dir=str(tmp_path),
        ),
        device="cpu",
        sft_latents=sft_latents,
    )


def _latents_for_targets() -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    return {_TARGET_VIDEO: torch.randn(*_LATENT)}


def test_sft_term_flows_gradient_and_scales_with_weight(tmp_path) -> None:
    latents = _latents_for_targets()
    trainer1 = _trainer(tmp_path, sft_weight=0.5, sft_latents=latents)
    trainer2 = _trainer(tmp_path, sft_weight=1.0, sft_latents=latents)
    batch = _batch(trainer1.evaluator.scheduler)

    torch.manual_seed(0)
    loss1 = trainer1._sft_regularizer_loss(batch)
    torch.manual_seed(0)
    loss2 = trainer2._sft_regularizer_loss(batch)

    assert loss1.requires_grad
    assert float(loss1.detach()) > 0
    # Same seed -> same (step, noise) draw; the term is linear in the weight.
    torch.testing.assert_close(loss2, 2 * loss1)

    loss1.backward()
    assert trainer1.model.weight.grad is not None
    assert float(trainer1.model.weight.grad.abs()) > 0

    # The forward ran on the noised CLEAN latents, at a schedule step index.
    step_idx, noisy = trainer1.model.forward_calls[0]
    assert 0 <= step_idx < _T
    assert noisy.shape == (_B, *_LATENT)


def test_sft_term_rejects_missing_target(tmp_path) -> None:
    latents = {"targets/other.mp4": torch.randn(*_LATENT)}
    trainer = _trainer(tmp_path, sft_weight=0.5, sft_latents=latents)
    with pytest.raises(ValueError, match="no entry for clean target"):
        trainer._sft_regularizer_loss(_batch(trainer.evaluator.scheduler))


def test_sft_term_accepts_image_target_identity(tmp_path) -> None:
    trainer = _trainer(
        tmp_path,
        sft_weight=0.5,
        sft_latents={_TARGET_IMAGE: torch.randn(*_LATENT)},
    )
    batch = _batch(trainer.evaluator.scheduler)
    batch.context["reward_metadata"] = {"target_image": _TARGET_IMAGE}

    assert torch.isfinite(trainer._sft_regularizer_loss(batch))


def test_sft_term_rejects_geometry_mismatch(tmp_path) -> None:
    latents = {_TARGET_VIDEO: torch.randn(3, 4, 4)}
    trainer = _trainer(tmp_path, sft_weight=0.5, sft_latents=latents)
    with pytest.raises(ValueError, match="does not match the"):
        trainer._sft_regularizer_loss(_batch(trainer.evaluator.scheduler))


def test_ctor_rejects_weight_without_latents(tmp_path) -> None:
    with pytest.raises(ValueError, match="sft_weight > 0 but no sft_latents"):
        _trainer(tmp_path, sft_weight=0.5, sft_latents=None)


def test_ctor_allows_zero_weight_without_latents(tmp_path) -> None:
    trainer = _trainer(tmp_path, sft_weight=0.0, sft_latents=None)
    assert trainer._sft_weight == 0.0


def test_sft_backward_uses_group_scale_not_timestep_scale(tmp_path) -> None:
    trainer = _trainer(
        tmp_path,
        sft_weight=0.5,
        sft_latents=_latents_for_targets(),
    )
    batch = _batch(trainer.evaluator.scheduler)
    backward_losses: list[torch.Tensor] = []
    trainer._backward = backward_losses.append
    agg = _ReplayMetrics()

    trainer._backward_sft_regularizer(
        batch,
        loss_weight=0.25,
        total_groups=2,
        is_dummy=False,
        agg=agg,
    )

    metric = agg.sft_loss
    assert len(backward_losses) == 1
    torch.testing.assert_close(
        backward_losses[0].detach(),
        torch.tensor(metric * 0.25 / 2),
    )


def test_sft_dummy_slot_runs_zero_weight_backward_without_metric(tmp_path) -> None:
    trainer = _trainer(
        tmp_path,
        sft_weight=0.5,
        sft_latents=_latents_for_targets(),
    )
    backward_losses: list[torch.Tensor] = []
    trainer._backward = backward_losses.append
    agg = _ReplayMetrics()

    trainer._backward_sft_regularizer(
        _batch(trainer.evaluator.scheduler),
        loss_weight=0.0,
        total_groups=2,
        is_dummy=True,
        agg=agg,
    )

    assert len(trainer.model.forward_calls) == 1
    assert len(backward_losses) == 1
    assert float(backward_losses[0].detach()) == 0.0
    assert agg.sft_loss == 0.0
