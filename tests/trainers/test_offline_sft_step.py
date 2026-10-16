"""One ``OfflineSFTTrainer`` step: the timestep window, the weights, the update."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vrl.algorithms.sft import DiffusionSFT, DiffusionSFTConfig
from vrl.config.precision import RolePrecision
from vrl.trainers.offline import OfflineSFTTrainer, OfflineTrainerConfig, SFTBatch

PRECISION = RolePrecision(dtype="fp32", float32_precision="ieee", outer_autocast=False)


def _scheduler(steps: int) -> SimpleNamespace:
    """A flow table whose sigma at position i is i / steps, so position and noise agree."""

    sigmas = torch.arange(steps + 1).float() / steps
    return SimpleNamespace(timesteps=sigmas[:-1] * 1000, sigmas=sigmas)


def _trainer(model, *, window=None, forward=None, max_grad_norm=1.0) -> OfflineSFTTrainer:
    model.precision = PRECISION
    return OfflineSFTTrainer(
        model=model,
        algorithm=DiffusionSFT(DiffusionSFTConfig(timestep_index_range=window)),
        forward_fn=forward or (lambda module, noisy, timesteps, cond: module(noisy)),
        noise_scheduler=_scheduler(10),
        config=OfflineTrainerConfig(
            prediction_type="flow_matching", lr=0.1, max_grad_norm=max_grad_norm
        ),
        device="cpu",
    )


def test_timesteps_are_drawn_inside_the_configured_window() -> None:
    seen: list[float] = []

    def forward(module, noisy, timesteps, cond):
        seen.extend(timesteps.tolist())
        return module(noisy)

    trainer = _trainer(torch.nn.Linear(2, 2), window=(3, 5), forward=forward)
    torch.manual_seed(0)
    for _ in range(20):
        trainer.step(SFTBatch(latents=torch.randn(1, 4, 2), conditioning=None))

    assert set(seen) <= {300.0, 400.0}
    assert len(set(seen)) == 2


def test_window_past_the_table_is_rejected() -> None:
    trainer = _trainer(torch.nn.Linear(2, 2), window=(0, 11))
    with pytest.raises(ValueError, match="timestep table"):
        trainer.step(SFTBatch(latents=torch.zeros(1, 4, 2), conditioning=None))


def test_step_optimizes_the_weighted_error_and_reports_the_unweighted_mse(monkeypatch) -> None:
    # A zero-weight token carries an arbitrary error that must not reach the loss.
    class _Fixed(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(1))

        def forward(self, noisy):
            prediction = torch.zeros_like(noisy)
            prediction[:, 1] = 10.0
            return prediction * self.scale

    trainer = _trainer(_Fixed())
    monkeypatch.setattr(torch, "randn_like", lambda latents: torch.zeros_like(latents))
    latents = torch.zeros(1, 2, 3)
    weights = torch.tensor([[[2.0], [0.0]]])

    metrics = trainer.step(SFTBatch(latents=latents, conditioning=None, weights=weights))

    # velocity target is noise - latents = 0 everywhere; only token 0 (error 0) is weighted.
    assert metrics.loss == pytest.approx(0.0)
    assert metrics.mse == pytest.approx(50.0)
    assert trainer.global_step == 1


def test_step_clips_and_reports_the_gradient_norm() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    trainer = _trainer(model, max_grad_norm=0.5)
    torch.manual_seed(1)

    metrics = trainer.step(SFTBatch(latents=torch.randn(1, 4, 2) * 5, conditioning=None))

    assert metrics.grad_norm > 0.5  # the pre-clip norm of a far-off prediction
    assert all(parameter.grad is None for parameter in model.parameters())


def test_non_finite_loss_stops_the_run() -> None:
    trainer = _trainer(torch.nn.Linear(2, 2), forward=lambda m, noisy, t, c: noisy * float("nan"))
    with pytest.raises(FloatingPointError, match="non-finite"):
        trainer.step(SFTBatch(latents=torch.zeros(1, 4, 2), conditioning=None))
