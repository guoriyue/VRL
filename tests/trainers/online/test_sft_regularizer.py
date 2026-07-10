"""GRPO diffusion-loss regularizer (algorithm.sft_weight) trainer term."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from diffusers import FlowMatchEulerDiscreteScheduler

from vrl.algorithms.grpo.continuous import GRPO, GRPOConfig
from vrl.generation.types import GenerationRequest, GenerationSampleRow
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.base import Evaluator
from vrl.trainers.core.types import EMAConfig, OptimConfig, TrainerConfig
from vrl.trainers.online import OnlineTrainer
from vrl.trajectory import build_diffusion_trajectory

_B, _T = 2, 4
_LATENT = (3, 2, 2)
_PROMPTS = ["a red fox", "a blue car"]


class _Collector:
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
        return {"noise_pred": self.weight * latents}


def _sample_rows() -> list[GenerationSampleRow]:
    return [
        GenerationSampleRow(
            prompt_index=i,
            sample_index=0,
            prompt=prompt,
            prompt_id=f"r:prompt:{i}",
            group_id=f"r:group:{i}",
            sample_id=f"r:sample:{i}:0",
            trajectory_id=f"r:trajectory:{i}:0",
            seed=None,
            metadata={},
        )
        for i, prompt in enumerate(_PROMPTS)
    ]


def _batch(scheduler) -> RolloutBatch:
    observations = torch.randn(_B, _T, *_LATENT)
    timesteps = scheduler.timesteps[: _T].unsqueeze(0).expand(_B, _T).clone()
    request = GenerationRequest(
        request_id="r",
        family="cosmos-predict2",
        task="v2w",
        prompts=list(_PROMPTS),
        samples_per_prompt=1,
        return_artifacts={"output", "trajectory"},
    )
    trajectory = build_diffusion_trajectory(
        request=request,
        sample_rows=_sample_rows(),
        observations=observations,
        actions=torch.randn(_B, _T, *_LATENT),
        old_log_prob=torch.zeros(_B, _T),
        timesteps=timesteps,
        kl=torch.zeros(_B, _T),
        replay_tensors={},
        context={},
    )
    return RolloutBatch(
        observations=observations,
        actions=torch.randn(_B, _T, *_LATENT),
        rewards=torch.zeros(_B),
        dones=torch.ones(_B, dtype=torch.bool),
        group_ids=torch.arange(_B),
        prompts=list(_PROMPTS),
        trajectory=trajectory,
    )


def _trainer(tmp_path, *, sft_weight: float, sft_latents) -> OnlineTrainer:
    return OnlineTrainer(
        algorithm=GRPO(GRPOConfig(sft_weight=sft_weight)),
        collector=_Collector(),
        evaluator=_Evaluator(),
        model=_Policy(),
        config=TrainerConfig(
            prompts_per_batch=1,
            timestep_fraction=1.0,
            total_epochs=1,
            drop_zero_advantage=False,
            optim=OptimConfig(lr=0.01),
            ema=EMAConfig(),
            n_samples_per_prompt=1,
            train_precision="no",
            output_dir=str(tmp_path),
        ),
        device="cpu",
        sft_latents=sft_latents,
    )


def _latents_for_prompts() -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    return {prompt: torch.randn(*_LATENT) for prompt in _PROMPTS}


def test_sft_term_flows_gradient_and_scales_with_weight(tmp_path) -> None:
    latents = _latents_for_prompts()
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


def test_sft_term_rejects_missing_prompt(tmp_path) -> None:
    latents = _latents_for_prompts()
    del latents[_PROMPTS[1]]
    trainer = _trainer(tmp_path, sft_weight=0.5, sft_latents=latents)
    with pytest.raises(ValueError, match="missing 1 of this batch's prompts"):
        trainer._sft_regularizer_loss(_batch(trainer.evaluator.scheduler))


def test_sft_term_rejects_geometry_mismatch(tmp_path) -> None:
    latents = {prompt: torch.randn(3, 4, 4) for prompt in _PROMPTS}
    trainer = _trainer(tmp_path, sft_weight=0.5, sft_latents=latents)
    with pytest.raises(ValueError, match="does not match the"):
        trainer._sft_regularizer_loss(_batch(trainer.evaluator.scheduler))


def test_ctor_rejects_weight_without_latents(tmp_path) -> None:
    with pytest.raises(ValueError, match="sft_weight > 0 but no sft_latents"):
        _trainer(tmp_path, sft_weight=0.5, sft_latents=None)


def test_ctor_allows_zero_weight_without_latents(tmp_path) -> None:
    trainer = _trainer(tmp_path, sft_weight=0.0, sft_latents=None)
    assert trainer._sft_weight == 0.0
