"""GRPO diffusion-loss regularizer (algorithm.sft_weight) on the real trainer.

The trainer is wired as the online recipe wires it with the regularizer on:
``algorithm.sft_weight`` and ``data.sft_latents`` in the config, the clean-latent
shard read by the recipe's own loader, and the SANA policy's
``replay_forward_with_latents`` as the forward. Batches are real collected
rollouts whose prompts name their clean target.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import torch

from tests.trainers.online._helpers import TrainerBench, real_trainer
from vrl.rollouts.batch import RolloutBatch
from vrl.trainers.data.prompts import PromptExample
from vrl.trainers.data.sft_latents import SFT_LATENTS_SCHEMA_VERSION
from vrl.trainers.online.trainer import OnlineTrainer, _ReplayMetrics

_TARGET_VIDEO = "targets/training.mp4"
_TARGET_IMAGE = "targets/training.png"
# The tiny SANA rollout latent: 4 channels at 8x8 for a 32x32 image.
_LATENT = (4, 8, 8)


def _sft_trainer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    sft_weight: float,
    latents: dict[str, torch.Tensor],
) -> TrainerBench:
    """``real_trainer`` plus the regularizer, as the recipe builds it.

    The clean-latent shard is written next to the run and pinned to its family
    and model path; the recipe's ``_load_sft_latents_from_config`` reads it back.
    """

    from vrl.scripts.common.online import _load_sft_latents_from_config

    tmp_path.mkdir(parents=True, exist_ok=True)
    shard = tmp_path / "sft_latents.pt"
    torch.save(
        {
            "schema_version": SFT_LATENTS_SCHEMA_VERSION,
            "family": "sana",
            "model_path": str(tmp_path / "sana-snapshot"),
            "model_revision": "",
            "latents": latents,
        },
        shard,
    )
    tb = real_trainer(
        monkeypatch,
        tmp_path,
        overrides=(f"algorithm.sft_weight={sft_weight}", f"data.sft_latents={shard}"),
    )
    built = tb.collector.stack.resolved.built
    base = tb.trainer
    trainer = OnlineTrainer(
        algorithm=base.algorithm,
        collector=tb.collector.collector,
        evaluator=base.evaluator,
        model=base.model,
        ref_model=base.ref_model,
        weight_syncer=base.weight_syncer,
        sync_state_getter=base.sync_state_getter,
        config=base.config,
        device=torch.device("cpu"),
        strategy=tb.strategy,
        sft_latents=_load_sft_latents_from_config(built, sft_weight=base.algorithm.sft_weight),
    )
    tb.trainer = trainer
    tb.collector.trace.watch(tb.model, "replay_forward_with_latents", "sft_forward")
    return tb


def _batch(tb: TrainerBench, **target: str) -> RolloutBatch:
    prompt = PromptExample(prompt="a cat", **target)
    return asyncio.run(tb.trainer.collect_training_batch([prompt])).batches[0]


def _latents(seed: int = 7, shape: tuple[int, ...] = _LATENT) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def test_sft_term_flows_gradient_and_scales_with_weight(monkeypatch, tmp_path) -> None:
    latents = {_TARGET_VIDEO: _latents()}
    half = _sft_trainer(monkeypatch, tmp_path / "half", sft_weight=0.5, latents=latents)
    batch = _batch(half, target_video=_TARGET_VIDEO)
    torch.manual_seed(0)
    loss_half = half.trainer._sft_regularizer_loss(batch)

    full = _sft_trainer(monkeypatch, tmp_path / "full", sft_weight=1.0, latents=latents)
    torch.manual_seed(0)
    loss_full = full.trainer._sft_regularizer_loss(batch)

    assert loss_half.requires_grad
    assert float(loss_half.detach()) > 0
    # Same seed: same (step, noise) draw on identical weights, so the term is
    # linear in the weight.
    torch.testing.assert_close(loss_full, 2 * loss_half)

    loss_half.backward()
    grads = [p.grad for p in half.trainable_parameters().values()]
    assert any(grad is not None and float(grad.abs().sum()) > 0 for grad in grads)

    # The forward ran on the noised CLEAN latents, at a schedule step index.
    _batch_arg, step_idx, noisy = next(
        args for event, args in half.collector.trace.calls if event == "sft_forward"
    )
    num_steps = batch.trajectory.segments["denoise"].tensors["timesteps"].value.shape[1]
    assert 0 <= step_idx < num_steps
    assert noisy.shape == (batch.rewards.shape[0], *_LATENT)


def test_sft_term_rejects_missing_target(monkeypatch, tmp_path) -> None:
    tb = _sft_trainer(
        monkeypatch, tmp_path, sft_weight=0.5, latents={"targets/other.mp4": _latents()}
    )
    with pytest.raises(ValueError, match="no entry for clean target"):
        tb.trainer._sft_regularizer_loss(_batch(tb, target_video=_TARGET_VIDEO))


def test_sft_term_accepts_image_target_identity(monkeypatch, tmp_path) -> None:
    tb = _sft_trainer(monkeypatch, tmp_path, sft_weight=0.5, latents={_TARGET_IMAGE: _latents()})

    loss = tb.trainer._sft_regularizer_loss(_batch(tb, target_image=_TARGET_IMAGE))

    assert torch.isfinite(loss)


def test_sft_term_rejects_geometry_mismatch_before_device_copy(monkeypatch, tmp_path) -> None:
    tb = _sft_trainer(
        monkeypatch, tmp_path, sft_weight=0.5, latents={_TARGET_VIDEO: _latents(shape=(4, 4, 4))}
    )
    batch = _batch(tb, target_video=_TARGET_VIDEO)

    def unexpected_copy(*args, **kwargs):
        pytest.fail("invalid SFT geometry must fail before Tensor.to")

    monkeypatch.setattr(torch.Tensor, "to", unexpected_copy)
    with pytest.raises(ValueError, match="does not match the"):
        tb.trainer._sft_regularizer_loss(batch)


def test_sft_backward_uses_group_scale_not_timestep_scale(monkeypatch, tmp_path) -> None:
    tb = _sft_trainer(monkeypatch, tmp_path, sft_weight=0.5, latents={_TARGET_VIDEO: _latents()})
    batch = _batch(tb, target_video=_TARGET_VIDEO)
    tb.collector.trace.watch(tb.trainer, "_backward", "backward")
    agg = _ReplayMetrics()

    tb.trainer._backward_sft_regularizer(
        batch,
        loss_weight=0.25,
        total_groups=2,
        is_dummy=False,
        agg=agg,
    )

    backward_losses = [args[0] for event, args in tb.collector.trace.calls if event == "backward"]
    assert len(backward_losses) == 1
    torch.testing.assert_close(
        backward_losses[0].detach(),
        torch.tensor(agg.sft_loss * 0.25 / 2),
    )


def test_sft_dummy_slot_runs_zero_weight_backward_without_metric(monkeypatch, tmp_path) -> None:
    tb = _sft_trainer(monkeypatch, tmp_path, sft_weight=0.5, latents={_TARGET_VIDEO: _latents()})
    batch = _batch(tb, target_video=_TARGET_VIDEO)
    tb.collector.trace.watch(tb.trainer, "_backward", "backward")
    forwards_before = tb.collector.trace.events.count("sft_forward")
    agg = _ReplayMetrics()

    tb.trainer._backward_sft_regularizer(
        batch,
        loss_weight=0.0,
        total_groups=2,
        is_dummy=True,
        agg=agg,
    )

    backward_losses = [args[0] for event, args in tb.collector.trace.calls if event == "backward"]
    assert tb.collector.trace.events.count("sft_forward") == forwards_before + 1
    assert len(backward_losses) == 1
    assert float(backward_losses[0].detach()) == 0.0
    assert agg.sft_loss == 0.0
