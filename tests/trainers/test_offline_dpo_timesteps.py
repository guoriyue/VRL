"""Red-line tests for vrl.trainers.offline._sample_timesteps.

Catches the silent fallback where an empty ``scheduler.timesteps`` would
quietly substitute ``num_train_timesteps`` and shift the RL sampling
distribution without the trainer noticing.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vrl.config.precision import RolePrecision
from vrl.trainers.data.preferences import PreferenceBatch
from vrl.trainers.offline import OfflineDPOTrainer, OfflineDPOTrainerConfig

PRECISION = RolePrecision(
    dtype="fp32",
    float32_precision="ieee",
    outer_autocast=False,
)


def _noop_forward(model, noisy, ts, encoder, extra=None):  # pragma: no cover
    return noisy


def _noop_encode_pix(pix):  # pragma: no cover
    return pix


def _noop_encode_text(captions):  # pragma: no cover
    return None


def _make_trainer(
    scheduler_timesteps,
    *,
    float32_precision: str = "ieee",
) -> OfflineDPOTrainer:
    scheduler = SimpleNamespace(
        timesteps=scheduler_timesteps,
        config=SimpleNamespace(num_train_timesteps=1000),
    )
    cfg = OfflineDPOTrainerConfig(
        prediction_type="epsilon",
    )
    model = torch.nn.Linear(4, 4)
    model.precision = RolePrecision(
        dtype="fp32",
        float32_precision=float32_precision,
        outer_autocast=False,
    )
    return OfflineDPOTrainer(
        model=model,
        ref_model=None,
        forward_fn=_noop_forward,
        noise_scheduler=scheduler,
        encode_pixels=_noop_encode_pix,
        encode_text=_noop_encode_text,
        config=cfg,
        device="cpu",
    )


class TestSampleTimesteps:
    """Groups tests for sample timesteps."""

    def test_uses_scheduler_timesteps_when_set(self) -> None:
        """When the trainer holds an explicit scheduler timestep table, sampled timesteps are
        indices into it (0 <= t < len(table)).
        """
        trainer = _make_trainer(torch.arange(20))
        ts = trainer._sample_timesteps(8)
        assert ts.shape == (8,)
        assert (ts >= 0).all() and (ts < 20).all()

    def test_empty_timesteps_raises(self) -> None:
        """Red-line: do not silently fall back to num_train_timesteps."""
        trainer = _make_trainer(torch.empty(0, dtype=torch.long))
        with pytest.raises(RuntimeError, match="set_timesteps"):
            trainer._sample_timesteps(4)


def test_offline_dpo_state_dict_restores_optimizer_and_global_step() -> None:
    """Offline DPO's ``state_dict`` carries ``global_step`` and the optimizer's Adam moments, and
    a fresh trainer restores both.
    """
    source = _make_trainer(torch.arange(20))
    source.global_step = 7
    loss = source.model(torch.ones(1, 4)).sum()
    loss.backward()
    source._optimizer.step()
    source._optimizer.zero_grad()
    state = source.state_dict()

    restored = _make_trainer(torch.arange(20))
    restored.load_state_dict(state, strict=True)

    assert restored.global_step == 7
    assert _adam_exp_avg_values(restored._optimizer) == pytest.approx(
        _adam_exp_avg_values(source._optimizer),
    )


def test_offline_dpo_accumulation_boundary_ignores_global_step_offset() -> None:
    """The gradient-accumulation boundary counts steps taken in this process, not from the resumed
    ``global_step`` offset: the third call after a resume at 7 is the first boundary.
    """
    trainer = _make_trainer(torch.arange(20))
    trainer.config.gradient_accumulation_steps = 3
    trainer.global_step = 7

    assert trainer._mark_gradient_accumulation_step() is False
    assert trainer._mark_gradient_accumulation_step() is False
    assert trainer._mark_gradient_accumulation_step() is True


@pytest.mark.parametrize("use_adafactor", [False, True])
def test_offline_dpo_consumes_resolved_optimizer_values(use_adafactor) -> None:
    model = torch.nn.Linear(1, 1)
    model.precision = PRECISION
    cfg = OfflineDPOTrainerConfig(
        lr=0.02,
        adam_beta1=0.6,
        adam_beta2=0.7,
        adam_weight_decay=0.04,
        adam_epsilon=1e-5,
        use_adafactor=use_adafactor,
    )
    trainer = OfflineDPOTrainer(
        model=model,
        ref_model=None,
        forward_fn=_noop_forward,
        noise_scheduler=SimpleNamespace(timesteps=torch.arange(20)),
        encode_pixels=_noop_encode_pix,
        encode_text=_noop_encode_text,
        config=cfg,
        device="cpu",
    )
    optimizer = trainer._optimizer

    assert optimizer.defaults["lr"] == pytest.approx(0.02)
    assert optimizer.defaults["weight_decay"] == pytest.approx(0.04)
    if use_adafactor:
        from transformers.optimization import Adafactor

        assert isinstance(optimizer, Adafactor)
        assert optimizer.defaults["scale_parameter"] is False
        assert optimizer.defaults["relative_step"] is False
        assert optimizer.defaults["warmup_init"] is False
    else:
        assert type(optimizer) is torch.optim.AdamW
        assert optimizer.defaults["betas"] == pytest.approx((0.6, 0.7))
        assert optimizer.defaults["eps"] == pytest.approx(1e-5)


@pytest.mark.parametrize("float32_precision", ["ieee", "tf32"])
def test_offline_dpo_applies_float32_policy(monkeypatch, float32_precision: str) -> None:
    import vrl.trainers.offline.dpo as dpo_module

    applied: list[str] = []
    monkeypatch.setattr(dpo_module, "apply_float32_precision", applied.append)

    _make_trainer(torch.arange(20), float32_precision=float32_precision)

    assert applied == [float32_precision]


@pytest.mark.parametrize(
    ("sft_weight", "expected_loss", "expected_sft_calls"),
    [(0.0, 2.0, 0), (0.5, 3.5, 1)],
)
@pytest.mark.parametrize(
    ("max_grad_norm", "expected_clip_calls", "expected_grad_norm"),
    [(0.0, 0, 0.0), (0.25, 1, 2.5)],
)
def test_step_metrics_report_the_optimized_loss(
    monkeypatch,
    sft_weight: float,
    expected_loss: float,
    expected_sft_calls: int,
    max_grad_norm: float,
    expected_clip_calls: int,
    expected_grad_norm: float,
) -> None:
    import vrl.trainers.offline.dpo as dpo_module

    model = torch.nn.Linear(1, 1, bias=False)
    ref_model = torch.nn.Linear(1, 1, bias=False)
    scheduler = SimpleNamespace(
        timesteps=torch.arange(1),
        add_noise=lambda latents, noise, timesteps: latents,
    )
    active_forward_scope = 0

    @contextmanager
    def track_model_autocast(forward_model, device):
        nonlocal active_forward_scope
        assert forward_model is model
        assert torch.device(device).type == "cpu"
        active_forward_scope += 1
        try:
            yield
        finally:
            active_forward_scope -= 1

    def forward_fn(model, noisy, timesteps, encoder):
        del timesteps, encoder
        assert active_forward_scope == 1
        return model(noisy.flatten(1)[:, :1]).reshape(-1, 1, 1, 1)

    def fake_dpo_loss(*, model_pred, **kwargs):
        del kwargs
        assert active_forward_scope == 0
        base = model_pred.sum() * 0.0 + 2.0
        return {
            "loss": base,
            "raw_model_loss": base,
            "raw_ref_loss": base,
            "model_diff": base,
            "ref_diff": base,
            "implicit_acc": base,
        }

    sft_calls = 0

    def fake_sft_loss(model_pred, target):
        nonlocal sft_calls
        del target
        assert active_forward_scope == 0
        sft_calls += 1
        return model_pred.sum() * 0.0 + 3.0

    monkeypatch.setattr(dpo_module, "model_autocast", track_model_autocast)
    monkeypatch.setattr(dpo_module, "diffusion_dpo_loss", fake_dpo_loss)
    monkeypatch.setattr(dpo_module, "diffusion_sft_loss", fake_sft_loss)
    clip_calls: list[float] = []

    def fake_clip_grad_norm(parameters, limit):
        list(parameters)
        clip_calls.append(float(limit))
        return torch.tensor(2.5)

    monkeypatch.setattr(dpo_module.nn.utils, "clip_grad_norm_", fake_clip_grad_norm)
    model.precision = PRECISION
    trainer = OfflineDPOTrainer(
        model=model,
        ref_model=ref_model,
        forward_fn=forward_fn,
        noise_scheduler=scheduler,
        encode_pixels=lambda pixels: pixels[:, :1],
        encode_text=lambda captions: torch.zeros(len(captions), 1),
        config=OfflineDPOTrainerConfig(
            prediction_type="epsilon",
            sft_weight=sft_weight,
            lr=0.0,
            max_grad_norm=max_grad_norm,
        ),
        device="cpu",
    )

    metrics = trainer.step(
        PreferenceBatch(
            pixel_values=torch.zeros(1, 6, 1, 1),
            captions=["prompt"],
        ),
    )

    assert metrics.loss == pytest.approx(expected_loss)
    assert sft_calls == expected_sft_calls
    assert clip_calls == [max_grad_norm] * expected_clip_calls
    assert metrics.grad_norm == pytest.approx(expected_grad_norm)


def _adam_exp_avg_values(optimizer) -> list[float]:
    values: list[float] = []
    for slot in optimizer.state.values():
        exp_avg = slot.get("exp_avg")
        if exp_avg is not None:
            values.extend(float(v) for v in exp_avg.reshape(-1).detach().cpu().tolist())
    return values


@pytest.mark.parametrize("invalid", [None, "captions", "pixels", "text"])
def test_step_preserves_caption_pairing(invalid) -> None:
    model = torch.nn.Linear(1, 1)
    model.precision = PRECISION
    seen = []

    def forward(model, noisy, timesteps, encoder):
        del timesteps
        seen.append((noisy.flatten().tolist(), encoder.flatten().tolist()))
        return model(noisy.flatten(1)).reshape_as(noisy)

    trainer = OfflineDPOTrainer(
        model=model,
        ref_model=torch.nn.Linear(1, 1),
        forward_fn=forward,
        noise_scheduler=SimpleNamespace(
            timesteps=torch.arange(20),
            add_noise=lambda latents, noise, timesteps: latents,
        ),
        encode_pixels=lambda pixels: pixels[:1, :1] if invalid == "pixels" else pixels[:, :1],
        encode_text=lambda captions: torch.tensor(
            [[10.0], [20.0]] if invalid != "text" else [[10.0]] * 4,
        ),
        config=OfflineDPOTrainerConfig(prediction_type="epsilon", lr=0.0),
        device="cpu",
    )
    batch = PreferenceBatch(
        pixel_values=torch.tensor(
            [
                [1.0, 1.0, 1.0, 3.0, 3.0, 3.0],
                [2.0, 2.0, 2.0, 4.0, 4.0, 4.0],
            ]
        ).reshape(2, 6, 1, 1),
        captions=["A"] if invalid == "captions" else ["A", "B"],
    )
    if invalid:
        message = {
            "captions": "one caption per image pair",
            "pixels": "encode_pixels must preserve",
            "text": "encode_text must return one embedding",
        }[invalid]
        with pytest.raises(ValueError, match=message):
            trainer.step(batch)
        assert not seen
        assert trainer.global_step == 0
    else:
        trainer.step(batch)
        assert seen == [([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 10.0, 20.0])] * 2
        assert trainer.global_step == 1
