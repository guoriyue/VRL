"""Weight-free unit tests for the JoyAI-Echo flow-matching RL policy.

Echo's real transformer needs an 80GB GPU + the merged checkpoint, so these
tests inject a fake Echo wrapper and exercise the parts that must be correct
regardless of weights: the x0->velocity conversion (rectified-flow convention),
the sigma = t/num_train derivation, latent-shape math, and the
encode/prepare/export/restore plumbing the shared diffusion executor drives.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers import FlowMatchEulerDiscreteScheduler

from tests.models.diffusion.fixtures import stamp_model_precision
from vrl.generation.diffusion.layout import VideoGenerationRequest
from vrl.models.diffusion.echo.model import (
    EchoModel,
    EchoReplayModel,
    EchoSamplingState,
    _sigma_from_timestep,
)


class _FakeEcho(nn.Module):
    """Stand-in for LTX2DiffusionWrapper: returns a preset x0 for any input."""

    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        # A real nn.Module so `_transformer_dtype()` / trainable_modules work.
        self.model = nn.Linear(channels, channels)
        self.x0: torch.Tensor | None = None
        self.last_timestep: torch.Tensor | None = None

    def forward(self, *, noisy_image_or_video, conditional_dict, timestep, noisy_audio=None):
        assert noisy_audio is None, "Echo RL path is video-only"
        assert conditional_dict["audio_context"] is None
        self.last_timestep = timestep
        x0 = self.x0 if self.x0 is not None else torch.zeros_like(noisy_image_or_video)
        return x0.to(noisy_image_or_video.dtype), None


class _FakeTextEncoder:
    def __init__(self, seq: int = 3, dim: int = 16) -> None:
        self.seq, self.dim = seq, dim

    def __call__(self, prompts):
        b = len(prompts)
        return {
            "video_context": torch.randn(b, self.seq, self.dim),
            "audio_context": torch.randn(b, self.seq, self.dim),
            "attention_mask": torch.ones(b, self.seq),
        }


def _build_model() -> tuple[EchoModel, _FakeEcho]:
    echo = _FakeEcho()
    model = EchoModel(
        echo=echo,
        text_encoder=_FakeTextEncoder(),
        video_vae=None,
        scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    stamp_model_precision(model)
    return model, echo


def _request(num_steps: int = 4) -> VideoGenerationRequest:
    return VideoGenerationRequest(
        prompt="a cat",
        negative_prompt="",
        task_type="text_to_video",
        width=256,
        height=256,
        frame_count=25,
        num_steps=num_steps,
        guidance_scale=1.0,
        seed=0,
        fps=24,
        extra={},
    )


def test_sigma_from_timestep_divides_and_clamps() -> None:
    like = torch.zeros(2, 1)
    assert abs(float(_sigma_from_timestep(torch.tensor(500.0), 1000, like)) - 0.5) < 1e-6
    # sigma=0 is undefined for velocity; clamp keeps it strictly positive.
    assert float(_sigma_from_timestep(torch.tensor(0.0), 1000, like)) > 0.0


def test_forward_step_velocity_is_rectified_flow_of_x0() -> None:
    """v = (x_t - x0)/sigma, and x0 must be recoverable as x_t - v*sigma."""

    model, echo = _build_model()
    latents = torch.randn(2, 4, 128, 8, 8)
    x0 = torch.randn_like(latents)
    echo.x0 = x0
    num_train = 1000
    t = torch.tensor(400.0)
    sigma = float(t) / num_train

    state = EchoSamplingState(
        latents=latents,
        timesteps=t.reshape(1),
        scheduler=None,
        video_context=torch.randn(2, 3, 16),
        attention_mask=torch.ones(2, 3),
        num_train_timesteps=num_train,
        guidance_scale=1.0,
    )
    out = model.forward_step(state, 0)
    v = out["noise_pred"]

    assert v.dtype == torch.float32
    assert v.shape == latents.shape
    expected_v = (latents - x0) / sigma
    assert torch.allclose(v, expected_v, atol=1e-4)
    # Round-trip: the SDE step reconstructs x0 from (x_t, v, sigma).
    x0_recovered = latents - v * sigma
    assert torch.allclose(x0_recovered, x0, atol=1e-4)
    # Echo received sigma (not the raw flow timestep), broadcast per batch.
    assert torch.allclose(echo.last_timestep, torch.full((2,), sigma), atol=1e-5)


def test_encode_prompt_keeps_video_context_drops_audio() -> None:
    model, _ = _build_model()
    enc = model.encode_prompt("a dog")
    assert set(enc) == {"video_context", "attention_mask"}
    assert enc["video_context"].shape == (1, 3, 16)


def test_prepare_sampling_builds_noise_latents_and_schedule() -> None:
    model, _ = _build_model()
    encoded = {"video_context": torch.randn(2, 3, 16), "attention_mask": torch.ones(2, 3)}
    state = model.prepare_sampling(_request(num_steps=4), encoded)
    assert state.latents.shape == (2, 4, 128, 8, 8)
    assert state.latents.dtype == torch.float32
    assert len(state.timesteps) == 4
    assert state.num_train_timesteps == 1000


def test_export_restore_roundtrip_feeds_forward_step() -> None:
    """Replay path: restore packs timesteps [1,B] so forward_step(.,0) works."""

    model, echo = _build_model()
    latents = torch.randn(2, 4, 128, 8, 8)
    echo.x0 = torch.zeros_like(latents)
    state = EchoSamplingState(
        latents=latents,
        timesteps=torch.tensor([300.0, 600.0, 700.0, 900.0]),
        scheduler=None,
        video_context=torch.randn(2, 3, 16),
        attention_mask=torch.ones(2, 3),
        num_train_timesteps=1000,
        guidance_scale=1.0,
    )
    replay = model.export_replay_tensors(state)
    ctx = model.export_batch_context(state)
    assert set(replay) == {"video_context", "attention_mask"}
    assert ctx["num_train_timesteps"] == 1000

    # Stored denoise timesteps are [B, num_steps]; restore one step.
    stored_ts = torch.tensor([[300.0, 600.0, 700.0, 900.0]] * 2)
    restored = model.restore_eval_state(
        {**replay, "timesteps": stored_ts},
        ctx,
        latents,
        step_idx=2,
    )
    assert restored.timesteps.shape == (1, 2)  # [1, B]
    out = model.forward_step(restored, 0)
    assert out["noise_pred"].shape == latents.shape


def test_move_frozen_components_offloads_wrappers_device_driven() -> None:
    """Generic frozen-component offload: moves encoder/VAE wrappers, not the
    trainable transformer, to whatever device the lifecycle passes."""

    class _Mod(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.moved_to: object = None

        def to(self, device, *args, **kwargs):  # type: ignore[override]
            self.moved_to = device
            return self

    class _Wrap:
        def __init__(self) -> None:
            self.encoder = _Mod()
            self.device = "orig"

    echo = _FakeEcho()
    te, vae = _Wrap(), _Wrap()
    model = EchoModel(
        echo=echo,
        text_encoder=te,
        video_vae=vae,
        scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    model.move_frozen_components("cpu_sentinel")

    assert te.encoder.moved_to == "cpu_sentinel"
    assert vae.encoder.moved_to == "cpu_sentinel"
    assert te.device == "cpu_sentinel" and vae.device == "cpu_sentinel"
    # The trainable transformer is never touched by the offload path.
    assert echo.model.weight.device == torch.device("cpu")


def test_replay_model_offload_is_noop_without_wrappers() -> None:
    echo = _FakeEcho()
    replay = EchoReplayModel(
        echo=echo,
        scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    replay.move_frozen_components("cpu")  # no text encoder / VAE -> must not raise


def test_replay_model_is_transformer_only_and_runs_velocity_forward() -> None:
    """The trainer replay model owns the velocity transformer, no rollout surface."""

    import pytest

    echo = _FakeEcho()
    replay = EchoReplayModel(
        echo=echo,
        scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    stamp_model_precision(replay)
    assert replay.raw_handle is None
    assert replay.trainable_modules == {"transformer": echo.model}
    # Rollout-only surface raises (ReplayRolloutStubs) — replay never encodes/decodes.
    with pytest.raises(RuntimeError):
        replay.encode_prompt("x")
    with pytest.raises(RuntimeError):
        replay.decode_latents(torch.zeros(1, 1, 128, 8, 8))

    # The velocity forward (what GRPO replay calls) works.
    latents = torch.randn(2, 4, 128, 8, 8)
    echo.x0 = torch.zeros_like(latents)
    state = EchoSamplingState(
        latents=latents,
        timesteps=torch.tensor([500.0]),
        scheduler=None,
        video_context=torch.randn(2, 3, 16),
        attention_mask=torch.ones(2, 3),
        num_train_timesteps=1000,
        guidance_scale=1.0,
    )
    out = replay.forward_step(state, 0)
    assert out["noise_pred"].shape == latents.shape
    # x0=0 -> v = x_t/sigma with sigma=0.5 -> v = 2*x_t.
    assert torch.allclose(out["noise_pred"], latents / 0.5, atol=1e-4)
