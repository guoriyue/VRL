"""Real-inference tests for the FLUX.1 guidance-distilled backbone wrapper.

A real (tiny, cache-free) ``FluxTransformer2DModel`` runs through
``forward_step``; the single-branch / guidance / timestep-normalization behavior
is verified against the model's OWN real outputs, never a re-derived formula.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.models.steps.denoise.fixtures import (
    TINY_FLUX_IN_CHANNELS,
    TINY_FLUX_JOINT_DIM,
    TINY_FLUX_POOLED_DIM,
    build_tiny_flux_transformer,
    record_forward_calls,
    stamp_model_precision,
)
from vrl.models.families.flux.model import FluxModel, FluxReplayModel, FluxSamplingState

_TEXT_LEN = 3
_SEQ = 16  # packed latent token count (h//2 * w//2)


def _model(transformer: torch.nn.Module) -> FluxModel:
    model = FluxModel(
        pipeline=SimpleNamespace(transformer=transformer, device=torch.device("cpu")),
        device=torch.device("cpu"),
    )
    stamp_model_precision(model)
    return model


def _state(model: FluxModel, *, guidance_embeds: bool) -> FluxSamplingState:
    return FluxSamplingState(
        latents=torch.randn(2, _SEQ, TINY_FLUX_IN_CHANNELS),
        timesteps=torch.tensor([500.0]),
        scheduler=None,
        prompt_embeds=torch.randn(2, _TEXT_LEN, TINY_FLUX_JOINT_DIM),
        pooled_prompt_embeds=torch.randn(2, TINY_FLUX_POOLED_DIM),
        text_ids=torch.zeros(_TEXT_LEN, 3),
        latent_image_ids=model._build_latent_image_ids(4, 4, torch.device("cpu"), torch.float32),
        guidance_scale=3.5,
        guidance_embeds=guidance_embeds,
        height=128,
        width=128,
    )


def test_flux_forward_step_runs_single_distilled_branch() -> None:
    """One forward, no uncond branch, noise == cond, timestep fed as t / 1000."""
    transformer = build_tiny_flux_transformer(guidance_embeds=True)
    calls = record_forward_calls(transformer)
    model = _model(transformer)

    out = model.forward_step(_state(model, guidance_embeds=True), 0)

    # Single guidance-distilled branch: batch is NOT doubled, one transformer call.
    assert len(calls) == 1
    assert calls[0]["hidden_states"].shape[0] == 2
    # noise == cond and uncond is a zero placeholder (no CFG).
    torch.testing.assert_close(out["noise_pred"], out["noise_pred_cond"])
    torch.testing.assert_close(
        out["noise_pred_uncond"], torch.zeros_like(out["noise_pred_uncond"])
    )
    # Timestep is normalized by 1000 before the transformer (500 -> 0.5).
    torch.testing.assert_close(
        calls[0]["timestep"],
        torch.full((2,), 0.5),
    )
    # guidance is present and carries the guidance scale.
    assert calls[0]["guidance"] is not None
    torch.testing.assert_close(calls[0]["guidance"], torch.full((2,), 3.5))
    assert out["noise_pred"].shape == (2, _SEQ, TINY_FLUX_IN_CHANNELS)


def test_flux_encode_prompt_accepts_only_empty_negative_conditioning() -> None:
    negative_prompt = ""
    transformer = build_tiny_flux_transformer(guidance_embeds=True)
    model = _model(transformer)
    model.pipeline.encode_prompt = lambda **_kwargs: (
        torch.zeros(1, _TEXT_LEN, TINY_FLUX_JOINT_DIM),
        torch.zeros(1, TINY_FLUX_POOLED_DIM),
        torch.zeros(_TEXT_LEN, 3),
    )

    encoded = model.encode_prompt("a cat", negative_prompt=negative_prompt)

    assert set(encoded) == {"prompt_embeds", "pooled_prompt_embeds", "text_ids"}


def test_flux_encode_prompt_rejects_non_empty_negative_conditioning() -> None:
    negative_prompt = "low quality"
    transformer = build_tiny_flux_transformer(guidance_embeds=True)
    model = _model(transformer)

    with pytest.raises(ValueError, match="does not support negative prompts"):
        model.encode_prompt("a cat", negative_prompt=negative_prompt)


def test_flux_forward_step_omits_guidance_when_not_distilled() -> None:
    """A non-distilled (schnell-like) checkpoint passes guidance=None."""
    transformer = build_tiny_flux_transformer(guidance_embeds=False)
    calls = record_forward_calls(transformer)
    model = _model(transformer)

    model.forward_step(_state(model, guidance_embeds=False), 0)

    assert calls[0]["guidance"] is None


def test_flux_forward_step_casts_replay_tensors_to_transformer_dtype() -> None:
    """bf16 rollout tensors replay cleanly through an fp32 transformer."""
    transformer = build_tiny_flux_transformer(guidance_embeds=True)
    calls = record_forward_calls(transformer)
    model = _model(transformer)
    state = FluxSamplingState(
        latents=torch.randn(2, _SEQ, TINY_FLUX_IN_CHANNELS, dtype=torch.bfloat16),
        timesteps=torch.tensor([[500.0, 600.0]], dtype=torch.bfloat16),
        scheduler=None,
        prompt_embeds=torch.randn(2, _TEXT_LEN, TINY_FLUX_JOINT_DIM, dtype=torch.bfloat16),
        pooled_prompt_embeds=torch.randn(2, TINY_FLUX_POOLED_DIM, dtype=torch.bfloat16),
        text_ids=torch.zeros(_TEXT_LEN, 3),
        latent_image_ids=model._build_latent_image_ids(4, 4, torch.device("cpu"), torch.float32),
        guidance_scale=3.5,
        guidance_embeds=True,
        height=128,
        width=128,
    )

    out = model.forward_step(state, 0)

    assert out["noise_pred"].dtype == torch.float32
    assert calls[0]["hidden_states"].dtype == torch.float32
    assert calls[0]["encoder_hidden_states"].dtype == torch.float32
    assert calls[0]["pooled_projections"].dtype == torch.float32


def test_flux_replay_model_restores_state_without_a_pipeline() -> None:
    """The pipeline-less replay model rebuilds position grids and runs forward_step."""
    transformer = build_tiny_flux_transformer(guidance_embeds=True)
    model = FluxReplayModel(transformer=transformer, scheduler=None, device=torch.device("cpu"))
    stamp_model_precision(model)

    replay_tensors = {
        "prompt_embeds": torch.randn(2, _TEXT_LEN, TINY_FLUX_JOINT_DIM),
        "pooled_prompt_embeds": torch.randn(2, TINY_FLUX_POOLED_DIM),
        # [B, num_steps] per-sample timesteps.
        "timesteps": torch.tensor([[500.0, 600.0], [500.0, 600.0]]),
    }
    batch_context = {
        "guidance_scale": 3.5,
        "guidance_embeds": True,
        "height": 128,
        "width": 128,
        # vae_scale_factor 8 -> latent grid 128/(8*2)*2 = 16, //2 = 8 per axis.
        "vae_scale_factor": 8,
    }
    # height/width 128 with vae_scale_factor 8 -> packed latent grid 8x8 = 64 tokens.
    latents = torch.randn(2, 64, TINY_FLUX_IN_CHANNELS)

    state = model.restore_eval_state(replay_tensors, batch_context, latents, 0)
    # Position grid rebuilt from height/width without touching a pipeline.
    assert state.latent_image_ids.shape == (8 * 8, 3)
    assert state.text_ids.shape == (_TEXT_LEN, 3)

    out = model.forward_step(state, 0)
    assert out["noise_pred"].shape == (2, 64, TINY_FLUX_IN_CHANNELS)
