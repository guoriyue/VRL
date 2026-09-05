"""Tests for family-neutral stepwise denoise generation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from vrl.models.steps.denoise.base import DiffusionSamplingStateBase
from vrl.scripts.eval import denoise_generation
from vrl.scripts.eval.denoise_generation import (
    generate_images,
    generate_one_video,
    seed_for,
    video_to_cthw,
)


class _StepwiseModel:
    """Minimal model boundary with real scheduler arithmetic and CPU tensors."""

    def encode_prompt(self, prompt, negative_prompt, **kwargs):
        self.conditioning = (prompt, negative_prompt, kwargs)
        return {"batch_size": len(prompt) if isinstance(prompt, list) else 1}

    def prepare_sampling(self, request, encoded):
        self.request = request
        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(request.num_steps)
        self.state = DiffusionSamplingStateBase(
            latents=torch.full((encoded["batch_size"], 3, 2, 2), 0.5, dtype=torch.bfloat16),
            timesteps=scheduler.timesteps,
            scheduler=scheduler,
        )
        return self.state

    def forward_step(self, state, step_idx):
        assert not torch.is_grad_enabled()
        return {"noise_pred": torch.full_like(state.latents, 0.25, dtype=torch.bfloat16)}

    def decode_latents(self, latents):
        self.decode_grad_enabled = torch.is_grad_enabled()
        self.decoded_latents = latents.clone()
        return latents if isinstance(self.conditioning[0], list) else latents.unsqueeze(2)


@pytest.mark.parametrize("media_kind", ["image", "video"])
def test_native_generation_preserves_conditioning_precision_and_decode_context(media_kind):
    model = _StepwiseModel()
    sampling = SimpleNamespace(
        width=2, height=2, num_steps=2, guidance_scale=4.5, max_sequence_length=128
    )

    if media_kind == "image":
        images = generate_images(
            model,
            prompt="a bird",
            negative_prompt="blur",
            seed=37,
            samples_per_prompt=2,
            sampling=sampling,
            torch=torch,
        )
        assert len(images) == 2
        assert model.conditioning[:2] == (["a bird", "a bird"], ["blur", "blur"])
        assert model.request.negative_prompt == "blur"
        assert model.decode_grad_enabled
    else:
        video = generate_one_video(
            model,
            prompt="a bird",
            seed=37,
            sampling={**vars(sampling), "num_frames": 1, "fps": 16, "denoise_mode": "native"},
        )
        assert video.shape == (3, 1, 2, 2)
        assert model.conditioning[:2] == ("a bird", None)
        assert not model.decode_grad_enabled

    assert model.conditioning[2] == {"max_sequence_length": 128, "guidance_scale": 4.5}
    assert model.request.seed == 37
    assert model.request.frame_count == 1
    assert model.decoded_latents.dtype == torch.float32
    torch.testing.assert_close(model.decoded_latents, torch.full_like(model.decoded_latents, 0.25))


def test_video_sde_keeps_its_seeded_transition_boundary(monkeypatch):
    calls = []

    def transition(scheduler, noise, timestep, latents, **kwargs):
        calls.append((scheduler, noise, timestep, kwargs))
        return SimpleNamespace(prev_sample=latents + 1)

    monkeypatch.setattr(denoise_generation, "sde_step_with_logprob", transition)
    model = _StepwiseModel()
    generate_one_video(
        model,
        prompt="a bird",
        seed=37,
        sampling={
            "width": 2,
            "height": 2,
            "num_frames": 1,
            "num_steps": 2,
            "guidance_scale": 4.5,
            "max_sequence_length": 128,
            "fps": 16,
            "denoise_mode": "sde",
            "noise_level": 0.5,
            "sde_type": "flow_grpo",
        },
    )

    assert len(calls) == 2
    generator = calls[0][3]["generator"]
    assert generator.initial_seed() == 37
    for step_index, (scheduler, noise, timestep, kwargs) in enumerate(calls):
        assert scheduler is model.state.scheduler
        assert noise.dtype == torch.float32
        assert timestep.shape == (1,)
        assert kwargs == {
            "generator": generator,
            "deterministic": False,
            "return_dt": False,
            "noise_level": 0.5,
            "sde_type": "flow_grpo",
            "step_index": step_index,
        }
    torch.testing.assert_close(model.decoded_latents, torch.full_like(model.decoded_latents, 2.5))


def test_seed_formula_lays_out_a_dense_non_colliding_grid() -> None:
    """``base_seed + prompt*samples_per_prompt + sample`` fills the grid."""

    grid = [
        seed_for(
            base_seed=17,
            prompt_index=prompt_index,
            sample_index=sample_index,
            samples_per_prompt=4,
        )
        for prompt_index in range(3)
        for sample_index in range(4)
    ]

    assert grid == list(range(17, 29))


def test_video_to_cthw_accepts_btchw_layout() -> None:
    video = torch.zeros(1, 5, 3, 8, 8)
    video[:, :, 0] = 1.0

    out = video_to_cthw(video)

    assert tuple(out.shape) == (3, 5, 8, 8)
    assert torch.all(out[0] == 1.0)
