"""Wan batch-local UniPC history preserves trajectories without retaining tensors."""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from diffusers import UniPCMultistepScheduler

from vrl.generation.types import DenoiseRequest
from vrl.models.families.wan_2_1.model import WanI2VDiffusersModel, WanT2VDiffusersModel


@pytest.mark.parametrize("model_cls", [WanT2VDiffusersModel, WanI2VDiffusersModel])
def test_batch_history_is_released_and_repeated_sampling_matches_shared_scheduler(model_cls):
    template = UniPCMultistepScheduler(
        prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0
    )
    legacy = UniPCMultistepScheduler.from_config(template.config)
    transformer = torch.nn.Linear(4, 4)
    transformer.config = SimpleNamespace(in_channels=4)

    def prepare_t2v(batch, channels, height, width, frames, dtype, device, generator, latents):
        return torch.randn((batch, channels, 1, 2, 2), generator=generator)

    def prepare_i2v(
        image, batch, channels, height, width, frames, dtype, device, generator, latents, condition
    ):
        sample = torch.randn((batch, channels, 1, 2, 2), generator=generator)
        return sample, torch.zeros_like(sample)

    pipe = SimpleNamespace(
        transformer=transformer,
        scheduler=template,
        vae=SimpleNamespace(config=SimpleNamespace(z_dim=4)),
        video_processor=SimpleNamespace(preprocess=lambda image, **kwargs: image),
        prepare_latents=prepare_i2v if model_cls is WanI2VDiffusersModel else prepare_t2v,
    )
    model = model_cls(pipeline=pipe, device=torch.device("cpu"))
    encoded = {
        "prompt_embeds": torch.zeros(1, 3, 4),
        "reference_image": torch.zeros(1, 3, 8, 8),
        "image_embeds": torch.zeros(1, 2, 4),
    }
    request = DenoiseRequest(
        width=8, height=8, frame_count=1, num_steps=4, guidance_scale=1.0, seed=7
    )
    for _ in range(2):
        state = model.prepare_sampling(request, encoded)
        legacy.set_timesteps(request.num_steps, device="cpu")
        expected = state.latents.clone()
        actual = state.latents.clone()
        for timestep in state.timesteps:
            expected = legacy.step(expected * 0.125, timestep, expected).prev_sample
            actual = state.scheduler.step(actual * 0.125, timestep, actual).prev_sample
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        history = weakref.ref(state.scheduler.last_sample)
        scheduler = weakref.ref(state.scheduler)
        assert history() is not None
        del state, actual, expected
        gc.collect()
        assert scheduler() is None
        assert history() is None
        assert template.last_sample is None
        assert all(value is None for value in template.model_outputs)
