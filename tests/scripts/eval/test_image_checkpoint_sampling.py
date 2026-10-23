"""The image checkpoint evaluator's sampling projection and native generation step."""

from __future__ import annotations

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf

from vrl.config.schema import parse_config
from vrl.models.steps.denoise.base import DenoiseSamplingStateBase
from vrl.scripts.eval.image_checkpoint_eval import ImageSampling, generate_image

_IMAGE = {"width": 512, "height": 512, "num_steps": 10, "guidance_scale": 4.5}


def _root(*, family: str = "sd3_5", sampling: dict, executor: dict | None = None):
    model = {"family": family}
    if executor is not None:
        model["executor"] = executor
    return parse_config(OmegaConf.create({"model": model, "sampling": sampling}))


def test_family_key_declared_in_sampling_wins_over_executor() -> None:
    root = _root(
        sampling={**_IMAGE, "max_sequence_length": 256},
        executor={"max_sequence_length": 128},
    )

    assert ImageSampling.from_root(root).max_sequence_length == 256


def test_family_key_falls_back_to_model_executor_like_the_runtime() -> None:
    root = _root(sampling=_IMAGE, executor={"max_sequence_length": 128})

    assert ImageSampling.from_root(root).max_sequence_length == 128


def test_family_key_missing_everywhere_names_the_config_path() -> None:
    with pytest.raises(
        ValueError, match=r"config missing required field: sampling\.max_sequence_length"
    ):
        ImageSampling.from_root(_root(sampling=_IMAGE))


def test_family_keys_come_from_the_family_schema_not_a_hand_list() -> None:
    """Qwen-Image-2.1 declares reference preprocessing and no prompt-length knob.

    negative_prompt and teacache belong to the shared section and never ride
    along into the record handed to the family executor.
    """

    root = _root(
        family="qwen_image_21",
        sampling={
            **_IMAGE,
            "guidance_scale": 1.0,
            "reference_resolution": 512,
            "negative_prompt": "blurry",
        },
    )

    sampling = ImageSampling.from_root(root)

    assert sampling.max_sequence_length is None
    assert sampling.to_record() == {
        "width": 512,
        "height": 512,
        "num_steps": 10,
        "guidance_scale": 1.0,
        "max_sequence_length": None,
        "reference_resolution": 512,
        "output_mode": "rgb",
    }


class _StepwiseModel:
    """Minimal model boundary with real scheduler arithmetic and CPU tensors."""

    def encode_prompt(self, prompt, negative_prompt, **kwargs):
        self.conditioning = (prompt, negative_prompt, kwargs)
        return {"batch_size": len(prompt)}

    def prepare_sampling(self, request, encoded):
        self.request = request
        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(request.num_steps)
        return DenoiseSamplingStateBase(
            latents=torch.full((encoded["batch_size"], 3, 2, 2), 0.5, dtype=torch.bfloat16),
            timesteps=scheduler.timesteps,
            scheduler=scheduler,
        )

    def forward_step(self, state, step_idx):
        assert not torch.is_grad_enabled()
        return {"noise_pred": torch.full_like(state.latents, 0.25, dtype=torch.bfloat16)}

    def decode_latents(self, latents):
        self.decode_grad_enabled = torch.is_grad_enabled()
        self.decoded_latents = latents.clone()
        return latents


def test_native_generation_preserves_conditioning_and_float32_scheduler_state() -> None:
    model = _StepwiseModel()

    image = generate_image(
        model,
        prompt="a bird",
        negative_prompt="blur",
        seed=37,
        sampling=ImageSampling(2, 2, 2, 4.5, 128),
    )

    assert image.size == (2, 2)
    assert model.conditioning == (
        ["a bird"],
        ["blur"],
        {"max_sequence_length": 128, "guidance_scale": 4.5},
    )
    assert (model.request.negative_prompt, model.request.seed) == ("blur", 37)
    assert model.request.frame_count == 1
    assert model.decode_grad_enabled
    assert model.decoded_latents.dtype == torch.float32
    torch.testing.assert_close(model.decoded_latents, torch.full_like(model.decoded_latents, 0.25))
