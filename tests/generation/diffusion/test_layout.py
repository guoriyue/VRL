"""Tests for diffusion request layout helpers."""

from __future__ import annotations

import pytest
import torch

from vrl.generation.diffusion import DiffusionRequestLayout
from vrl.generation.types import GenerationRequest


def test_diffusion_layout_rejects_oversized_sde_window() -> None:
    request = _request(
        {
            "sde_window_size": 10,
            "sde_window_range": (0, 5),
        },
    )

    with pytest.raises(ValueError, match="sde_window_size"):
        DiffusionRequestLayout().parse_sampling_params(request)


def test_diffusion_layout_normalizes_native_denoise_mode() -> None:
    request = _request({"denoise_mode": "native"})

    params = DiffusionRequestLayout().parse_sampling_params(request)

    assert params.denoise_mode == "native"


def test_diffusion_layout_rejects_unknown_denoise_mode() -> None:
    request = _request({"denoise_mode": "custom"})

    with pytest.raises(ValueError, match="denoise_mode"):
        DiffusionRequestLayout().parse_sampling_params(request)


def test_diffusion_layout_repeat_batch_rejects_unexpected_batch_size() -> None:
    layout = DiffusionRequestLayout()

    repeated = layout.repeat_batch(torch.ones(1, 2), 3)
    assert repeated.shape == (3, 2)

    already_sized = torch.ones(3, 2)
    assert layout.repeat_batch(already_sized, 3) is already_sized

    with pytest.raises(ValueError, match="cannot repeat tensor batch=2"):
        layout.repeat_batch(torch.ones(2, 2), 3)


def _request(extra_sampling: dict[str, object]) -> GenerationRequest:
    sampling = {
        "num_steps": 20,
        "guidance_scale": 4.5,
        "height": 64,
        "width": 64,
        **extra_sampling,
    }
    return GenerationRequest(
        request_id="req",
        family="sd3_5",
        task="t2i",
        prompts=["p0"],
        samples_per_prompt=1,
        sampling=sampling,
    )
