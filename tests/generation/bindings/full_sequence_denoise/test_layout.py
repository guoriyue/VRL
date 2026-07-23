"""Tests for diffusion request layout helpers."""

from __future__ import annotations

import pytest
import torch

from vrl.generation.bindings.full_sequence_denoise import DiffusionRequestLayout
from vrl.generation.types import GenerationRequest


def test_diffusion_layout_rejects_oversized_sde_window() -> None:
    """Checks diffusion layout rejects oversized SDE window."""
    request = _request(
        {
            "sde_window_size": 10,
            "sde_window_range": (0, 5),
        },
    )

    with pytest.raises(ValueError, match="sde_window_size"):
        _layout().parse_sampling_params(request)


@pytest.mark.parametrize("denoise_mode", ["native", "sde"])
def test_diffusion_layout_always_builds_sde_math_params(denoise_mode: str) -> None:
    """Checks both denoise modes carry the non-optional loop math contract."""
    request = _request({"denoise_mode": denoise_mode})

    params = _layout().parse_sampling_params(request)

    assert params.denoise_mode == denoise_mode
    assert params.sde.sde_type == "flow_grpo"
    assert params.sde_window_size == 0
    assert params.sde_window_range == (0, 20)


def test_diffusion_layout_selects_request_owned_sde_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks request window policy resolves before entering the denoise loop."""
    layout = _layout()
    params = layout.parse_sampling_params(
        _request(
            {
                "sde_window_size": 2,
                "sde_window_range": (3, 8),
            },
        ),
    )
    monkeypatch.setattr(
        "vrl.generation.bindings.full_sequence_denoise.layout.random.randint",
        lambda lo, hi: hi,
    )

    assert layout.select_sde_window(params) == (6, 8)

    no_window = layout.parse_sampling_params(_request({"sde_window_size": 0}))
    assert layout.select_sde_window(no_window) is None


@pytest.mark.parametrize(
    "window_range",
    [(-1, 2), (2, 2), (2, 21), (2,), "bad"],
)
def test_diffusion_layout_rejects_invalid_sde_window_range(
    window_range: object,
) -> None:
    """Checks malformed request window policies fail at request parsing."""
    with pytest.raises(ValueError, match="sde_window_range"):
        _layout().parse_sampling_params(
            _request({"sde_window_range": window_range}),
        )


def test_diffusion_layout_rejects_unknown_denoise_mode() -> None:
    """Checks diffusion layout rejects unknown denoise mode."""
    request = _request({"denoise_mode": "custom"})

    with pytest.raises(ValueError, match="denoise_mode"):
        _layout().parse_sampling_params(request)


def test_diffusion_layout_repeat_batch_rejects_unexpected_batch_size() -> None:
    """Checks diffusion layout repeat batch rejects unexpected batch size."""
    layout = _layout()

    repeated = layout.repeat_batch(torch.ones(1, 2), 3)
    assert repeated.shape == (3, 2)

    already_sized = torch.ones(3, 2)
    assert layout.repeat_batch(already_sized, 3) is already_sized

    with pytest.raises(ValueError, match="cannot repeat tensor batch=2"):
        layout.repeat_batch(torch.ones(2, 2), 3)


def _layout() -> DiffusionRequestLayout:
    """A layout with explicit fallbacks (the executor is the real source)."""
    return DiffusionRequestLayout(
        default_samples_per_chunk=1,
        default_num_frames=1,
        default_fps=None,
        default_max_sequence_length=512,
        sde_type="flow_grpo",
    )


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
        inputs=["p0"],
        samples_per_prompt=1,
        sampling=sampling,
    )
