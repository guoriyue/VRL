"""Tests for diffusion request layout helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vrl.generation.bindings.full_sequence_denoise import (
    DiffusionChunkExecutor,
    DiffusionRequestLayout,
)
from vrl.generation.types import GenerationRequest
from vrl.models.families.cosmos.cosmos3.runtime import Cosmos3ChunkExecutor
from vrl.models.families.cosmos.predict2.runtime import CosmosChunkExecutor
from vrl.models.families.cosmos.predict2_5.runtime import (
    CosmosPredict25ChunkExecutor,
)
from vrl.models.families.echo.runtime import EchoChunkExecutor
from vrl.models.families.wan_2_1.runtime import Wan_2_1I2VChunkExecutor


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


@pytest.mark.parametrize(
    ("max_sequence_length", "expected_extra"),
    [
        (None, {}),
        (123, {"max_sequence_length": 123}),
    ],
)
def test_diffusion_executor_only_projects_real_text_length(
    max_sequence_length: int | None,
    expected_extra: dict[str, int],
) -> None:
    """An absent family/request value stays absent at both model boundaries."""

    model = SimpleNamespace()
    executor = DiffusionChunkExecutor(
        model,
        family="test",
        task="t2i",
        max_sequence_length=max_sequence_length,
    )
    params = executor.parse_sampling_params(_request({}))
    video_request = executor.build_video_request("p0", params)

    assert params.base.max_sequence_length == max_sequence_length
    assert params.base.text_encode_kwargs() == {
        "guidance_scale": 4.5,
        **expected_extra,
    }
    assert video_request.extra == expected_extra


@pytest.mark.parametrize(
    ("executor_cls", "expected"),
    [
        (CosmosChunkExecutor, None),
        (Cosmos3ChunkExecutor, None),
        (EchoChunkExecutor, None),
        (CosmosPredict25ChunkExecutor, 512),
        (Wan_2_1I2VChunkExecutor, 512),
    ],
)
def test_custom_family_text_length_defaults_match_encoder_capability(
    executor_cls: type,
    expected: int | None,
) -> None:
    model = (
        SimpleNamespace(video_height=64, video_width=64)
        if executor_cls is EchoChunkExecutor
        else object()
    )
    executor = executor_cls(model)
    params = executor.parse_sampling_params(_request({}))

    assert params.base.max_sequence_length == expected
    assert executor.build_video_request("p0", params).extra == (
        {} if expected is None else {"max_sequence_length": expected}
    )


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
