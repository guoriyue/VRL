"""Sampling and rollout knobs: the family-selected sampling schema and denoise modes."""

from __future__ import annotations

import pytest

from tests.config.helpers import minimal_grpo_cfg
from vrl.config.sampling_schema import (
    DenoiseImageSamplingSection,
    TextEncodedImageSamplingSection,
    VideoSamplingSection,
)
from vrl.config.schema import (
    parse_config,
)

# ── rollout / sampling string-setting Literals ────────────────────────────────


@pytest.mark.parametrize("mode", ["native", "sde"])
def test_valid_denoise_modes_accepted(mode: str) -> None:
    """Both denoise modes validate; the Literal is the user-facing allow-list."""
    cfg = minimal_grpo_cfg()
    cfg.rollout.denoise_mode = mode
    assert parse_config(cfg).rollout.denoise_mode == mode


def test_sampling_schema_is_selected_from_model_family() -> None:
    cfg = minimal_grpo_cfg(model={"family": "sana"})
    cfg.sampling = {"max_sequence_length": 300}

    sampling = parse_config(cfg).sampling

    assert type(sampling) is TextEncodedImageSamplingSection
    assert sampling.max_sequence_length == 300


def test_sampling_section_requires_model_family_for_schema_selection() -> None:
    cfg = minimal_grpo_cfg(sampling={"num_steps": 10})

    with pytest.raises(ValueError, match=r"sampling requires model\.family"):
        parse_config(cfg)


@pytest.mark.parametrize(
    ("family", "expected_type"),
    [
        ("hunyuan_image", DenoiseImageSamplingSection),
        ("cosmos-predict2", VideoSamplingSection),
    ],
)
def test_family_without_text_length_rejects_max_sequence_length(
    family: str,
    expected_type: type,
) -> None:
    cfg = minimal_grpo_cfg(model={"family": family})
    cfg.sampling = {"max_sequence_length": 512}

    with pytest.raises(ValueError, match=r"unknown sampling\.max_sequence_length"):
        parse_config(cfg)

    parsed = parse_config(
        minimal_grpo_cfg(model={"family": family}, sampling={}),
    )
    assert type(parsed.sampling) is expected_type


def test_echo_accepts_only_baked_guidance_value() -> None:
    valid = minimal_grpo_cfg(model={"family": "echo"})
    valid.sampling = {"guidance_scale": 1.0}
    assert parse_config(valid).sampling.guidance_scale == 1.0

    invalid = minimal_grpo_cfg(model={"family": "echo"})
    invalid.sampling = {"guidance_scale": 4.5}
    with pytest.raises(ValueError, match=r"unknown sampling\.guidance_scale=4\.5"):
        parse_config(invalid)


@pytest.mark.parametrize(
    ("family", "field", "value"),
    [
        ("magi_1", "guidance_scale", 4.5),
        ("sana", "fps", 16),
    ],
)
def test_sampling_fields_are_rejected_outside_their_behavior_owner(
    family: str,
    field: str,
    value: object,
) -> None:
    cfg = minimal_grpo_cfg(model={"family": family}, sampling={field: value})

    with pytest.raises(ValueError, match=rf"unknown sampling\.{field}"):
        parse_config(cfg)
