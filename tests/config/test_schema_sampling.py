"""Sampling and rollout knobs: the family-selected sampling schema and denoise modes."""

from __future__ import annotations

import pytest

from tests.config.helpers import minimal_grpo_cfg
from vrl.config.sampling_schema import (
    TextEncodedImageSamplingSection,
    VideoSamplingSection,
)
from vrl.config.schema import (
    parse_config,
)

# ── rollout / sampling string-setting Literals ────────────────────────────────


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


def test_family_without_text_length_rejects_max_sequence_length() -> None:
    cfg = minimal_grpo_cfg(model={"family": "cosmos-predict2"})
    cfg.sampling = {"max_sequence_length": 512}

    with pytest.raises(ValueError, match=r"unknown sampling\.max_sequence_length"):
        parse_config(cfg)

    parsed = parse_config(
        minimal_grpo_cfg(model={"family": "cosmos-predict2"}, sampling={}),
    )
    assert type(parsed.sampling) is VideoSamplingSection


def test_sampling_fields_are_rejected_outside_their_behavior_owner() -> None:
    cfg = minimal_grpo_cfg(model={"family": "sana"}, sampling={"fps": 16})

    with pytest.raises(ValueError, match=r"unknown sampling\.fps"):
        parse_config(cfg)
