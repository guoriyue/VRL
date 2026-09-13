"""Sampling and rollout knobs: the family-selected sampling schema, attention
backend ownership, denoise modes and the scheduler batch size."""

from __future__ import annotations

import pytest

from tests.config.helpers import minimal_grpo_cfg
from vrl.config.sampling_schema import (
    ARSamplingSection,
    DenoiseImageSamplingSection,
    JanusProSamplingSection,
    TextEncodedImageSamplingSection,
    VideoSamplingSection,
)
from vrl.config.schema import (
    parse_config,
)


@pytest.mark.parametrize("value", [None, 1, 8])
def test_sampling_scheduler_batch_size_accepts_null_or_positive_integer(
    value: int | None,
) -> None:
    sampling = ARSamplingSection.model_validate({"ar_scheduler_batch_size": value})

    assert sampling.ar_scheduler_batch_size == value


@pytest.mark.parametrize("value", [True, 0])
def test_sampling_scheduler_batch_size_rejects_coercible_or_non_positive_values(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="must be a positive integer or null"):
        ARSamplingSection.model_validate({"ar_scheduler_batch_size": value})


# ── rollout / sampling string-setting Literals ────────────────────────────────


@pytest.mark.parametrize("mode", ["native", "sde"])
def test_valid_denoise_modes_accepted(mode: str) -> None:
    """Both denoise modes validate; the Literal is the user-facing allow-list."""
    cfg = minimal_grpo_cfg()
    cfg.rollout.denoise_mode = mode
    assert parse_config(cfg).rollout.denoise_mode == mode


def test_shared_attention_backend_omission_stays_unset() -> None:
    """The runtime fallback owns the default; typed config stores no duplicate."""
    cfg = minimal_grpo_cfg(model={"family": "janus_pro"})
    cfg.sampling = {}
    sampling = parse_config(cfg).sampling

    assert type(sampling) is JanusProSamplingSection
    assert sampling.attention_backend is None
    assert sampling.model_dump(exclude_unset=True) == {}


def test_shared_attention_family_accepts_explicit_backend() -> None:
    cfg = minimal_grpo_cfg(model={"family": "janus_pro"})
    cfg.sampling = {"attention_backend": "torch_native"}

    assert parse_config(cfg).sampling.attention_backend == "torch_native"


def test_unknown_attention_backend_raises() -> None:
    """An out-of-set attention_backend is rejected at parse with the dotted path."""
    cfg = minimal_grpo_cfg(model={"family": "janus_pro"})
    cfg.sampling = {"attention_backend": "bogus"}
    with pytest.raises(ValueError, match=r"unknown sampling\.attention_backend"):
        parse_config(cfg)


def test_native_cache_family_rejects_typed_attention_backend() -> None:
    cfg = minimal_grpo_cfg(model={"family": "llamagen"})
    cfg.sampling = {"attention_backend": "torch_native"}

    with pytest.raises(ValueError, match=r"unknown sampling\.attention_backend"):
        parse_config(cfg)


def test_llamagen_sampling_rejects_model_derived_topology() -> None:
    cfg = minimal_grpo_cfg(model={"family": "llamagen"}, sampling={"image_token_num": 256})

    with pytest.raises(ValueError, match=r"unknown sampling\.image_token_num"):
        parse_config(cfg)


@pytest.mark.parametrize(
    "family",
    ["janus_pro", "nextstep_1", "emu3", "glm_image"],
)
def test_text_encoded_ar_families_keep_request_text_length(family: str) -> None:
    cfg = minimal_grpo_cfg(
        model={"family": family},
        sampling={"max_text_length": 128},
    )

    sampling = parse_config(cfg).sampling

    assert sampling is not None
    assert sampling.max_text_length == 128


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
        ("janus_pro", "max_reflect_len", 80),
        ("magi_1", "guidance_scale", 4.5),
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


def test_reflection_length_is_owned_only_by_janus_r1() -> None:
    cfg = minimal_grpo_cfg(
        model={"family": "janus_pro_r1"},
        sampling={"max_reflect_len": 80},
    )
    cfg.algorithm.kind = "token_grpo_multisegment"
    cfg.rollout.final_image_policy = "always_generate"

    assert parse_config(cfg).sampling.max_reflect_len == 80
