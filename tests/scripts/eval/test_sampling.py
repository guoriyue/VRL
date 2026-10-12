"""Tests for the shared eval sampling projection."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from vrl.config.schema import parse_config
from vrl.scripts.eval._sampling import resolve_eval_sampling


def _root(*, sampling: dict, executor: dict | None = None):
    model = {"family": "sd3_5"}
    if executor is not None:
        model["executor"] = executor
    return parse_config(OmegaConf.create({"model": model, "sampling": sampling}))


_IMAGE = {"width": 512, "height": 512, "num_steps": 10, "guidance_scale": 4.5}


def test_family_key_declared_in_sampling_wins_over_executor() -> None:
    root = _root(
        sampling={**_IMAGE, "max_sequence_length": 256},
        executor={"max_sequence_length": 128},
    )

    assert resolve_eval_sampling(root)["max_sequence_length"] == 256


def test_family_key_falls_back_to_model_executor_like_the_runtime() -> None:
    root = _root(sampling=_IMAGE, executor={"max_sequence_length": 128})

    assert resolve_eval_sampling(root)["max_sequence_length"] == 128


def test_family_key_missing_everywhere_names_the_config_path() -> None:
    root = _root(sampling=_IMAGE)

    with pytest.raises(
        ValueError, match=r"config missing required field: sampling\.max_sequence_length"
    ):
        resolve_eval_sampling(root)


def test_cli_override_wins_and_explicit_zero_guidance_is_kept() -> None:
    root = _root(sampling=_IMAGE, executor={"max_sequence_length": 128})

    out = resolve_eval_sampling(root, overrides={"num_steps": 3, "guidance_scale": 0.0})

    assert out["num_steps"] == 3
    assert out["guidance_scale"] == 0.0


def test_family_without_a_prompt_length_knob_projects_no_key() -> None:
    """Qwen-Image-2.1 tokenizes its template whole: no max_sequence_length anywhere.

    The projection must not demand a key the family's sampling section does not
    declare, and the evaluator's ImageSampling must accept its absence.
    """
    from vrl.scripts.eval.denoise_generation import ImageSampling

    root = parse_config(
        OmegaConf.create({"model": {"family": "qwen_image_21"}, "sampling": _IMAGE})
    )

    out = resolve_eval_sampling(root)
    assert "max_sequence_length" not in out

    sampling = ImageSampling.from_root(root)
    assert sampling.max_sequence_length is None
    assert ImageSampling.from_mapping(sampling.to_record()) == sampling


def test_family_keys_come_from_the_family_schema_not_a_hand_list() -> None:
    root = parse_config(
        OmegaConf.create(
            {
                "model": {"family": "qwen_image_21"},
                "sampling": {
                    **_IMAGE,
                    "guidance_scale": 1.0,
                    "reference_resolution": 512,
                    "output_mode": "rgb",
                    "negative_prompt": "blurry",
                },
            }
        )
    )

    out = resolve_eval_sampling(root, overrides={"width": 1024, "height": 1024})

    # Qwen-Image-2.1 declares reference preprocessing and no prompt-length knob;
    # negative_prompt and teacache belong to the shared section and never ride along.
    assert out == {
        "width": 1024,
        "height": 1024,
        "num_steps": 10,
        "guidance_scale": 1.0,
        "reference_resolution": 512,
        "output_mode": "rgb",
    }
