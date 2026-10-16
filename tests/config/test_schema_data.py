"""The data section: loader discriminator, prompt manifest formats and mixtures,
and sampler types."""

from __future__ import annotations

import pytest

from vrl.config.schema import (
    DataConfig,
)

# ── Data loader discriminator ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fmt,expected",
    [
        ("image_caption_jsonl", "prompt_image_manifest"),
        ("jsonl", "prompt_manifest"),
        ("text", "prompt_manifest"),
    ],
)
def test_omitted_loader_derives_from_preprocessing_format(fmt: str, expected: str) -> None:
    """An omitted data.loader is derived from preprocessing.format for the prompt-* family."""
    if expected == "prompt_image_manifest":
        data = DataConfig(
            manifest="data/external/videophy_i2v/manifests/train.jsonl",
            eval_manifest="data/external/videophy_i2v/manifests/eval.jsonl",
            preprocessing={
                "format": fmt,
                "image_field": "image",
                "caption_field": "caption",
                "conditioning": "reference_image",
            },
            sampler={"type": "random_without_replacement"},
        )
    else:
        data = DataConfig(
            manifest="manifests/ocr/train.txt",
            preprocessing={"format": fmt},
            sampler={"type": "random_without_replacement"},
        )
    assert data.loader == expected


@pytest.mark.parametrize(
    ("loader", "fmt", "message"),
    [
        (
            "prompt_manifest",
            "image_caption_jsonl",
            r"requires.*prompt_image_manifest",
        ),
        (
            "prompt_image_manifest",
            "text",
            r"requires.*image_caption_jsonl",
        ),
    ],
)
def test_explicit_data_loader_rejects_preprocessing_format_conflict(
    loader: str,
    fmt: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DataConfig(
            loader=loader,
            manifest="train.jsonl",
            eval_manifest="eval.jsonl",
            preprocessing={
                "format": fmt,
                "image_field": "image",
                "caption_field": "caption",
                "conditioning": "reference_image",
            },
            sampler={"type": "random_without_replacement"},
        )


def test_prompt_image_manifest_requires_image_caption_fields() -> None:
    with pytest.raises(ValueError, match=r"data\.preprocessing\.caption_field"):
        DataConfig(
            loader="prompt_image_manifest",
            manifest="x",
            eval_manifest="y",
            preprocessing={
                "format": "image_caption_jsonl",
                "image_field": "image",
                "conditioning": "reference_image",
            },
            sampler={"type": "random_without_replacement"},
        )


def test_prompt_manifest_accepts_mixture_counts() -> None:
    """A {path: count} manifest is the recipe-level way to declare a prompt mix."""
    data = DataConfig(
        loader="prompt_manifest",
        manifest={"anatomy.jsonl": 6800, "safety.jsonl": 1200},
        mix_seed=20260818,
        preprocessing={"format": "jsonl"},
        sampler={"type": "random_without_replacement"},
    )
    assert data.manifest == {"anatomy.jsonl": 6800, "safety.jsonl": 1200}


def test_prompt_manifest_mixture_requires_a_seed() -> None:
    """A mixture without a seed would draw a different prompt set on every rank."""
    with pytest.raises(ValueError, match=r"data\.mix_seed"):
        DataConfig(
            loader="prompt_manifest",
            manifest={"anatomy.jsonl": 6800, "safety.jsonl": 1200},
            preprocessing={"format": "jsonl"},
            sampler={"type": "random_without_replacement"},
        )


@pytest.mark.parametrize("count", [0, "many"])
def test_prompt_manifest_rejects_non_positive_mixture_count(count: object) -> None:
    """A mixture count that cannot select prompts fails at config time."""
    with pytest.raises(ValueError, match=r"positive prompt count"):
        DataConfig(
            loader="prompt_manifest",
            manifest={"anatomy.jsonl": count},
            preprocessing={"format": "jsonl"},
            sampler={"type": "random_without_replacement"},
        )


def test_prompt_image_manifest_rejects_mixture() -> None:
    """Image-conditioned runs pair one manifest with its reference tree."""
    with pytest.raises(ValueError, match=r"single data\.manifest path"):
        DataConfig(
            loader="prompt_image_manifest",
            manifest={"a.jsonl": 10, "b.jsonl": 10},
            eval_manifest="eval.jsonl",
            preprocessing={
                "format": "image_caption_jsonl",
                "image_field": "image",
                "caption_field": "caption",
                "conditioning": "reference_image",
            },
            sampler={"type": "random_without_replacement"},
        )


# ── Sampler type literal ──────────────────────────────────────────────────────


def test_unknown_sampler_type_raises() -> None:
    with pytest.raises(ValueError, match=r"unknown data\.sampler\.type"):
        DataConfig(
            loader="prompt_manifest",
            manifest="x",
            preprocessing={},
            sampler={"type": "round_robin"},
        )


# ── Cached layered scenes ─────────────────────────────────────────────────────


def _layered_scenes(**overrides: object) -> DataConfig:
    values: dict[str, object] = {
        "loader": "layered_scenes",
        "manifest": {"/data/v3": 1000, "/data/v1": None},
        "split": "train",
        "token_weights": "weighted",
        "spare_layers_max": 2,
    }
    values.update(overrides)
    return DataConfig(**values)


def test_layered_scenes_takes_roots_with_optional_scene_counts() -> None:
    data = _layered_scenes()

    assert data.loader == "layered_scenes"
    assert data.token_weights == "weighted" and data.spare_layers_max == 2


@pytest.mark.parametrize("field", ["split", "token_weights", "spare_layers_max"])
def test_layered_scenes_requires_its_fields(field: str) -> None:
    with pytest.raises(ValueError, match=rf"data\.{field}"):
        _layered_scenes(**{field: None})


def test_layered_scenes_rejects_a_negative_spare_budget_and_unknown_weights() -> None:
    with pytest.raises(ValueError, match=r"spare_layers_max"):
        _layered_scenes(spare_layers_max=-1)
    with pytest.raises(ValueError, match=r"token_weights"):
        _layered_scenes(token_weights="uniform")
