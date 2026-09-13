"""The data section: loader discriminator, prompt manifest formats and mixtures,
and sampler types."""

from __future__ import annotations

import pytest

from tests.config.helpers import literal_args
from vrl.config.schema import (
    DataConfig,
)

# ── Data loader discriminator ─────────────────────────────────────────────────


@pytest.mark.parametrize("loader", literal_args(DataConfig.model_fields["loader"].annotation))
def test_valid_data_loaders_are_accepted(loader: str) -> None:
    """Every loader in the DataConfig.loader Literal allow-list is accepted; the
    per-loader construction branches below stay as real behavior coverage."""
    if loader == "prompt_manifest":
        data = DataConfig(
            loader=loader,
            manifest="datasets/ocr/train.txt",
            preprocessing={"format": "text"},
            sampler={"type": "random_without_replacement"},
        )
    elif loader == "prompt_image_manifest":
        data = DataConfig(
            loader=loader,
            manifest="data/external/videophy_i2v/manifests/train.jsonl",
            eval_manifest="data/external/videophy_i2v/manifests/eval.jsonl",
            preprocessing={
                "format": "image_caption_jsonl",
                "image_field": "image",
                "caption_field": "caption",
                "conditioning": "reference_image",
            },
            sampler={"type": "random_without_replacement"},
        )
    else:
        data = DataConfig(
            loader=loader,
            dataset_name="org/dataset",
            split="train",
            cache_dir="/tmp/cache",
            preprocessing={"resolution": 512, "random_crop": False, "horizontal_flip": True},
            sampler={"shuffle": True, "drop_last": True, "dataloader_num_workers": 4},
        )
    assert data.loader == loader


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
            manifest="datasets/ocr/train.txt",
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


@pytest.mark.parametrize(
    "sampler_type",
    ["random_without_replacement", "sequential_window"],
)
def test_valid_sampler_types_are_accepted(sampler_type: str) -> None:
    """Every sampler type in the registered set is accepted as-is."""
    data = DataConfig(
        loader="prompt_manifest",
        manifest="x",
        preprocessing={"format": "text"},
        sampler={"type": sampler_type},
    )
    assert data.sampler.type == sampler_type


def test_unknown_sampler_type_raises() -> None:
    with pytest.raises(ValueError, match=r"unknown data\.sampler\.type"):
        DataConfig(
            loader="prompt_manifest",
            manifest="x",
            preprocessing={},
            sampler={"type": "round_robin"},
        )
