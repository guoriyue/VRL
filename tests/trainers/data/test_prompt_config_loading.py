"""Config-driven prompt loading: the manifest format each loader reads."""

from __future__ import annotations

import json
from pathlib import Path

from vrl.config.schema import DataConfig
from vrl.trainers.data.prompts import load_prompt_examples_from_config


def _data_config(payload: dict) -> DataConfig:
    """A parsed ``data`` section; the prompt loaders require a sampler type."""

    payload = dict(payload)
    payload.setdefault("sampler", {"type": "random_without_replacement"})
    return DataConfig.model_validate(payload)


def test_runtime_data_loader_derives_plain_prompt_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "prompts.txt"
    manifest.write_text("a red fox\n", encoding="utf-8")

    examples = load_prompt_examples_from_config(
        _data_config(
            {
                "manifest": str(manifest),
                "preprocessing": {"format": "text"},
            },
        ),
    )

    assert [example.prompt for example in examples] == ["a red fox"]
    assert examples[0].reference_images == []


def test_runtime_data_loader_derives_image_prompt_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "prompts.jsonl"
    manifest.write_text(
        json.dumps({"image": "reference.png", "caption": "a red fox"}) + "\n",
        encoding="utf-8",
    )

    examples = load_prompt_examples_from_config(
        _data_config(
            {
                "manifest": str(manifest),
                "eval_manifest": str(manifest),
                "preprocessing": {
                    "format": "image_caption_jsonl",
                    "image_field": "image",
                    "caption_field": "caption",
                    "conditioning": "reference_image",
                },
            },
        ),
    )

    assert [example.prompt for example in examples] == ["a red fox"]
    assert examples[0].reference_images == ["reference.png"]
