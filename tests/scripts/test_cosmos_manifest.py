from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from vrl.scripts.cosmos.train import _load_cosmos_prompt_examples
from vrl.trainers.data import load_prompt_manifest


def test_text_manifest_still_loads_plain_prompts(tmp_path: Path) -> None:
    manifest = tmp_path / "prompts.txt"
    manifest.write_text("A robot walking through a warehouse.\n", encoding="utf-8")

    examples = load_prompt_manifest(manifest)

    assert len(examples) == 1
    assert examples[0].prompt == "A robot walking through a warehouse."
    assert examples[0].reference_image == ""


def test_jsonl_manifest_loads_prompt_and_reference_image(tmp_path: Path) -> None:
    manifest = tmp_path / "prompts.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "prompt": "A robot walking through a warehouse.",
                "reference_image": "reference.png",
            },
        )
        + "\n",
        encoding="utf-8",
    )

    examples = load_prompt_manifest(manifest)

    assert len(examples) == 1
    assert examples[0].prompt == "A robot walking through a warehouse."
    assert examples[0].reference_image == "reference.png"


def test_per_sample_cosmos_manifest_requires_reference_image(tmp_path: Path) -> None:
    manifest = tmp_path / "prompts.jsonl"
    manifest.write_text(json.dumps({"prompt": "missing reference"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="reference_image"):
        _load_cosmos_prompt_examples(
            manifest,
            reference_mode="per_sample",
            rollout_batch_size=1,
        )


def test_per_sample_cosmos_manifest_resolves_reference_paths(tmp_path: Path) -> None:
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(tmp_path / "reference.png")
    manifest = tmp_path / "prompts.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "prompt": "A red room.",
                "reference_image": "reference.png",
            },
        )
        + "\n",
        encoding="utf-8",
    )

    examples = _load_cosmos_prompt_examples(
        manifest,
        reference_mode="per_sample",
        rollout_batch_size=1,
    )

    assert examples[0].reference_image == str((tmp_path / "reference.png").resolve())
    assert examples[0].metadata["reference_image"] == examples[0].reference_image


def test_per_sample_cosmos_manifest_rejects_multi_prompt_batches(tmp_path: Path) -> None:
    manifest = tmp_path / "prompts.jsonl"
    manifest.write_text(
        json.dumps({"prompt": "A red room.", "reference_image": "reference.png"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"rollout\.rollout_batch_size=1"):
        _load_cosmos_prompt_examples(
            manifest,
            reference_mode="per_sample",
            rollout_batch_size=2,
        )
