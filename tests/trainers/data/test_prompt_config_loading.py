"""Config-driven prompt loading: manifest formats and {path: count} mixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

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


def _write_prompts(path: Path, prompts: list[str]) -> Path:
    path.write_text(
        "".join(json.dumps({"prompt": prompt}) + "\n" for prompt in prompts),
        encoding="utf-8",
    )
    return path


def test_runtime_data_loader_mixes_manifests_by_declared_counts(tmp_path: Path) -> None:
    """A {path: count} manifest draws exactly that many prompts from each source."""
    anatomy = _write_prompts(tmp_path / "anatomy.jsonl", [f"anatomy {i}" for i in range(50)])
    safety = _write_prompts(tmp_path / "safety.jsonl", [f"safety {i}" for i in range(50)])

    examples = load_prompt_examples_from_config(
        _data_config(
            {
                "loader": "prompt_manifest",
                "manifest": {str(anatomy): 8, str(safety): 2},
                "mix_seed": 20260818,
                "preprocessing": {"format": "jsonl"},
            },
        ),
    )

    prompts = [example.prompt for example in examples]
    assert len(prompts) == 10
    assert sum(prompt.startswith("anatomy") for prompt in prompts) == 8
    assert sum(prompt.startswith("safety") for prompt in prompts) == 2


def test_runtime_data_loader_mixture_is_seed_reproducible(tmp_path: Path) -> None:
    """Same spec and seed reproduce the prompt set; a different seed redraws it."""
    anatomy = _write_prompts(tmp_path / "anatomy.jsonl", [f"anatomy {i}" for i in range(50)])
    safety = _write_prompts(tmp_path / "safety.jsonl", [f"safety {i}" for i in range(50)])

    def load(seed: int) -> list[str]:
        data = {
            "loader": "prompt_manifest",
            "manifest": {str(anatomy): 8, str(safety): 2},
            "mix_seed": seed,
            "preprocessing": {"format": "jsonl"},
        }
        return [e.prompt for e in load_prompt_examples_from_config(_data_config(data))]

    assert load(7) == load(7)
    assert load(7) != load(8)


def test_runtime_data_loader_rejects_mixture_count_over_manifest_size(tmp_path: Path) -> None:
    """Asking for more prompts than a source holds fails loudly, not silently short."""
    anatomy = _write_prompts(tmp_path / "anatomy.jsonl", ["only one"])

    with pytest.raises(ValueError, match="only 1 available"):
        load_prompt_examples_from_config(
            OmegaConf.create(
                {
                    "loader": "prompt_manifest",
                    "manifest": {str(anatomy): 5},
                    "mix_seed": 20260818,
                    "preprocessing": {"format": "jsonl"},
                },
            ),
        )
