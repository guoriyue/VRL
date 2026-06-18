"""Tests for vrl.trainers.data (DistributedKRepeatSampler)."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf


class TestDistributedKRepeatSampler:
    """Groups tests for distributed krepeat sampler."""
    def test_k_repeat_distribution(self) -> None:
        """Checks k repeat distribution."""
        import torch
        from torch.utils.data import TensorDataset

        from vrl.trainers.data import DistributedKRepeatSampler

        dataset = TensorDataset(torch.arange(100))
        sampler = DistributedKRepeatSampler(
            dataset=dataset, batch_size=6, k=3, num_replicas=2, rank=0, seed=42
        )
        it = iter(sampler)
        batch = next(it)
        assert len(batch) == 6

    def test_rank_sync(self) -> None:
        """Both ranks should see the same unique prompts."""
        import torch
        from torch.utils.data import TensorDataset

        from vrl.trainers.data import DistributedKRepeatSampler

        dataset = TensorDataset(torch.arange(100))
        s0 = DistributedKRepeatSampler(dataset=dataset, batch_size=4, k=2, num_replicas=2, rank=0, seed=0)
        s1 = DistributedKRepeatSampler(dataset=dataset, batch_size=4, k=2, num_replicas=2, rank=1, seed=0)
        b0 = next(iter(s0))
        b1 = next(iter(s1))
        # Together they should have 8 items from 4 unique indices, each repeated 2x
        all_indices = b0 + b1
        assert len(all_indices) == 8
        unique = set(all_indices)
        assert len(unique) == 4


def test_image_caption_prompt_manifest_maps_to_reference_image(tmp_path) -> None:
    """Checks image caption prompt manifest maps to reference image."""
    from vrl.trainers.data import load_prompt_image_manifest

    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        (
            '{"image":"images/000.png","caption":"A ball rolls down a ramp.",'
            '"seed":7,"metadata":{"source":"sd3_5"}}\n'
        ),
        encoding="utf-8",
    )

    examples = load_prompt_image_manifest(manifest)

    assert len(examples) == 1
    assert examples[0].prompt == "A ball rolls down a ramp."
    assert examples[0].reference_image == "images/000.png"
    assert examples[0].task_type == "image_to_video"
    assert examples[0].metadata == {"source": "sd3_5", "seed": 7}


def test_image_caption_prompt_manifest_reports_missing_fields(tmp_path) -> None:
    """Checks image caption prompt manifest reports missing fields."""
    from vrl.trainers.data import load_prompt_image_manifest

    manifest = tmp_path / "train.jsonl"
    manifest.write_text('{"image":"images/000.png"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"row 0 missing required field 'caption'"):
        load_prompt_image_manifest(manifest)


def test_prompt_examples_from_config_dispatches_image_caption_loader(tmp_path) -> None:
    """Checks prompt examples from config dispatches image caption loader."""
    from vrl.trainers.data import load_prompt_examples_from_config

    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        '{"image":"images/000.png","caption":"Water splashes into a bowl."}\n',
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "loader": "prompt_image_manifest",
            "manifest": manifest.as_posix(),
            "task_type": "image_to_video",
            "preprocessing": {
                "image_field": "image",
                "caption_field": "caption",
            },
        },
    )

    examples = load_prompt_examples_from_config(cfg)

    assert examples[0].prompt == "Water splashes into a bowl."
    assert examples[0].reference_image == "images/000.png"


def test_prompt_examples_from_config_derives_image_caption_loader_when_omitted(tmp_path) -> None:
    """Omitting loader with format=image_caption_jsonl derives the image-caption loader."""
    from vrl.trainers.data import load_prompt_examples_from_config

    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        '{"image":"images/000.png","caption":"Water splashes into a bowl."}\n',
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "manifest": manifest.as_posix(),
            "task_type": "image_to_video",
            "preprocessing": {
                "format": "image_caption_jsonl",
                "image_field": "image",
                "caption_field": "caption",
            },
        },
    )

    examples = load_prompt_examples_from_config(cfg)

    assert examples[0].prompt == "Water splashes into a bowl."
    assert examples[0].reference_image == "images/000.png"


def test_prompt_examples_from_config_defaults_to_plain_manifest_when_omitted(tmp_path) -> None:
    """Omitting loader with a non-image-caption format falls back to the plain prompt manifest."""
    from vrl.trainers.data import load_prompt_examples_from_config

    manifest = tmp_path / "train.txt"
    manifest.write_text('a cat "cat"\n', encoding="utf-8")
    cfg = OmegaConf.create(
        {
            "manifest": manifest.as_posix(),
            "preprocessing": {"format": "text"},
        },
    )

    examples = load_prompt_examples_from_config(cfg)

    assert examples[0].prompt == 'a cat "cat"'
