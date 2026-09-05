from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from vrl.scripts.eval import anima_exact_count_checkpoint_eval as checkpoint_eval
from vrl.scripts.families.cosmos.anima.generation_protocol import (
    ANIMA_GENERATION_SCHEMA,
    AnimaGenerationArchive,
    AnimaPixelPairingProtocol,
    validate_paired_generation_archives,
)


def _write_archive(
    root: Path,
    *,
    color: tuple[int, int, int],
    partial: str | None = None,
    image_size: tuple[int, int] = (8, 8),
    run_config_updates: dict[str, object] | None = None,
) -> Path:
    image_dir = root / "images"
    image_dir.mkdir(parents=True)
    prompt = "An anime illustration shows exactly four adult ceramicists in a studio."
    metadata_rows = []
    anchor_rows = []
    for sample_index in range(2):
        image_path = image_dir / f"anima_0000_{sample_index:02d}.png"
        sample_color = tuple(min(channel + sample_index, 255) for channel in color)
        Image.new("RGB", image_size, sample_color).save(image_path, format="PNG")
        prompt_metadata = {"expected_people": 4, "bucket": "studio"}
        reward_metadata = dict(prompt_metadata)
        metadata_rows.append(
            {
                "prompt_index": 0,
                "sample_index": sample_index,
                "seed": 17,
                "prompt": prompt,
                "prompt_metadata": prompt_metadata,
                "reward_metadata": reward_metadata,
                "image_path": str(image_path),
            },
        )
        anchor_rows.append(
            {
                "prompt": prompt,
                "target_image": str(image_path.relative_to(root)),
                "metadata": {
                    **prompt_metadata,
                    "anchor_sample_index": sample_index,
                    "anchor_seed": 17,
                    "anchor_source": "anima_base_synthetic",
                },
            },
        )

    if partial == "metadata.jsonl":
        metadata_rows.pop()
    if partial == "anchor_manifest.jsonl":
        anchor_rows.pop()
    (root / "metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metadata_rows),
        encoding="utf-8",
    )
    (root / "anchor_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in anchor_rows),
        encoding="utf-8",
    )
    run_config: dict[str, object] = {
        "schema": ANIMA_GENERATION_SCHEMA,
        "prompt_count": 1,
        "samples_per_prompt": 2,
        "base_seed": 17,
        "sampling": {
            "width": image_size[0],
            "height": image_size[1],
            "num_steps": 1,
            "guidance_scale": 4.5,
            "max_sequence_length": 16,
        },
        "negative_prompt": "low quality",
        "execution": {"device": "cuda:0", "dtype": "bfloat16"},
        "generator_runtime": {
            "python": "3.12.2",
            "packages": {"torch": "2.11.0", "diffusers": "0.39.0"},
            "vrl_python_tree_sha256": "a" * 64,
        },
        "generation_policy": {"family": "cosmos-predict2-anima"},
        "model_identity": {
            "schema": "vrl.model-identity/v1",
            "sources": {"main": {"revision": "test"}},
            "build": {"transformer_file": "anima.safetensors", "use_lora": False},
        },
        "model": {"use_lora": False},
        "prompt_source": {"manifest_sha256": "test"},
    }
    run_config.update(run_config_updates or {})
    (root / "run_config.json").write_text(
        json.dumps(run_config),
        encoding="utf-8",
    )
    return root


@pytest.mark.parametrize("partial", ["metadata.jsonl", "anchor_manifest.jsonl"])
def test_generation_arm_rejects_partial_grid(tmp_path: Path, partial: str) -> None:
    generation_dir = _write_archive(
        tmp_path / partial.removesuffix(".jsonl"),
        color=(10, 20, 30),
        partial=partial,
    )

    with pytest.raises(ValueError, match=r"generation grid|anchor_manifest"):
        AnimaGenerationArchive.load(generation_dir)


def test_generation_arm_rejects_image_dimensions_outside_protocol(tmp_path: Path) -> None:
    generation_dir = _write_archive(
        tmp_path / "wrong_dimensions",
        color=(10, 20, 30),
        run_config_updates={
            "sampling": {
                "width": 16,
                "height": 8,
                "num_steps": 1,
                "guidance_scale": 4.5,
                "max_sequence_length": 16,
            },
        },
    )

    with pytest.raises(ValueError, match="image dimensions differ"):
        AnimaGenerationArchive.load(generation_dir)


def test_generation_archive_does_not_own_exact_count_target(tmp_path: Path) -> None:
    generation_dir = _write_archive(
        tmp_path / "generic",
        color=(10, 20, 30),
    )
    metadata_path = generation_dir / "metadata.jsonl"
    rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["reward_metadata"].pop("expected_people")
    metadata_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    archive = AnimaGenerationArchive.load(generation_dir)

    assert archive.protocol.generator_runtime["vrl_python_tree_sha256"] == "a" * 64
    assert all("expected_people" not in cell.reward_metadata for cell in archive.cells)


def test_pixel_pairing_ignores_device_ordinal_and_broad_tree_hash(tmp_path: Path) -> None:
    base = AnimaGenerationArchive.load(
        _write_archive(tmp_path / "base", color=(10, 20, 30)),
    )
    checkpoint = AnimaGenerationArchive.load(
        _write_archive(
            tmp_path / "checkpoint",
            color=(30, 20, 10),
            run_config_updates={
                "execution": {"device": "cuda:1", "dtype": "bfloat16"},
                "generator_runtime": {
                    "python": "3.12.2",
                    "packages": {"torch": "2.11.0", "diffusers": "0.39.0"},
                    "vrl_python_tree_sha256": "b" * 64,
                },
            },
        ),
    )

    protocol = AnimaPixelPairingProtocol.from_generation_protocol(base.protocol)

    assert protocol.execution_device_type == "cuda"
    assert "vrl_python_tree_sha256" not in protocol.generator_runtime
    validate_paired_generation_archives(base, checkpoint)


def test_pixel_pairing_rejects_a_different_device_backend(tmp_path: Path) -> None:
    base = AnimaGenerationArchive.load(
        _write_archive(tmp_path / "base", color=(10, 20, 30)),
    )
    checkpoint = AnimaGenerationArchive.load(
        _write_archive(
            tmp_path / "checkpoint",
            color=(30, 20, 10),
            run_config_updates={
                "execution": {"device": "cpu", "dtype": "bfloat16"},
            },
        ),
    )

    with pytest.raises(ValueError, match="execution_device_type"):
        validate_paired_generation_archives(base, checkpoint)


def test_pixel_pairing_rejects_different_cell_metadata(tmp_path: Path) -> None:
    base = AnimaGenerationArchive.load(
        _write_archive(tmp_path / "base", color=(10, 20, 30)),
    )
    checkpoint_dir = _write_archive(
        tmp_path / "checkpoint",
        color=(30, 20, 10),
    )
    metadata_path = checkpoint_dir / "metadata.jsonl"
    rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["reward_metadata"]["evaluation_tag"] = "checkpoint-only"
    metadata_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    checkpoint = AnimaGenerationArchive.load(checkpoint_dir)

    with pytest.raises(ValueError, match=r"cell differs.*reward_metadata"):
        validate_paired_generation_archives(base, checkpoint)


def test_exact_count_target_must_match_prompt_metadata(tmp_path: Path) -> None:
    archive_dir = _write_archive(tmp_path / "archive", color=(10, 20, 30))
    metadata_path = archive_dir / "metadata.jsonl"
    rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["reward_metadata"]["expected_people"] = 5
    metadata_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    archive = AnimaGenerationArchive.load(archive_dir)

    with pytest.raises(ValueError, match=r"conflicting prompt_metadata\.expected_people"):
        checkpoint_eval._bind_exact_count_arm(archive, label="base")
