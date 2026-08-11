from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from PIL import Image

from vrl.scripts.data import bootstrap, common, danbooru, setup, video_world
from vrl.trainers.data import load_prompt_examples_from_config, load_prompt_manifest
from vrl.trainers.data.artifacts import (
    require_reference_images,
    resolve_prompt_example_references,
)


def test_runtime_data_loader_derives_plain_prompt_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "prompts.txt"
    manifest.write_text("a red fox\n", encoding="utf-8")

    examples = load_prompt_examples_from_config(
        OmegaConf.create(
            {
                "manifest": str(manifest),
                "preprocessing": {"format": "text"},
            },
        ),
    )

    assert [example.prompt for example in examples] == ["a red fox"]
    assert examples[0].reference_image is None


def test_runtime_data_loader_derives_image_prompt_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "prompts.jsonl"
    manifest.write_text(
        json.dumps({"image": "reference.png", "caption": "a red fox"}) + "\n",
        encoding="utf-8",
    )

    examples = load_prompt_examples_from_config(
        OmegaConf.create(
            {
                "manifest": str(manifest),
                "preprocessing": {
                    "format": "image_caption_jsonl",
                    "image_field": "image",
                    "caption_field": "caption",
                },
            },
        ),
    )

    assert [example.prompt for example in examples] == ["a red fox"]
    assert examples[0].reference_image == "reference.png"


def test_runtime_data_loader_rejects_explicit_format_conflict() -> None:
    data = OmegaConf.create(
        {
            "loader": "prompt_manifest",
            "manifest": "unused.jsonl",
            "preprocessing": {"format": "image_caption_jsonl"},
        },
    )

    with pytest.raises(ValueError, match=r"requires.*prompt_image_manifest"):
        load_prompt_examples_from_config(data)


def test_video_world_bridge_rows_match_cosmos_consumer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Checks video world bridge rows match Cosmos consumer."""
    episodes = [
        {
            "image": Image.new("RGB", (4, 4), (10, 20, 30)),
            "prompt": "the robot arm reaches toward the cup",
            "episode_id": "000001",
        },
        {
            "image": Image.new("RGB", (4, 4), (40, 50, 60)),
            "prompt": "the gripper slides the block forward",
            "episode_id": "000002",
        },
    ]
    data_root = tmp_path
    reference_dir = data_root / "video_world" / "references"

    rows = video_world.build_video_world_rows(
        episodes,
        reference_dir=reference_dir,
        data_root=data_root,
        source="bridge",
    )

    assert len(rows) == 2
    for row in rows:
        assert not os.path.isabs(row["reference_image"])
        assert (data_root / row["reference_image"]).exists()
        assert row["task_type"] == "video2world"
        assert row["metadata"]["source"] == "bridge"
        assert row["metadata"]["conditioning"] == "first_frame"

    manifest = data_root / "video_world" / "manifests" / "bridge_train.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    common.write_jsonl(manifest, rows)

    monkeypatch.setenv("VRL_DATA_ROOT", str(data_root))
    examples = [
        resolve_prompt_example_references(example, allow_absolute=True)
        for example in load_prompt_manifest(manifest)
    ]
    require_reference_images(
        examples,
        manifest_path=manifest,
    )
    assert examples[0].metadata["source_episode"] == "000001"
    assert Path(examples[0].reference_image).exists()


@pytest.mark.real_cover(
    None,
    why=(
        "imageio + imageio-ffmpeg are declared dependencies, so real mp4 encoding would "
        "work here; it is skipped because this test asserts clip splitting, row paths and "
        "the per-episode fps override, and never decodes the written video — real encoding "
        "would add seconds and zero coverage"
    ),
    tracked_in="docs/sprints/done/SPRINT_tier-policy-and-real-cover-labels.md",
)
def test_video_world_targets_rows_include_real_source_target_clip(tmp_path: Path) -> None:
    """Target rows carry reference + target artifacts, and per-episode fps wins over CLI fps."""
    data_root = tmp_path / "external"
    reference_dir = data_root / "video_world" / "references"
    target_dir = data_root / "video_world" / "targets"
    episodes = [
        {
            "frames": [
                Image.new("RGB", (4, 4), (10, 20, 30)),
                Image.new("RGB", (4, 4), (40, 50, 60)),
            ],
            "prompt": "put the marker in the pot",
            "episode_id": "000001",
            "metadata": {
                "source_repo": "lerobot/droid_100",
                "source_split": "main",
                "source_video": "videos/observation.images.exterior/batch-000/file-000.mp4",
                "source_frame_index": 0,
                "decode_method": "pyav_http_target_clip",
                "source_fps": 15.0,
            },
        },
    ]

    # Record the encoder call instead of writing a formatted string and reading it
    # back: the invariant under test is that the per-episode source_fps=15.0 beats
    # the CLI fps=10.0, and an argument list states that directly.
    writes: list[tuple[Path, int, float]] = []

    def record_video_write(path: Path, frames: list[Image.Image], fps: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        writes.append((path, len(frames), fps))

    rows = video_world.build_target_video_world_rows(
        episodes,
        reference_dir=reference_dir,
        target_dir=target_dir,
        data_root=data_root,
        source="droid",
        fps=10.0,
        video_writer=record_video_write,
    )

    assert rows[0]["reference_image"].startswith("video_world/references/")
    assert rows[0]["target_video"].startswith("video_world/targets/")
    assert (data_root / rows[0]["reference_image"]).exists()
    assert (data_root / rows[0]["target_video"]).exists()
    assert writes == [(data_root / rows[0]["target_video"], 2, 15.0)]
    assert rows[0]["task_type"] == "video2world"
    assert rows[0]["metadata"]["source"] == "droid"
    assert rows[0]["metadata"]["source_repo"] == "lerobot/droid_100"


# NOTE: the bare ``download_danbooru_images`` selection path is owned by
# ``test_danbooru.py::test_download_danbooru_images_downloads_only_positive_selection``.
# This module only covers the ``setup.main`` CLI wiring that sits on top of it.
@pytest.mark.real_cover(
    None,
    why=(
        "the patched _http_download stands in for an HTTP GET against danbooru.donmai.us; a "
        "test that reaches the live site is neither reproducible nor free, and what is "
        "asserted here is the setup.main CLI wiring above it"
    ),
    tracked_in="docs/sprints/done/SPRINT_tier-policy-and-real-cover-labels.md",
)
def test_anime_positives_prepares_both_manifests_end_to_end(monkeypatch, tmp_path: Path) -> None:
    """The CLI writes both the positives and hand-crop manifests in one pass."""
    metadata = tmp_path / "posts.jsonl"
    rows = [
        {
            "id": 1,
            "score": 50,
            "tag_string": "1girl solo full_body standing",
            "file_ext": "jpg",
            "file_url": "https://example.test/1.jpg",
        },
        {
            "id": 2,
            "score": 1,
            "tag_string": "1girl solo upper_body",
            "file_ext": "jpg",
            "file_url": "https://example.test/2.jpg",
        },
    ]
    metadata.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    image_root = tmp_path / "images"
    positives = tmp_path / "positive_images.jsonl"
    hand_crops = tmp_path / "hand_crops.jsonl"

    def fake_fetch(url: str, target: Path) -> None:
        target.write_bytes(b"fake-image-bytes")

    monkeypatch.setattr(danbooru, "_http_download", fake_fetch)

    setup.main(
        [
            "anime-positives",
            "--metadata",
            str(metadata),
            "--image-root",
            str(image_root),
            "--output",
            str(positives),
            "--hand-crops-output",
            str(hand_crops),
            "--fetch-images",
        ],
    )

    pos_rows = [json.loads(line) for line in positives.read_text().splitlines() if line.strip()]
    assert len(pos_rows) == 1
    assert pos_rows[0]["post_id"] == 1
    assert (image_root / "1.jpg").exists()
    assert not (image_root / "2.jpg").exists()

    crop_rows = [json.loads(line) for line in hand_crops.read_text().splitlines() if line.strip()]
    assert len(crop_rows) == 1
    assert crop_rows[0]["labels"] == ["hand_ok"]


def test_for_experiment_plan_marks_committed_manifest_ready(tmp_path: Path) -> None:
    """Checks for experiment plan marks committed manifest ready."""
    manifest_dir = tmp_path / "datasets" / "pickscore_sfw"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "train.txt").write_text("a\nb\nc\n", encoding="utf-8")

    plan = bootstrap.resolve_experiment_dataset_plan(
        {"loader": "prompt_manifest", "manifest": "datasets/pickscore_sfw/train.txt"},
        repo_root=tmp_path,
    )

    assert plan["ready"] is True
    assert plan["steps"][0]["present"] is True
    assert plan["steps"][0]["rows"] == 3
    assert plan["steps"][0]["get"] == ""


def test_for_experiment_plan_flags_pickapic_download(tmp_path: Path) -> None:
    """Checks for experiment plan flags Pick-a-Pic download."""
    plan = bootstrap.resolve_experiment_dataset_plan(
        {"loader": "pickapic_preference"},
        repo_root=tmp_path,
    )

    assert plan["ready"] is False
    assert any("pickapic --with-images" in step["get"] for step in plan["steps"])


def test_for_experiment_plan_flags_missing_manifest_with_command(tmp_path: Path) -> None:
    """Checks for experiment plan flags missing manifest with command."""
    plan = bootstrap.resolve_experiment_dataset_plan(
        {
            "loader": "prompt_manifest",
            "manifest": "datasets/danbooru/anatomy/train_prompts.jsonl",
        },
        repo_root=tmp_path,
    )

    assert plan["ready"] is False
    assert "anime-prompts" in plan["steps"][0]["get"]


def test_for_experiment_resolves_real_wan_experiment(capsys) -> None:
    """Checks for experiment resolves real Wan experiment."""
    from vrl.config.loading import load_config

    experiment = "wan_2_1/online_grpo_kling_video_reward"
    setup.main(["for-experiment", experiment])
    out = json.loads(capsys.readouterr().out)

    # Derive expected loader/manifest from the same config the resolver loads, so the
    # test tracks the dataset group instead of re-typing its YAML strings.
    data = load_config(f"experiment/{experiment}").data
    assert out["experiment"] == experiment
    assert out["loader"] == data.loader
    # Resolver behavior contract (the real point of the test), not a config literal:
    assert out["ready"] is True
    assert all(step["present"] and step["complete"] and step["get"] == "" for step in out["steps"])
    assert any(step["path"] == data.manifest for step in out["steps"])
