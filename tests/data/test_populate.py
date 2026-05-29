from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image

from vrl.scripts.data import bootstrap, common, danbooru, populate, video_world
from vrl.scripts.data.danbooru import positive_image_rows
from vrl.scripts.diffusion.cosmos.train import _normalize_per_sample_reference_images
from vrl.trainers.data import load_prompt_manifest


def test_video_world_bridge_rows_match_cosmos_consumer(
    monkeypatch,
    tmp_path: Path,
) -> None:
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
    examples = load_prompt_manifest(manifest)
    _normalize_per_sample_reference_images(
        examples,
        manifest_path=manifest,
        rollout_batch_size=1,
    )
    assert examples[0].metadata["source_episode"] == "000001"
    assert Path(examples[0].reference_image).exists()


def test_anime_fetch_images_downloads_only_positive_selection(tmp_path: Path) -> None:
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

    selected = positive_image_rows(
        metadata,
        image_root=image_root,
        min_score=10,
        limit=None,
        source="danbooru",
    )
    targets = {str(row["post_id"]): Path(row["image_path"]) for row in selected}

    calls: list[tuple[str, Path]] = []

    def fake_fetch(url: str, target: Path) -> None:
        calls.append((url, target))
        target.write_bytes(b"fake-image-bytes")

    downloaded, skipped, failed = danbooru.download_danbooru_images(
        metadata,
        targets,
        fetch=fake_fetch,
    )

    assert downloaded == 1
    assert skipped == 0
    assert failed == 0
    assert calls == [("https://example.test/1.jpg", image_root / "1.jpg")]
    assert (image_root / "1.jpg").exists()
    assert not (image_root / "2.jpg").exists()


def test_anime_positives_prepares_both_manifests_end_to_end(monkeypatch, tmp_path: Path) -> None:
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

    populate.main(
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
    plan = bootstrap.resolve_experiment_dataset_plan(
        {"loader": "pickapic_preference"},
        repo_root=tmp_path,
    )

    assert plan["ready"] is False
    assert any("pickapic --with-images" in step["get"] for step in plan["steps"])


def test_for_experiment_plan_flags_missing_manifest_with_command(tmp_path: Path) -> None:
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
    populate.main(["for-experiment", "diffusion/wan_2_1/online_grpo_video_reward"])
    out = json.loads(capsys.readouterr().out)

    assert out["experiment"] == "diffusion/wan_2_1/online_grpo_video_reward"
    assert out["loader"] == "prompt_manifest"
    assert out["ready"] is True
    assert any(step["path"] == "datasets/videophy/train.txt" for step in out["steps"])
