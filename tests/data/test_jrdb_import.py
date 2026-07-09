"""CPU tests for the JRDB sidewalk-delivery importer (no downloads).

The official archive sits behind a registration wall, so tests synthesize the
extracted layout (images/<camera>/<sequence>/<frame>.jpg) with tiny PIL frames
and exercise the parser, the row transform, and the full command path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from vrl.scripts.data import jrdb, video_world
from vrl.scripts.data import setup as setup_cli
from vrl.trainers.data.artifacts import SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS


def _make_layout(root: Path, sequences: dict[str, int]) -> None:
    for sequence, frame_count in sequences.items():
        seq_dir = root / "images" / "image_stitched" / sequence
        seq_dir.mkdir(parents=True)
        for index in range(frame_count):
            Image.new("RGB", (8, 6), color=(index * 20 % 255, 40, 90)).save(
                seq_dir / f"{index:06d}.jpg",
            )


def test_iter_jrdb_clips_parses_layout_and_templates_prompts(tmp_path: Path) -> None:
    _make_layout(tmp_path, {"bytes-cafe-2019-02-07_0": 8, "clark-center-2019-02-28_1": 5})

    clips = list(
        jrdb.iter_jrdb_clips(
            tmp_path,
            camera="image_stitched",
            split="train",
            clip_frames=4,
            clip_stride=4,
            clips_per_sequence=2,
            limit=10,
            fps=15.0,
        )
    )

    # 8-frame sequence yields starts 0 and 4; 5-frame sequence only start 0.
    assert [clip["episode_id"] for clip in clips] == [
        "bytes-cafe-2019-02-07_0_000000",
        "bytes-cafe-2019-02-07_0_000004",
        "clark-center-2019-02-28_1_000000",
    ]
    first = clips[0]
    assert first["prompt"] == (
        "First-person view of a mobile robot driving through bytes cafe among pedestrians"
    )
    assert len(first["frames"]) == 4
    metadata = first["metadata"]
    assert metadata["source_repo"] == jrdb.SOURCE_REPO
    assert metadata["source_video"] == "images/image_stitched/bytes-cafe-2019-02-07_0"
    assert metadata["source_frame_index"] == 0
    assert metadata["decode_method"] == "local_jpg_sequence"
    assert metadata["prompt_source"] == "jrdb_sequence_name_template"
    assert clips[1]["metadata"]["source_frame_index"] == 4


def test_iter_jrdb_clips_missing_camera_dir_names_the_download_page(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"jrdb\.erc\.monash\.edu"):
        list(
            jrdb.iter_jrdb_clips(
                tmp_path,
                camera="image_stitched",
                split="train",
                clip_frames=4,
                clip_stride=4,
                clips_per_sequence=1,
                limit=1,
                fps=15.0,
            )
        )


def test_iter_jrdb_clips_rejects_overlapping_stride(tmp_path: Path) -> None:
    _make_layout(tmp_path, {"bytes-cafe-2019-02-07_0": 8})
    with pytest.raises(ValueError, match="clip-stride"):
        list(
            jrdb.iter_jrdb_clips(
                tmp_path,
                camera="image_stitched",
                split="train",
                clip_frames=4,
                clip_stride=2,
                clips_per_sequence=1,
                limit=1,
                fps=15.0,
            )
        )


def test_jrdb_targets_command_writes_manifests_report_and_validates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jrdb_root = tmp_path / "jrdb"
    _make_layout(jrdb_root, {"bytes-cafe-2019-02-07_0": 8, "clark-center-2019-02-28_1": 8})
    data_root = tmp_path / "data_root"

    def fake_video_writer(path: Path, frames, fps: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fake-video frames={len(frames)} fps={fps}")

    monkeypatch.setattr(video_world, "_write_mp4", fake_video_writer)

    setup_cli.main(
        [
            "jrdb-targets",
            "--jrdb-root",
            str(jrdb_root),
            "--data-root",
            str(data_root),
            "--clip-frames",
            "4",
            "--clip-stride",
            "4",
            "--clips-per-sequence",
            "2",
            "--limit",
            "10",
            "--eval-limit",
            "1",
        ]
    )

    manifests = data_root / "video_world" / "manifests"
    train_rows = [
        json.loads(line)
        for line in (manifests / "jrdb_targets_train.jsonl").read_text().splitlines()
    ]
    eval_rows = [
        json.loads(line)
        for line in (manifests / "jrdb_targets_eval.jsonl").read_text().splitlines()
    ]
    assert len(train_rows) == 3
    assert len(eval_rows) == 1

    row = train_rows[0]
    assert row["task_type"] == "video2world"
    assert (data_root / row["reference_image"]).exists()
    assert (data_root / row["target_video"]).exists()
    for field in SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS:
        assert row["metadata"].get(field) not in (None, ""), field

    report = json.loads((data_root / "video_world" / "jrdb_targets_report.json").read_text())
    assert report["repo_id"] == jrdb.SOURCE_REPO
    assert report["train_rows"] == 3
    assert report["eval_rows"] == 1
    assert report["validation_summary"]["row_count"] == len(train_rows)
