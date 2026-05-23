from __future__ import annotations

import json
import tarfile
from pathlib import Path

from vrl.scripts.data.anime_anatomy import (
    build_prompt_rows,
    hand_crop_rows,
    hard_negative_rows,
    label_queue_rows,
    positive_image_rows,
    split_prompt_rows,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_build_prompt_rows_tracks_provenance_and_buckets(tmp_path: Path) -> None:
    metadata = tmp_path / "posts.jsonl"
    _write_jsonl(
        metadata,
        [
            {
                "id": 1,
                "score": 10,
                "tag_string_general": (
                    "1girl solo full_body standing boots simple_background hands long_hair"
                ),
            },
            {
                "id": 2,
                "score": 12,
                "tags": ["1boy", "solo", "full_body", "running", "sportswear", "feet"],
            },
            {
                "id": 3,
                "score": 20,
                "tag_string_general": "1girl solo portrait standing hands",
            },
        ],
    )

    rows = build_prompt_rows(metadata, min_score=0, seed=7)
    train_rows, eval_rows = split_prompt_rows(rows, train_limit=1, eval_limit=1)

    assert len(rows) == 2
    assert len(train_rows) == 1
    assert len(eval_rows) == 1
    assert {row["metadata"]["domain"] for row in rows} == {"anime"}
    assert {row["metadata"]["template_id"] for row in rows} == {"anime_anatomy_v1"}
    assert {row["metadata"]["prompt_style"] for row in rows} == {"tag", "language"}
    assert any(row["prompt"].startswith("1girl, solo") for row in rows)
    assert any(row["prompt"].startswith("single anime boy") for row in rows)
    assert any("both hands visible" in row["prompt"] for row in rows)
    assert any("long hair" in row["prompt"] for row in rows)
    assert any(row["metadata"]["bucket"] == "running" for row in rows)


def test_build_prompt_rows_reads_danbooru_json_and_tar_gz(tmp_path: Path) -> None:
    rows = [
        {
            "id": 1,
            "score": 10,
            "tag_string": (
                "1girl solo full_body standing hands boots simple_background long_hair"
            ),
        },
        {
            "id": 2,
            "score": 12,
            "tag_string": "1boy solo full_body walking shoes city_street backpack",
        },
    ]
    metadata_json = tmp_path / "posts.json"
    _write_jsonl(metadata_json, rows)
    assert len(build_prompt_rows(metadata_json, min_score=0, seed=1)) == 2

    archive_path = tmp_path / "posts.tar.gz"
    inner = tmp_path / "posts_inner.json"
    _write_jsonl(inner, rows)
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(inner, arcname="posts.json")

    archive_rows = build_prompt_rows(archive_path, min_score=0, seed=1)

    assert len(archive_rows) == 2
    assert any("backpack" in row["prompt"] for row in archive_rows)


def test_build_prompt_rows_uses_bucket_quotas_with_score_fallback(tmp_path: Path) -> None:
    metadata = tmp_path / "posts.jsonl"
    rows: list[dict] = []
    hair_tags = ["black_hair", "brown_hair", "blonde_hair", "red_hair", "blue_hair", "pink_hair"]
    for index in range(6):
        rows.append(
            {
                "id": 100 + index,
                "score": 30,
                "tag_string": f"1girl solo full_body standing boots long_hair {hair_tags[index]}",
            },
        )
    action_hair_tags = ["black_hair", "brown_hair", "blonde_hair"]
    for index in range(3):
        rows.append(
            {
                "id": 200 + index,
                "score": 6,
                "tag_string": f"1girl solo full_body jumping shoes short_hair {action_hair_tags[index]}",
            },
        )
    walking_hair_tags = ["red_hair", "blue_hair", "pink_hair"]
    for index in range(3):
        rows.append(
            {
                "id": 300 + index,
                "score": 8,
                "tag_string": f"1boy solo full_body walking shoes backpack {walking_hair_tags[index]}",
            },
        )
    _write_jsonl(metadata, rows)

    built = build_prompt_rows(
        metadata,
        min_score=5,
        preferred_min_score=20,
        limit=6,
        seed=0,
        bucket_weights={"feet_visible": 1.0, "action_pose": 1.0, "walking": 1.0},
    )

    buckets = [row["metadata"]["bucket"] for row in built]
    assert buckets.count("feet_visible") == 2
    assert buckets.count("action_pose") == 2
    assert buckets.count("walking") == 2
    assert all(row["metadata"]["source_score"] >= 5 for row in built)


def test_build_prompt_rows_allows_hand_focus_without_full_body(tmp_path: Path) -> None:
    metadata = tmp_path / "posts.jsonl"
    _write_jsonl(
        metadata,
        [
            {
                "id": 400,
                "score": 15,
                "tag_string": "1girl solo upper_body hand_focus hands long_hair",
            },
        ],
    )

    rows = build_prompt_rows(metadata, min_score=5, limit=1, seed=0)

    assert rows[0]["metadata"]["bucket"] == "hand_focus"
    assert rows[0]["metadata"]["framing"] == "upper body"


def test_positive_hand_hard_negative_and_label_queue_rows(tmp_path: Path) -> None:
    metadata = tmp_path / "posts.jsonl"
    image = tmp_path / "image.png"
    image.write_bytes(b"fake")
    _write_jsonl(
        metadata,
        [
            {
                "id": 10,
                "score": 5,
                "image_path": str(image),
                "tags": ["1girl", "solo", "full_body", "standing", "hands", "boots"],
                "hand_boxes": [[1, 2, 3, 4]],
            },
        ],
    )

    positives = positive_image_rows(metadata, source="test")
    assert positives == [
        {
            "image_path": str(image),
            "source": "test",
            "post_id": 10,
            "score": 5.0,
            "tags": ["1girl", "solo", "full_body", "standing", "hands", "boots"],
            "domain": "anime",
        },
    ]

    positive_manifest = tmp_path / "positives.jsonl"
    _write_jsonl(positive_manifest, positives)
    crops = hand_crop_rows([positive_manifest], label="hand_ok", source="test")
    assert crops == []

    generated = tmp_path / "generated.jsonl"
    _write_jsonl(
        generated,
        [
            {
                "image_path": str(image),
                "prompt": "anime woman, full body",
                "labels": ["bad_hands", "missing_feet"],
                "severity": 2,
            },
            {
                "image_path": str(image),
                "prompt": "anime woman, full body",
                "labels": ["ok"],
                "severity": 1,
            },
        ],
    )
    negatives = hard_negative_rows(generated, min_severity=2)
    assert len(negatives) == 1
    assert negatives[0]["labels"] == ["bad_hands", "missing_feet"]
    assert negatives[0]["domain"] == "anime"

    negative_manifest = tmp_path / "negatives.jsonl"
    _write_jsonl(negative_manifest, negatives)
    queue = label_queue_rows(negative_manifest)
    assert len(queue) == 1
    assert queue[0]["image_path"] == str(image)
    assert "Are fingers plausible enough for the image scale?" in queue[0]["questions"]
