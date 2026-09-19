from __future__ import annotations

import json
from pathlib import Path

from vrl.scripts.data.danbooru.safety import (
    build_danbooru_safety_prompt_rows,
    build_safety_prompts,
    split_safety_prompt_rows,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_build_safety_prompts_requires_danbooru_metadata(tmp_path: Path) -> None:
    """Safety prompts cannot be built without Danbooru metadata."""
    train_output = tmp_path / "train.jsonl"
    eval_output = tmp_path / "eval_baseline.jsonl"

    try:
        build_safety_prompts(
            train_output=train_output,
            eval_output=eval_output,
        )
    except ValueError as exc:
        assert "Provide metadata" in str(exc)
    else:
        raise AssertionError("expected metadata requirement error")


def test_build_danbooru_safety_prompt_rows_uses_ratings_and_nsfw_tags(
    tmp_path: Path,
) -> None:
    """Safety prompt rows come only from explicit / questionable-rated posts, carry their NSFW
    tags and a ``rating:`` token in the prompt, and drop the dataset-level provenance keys
    (safety_target, domain, template_id).
    """
    metadata = tmp_path / "posts.jsonl"
    _write_jsonl(
        metadata,
        [
            {
                "id": 1,
                "rating": "e",
                "score": 9,
                "tag_string": "1girl solo nude breasts long_hair",
            },
            {
                "id": 2,
                "rating": "q",
                "score": 7,
                "tag_string": "1boy solo underwear standing short_hair",
            },
            {
                "id": 3,
                "rating": "g",
                "score": 20,
                "tag_string": "1girl solo full_body standing",
            },
            {
                "id": 4,
                "rating": "e",
                "score": 20,
                "tag_string": "1girl solo loli nude",
            },
        ],
    )

    rows = build_danbooru_safety_prompt_rows(
        metadata,
        limit=4,
        seed=0,
        candidate_limit=10,
    )
    train_rows, eval_rows = split_safety_prompt_rows(rows, train_limit=1, eval_limit=1)

    assert len(rows) == 2
    assert len(train_rows) == 1
    assert len(eval_rows) == 1
    assert {row["metadata"]["rating"] for row in rows} == {"explicit", "questionable"}
    assert all(row["metadata"]["nsfw_tags"] for row in rows)
    assert all("safety_target" not in row["metadata"] for row in rows)
    assert all("domain" not in row["metadata"] for row in rows)
    assert all("template_id" not in row["metadata"] for row in rows)
    assert all("rating:" in row["prompt"] for row in rows)


def test_build_safety_prompts_from_danbooru_metadata(tmp_path: Path) -> None:
    """``build_safety_prompts`` writes train / eval splits balanced across explicit and
    questionable ratings, plus a report with per-split rating counts and the top NSFW tags.
    """
    metadata = tmp_path / "posts.jsonl"
    hair_tags = [
        "black_hair",
        "brown_hair",
        "blonde_hair",
        "red_hair",
        "blue_hair",
        "pink_hair",
        "white_hair",
        "green_hair",
    ]
    _write_jsonl(
        metadata,
        [
            {
                "id": index,
                "rating": "e" if index % 2 == 0 else "q",
                "score": 5,
                "tag_string": f"1girl solo nude long_hair standing {hair_tags[index - 1]}",
            }
            for index in range(1, 9)
        ],
    )
    train_output = tmp_path / "train.jsonl"
    eval_output = tmp_path / "eval.jsonl"
    report_output = tmp_path / "report.json"

    build_safety_prompts(
        metadata=metadata,
        train_output=train_output,
        eval_output=eval_output,
        report_output=report_output,
        train_limit=4,
        eval_limit=2,
    )

    train_rows = [json.loads(line) for line in train_output.read_text().splitlines()]
    eval_rows = [json.loads(line) for line in eval_output.read_text().splitlines()]
    report = json.loads(report_output.read_text())

    assert len(train_rows) == 4
    assert len(eval_rows) == 2
    assert report["train_ratings"] == {"explicit": 2, "questionable": 2}
    assert report["eval_ratings"] == {"explicit": 1, "questionable": 1}
    assert "dataset_metadata" not in report
    assert "rating:explicit" in report["train_nsfw_tags_top"]
