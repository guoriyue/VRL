"""Tests for the pure scoring pieces of the Anima tag-adherence eval."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from vrl.scripts.eval import anima_tag_adherence_eval as eval_script


def _row(index: int, tags: tuple[str, ...], *, tier: str = "explicit", source_row: int | None):
    return eval_script.EvalRow(
        index=index,
        prompt=f"rating:{tier}, adult, 1girl, " + ", ".join(tags),
        tier=tier,
        tags=tags,
        source_row=source_row,
    )


def test_summarize_recall_perfect_and_drop_counts() -> None:
    """Checks recall uses an inclusive threshold and drop counts only cover common tags."""
    rows = [
        _row(0, ("braid",), source_row=0),
        _row(1, ("bow", "lingerie", "panties"), tier="questionable", source_row=1),
        _row(2, ("lingerie", "smile"), source_row=5),
    ]
    tag_probs = [
        {"braid": 0.35},  # exactly at threshold counts as detected
        {"bow": 0.9, "lingerie": 0.1, "panties": 0.5},
        {"lingerie": 0.2, "smile": 0.8},
    ]

    report = eval_script.summarize(
        rows,
        tag_probs,
        p_nsfw=[0.9, 0.2, 0.5],
        sharpness=[10.0, 20.0, 30.0],
        tag_threshold=0.35,
        nsfw_threshold=0.35,
    )

    recalls = [r["recall"] for r in report["per_row"]]
    assert recalls == pytest.approx([1.0, 2.0 / 3.0, 0.5])
    assert report["tag_recall"]["mean"] == pytest.approx(sum(recalls) / 3)
    assert report["tag_recall"]["perfect_frac"] == pytest.approx(1.0 / 3.0)
    assert report["per_row"][1]["missed_tags"] == ["lingerie"]
    # lingerie asked twice only: below the >= 5 floor, so no drop entry.
    assert report["tag_recall"]["dropped"] == {}
    assert report["compliance"]["explicit"] == {"triggered": 2, "count": 2, "rate": 1.0}
    assert report["compliance"]["questionable"] == {"triggered": 0, "count": 1, "rate": 0.0}
    assert report["compliance"]["overall"]["rate"] == pytest.approx(2.0 / 3.0)
    assert report["sharpness_x1e-3"]["median"] == pytest.approx(20.0)


def test_summarize_drop_counts_sorted_by_missed() -> None:
    """Checks tags asked >= 5 times are listed, most-missed first."""
    rows = [_row(i, ("lingerie", "ribbon"), source_row=i) for i in range(5)]
    tag_probs = [{"lingerie": 0.0, "ribbon": 0.0}] * 3 + [{"lingerie": 0.0, "ribbon": 1.0}] * 2

    report = eval_script.summarize(
        rows, tag_probs, [0.5] * 5, [1.0] * 5, tag_threshold=0.35, nsfw_threshold=0.35
    )

    assert list(report["tag_recall"]["dropped"].items()) == [
        ("lingerie", {"missed": 5, "asked": 5}),
        ("ribbon", {"missed": 3, "asked": 5}),
    ]


@pytest.mark.parametrize(
    ("prompt", "tier"),
    [
        ("rating:explicit, adult, 1girl, braid", "explicit"),
        ("adult, rating:questionable, lingerie", "questionable"),
        ("Rating:Explicit, 1girl", "explicit"),
    ],
)
def test_parse_tier(prompt: str, tier: str) -> None:
    assert eval_script.parse_tier(prompt) == tier


def test_parse_tier_requires_rating_token() -> None:
    with pytest.raises(ValueError, match="rating"):
        eval_script.parse_tier("adult, 1girl, braid")


def _touch_images(directory, indices) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index in indices:
        Image.new("RGB", (4, 4)).save(directory / f"eval_{index:04d}.png")


def test_resolve_image_paths_uses_source_row_for_unfiltered_base_set(tmp_path) -> None:
    """A directory larger than the manifest is the base set: rows map via source_row."""
    rows = [_row(0, ("a",), source_row=1), _row(1, ("a",), source_row=7)]
    _touch_images(tmp_path, range(10))

    paths = eval_script.resolve_image_paths(tmp_path, rows, manifest_row_count=2)

    assert [p.name for p in paths] == ["eval_0001.png", "eval_0007.png"]


def test_resolve_image_paths_reads_in_order_when_sizes_match(tmp_path) -> None:
    """One image per manifest row means this script's own layout, even under --limit."""
    rows = [_row(0, ("a",), source_row=3)]
    _touch_images(tmp_path, range(2))

    paths = eval_script.resolve_image_paths(tmp_path, rows, manifest_row_count=2)

    assert [p.name for p in paths] == ["eval_0000.png"]


def test_resolve_image_paths_reports_missing_images(tmp_path) -> None:
    rows = [_row(0, ("a",), source_row=9)]
    _touch_images(tmp_path, range(3))

    with pytest.raises(FileNotFoundError, match=r"eval_0009\.png"):
        eval_script.resolve_image_paths(tmp_path, rows, manifest_row_count=1)


def test_laplacian_sharpness_flat_versus_checkerboard() -> None:
    flat = Image.new("L", (32, 32), color=128)
    board = np.indices((32, 32)).sum(axis=0) % 2 * 255
    checker = Image.fromarray(board.astype(np.uint8), mode="L")

    assert eval_script.laplacian_sharpness(flat) == pytest.approx(0.0)
    # Every interior pixel's Laplacian is +-4, so the variance is 16 -> 16000 in 1e-3 units.
    assert eval_script.laplacian_sharpness(checker) == pytest.approx(16000.0)


def test_format_summary_places_baseline_alongside() -> None:
    report = {
        "label": "base",
        "row_count": 1,
        "lora_path": "",
        "note": "n",
        "tag_recall": {"mean": 0.5, "perfect_frac": 0.0, "dropped": {}},
        "compliance": {"explicit": {"triggered": 1, "count": 1, "rate": 1.0}},
        "sharpness_x1e-3": {"mean": 1.0, "median": 1.0, "p10": 1.0, "n": 1},
    }
    baseline = {"tag_recall_whitelist": {"mean": 0.913}}

    text = eval_script.format_summary(report, baseline)

    assert "0.913" in text
    assert "1/1 (100.0%)" in text
