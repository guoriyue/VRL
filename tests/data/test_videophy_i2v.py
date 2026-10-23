from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path
from urllib.error import HTTPError

import imageio.v2 as imageio
import numpy as np
import pytest
from PIL import Image

from vrl.scripts.data.videophy_i2v import (
    DECODE_METHOD,
    VideoPhySource,
    VideoPhyVideoRow,
    prepare_videophy_i2v_dataset,
    select_videos_for_prompts,
)


def _mp4_bytes(tmp_path: Path, *, width: int, height: int) -> bytes:
    """A real two-frame H.264 clip, so the production ffmpeg decode reads it."""

    path = tmp_path / f"source-{width}x{height}.mp4"
    frame = np.full((height, width, 3), (10, 20, 30), dtype=np.uint8)
    imageio.mimsave(path, [frame, frame], fps=8, macro_block_size=1)
    return path.read_bytes()


@pytest.mark.parametrize("status", [403, 503])
def test_unavailable_candidate_fallback_keeps_source_and_cache_identity(
    tmp_path: Path, monkeypatch, status: int
) -> None:
    csv_path = tmp_path / "videos.csv"
    csv_path.write_text(
        "video_url,caption,source,complexity,majority_sa,majority_pc\n"
        "https://example.test/best.mp4,A wheel rolls.,best,0,1,1\n"
        "https://example.test/next.mp4,A wheel rolls.,next,0,1,0\n"
    )
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("A wheel rolls.\n")
    video = _mp4_bytes(tmp_path, width=8, height=6)
    fetched = []
    best_available = False

    # The network is the one boundary: every other step (download into the
    # cache path, ffmpeg first-frame decode, resize) runs for real.
    def urlopen(url: str, timeout: float) -> io.BytesIO:
        fetched.append(url)
        if url.endswith("best.mp4") and not best_available:
            raise HTTPError(url, status, "unavailable", None, None)
        return io.BytesIO(video)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    def prepare() -> dict:
        return prepare_videophy_i2v_dataset(
            sources=[VideoPhySource("official/repo", "videos.csv", csv_path, "test")],
            train_prompts=prompts,
            eval_prompts=prompts,
            data_root=tmp_path / "data",
            width=4,
            height=4,
        )

    if status == 503:
        with pytest.raises(HTTPError):
            prepare()
        assert fetched == ["https://example.test/best.mp4"]
        return

    report = prepare()
    manifest = Path(report["train_manifest"])
    original = json.loads(manifest.read_text())
    assert original["metadata"]["source_video_url"].endswith("next.mp4")
    assert original["metadata"]["source_csv_row"] == 1
    assert original["metadata"]["unavailable_candidates"][0]["status"] == status
    prepare()
    assert json.loads(manifest.read_text())["metadata"]["source_frame_size"] == {
        "width": 8,
        "height": 6,
    }
    assert fetched.count("https://example.test/next.mp4") == 2  # one per split

    best_available = True
    prepare()
    recovered = json.loads(manifest.read_text())
    assert recovered["image"] != original["image"]
    assert recovered["metadata"]["source_video_url"].endswith("best.mp4")
    assert recovered["metadata"]["source_csv_row"] == 0
    assert "unavailable_candidates" not in recovered["metadata"]


def test_select_videophy_videos_prefers_aligned_physical_candidate() -> None:
    """Among candidates for one caption the selection prefers the one with both semantic alignment
    and physical commonsense (sa=1, pc=1), matching captions after whitespace trimming.
    """
    candidates = [
        VideoPhyVideoRow(
            caption="A wheel steadily rolls across a cement path.",
            video_url="https://example.test/weak.mp4",
            row_index=0,
            majority_sa=1,
            majority_pc=0,
        ),
        VideoPhyVideoRow(
            caption="A wheel steadily rolls across a cement path.",
            video_url="https://example.test/best.mp4",
            row_index=1,
            majority_sa=1,
            majority_pc=1,
        ),
        VideoPhyVideoRow(
            caption="A wheel steadily rolls across a cement path.",
            video_url="https://example.test/misaligned.mp4",
            row_index=2,
            majority_sa=0,
            majority_pc=1,
        ),
    ]

    selected = select_videos_for_prompts(
        [" A wheel steadily rolls across a cement path. "],
        candidates,
        split="train",
    )

    assert len(selected) == 1
    assert selected[0].source_row.video_url == "https://example.test/best.mp4"


def test_prepare_videophy_i2v_dataset_writes_source_backed_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    """Preparation fetches each split's video, extracts its first frame resized to 832x480, writes
    source-backed manifest rows (URL, decode method, frame sizes) plus a report, and does not
    keep the videos.
    """
    csv_path = tmp_path / "videophy2_test.csv"
    csv_path.write_text(
        "\n".join(
            [
                "video_url,caption,states_of_matter,source,complexity,majority_sa,majority_pc",
                "https://example.test/train.mp4,A wheel rolls.,solid_solid,lavie,0,1,1",
                "https://example.test/eval.mp4,Honey diffuses.,fluid_fluid,lavie,1,1,1",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    train_prompts = tmp_path / "train.txt"
    eval_prompts = tmp_path / "eval.txt"
    train_prompts.write_text("A wheel rolls.\n", encoding="utf-8")
    eval_prompts.write_text("Honey diffuses.\n", encoding="utf-8")
    data_root = tmp_path / "external"

    video = _mp4_bytes(tmp_path, width=4, height=4)
    fetched: list[str] = []

    def urlopen(url: str, timeout: float) -> io.BytesIO:
        fetched.append(url)
        return io.BytesIO(video)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    report = prepare_videophy_i2v_dataset(
        sources=[
            VideoPhySource("videophysics/videophy2_test", "videophy2_test.csv", csv_path, "test")
        ],
        train_prompts=train_prompts,
        eval_prompts=eval_prompts,
        data_root=data_root,
    )

    dataset_root = data_root / "videophy_i2v"
    train_manifest = dataset_root / "manifests" / "train.jsonl"
    eval_manifest = dataset_root / "manifests" / "eval.jsonl"
    train_rows = [json.loads(line) for line in train_manifest.read_text().splitlines()]
    eval_rows = [json.loads(line) for line in eval_manifest.read_text().splitlines()]

    assert report["train_rows"] == 1
    assert report["eval_rows"] == 1
    assert fetched == ["https://example.test/train.mp4", "https://example.test/eval.mp4"]
    assert train_rows[0]["image"].startswith("images/train/000-")
    assert train_rows[0]["caption"] == "A wheel rolls."
    assert train_rows[0]["task_type"] == "image_to_video"
    assert train_rows[0]["metadata"]["source_video_url"] == "https://example.test/train.mp4"
    assert train_rows[0]["metadata"]["decode_method"] == DECODE_METHOD
    assert train_rows[0]["metadata"]["source_frame_size"] == {"width": 4, "height": 4}
    assert train_rows[0]["metadata"]["image_size"] == {"width": 832, "height": 480}
    assert eval_rows[0]["image"].startswith("images/eval/000-")
    assert (dataset_root / train_rows[0]["image"]).exists()
    assert Image.open(dataset_root / train_rows[0]["image"]).size == (832, 480)
    assert not list((dataset_root / "videos" / "train").glob("*.mp4"))
    assert (dataset_root / "report.json").exists()
