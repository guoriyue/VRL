"""Video2World row builders: written media, row shape, the text-only view and the
Cosmos consumer's view."""

from __future__ import annotations

import os
from pathlib import Path

import imageio.v2 as imageio
import pytest
from PIL import Image

from vrl.scripts.data import video_world
from vrl.trainers.data.prompts import load_prompt_dataset_index
from vrl.utils.json_files import write_jsonl
from vrl.utils.paths import RootedPaths


def test_video_world_bridge_rows_match_cosmos_consumer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Bridge episodes become video-world rows with a data-root-relative reference image,
    ``video2world`` task type and first-frame conditioning, and load back through the Cosmos
    consumer's manifest path with the reference resolved.
    """
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
    reference_dir = tmp_path / "video_world" / "references"

    rows = video_world.build_video_world_rows(
        episodes,
        reference_dir=reference_dir,
        data_root=tmp_path,
        source="bridge",
    )

    assert len(rows) == 2
    for row in rows:
        assert not os.path.isabs(row["reference_image"])
        assert (tmp_path / row["reference_image"]).exists()
        assert row["task_type"] == "video2world"
        assert row["metadata"]["source"] == "bridge"
        assert row["metadata"]["conditioning"] == "first_frame"

    manifest = tmp_path / "video_world" / "manifests" / "bridge_train.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(manifest, rows)

    monkeypatch.setenv("VRL_DATA_ROOT", str(tmp_path))
    examples = [
        example.with_resolved_references(RootedPaths.data_root(allow_absolute=True))
        for example in load_prompt_dataset_index(manifest)
    ]
    examples = [
        example.with_required_reference_images(manifest=manifest, row_index=index)
        for index, example in enumerate(examples)
    ]
    assert examples[0].metadata["source_episode"] == "000001"
    assert Path(examples[0].reference_images[0]).exists()


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

    rows = video_world.build_target_video_world_rows(
        episodes,
        reference_dir=reference_dir,
        target_dir=target_dir,
        data_root=data_root,
        source="droid",
        fps=10.0,
    )

    assert rows[0]["reference_image"].startswith("video_world/references/")
    assert rows[0]["target_video"].startswith("video_world/targets/")
    assert (data_root / rows[0]["reference_image"]).exists()
    # The real encoder wrote both frames at the per-episode source_fps=15.0,
    # which beats the CLI fps=10.0: read it back from the mp4 itself.
    reader = imageio.get_reader(data_root / rows[0]["target_video"])
    assert reader.count_frames() == 2
    assert round(reader.get_meta_data()["fps"]) == 15
    assert rows[0]["task_type"] == "video2world"
    assert rows[0]["metadata"]["source"] == "droid"
    assert rows[0]["metadata"]["source_repo"] == "lerobot/droid_100"


@pytest.mark.parametrize("source_fps", [0, float("nan")])
def test_video_world_rejects_invalid_source_fps_before_media_write(tmp_path, source_fps):
    references = tmp_path / "references"
    targets = tmp_path / "targets"

    with pytest.raises(ValueError, match="FPS must be finite and > 0"):
        video_world.build_target_video_world_rows(
            [
                {
                    "prompt": "move",
                    "episode_id": "1",
                    "frames": [Image.new("RGB", (2, 2))],
                    "metadata": {"source_fps": source_fps},
                }
            ],
            reference_dir=references,
            target_dir=targets,
            data_root=tmp_path,
            source="unit",
            fps=24.0,
        )
    assert list(references.iterdir()) == []
    assert list(targets.iterdir()) == []


@pytest.mark.parametrize("target_video", [False, True])
def test_video_world_rejects_metadata_before_media_write(tmp_path, target_video):
    image = Image.new("RGB", (2, 2))
    episodes = [
        {
            "prompt": "move",
            "episode_id": "1",
            "image": image,
            "frames": [image],
            # Falsy but not a mapping: must not be coerced to {}.
            "metadata": False,
        }
    ]
    kwargs = dict(reference_dir=tmp_path / "references", data_root=tmp_path, source="unit")
    with pytest.raises(TypeError, match="metadata must be a mapping"):
        if target_video:
            video_world.build_target_video_world_rows(
                episodes,
                **kwargs,
                target_dir=tmp_path / "targets",
                fps=24,
            )
        else:
            video_world.build_video_world_rows(episodes, **kwargs)
    assert not list(tmp_path.rglob("*.png"))
    assert not list(tmp_path.rglob("*.mp4"))


@pytest.mark.parametrize("missing_field", ["prompt", "episode_id"])
@pytest.mark.parametrize("target_video", [False, True])
def test_video_world_skips_null_identity_and_preserves_zero_id(
    tmp_path, missing_field, target_video
):
    image = Image.new("RGB", (2, 2))
    complete = {"prompt": "move", "episode_id": 0, "image": image, "frames": [image]}
    episodes = [{**complete, missing_field: None}, complete]
    kwargs = dict(reference_dir=tmp_path / "references", data_root=tmp_path, source="unit")
    if target_video:
        rows = video_world.build_target_video_world_rows(
            episodes,
            **kwargs,
            target_dir=tmp_path / "targets",
            fps=24,
        )
    else:
        rows = video_world.build_video_world_rows(episodes, **kwargs)
    assert len(rows) == 1
    assert rows[0]["prompt"] == "move"
    assert rows[0]["metadata"]["source_episode"] == "0"
    assert len(list(tmp_path.rglob("*.png"))) == 1
    assert not list(tmp_path.rglob("*None*"))


def test_derive_text_video_rows_removes_unconsumed_first_frame() -> None:
    rows = video_world.derive_text_video_rows(
        [
            {
                "prompt": "Put the blue block in the bowl",
                "reference_image": "video_world/references/episode.png",
                "target_video": "video_world/targets/episode.mp4",
                "task_type": "video2world",
                "metadata": {
                    "source": "droid",
                    "source_episode": "000001",
                    "conditioning": "first_frame",
                },
            },
        ],
    )

    assert rows == [
        {
            "prompt": "Put the blue block in the bowl",
            "target_video": "video_world/targets/episode.mp4",
            "task_type": "text_to_video",
            "metadata": {
                "source": "droid",
                "source_episode": "000001",
                "source_conditioning": "first_frame",
                "conditioning": "text_only",
                "derived_from_task_type": "video2world",
            },
        },
    ]


def test_full_targets_build_writes_the_text_only_view_its_preset_reads(
    tmp_path, monkeypatch
) -> None:
    """``droid_full_targets`` splits the target rows and writes the ``_t2v`` view beside them,
    at the paths ``dataset/droid_full_target_t2v`` reads (relative to the data root).
    """
    from vrl.config.loading import load_config

    frames = [Image.new("RGB", (4, 4), (10, 20, 30))] * 2

    def clips(repo_id, *, camera, limit):
        for index in range(limit):
            yield {
                "frames": frames,
                "prompt": f"move object {index}",
                "episode_id": f"{index:06d}",
                "metadata": {"source_repo": repo_id, "source_camera": camera, "source_fps": 15.0},
            }

    monkeypatch.setattr(video_world, "iter_lerobot_target_clips", clips)
    data_root = tmp_path / "data" / "external"
    video_world.main(["droid_full_targets", "--data-root", str(data_root), "--limit", "3"])

    data = load_config("dataset/droid_full_target_t2v").data
    under_root = {
        role: data_root / str(data[role]).removeprefix("data/external/")
        for role in ("manifest", "eval_manifest", "source_report")
    }
    train = load_prompt_dataset_index(under_root["manifest"])
    evals = load_prompt_dataset_index(under_root["eval_manifest"])
    assert under_root["source_report"].exists()
    assert [len(train), len(evals)] == [1, 2]
    assert train[0].reference_images == []
    assert train[0].task_type == "text_to_video"
    assert train[0].metadata["conditioning"] == "text_only"
    assert (data_root / train[0].target_video).exists()
