from __future__ import annotations

import json
import os
from pathlib import Path

import imageio.v2 as imageio
import pytest
from omegaconf import OmegaConf
from PIL import Image

from vrl.config.schema import DataConfig
from vrl.scripts.data import bootstrap, setup, video_world
from vrl.trainers.data.artifacts import (
    resolve_prompt_example_references,
    resolve_required_reference_images_,
)
from vrl.trainers.data.prompts import load_prompt_dataset_index, load_prompt_examples_from_config
from vrl.utils.json_files import write_jsonl


def _data_config(payload: dict) -> DataConfig:
    """A parsed ``data`` section; the prompt loaders require a sampler type."""

    payload = dict(payload)
    payload.setdefault("sampler", {"type": "random_without_replacement"})
    return DataConfig.model_validate(payload)


def test_runtime_data_loader_derives_plain_prompt_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "prompts.txt"
    manifest.write_text("a red fox\n", encoding="utf-8")

    examples = load_prompt_examples_from_config(
        _data_config(
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
        _data_config(
            {
                "manifest": str(manifest),
                "eval_manifest": str(manifest),
                "preprocessing": {
                    "format": "image_caption_jsonl",
                    "image_field": "image",
                    "caption_field": "caption",
                    "conditioning": "reference_image",
                },
            },
        ),
    )

    assert [example.prompt for example in examples] == ["a red fox"]
    assert examples[0].reference_image == "reference.png"


def _write_prompts(path: Path, prompts: list[str]) -> Path:
    path.write_text(
        "".join(json.dumps({"prompt": prompt}) + "\n" for prompt in prompts),
        encoding="utf-8",
    )
    return path


def test_runtime_data_loader_mixes_manifests_by_declared_counts(tmp_path: Path) -> None:
    """A {path: count} manifest draws exactly that many prompts from each source."""
    anatomy = _write_prompts(tmp_path / "anatomy.jsonl", [f"anatomy {i}" for i in range(50)])
    safety = _write_prompts(tmp_path / "safety.jsonl", [f"safety {i}" for i in range(50)])

    examples = load_prompt_examples_from_config(
        _data_config(
            {
                "loader": "prompt_manifest",
                "manifest": {str(anatomy): 8, str(safety): 2},
                "mix_seed": 20260818,
                "preprocessing": {"format": "jsonl"},
            },
        ),
    )

    prompts = [example.prompt for example in examples]
    assert len(prompts) == 10
    assert sum(prompt.startswith("anatomy") for prompt in prompts) == 8
    assert sum(prompt.startswith("safety") for prompt in prompts) == 2


def test_runtime_data_loader_mixture_is_seed_reproducible(tmp_path: Path) -> None:
    """Same spec and seed reproduce the prompt set; a different seed redraws it."""
    anatomy = _write_prompts(tmp_path / "anatomy.jsonl", [f"anatomy {i}" for i in range(50)])
    safety = _write_prompts(tmp_path / "safety.jsonl", [f"safety {i}" for i in range(50)])

    def load(seed: int) -> list[str]:
        data = {
            "loader": "prompt_manifest",
            "manifest": {str(anatomy): 8, str(safety): 2},
            "mix_seed": seed,
            "preprocessing": {"format": "jsonl"},
        }
        return [e.prompt for e in load_prompt_examples_from_config(_data_config(data))]

    assert load(7) == load(7)
    assert load(7) != load(8)


def test_runtime_data_loader_requires_a_seed_for_a_mixture(tmp_path: Path) -> None:
    """Every rank draws the mixture itself, so an unseeded draw would desync them."""
    anatomy = _write_prompts(tmp_path / "anatomy.jsonl", ["a", "b", "c"])
    safety = _write_prompts(tmp_path / "safety.jsonl", ["x", "y", "z"])

    with pytest.raises(ValueError, match=r"data\.mix_seed"):
        load_prompt_examples_from_config(
            _data_config(
                {
                    "loader": "prompt_manifest",
                    "manifest": {str(anatomy): 2, str(safety): 1},
                    "preprocessing": {"format": "jsonl"},
                },
            ),
        )


def test_runtime_data_loader_rejects_mixture_count_over_manifest_size(tmp_path: Path) -> None:
    """Asking for more prompts than a source holds fails loudly, not silently short."""
    anatomy = _write_prompts(tmp_path / "anatomy.jsonl", ["only one"])

    with pytest.raises(ValueError, match="only 1 available"):
        load_prompt_examples_from_config(
            OmegaConf.create(
                {
                    "loader": "prompt_manifest",
                    "manifest": {str(anatomy): 5},
                    "mix_seed": 20260818,
                    "preprocessing": {"format": "jsonl"},
                },
            ),
        )


def test_for_experiment_plan_covers_every_mixture_source(tmp_path: Path) -> None:
    """Each source of a manifest mixture gets its own present/populate step."""
    plan = bootstrap.resolve_experiment_dataset_plan(
        {
            "loader": "prompt_manifest",
            "manifest": {
                "manifests/danbooru/safety/train_c_adherence.jsonl": 6800,
                "manifests/danbooru/safety/train.jsonl": 1200,
            },
        },
        repo_root=tmp_path,
    )

    assert [step["path"] for step in plan["steps"]] == [
        "manifests/danbooru/safety/train_c_adherence.jsonl",
        "manifests/danbooru/safety/train.jsonl",
    ]
    assert plan["ready"] is False


def test_runtime_data_loader_rejects_explicit_format_conflict() -> None:
    with pytest.raises(ValueError, match=r"requires.*prompt_image_manifest"):
        _data_config(
            {
                "loader": "prompt_manifest",
                "manifest": "unused.jsonl",
                "preprocessing": {"format": "image_caption_jsonl"},
            },
        )


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
        resolve_prompt_example_references(example, allow_absolute=True)
        for example in load_prompt_dataset_index(manifest)
    ]
    resolve_required_reference_images_(
        examples,
        manifest_path=manifest,
    )
    assert examples[0].metadata["source_episode"] == "000001"
    assert Path(examples[0].reference_image).exists()


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


def test_for_experiment_plan_marks_committed_manifest_ready(tmp_path: Path) -> None:
    """A committed prompt manifest marks the experiment's dataset plan ready, with its row count
    and no fetch command.
    """
    manifest_dir = tmp_path / "manifests" / "pickscore_sfw"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "train.txt").write_text("a\nb\nc\n", encoding="utf-8")

    plan = bootstrap.resolve_experiment_dataset_plan(
        {"loader": "prompt_manifest", "manifest": "manifests/pickscore_sfw/train.txt"},
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
    """A missing Danbooru manifest leaves the plan not ready and names the ``anime-safety-prompts``
    command that produces it.
    """
    plan = bootstrap.resolve_experiment_dataset_plan(
        {
            "loader": "prompt_manifest",
            "manifest": "manifests/danbooru/safety/train.jsonl",
        },
        repo_root=tmp_path,
    )

    assert plan["ready"] is False
    assert "anime-safety-prompts" in plan["steps"][0]["get"]


def test_for_experiment_resolves_real_wan_experiment(capsys) -> None:
    """``for-experiment`` on the real Wan Kling experiment reports the config's loader and
    manifest and finds every step present and complete.
    """
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


@pytest.mark.parametrize("source_fps", [0, -1, float("nan"), float("inf")])
def test_video_world_rejects_invalid_source_fps_before_media_write(tmp_path, source_fps):
    references = tmp_path / "references"
    targets = tmp_path / "targets"

    def unexpected_writer(path, frames, fps):
        pytest.fail("invalid FPS must fail before writing video")

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
            video_writer=unexpected_writer,
        )
    assert list(references.iterdir()) == []
    assert list(targets.iterdir()) == []


@pytest.mark.parametrize("metadata", [False, 0, "", []])
@pytest.mark.parametrize("target_video", [False, True])
def test_video_world_rejects_metadata_before_media_write(tmp_path, metadata, target_video):
    image = Image.new("RGB", (2, 2))
    episodes = [
        {
            "prompt": "move",
            "episode_id": "1",
            "image": image,
            "frames": [image],
            "metadata": metadata,
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
            video_writer=lambda path, frames, fps: path.touch(),
        )
    else:
        rows = video_world.build_video_world_rows(episodes, **kwargs)
    assert len(rows) == 1
    assert rows[0]["prompt"] == "move"
    assert rows[0]["metadata"]["source_episode"] == "0"
    assert len(list(tmp_path.rglob("*.png"))) == 1
    assert not list(tmp_path.rglob("*None*"))
