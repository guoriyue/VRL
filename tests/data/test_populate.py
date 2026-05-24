from __future__ import annotations

from pathlib import Path

from vrl.scripts.data import populate
from vrl.scripts.diffusion.cosmos.train import _normalize_per_sample_reference_images
from vrl.trainers.data import load_prompt_manifest


def test_video_world_tiny_population_writes_manifest_and_reference(
    monkeypatch,
    tmp_path: Path,
) -> None:
    populate.main(["video-world-tiny", "--data-root", str(tmp_path), "--name", "smoke"])
    monkeypatch.setenv("VRL_DATA_ROOT", str(tmp_path))

    manifest = tmp_path / "video_world" / "manifests" / "smoke_train.jsonl"
    examples = load_prompt_manifest(manifest)

    _normalize_per_sample_reference_images(
        examples,
        manifest_path=manifest,
        rollout_batch_size=1,
    )

    assert examples[0].reference_image == str(
        (tmp_path / "video_world" / "references" / "smoke_train_ref.ppm").resolve(),
    )
    assert examples[0].metadata["source_episode"] == "smoke_train_episode"
