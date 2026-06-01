from __future__ import annotations

import json
from pathlib import Path

import pytest

from vrl.scripts.data import setup as setup_cli
from vrl.trainers.data import load_prompt_manifest
from vrl.trainers.data.artifacts import (
    ArtifactManifestError,
    resolve_artifact_path,
    validate_artifact_manifest,
)


def _write_ppm(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")


def _write_manifest(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_artifact_manifest_resolves_relative_references_via_data_root(tmp_path: Path) -> None:
    data_root = tmp_path / "external"
    reference = data_root / "video_world" / "references" / "ref.ppm"
    manifest = tmp_path / "v2w_train.jsonl"
    _write_ppm(reference)
    _write_manifest(
        manifest,
        {
            "prompt": "reference prompt",
            "reference_image": "video_world/references/ref.ppm",
        },
    )

    examples = load_prompt_manifest(manifest)
    report = validate_artifact_manifest(
        manifest,
        data_root=data_root,
        required_artifact_fields=("reference_image",),
    )

    assert examples[0].reference_image == "video_world/references/ref.ppm"
    assert report.row_count == 1
    assert report.artifact_count == 1
    assert report.resolved_artifacts[0].resolved_path == reference.resolve()


def test_missing_reference_image_fails_with_manifest_row(tmp_path: Path) -> None:
    manifest = tmp_path / "missing.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "prompt": "missing ref",
                "reference_image": "video_world/references/missing.png",
            },
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ArtifactManifestError, match=r"row 0 reference_image does not exist"):
        validate_artifact_manifest(
            manifest,
            data_root=tmp_path,
            required_artifact_fields=("reference_image",),
        )


def test_absolute_artifact_paths_are_rejected_by_default(tmp_path: Path) -> None:
    path = tmp_path / "ref.ppm"
    path.write_text("P3\n1 1\n255\n0 0 0\n", encoding="utf-8")

    with pytest.raises(ArtifactManifestError, match="absolute artifact paths"):
        resolve_artifact_path(path)


def test_production_metadata_domain_is_rejected(tmp_path: Path) -> None:
    reference = tmp_path / "video_world" / "references" / "ref.ppm"
    _write_ppm(reference)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(
        manifest,
        {
            "prompt": "bad metadata",
            "reference_image": "video_world/references/ref.ppm",
            "metadata": {"domain": "robotics"},
        },
    )

    with pytest.raises(ArtifactManifestError, match=r"metadata\.domain"):
        validate_artifact_manifest(
            manifest,
            data_root=tmp_path,
            required_artifact_fields=("reference_image",),
        )


def test_setup_cli_creates_ignored_external_dirs(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "external"

    assert setup_cli.main(
        [
            "init-dirs",
            "video-world",
            "--data-root",
            str(data_root),
        ],
    ) is None

    assert (data_root / "video_world" / "references").is_dir()
    assert (data_root / "video_world" / "source_videos").is_dir()
