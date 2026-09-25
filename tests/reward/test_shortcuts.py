"""Shortcut manifests pair every task-avoiding candidate with the genuine one, like stress."""

import json

import numpy as np
import pytest
from PIL import Image

from reward.analysis import Analysis
from reward.shortcuts import SHORTCUTS, build_shortcut_manifest
from vrl.rewards.evaluation import Evaluation, ScoringConfig, load_media_manifest


def _scene(seed: int, size: int = 24) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))


def _manifest(tmp_path):
    rows = []
    for n, seed in (("a", 1), ("b", 2)):
        _scene(seed).save(tmp_path / f"source_{n}.png")
        edited = _scene(seed)
        edited.paste((255, 0, 0), (4, 4, 12, 12))  # the genuine edit: a red patch
        edited.save(tmp_path / f"edit_{n}.png")
        rows.append(
            {
                "sample_id": n,
                "prompt_id": n,
                "prompt": "add a red patch",
                "path": f"edit_{n}.png",
                "assets": {"reference_image": f"source_{n}.png"},
            }
        )
    manifest = tmp_path / "media.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows))
    return manifest


def test_shortcut_manifest_writes_every_transform_beside_its_baseline(tmp_path):
    manifest = _manifest(tmp_path)
    out = build_shortcut_manifest(manifest, tmp_path / "shortcuts", seed=7)
    rows = load_media_manifest(out)
    assert len(rows) == 2 * (1 + len(SHORTCUTS))
    by_id = {row.sample_id: row for row in rows}
    unchanged = np.asarray(Image.open(by_id["a:unchanged_source"].path))
    assert np.array_equal(unchanged, np.asarray(Image.open(tmp_path / "source_a.png")))
    baseline = np.asarray(Image.open(by_id["a:baseline"].path))
    assert not np.array_equal(unchanged, baseline)
    shifted = np.asarray(Image.open(by_id["a:shift_frame"].path))
    assert np.array_equal(shifted[1:, 1:], baseline[:-1, :-1])  # 3% of 24 px rounds to 1
    other = np.asarray(Image.open(by_id["a:other_scene"].path))
    assert np.array_equal(other, np.asarray(Image.open(by_id["b:baseline"].path)))
    probe = by_id["a:crop_zoom"].metadata["reward_stress"]
    assert probe["baseline_id"] == "a:baseline" and probe["transform"] == "crop_zoom"
    assert json.loads((tmp_path / "shortcuts" / "recipe.json").read_text())["shortcuts"] == list(
        SHORTCUTS
    )


def test_shortcut_manifest_refuses_unknown_names_and_missing_reference(tmp_path):
    manifest = _manifest(tmp_path)
    with pytest.raises(ValueError, match="unknown shortcuts"):
        build_shortcut_manifest(manifest, tmp_path / "x", shortcuts=["teleport"])
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    for row in rows:
        row.pop("assets")
    bare = tmp_path / "bare.jsonl"
    bare.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="reference_image"):
        build_shortcut_manifest(bare, tmp_path / "y", shortcuts=["unchanged_source"])


@pytest.mark.asyncio
async def test_stress_report_pairs_shortcuts_with_the_genuine_candidate(tmp_path):
    manifest = _manifest(tmp_path)
    out = build_shortcut_manifest(manifest, tmp_path / "shortcuts", shortcuts=["shift_frame"])
    config = ScoringConfig(
        name="sharpness",
        revision="v1",
        preprocessing_revision="native",
        rubric_revision="laplacian",
        worker_config={
            "model_factory": "vrl.rewards.models.image_sharpness:ImageSharpnessRewardModel",
            "device": "cpu",
        },
    )
    evaluation = await Evaluation.score(out, config, tmp_path / "scores")
    report = Analysis(evaluation).stress()
    entry = report["transforms"]["shift_frame"]["image_sharpness"]
    assert entry["increases"] + entry["decreases"] + entry["ties"] == 2
