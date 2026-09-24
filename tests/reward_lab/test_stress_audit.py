"""Real perturbations remain reproducible and failed score pairs remain visible."""

import copy
import json

import numpy as np
import pytest
from PIL import Image

from reward_lab.diagnostics import stress_report
from reward_lab.scripts.stress_media import build_stress_manifest
from vrl.rewards.evaluation import (
    ScoringConfig,
    load_media_manifest,
    read_evaluation,
    rescore_media,
)
from vrl.utils.artifacts import sha256_file


@pytest.mark.asyncio
async def test_stress_preserves_alpha_except_explicit_probes_and_records_score_changes(
    tmp_path,
):
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[8:24, 8:24] = [255, 0, 0, 255]
    Image.fromarray(rgba).save(tmp_path / "target.png")
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "sample_id": "original",
                "prompt_id": "extract",
                "prompt": "Extract red square",
                "path": "target.png",
                "sha256": sha256_file(tmp_path / "target.png"),
                "assets": {"target_image": "target.png"},
            }
        )
        + "\n"
    )
    manifest = build_stress_manifest(source, tmp_path / "stress")
    repeated = build_stress_manifest(source, tmp_path / "repeat")
    rows, copies = load_media_manifest(manifest), load_media_manifest(repeated)
    assert [sha256_file(row.path) for row in rows] == [sha256_file(row.path) for row in copies]
    for row in rows:
        variant = row.metadata["reward_stress"]["transform"]
        with Image.open(row.path) as image:
            alpha = np.asarray(image)[..., 3]
        if variant == "opaque_alpha":
            assert (alpha == 255).all()
        elif variant == "empty_alpha":
            assert not alpha.any()
        else:
            np.testing.assert_array_equal(alpha, rgba[..., 3])
    with pytest.raises(FileExistsError):
        build_stress_manifest(source, tmp_path / "stress")
    with pytest.raises(ValueError, match="nested"):
        build_stress_manifest(manifest, tmp_path / "nested")
    config = ScoringConfig(
        name="sharpness",
        revision="v1",
        preprocessing_revision="rgba-white-composite",
        rubric_revision="laplacian",
        worker_config={
            "model_factory": "vrl.rewards.models.image_sharpness:ImageSharpnessRewardModel",
            # Unsaturated: the default scale clips this hard-edged square at 1.
            "scale": 10.0,
        },
    )
    await rescore_media(manifest, config, tmp_path / "scores")
    evaluation = read_evaluation(tmp_path / "scores")
    report = stress_report(evaluation)
    # An empty layer composites to a blank canvas; RGB noise is the shortcut the
    # sharpness docstring warns about and must show up as an increase.
    assert report["transforms"]["empty_alpha"]["image_sharpness"]["decreases"] == 1
    assert report["transforms"]["noise_rgb"]["image_sharpness"]["increases"] == 1
    broken = copy.deepcopy(evaluation)
    changed = next(
        row
        for row in broken["records"].values()
        if row["input"]["metadata"]["reward_stress"]["transform"] == "noise_rgb"
    )
    changed["status"] = "error"
    changed.pop("result")
    changed["error"] = {"type": "RuntimeError"}
    failed_report = stress_report(broken)
    failed = next(row for row in failed_report["observations"] if row["transform"] == "noise_rgb")
    assert failed["deltas"] == {} and failed["status"] == "error"
    assert "noise_rgb" not in failed_report["transforms"]
    changed["input"]["prompt"] = "different task"
    with pytest.raises(ValueError, match="identity or task"):
        stress_report(broken)
    Image.new("RGBA", (32, 32), "white").save(tmp_path / "target.png")
    with pytest.raises(ValueError, match="declared media SHA-256 mismatch"):
        build_stress_manifest(source, tmp_path / "changed-source")
    assert not (tmp_path / "changed-source").exists()
