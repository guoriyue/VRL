"""IDM scripts share the repository JSONL record contract."""

import json

import pytest

from vrl.scripts.rewards import idm_discrimination_probe, train_droid_idm


@pytest.mark.parametrize("script", ["training", "probe"])
def test_idm_manifest_accepts_unicode_prompt_and_blank_lines(tmp_path, monkeypatch, script):
    manifest = tmp_path / "manifest.jsonl"
    row = {"prompt": "move\u2028left", "metadata": {}}
    manifest.write_text("\n" + json.dumps(row, ensure_ascii=False) + "\n\n", encoding="utf-8")
    # Parsing must reach each script's action-data check, without decoding video
    # or loading a real model. This row deliberately has no target actions.
    if script == "training":
        with pytest.raises(SystemExit, match="no rows with target_actions"):
            train_droid_idm.load_pair_dataset(manifest, image_size=8)
    else:
        monkeypatch.setattr(idm_discrimination_probe, "load_idm_checkpoint", lambda *a, **k: {})
        with pytest.raises(SystemExit, match="need >= 2 eval rows"):
            idm_discrimination_probe.main(
                ["--checkpoint", "unused.pt", "--eval-manifest", str(manifest), "--device", "cpu"]
            )
