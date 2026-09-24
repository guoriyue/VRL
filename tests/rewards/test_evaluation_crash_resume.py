"""A killed scoring writer cannot retain ownership or erase completed sample evidence."""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from PIL import Image

from vrl.rewards.evaluation import ScoringConfig, read_evaluation, rescore_media


@pytest.mark.asyncio
async def test_sigkill_releases_writer_and_resume_reuses_completed_samples(tmp_path, monkeypatch):
    Image.new("RGB", (12, 12), "white").save(tmp_path / "source.png")
    manifest = tmp_path / "media.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": name,
                    "prompt_id": "white",
                    "prompt": "White image",
                    "path": "source.png",
                }
            )
            + "\n"
            for name in ("first", "second")
        )
    )
    # The real CPU scorer completes one batch, then blocks once in the second.
    # This is an interrupted backend fixture, not a mock of OS lock release.
    (tmp_path / "crash_reward.py").write_text("""
from pathlib import Path
import time
from vrl.rewards.models.image_sharpness import ImageSharpnessRewardModel
class InterruptedSharpness(ImageSharpnessRewardModel):
    def __init__(self, config):
        super().__init__(config)
        self.marker = Path(config["block_once_marker"])
    def score_batch(self, artifacts):
        if artifacts[0].artifact_id == "second" and not self.marker.exists():
            self.marker.write_text("ready")
            time.sleep(60)
        return super().score_batch(artifacts)
""")
    config = ScoringConfig(
        name="interrupted-sharpness",
        revision="fixture-v1",
        preprocessing_revision="native",
        rubric_revision="laplacian",
        worker_config={
            "model_factory": "crash_reward:InterruptedSharpness",
            "device": "cpu",
            "block_once_marker": str(tmp_path / "ready"),
        },
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(config.model_dump_json())
    output = tmp_path / "scores"
    code = """
import asyncio,sys
from pathlib import Path
from vrl.rewards.evaluation import ScoringConfig,rescore_media
config=ScoringConfig.model_validate_json(Path(sys.argv[2]).read_text())
asyncio.run(rescore_media(Path(sys.argv[1]),config,Path(sys.argv[3])))
"""
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(tmp_path), str(Path.cwd()), environment.get("PYTHONPATH", ""))
    )
    with (tmp_path / "child.log").open("w") as log:
        child = subprocess.Popen(
            [sys.executable, "-c", code, str(manifest), str(config_path), str(output)],
            env=environment,
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 30
            while not (tmp_path / "ready").exists():
                assert child.poll() is None, (tmp_path / "child.log").read_text()
                assert time.monotonic() < deadline, "scoring child did not reach the second batch"
                await asyncio.sleep(0.02)
            with pytest.raises(ValueError, match="still being written"):
                read_evaluation(output)
            with pytest.raises(FileExistsError, match="already in use"):
                await rescore_media(manifest, config, output, resume=True)
            child.kill()
            child.wait(timeout=10)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
    partial = read_evaluation(output)
    assert partial["records"]["first"]["status"] == "success"
    assert partial["records"]["second"]["status"] == "missing"
    monkeypatch.syspath_prepend(str(tmp_path))
    summary = await rescore_media(manifest, config, output, resume=True)
    assert summary["reused"] == summary["scored"] == 1
    assert all(row["status"] == "success" for row in read_evaluation(output)["records"].values())
