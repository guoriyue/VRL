"""LeRobot v2.0 episode parsing preserves JSONL content and prefix limits."""

import json

import numpy as np
import pytest

from vrl.scripts.data.video_world import lerobot


@pytest.mark.parametrize("target_clips", [False, True])
def test_v20_preserves_unicode_prompt_and_stops_at_limit(tmp_path, monkeypatch, target_clips):
    prompt = "move\u2028left"
    path = tmp_path / "episodes.jsonl"
    path.write_text(
        "\n"
        + json.dumps({"episode_index": 0, "tasks": [prompt]}, ensure_ascii=False)
        + "\ninvalid trailing record\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        lerobot,
        "_decode_frames",
        lambda *args, **kwargs: iter([(0, np.zeros((2, 2, 3), dtype=np.uint8))]),
    )
    kwargs = dict(video_key="camera", limit=1, dl=lambda relative: str(path))
    info = {"video_path": "videos/{episode_index}.mp4", "fps": 15}
    if target_clips:
        rows = list(
            lerobot._iter_lerobot_v20_target_clips(
                "unit/repo",
                info,
                **kwargs,
                max_target_frames=2,
            )
        )
    else:
        rows = list(lerobot._iter_lerobot_v20("unit/repo", info, **kwargs))
    assert len(rows) == 1
    assert rows[0]["prompt"] == prompt
    assert rows[0]["episode_id"] == "000000"
