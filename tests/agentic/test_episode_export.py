"""Episode exports list the source and every edit output with their lineage."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from agentic.episode import Artifact, Episode, PolicyStamp, Score, Task
from agentic.export import export_episode_media
from agentic.scripts.export_episode_media import main
from agentic.scripts.visual_control import BaselineController
from vrl.rewards.evaluation import load_media_manifest
from vrl.utils.json_files import canonical_json_sha256


@pytest.mark.asyncio
async def test_export_cli_keeps_the_original_reference_and_edit_lineage(tmp_path):
    source, edited = tmp_path / "source.png", tmp_path / "edited.png"
    Image.new("RGB", (16, 16), "white").save(source)
    Image.new("RGB", (16, 16), "blue").save(edited)
    task = Task.from_manifest_record(
        {
            "task_id": "blue",
            "instruction": "Make blue",
            "source": "source.png",
            "actions": [
                {"name": "blue", "kind": "edit", "instruction": "Make blue"},
                {"name": "stop", "kind": "stop"},
            ],
        },
        base_dir=tmp_path,
    )
    editor = SimpleNamespace(
        policy_stamp=PolicyStamp("fixture-editor", "v1", 0),
        activate=AsyncMock(),
        park=AsyncMock(),
        edit=AsyncMock(return_value=Artifact.from_path(edited)),
    )
    judge = SimpleNamespace(
        revision="fixture-judge",
        activate=AsyncMock(),
        park=AsyncMock(),
        score=AsyncMock(return_value=[Score(0.5), Score(0.5)]),
    )
    trace = await Episode().run(
        task, BaselineController("fixed"), editor, judge, output_dir=tmp_path / "episode", seed=1
    )
    output = tmp_path / "export"
    main(["--episode", str(tmp_path / "episode/episode.json"), "--output-dir", str(output)])
    rows = load_media_manifest(output / "media.jsonl")
    assert [row.path for row in rows] == [source, edited]
    assert all(row.assets["reference_image"] == source for row in rows)
    assert rows[0].metadata == rows[1].metadata
    report = json.loads((output / "provenance.json").read_text())
    assert report["lineage"][1]["parent_sha256"] == rows[0].sha256
    assert report["lineage"][1]["decision_index"] == 0
    assert report["episode_id"] == canonical_json_sha256(trace, allow_nan=False)
    assert json.loads((output / "episode.json").read_text()) == json.loads(json.dumps(trace))
    with pytest.raises(FileExistsError):
        export_episode_media(trace, output)
    with pytest.raises(ValueError, match="successful"):
        export_episode_media({**trace, "status": "error"}, tmp_path / "failed")
