"""Sequential edits consume prior outputs and report lost earlier requirements."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from agentic.episode import Action, Artifact, PolicyStamp, Score, Task
from agentic.scripts.visual_sequence import SequencePlan, run_visual_sequence


@pytest.mark.asyncio
async def test_stages_preserve_lineage_and_report_regression_without_changing_credit(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (4, 4), "black").save(source)
    task = Task(
        "stages",
        "Keep red and add green",
        Artifact.from_path(source),
        (
            Action("red", "edit", "Set red channel"),
            Action("green", "edit", "Set green channel; preserve red"),
            Action("stop", "stop"),
        ),
    )
    plan = SequencePlan.model_validate(
        {
            "sequence_id": "two-stage",
            "actions": ["red", "green"],
            "requirements": [
                {"name": "red", "axis": "red", "threshold": 1.0, "active_from": 1},
                {"name": "green", "axis": "green", "threshold": 1.0, "active_from": 2},
            ],
        }
    )
    stamp = PolicyStamp("fixture-tool", "v1", 0)
    parents = []

    async def edit(task, observation, action, *, seed, output_dir):
        parents.append(observation.current)
        # Deliberate fixture defect: the second edit forgets the first channel.
        path = output_dir / f"{action.name}.png"
        Image.new("RGB", (4, 4), (255, 0, 0) if action.name == "red" else (0, 255, 0)).save(path)
        return Artifact.from_path(path)

    async def score(task, artifacts):
        scores = []
        for artifact in artifacts:
            with Image.open(artifact.path) as image:
                r, g, _ = image.getpixel((0, 0))
            scores.append(Score((r + g) / 510, {"red": r / 255, "green": g / 255}))
        return scores

    editor = SimpleNamespace(policy_stamp=stamp, activate=AsyncMock(), park=AsyncMock(), edit=edit)
    judge = SimpleNamespace(
        revision="pixel-fixture", activate=AsyncMock(), park=AsyncMock(), score=score
    )
    output = tmp_path / "run"
    result = await run_visual_sequence(
        task, plan, editor, judge, output_dir=output, seed=1, tool_cost=0.1
    )
    trace = json.loads((output / "episode/episode.json").read_text())
    assert parents[0] == task.source
    assert parents[1].sha256 == trace["steps"][0]["tool_result"]["artifact"]["sha256"]
    report = result["sequence_report"]
    assert report["requirements"]["red"]["regression_steps"] == [2]
    assert report["requirements"]["green"]["first_satisfied_step"] == 2
    assert report["final_requirements_met"] is False and report["coverage_complete"]
    assert trace["discounted_return"] == pytest.approx(0.3)
    assert [step["reward"] for step in trace["steps"]] == [-0.1, 0.4]
    provenance = json.loads((output / "media/provenance.json").read_text())
    assert len(provenance["sample_order"]) == 3
    assert provenance["lineage"][2]["parent_sha256"] == parents[1].sha256
    invalid = plan.model_copy(update={"actions": ("stop",)})
    before = editor.activate.await_count
    with pytest.raises(ValueError, match="existing edit actions"):
        await run_visual_sequence(
            task, invalid, editor, judge, output_dir=tmp_path / "invalid", seed=1
        )
    assert editor.activate.await_count == before
    assert not (tmp_path / "invalid").exists()

    editor.edit = AsyncMock(side_effect=RuntimeError("tool failed"))
    with pytest.raises(RuntimeError, match="tool failed"):
        await run_visual_sequence(
            task, plan, editor, judge, output_dir=tmp_path / "failed", seed=1
        )
    assert not (tmp_path / "failed/report.json").exists()
    assert json.loads((tmp_path / "failed/episode/episode.json").read_text())["status"] == "error"
