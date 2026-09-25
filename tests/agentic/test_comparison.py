"""Best-of-N pays for discarded candidates and a failed handoff ends the comparison."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from agentic.episode import Action, Artifact, PolicyStamp, Score, Task
from agentic.scripts.visual_control import BaselineController, compare_visual_policies


@pytest.mark.asyncio
async def test_comparison_charges_all_candidates_and_stops_after_failed_handoff(tmp_path):
    source = tmp_path / "black.png"
    Image.new("RGB", (4, 4), "black").save(source)
    task = Task(
        "white",
        "Make white",
        Artifact.from_path(source),
        (Action("white", "edit", "Make white"), Action("stop", "stop")),
    )
    stamp = PolicyStamp("editor", "fake", 0)
    seen = []

    async def edit(task, observation, action, *, seed, output_dir):
        seen.append((observation.current.sha256, seed))
        path = output_dir / "candidate.png"
        Image.new("RGB", (4, 4), "white").save(path)
        return Artifact.from_path(path)

    async def score(task, artifacts):
        scores = []
        for artifact in artifacts:
            with Image.open(artifact.path) as image:
                scores.append(Score(image.getpixel((0, 0))[0] / 255))
        return scores

    editor = SimpleNamespace(policy_stamp=stamp, activate=AsyncMock(), park=AsyncMock(), edit=edit)
    judge = SimpleNamespace(revision="fake", activate=AsyncMock(), park=AsyncMock(), score=score)
    report = await compare_visual_policies(
        task,
        BaselineController("fixed"),
        editor,
        judge,
        output_dir=tmp_path / "comparison",
        seed=3,
        max_tool_calls=3,
        tool_cost=0.2,
    )
    methods = report["methods"]
    assert methods["stop"]["net_return"] == 0
    assert methods["fixed"]["net_return"] == 0.8
    assert methods["controller"]["controller_decisions"] == 2
    best = methods["best_of_n"]
    assert best["net_return"] == pytest.approx(0.4)
    assert best["tool_calls"] == 3 and best["judge_calls"] == 3
    assert len(best["trace_paths"]) == 3 and best["selected_index"] == 0
    assert [digest for digest, _ in seen[-3:]] == [task.source.sha256] * 3
    assert len({seed for _, seed in seen[-3:]}) == 3
    assert json.loads((tmp_path / "comparison/comparison.json").read_text()) == report
    assert not report["capability_improvement_verified"]

    editor.park = AsyncMock(side_effect=RuntimeError("unsafe memory handoff"))
    previous = editor.activate.await_count
    with pytest.raises(RuntimeError, match="unsafe memory"):
        await compare_visual_policies(
            task,
            BaselineController("fixed"),
            editor,
            judge,
            output_dir=tmp_path / "failed",
            seed=3,
            max_tool_calls=3,
            tool_cost=0.2,
        )
    failed = json.loads((tmp_path / "failed/comparison.json").read_text())
    assert failed["status"] == "error" and failed["methods"] == {}
    assert editor.activate.await_count == previous
