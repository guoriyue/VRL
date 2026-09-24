"""Episode decisions observe new images and own causal, bounded credit."""

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from agentic.episode import Action, Artifact, Decision, Episode, PolicyStamp, Score, Task


@pytest.mark.asyncio
async def test_observed_edit_stop_budget_and_failed_handoff_are_distinct(tmp_path):
    source = tmp_path / "source.png"
    candidate = tmp_path / "candidate.png"
    Image.new("RGB", (8, 8), "black").save(source)
    Image.new("RGB", (8, 8), "white").save(candidate)
    task = Task(
        "white",
        "Make the image white",
        Artifact.from_path(source),
        (Action("whiten", "edit", "Make the image white"), Action("accept", "stop")),
    )
    active, events = set(), []

    class Role:
        def __init__(self, name):
            self.name = name
            self.fail_park = False
            self.policy_stamp = PolicyStamp(name, "fake-v1", 0)

        async def activate(self):
            assert not active, "a previous owner did not release memory"
            active.add(self.name)
            events.append((self.name, "activate"))

        async def park(self):
            if self.fail_park and self.name in active:
                raise RuntimeError("park failed")
            active.discard(self.name)
            events.append((self.name, "park"))

    class Controller(Role):
        async def decide(self, task, observation, *, seed):
            # Decision uses the actual current pixels, not a scripted turn index.
            with Image.open(observation.current.path) as image:
                action = "accept" if image.getpixel((0, 0))[0] == 255 else "whiten"
            return Decision(
                action, self.policy_stamp, -0.4, observation.digest(task), {"seed": seed}
            )

    class Editor(Role):
        async def edit(self, task, observation, action, *, seed, output_dir):
            events.append(("editor", "edit"))
            return Artifact.from_path(candidate)

    class Judge(Role):
        revision = "pixel-fixture-v1"

        async def score(self, task, artifact):
            with Image.open(artifact.path) as image:
                return Score(image.getpixel((0, 0))[0] / 255)

    controller, editor, judge = Controller("controller"), Editor("editor"), Judge("judge")
    trace = await Episode(tool_cost=0.2).run(
        task, controller, editor, judge, output_dir=tmp_path / "normal", seed=11
    )
    assert trace["status"] == "success" and trace["termination"] == "stop"
    assert [step["decision"]["action"] for step in trace["steps"]] == ["whiten", "accept"]
    assert [step["reward"] for step in trace["steps"]] == [-0.2, 1.0]
    assert [step["return_to_go"] for step in trace["steps"]] == [0.8, 1.0]
    assert trace["steps"][0]["tool_result"]["artifact"]["path"] == str(candidate)
    assert "tool_result" not in trace["steps"][1]
    assert trace["tool_calls"] == 1 and not active
    assert json.loads((tmp_path / "normal/episode.json").read_text()) == json.loads(
        json.dumps(trace)
    )

    budget = await Episode(tool_cost=0.2, max_tool_calls=1).run(
        task, controller, editor, judge, output_dir=tmp_path / "budget", seed=11
    )
    assert budget["termination"] == "tool_budget"
    assert len(budget["steps"]) == 1  # No fabricated stop decision at the horizon.
    assert budget["discounted_return"] == 0.8
    stop = await Episode().run(
        replace(task, source=Artifact.from_path(candidate)),
        controller,
        editor,
        judge,
        output_dir=tmp_path / "stop",
        seed=11,
    )
    assert stop["tool_calls"] == 0 and stop["discounted_return"] == 1.0

    controller.fail_park = True
    events.clear()
    with pytest.raises(RuntimeError, match="park failed"):
        await Episode().run(
            task, controller, editor, judge, output_dir=tmp_path / "failed", seed=11
        )
    failed = json.loads((tmp_path / "failed/episode.json").read_text())
    assert failed["status"] == "error" and "discounted_return" not in failed
    assert ("editor", "activate") not in events
    assert active == {"controller"}  # No false claim that a failed handoff released it.


def test_return_to_go_never_credits_a_later_action_with_an_earlier_cost():
    assert Episode.returns([-0.2, -0.2, 1.0], gamma=0.5) == pytest.approx([-0.05, 0.3, 1.0])
    assert Episode.returns([1.0, 0.0], gamma=1.0) == [1.0, 0.0]
    with pytest.raises(ValueError, match="finite"):
        Episode.returns([float("nan")])


@pytest.mark.asyncio
async def test_failed_edit_records_attempt_without_fabricating_training_returns(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (8, 8), "black").save(source)
    task = Task(
        "edit",
        "edit",
        Artifact.from_path(source),
        (Action("edit", "edit", "edit"), Action("stop", "stop")),
    )
    stamp = PolicyStamp("controller", "fixture", 0)

    async def decide(task, observation, *, seed):
        return Decision("edit", stamp, -0.5, observation.digest(task))

    controller = SimpleNamespace(
        policy_stamp=stamp, activate=AsyncMock(), park=AsyncMock(), decide=decide
    )
    editor = SimpleNamespace(
        policy_stamp=PolicyStamp("generator", "fixture", 0),
        activate=AsyncMock(),
        park=AsyncMock(),
        edit=AsyncMock(side_effect=RuntimeError("tool failed")),
    )
    judge = SimpleNamespace(
        revision="fixture",
        activate=AsyncMock(),
        park=AsyncMock(),
        score=AsyncMock(return_value=Score(0.0)),
    )
    with pytest.raises(RuntimeError, match="tool failed"):
        await Episode().run(
            task, controller, editor, judge, output_dir=tmp_path / "episode", seed=3
        )
    trace = json.loads((tmp_path / "episode/episode.json").read_text())
    assert trace["status"] == "error"
    assert trace["steps"][0]["status"] == "error"
    assert "return_to_go" not in trace["steps"][0]
    assert "final_score" not in trace
    assert judge.score.await_count == 1  # Only the unchanged initial image was scored.
    assert editor.park.await_count == 2  # Initial handoff and failed operation cleanup.
    with pytest.raises(ValueError, match="successful"):
        Episode.training_steps(trace, policy=stamp)


def test_task_manifest_paths_are_relative_to_the_task_file_and_unknown_fields_fail(
    tmp_path, monkeypatch
):
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    Image.new("RGB", (8, 8), "white").save(tasks / "source.png")
    record = {
        "task_id": "white",
        "instruction": "Keep white",
        "source": "source.png",
        "actions": [{"name": "stop", "kind": "stop"}],
    }
    monkeypatch.chdir(tmp_path)
    task = Task.from_manifest_record(record, base_dir=tasks)
    assert task.source.path == str(tasks / "source.png")
    assert Task.from_record(task.as_record()) == task
    assert "requirement" not in task.as_record()
    rich = Task.from_manifest_record(
        {**record, "requirement": "Keep it white", "context_images": {"style": "source.png"}},
        base_dir=tasks,
    )
    assert rich.requirement == "Keep it white"
    assert rich.context_images["style"] == task.source
    assert Task.from_record(rich.as_record()) == rich
    with pytest.raises(ValueError, match="unknown visual task fields"):
        Task.from_manifest_record(
            {**record, "unimplemented_constraint": "preserve geometry"}, base_dir=tasks
        )
