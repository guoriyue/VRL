"""The editor consumes the current image; the judge scores against the original task."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from PIL import Image

from agentic.episode import Action, Artifact, Observation, PolicyStamp, Score, Task
from agentic.roles import LocalEditor, RewardJudge
from vrl.rewards.types import RewardOutput


@pytest.mark.asyncio
async def test_editor_consumes_current_image_judge_checks_original_and_parking_failure(tmp_path):
    original, current = tmp_path / "original.png", tmp_path / "current.png"
    Image.new("RGB", (8, 8), "black").save(original)
    Image.new("RGBA", (8, 8), (255, 0, 0, 0)).save(current)
    action = Action("blue", "edit", "Make the object blue")
    task = Task(
        "blue-task",
        "A blue object with its background unchanged",
        Artifact.from_path(original),
        (action, Action("stop", "stop")),
    )
    observation = Observation(1, 1, Artifact.from_path(current), Score(0.2))
    requests = []

    class Model(torch.nn.Linear):
        def move_frozen_components(self, device):
            pass

    class Executor:
        def forward_batch(self, request, batch):
            requests.append(request)
            return torch.full((1, 4, 8, 8), 127, dtype=torch.uint8)

        def merge_generation_batches(self, request, rows, batches):
            return SimpleNamespace(output=batches[0])

    editor = LocalEditor(
        Model(2, 2),
        Executor(),
        policy=PolicyStamp("editor", "fake", 0),
        family="qwen_image_21",
        sampling={},
        device=torch.device("cpu"),
    )
    await editor.park()
    await editor.activate()
    output = await editor.edit(task, observation, action, seed=91, output_dir=tmp_path / "edit")
    assert requests[0].inputs[0].reference_images == [str(current)]
    assert requests[0].inputs[0].prompt == action.instruction
    assert requests[0].sampling["seed"] == 91
    with Image.open(output.path) as image:
        assert image.mode == "RGBA" and image.getpixel((0, 0))[3] == 127
    second = await editor.edit(
        task,
        replace(observation, step=2, current=output),
        action,
        seed=92,
        output_dir=tmp_path / "edit",
    )
    assert second.path != output.path
    await editor.park()
    with pytest.raises(RuntimeError, match="healthy and active"):
        await editor.edit(task, observation, action, seed=91, output_dir=tmp_path / "bad")

    runtime = SimpleNamespace(
        activate=AsyncMock(),
        park_memory=AsyncMock(),
        score=AsyncMock(return_value=RewardOutput((0.7,), {"locality": (0.9,)})),
    )
    judge = RewardJudge(runtime, revision="fake-v1", require_memory_release=True)
    await judge.park()  # Even an external service must acknowledge the initial handoff.
    runtime.activate.assert_awaited_once()
    runtime.park_memory.assert_awaited_once_with(required=True)
    await judge.activate()
    score = await judge.score(task, observation.current)
    sample = runtime.score.call_args.args[0][0]
    assert sample.prompt == task.instruction
    assert sample.metadata["reference_images"] == [str(original)]
    assert sample.output.shape == (4, 1, 8, 8)
    assert torch.all(sample.output[0] == 1) and torch.all(sample.output[3] == 0)
    assert score.total == 0.7 and score.components == {"locality": 0.9}
    runtime.park_memory.side_effect = RuntimeError("partial release")
    with pytest.raises(RuntimeError, match="partial release"):
        await judge.park()
    with pytest.raises(RuntimeError, match="replace its owner"):
        await judge.activate()


@pytest.mark.asyncio
async def test_judge_preserves_alpha_and_passes_reward_only_targets_by_path(tmp_path):
    source, target, candidate = (
        tmp_path / name for name in ("source.png", "target.png", "candidate.png")
    )
    Image.new("RGBA", (16, 16), "white").save(source)
    layer = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    layer.paste((255, 0, 0, 255), (4, 4, 12, 12))
    layer.save(target)
    layer.save(candidate)
    task = Task.from_manifest_record(
        {
            "task_id": "extract",
            "instruction": "Extract the red square",
            "source": "source.png",
            "actions": [{"name": "stop", "kind": "stop"}],
            "requirement": "Keep the square edges crisp",
            "reward_assets": {"target_image": "target.png"},
        },
        base_dir=tmp_path,
    )
    runtime = SimpleNamespace(
        activate=AsyncMock(),
        park_memory=AsyncMock(),
        score=AsyncMock(return_value=RewardOutput((1.0,), {})),
    )
    judge = RewardJudge(runtime, revision="pixel-fixture", require_memory_release=False)
    await judge.activate()
    await judge.score(task, Artifact.from_path(candidate))
    sample = runtime.score.call_args.args[0][0]
    # The judge sends straight RGBA and names the reward-only target by path.
    assert sample.metadata["target_image"] == str(target)
    assert sample.metadata["requirement"] == "Keep the square edges crisp"
    assert sample.output.shape == (4, 1, 16, 16)
    assert torch.all(sample.output[3, 0, 4:12, 4:12] == 1)
    assert torch.all(sample.output[3, 0, :4] == 0)
    await judge.park()
