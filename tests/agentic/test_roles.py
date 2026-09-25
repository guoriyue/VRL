"""The editor consumes the current image; the judge scores states against the source."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from PIL import Image

from agentic.chains import Artifact, EditChain, PolicyStamp
from agentic.roles import LocalEditor, RewardJudge
from vrl.rewards.types import RewardOutput
from vrl.trainers.data.prompts import PromptExample


@pytest.mark.asyncio
async def test_editor_consumes_current_image_judge_scores_against_source(tmp_path):
    original, current = tmp_path / "original.png", tmp_path / "current.png"
    Image.new("RGB", (8, 8), "black").save(original)
    Image.new("RGBA", (8, 8), (255, 0, 0, 0)).save(current)
    chain = EditChain(
        "blue-task",
        Artifact.from_path(original),
        [PromptExample(prompt="Make the object blue"), PromptExample(prompt="Sharpen")],
        tmp_path,
        requirement="Keep the background unchanged",
    )
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
    first = Artifact.from_path(current)
    output = await editor.edit(chain, 0, first, seed=91, output_dir=tmp_path / "edit")
    assert requests[0].inputs[0].reference_images == [str(current)]
    assert requests[0].inputs[0].prompt == "Make the object blue"
    assert requests[0].sampling["seed"] == 91
    with Image.open(output.path) as image:
        assert image.mode == "RGBA" and image.getpixel((0, 0))[3] == 127
    second = await editor.edit(chain, 1, output, seed=92, output_dir=tmp_path / "edit")
    assert second.path != output.path
    assert requests[1].inputs[0].reference_images == [output.path]
    await editor.park()
    with pytest.raises(RuntimeError, match="healthy and active"):
        await editor.edit(chain, 0, first, seed=91, output_dir=tmp_path / "bad")

    runtime = SimpleNamespace(
        activate=AsyncMock(),
        park_memory=AsyncMock(),
        score=AsyncMock(return_value=RewardOutput((0.2, 0.7), {"locality": (0.5, 0.9)})),
    )
    judge = RewardJudge(runtime, revision="fake-v1", require_memory_release=True)
    await judge.park()  # Even an external service must acknowledge the initial handoff.
    runtime.activate.assert_awaited_once()
    runtime.park_memory.assert_awaited_once_with(required=True)
    await judge.activate()
    scores = await judge.score(chain, [chain.source, first])
    samples = runtime.score.call_args.args[0]
    assert [sample.prompt for sample in samples] == ["Make the object blue Sharpen"] * 2
    assert samples[1].metadata["reference_images"] == [str(original)]
    assert samples[1].metadata["requirement"] == "Keep the background unchanged"
    assert samples[1].output.shape == (4, 1, 8, 8)
    assert torch.all(samples[1].output[0] == 1) and torch.all(samples[1].output[3] == 0)
    assert [score.total for score in scores] == [0.2, 0.7]
    assert scores[1].components == {"locality": 0.9}
    runtime.park_memory.side_effect = RuntimeError("partial release")
    with pytest.raises(RuntimeError, match="partial release"):
        await judge.park()
    with pytest.raises(RuntimeError, match="replace its owner"):
        await judge.activate()


@pytest.mark.asyncio
async def test_judge_preserves_alpha_and_passes_reward_assets_by_path(tmp_path):
    source, target, candidate = (
        tmp_path / name for name in ("source.png", "target.png", "candidate.png")
    )
    Image.new("RGBA", (16, 16), "white").save(source)
    layer = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    layer.paste((255, 0, 0, 255), (4, 4, 12, 12))
    layer.save(target)
    layer.save(candidate)
    chain = EditChain(
        "extract",
        Artifact.from_path(source),
        [PromptExample(prompt="Extract the red square")],
        tmp_path,
        reward_assets={"target_image": Artifact.from_path(target)},
    )
    runtime = SimpleNamespace(
        activate=AsyncMock(),
        park_memory=AsyncMock(),
        score=AsyncMock(return_value=RewardOutput((1.0,), {})),
    )
    judge = RewardJudge(runtime, revision="pixel-fixture", require_memory_release=False)
    await judge.activate()
    await judge.score(chain, [Artifact.from_path(candidate)])
    sample = runtime.score.call_args.args[0][0]
    # The judge sends straight RGBA and names the reward-only target by path.
    assert sample.metadata["target_image"] == str(target)
    assert sample.output.shape == (4, 1, 16, 16)
    assert torch.all(sample.output[3, 0, 4:12, 4:12] == 1)
    assert torch.all(sample.output[3, 0, :4] == 0)
    await judge.park()
