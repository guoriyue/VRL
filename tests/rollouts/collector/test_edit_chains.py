"""Edit-chain collection conditions each step on a drawn sample of the previous one."""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from vrl.generation import GenerationOutput, GenerationRequest, GenerationSampleRow
from vrl.models.families.registry import get_model_family_entry
from vrl.rewards import RewardOutput, RewardSample
from vrl.rollouts.collector.chains import sample_image
from vrl.rollouts.collector.config import RolloutCollectorConfig
from vrl.rollouts.collector.core import RolloutCollector
from vrl.rollouts.collector.requests import GenerationRequestBuilder
from vrl.rollouts.stats import RolloutStats
from vrl.trainers.data.edit_chains import EditChain
from vrl.trainers.data.prompts import PromptExample
from vrl.trajectory.builders import build_diffusion_trajectory
from vrl.utils.media_reference import MediaReference


class _Runtime:
    """Each sample of a request decodes to a flat image whose value names the sample."""

    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    async def preflight(self) -> None:
        return None

    async def activate(self) -> None:
        return None

    async def generate(self, request: GenerationRequest) -> GenerationOutput:
        self.requests.append(request)
        rows = [
            GenerationSampleRow(
                prompt_index=0,
                sample_index=index,
                prompt=request.prompts[0],
                sample_id=f"{request.request_id}:0:{index}",
            )
            for index in range(request.samples_per_prompt)
        ]
        batch_size = len(rows)
        output = torch.stack(
            [torch.full((3, 2, 2), (index + 1) / 10) for index in range(batch_size)]
        )
        actions = torch.zeros(batch_size, 2, 1)
        trajectory = build_diffusion_trajectory(
            request=request,
            sample_rows=rows,
            observations=torch.zeros_like(actions),
            actions=actions,
            old_log_prob=torch.zeros(batch_size, 2),
            timesteps=torch.zeros(batch_size, 2),
            replay_tensors={"prompt_ids": torch.ones(batch_size, 3, dtype=torch.long)},
            context={},
        )
        return GenerationOutput(output=output, trajectory=trajectory)

    async def offload(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None


class _RewardRuntime:
    def __init__(self) -> None:
        self.calls: list[list[RewardSample]] = []

    async def preflight(self) -> None:
        return None

    async def activate(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def score(
        self, samples: Sequence[RewardSample], *, require_memory_release: bool = False
    ) -> RewardOutput:
        self.calls.append(list(samples))
        return RewardOutput(scores=tuple(float(index) for index in range(len(samples))))

    async def park_memory(self, *, required: bool) -> None:
        return None


def _collector(
    tmp_path: Path, *, media_dir: bool = True
) -> tuple[RolloutCollector, _Runtime, _RewardRuntime]:
    entry = get_model_family_entry("qwen_image_21")
    config = RolloutCollectorConfig(
        request_sampling={"seed": 3},
        edit_chain_media_dir=tmp_path / "edit_chains" if media_dir else None,
    )
    runtime, reward = _Runtime(), _RewardRuntime()
    collector = RolloutCollector(
        config=config,
        request_builder=GenerationRequestBuilder(entry=entry, config=config),
        reward_runtime=reward,
        generation_runtime=runtime,
    )
    return collector, runtime, reward


def _chain(tmp_path: Path, steps: list[str], chain_id: str = "chain") -> EditChain:
    source = tmp_path / f"{chain_id}.png"
    Image.new("RGB", (2, 2), "white").save(source)
    return EditChain(chain_id, str(source), [PromptExample(prompt=step) for step in steps])


@pytest.mark.asyncio
async def test_each_step_conditions_on_a_drawn_sample_and_scores_against_it(tmp_path) -> None:
    collector, runtime, reward = _collector(tmp_path)
    chain = _chain(tmp_path, ["recolor", "letter", "sharpen"])
    stats = RolloutStats()
    random.seed(0)
    expected_draws = [random.randrange(2) for _ in range(2)]
    random.seed(0)

    batches = await collector.prepare_training_batches(
        prompts=[chain], group_size=2, runtime_debug=False, policy_version=4, stats=stats
    )

    assert [request.prompts for request in runtime.requests] == [
        ["recolor"],
        ["letter"],
        ["sharpen"],
    ]
    references = [request.inputs[0].reference_images for request in runtime.requests]
    assert references[0] == [chain.source]
    for step, (draw, reference) in enumerate(zip(expected_draws, references[1:], strict=True)):
        parent = Path(reference[0])
        assert parent.parent == tmp_path / "edit_chains" / "chain"
        assert parent.name.startswith(f"step{step:02d}-")
        # The written parent is the drawn sample of the previous step.
        pixels = np.array(Image.open(parent))
        assert int(pixels[0, 0, 0]) == round((draw + 1) / 10 * 255)

    # One reward call over all steps; each sample sees its own parent as reference.
    assert len(reward.calls) == 1
    metadata = [sample.metadata for sample in reward.calls[0]]
    assert [item["chain_step"] for item in metadata] == [0, 0, 1, 1, 2, 2]
    assert [item["reference_images"] for item in metadata[::2]] == references
    assert [item["prompt_id"] for item in metadata[::2]] == ["chain:0", "chain:1", "chain:2"]
    assert metadata[0]["chain_parent_sample_id"] is None
    assert (
        metadata[2]["chain_parent_sample_id"]
        == runtime.requests[0].request_id + f":0:{expected_draws[0]}"
    )

    # Three groups with distinct ids, one per step, in step order.
    assert [batch.group_ids.tolist() for batch in batches] == [[0, 0], [1, 1], [2, 2]]
    assert [batch.rewards.tolist() for batch in batches] == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
    assert all(
        batch.context["reward_metadata"]["rollout_policy_version"] == 4 for batch in batches
    )
    assert stats.counters["collect.group_count"] == 3
    assert stats.counters["collect.sample_count"] == 6


@pytest.mark.asyncio
async def test_chains_and_one_shot_prompts_do_not_share_a_call(tmp_path) -> None:
    collector, _, _ = _collector(tmp_path)
    chain = _chain(tmp_path, ["a"])
    with pytest.raises(ValueError, match="cannot share"):
        await collector.prepare_training_batches(
            prompts=[chain, "plain prompt"],
            group_size=2,
            runtime_debug=False,
            policy_version=None,
            stats=RolloutStats(),
        )
    with pytest.raises(ValueError, match="step by step"):
        list(
            collector.build_generation_requests(
                prompts=[chain], group_size=2, runtime_debug=False, policy_version=None
            )
        )


@pytest.mark.asyncio
async def test_chain_collection_requires_a_media_dir(tmp_path) -> None:
    collector, _, _ = _collector(tmp_path, media_dir=False)
    with pytest.raises(ValueError, match="edit_chain_media_dir"):
        await collector.prepare_training_batches(
            prompts=[_chain(tmp_path, ["a", "b"])],
            group_size=2,
            runtime_debug=False,
            policy_version=None,
            stats=RolloutStats(),
        )


def test_sample_image_accepts_batches_frames_and_references(monkeypatch) -> None:
    batch = torch.arange(2 * 3 * 1 * 2 * 2, dtype=torch.float32).reshape(2, 3, 1, 2, 2)
    assert sample_image(batch, 1).shape == (3, 2, 2)
    assert torch.equal(sample_image(batch, 1), batch[1, :, 0])

    reference = MediaReference(object_ref="ref", sample_index=0)
    monkeypatch.setattr(MediaReference, "resolve", lambda self, cache=None: batch[0, :, 0])
    assert torch.equal(sample_image([reference], 0), batch[0, :, 0])
    with pytest.raises(ValueError, match="one image"):
        sample_image(torch.zeros(2, 6, 2, 2, 1), 0)
