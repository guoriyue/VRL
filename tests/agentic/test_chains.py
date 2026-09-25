"""Edit chains condition each step on a drawn sample of the previous one."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from agentic.chains import EditChain, load_edit_chains, sample_image
from vrl.generation import GenerationOutput, GenerationRequest, GenerationSampleRow
from vrl.models.families.registry import get_model_family_entry
from vrl.rewards import RewardOutput, RewardSample
from vrl.rollouts.collector.config import RolloutCollectorConfig
from vrl.rollouts.collector.core import RolloutCollector
from vrl.rollouts.collector.requests import GenerationRequestBuilder
from vrl.rollouts.stats import RolloutStats
from vrl.trainers.data.prompts import PromptExample
from vrl.trajectory.builders import build_diffusion_trajectory
from vrl.utils.media_reference import MediaReference


class _Runtime:
    """Each sample decodes to a flat image whose value names the sample."""

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


def _collector() -> tuple[RolloutCollector, _Runtime, _RewardRuntime]:
    entry = get_model_family_entry("qwen_image_21")
    config = RolloutCollectorConfig(request_sampling={"seed": 3})
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
    return EditChain(
        chain_id,
        str(source),
        [PromptExample(prompt=step) for step in steps],
        tmp_path / "edit_chains",
    )


@pytest.mark.asyncio
async def test_each_step_conditions_on_a_drawn_sample_and_scores_against_it(tmp_path) -> None:
    collector, runtime, reward = _collector()
    chain = _chain(tmp_path, ["recolor", "letter", "sharpen"])
    stats = RolloutStats()
    random.seed(0)
    expected_draws = [random.randrange(2) for _ in range(3)]
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
    for step, (draw, reference) in enumerate(zip(expected_draws[:2], references[1:], strict=True)):
        parent = Path(reference[0])
        assert parent.parent.parent == tmp_path / "edit_chains" / "chain"
        assert parent.name == f"step{step:02d}.png"
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

    # The chain ran as one episode: its trace records the schedule and the drawn scores.
    (episode_path,) = (tmp_path / "edit_chains" / "chain").glob("*/episode.json")
    trace = json.loads(episode_path.read_text())
    assert trace["schema"] == "vrl.visual-episode.v2"
    assert trace["termination"] == "tool_budget" and trace["tool_calls"] == 3
    assert [step["decision"]["action"] for step in trace["steps"]] == [
        "step-00",
        "step-01",
        "step-02",
    ]
    # The source is not scored; each edited state carries its drawn sample's score.
    assert trace["state_scores"][0] is None
    assert [score["total"] for score in trace["state_scores"][1:]] == [
        float(expected_draws[0]),
        2.0 + expected_draws[1],
        4.0 + expected_draws[2],
    ]
    assert trace["final_score"]["total"] == 4.0 + expected_draws[2]


@pytest.mark.asyncio
async def test_two_chains_in_one_call_get_distinct_group_ids(tmp_path) -> None:
    collector, _, _ = _collector()
    batches = await collector.prepare_training_batches(
        prompts=[_chain(tmp_path, ["a", "b"], "one"), _chain(tmp_path, ["c"], "two")],
        group_size=2,
        runtime_debug=False,
        policy_version=None,
        stats=RolloutStats(),
    )
    assert [batch.group_ids.tolist() for batch in batches] == [[0, 0], [1, 1], [2, 2]]


def test_manifest_rows_resolve_source_and_accept_string_or_object_steps(tmp_path) -> None:
    Image.new("RGB", (4, 4), "white").save(tmp_path / "page.png")
    path = tmp_path / "chains.jsonl"
    rows = [
        {
            "chain_id": "comic-1",
            "source": "page.png",
            "steps": [
                "Recolor the coat blue.",
                {"prompt": "Fix the second bubble.", "target_text": "HELLO", "panel": 2},
            ],
            "metadata": {"page": "p1"},
        },
        {"source": "./page.png", "steps": ["Sharpen."]},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    chains = load_edit_chains(path, media_dir=tmp_path / "media")
    assert [chain.chain_id for chain in chains] == ["comic-1", "chains:2"]
    first = chains[0]
    assert first.source == str(tmp_path / "page.png")
    assert first.media_dir == tmp_path / "media"
    assert first.steps[0] == PromptExample(prompt="Recolor the coat blue.")
    assert first.steps[1].target_text == "HELLO"
    assert first.steps[1].metadata == {"panel": 2}
    step = first.step_example(1, "/tmp/parent.png", parent_sample_id="req:0:1")
    assert step.reference_images == ["/tmp/parent.png"]
    assert step.reward_metadata()["prompt_id"] == "comic-1:1"
    assert step.reward_metadata()["chain_source"] == first.source


@pytest.mark.parametrize(
    ("row", "match"),
    [
        ({"source": "page.png", "steps": []}, "non-empty list"),
        ({"source": "missing.png", "steps": ["a"]}, "does not exist"),
        ({"source": "page.png", "steps": ["a"], "extra": 1}, "unknown edit chain fields"),
        (
            {"source": "page.png", "steps": [{"prompt": "a", "reference_image": "x.png"}]},
            "conditioning",
        ),
        ({"source": "page.png", "steps": [{"prompt": "a", "chain_step": 3}]}, "collector-owned"),
    ],
)
def test_invalid_rows_are_rejected(tmp_path, row, match) -> None:
    Image.new("RGB", (4, 4), "white").save(tmp_path / "page.png")
    path = tmp_path / "chains.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        load_edit_chains(path, media_dir=tmp_path / "media")


def test_sample_image_accepts_batches_frames_and_references(monkeypatch) -> None:
    batch = torch.arange(2 * 3 * 1 * 2 * 2, dtype=torch.float32).reshape(2, 3, 1, 2, 2)
    assert torch.equal(sample_image(batch, 1), batch[1, :, 0])
    reference = MediaReference(object_ref="ref", sample_index=0)
    monkeypatch.setattr(MediaReference, "resolve", lambda self, cache=None: batch[0, :, 0])
    assert torch.equal(sample_image([reference], 0), batch[0, :, 0])
    with pytest.raises(ValueError, match="one image"):
        sample_image(torch.zeros(2, 6, 2, 2, 1), 0)
