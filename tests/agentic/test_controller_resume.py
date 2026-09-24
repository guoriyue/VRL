"""An interrupted episode group resumes from the last committed controller update."""

import argparse
import json
import random
from contextlib import asynccontextmanager
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image

from agentic.episode import Artifact, Decision, Episode, PolicyStamp, Score
from agentic.scripts import train_visual_controller


@pytest.mark.asyncio
async def test_collection_failure_resumes_without_duplicate_or_missing_episodes(
    tmp_path, monkeypatch, request
):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    rng, python_rng, numpy_rng = torch.get_rng_state(), random.getstate(), np.random.get_state()
    for owner, name in (
        (torch.backends.cudnn, "deterministic"),
        (torch.backends.cudnn, "benchmark"),
        (torch.backends.cuda.matmul, "fp32_precision"),
    ):
        monkeypatch.setattr(owner, name, getattr(owner, name))

    def restore_process_state():
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_rng_state(rng)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)

    request.addfinalizer(restore_process_state)
    source, target = tmp_path / "black.png", tmp_path / "white.png"
    Image.new("RGB", (32, 32), "black").save(source)
    Image.new("RGB", (32, 32), "white").save(target)
    manifest = tmp_path / "tasks.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "task_id": "white",
                "instruction": "Make white",
                "source": str(source),
                "actions": [
                    {"name": "edit", "kind": "edit", "instruction": "Make white"},
                    {"name": "stop", "kind": "stop"},
                ],
            }
        )
        + "\n"
    )

    class Model(torch.nn.Linear):
        def save_pretrained(self, path):
            path.mkdir()
            torch.save(self.state_dict(), path / "adapter.pt")

    class Controller:
        temperature = 1.0
        max_image_pixels = 1024
        observation_mode = "rgb"

        def __init__(self):
            self.base_identity = {"fixture": "categorical-resume-v1"}
            self.model = Model(1, 1, bias=False)
            torch.nn.init.zeros_(self.model.weight)
            self.policy_stamp = PolicyStamp("fixture", "v1", 0)

        async def activate(self):
            pass

        async def park(self):
            pass

        def advance_policy(self):
            self.policy_stamp = replace(self.policy_stamp, version=self.policy_stamp.version + 1)

        def restore_policy_stamp(self, policy):
            self.policy_stamp = policy

        def replay_log_prob(self, task, observation, decision):
            logits = torch.cat((self.model.weight.reshape(1), torch.zeros(1)))
            return logits.log_softmax(0)[0 if decision.action == "edit" else 1]

        async def decide(self, task, observation, *, seed):
            logits = torch.cat((self.model.weight.reshape(1), torch.zeros(1)))
            index = torch.multinomial(
                logits.detach().softmax(0), 1, generator=torch.Generator().manual_seed(seed)
            ).item()
            return Decision(
                task.actions[index].name,
                self.policy_stamp,
                logits.log_softmax(0)[index].item(),
                observation.digest(task),
                {},
            )

    class Editor:
        policy_stamp = PolicyStamp("fixture-editor", "v1", 0)

        async def activate(self):
            pass

        async def park(self):
            pass

        async def edit(self, task, observation, action, *, seed, output_dir):
            return Artifact.from_path(target)

    class Judge:
        revision = "pixels-v1"

        async def activate(self):
            pass

        async def park(self):
            pass

        async def score(self, task, artifact):
            with Image.open(artifact.path) as image:
                return Score(image.getpixel((0, 0))[0] / 255)

    @asynccontextmanager
    async def session(args, *, output):
        yield Controller(), Editor(), Judge()

    monkeypatch.setattr(train_visual_controller, "open_visual_session", session)
    args = argparse.Namespace(
        updates=2,
        episodes_per_task=4,
        max_tool_calls=2,
        resolution=32,
        steps=1,
        output_mode="rgb",
        seed=42,
        deterministic=True,
        tasks=str(manifest),
        revision="v1",
        reward_model="pixels",
        reward_revision="v1",
        tool_cost=0.1,
        gamma=0.8,
        learning_rate=0.01,
        resume=None,
        output=str(tmp_path / "control"),
    )
    await train_visual_controller.run(args)
    real_episode = Episode.run
    collected = 0

    async def failing_episode(*positional, **kwargs):
        nonlocal collected
        trace = await real_episode(*positional, **kwargs)
        collected += 1
        if collected == 6:
            raise RuntimeError("injected failure after uncommitted episode")
        return trace

    monkeypatch.setattr(Episode, "run", failing_episode)
    args.output = str(tmp_path / "interrupted")
    with pytest.raises(RuntimeError, match="injected failure"):
        await train_visual_controller.run(args)
    interrupted = tmp_path / "interrupted"
    assert (interrupted / "checkpoint-1.pt").exists()
    assert not (interrupted / "checkpoint-2.pt").exists()
    assert not (interrupted / "result.json").exists()

    monkeypatch.setattr(Episode, "run", real_episode)
    args.resume = str(interrupted / "checkpoint-1.pt")
    args.output = str(tmp_path / "resumed")
    await train_visual_controller.run(args)
    control = torch.load(tmp_path / "control/checkpoint-2.pt", weights_only=False)
    resumed = torch.load(tmp_path / "resumed/checkpoint-2.pt", weights_only=False)
    assert control["progress"] == resumed["progress"] == {"next_update": 2, "episode_cursor": 8}
    assert control["policy"] == resumed["policy"]
    assert control["policy"]["version"] == 2
    for name in control["parameters"]:
        assert torch.equal(control["parameters"][name], resumed["parameters"][name])
    for index, state in control["optimizer"]["state"].items():
        for name, value in state.items():
            assert torch.equal(value, resumed["optimizer"]["state"][index][name])
    assert sorted(path.name for path in (tmp_path / "resumed/episodes").iterdir()) == [
        "00000004",
        "00000005",
        "00000006",
        "00000007",
    ]
    for cursor in range(4, 8):
        before = json.loads((tmp_path / f"control/episodes/{cursor:08d}/episode.json").read_text())
        after = json.loads((tmp_path / f"resumed/episodes/{cursor:08d}/episode.json").read_text())
        assert before["seed"] == after["seed"]
        assert before["discounted_return"] == after["discounted_return"]
