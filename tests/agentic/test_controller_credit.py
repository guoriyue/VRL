"""Controller learning sums causal discounted credit with a leave-one-out baseline."""

import math
from dataclasses import replace

import pytest
import torch
from PIL import Image

from agentic.episode import Action, Artifact, Decision, Episode, PolicyStamp, Score, Task
from agentic.trainer import ControllerTrainer


@pytest.mark.asyncio
async def test_variable_length_credit_matches_hand_derived_policy_gradient(tmp_path):
    source, white = tmp_path / "black.png", tmp_path / "white.png"
    Image.new("RGB", (8, 8), "black").save(source)
    Image.new("RGB", (8, 8), "white").save(white)
    task = Task(
        "white",
        "Make white",
        Artifact.from_path(source),
        (Action("edit", "edit", "Make white"), Action("stop", "stop")),
    )

    class Controller:
        temperature = 1.0
        max_image_pixels = 65536
        observation_mode = "rgb"

        def __init__(self):
            self.base_identity = {"fixture": "one-parameter-categorical-v1"}
            self.model = torch.nn.Linear(1, 1, bias=False)
            torch.nn.init.zeros_(self.model.weight)
            self.policy_stamp = PolicyStamp("fake-controller", "v1", 0)

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
            # Force contrasting fixture episodes; both likelihoods come from
            # the same real differentiable categorical distribution.
            action = "edit" if seed == 2 else "stop"
            logits = torch.cat((self.model.weight.reshape(1), torch.zeros(1)))
            decision = Decision(
                action,
                self.policy_stamp,
                0,
                observation.digest(task),
                {"action_log_probs": logits.detach().log_softmax(0).tolist()},
            )
            return replace(
                decision, old_log_prob=self.replay_log_prob(task, observation, decision).item()
            )

    class Editor:
        policy_stamp = PolicyStamp("fake-editor", "v1", 0)

        async def activate(self):
            pass

        async def park(self):
            pass

        async def edit(self, task, observation, action, *, seed, output_dir):
            return Artifact.from_path(white)

    class Judge:
        revision = "pixels-v1"

        async def activate(self):
            pass

        async def park(self):
            pass

        async def score(self, task, artifact):
            with Image.open(artifact.path) as image:
                return Score(image.getpixel((0, 0))[0] / 255)

    controller, editor, judge = Controller(), Editor(), Judge()
    traces = [
        await Episode(gamma=0.8, tool_cost=0.25).run(
            task, controller, editor, judge, output_dir=tmp_path / str(seed), seed=seed
        )
        for seed in (1, 2)
    ]
    trainer = ControllerTrainer(controller, max_grad_norm=10)
    trainer.optimizer = torch.optim.SGD(controller.model.parameters(), lr=1.0)
    metrics = await trainer.update(traces)
    # stop: advantage -.55; edit: +.55; later stop: .8 * 1.
    # Sum transition gradients and divide by TWO EPISODES, not three decisions.
    assert controller.model.weight.item() == pytest.approx(0.075, abs=1e-6)
    assert metrics["decisions"] == 3 and metrics["gradient_norm"] == pytest.approx(0.075)
    assert metrics["replay_abs_error_max"] == 0 and metrics["optimizer_stepped"]
    assert metrics["action_distribution_decisions"] == 3
    assert metrics["action_entropy_mean"] == pytest.approx(math.log(2))
    assert metrics["stop_probability_mean"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="current on-policy"):
        await trainer.update(traces)

    # Populate Adam moments, then compare a restored continuation against the
    # original optimizer on identical fresh episodes. RNG must resume too.
    trainer.optimizer = torch.optim.AdamW(controller.model.parameters(), lr=1e-4)
    fresh = [
        await Episode(gamma=0.8, tool_cost=0.25).run(
            task, controller, editor, judge, output_dir=tmp_path / f"warm-{seed}", seed=seed
        )
        for seed in (1, 2)
    ]
    for trace in fresh:
        for step in trace["steps"]:
            step["decision"]["replay"].pop("action_log_probs")
    legacy_metrics = await trainer.update(fresh)
    assert legacy_metrics["action_distribution_decisions"] == 0
    assert legacy_metrics["action_entropy_mean"] is None
    checkpoint = tmp_path / "controller.pt"
    trainer.save_checkpoint(checkpoint, contract={"task": "white"}, progress={"update": 2})
    expected_random = torch.rand(4)
    restored = ControllerTrainer(Controller(), max_grad_norm=10)
    restored.controller.base_identity = {"fixture": "different-frozen-backbone"}
    with pytest.raises(ValueError, match="different base model"):
        restored.load_checkpoint(checkpoint)
    assert restored.controller.model.weight.item() == 0
    restored.controller.base_identity = controller.base_identity.copy()
    restored.controller.observation_mode = "rgba"
    assert restored.load_checkpoint(checkpoint) == {"update": 2}
    # Controller settings follow the checkpoint instead of being re-declared.
    assert restored.controller.observation_mode == "rgb"
    assert restored.controller.policy_stamp == controller.policy_stamp
    assert torch.equal(torch.rand(4), expected_random)
    continuation = [
        await Episode(gamma=0.8, tool_cost=0.25).run(
            task, controller, editor, judge, output_dir=tmp_path / f"continue-{seed}", seed=seed
        )
        for seed in (1, 2)
    ]
    await trainer.update(continuation)
    await restored.update(continuation)
    assert torch.equal(controller.model.weight, restored.controller.model.weight)
