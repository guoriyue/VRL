"""Repeated PPO updates retain the original behavior density and reference policy."""

import hashlib
import json
from types import SimpleNamespace

import torch

from vrl.math.denoise.flow_matching import sde_step_with_logprob
from vrl.scripts.generation import qwen_layered_rl_update as runner


def test_multiple_updates_keep_frozen_behavior_and_reload_exactly(tmp_path, monkeypatch):
    from diffusers import FlowMatchEulerDiscreteScheduler

    # The runner uses CUDA placement; this CPU test exercises the real SDE loss,
    # optimizer and checkpoint logic with a scalar policy instead of Qwen.
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    scheduler = SimpleNamespace(sigmas=torch.tensor([0.8, 0.6]), timesteps=torch.tensor([800.0]))
    monkeypatch.setattr(
        FlowMatchEulerDiscreteScheduler, "from_pretrained", lambda *args, **kwargs: scheduler
    )
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.zero_()
    monkeypatch.setattr(runner, "load_transformer", lambda plan: model)
    monkeypatch.setattr(runner, "load_upstream", lambda path: None)
    monkeypatch.setattr(
        runner,
        "predict_velocity",
        lambda model, latent, timestep, condition, upstream: (
            model.weight.reshape(1, 1, 1).expand_as(latent).clone()
        ),
    )
    upstream = tmp_path / "upstream.py"
    upstream.write_text("# Frozen test upstream\n")
    plan = {
        "cases": [{"name": "room", "split": "train", "sha256": "source"}],
        "steps": 1,
        "noise_level": 0.2,
        "adapter_seed": 7,
        "base_path": "unused",
        "upstream": str(upstream),
        "upstream_sha256": hashlib.sha256(upstream.read_bytes()).hexdigest(),
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    torch.save({"weight": model.weight.detach().clone()}, tmp_path / "initial_adapter.pt")
    condition = tmp_path / "conditions/room"
    condition.mkdir(parents=True)
    torch.save({"source_sha256": "source"}, condition / "condition.pt")
    rows = []
    for index, action_value in enumerate([-0.1, 0.1]):
        directory = tmp_path / "rollouts/room" / f"candidate{index:02d}"
        directory.mkdir(parents=True)
        observation = torch.zeros(1, 2, 2)
        velocity = torch.zeros_like(observation)
        action = torch.full_like(observation, action_value)
        transition = sde_step_with_logprob(
            scheduler,
            velocity,
            scheduler.timesteps[:1],
            observation,
            prev_sample=action,
            noise_level=0.2,
            step_index=0,
            return_dt=True,
        )
        torch.save(
            {
                "observations": observation[None],
                "actions": action[None],
                "old_velocity": velocity[None],
                "old_log_density_mean": transition.log_prob[None],
                "sigmas": scheduler.sigmas,
                "timesteps": scheduler.timesteps,
                "noise_level": 0.2,
                "condition_case": "room",
                "initial_adapter_seed": 7,
            },
            directory / "trajectory.pt",
        )
        view = directory / "review.txt"
        view.write_text(f"Frozen toy candidate {index}")
        rows.append(
            {
                "kind": "candidate",
                "valid": True,
                "case": "room",
                "candidate": directory.name,
                "reward": float(index),
                "view": str(view),
                "view_sha256": hashlib.sha256(view.read_bytes()).hexdigest(),
            }
        )
    rewards = tmp_path / "reward_calibration"
    rewards.mkdir()
    scores = rewards / "scores.json"
    scores.write_text(json.dumps({"status": "complete", "rows": rows}))
    (rewards / "validation.json").write_text(
        json.dumps(
            {
                "usable_for_small_update": True,
                "scores_sha256": hashlib.sha256(scores.read_bytes()).hexdigest(),
            }
        )
    )
    runner.run_update(
        tmp_path,
        {
            "ppo_epochs": 3,
            "learning_rate": 0.01,
            "clip_ratio": 0.2,
            "kl_coef": 0.01,
            "max_grad_norm": 1.0,
            "timesteps": [0],
        },
    )
    output = tmp_path / "update01"
    replay = json.loads((output / "replay.json").read_text())
    assert all(row["preupdate_logprob_error"] == 0 for row in replay[:2])
    assert any(abs(row["logprob_difference_from_behavior"]) > 1e-5 for row in replay[2:])
    assert all(row["kl_penalty"] > 0 for row in replay[2:])
    state = torch.load(output / "optimizer.pt", weights_only=True)
    assert state["state"][0]["step"].item() == 3
    initial = torch.load(tmp_path / "initial_adapter.pt", weights_only=True)["weight"]
    first = torch.load(output / "epoch01_adapter.pt", weights_only=True)["weight"]
    final = torch.load(output / "adapter.pt", weights_only=True)["weight"]
    assert torch.count_nonzero(initial) == 0
    assert not torch.equal(first, final)
    assert torch.equal(model.weight.detach(), final)
