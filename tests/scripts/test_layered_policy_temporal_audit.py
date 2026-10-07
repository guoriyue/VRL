"""Saved-trajectory audits weight the clipped policy change but never the KL."""

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from vrl.scripts.eval import layered_policy_audit as audit


def test_temporal_weight_changes_policy_only(tmp_path, monkeypatch):
    update = tmp_path / "update01"
    condition = tmp_path / "conditions/fixture"
    rollout = tmp_path / "rollouts/fixture/candidate00"
    for directory in (update, condition, rollout):
        directory.mkdir(parents=True)
    upstream = tmp_path / "upstream.py"
    upstream.write_text("# Fixture upstream\n")
    plan = {
        "upstream": str(upstream),
        "upstream_sha256": hashlib.sha256(upstream.read_bytes()).hexdigest(),
        "base_path": "unused",
        "noise_level": 0.2,
        "cases": [{"name": "fixture", "split": "train", "sha256": "source"}],
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    torch.save({"weight": torch.tensor([[0.2]])}, update / "adapter.pt")
    (update / "result.json").write_text(
        json.dumps(
            {"adapter_sha256": hashlib.sha256((update / "adapter.pt").read_bytes()).hexdigest()}
        )
    )
    torch.save({"source_sha256": "source"}, condition / "condition.pt")
    torch.save(
        {
            "noise_level": 0.2,
            "sigmas": torch.tensor([0.8, 0.4, 0.0]),
            "timesteps": torch.tensor([800.0]),
            "observations": [torch.zeros(1, 2)],
            "actions": [torch.full((1, 2), 0.1)],
            "old_velocity": [torch.zeros(1, 2)],
        },
        rollout / "trajectory.pt",
    )
    (update / "replay.json").write_text(
        json.dumps(
            [
                {
                    "case": "fixture",
                    "candidate": "candidate00",
                    "step": 0,
                    "advantage": 1.0,
                    "temporal_weight": 3.0,
                }
            ]
        )
    )
    # Substitute only model loading and device placement; use the real Gaussian
    # transition density, clipping, and KL calculation on the saved CPU tensors.
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    monkeypatch.setattr(audit, "load_upstream", lambda path: None)
    monkeypatch.setattr(audit, "load_transformer", lambda plan: torch.nn.Linear(1, 1, bias=False))
    monkeypatch.setattr(
        audit,
        "predict_velocity",
        lambda model, latent, timestep, condition, upstream: model.weight.expand_as(latent),
    )
    monkeypatch.setattr(
        FlowMatchEulerDiscreteScheduler,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    records = []
    for rectified in (False, True):
        (update / "config.json").write_text(
            json.dumps(
                {
                    "timesteps": [0],
                    "clip_ratio": 0.2,
                    "kl_coef": 0.1,
                    "temporal_rectification": rectified,
                }
            )
        )
        audit.audit_likelihood(tmp_path)
        records.append(json.loads((update / "policy_audit.json").read_text()))
    plain, rectified = records
    policy = plain["mean_clipped_policy_loss_change"]
    kl = plain["mean_gaussian_kl"]
    assert abs(policy) > 1e-6 and kl > 0
    assert rectified["mean_clipped_policy_loss_change"] == pytest.approx(3 * policy)
    assert rectified["mean_gaussian_kl"] == pytest.approx(kl)
    assert rectified["mean_regularized_loss_change"] == pytest.approx(3 * policy + 0.1 * kl)
    assert rectified["rows"][0]["temporal_weight"] == 3.0
