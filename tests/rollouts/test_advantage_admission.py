"""Admission keeps the exact zero-advantage row mask and records every group's decision."""

import json

import torch

from vrl.generation.types import GenerationRequest
from vrl.rollouts.admission import AdmissionLedger
from vrl.rollouts.batch import RolloutBatch
from vrl.trajectory.builders import build_diffusion_trajectory


def test_zero_advantage_rows_are_dropped_and_each_attempt_writes_its_own_ledger(tmp_path):
    request = GenerationRequest("audit", "sd3_5", "t2i", ["first", "second"], 3)
    trajectory = build_diffusion_trajectory(
        request=request,
        sample_rows=request.sample_rows(),
        observations=torch.zeros(6, 2, 1),
        actions=torch.zeros(6, 2, 1),
        old_log_prob=torch.zeros(6, 2),
        timesteps=torch.zeros(6, 2),
        replay_tensors={},
        context={},
    )
    batch = RolloutBatch(
        rewards=torch.tensor([0.1, 0.2, 0.3, -1, -1, -1]),
        group_ids=torch.tensor([0, 0, 0, 1, 1, 1]),
        extras={"reward_components": {"observer/detail": torch.arange(6).float()}},
        context={"reward_metadata": {"rollout_policy_version": 7}},
        trajectory=trajectory,
    )
    advantages = torch.tensor([0, 0, 0, -1e-10, 0, 1e-10], dtype=torch.float64)
    ledger = AdmissionLedger(tmp_path, rank=0)
    batches, values = ledger.admit(
        [batch], [advantages], drop_zero_advantage=True, trainer_step=3, global_step=2
    )
    assert batches[0].trajectory.sample_rows == [trajectory.sample_rows[i] for i in (3, 5)]
    assert torch.equal(values[0], advantages[[3, 5]])
    unchanged, full = ledger.admit(
        [batch], [advantages], drop_zero_advantage=False, trainer_step=4, global_step=2
    )
    assert unchanged[0] is batch and full[0] is advantages

    lines = [json.loads(line) for line in ledger.path.read_text().splitlines()]
    assert [(line["collection"], line["trainer_step"]) for line in lines] == [
        (0, 3),
        (0, 3),
        (1, 4),
        (1, 4),
    ]
    dropped, partial = lines[:2]
    assert dropped["decision"] == "drop" and dropped["reason"] == "zero_advantage"
    assert partial["decision"] == "partial" and partial["selected_rows"] == [True, False, True]
    assert partial["advantages"] == advantages[3:].tolist()
    assert partial["reward_components"]["observer/detail"] == [3, 4, 5]
    assert partial["rollout_policy_version"] == 7
    assert all(line["reason"] == "filter_disabled" for line in lines[2:])
    before = ledger.path.read_bytes()
    second = AdmissionLedger(tmp_path, rank=0)
    second.admit([batch], [advantages], drop_zero_advantage=True, trainer_step=3, global_step=2)
    assert second.path != ledger.path and ledger.path.read_bytes() == before


def test_dropped_task_retains_source_identity_without_mutable_metadata_alias(tmp_path):
    request = GenerationRequest("source-audit", "sd3_5", "t2i", ["Extract the object"], 2)
    metadata = {
        "task_id": "circle-00001",
        "source_group": "synthetic-42-1",
        "target_image": tmp_path / "target.png",
        "rubric": {"revision": "v1"},
        "rollout_policy_version": 3,
    }
    trajectory = build_diffusion_trajectory(
        request=request,
        sample_rows=request.sample_rows(),
        observations=torch.zeros(2, 2, 1),
        actions=torch.zeros(2, 2, 1),
        old_log_prob=torch.zeros(2, 2),
        timesteps=torch.zeros(2, 2),
        replay_tensors={},
        context={},
    )
    batch = RolloutBatch(
        rewards=torch.zeros(2),
        group_ids=torch.zeros(2, dtype=torch.long),
        context={"reward_metadata": metadata},
        trajectory=trajectory,
    )
    ledger = AdmissionLedger(tmp_path, rank=0)
    ledger.admit(
        [batch], [torch.zeros(2)], drop_zero_advantage=True, trainer_step=0, global_step=0
    )
    metadata["source_group"] = "synthetic-42-2"
    metadata["rubric"]["revision"] = "v2"
    ledger.admit(
        [batch], [torch.zeros(2)], drop_zero_advantage=True, trainer_step=1, global_step=0
    )
    first, second = [json.loads(line) for line in ledger.path.read_text().splitlines()]
    assert first["prompt_key"] != second["prompt_key"]
    assert first["decision"] == "drop"
    assert first["prompt_id"] == "circle-00001"
    assert first["input_metadata"] == {
        "task_id": "circle-00001",
        "source_group": "synthetic-42-1",
        "target_image": str(tmp_path / "target.png"),
        "rubric": {"revision": "v1"},
    }
    assert first["rollout_policy_version"] == 3
    assert "rollout_policy_version" in metadata
