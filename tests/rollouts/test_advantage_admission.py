"""Admission preserves exact post-normalization row selection and auditable attempts."""

import json

import torch

from vrl.generation.types import GenerationRequest
from vrl.rollouts.admission import AdmissionLedger, select_advantage_rows
from vrl.rollouts.batch import RolloutBatch
from vrl.trajectory.builders import build_diffusion_trajectory


def test_continuous_reward_failures_and_tiny_advantages_keep_their_existing_semantics(tmp_path):
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
    batches, values, records = select_advantage_rows(
        [batch], [advantages], drop_zero_advantage=True
    )
    assert batches[0].trajectory.sample_rows == [trajectory.sample_rows[i] for i in (3, 5)]
    assert torch.equal(values[0], advantages[[3, 5]])
    assert records[0]["decision"] == "drop" and records[0]["reason"] == "zero_advantage"
    assert records[1]["decision"] == "partial" and records[1]["selected_rows"] == [
        True,
        False,
        True,
    ]
    assert records[1]["advantages"] == advantages[3:].tolist()
    assert records[1]["reward_components"]["observer/detail"] == [3, 4, 5]
    assert records[1]["failure_attribution"] == "undetermined"
    assert records[1]["rollout_policy_version"] == 7
    unchanged, full_advantages, disabled = select_advantage_rows(
        [batch],
        [advantages],
        drop_zero_advantage=False,
    )
    assert unchanged[0] is batch and full_advantages[0] is advantages
    assert all(record["reason"] == "filter_disabled" for record in disabled)

    first = AdmissionLedger(tmp_path, rank=0)
    first.record(records, trainer_step=3, global_step=2)
    first.record(disabled, trainer_step=4, global_step=2)
    lines = [json.loads(line) for line in first.path.read_text().splitlines()]
    assert [line["collection"] for line in lines] == [0, 0, 1, 1]
    assert all(line["optimizer_applied"] is None for line in lines)
    before = first.path.read_bytes()
    second = AdmissionLedger(tmp_path, rank=0)
    second.record(records, trainer_step=3, global_step=2)
    assert second.path != first.path and first.path.read_bytes() == before


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
    _, _, records = select_advantage_rows([batch], [torch.zeros(2)], drop_zero_advantage=True)
    metadata["source_group"] = "synthetic-42-2"
    metadata["rubric"]["revision"] = "v2"
    _, _, other = select_advantage_rows([batch], [torch.zeros(2)], drop_zero_advantage=True)
    assert records[0]["prompt_key"] != other[0]["prompt_key"]
    ledger = AdmissionLedger(tmp_path, rank=0)
    ledger.record(records, trainer_step=0, global_step=0)
    recorded = json.loads(ledger.path.read_text())
    assert recorded["decision"] == "drop"
    assert recorded["prompt_id"] == "circle-00001"
    assert recorded["input_metadata"] == {
        "task_id": "circle-00001",
        "source_group": "synthetic-42-1",
        "target_image": str(tmp_path / "target.png"),
        "rubric": {"revision": "v1"},
    }
    assert recorded["rollout_policy_version"] == 3
    assert "rollout_policy_version" in metadata
