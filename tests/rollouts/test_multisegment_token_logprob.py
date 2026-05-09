"""Tests for R1 multi-segment token log-prob replay."""

from __future__ import annotations

import torch

from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.ar.multi_segment_token_logprob import (
    MultiSegmentTokenLogProbEvaluator,
)


def _segment(
    name: str,
    token_ids: torch.Tensor,
    *,
    modality: str,
    train: bool = True,
) -> dict[str, torch.Tensor | str | bool]:
    batch_size = token_ids.shape[0]
    return {
        "name": name,
        "modality": modality,
        "train": train,
        "token_ids": token_ids,
        "token_log_probs": torch.zeros_like(token_ids, dtype=torch.float32),
        "token_mask": torch.ones_like(token_ids, dtype=torch.float32),
        "prompt_input_ids": torch.ones(batch_size, 3, dtype=torch.long),
        "prompt_attention_mask": torch.ones(batch_size, 3, dtype=torch.long),
    }


def _batch() -> RolloutBatch:
    initial_ids = torch.tensor([[1, 2], [2, 3]])
    final_ids = torch.tensor([[3, 4, 5], [4, 5, 6]])
    selfcheck_ids = torch.tensor([[7, 8], [8, 9]])
    return RolloutBatch(
        observations=torch.ones(2, 1, 3, dtype=torch.long),
        actions=initial_ids,
        rewards=torch.zeros(2),
        dones=torch.ones(2, dtype=torch.bool),
        group_ids=torch.tensor([0, 0]),
        extras={
            "log_probs": torch.zeros(2, 1, 2),
            "r1_segments": {
                "initial_image": _segment(
                    "initial_image",
                    initial_ids,
                    modality="image",
                ),
                "selfcheck_text": _segment(
                    "selfcheck_text",
                    selfcheck_ids,
                    modality="text",
                    train=False,
                ),
                "final_image": _segment(
                    "final_image",
                    final_ids,
                    modality="image",
                ),
            },
        },
    )


class _ImageReplayModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def replay_forward(self, batch: RolloutBatch, timestep_idx: int = 0):
        del timestep_idx
        name = batch.extras["segment_name"]
        self.calls.append(name)
        token_ids = batch.actions
        logits = torch.zeros(token_ids.shape[0], token_ids.shape[1], 16)
        logits.scatter_(-1, token_ids.unsqueeze(-1), 2.0)
        return {"logits": logits}


class _SegmentReplayModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def replay_r1_segment(self, *, batch, segment_name, segment):
        del batch
        modality = str(segment["modality"])
        self.calls.append((segment_name, modality))
        token_ids = segment["token_ids"]
        logits = torch.zeros(token_ids.shape[0], token_ids.shape[1], 20)
        logits.scatter_(-1, token_ids.unsqueeze(-1), 3.0)
        return {"logits": logits}


def test_evaluator_replays_enabled_image_segments_separately() -> None:
    batch = _batch()
    model = _ImageReplayModel()
    evaluator = MultiSegmentTokenLogProbEvaluator(
        enabled_segments=("initial_image", "final_image"),
    )

    signals = evaluator.evaluate(model, batch)

    assert model.calls == ["initial_image", "final_image"]
    segment_signals = signals.aux["segments"]
    assert set(segment_signals) == {"initial_image", "final_image"}
    assert segment_signals["initial_image"].log_prob.shape == (2, 2)
    assert segment_signals["final_image"].log_prob.shape == (2, 3)
    assert signals.log_prob.shape == (2, 2)
    assert signals.aux["old_log_probs"]["final_image"].shape == (2, 3)


def test_evaluator_can_replay_text_segment_without_using_image_path() -> None:
    batch = _batch()
    batch.extras["r1_segments"]["selfcheck_text"]["train"] = True
    model = _SegmentReplayModel()
    evaluator = MultiSegmentTokenLogProbEvaluator(
        enabled_segments=("selfcheck_text",),
    )

    signals = evaluator.evaluate(model, batch)

    assert model.calls == [("selfcheck_text", "text")]
    assert signals.aux["segments"]["selfcheck_text"].log_prob.shape == (2, 2)
