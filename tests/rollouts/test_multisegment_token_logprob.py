"""Tests for R1 multi-segment token log-prob replay."""

from __future__ import annotations

import torch

from vrl.engine import GenerationRequest, GenerationSampleSpec
from vrl.engine.trajectory import build_ar_multisegment_trajectory, build_training_view
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.evaluators.ar.multi_segment_token_logprob import (
    MultiSegmentTokenLogProbEvaluator,
)


def _sample_specs() -> list[GenerationSampleSpec]:
    return [
        GenerationSampleSpec(
            prompt_index=0,
            sample_index=0,
            prompt="draw a chart",
            prompt_id="p0",
            group_id="g0",
            sample_id="s0",
            trajectory_id="t0",
            seed=None,
        ),
        GenerationSampleSpec(
            prompt_index=0,
            sample_index=1,
            prompt="draw a chart",
            prompt_id="p0",
            group_id="g0",
            sample_id="s1",
            trajectory_id="t1",
            seed=None,
        ),
    ]


def _trajectory_segment(
    name: str,
    token_ids: torch.Tensor,
    *,
    modality: str,
    train: bool = True,
) -> dict[str, torch.Tensor | str | bool]:
    batch_size = token_ids.shape[0]
    return {
        "name": name,
        "visual": modality == "image",
        "train": train,
        "token_ids": token_ids,
        "token_log_probs": torch.zeros_like(token_ids, dtype=torch.float32),
        "token_mask": torch.ones_like(token_ids, dtype=torch.float32),
        "prompt_embeds": torch.ones(batch_size, 3, 4),
        "attention_mask": torch.ones(batch_size, 3, dtype=torch.long),
        "prompt_attention_mask": torch.ones(batch_size, 3, dtype=torch.long),
    }


def _trajectory_batch() -> RolloutBatch:
    initial_ids = torch.tensor([[1, 2], [2, 3]])
    final_ids = torch.tensor([[3, 4, 5], [4, 5, 6]])
    selfcheck_ids = torch.tensor([[7, 8], [8, 9]])
    request = GenerationRequest(
        request_id="r1",
        family="janus_pro_r1",
        task="ar_t2i_r1",
        prompts=["draw a chart"],
        samples_per_prompt=2,
        return_artifacts={"output", "trajectory"},
    )
    trajectory = build_ar_multisegment_trajectory(
        request=request,
        sample_specs=_sample_specs(),
        segments={
            "initial_image": _trajectory_segment(
                "initial_image",
                initial_ids,
                modality="image",
            ),
            "selfcheck_text": _trajectory_segment(
                "selfcheck_text",
                selfcheck_ids,
                modality="text",
                train=True,
            ),
            "final_image": _trajectory_segment(
                "final_image",
                final_ids,
                modality="image",
            ),
        },
        decoded_outputs={
            "initial_image": torch.zeros(2, 3, 4, 4),
            "final_image": torch.ones(2, 3, 4, 4),
            "selfcheck": selfcheck_ids,
        },
        primary_segment="final_image",
        context={},
    )
    return RolloutBatch(
        observations=torch.ones(2, 1, 3, dtype=torch.long),
        actions=final_ids,
        rewards=torch.zeros(2),
        dones=torch.ones(2, dtype=torch.bool),
        group_ids=torch.tensor([0, 0]),
        extras={},
        trajectory=trajectory,
        training_view=build_training_view(trajectory, primary_segment="final_image"),
    )


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


def test_evaluator_can_replay_text_segment_without_using_image_path() -> None:
    batch = _trajectory_batch()
    assert batch.trajectory is not None
    batch.trajectory.segments["selfcheck_text"].metadata["train"] = True
    model = _SegmentReplayModel()
    evaluator = MultiSegmentTokenLogProbEvaluator(
        enabled_segments=("selfcheck_text",),
    )

    signals = evaluator.evaluate(model, batch)

    assert model.calls == [("selfcheck_text", "text")]
    assert signals.segments["selfcheck_text"].log_prob.shape == (2, 2)


def test_evaluator_reads_r1_segments_from_trajectory_without_legacy_extras() -> None:
    batch = _trajectory_batch()
    model = _SegmentReplayModel()
    evaluator = MultiSegmentTokenLogProbEvaluator(
        enabled_segments=("selfcheck_text", "final_image"),
    )

    signals = evaluator.evaluate(model, batch)

    assert model.calls == [("selfcheck_text", "text"), ("final_image", "image")]
    assert signals.primary_segment == "final_image"
    assert signals.primary.log_prob.shape == (2, 3)
    assert signals.segments["selfcheck_text"].log_prob.shape == (2, 2)
    assert signals.segments["final_image"].old_log_prob.shape == (2, 3)
