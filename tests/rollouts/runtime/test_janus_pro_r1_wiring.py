from __future__ import annotations

import pytest
import torch

from vrl.config.loading import load_config
from vrl.config.schema import parse_config
from vrl.generation import GenerationOutput, GenerationRequest, GenerationSampleRow
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.models.families.janus_pro.runtime import (
    JanusProR1BatchPayload,
    JanusProR1GenerationBatchGatherer,
)
from vrl.models.families.registry import get_model_family_entry
from vrl.rewards.runtime import RewardFunctionRuntime
from vrl.rollouts.collector import RolloutCollector
from vrl.rollouts.collector.batch_builder import (
    RolloutBatchBuildContext,
    TrajectoryRolloutBatchBuilder,
)
from vrl.rollouts.collector.config import RolloutCollectorConfig
from vrl.trajectory import TrajectoryResolver, build_ar_multisegment_trajectory


def _sample_rows() -> list[GenerationSampleRow]:
    return [
        GenerationSampleRow(
            prompt_index=0,
            sample_index=0,
            prompt="draw text",
            sample_id="s0",
        ),
        GenerationSampleRow(
            prompt_index=0,
            sample_index=1,
            prompt="draw text",
            sample_id="s1",
        ),
    ]


def _segment(batch: int, length: int, *, visual: bool) -> dict[str, torch.Tensor | bool]:
    return {
        "token_ids": torch.arange(batch * length, dtype=torch.long).reshape(batch, length),
        "token_log_probs": torch.zeros(batch, length),
        "token_mask": torch.ones(batch, length),
        "prompt_input_ids": torch.ones(batch, 4, dtype=torch.long),
        "prompt_attention_mask": torch.ones(batch, 4, dtype=torch.long),
        "uncond_input_ids": torch.zeros(batch, 4, dtype=torch.long),
        "uncond_attention_mask": torch.ones(batch, 4, dtype=torch.long),
        "prompt_embeds": torch.ones(batch, 4, 8),
        "attention_mask": torch.ones(batch, 4, dtype=torch.long),
        "visual": visual,
        "cfg": visual,
    }


def test_r1_train_segments_derive_from_algorithm_config() -> None:
    cfg = load_config("experiment/janus_pro/online_r1_grpo_ocr")
    cfg.algorithm.train_segments.initial_image = False
    cfg.algorithm.train_segments.selfcheck_text = True

    rollout = RolloutCollectorConfig.from_root(parse_config(cfg))

    assert rollout.train_segments == {
        "initial_image": False,
        "selfcheck_text": True,
        "final_image": True,
    }


@pytest.mark.parametrize(
    "mismatch",
    [
        "visual",
        "cfg",
        "missing_log_probs",
        "token_ids",
        "token_log_probs",
        "token_mask",
        "prompt_embeds",
        "attention_mask",
        "prompt_attention_mask",
    ],
)
def test_r1_gather_rejects_inconsistent_segment_batches(mismatch: str) -> None:
    request = GenerationRequest(
        request_id="r1",
        family="janus_pro_r1",
        task="ar_t2i_r1",
        inputs=["draw text"],
        samples_per_prompt=2,
    )
    batches = [
        JanusProR1BatchPayload(
            batch=GenerationSampleBatch(prompt_index=0, sample_start=index, sample_count=1),
            initial_image=torch.zeros(1, 3, 2, 2),
            final_image=torch.zeros(1, 3, 2, 2),
            selfcheck=torch.zeros(1, 2),
            segments={
                "initial_image": _segment(1, 3, visual=True),
                "selfcheck_text": _segment(1, 2, visual=False),
                "final_image": _segment(1, 5, visual=True),
            },
            context={},
        )
        for index in range(2)
    ]
    if mismatch == "missing_log_probs":
        # A missing first value must not discard the second batch's log-probs.
        batches[0].segments["final_image"]["token_log_probs"] = None
    elif mismatch in {"visual", "cfg"}:
        batches[1].segments["final_image"][mismatch] = False
    else:
        value = batches[1].segments["final_image"][mismatch]
        batches[1].segments["final_image"][mismatch] = value.to(torch.float64)
    with pytest.raises(ValueError, match="segment 'final_image'"):
        JanusProR1GenerationBatchGatherer().gather_batches(request, _sample_rows(), batches)


def test_r1_collector_uses_r1_task_request_and_trajectory_batch() -> None:
    """A collector built for ``janus_pro_r1`` issues requests with the R1 family and the
    registry's R1 task, carrying the reflect length through sampling.
    """
    rollout_config = RolloutCollectorConfig(
        request_sampling={
            "guidance_scale": 5.0,
            "temperature": 0.9,
            "image_token_num": 576,
            "image_size": 384,
            "max_text_length": 256,
            "max_reflect_len": 32,
            "final_image_policy": "always_generate",
        },
        train_segments={
            "initial_image": True,
            "selfcheck_text": False,
            "final_image": True,
        },
    )
    collector = RolloutCollector.from_family(
        get_model_family_entry("janus_pro_r1"),
        reward_runtime=RewardFunctionRuntime(None),
        config=rollout_config,
    )
    plan = collector.request_builder.build(["draw text"], 2)

    entry = get_model_family_entry("janus_pro_r1")
    assert plan.request.family == "janus_pro_r1"
    assert plan.request.task == entry.task
    assert plan.request.sampling["max_reflect_len"] == 32


def test_r1_trajectory_batch_keeps_segments_separate() -> None:
    """The multisegment layout keeps the three R1 segments as separate trajectory segments with
    their own token lengths; nothing is flattened into ``extras``.
    """
    batch_size = 2
    final_images = torch.zeros(batch_size, 3, 2, 2)
    request = GenerationRequest(
        request_id="r1",
        family="janus_pro_r1",
        task="ar_t2i_r1",
        inputs=["draw text"],
        samples_per_prompt=2,
    )
    segments = {
        "initial_image": _segment(batch_size, 3, visual=True),
        "selfcheck_text": _segment(batch_size, 2, visual=False),
        "final_image": _segment(batch_size, 5, visual=True),
    }
    trajectory = build_ar_multisegment_trajectory(
        request=request,
        sample_rows=_sample_rows(),
        segments=segments,
        primary_segment="final_image",
        context={"mode": "r1"},
    )
    output = GenerationOutput(
        output=final_images,
        trajectory=trajectory,
    )

    packed = TrajectoryRolloutBatchBuilder(
        output,
        RolloutBatchBuildContext(
            metadata={},
            device="cpu",
            trajectory_layout="multisegment_token",
        ),
    ).build(torch.tensor([1.0, 2.0]))

    assert "r1_segments" not in packed.extras
    assert packed.trajectory is trajectory
    assert trajectory.segments["initial_image"].tensors["token_ids"].value.shape == (batch_size, 3)
    assert trajectory.segments["selfcheck_text"].tensors["token_ids"].value.shape == (
        batch_size,
        2,
    )
    assert trajectory.segments["final_image"].tensors["token_ids"].value.shape == (batch_size, 5)
    assert "decoded" not in trajectory.segments
    assert trajectory.reward_views["image"].tensor_refs == ()
    assert trajectory.reward_views["image"].metadata == {"output_ref": "GenerationOutput.output"}
    assert "log_probs" not in packed.extras
    actions = TrajectoryResolver.from_batch(packed).role_value("final_image", "action")
    assert actions.shape == (batch_size, 5)
    assert packed.context["mode"] == "r1"
