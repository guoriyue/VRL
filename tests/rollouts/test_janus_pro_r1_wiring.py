from __future__ import annotations

import torch

from vrl.engine import GenerationRequest, GenerationSampleRow, OutputBatch
from vrl.engine.trajectory import build_ar_multisegment_trajectory
from vrl.rollouts.collector.batch_builder import (
    RolloutBatchBuildContext,
    TrajectoryRolloutBatchBuilder,
)
from vrl.rollouts.collector.factory import build_rollout_collector
from vrl.rollouts.settings import RolloutSettings


def _sample_rows() -> list[GenerationSampleRow]:
    return [
        GenerationSampleRow(
            prompt_index=0,
            sample_index=0,
            prompt="draw text",
            prompt_id="p0",
            group_id="g0",
            sample_id="s0",
            trajectory_id="t0",
            seed=None,
        ),
        GenerationSampleRow(
            prompt_index=0,
            sample_index=1,
            prompt="draw text",
            prompt_id="p0",
            group_id="g0",
            sample_id="s1",
            trajectory_id="t1",
            seed=None,
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


def test_r1_collector_uses_r1_task_request_and_trajectory_batch() -> None:
    settings = RolloutSettings(
        family="janus_pro_r1",
        values={
            "n_samples_per_prompt": 2,
            "cfg_weight": 5.0,
            "temperature": 0.9,
            "image_token_num": 576,
            "image_size": 384,
            "rescale_to_unit": True,
            "max_text_length": 256,
            "max_reflect_len": 32,
            "final_image_policy": "always_generate",
            "train_segments": {
                "initial_image": True,
                "selfcheck_text": False,
                "final_image": True,
            },
        },
    )
    collector = build_rollout_collector(
        "janus_pro_r1",
        model=None,
        reward_fn=None,
        config=settings,
    )
    plan = collector.request_builder.build(["draw text"], 2, {})

    assert collector.family == "janus_pro_r1"
    assert collector.task == "ar_t2i_r1"
    assert plan.request.family == "janus_pro_r1"
    assert plan.request.task == "ar_t2i_r1"
    assert plan.request.sampling["max_reflect_len"] == 32
    assert "trajectory" in plan.request.return_artifacts


def test_r1_trajectory_batch_keeps_segments_separate() -> None:
    batch_size = 2
    initial_images = torch.ones(batch_size, 3, 2, 2)
    final_images = torch.zeros(batch_size, 3, 2, 2)
    selfcheck = torch.ones(batch_size, 2, dtype=torch.long)
    request = GenerationRequest(
        request_id="r1",
        family="janus_pro_r1",
        task="ar_t2i_r1",
        prompts=["draw text"],
        samples_per_prompt=2,
        return_artifacts={"output", "trajectory"},
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
        decoded_outputs={
            "initial_image": initial_images,
            "final_image": final_images,
            "selfcheck": selfcheck,
        },
        primary_segment="final_image",
        context={"mode": "r1"},
    )
    output = OutputBatch(
        request_id=request.request_id,
        family=request.family,
        task=request.task,
        prompts=list(request.prompts),
        sample_rows=_sample_rows(),
        output=final_images,
        trajectory=trajectory,
        extra={},
    )

    packed = TrajectoryRolloutBatchBuilder(
        output,
        RolloutBatchBuildContext(metadata={}, device="cpu", rescale_to_unit=True),
    ).build(torch.tensor([1.0, 2.0]))

    assert "r1_segments" not in packed.extras
    assert packed.trajectory is trajectory
    assert trajectory.segments["initial_image"].tensors["token_ids"].value.shape == (batch_size, 3)
    assert trajectory.segments["selfcheck_text"].tensors["token_ids"].value.shape == (batch_size, 2)
    assert trajectory.segments["final_image"].tensors["token_ids"].value.shape == (batch_size, 5)
    assert "log_probs" not in packed.extras
    assert packed.actions.shape == (batch_size, 5)
    assert packed.context["mode"] == "r1"
