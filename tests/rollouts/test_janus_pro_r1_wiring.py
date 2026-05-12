from __future__ import annotations

import asyncio

import torch

from vrl.engine import GenerationSampleSpec, OutputBatch
from vrl.rollouts.collector.configs import JanusProR1CollectorConfig
from vrl.rollouts.collector.factory import build_rollout_collector
from vrl.rollouts.packers.ar.r1 import ARR1RolloutPacker
from vrl.rollouts.packers.base import RolloutPackContext


def _sample_specs() -> list[GenerationSampleSpec]:
    return [
        GenerationSampleSpec(
            prompt_index=0,
            sample_index=0,
            prompt="draw text",
            prompt_id="p0",
            group_id="g0",
            sample_id="s0",
            trajectory_id="t0",
            seed=None,
        ),
        GenerationSampleSpec(
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
        "visual": visual,
        "cfg": visual,
    }


def test_r1_collector_uses_r1_task_request_and_packer() -> None:
    config = JanusProR1CollectorConfig(max_reflect_len=32)
    collector = build_rollout_collector(
        "janus_pro_r1",
        model=None,
        reward_fn=None,
        config=config,
    )
    plan = collector.request_builder.build(["draw text"], 2, {})

    assert collector.family == "janus_pro_r1"
    assert collector.task == "ar_t2i_r1"
    assert isinstance(collector.packer, ARR1RolloutPacker)
    assert plan.request.family == "janus_pro_r1"
    assert plan.request.task == "ar_t2i_r1"
    assert plan.request.sampling["max_reflect_len"] == 32
    assert "r1_segments" in plan.request.return_artifacts


def test_r1_packer_keeps_segments_separate() -> None:
    batch_size = 2
    final_images = torch.zeros(batch_size, 3, 2, 2)
    output = OutputBatch(
        request_id="r1",
        family="janus_pro",
        task="ar_t2i_r1",
        prompts=["draw text"],
        sample_specs=_sample_specs(),
        output=final_images,
        extra={
            "final_image": final_images,
            "r1_segments": {
                "initial_image": _segment(batch_size, 3, visual=True),
                "selfcheck_text": _segment(batch_size, 2, visual=False),
                "final_image": _segment(batch_size, 5, visual=True),
            },
            "context": {"mode": "r1"},
        },
    )

    packer = ARR1RolloutPacker()
    packed = asyncio.run(
        packer.pack(
            output,
            torch.tensor([1.0, 2.0]),
            RolloutPackContext(metadata={}, device="cpu", rescale_to_unit=True),
        ),
    )

    segments = packed.extras["r1_segments"]
    assert segments["initial_image"]["token_ids"].shape == (batch_size, 3)
    assert segments["selfcheck_text"]["token_ids"].shape == (batch_size, 2)
    assert segments["final_image"]["token_ids"].shape == (batch_size, 5)
    assert packed.extras["log_probs"].shape == (batch_size, 1, 5)
    assert packed.actions.shape == (batch_size, 5)
    assert packed.context["mode"] == "r1"
