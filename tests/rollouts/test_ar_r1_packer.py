"""Tests for Janus-Pro-R1 multi-segment rollout packing."""

from __future__ import annotations

import pytest
import torch

from vrl.engine import GenerationSampleSpec, OutputBatch
from vrl.rollouts.packers.ar_r1 import ARR1RolloutPacker
from vrl.rollouts.packers.base import RolloutPackContext


def _spec(prompt_index: int, sample_index: int, prompt: str) -> GenerationSampleSpec:
    return GenerationSampleSpec(
        prompt_index=prompt_index,
        sample_index=sample_index,
        prompt=prompt,
        prompt_id=f"p{prompt_index}",
        group_id=f"g{prompt_index}",
        sample_id=f"s{prompt_index}-{sample_index}",
        trajectory_id=f"t{prompt_index}-{sample_index}",
        seed=None,
    )


def _segment(
    name: str,
    *,
    batch_size: int,
    token_count: int,
    modality: str,
    train: bool = True,
) -> dict[str, torch.Tensor | str | bool]:
    return {
        "name": name,
        "modality": modality,
        "train": train,
        "token_ids": torch.arange(batch_size * token_count, dtype=torch.long).view(
            batch_size,
            token_count,
        ),
        "token_log_probs": torch.full((batch_size, token_count), -0.5),
        "token_mask": torch.ones(batch_size, token_count),
        "prompt_input_ids": torch.ones(batch_size, 3, dtype=torch.long),
        "prompt_attention_mask": torch.ones(batch_size, 3, dtype=torch.long),
        "uncond_input_ids": torch.zeros(batch_size, 3, dtype=torch.long),
        "uncond_attention_mask": torch.ones(batch_size, 3, dtype=torch.long),
    }


def _output() -> OutputBatch:
    batch_size = 2
    final_image = torch.ones(batch_size, 3, 4, 4)
    return OutputBatch(
        request_id="r1",
        family="janus_pro",
        task="ar_t2i_r1",
        prompts=["a"],
        sample_specs=[_spec(0, 0, "a"), _spec(0, 1, "a")],
        output=torch.zeros_like(final_image),
        extra={
            "initial_image": torch.zeros(batch_size, 3, 4, 4),
            "final_image": final_image,
            "selfcheck": torch.tensor([[1, 2], [3, 4]]),
            "segments": {
                "initial_image": _segment(
                    "initial_image",
                    batch_size=batch_size,
                    token_count=4,
                    modality="image",
                ),
                "selfcheck_text": _segment(
                    "selfcheck_text",
                    batch_size=batch_size,
                    token_count=2,
                    modality="text",
                    train=False,
                ),
                "final_image": _segment(
                    "final_image",
                    batch_size=batch_size,
                    token_count=5,
                    modality="image",
                ),
            },
            "context": {"cfg_weight": 5.0},
        },
    )


def test_reward_outputs_default_to_final_image() -> None:
    output = _output()
    packer = ARR1RolloutPacker()

    reward_outputs = packer.reward_outputs(output, RolloutPackContext(metadata={}))

    assert torch.equal(reward_outputs, output.extra["final_image"])
    assert not torch.equal(reward_outputs, output.output)


@pytest.mark.asyncio
async def test_pack_preserves_r1_segments_without_concatenating_tokens() -> None:
    output = _output()
    packer = ARR1RolloutPacker()

    batch = await packer.pack(
        output,
        rewards_raw=torch.tensor([1.0, 2.0]),
        context=RolloutPackContext(metadata={}),
    )

    segments = batch.extras["r1_segments"]
    assert set(segments) == {"initial_image", "selfcheck_text", "final_image"}
    assert segments["initial_image"]["token_ids"].shape == (2, 4)
    assert segments["selfcheck_text"]["token_ids"].shape == (2, 2)
    assert segments["final_image"]["token_ids"].shape == (2, 5)
    assert batch.actions.shape == (2, 5)
    assert batch.extras["log_probs"].shape == (2, 1, 5)
    assert batch.extras["primary_segment"] == "final_image"
    assert torch.equal(batch.videos[:, :, 0], output.extra["final_image"])
    assert batch.extras["selfcheck"] is output.extra["selfcheck"]
