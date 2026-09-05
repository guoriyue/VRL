from __future__ import annotations

import pytest
import torch

from vrl.trainers.checkpointing import capture_rng_state, restore_rng_state
from vrl.trainers.data.prompt_sampler import PromptBatchSampler


def test_prompt_batch_sampler_uses_configured_strategy() -> None:
    sequential = PromptBatchSampler(
        generator=torch.Generator().manual_seed(0),
        num_examples=5,
        prompts_per_rank=3,
        strategy="sequential_window",
    ).sample(epoch=1)
    random_batch = PromptBatchSampler(
        generator=torch.Generator().manual_seed(0),
        num_examples=5,
        prompts_per_rank=3,
        strategy="random_without_replacement",
    ).sample()

    assert sequential == [3, 4, 0]
    assert len(random_batch) == 3
    assert len(set(random_batch)) == 3


def test_distributed_ranks_slice_one_global_draw() -> None:
    samplers = [
        PromptBatchSampler(
            generator=torch.Generator().manual_seed(11),
            num_examples=8,
            prompts_per_rank=2,
            num_replicas=2,
            rank=rank,
            strategy="random_without_replacement",
        )
        for rank in range(2)
    ]

    rank_batches = [sampler.sample() for sampler in samplers]
    expected_global_batch = torch.randperm(
        8,
        generator=torch.Generator().manual_seed(11),
    )[:4].tolist()

    assert rank_batches[0] + rank_batches[1] == expected_global_batch
    assert set(rank_batches[0]).isdisjoint(rank_batches[1])


@pytest.mark.parametrize(
    ("strategy", "epoch"),
    [
        ("random_without_replacement", 0),
        ("sequential_window", 2),
    ],
)
def test_preview_matches_next_sample_without_advancing_rng(
    strategy: str,
    epoch: int,
) -> None:
    generator = torch.Generator().manual_seed(17)
    sampler = PromptBatchSampler(
        generator=generator,
        num_examples=10,
        prompts_per_rank=2,
        num_replicas=2,
        rank=1,
        strategy=strategy,
    )
    state_before = generator.get_state().clone()

    preview = sampler.preview(epoch=epoch)

    assert torch.equal(generator.get_state(), state_before)
    assert sampler.sample(epoch=epoch) == preview


def test_checkpointed_generator_resumes_at_previewed_batch() -> None:
    generator = torch.Generator().manual_seed(23)
    sampler = PromptBatchSampler(
        generator=generator,
        num_examples=12,
        prompts_per_rank=3,
        num_replicas=2,
        rank=1,
        strategy="random_without_replacement",
    )
    sampler.sample(epoch=0)
    next_batch = sampler.preview(epoch=1)
    checkpoint_rng = capture_rng_state(prompt_generator=generator)

    restored_generator = torch.Generator().manual_seed(999)
    restore_rng_state(checkpoint_rng, prompt_generator=restored_generator)
    resumed_sampler = PromptBatchSampler(
        generator=restored_generator,
        num_examples=12,
        prompts_per_rank=3,
        num_replicas=2,
        rank=1,
        strategy="random_without_replacement",
    )

    assert resumed_sampler.sample(epoch=1) == next_batch


def test_random_without_replacement_rejects_a_short_global_batch() -> None:
    with pytest.raises(ValueError, match="global batch slot"):
        PromptBatchSampler(
            generator=torch.Generator().manual_seed(0),
            num_examples=3,
            prompts_per_rank=2,
            num_replicas=2,
            rank=0,
            strategy="random_without_replacement",
        )


@pytest.mark.parametrize(
    ("num_replicas", "rank"),
    [(0, 0), (2, 2)],
)
def test_prompt_batch_sampler_rejects_invalid_rank_geometry(
    num_replicas: int,
    rank: int,
) -> None:
    with pytest.raises(ValueError):
        PromptBatchSampler(
            generator=torch.Generator().manual_seed(0),
            num_examples=4,
            prompts_per_rank=1,
            num_replicas=num_replicas,
            rank=rank,
            strategy="sequential_window",
        )
