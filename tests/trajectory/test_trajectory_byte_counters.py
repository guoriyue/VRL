"""Tests for trajectory tensor byte counters."""

from __future__ import annotations

import torch

from vrl.generation import GenerationRequest, GenerationSampleRow
from vrl.trajectory.builders import build_diffusion_trajectory
from vrl.trajectory.storage import trajectory_tensor_bytes


def test_byte_counter_counts_trajectory_tensor_leaves() -> None:
    """``trajectory_tensor_bytes`` sums ``numel * element_size`` over every tensor leaf of the
    trajectory.
    """
    observations = torch.zeros(1, 2, 4, dtype=torch.float32)
    actions = torch.ones(1, 2, 4, dtype=torch.float32)
    old_log_prob = torch.zeros(1, 2, dtype=torch.float32)
    timesteps = torch.zeros(1, 2, dtype=torch.float32)
    kl = torch.zeros(1, 2, dtype=torch.float32)
    prompt_ids = torch.ones(1, 3, dtype=torch.long)
    trajectory = build_diffusion_trajectory(
        request=GenerationRequest(
            request_id="req",
            family="sd3_5",
            task="t2i",
            inputs=["p"],
            samples_per_prompt=1,
        ),
        sample_rows=[
            GenerationSampleRow(
                prompt_index=0,
                sample_index=0,
                prompt="p",
                sample_id="s0",
            )
        ],
        observations=observations,
        actions=actions,
        old_log_prob=old_log_prob,
        timesteps=timesteps,
        kl=kl,
        replay_tensors={"prompt_ids": prompt_ids},
        context={},
    )

    # The builder adds one mask leaf shaped like old_log_prob.
    expected = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            observations,
            actions,
            old_log_prob,
            old_log_prob,
            timesteps,
            kl,
            prompt_ids,
        )
    )

    assert trajectory_tensor_bytes(trajectory) == expected


def test_byte_counter_deduplicates_shared_references() -> None:
    """The same tensor referenced twice is counted once."""
    tensor = torch.zeros(4, dtype=torch.float32)

    assert trajectory_tensor_bytes({"a": tensor, "b": tensor}) == 16
