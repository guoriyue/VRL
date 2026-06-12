"""Replay-export alignment contract for predict2 (sample_batch_size > 1).

The trajectory builder silently drops replay tensors whose dim-0 does not
match the sample batch (vrl/trajectory/builders.py). predict2 holds
``init_latents`` with a leading-1 dim and expands lazily in the forward, so
an unaligned export lost the key whenever sample_batch_size > 1 and replay
restore KeyError'd on ``init_latents`` (found by the OOM-split GPU gate,
2026-06-11). This pins the export contract: every replay tensor leaves the
model sample-aligned.
"""

from __future__ import annotations

import torch

from vrl.models.diffusion.cosmos.predict2.model import (
    CosmosPredict2Model,
    CosmosPredict2SamplingState,
)


def _state(batch_size: int) -> CosmosPredict2SamplingState:
    return CosmosPredict2SamplingState(
        latents=torch.randn(batch_size, 2, 3, 4, 4),
        timesteps=torch.linspace(1000.0, 0.0, 5),
        scheduler=object(),
        prompt_embeds=torch.randn(batch_size, 7, 8),
        negative_prompt_embeds=None,
        guidance_scale=7.0,
        do_cfg=True,
        # Shared conditioning tensors carry a leading-1 dim in the state and
        # are expanded lazily at forward time.
        init_latents=torch.randn(1, 2, 3, 4, 4),
        cond_mask=torch.ones(1, 1, 3, 4, 4),
        uncond_mask=torch.zeros(1, 1, 3, 4, 4),
        padding_mask=torch.zeros(1, 1, 4, 4),
        cond_indicator=torch.ones(1, 1, 3, 1, 1),
        uncond_indicator=torch.zeros(1, 1, 3, 1, 1),
        fps=16,
        seed=0,
    )


def test_replay_export_is_sample_aligned_at_batch_gt_1() -> None:
    """Checks every exported replay tensor has dim-0 == sample batch."""
    batch_size = 4
    exported = CosmosPredict2Model.export_replay_tensors(
        object.__new__(CosmosPredict2Model),
        _state(batch_size),
    )

    assert "init_latents" in exported
    for name, value in exported.items():
        if isinstance(value, torch.Tensor):
            assert value.shape[0] == batch_size, (
                f"replay tensor {name!r} is not sample-aligned "
                f"(shape {tuple(value.shape)}); the trajectory builder would "
                "silently drop it and replay restore would KeyError"
            )
