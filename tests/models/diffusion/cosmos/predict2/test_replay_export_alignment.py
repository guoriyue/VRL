"""Replay-export alignment contract across the Cosmos family.

The trajectory builder silently drops replay tensors whose dim-0 does not
match the sample batch (vrl/trajectory/builders.py). predict2 holds its
conditioning bundle (``init_latents``, masks, indicators) with a leading-1 dim
and expands lazily in the forward, so an unaligned export lost the key whenever
sample_batch_size > 1 and replay restore KeyError'd on ``init_latents`` (found
by the OOM-split GPU gate, 2026-06-11).

predict2.5 and anima share the same shared-conditioning shape: their export
already calls ``align_replay_tensor``/``_align_replay_tensor`` so production is
guarded, but nothing pinned it. This parametrizes the contract across all three
Cosmos families so the alignment cannot silently regress in any of them: every
exported replay tensor must leave the model sample-aligned.

Wan-I2V is deliberately excluded — its ``condition`` comes from
``pipe.prepare_latents(..., batch_size, ...)`` already sample-batched and its
``image_embeds`` are forced via ``_align_optional_batch`` (raises if it cannot
align), so it never had predict2's leading-1 silent-drop risk; its roundtrip is
covered by ``wan_2_1/test_backbone_parity.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from vrl.models.diffusion.cosmos.anima.model import AnimaModel, AnimaSamplingState
from vrl.models.diffusion.cosmos.predict2.model import (
    CosmosPredict2Model,
    CosmosPredict2SamplingState,
)
from vrl.models.diffusion.cosmos.predict2_5.model import (
    CosmosPredict25Model,
    CosmosPredict25SamplingState,
)


def _predict2_state(batch_size: int) -> CosmosPredict2SamplingState:
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
    )


def _predict25_state(batch_size: int) -> CosmosPredict25SamplingState:
    return CosmosPredict25SamplingState(
        latents=torch.randn(batch_size, 2, 3, 4, 4),
        timesteps=torch.linspace(1000.0, 0.0, 5),
        scheduler=object(),
        prompt_embeds=torch.randn(batch_size, 7, 8),
        negative_prompt_embeds=None,
        guidance_scale=7.0,
        do_cfg=True,
        # Shared conditioning carries a leading-1 dim, aligned lazily at export.
        cond_latent=torch.randn(1, 2, 3, 4, 4),
        cond_mask=torch.ones(1, 1, 3, 4, 4),
        cond_indicator=torch.ones(1, 1, 3, 1, 1),
        padding_mask=torch.zeros(1, 1, 4, 4),
        height=32,
        width=32,
        num_frames=3,
        fps=16,
    )


def _anima_state(batch_size: int) -> AnimaSamplingState:
    return AnimaSamplingState(
        latents=torch.randn(batch_size, 2, 3, 4, 4),
        timesteps=torch.linspace(1000.0, 0.0, 5),
        scheduler=object(),
        prompt_embeds=torch.randn(batch_size, 7, 8),
        negative_prompt_embeds=None,
        guidance_scale=7.0,
        do_cfg=True,
        # Shared conditioning carries a leading-1 dim, aligned lazily at export.
        padding_mask=torch.zeros(1, 1, 4, 4),
    )


# (family name, model class, synthetic-state builder). Each family's
# export_replay_tensors is invoked via object.__new__ (transformer-free, pure
# state plumbing) so this stays cheap and CPU-only.
_CASES: list[tuple[str, type, Callable[[int], Any]]] = [
    ("predict2", CosmosPredict2Model, _predict2_state),
    ("predict2_5", CosmosPredict25Model, _predict25_state),
    ("anima", AnimaModel, _anima_state),
]


@pytest.mark.parametrize(
    ("model_cls", "build_state"),
    [(cls, build) for _, cls, build in _CASES],
    ids=[name for name, _, _ in _CASES],
)
def test_replay_export_is_sample_aligned_at_batch_gt_1(
    model_cls: type,
    build_state: Callable[[int], Any],
) -> None:
    """Checks every exported replay tensor has dim-0 == sample batch."""
    batch_size = 4
    exported = model_cls.export_replay_tensors(
        object.__new__(model_cls),
        build_state(batch_size),
    )

    tensors = {n: v for n, v in exported.items() if isinstance(v, torch.Tensor)}
    assert tensors, "export produced no replay tensors to align"
    for name, value in tensors.items():
        assert value.shape[0] == batch_size, (
            f"replay tensor {name!r} is not sample-aligned "
            f"(shape {tuple(value.shape)}); the trajectory builder would "
            "silently drop it and replay restore would KeyError"
        )
