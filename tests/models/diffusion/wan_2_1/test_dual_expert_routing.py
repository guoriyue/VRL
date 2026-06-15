"""Wan 2.2 two-stage MoE expert routing.

The only correctness-critical piece that does not need a GPU/model: the
deterministic expert dispatch must match diffusers ``pipeline_wan_i2v.py``
(``t >= boundary_timestep`` → high-noise ``transformer``; below → low-noise
``transformer_2``). If this drifts, rollout and replay route differently and
GRPO log-probs silently diverge.
"""

from __future__ import annotations

import torch

from vrl.models.diffusion.wan_2_1.model import (
    WanI2VReplayModel,
    _wan_boundary_timestep,
)

# Sentinels stand in for the two expert transformers (never called here).
_HIGH = object()
_LOW = object()


def _dual(boundary: float = 900.0) -> WanI2VReplayModel:
    return WanI2VReplayModel(
        transformer=_HIGH,
        scheduler=None,
        transformer_2=_LOW,
        boundary_timestep=boundary,
    )


def test_boundary_timestep_matches_diffusers_formula() -> None:
    # boundary_timestep = boundary_ratio * num_train_timesteps; Wan2.2 ckpt = 0.9 * 1000.
    assert _wan_boundary_timestep(0.9, 1000) == 900.0
    assert _wan_boundary_timestep(0.5, 1000) == 500.0
    # Single-tower Wan 2.1: no boundary.
    assert _wan_boundary_timestep(None, 1000) is None


def test_dual_expert_routing_at_and_around_boundary() -> None:
    m = _dual(900.0)
    # t >= boundary -> high-noise transformer (note: boundary itself is inclusive).
    assert m._expert_for_timestep(torch.tensor(1000.0)) is _HIGH
    assert m._expert_for_timestep(torch.tensor(900.0)) is _HIGH
    # t < boundary -> low-noise transformer_2.
    assert m._expert_for_timestep(torch.tensor(899.999)) is _LOW
    assert m._expert_for_timestep(torch.tensor(0.0)) is _LOW


def test_routing_accepts_batched_and_scalar_timesteps() -> None:
    m = _dual(900.0)
    # All samples in one denoise step share the same timestep.
    assert m._expert_for_timestep(torch.tensor([950.0, 950.0, 950.0])) is _HIGH
    assert m._expert_for_timestep(torch.tensor([10.0, 10.0])) is _LOW
    # Python scalar timestep (no .reshape).
    assert m._expert_for_timestep(950.0) is _HIGH
    assert m._expert_for_timestep(10.0) is _LOW


def test_single_tower_wan21_always_uses_transformer() -> None:
    m = WanI2VReplayModel(transformer=_HIGH, scheduler=None)  # no second expert / boundary
    assert m._expert_for_timestep(torch.tensor(999.0)) is _HIGH
    assert m._expert_for_timestep(torch.tensor(0.0)) is _HIGH


def test_trainable_modules_reflect_expert_count() -> None:
    assert _dual().trainable_modules == {"transformer": _HIGH, "transformer_2": _LOW}
    single = WanI2VReplayModel(transformer=_HIGH, scheduler=None)
    assert single.trainable_modules == {"transformer": _HIGH}
