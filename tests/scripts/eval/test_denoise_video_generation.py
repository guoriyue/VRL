"""Tests for family-neutral deterministic video generation helpers."""

from __future__ import annotations

import torch

from vrl.scripts.eval.denoise_video_generation import seed_for, video_to_cthw


def test_seed_formula_lays_out_a_dense_non_colliding_grid() -> None:
    """``base_seed + prompt*samples_per_prompt + sample`` fills the grid."""

    grid = [
        seed_for(
            base_seed=17,
            prompt_index=prompt_index,
            sample_index=sample_index,
            samples_per_prompt=4,
        )
        for prompt_index in range(3)
        for sample_index in range(4)
    ]

    assert grid == list(range(17, 29))


def test_video_to_cthw_accepts_btchw_layout() -> None:
    video = torch.zeros(1, 5, 3, 8, 8)
    video[:, :, 0] = 1.0

    out = video_to_cthw(video)

    assert tuple(out.shape) == (3, 5, 8, 8)
    assert torch.all(out[0] == 1.0)
