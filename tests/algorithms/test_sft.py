"""Supervised flow matching: the token-weighted error and the timestep window."""

from __future__ import annotations

import pytest
import torch

from vrl.algorithms.sft import DiffusionSFT, DiffusionSFTConfig


def test_unweighted_loss_is_the_plain_mse() -> None:
    torch.manual_seed(0)
    pred, target = torch.randn(2, 6, 4), torch.randn(2, 6, 4)

    stats = DiffusionSFT.compute_loss(pred, target)

    torch.testing.assert_close(stats["loss"], (pred - target).square().mean())
    torch.testing.assert_close(stats["mse"], stats["loss"])


def test_token_weights_scale_each_token_error_and_leave_the_mse_unweighted() -> None:
    pred = torch.zeros(1, 4, 2)
    target = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [0.0, 0.0], [0.0, 0.0]]])
    # Mean one: the first error token counts twice, the second (error 9) not at all.
    weights = torch.tensor([[[2.0], [0.0], [1.0], [1.0]]])

    stats = DiffusionSFT.compute_loss(pred, target, weights)

    torch.testing.assert_close(stats["loss"], torch.tensor(0.5))
    torch.testing.assert_close(stats["mse"], torch.tensor(2.5))


def test_loss_rejects_a_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        DiffusionSFT.compute_loss(torch.zeros(1, 3), torch.zeros(1, 4))


@pytest.mark.parametrize("window", [(-1, 4), (4, 4), (5, 2)])
def test_timestep_window_must_be_a_forward_range(window: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="timestep_index_range"):
        DiffusionSFTConfig(timestep_index_range=window)


def test_timestep_window_is_stored_as_ints() -> None:
    assert DiffusionSFTConfig(timestep_index_range=(0.0, 48.0)).timestep_index_range == (0, 48)
    assert DiffusionSFT(DiffusionSFTConfig()).sft_weight == 0.0
