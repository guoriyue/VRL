"""A sample's SDE log-prob does not depend on the batch it is computed in."""

from __future__ import annotations

import pytest
import torch

from vrl.math.denoise.flow_matching import per_sample_mean

pytestmark = pytest.mark.gpu


def test_per_sample_mean_is_bitwise_the_lone_samples_mean() -> None:
    torch.manual_seed(0)
    # A Qwen-Image-Layered log-prob's shape: 9 layers of 40x40 latents, 16 channels.
    values = 50 * torch.randn(4, 9, 16, 40, 40, device="cuda")
    dims = tuple(range(1, values.ndim))
    alone = torch.cat([values[i : i + 1].mean(dim=dims) for i in range(4)])

    for batch in (2, 4):
        assert torch.equal(per_sample_mean(values[:batch]), alone[:batch])
