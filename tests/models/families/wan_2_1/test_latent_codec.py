"""CPU contract tests for Wan SFT target latent encoding."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from vrl.models.families.wan_2_1.model import WanT2VDiffusersModel


class _IdentityVAE:
    dtype = torch.float32

    def __init__(self, mean: list[float], std: list[float]) -> None:
        self.config = SimpleNamespace(
            latents_mean=mean,
            latents_std=std,
            z_dim=len(mean),
        )

    def encode(self, video: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: video))


def test_encode_video_to_latents_inverts_wan_decode_normalization() -> None:
    mean = [0.1, -0.2, 0.3]
    std = [0.5, 2.0, 4.0]
    pipeline = SimpleNamespace(
        transformer=nn.Identity(),
        transformer_2=None,
        config=SimpleNamespace(boundary_ratio=None),
        vae=_IdentityVAE(mean, std),
    )
    model = WanT2VDiffusersModel(pipeline=pipeline, device=torch.device("cpu"))
    video = torch.rand(2, 3, 2, 4, 4)

    latents = model.encode_video_to_latents(video)

    mean_tensor = torch.tensor(mean).view(1, 3, 1, 1, 1)
    std_tensor = torch.tensor(std).view(1, 3, 1, 1, 1)
    reconstructed_vae_input = latents * std_tensor + mean_tensor
    torch.testing.assert_close(reconstructed_vae_input, video * 2.0 - 1.0)
