"""CPU contract tests for Predict2.5 SFT target latent encoding."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from vrl.models.families.cosmos.predict2_5.model import CosmosPredict25Model


class _IdentityVAE:
    dtype = torch.float32

    def encode(self, video: torch.Tensor) -> SimpleNamespace:
        latent_dist = SimpleNamespace(mode=lambda: video)
        return SimpleNamespace(latent_dist=latent_dist)


def test_encode_video_to_latents_inverts_decode_normalization() -> None:
    mean = torch.tensor([0.1, -0.2, 0.3]).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.5, 2.0, 4.0]).view(1, 3, 1, 1, 1)
    pipeline = SimpleNamespace(
        transformer=nn.Identity(),
        vae=_IdentityVAE(),
        latents_mean=mean,
        latents_std=std,
        device=torch.device("cpu"),
    )
    model = CosmosPredict25Model(pipeline=pipeline, device=torch.device("cpu"))
    video = torch.rand(2, 3, 2, 4, 4)

    latents = model.encode_video_to_latents(video)

    reconstructed_vae_input = latents * std + mean
    torch.testing.assert_close(reconstructed_vae_input, video * 2.0 - 1.0)
