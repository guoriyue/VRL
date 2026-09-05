"""CPU contract test for Anima clean-target latent encoding."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from vrl.models.families.cosmos.anima.model import AnimaModel


class _IdentityVAE:
    dtype = torch.float32

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.config = SimpleNamespace(
            latents_mean=mean.flatten().tolist(),
            latents_std=std.flatten().tolist(),
            z_dim=int(mean.shape[1]),
        )

    def encode(self, video: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: video))


def test_encode_video_to_latents_inverts_anima_decode_normalization() -> None:
    mean = torch.tensor([0.1, -0.2, 0.3]).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.5, 2.0, 4.0]).view(1, 3, 1, 1, 1)
    scheduler = SimpleNamespace(config=SimpleNamespace(sigma_data=1.0))
    model = AnimaModel(
        transformer=nn.Identity(),
        text_encoder=None,
        llm_adapter=nn.Identity(),
        vae=_IdentityVAE(mean, std),
        scheduler=scheduler,
        image_processor=None,
        qwen_tokenizer=None,
        t5_tokenizer=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    image = torch.rand(2, 3, 1, 4, 4)

    latents = model.encode_video_to_latents(image)

    reconstructed_vae_input = latents * std + mean
    torch.testing.assert_close(reconstructed_vae_input, image * 2.0 - 1.0)
