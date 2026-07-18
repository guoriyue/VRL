"""Wan DPO pixel encoder must normalize latents exactly inverse to decode.

The canonical Wan decode (``decode_latents`` in
``vrl/models/wan_2_1/model.py``) denormalizes ``raw = z * std + mean``
with the VAE config's per-channel ``latents_mean`` / ``latents_std``. The DPO
``encode_pixels`` closure therefore has to produce ``z = (raw - mean) / std`` —
a dropped reciprocal (``* std``) feeds the transformer latents scaled by
``std**2`` per channel, which is out-of-distribution for the pretrained model.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from vrl.scripts.families.wan_2_1.train_dpo import _build_encoders


class _FakeVAE:
    def __init__(self, raw_latents: torch.Tensor) -> None:
        self._raw_latents = raw_latents
        self.dtype = torch.float32
        self.config = SimpleNamespace(
            z_dim=raw_latents.shape[1],
            latents_mean=[0.5, -0.25],
            latents_std=[2.0, 4.0],
        )

    def encode(self, x: torch.Tensor) -> SimpleNamespace:
        raw = self._raw_latents
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: raw))


def test_encode_pixels_normalizes_inverse_of_decode() -> None:
    raw = torch.tensor(
        [[[[[3.0]]], [[[7.0]]]], [[[[-1.0]]], [[[0.5]]]]],
    )  # [B=2, z_dim=2, T=1, H=1, W=1]
    vae = _FakeVAE(raw)
    pipeline = SimpleNamespace(vae=vae)
    encode_pixels, _ = _build_encoders(
        pipeline,
        num_frames=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    z = encode_pixels(torch.zeros(2, 3, 4, 4))

    mean = torch.tensor(vae.config.latents_mean).view(1, 2, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, 2, 1, 1, 1)
    torch.testing.assert_close(z, (raw - mean) / std)
    # Round-trip through the canonical decode denormalization recovers raw.
    torch.testing.assert_close(z * std + mean, raw)
    # False side: the historical bug multiplied by std instead of dividing.
    assert not torch.allclose(z, (raw - mean) * std)
