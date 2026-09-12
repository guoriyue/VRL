"""Wan DPO pixel encoder must normalize latents exactly inverse to decode.

The canonical Wan decode (``decode_latents`` in
``vrl/models/wan_2_1/model.py``) denormalizes ``raw = z * std + mean``
with the VAE config's per-channel ``latents_mean`` / ``latents_std``. The DPO
``encode_pixels`` method therefore has to produce ``z = (raw - mean) / std`` —
a dropped reciprocal (``* std``) feeds the transformer latents scaled by
``std**2`` per channel, which is out-of-distribution for the pretrained model.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tests.models.steps.denoise.fixtures import build_tiny_wan_vae
from vrl.scripts.families.wan_2_1.train_dpo import WanDPOEncoders


def _wan_vae_with_fixed_raw(raw_latents: torch.Tensor) -> object:
    """A real ``AutoencoderKLWan`` whose ``encode`` returns a known raw latent.

    The config half must be real: production reads ``vae.config.latents_mean`` /
    ``latents_std`` / ``z_dim`` off the genuine diffusers config, so a
    self-declared namespace could not catch an upstream rename of any of them.
    The encode half stays controlled on purpose — the assertion is that
    ``encode_pixels`` computes ``(raw - mean) / std``, which needs a raw value the
    test knows exactly, not one a random-init encoder happens to produce.
    """

    vae = build_tiny_wan_vae(z_dim=raw_latents.shape[1])
    vae.register_to_config(
        latents_mean=[0.5, -0.25],
        latents_std=[2.0, 4.0],
    )
    vae.encode = lambda _x: SimpleNamespace(  # type: ignore[method-assign]
        latent_dist=SimpleNamespace(sample=lambda: raw_latents),
    )
    return vae


def test_encode_pixels_normalizes_inverse_of_decode() -> None:
    raw = torch.tensor(
        [[[[[3.0]]], [[[7.0]]]], [[[[-1.0]]], [[[0.5]]]]],
    )  # [B=2, z_dim=2, T=1, H=1, W=1]
    vae = _wan_vae_with_fixed_raw(raw)
    pipeline = SimpleNamespace(vae=vae)
    encoders = WanDPOEncoders(
        pipeline,
        num_frames=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    z = encoders.encode_pixels(torch.zeros(2, 3, 4, 4))

    mean = torch.tensor(vae.config.latents_mean).view(1, 2, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, 2, 1, 1, 1)
    torch.testing.assert_close(z, (raw - mean) / std)
    # Round-trip through the canonical decode denormalization recovers raw.
    torch.testing.assert_close(z * std + mean, raw)
    # False side: the historical bug multiplied by std instead of dividing.
    assert not torch.allclose(z, (raw - mean) * std)


def test_text_encoder_preserves_caption_rows_and_disables_gradients() -> None:
    source = torch.arange(6.0).reshape(2, 3).requires_grad_()
    calls = []

    def encode_prompt(**kwargs):
        calls.append((kwargs, torch.is_grad_enabled()))
        return source * 2, None

    pipeline = SimpleNamespace(
        vae=_wan_vae_with_fixed_raw(torch.zeros(2, 2, 1, 1, 1)),
        encode_prompt=encode_prompt,
    )
    encoders = WanDPOEncoders(pipeline, num_frames=1, device="cpu", dtype=torch.float64)
    embeddings = encoders.encode_text(["first", "second"])

    torch.testing.assert_close(embeddings, (source * 2).to(torch.float64))
    assert not embeddings.requires_grad
    assert calls[0][1] is False
    assert calls[0][0] == {
        "prompt": ["first", "second"],
        "negative_prompt": ["", ""],
        "do_classifier_free_guidance": False,
        "num_videos_per_prompt": 1,
        "max_sequence_length": 512,
        "device": "cpu",
    }
