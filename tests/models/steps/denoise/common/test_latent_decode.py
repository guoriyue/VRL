from __future__ import annotations

import torch

from vrl.models.steps.denoise.common import (
    ChunkedLatentDecoder,
    LatentDecodePlan,
)


def test_chunked_latent_decoder_decodes_in_batch_chunks() -> None:
    """Checks chunked latent decoder decodes in batch batches."""
    calls: list[torch.Tensor] = []

    def decode(batch: torch.Tensor) -> torch.Tensor:
        calls.append(batch)
        return batch + 10

    decoder = ChunkedLatentDecoder(
        LatentDecodePlan(
            prepare_latents=lambda x: x * 2,
            vae_decode=decode,
            postprocess=lambda x: x - 1,
            output_layout="image_bchw",
            decode_batch_size=1,
        )
    )

    latents = torch.arange(3.0).view(3, 1, 1, 1)
    out = decoder(latents)

    assert len(calls) == 3
    torch.testing.assert_close(out, latents * 2 + 9)


def test_chunked_latent_decoder_standardizes_video_layout() -> None:
    """Checks chunked latent decoder standardizes video layout."""
    decoder = ChunkedLatentDecoder(
        LatentDecodePlan(
            prepare_latents=lambda x: x,
            vae_decode=lambda x: x,
            postprocess=lambda x: x.permute(0, 2, 1, 3, 4),
            output_layout="video_btchw",
        )
    )

    latents = torch.zeros(2, 3, 4, 1, 1)
    out = decoder(latents)

    assert out.shape == (2, 3, 4, 1, 1)


def test_latent_decode_plan_rejects_non_callable_fields() -> None:
    """A trailing comma silently turns a lambda field into a 1-tuple; the plan
    must reject that at construction instead of raising mid-decode."""
    import pytest

    with pytest.raises(TypeError, match="prepare_latents must be callable"):
        LatentDecodePlan(
            prepare_latents=(lambda x: x,),  # type: ignore[arg-type]
            vae_decode=lambda x: x,
            postprocess=lambda x: x,
            output_layout="image_bchw",
        )
