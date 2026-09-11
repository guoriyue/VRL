from __future__ import annotations

import pytest
import torch

from vrl.models.steps.denoise.common import (
    ChunkedLatentDecoder,
    LatentDecodePlan,
)


@pytest.mark.parametrize(
    "batch_size,expected_rows",
    [(1, [1, 1, 1]), (2, [2, 1]), (3, [3]), (4, [3]), (None, [3]), (0, [3]), (-1, [3])],
)
def test_chunked_latent_decoder_decodes_in_batch_chunks(batch_size, expected_rows) -> None:
    """Preserve row order and prepare/decode/postprocess across chunk boundaries."""
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
            decode_batch_size=batch_size,
        )
    )

    latents = torch.arange(3.0).view(3, 1, 1, 1)
    out = decoder(latents)

    assert [batch.shape[0] for batch in calls] == expected_rows
    torch.testing.assert_close(out, latents * 2 + 9)


def test_chunked_latent_decoder_standardizes_video_layout() -> None:
    """A ``video_btchw`` plan's postprocess permute is undone by the decoder, so callers always
    get [B,C,T,H,W] back whatever layout the VAE postprocess emits.
    """
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
