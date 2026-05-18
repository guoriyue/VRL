"""Chunked VAE decode policy for diffusion model latents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch

LatentOutputLayout = Literal["image_bchw", "video_btchw", "video_bcthw"]


@dataclass(slots=True)
class LatentDecodeTransform:
    """Callable latent normalization step before VAE decode."""

    fn: Callable[[torch.Tensor], torch.Tensor]

    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        return self.fn(latents)


@dataclass(slots=True)
class LatentDecodeSpec:
    """All family-owned callables needed for chunked latent decoding."""

    transform: LatentDecodeTransform
    vae_decode: Callable[[torch.Tensor], torch.Tensor]
    postprocess: Callable[[torch.Tensor], torch.Tensor]
    output_layout: LatentOutputLayout
    decode_batch_size: int | None = None
    match_num_frames: Callable[[torch.Tensor], torch.Tensor] | None = None


class ChunkedLatentDecoder:
    """Apply family normalization, VAE decode, postprocess, and layout policy."""

    def __init__(self, spec: LatentDecodeSpec) -> None:
        self.spec = spec

    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        chunks = self._chunks(latents)
        decoded = [self._decode_chunk(chunk) for chunk in chunks]
        output = decoded[0] if len(decoded) == 1 else torch.cat(decoded, dim=0)
        if self.spec.output_layout == "video_btchw":
            return output.permute(0, 2, 1, 3, 4)
        return output

    def _chunks(self, latents: torch.Tensor) -> list[torch.Tensor]:
        batch = latents.shape[0]
        size = self.spec.decode_batch_size
        if size is None or size <= 0 or size >= batch:
            return [latents]
        return [latents[start : start + size] for start in range(0, batch, size)]

    def _decode_chunk(self, latents: torch.Tensor) -> torch.Tensor:
        x = self.spec.transform(latents)
        decoded = self.spec.vae_decode(x)
        if self.spec.match_num_frames is not None:
            decoded = self.spec.match_num_frames(decoded)
        return self.spec.postprocess(decoded)
