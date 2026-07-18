"""Chunked VAE decode policy for diffusion model latents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch

LatentOutputLayout = Literal["image_bchw", "video_btchw", "video_bcthw"]


@dataclass(slots=True)
class LatentDecodePlan:
    """All family-owned callables needed for chunked latent decoding."""

    prepare_latents: Callable[[torch.Tensor], torch.Tensor]
    vae_decode: Callable[[torch.Tensor], torch.Tensor]
    postprocess: Callable[[torch.Tensor], torch.Tensor]
    output_layout: LatentOutputLayout
    decode_batch_size: int | None = None
    prepare_decoded: Callable[[torch.Tensor], torch.Tensor] | None = None

    def __post_init__(self) -> None:
        # Fail at plan construction, not mid-decode: a trailing comma turns a
        # lambda field into a 1-tuple silently (bit sana/hunyuan 2026-07-12).
        for name in ("prepare_latents", "vae_decode", "postprocess"):
            if not callable(getattr(self, name)):
                raise TypeError(f"LatentDecodePlan.{name} must be callable")
        if self.prepare_decoded is not None and not callable(self.prepare_decoded):
            raise TypeError("LatentDecodePlan.prepare_decoded must be callable or None")


class ChunkedLatentDecoder:
    """Apply family normalization, VAE decode, postprocess, and layout policy."""

    def __init__(self, plan: LatentDecodePlan) -> None:
        self.plan = plan

    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        chunks = self._chunks(latents)
        decoded = [self._decode_chunk(chunk) for chunk in chunks]
        output = decoded[0] if len(decoded) == 1 else torch.cat(decoded, dim=0)
        if self.plan.output_layout == "video_btchw":
            return output.permute(0, 2, 1, 3, 4)
        return output

    def _chunks(self, latents: torch.Tensor) -> list[torch.Tensor]:
        batch = latents.shape[0]
        size = self.plan.decode_batch_size
        if size is None or size <= 0 or size >= batch:
            return [latents]
        return [latents[start : start + size] for start in range(0, batch, size)]

    def _decode_chunk(self, latents: torch.Tensor) -> torch.Tensor:
        prepared = self.plan.prepare_latents(latents)
        decoded = self.plan.vae_decode(prepared)
        if self.plan.prepare_decoded is not None:
            decoded = self.plan.prepare_decoded(decoded)
        return self.plan.postprocess(decoded)
