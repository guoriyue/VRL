"""Chunked VAE decode policy for diffusion model latents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch

LatentOutputLayout = Literal["image_bchw", "video_btchw"]


@dataclass(slots=True)
class LatentDecodePlan:
    """All family-owned callables needed for chunked latent decoding."""

    prepare_latents: Callable[[torch.Tensor], torch.Tensor]
    vae_decode: Callable[[torch.Tensor], torch.Tensor]
    postprocess: Callable[[torch.Tensor], torch.Tensor]
    output_layout: LatentOutputLayout
    decode_batch_size: int | None = None
    prepare_decoded: Callable[[torch.Tensor], torch.Tensor] | None = None


class ChunkedLatentDecoder:
    """Apply family normalization, VAE decode, postprocess, and layout policy."""

    def __init__(self, plan: LatentDecodePlan) -> None:
        self.plan = plan

    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        batches = self._chunks(latents)
        decoded = [self._decode_chunk(batch) for batch in batches]
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


class VaeDecodeMixin:
    """``decode_latents`` for families whose denormalization is scale + shift.

    sd3_5, pixart_sigma, lumina2, hunyuan_image, sana and hunyuan_video carried
    byte-identical decode bodies; the only real differences were the output
    layout and whether the VAE ships a ``shift_factor``.

    OPT IN BY LISTING THIS MIXIN FIRST in the bases. ``decode_latents`` is an
    ``@abstractmethod`` on ``DiffusionModelBase``, so a mixin placed after
    ``DiffusersPipelineModelBase`` loses the MRO race, the abstract method
    survives, and the class raises ``TypeError`` at instantiation — the same
    reason ``LoraModelMixin`` precedes it.

    Families whose decode is more than scale + shift keep their own override:
    flux/qwen_image unpack packed latents first, mochi/cogvideox denormalize
    with per-channel ``latents_mean``/``latents_std``, and cosmos/wan/echo/
    causvid decode outside a diffusers image processor entirely.
    """

    # hunyuan_video is the only member that decodes 5D latents; everything else
    # produces images.
    _decode_output_layout: LatentOutputLayout = "image_bchw"

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents through the pipeline VAE and its output processor."""

        pipe = self.pipeline
        vae = pipe.vae
        scaling_factor = vae.config.scaling_factor
        # Read unconditionally. DC-AE (sana) and the two Hunyuan VAEs ship no
        # ``shift_factor`` at all, so the default makes the shift a no-op there
        # — one arithmetic path instead of a per-family branch, and a
        # checkpoint that ever gains the field is then handled like sd3's.
        shift_factor = getattr(vae.config, "shift_factor", 0.0) or 0.0

        def postprocess(decoded: torch.Tensor) -> torch.Tensor:
            if self._decode_output_layout == "video_btchw":
                return pipe.video_processor.postprocess_video(decoded, output_type="pt")
            return pipe.image_processor.postprocess(decoded, output_type="pt")

        decoder = ChunkedLatentDecoder(
            LatentDecodePlan(
                prepare_latents=lambda batch: batch.to(vae.dtype) / scaling_factor + shift_factor,
                vae_decode=lambda batch: vae.decode(batch, return_dict=False)[0],
                postprocess=postprocess,
                output_layout=self._decode_output_layout,
                decode_batch_size=getattr(pipe, "decode_batch_size", None),
            ),
        )
        return decoder(latents)
