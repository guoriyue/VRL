"""VAE decode policy for diffusion model latents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch

LatentOutputLayout = Literal["image_bchw", "video_btchw"]


@dataclass(slots=True)
class ChunkedLatentDecoder:
    """Apply family normalization, VAE decode, postprocess, and layout policy.

    The family supplies the callables; the decoder owns the output layout.
    """

    prepare_latents: Callable[[torch.Tensor], torch.Tensor]
    vae_decode: Callable[[torch.Tensor], torch.Tensor]
    postprocess: Callable[[torch.Tensor], torch.Tensor]
    output_layout: LatentOutputLayout
    prepare_decoded: Callable[[torch.Tensor], torch.Tensor] | None = None

    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        decoded = self.vae_decode(self.prepare_latents(latents))
        if self.prepare_decoded is not None:
            decoded = self.prepare_decoded(decoded)
        output = self.postprocess(decoded)
        if self.output_layout == "video_btchw":
            return output.permute(0, 2, 1, 3, 4)
        return output


class VaeDecodeMixin:
    """``decode_latents`` for families whose denormalization is scale + shift.

    sd3_5 and sana carried byte-identical decode bodies; the only real
    differences were the output layout and whether the VAE ships a
    ``shift_factor``.

    OPT IN BY LISTING THIS MIXIN FIRST in the bases. ``decode_latents`` is an
    ``@abstractmethod`` on ``DenoisePipelineModelBase``, so a mixin placed after
    ``DiffusersPipelineModelBase`` loses the MRO race, the abstract method
    survives, and the class raises ``TypeError`` at instantiation.

    Families whose decode is more than scale + shift keep their own override:
    flux/qwen_image unpack packed latents first, and cosmos/wan/causvid
    decode outside a diffusers image processor entirely.
    """

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents through the pipeline VAE and its image processor."""

        pipe = self.pipeline
        vae = pipe.vae
        scaling_factor = vae.config.scaling_factor
        # Read unconditionally. DC-AE (sana) ships no ``shift_factor`` at all,
        # so the default makes the shift a no-op there — one arithmetic path
        # instead of a per-family branch, and a checkpoint that ever gains the
        # field is then handled like sd3's.
        shift_factor = getattr(vae.config, "shift_factor", 0.0) or 0.0

        decoder = ChunkedLatentDecoder(
            prepare_latents=lambda batch: batch.to(vae.dtype) / scaling_factor + shift_factor,
            vae_decode=lambda batch: vae.decode(batch, return_dict=False)[0],
            postprocess=lambda decoded: pipe.image_processor.postprocess(
                decoded, output_type="pt"
            ),
            output_layout="image_bchw",
        )
        return decoder(latents)


def denormalize_video_vae_latents(unpacked: torch.Tensor, vae: Any) -> torch.Tensor:
    """Undo a video VAE's per-channel ``latents_mean`` / ``latents_std`` normalization."""

    z_dim = vae.config.z_dim
    unpacked = unpacked.to(vae.dtype)
    mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(unpacked)
    std = torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(unpacked)
    return unpacked * std + mean
