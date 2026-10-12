"""Qwen-Image-Layered: source-conditioned RGBA layer decomposition on the Qwen-Image backbone.

The layered checkpoint is the ``qwen_image`` MMDiT and video VAE loaded through its
own ``QwenImageLayeredPipeline`` (the checkpoint's ``model_index.json`` picks the
class). What differs from text-to-image is the token layout, not the model:

- One source image is VAE-encoded into packed condition tokens that are appended
  AFTER the generated tokens; the transformer's output is sliced back to the
  generated prefix, so the condition is conditioning, never an action.
- The generated sequence holds ``layers + 1`` frames of one ``h x w`` grid: frame 0
  is the model's own copy of the source (discarded at decode, as upstream does),
  frames ``1..layers`` are RGBA layers. The VAE has four input/output channels.
- Timesteps are ``sigmas = linspace(1, 0)`` shifted by ``mu = sqrt(tokens / 256)``
  of the condition image, not ``calculate_shift``.
- ``additional_t_cond`` is the pipeline's constant RGB flag (zeros).

``encode_video_to_latents`` is the exact inverse of ``decode_latents`` on a
``[B, C, layers + 1, H, W]`` clip, so the GRPO diffusion-loss regularizer can be
pointed at known complete layers.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from typing import Any

import torch

from vrl.generation.types import DenoiseRequest
from vrl.models.families.qwen_image.model import (
    QwenImageModel,
    QwenImagePipelineModel,
    QwenImageSamplingState,
)
from vrl.models.steps.denoise.common import (
    ChunkedLatentDecoder,
    expand_batch_timestep,
    pack_eval_timestep,
    predict_with_cfg,
)


@dataclass
class QwenImageLayeredSamplingState(QwenImageSamplingState):
    """Qwen-Image state plus the packed source tokens and the layer budget."""

    condition_latents: torch.Tensor
    layers: int


class QwenImageLayeredModel(QwenImageModel):
    """Layered transformer, scheduler and per-step math: the model replay runs.

    The generated ``layers + 1`` frames run with the source's condition tokens
    appended; the transformer output is sliced back to the generated prefix.
    """

    def _set_dynamic_timesteps(self, num_steps: int, image_seq_len: int, device: Any) -> Any:
        """Linear sigmas shifted by the layered pipeline's ``sqrt(tokens / 256)`` mu."""
        import numpy as np

        self.scheduler.set_timesteps(
            sigmas=np.linspace(1.0, 0.0, num_steps + 1)[:-1],
            device=device,
            mu=(image_seq_len / 256.0) ** 0.5,
        )
        return self.scheduler.timesteps

    def _img_shapes(self, state: QwenImageLayeredSamplingState, bsz: int) -> list:
        vsf = state.vae_scale_factor
        grid = (1, state.height // vsf // 2, state.width // vsf // 2)
        # layers + 1 generated frames, then the one condition image.
        return [[grid] * (state.layers + 1) + [grid]] * bsz

    def forward_step(
        self,
        state: QwenImageLayeredSamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        """Transformer forward over ``[generated | condition]`` tokens, sliced to generated."""
        t = state.timesteps[step_idx]
        bsz, target_tokens = state.latents.shape[0], state.latents.shape[1]
        td = self._transformer_dtype()

        latent_input = torch.cat([state.latents.to(td), state.condition_latents.to(td)], dim=1)
        timestep_batch = (
            expand_batch_timestep(t, bsz).to(device=latent_input.device, dtype=td) / 1000.0
        )
        guidance = None
        if state.guidance_embeds:
            guidance = torch.full(
                (bsz,), float(state.guidance_scale), device=latent_input.device, dtype=td
            )
        cond = {
            "hidden_states": latent_input,
            "timestep": timestep_batch,
            "encoder_hidden_states": state.prompt_embeds.to(td),
            "encoder_hidden_states_mask": state.prompt_embeds_mask,
            "img_shapes": self._img_shapes(state, bsz),
            # The pipeline's constant "RGB source" flag.
            "additional_t_cond": torch.zeros(bsz, dtype=torch.long, device=latent_input.device),
        }
        if guidance is not None:
            cond["guidance"] = guidance
        uncond = None
        if state.do_cfg:
            uncond = {
                **cond,
                "encoder_hidden_states": state.negative_prompt_embeds.to(td),
                "encoder_hidden_states_mask": state.negative_prompt_embeds_mask,
            }
        noise_pred = predict_with_cfg(
            self.transformer,
            cond,
            uncond,
            guidance_scale=state.guidance_scale,
            batched=False,
            dtype=td,
            normalize=True,
        )
        # The condition tokens' prediction is discarded. The norm-preserving combine
        # is per token, so slicing after it equals the pipeline's slice-then-combine.
        return {"noise_pred": noise_pred[:, :target_tokens]}

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> QwenImageLayeredSamplingState:
        timesteps = pack_eval_timestep(replay_tensors["timesteps"], step_idx)
        return QwenImageLayeredSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=None,
            prompt_embeds=replay_tensors["prompt_embeds"],
            prompt_embeds_mask=replay_tensors.get("prompt_embeds_mask"),
            negative_prompt_embeds=replay_tensors.get("negative_prompt_embeds"),
            negative_prompt_embeds_mask=replay_tensors.get("negative_prompt_embeds_mask"),
            guidance_scale=batch_context["guidance_scale"],
            do_cfg=batch_context["cfg"]
            and replay_tensors.get("negative_prompt_embeds") is not None,
            guidance_embeds=batch_context["guidance_embeds"],
            height=batch_context["height"],
            width=batch_context["width"],
            vae_scale_factor=int(batch_context["vae_scale_factor"]),
            condition_latents=replay_tensors["condition_latents"],
            layers=int(batch_context["layers"]),
        )


class QwenImageLayeredPipelineModel(QwenImagePipelineModel, QwenImageLayeredModel):
    """Layered model plus the pipeline: source encoding, sampling setup and RGBA decode."""

    def __init__(self, *, pipeline: Any, device: Any = None) -> None:
        super().__init__(pipeline=pipeline, device=device)
        self._decode_layers = 4

    def _latent_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        vae = self.pipeline.vae
        z_dim = vae.config.z_dim
        mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1)
        return mean, std

    def _with_alpha(self, pixels: torch.Tensor) -> torch.Tensor:
        """Give an RGB clip the opaque alpha the four-channel VAE expects."""
        channels = int(getattr(self.pipeline.vae.config, "input_channels", 3))
        if pixels.shape[1] == 3 and channels == 4:
            alpha = torch.ones_like(pixels[:, :1])
            return torch.cat([pixels, alpha], dim=1)
        if pixels.shape[1] == 4 and channels == 3:
            return pixels[:, :3]
        return pixels

    def _encode_frames(self, pixels: torch.Tensor) -> torch.Tensor:
        """``[B, C, T, H, W]`` in ``[-1, 1]`` -> packed, normalized ``[B, T * h * w / 4, 4C]``.

        Every frame is encoded as its own one-frame clip, exactly as the pipeline
        encodes the condition image and decodes each layer: the VAE is causal in
        time, so encoding ``T`` frames as one clip would entangle them.
        """
        from diffusers import QwenImageLayeredPipeline

        vae = self.pipeline.vae
        mean, std = self._latent_stats()
        pixels = self._with_alpha(pixels).to(device=self.device, dtype=vae.dtype)
        frames = []
        with torch.no_grad():
            for index in range(pixels.shape[2]):
                encoded = vae.encode(pixels[:, :, index : index + 1])
                dist = getattr(encoded, "latent_dist", encoded)
                frames.append(dist.mode() if hasattr(dist, "mode") else dist)
        latents = torch.cat(frames, dim=2)  # [B, z, T, h, w]
        latents = (latents - mean.to(latents.device, latents.dtype)) / std.to(
            latents.device, latents.dtype
        )
        batch, z_dim, frame_count, height, width = latents.shape
        return QwenImageLayeredPipeline._pack_latents(
            latents.permute(0, 2, 1, 3, 4), batch, z_dim, height, width, frame_count
        )

    def encode_video_to_latents(self, video: torch.Tensor) -> torch.Tensor:
        """Encode a clean ``[B, C, layers + 1, H, W]`` clip in ``[0, 1]`` into rollout latents.

        Frame 0 is the source composite, the rest are RGBA layers; the result has
        the packed layout ``decode_latents`` consumes, so it is a valid SFT target.
        """
        return self._encode_frames(video.float() * 2.0 - 1.0)

    def encode_condition_image(
        self,
        image: Any,
        *,
        height: int,
        width: int,
        input_mode: str = "rgb",
    ) -> torch.Tensor:
        """Source image (PIL or ``[1, C, H, W]`` in ``[0, 1]``) -> packed condition tokens."""
        if input_mode not in {"rgb", "rgba"}:
            raise ValueError("input_mode must be 'rgb' or 'rgba'")
        if isinstance(image, torch.Tensor):
            if image.ndim != 4 or image.shape[0] != 1:
                raise ValueError("a tensor source image must be [1, C, H, W]")
            pixels = image.float() * 2.0 - 1.0
            if pixels.shape[-2:] != (height, width):
                raise ValueError("a tensor source image must already be at the request geometry")
        else:
            processor = self.pipeline.image_processor
            resized = processor.resize(image, height, width)
            pixels = processor.preprocess(resized, height, width)
        if input_mode == "rgb":
            pixels = pixels[:, :3]
        return self._encode_frames(pixels.unsqueeze(2)).to(self._transformer_dtype())

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Text conditioning plus the packed source-image tokens."""
        reference_image = kwargs.pop("reference_image", None)
        height = kwargs.pop("height", None)
        width = kwargs.pop("width", None)
        input_mode = kwargs.pop("input_mode", "rgb")
        if reference_image is None:
            raise ValueError(
                "qwen_image_layered decomposes one source image; reference_image is required"
            )
        if height is None or width is None:
            raise ValueError(
                "qwen_image_layered needs the request height and width to encode the source"
            )
        result = super().encode_prompt(prompt, negative_prompt, **kwargs)
        result["condition_latents"] = self.encode_condition_image(
            reference_image, height=int(height), width=int(width), input_mode=input_mode
        )
        return result

    def prepare_sampling(
        self,
        request: DenoiseRequest,
        encoded: dict[str, Any],
        *,
        initial_latents: torch.Tensor | None = None,
        layers: int = 4,
        **kwargs: Any,
    ) -> QwenImageLayeredSamplingState:
        """Packed ``layers + 1`` frame noise plus the source tokens for one denoise loop."""
        del kwargs
        pipe = self.pipeline
        device = self.device
        layers = int(layers)
        if layers < 1:
            raise ValueError("layers must be >= 1")

        prompt_embeds = encoded["prompt_embeds"]
        negative_prompt_embeds = encoded.get("negative_prompt_embeds")
        condition = encoded["condition_latents"].to(device)

        seed = request.seed if request.seed is not None else random.randint(0, sys.maxsize)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        batch_size = prompt_embeds.shape[0]
        num_channels_latents = pipe.transformer.config.in_channels // 4
        latents, _ = pipe.prepare_latents(
            None,
            batch_size,
            num_channels_latents,
            request.height,
            request.width,
            layers,
            torch.float32,
            device,
            generator,
            initial_latents,
        )
        if condition.shape[0] != batch_size:
            raise ValueError(
                f"condition tokens carry {condition.shape[0]} rows for {batch_size} samples",
            )
        timesteps = self._set_dynamic_timesteps(request.num_steps, condition.shape[1], device)

        self._decode_height = int(request.height)
        self._decode_width = int(request.width)
        self._decode_layers = layers

        return QwenImageLayeredSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=pipe.scheduler,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=encoded.get("prompt_embeds_mask"),
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=encoded.get("negative_prompt_embeds_mask"),
            guidance_scale=request.guidance_scale,
            do_cfg=negative_prompt_embeds is not None,
            guidance_embeds=self._guidance_embeds,
            height=int(request.height),
            width=int(request.width),
            vae_scale_factor=int(pipe.vae_scale_factor),
            condition_latents=condition,
            layers=layers,
        )

    def export_batch_context(self, state: QwenImageLayeredSamplingState) -> dict[str, Any]:
        return {**super().export_batch_context(state), "layers": int(state.layers)}

    def export_replay_tensors(self, state: QwenImageLayeredSamplingState) -> dict[str, Any]:
        return {
            **super().export_replay_tensors(state),
            "condition_latents": state.condition_latents,
        }

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Packed latents -> ``[B, layers, 4, H, W]`` RGBA layers in ``[0, 1]``.

        Frame 0 (the model's source copy) is dropped as the pipeline does; every
        layer decodes as its own one-frame clip.
        """
        from diffusers import QwenImageLayeredPipeline

        pipe = self.pipeline
        vae = pipe.vae
        vae_scale_factor = pipe.vae_scale_factor
        height, width, layers = self._decode_height, self._decode_width, self._decode_layers
        mean, std = self._latent_stats()

        def _transform(batch: torch.Tensor) -> torch.Tensor:
            unpacked = QwenImageLayeredPipeline._unpack_latents(
                batch, height, width, layers, vae_scale_factor
            ).to(vae.dtype)
            unpacked = unpacked * std.to(unpacked.device, unpacked.dtype) + mean.to(
                unpacked.device, unpacked.dtype
            )
            layer_latents = unpacked[:, :, 1:].permute(0, 2, 1, 3, 4)  # [b, L, z, h, w]
            b, _, z_dim, h, w = layer_latents.shape
            return layer_latents.reshape(b * layers, z_dim, 1, h, w)

        def _postprocess(image: torch.Tensor) -> torch.Tensor:
            flat = pipe.image_processor.postprocess(image, output_type="pt")
            return flat.reshape(-1, layers, *flat.shape[1:])

        decoder = ChunkedLatentDecoder(
            prepare_latents=_transform,
            vae_decode=lambda batch: vae.decode(batch, return_dict=False)[0],
            prepare_decoded=lambda decoded: decoded[:, :, 0],
            postprocess=_postprocess,
            output_layout="image_bchw",
        )
        return decoder(latents)


__all__ = [
    "QwenImageLayeredModel",
    "QwenImageLayeredPipelineModel",
    "QwenImageLayeredSamplingState",
]
