"""Qwen-Image-2.1 image generation and editing, backed by diffusers.

Diffusion implementation for Alibaba Qwen-Image-2.1 (7B single-stream DiT).
The generation helper flow mirrors every diffusion family:

    encode_prompt -> prepare_sampling -> forward_step xN -> decode_latents

Qwen-Image-2.1 specifics vs the 20B ``qwen_image`` family:
- Latents are packed by a plain spatial flatten ``[B, h*w, C]`` with ``C=64``
  (``patch_size=1``): one token per 16x16 pixel tile. ``decode_latents`` unpacks
  to ``[B, C, 1, H, W]`` for the (video-style) 64-channel VAE.
- The text encoder is Qwen3-VL. ``encode_prompt`` returns a SEQUENCE embed, an
  optional attention mask (``None`` when nothing is padded) and an
  ``image_pad_mask`` marking vision slots in the text stream. The transformer
  needs ``img_mask`` over the JOINT sequence: the text-side pad mask plus one
  ``True`` slot per 2x2 group of target latents (diffusers parity).
- Classifier-free guidance is TRUE CFG (``true_cfg_scale``) but OFF by default
  (1.0). When enabled, cond/uncond tokenize to different lengths, so the runner
  is ``separate_cfg``; the combine is plain linear (no norm rescale, unlike 20B).
- Prefix KV caching (``kv_cache_mode``) is never used: text-token KV under a LoRA
  differs from the cached prefill, so rollout and replay both recompute the full
  sequence every step and stay bit-comparable.
- Optional reference images are encoded once into a fixed latent prefix and
  replayed unchanged. Only target latents are sampled and assigned log-probs.
- The transformer time embedding multiplies its input by 1000 internally, so
  ``forward_step`` feeds ``t / 1000`` (matching ``QwenImage21Pipeline``).
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vrl.generation.types import DenoiseRequest
from vrl.models.steps.denoise import (
    DiffusersPipelineModelBase,
    DiffusersReplayModelBase,
    GuidedDenoiseSamplingStateBase,
)
from vrl.models.steps.denoise.common import (
    ChunkedLatentDecoder,
    DenoiseBackboneCaller,
    DenoiseBackboneInput,
    DenoiseBackboneRunnerBase,
    DenoiseBranch,
    LatentDecodePlan,
    expand_batch_timestep,
    pack_eval_timestep,
)

# Latent tokens per ``img_mask`` slot: the vision-language encoder reserves one
# slot per 2x2 group of latent tokens (``QwenImage21Pipeline.append_target_slots``).
_LATENT_TOKENS_PER_MASK_SLOT = 4


@dataclass
class QwenImage21SamplingState(GuidedDenoiseSamplingStateBase):
    """Private Qwen-Image-2.1 sampling state. Engine MUST NOT introspect."""

    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor | None
    # Text-side vision-slot mask ``[B, txt_len]`` (all False for pure text).
    image_pad_mask: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None
    negative_prompt_embeds_mask: torch.Tensor | None
    negative_image_pad_mask: torch.Tensor | None
    do_cfg: bool
    height: int
    width: int
    vae_scale_factor: int
    reference_latents: torch.Tensor | None = None
    reference_shapes: tuple[tuple[int, int, int], ...] = ()


class QwenImage21Model(DiffusersPipelineModelBase, DenoiseBackboneRunnerBase):
    """Diffusers-backed Qwen-Image-2.1 text and reference-conditioned model.

    Implements the backbone-runner protocol itself. True CFG with cond/uncond
    prompts of DIFFERENT token lengths cannot be packed into one batched call,
    hence ``separate_cfg``; with the checkpoint's default ``true_cfg_scale=1.0``
    the runner makes one forward per step.
    """

    cfg_mode = "separate_cfg"
    cfg_base = "uncond"
    # ``QwenImage21Pipeline`` combines linearly:
    #     noise = uncond + s * (cond - uncond)
    cfg_normalization = False

    def build_branch(
        self,
        request: DenoiseBackboneInput,
        branch: str,
    ) -> DenoiseBranch:
        """Map Qwen-Image-2.1 transformer kwargs into the shared backbone contract."""
        if branch == "cond":
            embeds = request.prompt_embeds
            mask = request.extra.get("encoder_hidden_states_mask")
            img_mask = request.extra["img_mask"]
        else:
            embeds = request.negative_prompt_embeds
            mask = request.extra.get("negative_encoder_hidden_states_mask")
            img_mask = request.extra["negative_img_mask"]
        return DenoiseBranch(
            hidden_states=request.hidden_states,
            timestep=request.timestep,
            encoder_hidden_states=embeds,
            extra_kwargs={
                "encoder_hidden_states_mask": mask,
                "img_shapes": request.extra["img_shapes"],
                "img_mask": img_mask,
            },
        )

    def postprocess_branch(
        self,
        request: DenoiseBackboneInput,
        branch: DenoiseBranch,
        raw_output: torch.Tensor,
    ) -> torch.Tensor:
        """Keep the target-image tail of the joint (text + image) output.

        The single-stream transformer returns a prediction for every joint
        token; ``QwenImage21Pipeline`` slices ``[:, -latents.size(1):]`` and so
        does the rollout/replay step here.
        """
        del branch
        target_tokens = request.extra["target_tokens"]
        return raw_output[:, -target_tokens:]

    def __init__(
        self,
        *,
        pipeline: Any,
        device: Any = None,
    ) -> None:
        super().__init__(pipeline=pipeline, device=device)
        # decode_latents only receives the packed latent tensor; prepare_sampling
        # records the spatial shape it must unpack to (single model instance runs
        # prepare -> denoise -> decode sequentially per batch).
        self._decode_height = 1024
        self._decode_width = 1024
        self._output_mode = "rgb"

    def packed_token_count(self, height: int, width: int) -> int:
        """2.1 packs the 16x VAE grid unpatched, each side rounded to 2 latent cells.

        Mirrors ``QwenImage21Pipeline.prepare_latents`` (its ``vae_scale_factor``
        is hardcoded to 16, so replay needs no VAE to compute it).
        """
        return (2 * (height // 32)) * (2 * (width // 32))

    def _set_dynamic_timesteps(self, num_steps: int, image_seq_len: int, device: Any) -> Any:
        """Set timesteps exactly as ``QwenImage21Pipeline`` does.

        The pipeline passes an explicit ``sigmas = linspace(1, 1/N)`` grid plus
        the resolution-derived ``mu``; the scheduler's own default grid ends at
        ``sigma_min`` instead, so the explicit sigmas are required for parity.
        """
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift

        scheduler = self.scheduler
        config = scheduler.config
        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        mu = calculate_shift(
            image_seq_len,
            config.get("base_image_seq_len", 256),
            config.get("max_image_seq_len", 4096),
            config.get("base_shift", 0.5),
            config.get("max_shift", 1.15),
        )
        scheduler.set_timesteps(num_steps, device=device, sigmas=sigmas, mu=mu)
        return scheduler.timesteps

    # -- encode_prompt -------------------------------------------------

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode prompt via Qwen3-VL (sequence embeds + mask + vision-slot mask).

        Returns the conditional tensors and, when CFG is active and a negative
        prompt is supplied, the unconditional tensors for the separate branch.
        ``max_sequence_length`` is not a pipeline input for 2.1 (the template is
        tokenized whole); it is accepted and ignored for executor uniformity.
        """
        guidance_scale = kwargs.get("guidance_scale", 1.0)
        do_cfg = guidance_scale > 1.0 and negative_prompt is not None
        pipe = self.pipeline
        # The frozen Qwen3-VL encoder may live on CPU (model.memory.cpu_resident);
        # run encode there, then move embeds onto the transformer device.
        enc_device = self._encoder_device()
        td = self.transformer.dtype

        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions

        from vrl.utils.validation import require_int

        references = kwargs.get("reference_images") or []
        if len(references) > 10:
            raise ValueError("Qwen-Image-2.1 accepts at most 10 reference images")
        resolution = require_int(
            kwargs.get("reference_resolution", 1024), path="reference_resolution", minimum=32
        )
        output_mode = kwargs.get("output_mode", "rgb")
        if output_mode not in {"rgb", "rgba"}:
            raise ValueError("output_mode must be 'rgb' or 'rgba'")
        input_images, reference_latents, reference_shapes = [], [], []
        for image in references:
            image = image.convert("RGBA")
            width, height, _ = calculate_dimensions(resolution**2, image.width / image.height)
            if min(width, height) < 32:
                raise ValueError("Reference aspect ratio is too extreme for reference_resolution")
            vision_size = pipe.processor.image_processor.size
            if not vision_size["shortest_edge"] <= width * height <= vision_size["longest_edge"]:
                raise ValueError(
                    "reference_resolution produces an image outside the checkpoint vision "
                    f"processor's pixel limits {vision_size}; change reference_resolution so "
                    "the vision encoder and VAE use the same image geometry"
                )
            input_images.append(pipe.image_processor.resize(image, width=width, height=height))
            vae_image = pipe.image_processor.preprocess(image, width=width, height=height)
            # Keep reference alpha for the VAE. The upstream prompt encoder
            # composites only its own vision-encoder copy over white.
            vae_image = vae_image.unsqueeze(2).to(device=self.device, dtype=pipe.vae.dtype)
            latent = pipe._encode_vae_image(vae_image, generator=None)
            latent_h, latent_w = latent.shape[-2:]
            reference_shapes.append((1, latent_h, latent_w))
            reference_latents.append(
                pipe._pack_latents(latent, 1, latent.shape[1], latent_h, latent_w).to(td)
            )

        def _encode(
            text: str | list[str],
        ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
            embeds, mask, image_pad_mask = pipe.encode_prompt(
                prompt=text,
                num_images_per_prompt=1,
                device=enc_device,
                image=input_images or None,
            )
            return (
                embeds.to(self.device, dtype=td),
                None if mask is None else mask.to(self.device),
                image_pad_mask.to(self.device),
            )

        prompt_embeds, prompt_embeds_mask, image_pad_mask = _encode(prompt)
        reference_tokens = sum(latent.shape[1] for latent in reference_latents)
        if not torch.all(
            image_pad_mask.sum(dim=1) * _LATENT_TOKENS_PER_MASK_SLOT == reference_tokens
        ):
            raise ValueError("Reference vision slots and VAE latent geometry disagree")
        result: dict[str, Any] = {
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
            "image_pad_mask": image_pad_mask,
            "reference_shapes": tuple(reference_shapes),
            "output_mode": output_mode,
        }
        if reference_latents:
            # encode_prompt may also be called directly with a prompt batch.
            result["reference_latents"] = (
                torch.cat(reference_latents, dim=1)
                .expand(prompt_embeds.shape[0], -1, -1)
                .contiguous()
            )
        if do_cfg:
            negative_embeds, negative_mask, negative_image_pad_mask = _encode(negative_prompt)
            result["negative_prompt_embeds"] = negative_embeds
            result["negative_prompt_embeds_mask"] = negative_mask
            result["negative_image_pad_mask"] = negative_image_pad_mask
        return result

    # -- prepare_sampling ----------------------------------------------

    def prepare_sampling(
        self,
        request: DenoiseRequest,
        encoded: dict[str, Any],
        *,
        initial_latents: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> QwenImage21SamplingState:
        """Build the per-request packed-latent SamplingState for a denoise loop."""
        del kwargs
        pipe = self.pipeline
        device = self.device

        prompt_embeds = encoded["prompt_embeds"]
        negative_prompt_embeds = encoded.get("negative_prompt_embeds")

        seed = request.seed if request.seed is not None else random.randint(0, sys.maxsize)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        batch_size = prompt_embeds.shape[0]
        # 2.1 consumes latents unpatched: in_channels IS the latent channel count.
        num_channels_latents = pipe.transformer.config.in_channels
        latents, _ = pipe.prepare_latents(
            None,
            batch_size,
            num_channels_latents,
            request.height,
            request.width,
            torch.float32,
            device,
            generator,
            initial_latents,
        )

        # Dynamic-shifting timesteps depend on the packed image sequence length.
        timesteps = self._set_dynamic_timesteps(
            request.num_steps,
            latents.shape[1],
            device,
        )

        self._decode_height = int(request.height)
        self._decode_width = int(request.width)
        self._output_mode = encoded.get("output_mode", "rgb")

        return QwenImage21SamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=pipe.scheduler,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=encoded.get("prompt_embeds_mask"),
            image_pad_mask=encoded["image_pad_mask"],
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=encoded.get("negative_prompt_embeds_mask"),
            negative_image_pad_mask=encoded.get("negative_image_pad_mask"),
            guidance_scale=request.guidance_scale,
            do_cfg=negative_prompt_embeds is not None,
            height=int(request.height),
            width=int(request.width),
            vae_scale_factor=int(pipe.vae_scale_factor),
            reference_latents=encoded.get("reference_latents"),
            reference_shapes=encoded.get("reference_shapes", ()),
        )

    # -- forward_step --------------------------------------------------

    def _img_shapes(self, state: QwenImage21SamplingState, bsz: int) -> list:
        vsf = state.vae_scale_factor
        return [[*state.reference_shapes, (1, state.height // vsf, state.width // vsf)]] * bsz

    @staticmethod
    def _joint_img_mask(text_mask: torch.Tensor, latent_tokens: int) -> torch.Tensor:
        """Text-side vision slots plus one ``True`` slot per 2x2 group of target latents."""
        text_mask = text_mask.bool()
        target = text_mask.new_ones(
            text_mask.shape[0], latent_tokens // _LATENT_TOKENS_PER_MASK_SLOT
        )
        return torch.cat([text_mask, target], dim=1)

    def forward_step(
        self,
        state: QwenImage21SamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        """Qwen-Image-2.1 transformer forward (true-CFG separate branches when do_cfg)."""
        t = state.timesteps[step_idx]
        bsz, latent_tokens = state.latents.shape[0], state.latents.shape[1]
        td = self._transformer_dtype()

        latent_input = state.latents.to(td)
        if state.reference_latents is not None:
            latent_input = torch.cat([state.reference_latents.to(td), latent_input], dim=1)
        timestep_batch = (
            expand_batch_timestep(t, bsz).to(device=latent_input.device, dtype=td) / 1000.0
        )
        negative_embeds = (
            None if state.negative_prompt_embeds is None else state.negative_prompt_embeds.to(td)
        )
        negative_img_mask = (
            None
            if state.negative_image_pad_mask is None
            else self._joint_img_mask(state.negative_image_pad_mask, latent_tokens)
        )
        output = DenoiseBackboneCaller(
            self.transformer,
            self,
        )(
            DenoiseBackboneInput(
                hidden_states=latent_input,
                timestep=timestep_batch,
                prompt_embeds=state.prompt_embeds.to(td),
                negative_prompt_embeds=negative_embeds,
                guidance_scale=state.guidance_scale,
                do_cfg=state.do_cfg,
                output_dtype=td,
                extra={
                    "target_tokens": latent_tokens,
                    "encoder_hidden_states_mask": state.prompt_embeds_mask,
                    "negative_encoder_hidden_states_mask": state.negative_prompt_embeds_mask,
                    "img_shapes": self._img_shapes(state, bsz),
                    "img_mask": self._joint_img_mask(state.image_pad_mask, latent_tokens),
                    "negative_img_mask": negative_img_mask,
                },
            ),
        )
        return output.as_dict()

    # -- collector boundary --------------------------------------------

    def export_batch_context(self, state: QwenImage21SamplingState) -> dict[str, Any]:
        """Project Qwen-Image-2.1 sampling state into trajectory context."""
        return {
            "guidance_scale": state.guidance_scale,
            "cfg": state.do_cfg,
            "height": state.height,
            "width": state.width,
            "vae_scale_factor": state.vae_scale_factor,
            "reference_shapes": state.reference_shapes,
        }

    def export_replay_tensors(self, state: QwenImage21SamplingState) -> dict[str, Any]:
        """Project Qwen-Image-2.1 sampling state into per-sample trajectory tensors.

        Masks / negative embeds are only stored when present, so ``restore``
        reads them with ``.get``. Bool masks travel as ``int64`` (the trajectory
        store keeps integer tensors; ``_joint_img_mask`` casts back).
        """
        tensors: dict[str, Any] = {
            "prompt_embeds": state.prompt_embeds,
            "image_pad_mask": state.image_pad_mask.to(torch.int64),
        }
        if state.reference_latents is not None:
            tensors["reference_latents"] = state.reference_latents
        if state.prompt_embeds_mask is not None:
            tensors["prompt_embeds_mask"] = state.prompt_embeds_mask
        if state.negative_prompt_embeds is not None:
            tensors["negative_prompt_embeds"] = state.negative_prompt_embeds
        if state.negative_prompt_embeds_mask is not None:
            tensors["negative_prompt_embeds_mask"] = state.negative_prompt_embeds_mask
        if state.negative_image_pad_mask is not None:
            tensors["negative_image_pad_mask"] = state.negative_image_pad_mask.to(torch.int64)
        return tensors

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> QwenImage21SamplingState:
        """Rebuild QwenImage21SamplingState from a batch slice for the eval forward path."""
        ts = replay_tensors["timesteps"]
        timesteps = pack_eval_timestep(ts, step_idx)
        negative_prompt_embeds = replay_tensors.get("negative_prompt_embeds")
        return QwenImage21SamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=None,  # not needed for forward_step (no scheduler.step here)
            prompt_embeds=replay_tensors["prompt_embeds"],
            prompt_embeds_mask=replay_tensors.get("prompt_embeds_mask"),
            image_pad_mask=replay_tensors["image_pad_mask"],
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=replay_tensors.get("negative_prompt_embeds_mask"),
            negative_image_pad_mask=replay_tensors.get("negative_image_pad_mask"),
            guidance_scale=batch_context["guidance_scale"],
            do_cfg=batch_context["cfg"] and negative_prompt_embeds is not None,
            height=batch_context["height"],
            width=batch_context["width"],
            vae_scale_factor=int(batch_context["vae_scale_factor"]),
            reference_latents=replay_tensors.get("reference_latents"),
            reference_shapes=tuple(
                tuple(shape) for shape in batch_context.get("reference_shapes", ())
            ),
        )

    # -- decode_latents ------------------------------------------------

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode packed latents into RGB or opt-in RGBA output.

        Unpack to ``[B, C, 1, H, W]``, denormalize with the VAE's per-channel
        mean/std, decode to RGBA, drop the singleton temporal frame, then
        composite alpha over white only in RGB mode. RGBA output keeps all four
        channels through generation and reward transport.
        """
        pipe = self.pipeline
        vae = pipe.vae
        vae_scale_factor = pipe.vae_scale_factor
        height = self._decode_height
        width = self._decode_width
        z_dim = vae.config.z_dim
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1)
        latents_std = torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1)

        def _transform(batch: torch.Tensor) -> torch.Tensor:
            unpacked = pipe._unpack_latents(batch, height, width, vae_scale_factor)
            unpacked = unpacked.to(vae.dtype)
            mean = latents_mean.to(unpacked.device, unpacked.dtype)
            std = latents_std.to(unpacked.device, unpacked.dtype)
            return unpacked * std + mean

        def _composite_over_white(image: torch.Tensor) -> torch.Tensor:
            # ``postprocess`` returns [0, 1] tensors; channel 4 is alpha.
            if image.shape[1] != 4 or self._output_mode == "rgba":
                return image
            rgb, alpha = image[:, :3], image[:, 3:4]
            return rgb * alpha + (1.0 - alpha)

        decoder = ChunkedLatentDecoder(
            LatentDecodePlan(
                prepare_latents=_transform,
                vae_decode=lambda batch: vae.decode(batch, return_dict=False)[0],
                # Video-style VAE returns [B, 4, 1, H, W]; drop the temporal frame.
                prepare_decoded=lambda decoded: decoded[:, :, 0],
                postprocess=lambda image: _composite_over_white(
                    pipe.image_processor.postprocess(image, output_type="pt"),
                ),
                output_layout="image_bchw",
                decode_batch_size=getattr(pipe, "decode_batch_size", None),
            ),
        )
        return decoder(latents)


class QwenImage21ReplayModel(DiffusersReplayModelBase, QwenImage21Model):
    """Replay-only Qwen-Image-2.1 model that owns no prompt encoder, VAE, or pipeline."""


__all__ = ["QwenImage21Model", "QwenImage21ReplayModel", "QwenImage21SamplingState"]
