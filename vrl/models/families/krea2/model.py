"""Krea 2 t2i diffusers-backed model.

Diffusion implementation for Krea AI's Krea 2 (Raw: the undistilled, CFG-guided
checkpoint). The generation helper flow mirrors every diffusion family:

    encode_prompt -> prepare_sampling -> forward_step xN -> decode_latents

``Krea2PipelineModel`` runs that flow. ``Krea2Model`` — the transformer, its
scheduler and ``forward_step`` — is what replay runs; the pipeline class adds
the text encoder, latent packing and VAE decode on top of it.

Krea 2 specifics vs SD3 (the reference family):
- A 12B single-stream MMDiT (``Krea2Transformer2DModel``) in the Qwen-Image f8
  video-VAE latent space. Latents are PACKED ``[B, (h/2)*(w/2), 64]`` like FLUX and
  stay packed through the denoise loop; only ``decode_latents`` unpacks them (the
  unpacked latent carries a singleton temporal dim ``[B, C, 1, H, W]``).
- Text conditioning is a STACK of hidden states tapped from 12 layers of
  Qwen3-VL-4B: ``[B, S, 12, 2560]`` plus a bool mask, no pooled vector. Prompts
  are padded to a fixed ``max_sequence_length`` inside the chat template, so
  ``S`` is constant for a run and padded keys are masked everywhere.
- Rotary ``position_ids`` ``[S + img_len, 3]`` are batch-shared and deterministic
  from the prompt length and the latent grid, so replay rebuilds them instead of
  storing per-sample copies.
- The transformer takes flow time in ``[0, 1]``, so ``forward_step`` feeds
  ``t / 1000`` (matching diffusers' ``Krea2Pipeline`` loop).
- TRUE classifier-free guidance as two forwards. Upstream writes the combine as
  ``cond + g * (cond - uncond)`` with guidance on when ``g > 0``; VRL's
  ``guidance_scale`` is the standard scale ``s = 1 + g`` (``uncond + s * (cond -
  uncond)``, on when ``s > 1``), the same velocity. Krea's Raw default ``g=3.5``
  is therefore ``sampling.guidance_scale: 4.5`` here.
- Dynamic-shifting FlowMatch scheduler: ``mu`` derives from the packed token count
  with Krea's ``max_image_seq_len=6400``, so the grid is set in ``prepare_sampling``
  and rebuilt for replay in ``prepare_replay``.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from typing import Any

import torch

from vrl.generation.types import DenoiseRequest
from vrl.models.steps.denoise import (
    DiffusersModelBase,
    DiffusersPipelineModelBase,
    GuidedDenoiseSamplingStateBase,
)
from vrl.models.steps.denoise.common import (
    ChunkedLatentDecoder,
    denormalize_video_vae_latents,
    expand_batch_timestep,
    pack_eval_timestep,
    predict_with_cfg,
    set_mu_shifted_timesteps,
)

# f8 VAE x 2x2 patch pack: one transformer token covers 16x16 pixels.
PIXELS_PER_TOKEN = 16


@dataclass
class Krea2SamplingState(GuidedDenoiseSamplingStateBase):
    """Private Krea 2 sampling state. Engine MUST NOT introspect."""

    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None
    negative_prompt_embeds_mask: torch.Tensor | None
    position_ids: torch.Tensor
    do_cfg: bool
    height: int
    width: int


class Krea2Model(DiffusersModelBase):
    """Krea 2 transformer, scheduler and per-step math: the model replay runs."""

    def packed_token_count(self, height: int, width: int) -> int:
        """Krea 2 packs an 8x VAE + 2x2 patch grid: ``(H // 16) * (W // 16)`` tokens."""
        return (height // PIXELS_PER_TOKEN) * (width // PIXELS_PER_TOKEN)

    def _set_dynamic_timesteps(self, num_steps: int, image_seq_len: int, device: Any) -> Any:
        """Set Krea 2 timesteps with the resolution-derived ``mu`` (diffusers parity)."""
        from diffusers.pipelines.krea2.pipeline_krea2 import calculate_shift

        return set_mu_shifted_timesteps(
            self.scheduler,
            num_steps=num_steps,
            image_seq_len=image_seq_len,
            device=device,
            calculate_shift=calculate_shift,
        )

    @staticmethod
    def _position_ids(text_seq_len: int, height: int, width: int, device: Any) -> torch.Tensor:
        """The batch-shared ``[S + img_len, 3]`` rotary grid of diffusers' Krea2Pipeline.

        A static method of the pipeline class, so the pipeline-less replay model
        rebuilds the grid without owning a pipeline instance.
        """
        from diffusers.pipelines.krea2.pipeline_krea2 import Krea2Pipeline

        return Krea2Pipeline.prepare_position_ids(
            text_seq_len, height // PIXELS_PER_TOKEN, width // PIXELS_PER_TOKEN, device
        )

    # -- forward_step --------------------------------------------------

    def forward_step(
        self,
        state: Krea2SamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        """Krea 2 transformer forward (separate-branch CFG when do_cfg).

        Returns noise_pred (packed); the caller owns scheduler.step / SDE.
        """
        t = state.timesteps[step_idx]
        bsz = state.latents.shape[0]
        td = self._transformer_dtype()

        latent_input = state.latents.to(td)
        # The transformer takes flow time in [0, 1]; if t is already [B] (eval
        # path packs [1, B]) expand is a no-op.
        timestep_batch = (
            expand_batch_timestep(t, bsz).to(device=latent_input.device, dtype=td) / 1000.0
        )
        cond = {
            "hidden_states": latent_input,
            "encoder_hidden_states": state.prompt_embeds.to(td),
            "timestep": timestep_batch,
            # Rotary positions stay float32 (computed in float).
            "position_ids": state.position_ids.to(device=latent_input.device, dtype=torch.float32),
            "encoder_attention_mask": state.prompt_embeds_mask.to(torch.bool),
        }
        uncond = None
        if state.do_cfg:
            uncond = {
                **cond,
                "encoder_hidden_states": state.negative_prompt_embeds.to(td),
                "encoder_attention_mask": state.negative_prompt_embeds_mask.to(torch.bool),
            }
        noise_pred = predict_with_cfg(
            self.transformer,
            cond,
            uncond,
            guidance_scale=state.guidance_scale,
            # ``position_ids`` is one batch-shared [S + img_len, 3] tensor; the
            # batched packer would concatenate it along its sequence axis.
            batched=False,
            dtype=td,
        )
        return {"noise_pred": noise_pred}

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> Krea2SamplingState:
        """Rebuild Krea2SamplingState from a batch slice for the eval forward path.

        Rebuilds the batch-shared rotary grid from the stored prompt length and the
        spatial shape, pipeline-free.
        """
        ts = replay_tensors["timesteps"]
        timesteps = pack_eval_timestep(ts, step_idx)
        prompt_embeds = replay_tensors["prompt_embeds"]
        negative_prompt_embeds = replay_tensors.get("negative_prompt_embeds")
        height = int(batch_context["height"])
        width = int(batch_context["width"])
        return Krea2SamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=None,  # not needed for forward_step (no scheduler.step here)
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=replay_tensors["prompt_embeds_mask"],
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=replay_tensors.get("negative_prompt_embeds_mask"),
            position_ids=self._position_ids(
                prompt_embeds.shape[1], height, width, prompt_embeds.device
            ),
            guidance_scale=batch_context["guidance_scale"],
            do_cfg=batch_context["cfg"] and negative_prompt_embeds is not None,
            height=height,
            width=width,
        )


class Krea2PipelineModel(DiffusersPipelineModelBase, Krea2Model):
    """Krea 2 plus the diffusers pipeline that sampling needs.

    The pipeline brings prompt encoding, latent packing and VAE decode; this
    class also exports the trajectory tensors replay reads back.
    """

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

    # -- encode_prompt -------------------------------------------------

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode prompt via Qwen3-VL (stacked tapped hidden states + bool mask).

        Returns the conditional stack/mask and, when CFG is active, the
        unconditional pair for the separate branch. Upstream's default negative
        prompt is the empty string through the same template.
        """
        max_seq = kwargs.get("max_sequence_length", 512)
        guidance_scale = kwargs.get("guidance_scale", 4.5)
        do_cfg = guidance_scale > 1.0
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        prompt_embeds, prompt_embeds_mask = self._encode(prompts, max_seq)
        result: dict[str, Any] = {
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
        }
        if do_cfg:
            negative = "" if negative_prompt is None else negative_prompt
            negatives = [negative] * len(prompts) if isinstance(negative, str) else list(negative)
            negative_prompt_embeds, negative_prompt_embeds_mask = self._encode(negatives, max_seq)
            result["negative_prompt_embeds"] = negative_prompt_embeds
            result["negative_prompt_embeds_mask"] = negative_prompt_embeds_mask
        return result

    def _encode(self, prompts: list[str], max_sequence_length: int) -> tuple[Any, Any]:
        """Tap the encoder on its own device (CPU when resident there), then move the
        ``[B, S, layers, D]`` stack and mask onto the transformer device and dtype."""
        embeds, mask = self.pipeline.get_text_hidden_states(
            prompts, max_sequence_length, self._encoder_device()
        )
        return embeds.to(self.device, dtype=self.transformer.dtype), mask.to(self.device)

    # -- prepare_sampling ----------------------------------------------

    def prepare_sampling(
        self,
        request: DenoiseRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> Krea2SamplingState:
        """Build the per-request packed-latent SamplingState for a denoise loop."""
        del kwargs
        pipe = self.pipeline
        device = self.device
        if request.height % PIXELS_PER_TOKEN or request.width % PIXELS_PER_TOKEN:
            # The upstream pipeline silently rounds up; a trajectory must carry
            # the geometry the config asked for.
            raise ValueError(
                f"Krea 2 height and width must be multiples of {PIXELS_PER_TOKEN}; "
                f"got {request.height}x{request.width}",
            )

        prompt_embeds = encoded["prompt_embeds"]
        negative_prompt_embeds = encoded.get("negative_prompt_embeds")

        seed = request.seed if request.seed is not None else random.randint(0, sys.maxsize)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        batch_size = prompt_embeds.shape[0]
        # 2x2 patches are packed into channels, so the transformer's in_channels
        # is 4x the unpacked latent channel count.
        num_channels_latents = pipe.transformer.config.in_channels // 4
        latents = pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            request.height,
            request.width,
            torch.float32,
            device,
            generator,
        )
        position_ids = self._position_ids(
            prompt_embeds.shape[1], request.height, request.width, device
        )

        # Dynamic-shifting timesteps depend on the packed image sequence length.
        timesteps = self._set_dynamic_timesteps(
            request.num_steps,
            latents.shape[1],
            device,
        )

        self._decode_height = int(request.height)
        self._decode_width = int(request.width)

        return Krea2SamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=pipe.scheduler,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=encoded["prompt_embeds_mask"],
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=encoded.get("negative_prompt_embeds_mask"),
            position_ids=position_ids,
            guidance_scale=request.guidance_scale,
            do_cfg=negative_prompt_embeds is not None,
            height=int(request.height),
            width=int(request.width),
        )

    # -- collector boundary --------------------------------------------

    def export_batch_context(self, state: Krea2SamplingState) -> dict[str, Any]:
        """Project Krea 2 sampling state into trajectory context.

        ``position_ids`` is batch-shared and deterministic from the prompt length
        and spatial shape, so only scalars travel here.
        """
        return {
            "guidance_scale": state.guidance_scale,
            "cfg": state.do_cfg,
            "height": state.height,
            "width": state.width,
        }

    def export_replay_tensors(self, state: Krea2SamplingState) -> dict[str, Any]:
        """Project Krea 2 sampling state into per-sample trajectory tensors.

        The negative pair is stored only under CFG, so ``restore`` reads it with
        ``.get`` and the no-CFG path stays tensor-free.
        """
        tensors: dict[str, Any] = {
            "prompt_embeds": state.prompt_embeds,
            "prompt_embeds_mask": state.prompt_embeds_mask,
            "latents_clean": state.latents.detach(),
        }
        if state.negative_prompt_embeds is not None:
            tensors["negative_prompt_embeds"] = state.negative_prompt_embeds
            tensors["negative_prompt_embeds_mask"] = state.negative_prompt_embeds_mask
        return tensors

    # -- decode_latents ------------------------------------------------

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode packed latents -> image via the Qwen-Image video VAE.

        Unpack to ``[B, C, 1, H, W]``, denormalize with the VAE's per-channel
        mean/std, decode, then drop the singleton temporal frame.
        """
        pipe = self.pipeline
        vae = pipe.vae
        height = self._decode_height
        width = self._decode_width

        def _transform(batch: torch.Tensor) -> torch.Tensor:
            return denormalize_video_vae_latents(pipe._unpack_latents(batch, height, width), vae)

        decoder = ChunkedLatentDecoder(
            prepare_latents=_transform,
            vae_decode=lambda batch: vae.decode(batch, return_dict=False)[0],
            # Video VAE returns [B, C, 1, H, W]; drop the temporal frame.
            prepare_decoded=lambda decoded: decoded[:, :, 0],
            postprocess=lambda image: pipe.image_processor.postprocess(
                image,
                output_type="pt",
            ),
            output_layout="image_bchw",
        )
        return decoder(latents)


__all__ = ["PIXELS_PER_TOKEN", "Krea2Model", "Krea2PipelineModel", "Krea2SamplingState"]
