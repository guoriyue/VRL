"""CogVideoX t2v diffusers-backed model.

Diffusion implementation for Zhipu CogVideoX (2b/5b DiT).
The generation helper flow mirrors every diffusion family:

    encode_prompt -> prepare_sampling -> forward_step xN -> decode_latents

CogVideoX specifics vs Wan (the reference video family):
- NOT flow-matching: a v-prediction DDPM ladder (CogVideoXDDIMScheduler with
  snr_shift + zero-SNR rescale). Rollout/replay log-probs run through
  ``sde_type="ddim"`` (``vrl/math/diffusion/ddim.py``) — experiments MUST set
  ``sampling.sde_type: ddim``; ``prepare_sampling`` fails loud when the
  scheduler carries ``clip_sample=True`` (the Gaussian ignores clipping).
- Latents are ``[B, F, C, H, W]`` — the FRAME axis precedes channels — and
  ``prepare_latents`` takes ``num_frames`` third, not last. The layout rides
  the whole seam untouched (the SDE math is elementwise); only
  ``decode_latents`` permutes to the VAE's ``[B, C, F, H, W]``.
- RoPE is computed OUTSIDE the transformer (the only such family): 5b sets
  ``use_rotary_positional_embeddings`` and takes ``image_rotary_emb`` as a
  forward kwarg; 2b passes None. The freqs derive deterministically from the
  request shape + transformer config, so both rollout and replay recompute
  them via ``cogvideox_rotary_embeds`` (no pipeline needed on the replay
  side).
- T5-XXL single encoder, NO attention mask (fixed 226-token padding); true
  batched CFG with uncond rows first; the transformer takes the raw integer
  timestep.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from typing import Any

import torch

from vrl.generation.diffusion.layout import VideoGenerationRequest
from vrl.models.diffusion import (
    DiffusersPipelineModelBase,
    DiffusersReplayModelBase,
    DiffusionSamplingStateBase,
    diffusers_pipeline_dtypes,
)
from vrl.models.diffusion.common import (
    ChunkedLatentDecoder,
    DiffusionBackboneCaller,
    DiffusionBackboneInput,
    DiffusionBackboneRunnerBase,
    DiffusionBranch,
    LatentDecodePlan,
    expand_batch_timestep,
    pack_eval_timestep,
)
from vrl.models.diffusion.common.lora import LoraModelMixin
from vrl.models.diffusion.common.tensors import require_tensor
from vrl.models.interfaces.runtime import ModelBuild


def cogvideox_rotary_embeds(
    transformer_config: Any,
    *,
    height: int,
    width: int,
    num_latent_frames: int,
    vae_scale_factor_spatial: int,
    device: Any,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Recompute CogVideoX's external RoPE freqs from the request shape.

    Mirrors ``CogVideoXPipeline._prepare_rotary_positional_embeddings`` so the
    REPLAY side (which owns no pipeline) derives the identical freqs from the
    transformer config + stored shape. Returns None for checkpoints without
    rotary embeddings (2b).
    """
    if not getattr(transformer_config, "use_rotary_positional_embeddings", False):
        return None
    from diffusers.models.embeddings import get_3d_rotary_pos_embed
    from diffusers.pipelines.cogvideo.pipeline_cogvideox import (
        get_resize_crop_region_for_grid,
    )

    p = transformer_config.patch_size
    p_t = getattr(transformer_config, "patch_size_t", None)
    grid_height = height // (vae_scale_factor_spatial * p)
    grid_width = width // (vae_scale_factor_spatial * p)
    base_size_width = transformer_config.sample_width // p
    base_size_height = transformer_config.sample_height // p

    if p_t is None:
        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width),
            base_size_width,
            base_size_height,
        )
        return get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_latent_frames,
            device=device,
        )
    base_num_frames = (num_latent_frames + p_t - 1) // p_t
    return get_3d_rotary_pos_embed(
        embed_dim=transformer_config.attention_head_dim,
        crops_coords=None,
        grid_size=(grid_height, grid_width),
        temporal_size=base_num_frames,
        grid_type="slice",
        max_size=(base_size_height, base_size_width),
        device=device,
    )


@dataclass
class CogVideoXSamplingState(DiffusionSamplingStateBase):
    """Private CogVideoX sampling state. Engine MUST NOT introspect."""

    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None
    do_cfg: bool
    height: int
    width: int
    vae_scale_factor_spatial: int


class CogVideoXModel(LoraModelMixin, DiffusersPipelineModelBase, DiffusionBackboneRunnerBase):
    """Diffusers-backed CogVideoX t2v model (v-prediction DDPM family)."""

    cfg_mode = "batched_cfg"
    cfg_base = "uncond"

    def build_branch(
        self,
        request: DiffusionBackboneInput,
        branch: str,
    ) -> DiffusionBranch:
        """Map CogVideoX transformer kwargs into the shared backbone contract."""
        if branch == "cond":
            embeds = request.prompt_embeds
        else:
            embeds = require_tensor(
                request.negative_prompt_embeds,
                "negative_prompt_embeds",
            )
        return DiffusionBranch(
            hidden_states=request.hidden_states,
            timestep=request.timestep,
            encoder_hidden_states=embeds,
            extra_kwargs={"image_rotary_emb": request.extra.get("image_rotary_emb")},
        )

    # -- backend ownership (called by runtime, not by collectors) -------

    @classmethod
    def from_build(cls, build: ModelBuild) -> CogVideoXModel:
        """Load the diffusers CogVideoX pipeline + freeze non-trainable modules."""
        from diffusers import CogVideoXPipeline

        model_dtype = build.parameter_dtype
        prompt_encoder_dtype, load_kwargs = diffusers_pipeline_dtypes(build, model_dtype)
        pipeline = CogVideoXPipeline.from_pretrained(
            build.model_name_or_path,
            **load_kwargs,
        )
        pipeline.vae.requires_grad_(False)
        text_encoder = getattr(pipeline, "text_encoder", None)
        if text_encoder is not None:
            # T5-XXL (~9.5 GB bf16); park on CPU (Qwen-Image discipline).
            text_encoder.requires_grad_(False)
            text_encoder.to("cpu", dtype=prompt_encoder_dtype)
        pipeline.vae.to(build.device, dtype=torch.float32)
        return cls(
            pipeline=pipeline,
            device=build.device,
        )

    def _encoder_device(self) -> Any:
        """Device the frozen T5 encoder lives on (CPU when offloaded)."""
        enc = getattr(self.pipeline, "text_encoder", None)
        if enc is not None:
            try:
                return next(enc.parameters()).device
            except StopIteration:
                pass
        return self.device

    # -- encode_prompt -------------------------------------------------

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode prompt via T5-XXL (fixed-length padded sequence, no mask)."""
        max_seq = kwargs.get("max_sequence_length", 226)
        guidance_scale = kwargs.get("guidance_scale", 6.0)
        do_cfg = guidance_scale > 1.0
        neg = negative_prompt if negative_prompt is not None else ""

        prompt_embeds, negative_prompt_embeds = self.pipeline.encode_prompt(
            prompt=prompt,
            negative_prompt=neg,
            do_classifier_free_guidance=do_cfg,
            num_videos_per_prompt=1,
            max_sequence_length=max_seq,
            device=self._encoder_device(),
        )

        td = self.transformer.dtype
        result: dict[str, Any] = {
            "prompt_embeds": prompt_embeds.to(self.device, dtype=td),
        }
        if do_cfg and negative_prompt_embeds is not None:
            result["negative_prompt_embeds"] = negative_prompt_embeds.to(
                self.device,
                dtype=td,
            )
        return result

    # -- prepare_sampling ----------------------------------------------

    def prepare_sampling(
        self,
        request: VideoGenerationRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> CogVideoXSamplingState:
        """Build the per-request BFCHW-latent SamplingState for a denoise loop."""
        del kwargs
        pipe = self.pipeline
        device = self.device
        scheduler = pipe.scheduler

        if getattr(scheduler.config, "clip_sample", False):
            raise ValueError(
                "CogVideoX RL rollouts need clip_sample=false on the scheduler "
                "(the ddim log-prob Gaussian ignores clipping); the shipped "
                "THUDM configs already set it — check the checkpoint.",
            )

        prompt_embeds = encoded["prompt_embeds"]
        negative_prompt_embeds = encoded.get("negative_prompt_embeds")

        scheduler.set_timesteps(request.num_steps, device=device)
        timesteps = scheduler.timesteps

        seed = request.seed if request.seed is not None else random.randint(0, sys.maxsize)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        num_channels_latents = pipe.transformer.config.in_channels
        batch_size = prompt_embeds.shape[0]
        # CogVideoX prepare_latents takes num_frames THIRD (before h/w).
        latents = pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            request.frame_count,
            request.height,
            request.width,
            torch.float32,
            device,
            generator,
            None,
        )

        do_cfg = request.guidance_scale > 1.0 and negative_prompt_embeds is not None

        return CogVideoXSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=scheduler,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=request.guidance_scale,
            do_cfg=do_cfg,
            height=int(request.height),
            width=int(request.width),
            vae_scale_factor_spatial=int(pipe.vae_scale_factor_spatial),
        )

    # -- forward_step --------------------------------------------------

    def forward_step(
        self,
        state: CogVideoXSamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        """CogVideoX transformer forward (batched CFG, external RoPE)."""
        t = state.timesteps[step_idx]
        bsz = state.latents.shape[0]
        td = self._transformer_dtype()

        latent_input = state.latents.to(td)
        # Raw integer timestep — the sinusoidal proj consumes it directly.
        timestep_batch = expand_batch_timestep(t, bsz).to(
            device=latent_input.device,
        )
        rope = cogvideox_rotary_embeds(
            self.transformer.config,
            height=state.height,
            width=state.width,
            num_latent_frames=state.latents.shape[1],
            vae_scale_factor_spatial=state.vae_scale_factor_spatial,
            device=latent_input.device,
        )
        negative_embeds = (
            None if state.negative_prompt_embeds is None else state.negative_prompt_embeds.to(td)
        )
        output = DiffusionBackboneCaller(
            self.transformer,
            self,
        )(
            DiffusionBackboneInput(
                hidden_states=latent_input,
                timestep=timestep_batch,
                prompt_embeds=state.prompt_embeds.to(td),
                negative_prompt_embeds=negative_embeds,
                guidance_scale=state.guidance_scale,
                do_cfg=state.do_cfg,
                output_dtype=td,
                extra={"image_rotary_emb": rope},
            ),
        )
        return output.as_dict()

    # -- collector boundary --------------------------------------------

    def export_batch_context(self, state: CogVideoXSamplingState) -> dict[str, Any]:
        """Project CogVideoX sampling state into trajectory context."""
        return {
            "guidance_scale": state.guidance_scale,
            "cfg": state.do_cfg,
            "height": state.height,
            "width": state.width,
            "vae_scale_factor_spatial": state.vae_scale_factor_spatial,
        }

    def export_replay_tensors(self, state: CogVideoXSamplingState) -> dict[str, Any]:
        """Project CogVideoX sampling state into per-sample trajectory tensors."""
        tensors: dict[str, Any] = {"prompt_embeds": state.prompt_embeds}
        if state.negative_prompt_embeds is not None:
            tensors["negative_prompt_embeds"] = state.negative_prompt_embeds
        return tensors

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> CogVideoXSamplingState:
        """Rebuild CogVideoXSamplingState from a batch slice for the eval path.

        RoPE freqs are NOT stored: ``forward_step`` recomputes them from the
        transformer config + the shape carried in batch context.
        """
        ts = replay_tensors["timesteps"]
        timesteps = pack_eval_timestep(ts, step_idx)
        return CogVideoXSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=None,  # not needed for forward_step (no scheduler.step here)
            prompt_embeds=replay_tensors["prompt_embeds"],
            negative_prompt_embeds=replay_tensors.get("negative_prompt_embeds"),
            guidance_scale=batch_context["guidance_scale"],
            do_cfg=batch_context["cfg"]
            and replay_tensors.get("negative_prompt_embeds") is not None,
            height=int(batch_context["height"]),
            width=int(batch_context["width"]),
            vae_scale_factor_spatial=int(batch_context["vae_scale_factor_spatial"]),
        )

    # -- decode_latents ------------------------------------------------

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode BFCHW latents -> video [B, C, T, H, W] via the CogVideoX VAE."""
        pipe = self.pipeline
        vae = pipe.vae
        scaling = float(pipe.vae_scaling_factor_image)

        def _transform(chunk: torch.Tensor) -> torch.Tensor:
            # [B, F, C, H, W] -> the VAE's [B, C, F, H, W], then unscale.
            return chunk.permute(0, 2, 1, 3, 4).to(vae.dtype) / scaling

        decoder = ChunkedLatentDecoder(
            LatentDecodePlan(
                prepare_latents=_transform,
                vae_decode=lambda chunk: vae.decode(chunk, return_dict=False)[0],
                postprocess=lambda video: pipe.video_processor.postprocess_video(
                    video,
                    output_type="pt",
                ),
                output_layout="video_btchw",
                decode_batch_size=getattr(pipe, "decode_batch_size", None),
            ),
        )
        return decoder(latents)


class CogVideoXReplayModel(DiffusersReplayModelBase, CogVideoXModel):
    """Replay-only CogVideoX model that owns no prompt encoder, VAE, or pipeline."""


__all__ = [
    "CogVideoXModel",
    "CogVideoXReplayModel",
    "CogVideoXSamplingState",
    "cogvideox_rotary_embeds",
]
