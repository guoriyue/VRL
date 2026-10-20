"""MOSS-Image t2i model (OpenMOSS 5B single-stream DiT).

The generation helper flow mirrors every diffusion family:

    encode_prompt -> prepare_sampling -> forward_step xN -> decode_latents

``MossImagePipelineModel`` runs that flow. ``MossImageModel`` — the transformer,
its scheduler and ``forward_step`` — is what replay runs; the pipeline class adds
the text encoder, latent preparation and VAE decode on top of it.

MOSS-Image specifics vs SD3 (the reference family):
- The DiT, its scheduler and the reference pipeline are not in diffusers; the
  checkpoint ships them as remote code, vendored in ``pipeline_moss_image.py``. The registry
  names them by import path and the shared component loaders resolve it.
- Latents are UNPACKED 4D ``[B, 128, H/16, W/16]``: the FLUX.2 VAE's 32-channel
  f8 latent with its 2x2 patches folded into channels. Noise is drawn directly in
  that shape in fp32; only ``decode_latents`` unfolds it (``latents_to_vae_input``).
- The text encoder is Qwen3.5-9B: ``encode_prompt`` returns a SEQUENCE embed plus
  a bool mask and NO pooled vector. Every prompt is encoded unpadded on its own
  (the released recipe; padding changes Qwen's gated-delta-net state) and padded
  to the batch's longest row afterwards.
- TRUE classifier-free guidance with a twist: the unconditional branch skips DiT
  block 22 (upstream ``uncond_skip_layers=(22,)``). The branches therefore differ
  in a non-tensor kwarg and run as two forwards; both predictions are cast to fp32
  before ``uncond + g * (cond - uncond)``, as upstream combines them.
- ``forward_step`` feeds the raw scheduler timestep (0..1000); the transformer
  builds its own sinusoid from it.
- ``MOSSImageScheduler`` is a static flow-match schedule (shift 5 snapped to the
  1000-point training grid, no resolution ``mu``), so replay loads it eagerly.
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
    expand_batch_timestep,
    pack_eval_timestep,
    predict_with_cfg,
)

TRANSFORMER_CLS = "vrl.models.families.moss_image.pipeline_moss_image:MOSSImageTransformer"
SCHEDULER_CLS = "vrl.models.families.moss_image.pipeline_moss_image:MOSSImageScheduler"
# f8 VAE x 2x2 patch fold: one DiT latent cell covers 16x16 pixels.
PIXELS_PER_LATENT = 16
# Released sampling recipe: the negative branch skips this DiT block (zero-based).
UNCOND_SKIP_LAYERS: tuple[int, ...] = (22,)


@dataclass
class MossImageSamplingState(GuidedDenoiseSamplingStateBase):
    """Private MOSS-Image sampling state. Engine MUST NOT introspect."""

    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None
    negative_prompt_embeds_mask: torch.Tensor | None
    do_cfg: bool
    uncond_skip_layers: tuple[int, ...]


class MossImageModel(DiffusersModelBase):
    """MOSS-Image transformer, scheduler and per-step math: the model replay runs."""

    def forward_step(
        self,
        state: MossImageSamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        """MOSS-Image transformer forward (separate-branch CFG with the block skip).

        Returns a fp32 ``noise_pred``; the caller owns scheduler.step / SDE.
        """
        t = state.timesteps[step_idx]
        bsz = state.latents.shape[0]
        td = self._transformer_dtype()

        latent_input = state.latents.to(td)
        # The transformer rebuilds its sinusoid from the raw scheduler value in
        # fp32; if t is already [B] (eval path packs [1, B]) expand is a no-op.
        timestep_batch = expand_batch_timestep(t, bsz).to(
            device=latent_input.device, dtype=torch.float32
        )
        cond = {
            "hidden_states": latent_input,
            "timestep": timestep_batch,
            "encoder_hidden_states": state.prompt_embeds.to(td),
            "encoder_attention_mask": state.prompt_embeds_mask.to(torch.bool),
        }
        uncond = None
        if state.do_cfg:
            uncond = {
                **cond,
                "encoder_hidden_states": state.negative_prompt_embeds.to(td),
                "encoder_attention_mask": state.negative_prompt_embeds_mask.to(torch.bool),
            }
            if state.uncond_skip_layers:
                uncond["skip_layers"] = state.uncond_skip_layers
        noise_pred = predict_with_cfg(
            self.transformer,
            cond,
            uncond,
            guidance_scale=state.guidance_scale,
            # The negative branch carries a kwarg the positive one lacks.
            batched=False,
            # Upstream combines the two bf16 predictions in fp32.
            dtype=torch.float32,
        )
        return {"noise_pred": noise_pred}

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> MossImageSamplingState:
        """Rebuild MossImageSamplingState from a batch slice for the eval forward path."""
        ts = replay_tensors["timesteps"]
        timesteps = pack_eval_timestep(ts, step_idx)
        negative_prompt_embeds = replay_tensors.get("negative_prompt_embeds")
        return MossImageSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=None,  # not needed for forward_step (no scheduler.step here)
            prompt_embeds=replay_tensors["prompt_embeds"],
            prompt_embeds_mask=replay_tensors["prompt_embeds_mask"],
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=replay_tensors.get("negative_prompt_embeds_mask"),
            guidance_scale=batch_context["guidance_scale"],
            do_cfg=batch_context["cfg"] and negative_prompt_embeds is not None,
            uncond_skip_layers=tuple(batch_context["uncond_skip_layers"]),
        )


class MossImagePipelineModel(DiffusersPipelineModelBase, MossImageModel):
    """MOSS-Image plus the vendored reference pipeline that sampling needs.

    The pipeline brings prompt encoding and the VAE; this class owns latent
    preparation, decode and the trajectory tensors replay reads back.
    """

    @classmethod
    def from_build(cls, build: Any) -> MossImagePipelineModel:
        """Assemble the vendored pipeline from the checkpoint's components.

        ``DiffusionPipeline.from_pretrained`` cannot build this checkpoint
        without fetching its remote code, so each component loads through the
        shared loaders and the vendored classes; freeze and placement then follow
        the diffusers-family rule (fp32 VAE, encoder dtype, ``cpu_resident``).
        """
        from diffusers import AutoencoderKLFlux2
        from transformers import AutoModel, AutoTokenizer

        from vrl.models.families.moss_image.pipeline_moss_image import MOSSImagePipeline
        from vrl.models.loader import load_diffusers_scheduler, load_diffusers_transformer

        prompt_encoder_dtype, _ = cls._pipeline_load_dtypes(build)
        path = build.model_name_or_path
        kwargs = build.pretrained_kwargs
        pipeline = MOSSImagePipeline(
            transformer=load_diffusers_transformer(build, TRANSFORMER_CLS),
            scheduler=load_diffusers_scheduler(build, SCHEDULER_CLS),
            text_encoder=AutoModel.from_pretrained(
                path, subfolder="text_encoder", dtype=prompt_encoder_dtype, **kwargs
            ),
            tokenizer=AutoTokenizer.from_pretrained(path, subfolder="tokenizer", **kwargs),
            vae=AutoencoderKLFlux2.from_pretrained(
                path, subfolder="vae", torch_dtype=torch.float32, **kwargs
            ),
        )
        cls.freeze_pipeline_components(pipeline)
        cpu_resident = cls.place_pipeline_components(
            pipeline,
            build,
            prompt_encoder_dtype=prompt_encoder_dtype,
        )
        model = cls(pipeline=pipeline, device=build.device)
        model._cpu_resident = cpu_resident
        return model

    # -- encode_prompt -------------------------------------------------

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode prompt via Qwen3.5 (sequence embeds + bool mask, no pooled).

        Returns the conditional embeds/mask and, when CFG is active, the
        unconditional pair for the separate branch. The released recipe's
        negative prompt is the empty string run through the chat template.
        """
        max_seq = kwargs.get("max_sequence_length", 4096)
        guidance_scale = kwargs.get("guidance_scale", 3.0)
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
        """Run the vendored encoder on the frozen encoder's device (CPU when resident
        there) and move the padded rows onto the transformer device and dtype."""
        embeds, mask = self.pipeline.encode_prompt(
            prompts, max_sequence_length, device=self._encoder_device()
        )
        return embeds.to(self.device, dtype=self.transformer.dtype), mask.to(self.device)

    # -- prepare_sampling ----------------------------------------------

    def prepare_sampling(
        self,
        request: DenoiseRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> MossImageSamplingState:
        """Build the per-request 4D-latent SamplingState for a denoise loop."""
        del kwargs
        from diffusers.utils.torch_utils import randn_tensor

        pipe = self.pipeline
        device = self.device
        if request.height % PIXELS_PER_LATENT or request.width % PIXELS_PER_LATENT:
            raise ValueError(
                f"MOSS-Image height and width must be multiples of {PIXELS_PER_LATENT}; "
                f"got {request.height}x{request.width}",
            )

        prompt_embeds = encoded["prompt_embeds"]
        negative_prompt_embeds = encoded.get("negative_prompt_embeds")

        pipe.scheduler.set_timesteps(request.num_steps, device=device)
        timesteps = pipe.scheduler.timesteps
        if len(timesteps) != request.num_steps:
            # The nearest-grid snap merges steps that land on one training sigma.
            raise ValueError(
                f"MOSS-Image's training-grid schedule holds {len(timesteps)} distinct "
                f"sigmas for sampling.num_steps={request.num_steps}; lower the step count",
            )

        seed = request.seed if request.seed is not None else random.randint(0, sys.maxsize)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        shape = (
            prompt_embeds.shape[0],
            pipe.transformer.config.latent_channels,
            request.height // PIXELS_PER_LATENT,
            request.width // PIXELS_PER_LATENT,
        )
        # Sample in fp32: promoting bf16 noise afterward cannot restore its precision.
        latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)

        return MossImageSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=pipe.scheduler,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=encoded["prompt_embeds_mask"],
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=encoded.get("negative_prompt_embeds_mask"),
            guidance_scale=request.guidance_scale,
            do_cfg=negative_prompt_embeds is not None,
            uncond_skip_layers=UNCOND_SKIP_LAYERS,
        )

    # -- collector boundary --------------------------------------------

    def export_batch_context(self, state: MossImageSamplingState) -> dict[str, Any]:
        """Project MOSS-Image sampling state into trajectory context."""
        return {
            "guidance_scale": state.guidance_scale,
            "cfg": state.do_cfg,
            "uncond_skip_layers": list(state.uncond_skip_layers),
        }

    def export_replay_tensors(self, state: MossImageSamplingState) -> dict[str, Any]:
        """Project MOSS-Image sampling state into per-sample trajectory tensors.

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
        """Decode 4D latents -> image via the FLUX.2 VAE (denormalize, unfold, decode)."""
        from vrl.models.families.moss_image.pipeline_moss_image import latents_to_vae_input

        pipe = self.pipeline
        vae = pipe.vae
        decoder = ChunkedLatentDecoder(
            prepare_latents=lambda batch: latents_to_vae_input(batch, vae),
            vae_decode=lambda batch: vae.decode(batch, return_dict=False)[0].clamp(-1, 1),
            postprocess=lambda image: pipe.image_processor.postprocess(
                image,
                output_type="pt",
            ),
            output_layout="image_bchw",
        )
        return decoder(latents)


__all__ = [
    "PIXELS_PER_LATENT",
    "SCHEDULER_CLS",
    "TRANSFORMER_CLS",
    "UNCOND_SKIP_LAYERS",
    "MossImageModel",
    "MossImagePipelineModel",
    "MossImageSamplingState",
]
