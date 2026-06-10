"""SD 3.5 t2i diffusers-backed model.

Diffusion implementation for Stable Diffusion 3.5-Medium image generation.
The generation helper flow is:

    encode_prompt -> prepare_sampling -> forward_step xN -> decode_latents

The collector owns the scheduler step / SDE step. ``forward_step`` does only
one transformer forward (with optional batched CFG concat) and returns noise
predictions.

Per-family ``SD3SamplingState`` is private to this file — engine /
collector code MUST NOT introspect it beyond the documented attributes
(``latents``, ``timesteps``, ``scheduler``, plus the embeds the eval path
re-builds explicitly).

Timestep shape convention used by ``forward_step``:
- During rollouts ``timesteps`` is a 1-D tensor ``[T]`` of scheduler
  timesteps; ``state.timesteps[step_idx]`` is a scalar that we expand to
  ``[B]`` for the transformer call.
- During eval/training the collector pre-builds a ``SD3SamplingState``
  whose ``timesteps`` is a ``[1, B]`` tensor (per-sample timestep at the
  selected denoise step) and calls ``forward_step(state, 0, ...)``;
  ``state.timesteps[0]`` is then ``[B]`` and ``forward_step``'s
  ``t.expand(bsz)`` is a no-op (because the source already has shape
  ``[B]`` — ``Tensor.expand`` accepts equal sizes).
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from typing import Any

import torch

from vrl.generation.diffusion.layout import VideoGenerationRequest
from vrl.models.diffusion import DiffusionModelBase
from vrl.models.diffusion.common import (
    ChunkedLatentDecoder,
    DiffusionBackboneCaller,
    DiffusionBackboneInput,
    LatentDecodeSpec,
    LatentDecodeTransform,
    expand_batch_timestep,
    pack_eval_timestep,
)
from vrl.models.diffusion.common.lora import LoraModelMixin
from vrl.models.diffusion.sd3_5.runner import (
    SD3DiffusionBackboneRunner,
    install_sd3_joint_attention_processor,
)
from vrl.models.dtypes import resolve_torch_dtype


@dataclass
class SD3SamplingState:
    """Private SD3 sampling state. Engine MUST NOT introspect."""

    latents: torch.Tensor
    timesteps: torch.Tensor
    scheduler: Any
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None
    negative_pooled_prompt_embeds: torch.Tensor | None
    guidance_scale: float
    do_cfg: bool
    seed: int


class SD3_5Model(LoraModelMixin, DiffusionModelBase):
    """Diffusers-backed SD 3.5 t2i model."""

    family = "sd3_5-diffusers-t2i"

    def __init__(self, *, pipeline: Any, device: Any = None) -> None:
        super().__init__()
        object.__setattr__(self, "_pipeline", pipeline)
        self.transformer = pipeline.transformer
        self._device = device
        self.uses_vrl_attention_processor = install_sd3_joint_attention_processor(
            self.transformer,
        )

    @property
    def pipeline(self) -> Any:
        return self._pipeline

    def _set_transformer(self, transformer: Any) -> None:
        self.transformer = transformer
        self.pipeline.transformer = transformer
        self.uses_vrl_attention_processor = install_sd3_joint_attention_processor(
            transformer,
        )

    @property
    def device(self) -> Any:
        return self._device if self._device is not None else self.pipeline.device

    def describe(self) -> dict[str, Any]:
        return {"family": self.family, "device": str(self.device)}

    # -- backend ownership (called by runtime, not by collectors) -------

    @classmethod
    def from_spec(cls, spec: Any) -> SD3_5Model:
        """Load the diffusers SD3.5 pipeline + freeze non-trainable modules."""
        from diffusers import StableDiffusion3Pipeline

        model_dtype = resolve_torch_dtype(spec.dtype)
        # Frozen text encoders / VAE follow the ``frozen`` precision axis. Its
        # default already encodes the Flow-GRPO SD3 contract (fp16 when the
        # denoiser runs fp32, else the model dtype), so ``spec.frozen_dtype`` is
        # authoritative when present; fall back for bare/test specs.
        frozen_dtype = getattr(spec, "frozen_dtype", None)
        if frozen_dtype is None:
            frozen_dtype = torch.float16 if model_dtype == torch.float32 else model_dtype
        load_kwargs: dict[str, Any] = {}
        if model_dtype == torch.float32 and frozen_dtype != torch.float32:
            load_kwargs["torch_dtype"] = {
                "transformer": torch.float32,
                "vae": torch.float32,
                "default": frozen_dtype,
            }
        elif model_dtype != torch.float32:
            load_kwargs["torch_dtype"] = model_dtype
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            spec.model_name_or_path,
            **load_kwargs,
        )
        pipeline.vae.requires_grad_(False)
        for enc in (
            pipeline.text_encoder,
            pipeline.text_encoder_2,
            pipeline.text_encoder_3,
        ):
            if enc is not None:
                enc.requires_grad_(False)
                enc.to(spec.device, dtype=frozen_dtype)
        pipeline.vae.to(spec.device, dtype=torch.float32)
        return cls(pipeline=pipeline, device=spec.device)

    def _lora_dtype(self, spec: Any) -> Any:
        return resolve_torch_dtype(spec.dtype)

    def enable_full_finetune(self) -> None:
        """Mark transformer fully trainable (no-LoRA path)."""
        self.pipeline.transformer.requires_grad_(True)
        self.pipeline.transformer.to(self.device)

    def torch_compile_transformer(self, mode: str) -> None:
        """Apply torch.compile to the transformer in-place."""
        self._set_transformer(
            torch.compile(self.pipeline.transformer, mode=mode, fullgraph=False),
        )

    def set_num_steps(self, n: int) -> None:
        """Initialize the scheduler timesteps for sampling."""
        self.pipeline.scheduler.set_timesteps(n, device=self.device)

    @property
    def trainable_modules(self) -> dict[str, Any]:
        return {"transformer": self.transformer}

    @property
    def scheduler(self) -> Any:
        return self.pipeline.scheduler

    @property
    def backend_handle(self) -> Any:
        return self.pipeline

    # -- encode_prompt -------------------------------------------------

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode prompt via SD3's three text encoders (T5 + 2x CLIP).

        Returns prompt_embeds (joint T5+CLIP sequence), pooled_prompt_embeds
        (CLIP pooled), and their negative counterparts when CFG is active.
        """
        max_seq = kwargs.get("max_sequence_length", 128)
        guidance_scale = kwargs.get("guidance_scale", 4.5)
        do_cfg = guidance_scale > 1.0
        neg = negative_prompt if negative_prompt is not None else ""

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            negative_prompt=neg,
            negative_prompt_2=neg,
            negative_prompt_3=neg,
            do_classifier_free_guidance=do_cfg,
            num_images_per_prompt=1,
            max_sequence_length=max_seq,
            device=self.device,
        )

        td = self.pipeline.transformer.dtype
        prompt_embeds = prompt_embeds.to(td)
        pooled_prompt_embeds = pooled_prompt_embeds.to(td)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(td)
        if negative_pooled_prompt_embeds is not None:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(td)

        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
        }

    # -- prepare_sampling ----------------------------------------------

    def prepare_sampling(
        self,
        request: VideoGenerationRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> SD3SamplingState:
        """Build the per-request SamplingState for a denoise loop."""
        pipe = self.pipeline
        device = self.device

        prompt_embeds = encoded["prompt_embeds"]
        pooled_prompt_embeds = encoded["pooled_prompt_embeds"]
        negative_prompt_embeds = encoded.get("negative_prompt_embeds")
        negative_pooled_prompt_embeds = encoded.get("negative_pooled_prompt_embeds")

        pipe.scheduler.set_timesteps(request.num_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        seed = (
            request.seed if request.seed is not None
            else random.randint(0, sys.maxsize)
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        num_channels_latents = pipe.transformer.config.in_channels
        batch_size = prompt_embeds.shape[0]
        # SD3 prepare_latents: (batch, channels, height, width, dtype, device, generator, latents)
        latents = pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            request.height,
            request.width,
            torch.float32,
            device,
            generator,
            None,
        )

        do_cfg = request.guidance_scale > 1.0

        return SD3SamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=pipe.scheduler,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            guidance_scale=request.guidance_scale,
            do_cfg=do_cfg,
            seed=seed,
        )

    # -- forward_step --------------------------------------------------

    def forward_step(
        self,
        state: SD3SamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        """SD3 transformer forward + optional batched CFG.

        Returns noise_pred plus the un/conditional branches; the caller owns
        scheduler.step / SDE.
        """
        t = state.timesteps[step_idx]
        bsz = state.latents.shape[0]
        td = self._transformer_dtype()

        latent_input = state.latents.to(td)
        # SD3 timestep is broadcast across batch as the raw float (not /1000).
        # If t is already shape [B] (eval path packs timesteps as [1, B]),
        # Tensor.expand(bsz) is a no-op on the equal-sized dim.
        timestep_batch = expand_batch_timestep(t, bsz).to(device=latent_input.device, dtype=td)
        prompt_embeds = state.prompt_embeds.to(td)
        pooled_prompt_embeds = state.pooled_prompt_embeds.to(td)
        negative_prompt_embeds = (
            None if state.negative_prompt_embeds is None else state.negative_prompt_embeds.to(td)
        )
        negative_pooled_prompt_embeds = (
            None
            if state.negative_pooled_prompt_embeds is None
            else state.negative_pooled_prompt_embeds.to(td)
        )
        output = DiffusionBackboneCaller(
            self.transformer,
            SD3DiffusionBackboneRunner(),
        )(
            DiffusionBackboneInput(
                hidden_states=latent_input,
                timestep=timestep_batch,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=state.guidance_scale,
                do_cfg=state.do_cfg,
                output_dtype=td,
                extra={
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
                },
            ),
        )
        return output.as_dict()

    # -- collector boundary --------------------------------------------

    def export_batch_context(self, state: SD3SamplingState) -> dict[str, Any]:
        """Project SD3 sampling state into trajectory context."""
        return {
            "guidance_scale": state.guidance_scale,
            "cfg": state.do_cfg,
            "model_family": self.family,
        }

    def export_replay_tensors(self, state: SD3SamplingState) -> dict[str, Any]:
        """Project SD3 sampling state into trajectory replay tensors."""
        return {
            "prompt_embeds": state.prompt_embeds,
            "pooled_prompt_embeds": state.pooled_prompt_embeds,
            "negative_prompt_embeds": state.negative_prompt_embeds,
            "negative_pooled_prompt_embeds": state.negative_pooled_prompt_embeds,
        }

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> SD3SamplingState:
        """Rebuild SD3SamplingState from a batch slice for the eval forward path.

        Packs timesteps as ``[1, B]`` so ``state.timesteps[0]`` is ``[B]`` —
        matches the eval-path convention documented in the class docstring.
        """
        ts = replay_tensors["timesteps"]
        timesteps = pack_eval_timestep(ts, step_idx)
        return SD3SamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=None,  # not needed for forward_step (no scheduler.step here)
            prompt_embeds=replay_tensors["prompt_embeds"],
            pooled_prompt_embeds=replay_tensors["pooled_prompt_embeds"],
            negative_prompt_embeds=replay_tensors.get("negative_prompt_embeds"),
            negative_pooled_prompt_embeds=replay_tensors.get(
                "negative_pooled_prompt_embeds",
            ),
            guidance_scale=batch_context["guidance_scale"],
            do_cfg=batch_context["cfg"] and batch_context["guidance_scale"] > 1.0,
            seed=0,
        )

    # -- decode_latents ------------------------------------------------

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents → image via SD3 VAE (4D, no T dim)."""
        pipe = self.pipeline
        scaling_factor = pipe.vae.config.scaling_factor
        shift_factor = getattr(pipe.vae.config, "shift_factor", 0.0) or 0.0
        decoder = ChunkedLatentDecoder(
            LatentDecodeSpec(
                transform=LatentDecodeTransform(
                    lambda chunk: chunk.to(pipe.vae.dtype) / scaling_factor
                    + shift_factor,
                ),
                vae_decode=lambda chunk: pipe.vae.decode(
                    chunk,
                    return_dict=False,
                )[0],
                postprocess=lambda image: pipe.image_processor.postprocess(
                    image,
                    output_type="pt",
                ),
                output_layout="image_bchw",
                decode_batch_size=getattr(pipe, "decode_batch_size", None),
            ),
        )
        return decoder(latents)


class SD3_5ReplayModel(SD3_5Model):
    """Replay-only SD3.5 model that owns no prompt encoders, VAE, or pipeline."""

    def __init__(self, *, transformer: Any, scheduler: Any, device: Any = None) -> None:
        DiffusionModelBase.__init__(self)
        self.transformer = transformer
        self._scheduler = scheduler
        self._device = device
        self.uses_vrl_attention_processor = install_sd3_joint_attention_processor(
            transformer,
        )

    @property
    def pipeline(self) -> Any:
        raise RuntimeError("SD3_5ReplayModel does not own a diffusers pipeline")

    def _set_transformer(self, transformer: Any) -> None:
        self.transformer = transformer
        self.uses_vrl_attention_processor = install_sd3_joint_attention_processor(
            transformer,
        )

    @property
    def scheduler(self) -> Any:
        return self._scheduler

    @property
    def backend_handle(self) -> Any:
        return None

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del prompt, negative_prompt, kwargs
        raise RuntimeError("SD3_5ReplayModel cannot encode prompts")

    def prepare_sampling(
        self,
        request: VideoGenerationRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> SD3SamplingState:
        del request, encoded, kwargs
        raise RuntimeError("SD3_5ReplayModel cannot run rollout sampling")

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        del latents
        raise RuntimeError("SD3_5ReplayModel cannot decode latents")


__all__ = ["SD3SamplingState", "SD3_5Model", "SD3_5ReplayModel"]
