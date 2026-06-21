"""FLUX.1 t2i diffusers-backed model.

Diffusion implementation for Black Forest Labs FLUX.1 image generation. The
generation helper flow mirrors every diffusion family:

    encode_prompt -> prepare_sampling -> forward_step xN -> decode_latents

The collector owns the scheduler step / SDE step. ``forward_step`` does one
guidance-distilled transformer forward and returns the noise prediction.

FLUX specifics vs SD3 (the reference family):
- Latents are PACKED: prepare_latents returns ``[B, (h/2)*(w/2), C*4]`` 2x2
  patch-packed tokens (not ``[B, C, H, W]``). They stay packed through the whole
  denoise loop; only ``decode_latents`` unpacks before the VAE.
- The transformer needs rotary position ids: ``img_ids`` (per latent patch) and
  ``txt_ids`` (per text token, all zeros for FLUX). Both are batch-independent
  and deterministic from the spatial shape, so the replay path rebuilds them from
  ``height``/``width`` instead of storing per-sample copies.
- The transformer time embedding multiplies its input by 1000 internally, so
  ``forward_step`` feeds ``t / 1000`` (matching diffusers' FluxPipeline loop).
- FLUX.1-dev is guidance-distilled (``config.guidance_embeds``): a ``guidance``
  scalar is embedded, there is no classifier-free unconditional branch. So
  ``do_cfg`` is always False and the single-branch runner runs one forward.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from typing import Any

import torch

from vrl.generation.diffusion.layout import VideoGenerationRequest
from vrl.models.diffusion import DiffusionModelBase, ReplayRolloutStubs
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
from vrl.models.diffusion.flux.runner import FluxDiffusionBackboneRunner
from vrl.models.dtypes import resolve_torch_dtype


@dataclass
class FluxSamplingState:
    """Private FLUX sampling state. Engine MUST NOT introspect."""

    latents: torch.Tensor
    timesteps: torch.Tensor
    scheduler: Any
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    text_ids: torch.Tensor
    latent_image_ids: torch.Tensor
    guidance_scale: float
    guidance_embeds: bool
    height: int
    width: int


class FluxModel(LoraModelMixin, DiffusionModelBase):
    """Diffusers-backed FLUX.1 t2i model."""

    def __init__(
        self,
        *,
        pipeline: Any,
        device: Any = None,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_pipeline", pipeline)
        self.transformer = pipeline.transformer
        self._device = device
        # decode_latents only receives the packed latent tensor; the executor runs
        # prepare -> denoise -> decode sequentially on this one model instance per
        # chunk (no concurrency), so prepare_sampling records the spatial shape it
        # must unpack to here. Defaults are overwritten on the first prepare call.
        self._decode_height = 1024
        self._decode_width = 1024

    @property
    def pipeline(self) -> Any:
        return self._pipeline

    def _set_transformer(self, transformer: Any) -> None:
        self.transformer = transformer
        self.pipeline.transformer = transformer

    @property
    def device(self) -> Any:
        return self._device if self._device is not None else self.pipeline.device

    # -- backend ownership (called by runtime, not by collectors) -------

    @classmethod
    def from_spec(cls, spec: Any) -> FluxModel:
        """Load the diffusers FLUX pipeline + freeze non-trainable modules."""
        from diffusers import FluxPipeline

        model_dtype = resolve_torch_dtype(spec.dtype)
        # Frozen text encoders / VAE follow the ``frozen`` precision axis, same
        # contract as SD3: fp16 when the denoiser runs fp32, else the model dtype.
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
        pipeline = FluxPipeline.from_pretrained(
            spec.model_name_or_path,
            **load_kwargs,
        )
        pipeline.vae.requires_grad_(False)
        # FLUX.1-dev's T5-XXL encoder is ~9.4 GB and the transformer is ~24 GB
        # (bf16); together they exceed a single 32 GB card. The encoders feed only
        # the one-shot encode_prompt, so park them on CPU (the enable_model_cpu_offload
        # discipline) and leave the whole card for the denoiser. encode_prompt runs
        # them on their own device and moves the embeds to the GPU.
        for enc in (
            getattr(pipeline, "text_encoder", None),
            getattr(pipeline, "text_encoder_2", None),
        ):
            if enc is not None:
                enc.requires_grad_(False)
                enc.to("cpu", dtype=frozen_dtype)
        pipeline.vae.to(spec.device, dtype=torch.float32)
        return cls(
            pipeline=pipeline,
            device=spec.device,
        )

    def _lora_dtype(self, spec: Any) -> Any:
        return resolve_torch_dtype(spec.dtype)

    def enable_full_finetune(self) -> None:
        """Mark transformer fully trainable (no-LoRA path)."""
        self.pipeline.transformer.requires_grad_(True)
        self.pipeline.transformer.to(self.device)

    def set_num_steps(self, n: int) -> None:
        """Initialize the scheduler timesteps for sampling.

        FLUX's FlowMatch scheduler uses ``use_dynamic_shifting``: the timestep
        schedule depends on a resolution-derived ``mu`` (image sequence length),
        which is unknown at build time. Defer the real set to ``prepare_sampling``
        for such schedulers; only eager-set the static ones.
        """
        scheduler = self.pipeline.scheduler
        if getattr(scheduler.config, "use_dynamic_shifting", False):
            return
        scheduler.set_timesteps(n, device=self.device)

    def _set_dynamic_timesteps(self, num_steps: int, image_seq_len: int, device: Any) -> Any:
        """Set FLUX timesteps with the resolution-derived ``mu`` (diffusers parity)."""
        from diffusers.pipelines.flux.pipeline_flux import calculate_shift

        scheduler = self.pipeline.scheduler
        config = scheduler.config
        if getattr(config, "use_dynamic_shifting", False):
            mu = calculate_shift(
                image_seq_len,
                getattr(config, "base_image_seq_len", 256),
                getattr(config, "max_image_seq_len", 4096),
                getattr(config, "base_shift", 0.5),
                getattr(config, "max_shift", 1.15),
            )
            scheduler.set_timesteps(num_steps, device=device, mu=mu)
        else:
            scheduler.set_timesteps(num_steps, device=device)
        return scheduler.timesteps

    @property
    def trainable_modules(self) -> dict[str, Any]:
        return {"transformer": self.transformer}

    @property
    def scheduler(self) -> Any:
        return self.pipeline.scheduler

    @property
    def raw_handle(self) -> Any:
        return self.pipeline

    @property
    def _guidance_embeds(self) -> bool:
        """Whether the FLUX checkpoint embeds a guidance scalar (dev) or not (schnell)."""
        return bool(getattr(self.transformer.config, "guidance_embeds", False))

    @staticmethod
    def _build_latent_image_ids(
        height: int,
        width: int,
        device: Any,
        dtype: Any,
    ) -> torch.Tensor:
        """Rebuild FLUX latent position ids (diffusers ``_prepare_latent_image_ids``).

        Inlined as a pure function so the pipeline-less replay model can rebuild
        the batch-shared position grid without owning a FluxPipeline.
        """
        latent_image_ids = torch.zeros(height, width, 3, device=device, dtype=dtype)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(
            height, device=device, dtype=dtype,
        )[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(
            width, device=device, dtype=dtype,
        )[None, :]
        return latent_image_ids.reshape(height * width, 3)

    # -- encode_prompt -------------------------------------------------

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode prompt via FLUX's CLIP (pooled) + T5 (sequence) encoders.

        FLUX.1-dev is guidance-distilled, so there is no unconditional branch and
        ``negative_prompt`` is ignored. Returns the T5 sequence embeds, the CLIP
        pooled vector, and the batch-shared ``text_ids`` position grid (float32 —
        rotary positions are computed in float).
        """
        del negative_prompt
        max_seq = kwargs.get("max_sequence_length", 512)
        # The frozen encoders live on CPU (see from_spec); run encode there, then
        # move the embeds onto the model/transformer device for the denoise forward.
        enc_device = self._encoder_device()
        prompt_embeds, pooled_prompt_embeds, text_ids = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            num_images_per_prompt=1,
            max_sequence_length=max_seq,
            device=enc_device,
        )
        td = self.transformer.dtype
        return {
            "prompt_embeds": prompt_embeds.to(self.device, dtype=td),
            "pooled_prompt_embeds": pooled_prompt_embeds.to(self.device, dtype=td),
            "text_ids": text_ids.to(self.device, dtype=torch.float32),
        }

    def _encoder_device(self) -> Any:
        """Device the frozen prompt encoders live on (CPU when offloaded)."""
        for name in ("text_encoder_2", "text_encoder"):
            enc = getattr(self.pipeline, name, None)
            if enc is not None:
                try:
                    return next(enc.parameters()).device
                except StopIteration:
                    continue
        return self.device

    # -- prepare_sampling ----------------------------------------------

    def prepare_sampling(
        self,
        request: VideoGenerationRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> FluxSamplingState:
        """Build the per-request packed-latent SamplingState for a denoise loop."""
        del kwargs
        pipe = self.pipeline
        device = self.device

        prompt_embeds = encoded["prompt_embeds"]
        pooled_prompt_embeds = encoded["pooled_prompt_embeds"]
        text_ids = encoded["text_ids"]

        seed = (
            request.seed if request.seed is not None
            else random.randint(0, sys.maxsize)
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        batch_size = prompt_embeds.shape[0]
        # FLUX packs 2x2 patches into channels, so the transformer's in_channels
        # is 4x the unpacked latent channel count.
        num_channels_latents = pipe.transformer.config.in_channels // 4
        latents, latent_image_ids = pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            request.height,
            request.width,
            torch.float32,
            device,
            generator,
            None,
        )

        # Timesteps depend on the packed image sequence length (dynamic shifting),
        # so set them only now that the latents (hence seq len) exist.
        timesteps = self._set_dynamic_timesteps(
            request.num_steps, latents.shape[1], device,
        )

        # Record the spatial shape decode_latents must unpack to (safe: same model
        # instance runs prepare -> denoise -> decode sequentially per chunk).
        self._decode_height = int(request.height)
        self._decode_width = int(request.width)

        return FluxSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=pipe.scheduler,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids,
            latent_image_ids=latent_image_ids,
            guidance_scale=request.guidance_scale,
            guidance_embeds=self._guidance_embeds,
            height=int(request.height),
            width=int(request.width),
        )

    # -- forward_step --------------------------------------------------

    def forward_step(
        self,
        state: FluxSamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        """FLUX transformer forward (single guidance-distilled branch).

        Returns noise_pred (packed) plus the cond/uncond placeholder branches; the
        caller owns scheduler.step / SDE.
        """
        t = state.timesteps[step_idx]
        bsz = state.latents.shape[0]
        td = self._transformer_dtype()

        latent_input = state.latents.to(td)
        # The transformer multiplies its timestep input by 1000, so feed t / 1000
        # (matches diffusers FluxPipeline). If t is already [B] (eval path packs
        # [1, B]) expand is a no-op.
        timestep_batch = (
            expand_batch_timestep(t, bsz).to(device=latent_input.device, dtype=td) / 1000.0
        )
        guidance = None
        if state.guidance_embeds:
            guidance = torch.full(
                (bsz,),
                float(state.guidance_scale),
                device=latent_input.device,
                dtype=td,
            )
        output = DiffusionBackboneCaller(
            self.transformer,
            FluxDiffusionBackboneRunner(),
        )(
            DiffusionBackboneInput(
                hidden_states=latent_input,
                timestep=timestep_batch,
                prompt_embeds=state.prompt_embeds.to(td),
                negative_prompt_embeds=None,
                guidance_scale=state.guidance_scale,
                do_cfg=False,
                output_dtype=td,
                extra={
                    "pooled_projections": state.pooled_prompt_embeds.to(td),
                    # Rotary position ids stay float32 (computed in float).
                    "img_ids": state.latent_image_ids.to(
                        device=latent_input.device, dtype=torch.float32,
                    ),
                    "txt_ids": state.text_ids.to(
                        device=latent_input.device, dtype=torch.float32,
                    ),
                    "guidance": guidance,
                },
            ),
        )
        return output.as_dict()

    # -- collector boundary --------------------------------------------

    def export_batch_context(self, state: FluxSamplingState) -> dict[str, Any]:
        """Project FLUX sampling state into trajectory context.

        ``text_ids`` / ``latent_image_ids`` are batch-shared and deterministic
        from the spatial shape, so only scalars travel here; the replay path
        rebuilds the position grids from height/width + ``vae_scale_factor``.
        """
        return {
            "guidance_scale": state.guidance_scale,
            "guidance_embeds": state.guidance_embeds,
            "height": state.height,
            "width": state.width,
            "vae_scale_factor": int(self.pipeline.vae_scale_factor),
        }

    def export_replay_tensors(self, state: FluxSamplingState) -> dict[str, Any]:
        """Project FLUX sampling state into per-sample trajectory replay tensors."""
        return {
            "prompt_embeds": state.prompt_embeds,
            "pooled_prompt_embeds": state.pooled_prompt_embeds,
        }

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> FluxSamplingState:
        """Rebuild FluxSamplingState from a batch slice for the eval forward path.

        Rebuilds the batch-shared position grids deterministically: ``text_ids``
        is zeros sized to the stored prompt sequence length, ``latent_image_ids``
        from the stored spatial shape — pipeline-free, so the replay model never
        touches a FluxPipeline.
        """
        ts = replay_tensors["timesteps"]
        timesteps = pack_eval_timestep(ts, step_idx)
        prompt_embeds = replay_tensors["prompt_embeds"]
        height = batch_context["height"]
        width = batch_context["width"]
        vae_scale_factor = int(batch_context["vae_scale_factor"])
        device = prompt_embeds.device

        text_seq_len = prompt_embeds.shape[1]
        text_ids = torch.zeros(text_seq_len, 3, device=device, dtype=torch.float32)
        latent_h = 2 * (int(height) // (vae_scale_factor * 2))
        latent_w = 2 * (int(width) // (vae_scale_factor * 2))
        latent_image_ids = self._build_latent_image_ids(
            latent_h // 2,
            latent_w // 2,
            device,
            torch.float32,
        )
        return FluxSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=None,  # not needed for forward_step (no scheduler.step here)
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=replay_tensors["pooled_prompt_embeds"],
            text_ids=text_ids,
            latent_image_ids=latent_image_ids,
            guidance_scale=batch_context["guidance_scale"],
            guidance_embeds=batch_context["guidance_embeds"],
            height=height,
            width=width,
        )

    # -- decode_latents ------------------------------------------------

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode packed latents → image via FLUX VAE (unpack, then 4D decode)."""
        pipe = self.pipeline
        vae_scale_factor = pipe.vae_scale_factor
        scaling_factor = pipe.vae.config.scaling_factor
        shift_factor = getattr(pipe.vae.config, "shift_factor", 0.0) or 0.0
        height = self._decode_height
        width = self._decode_width

        def _transform(chunk: torch.Tensor) -> torch.Tensor:
            unpacked = pipe._unpack_latents(chunk, height, width, vae_scale_factor)
            return unpacked.to(pipe.vae.dtype) / scaling_factor + shift_factor

        decoder = ChunkedLatentDecoder(
            LatentDecodeSpec(
                transform=LatentDecodeTransform(_transform),
                vae_decode=lambda chunk: pipe.vae.decode(chunk, return_dict=False)[0],
                postprocess=lambda image: pipe.image_processor.postprocess(
                    image,
                    output_type="pt",
                ),
                output_layout="image_bchw",
                decode_batch_size=getattr(pipe, "decode_batch_size", None),
            ),
        )
        return decoder(latents)


class FluxReplayModel(ReplayRolloutStubs, FluxModel):
    """Replay-only FLUX model that owns no prompt encoders, VAE, or pipeline."""

    def __init__(self, *, transformer: Any, scheduler: Any, device: Any = None) -> None:
        DiffusionModelBase.__init__(self)
        self.transformer = transformer
        self._scheduler = scheduler
        self._device = device

    @property
    def pipeline(self) -> Any:
        raise RuntimeError("FluxReplayModel does not own a diffusers pipeline")

    def _set_transformer(self, transformer: Any) -> None:
        self.transformer = transformer

    @property
    def scheduler(self) -> Any:
        return self._scheduler

    @property
    def raw_handle(self) -> Any:
        return None


__all__ = ["FluxModel", "FluxReplayModel", "FluxSamplingState"]
