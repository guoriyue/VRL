"""Wan 2.1 t2v diffusers-backed model.

Diffusion implementation for Wan T2V on the RL path. The generation helper
flow is:

    encode_prompt -> prepare_sampling -> forward_step xN -> decode_latents

The collector owns the scheduler step / SDE step. ``forward_step`` does only
one transformer forward (with optional batched CFG concat) and returns noise
predictions.

Per-family ``WanT2VSamplingState`` is private to this file. The engine /
collector code MUST NOT introspect it beyond the documented attributes
(``latents``, ``timesteps``, ``scheduler``, plus the embeds the eval path
re-builds explicitly).

Differences from SD3:
- ``prompt_embeds`` only (no pooled/CLIP); transformer signature lacks
  ``pooled_projections``.
- 5D latents ``[B, C, T, H, W]`` (Wan VAE temporal axis).
- VAE decode applies Wan-specific per-channel ``latents_mean`` /
  ``latents_std`` denormalization (over ``z_dim``).
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
from vrl.models.diffusion.common.vae_decode_memory import apply_vae_decode_memory
from vrl.models.diffusion.wan_2_1.runner import (
    WanDiffusionBackboneRunner,
    WanI2VDiffusionBackboneRunner,
)


@dataclass
class WanT2VSamplingState:
    """Private Wan T2V sampling state. Engine MUST NOT introspect."""

    latents: torch.Tensor
    timesteps: torch.Tensor
    scheduler: Any
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None
    guidance_scale: float
    do_cfg: bool
    seed: int


@dataclass
class WanI2VSamplingState:
    """Private Wan I2V sampling state. Engine MUST NOT introspect."""

    latents: torch.Tensor
    timesteps: torch.Tensor
    scheduler: Any
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None
    image_embeds: torch.Tensor | None
    condition: torch.Tensor
    guidance_scale: float
    do_cfg: bool
    seed: int


class WanT2VDiffusersModel(DiffusionModelBase):
    """Diffusers-backed Wan 2.1 T2V model (1.3B variant)."""

    family = "wan-diffusers-t2v"

    def __init__(
        self,
        *,
        pipeline: Any,
        device: Any = None,
        memory_metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_pipeline", pipeline)
        self.transformer = pipeline.transformer
        self._device = device
        self.memory_metadata = dict(memory_metadata or {})

    @property
    def pipeline(self) -> Any:
        return self._pipeline

    def _set_transformer(self, transformer: Any) -> None:
        self.transformer = transformer
        self.pipeline.transformer = transformer

    @property
    def device(self) -> Any:
        return self._device if self._device is not None else self.pipeline.device

    def describe(self) -> dict[str, Any]:
        return {"family": self.family, "device": str(self.device)}

    # -- backend ownership (called by runtime, not by collectors) -------

    @classmethod
    def from_spec(cls, spec: Any) -> WanT2VDiffusersModel:
        """Load the diffusers WanPipeline + freeze non-trainable modules."""
        from diffusers import WanPipeline

        pipeline = WanPipeline.from_pretrained(
            spec.model_name_or_path, torch_dtype=spec.dtype,
        )
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        pipeline.vae.to(spec.device, dtype=torch.float32)
        memory_metadata = apply_vae_decode_memory(
            pipeline.vae,
            memory_config=spec.memory,
            owner="Wan VAE",
        )
        pipeline.text_encoder.to(spec.device, dtype=spec.dtype)
        return cls(
            pipeline=pipeline,
            device=spec.device,
            memory_metadata=memory_metadata,
        )

    def apply_lora(self, spec: Any) -> None:
        """Wrap the Wan transformer with PEFT LoRA per spec.lora_*."""
        from peft import LoraConfig, PeftModel, get_peft_model

        self.pipeline.transformer.requires_grad_(False)
        self.pipeline.transformer.to(self.device)

        lora_path = spec.lora_path
        if lora_path:
            transformer = PeftModel.from_pretrained(
                self.pipeline.transformer, lora_path, is_trainable=True,
            )
            transformer.set_adapter("default")
            self._set_transformer(transformer)
        else:
            lora_config = spec.lora
            assert lora_config is not None
            cfg = LoraConfig(
                r=lora_config["rank"],
                lora_alpha=lora_config["alpha"],
                # Empty training adapters must initially preserve base Wan output.
                init_lora_weights=lora_config.get("init_lora_weights", True),
                target_modules=lora_config["target_modules"],
            )
            self._set_transformer(
                get_peft_model(self.pipeline.transformer, cfg),
            )

    def enable_full_finetune(self) -> None:
        self.pipeline.transformer.requires_grad_(True)
        self.pipeline.transformer.to(self.device)

    def torch_compile_transformer(self, mode: str) -> None:
        self._set_transformer(
            torch.compile(self.pipeline.transformer, mode=mode, fullgraph=False),
        )

    def set_num_steps(self, n: int) -> None:
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
        """Encode prompt via Wan's T5 text encoder.

        Returns ``prompt_embeds`` and the matching ``negative_prompt_embeds``
        when CFG is active. Wan does not use a pooled CLIP embedding.
        """
        max_seq = kwargs.get("max_sequence_length", 512) or 512
        guidance_scale = kwargs.get("guidance_scale", 4.5)
        do_cfg = guidance_scale > 1.0
        neg = negative_prompt if negative_prompt is not None else ""

        prompt_embeds, negative_prompt_embeds = self.pipeline.encode_prompt(
            prompt=prompt,
            negative_prompt=neg,
            do_classifier_free_guidance=do_cfg,
            num_videos_per_prompt=1,
            max_sequence_length=max_seq,
            device=self.device,
        )

        td = self.pipeline.transformer.dtype
        prompt_embeds = prompt_embeds.to(td)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(td)

        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
        }

    # -- prepare_sampling ----------------------------------------------

    def prepare_sampling(
        self,
        request: VideoGenerationRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> WanT2VSamplingState:
        """Build the per-request SamplingState for a Wan T2V denoise loop."""
        pipe = self.pipeline
        device = self.device

        prompt_embeds = encoded["prompt_embeds"]
        negative_prompt_embeds = encoded.get("negative_prompt_embeds")

        guidance_scale = request.guidance_scale
        do_cfg = guidance_scale > 1.0

        pipe.scheduler.set_timesteps(request.num_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        num_channels_latents = pipe.transformer.config.in_channels
        batch_size = prompt_embeds.shape[0]

        seed = (
            request.seed if request.seed is not None
            else random.randint(0, sys.maxsize)
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        # Wan prepare_latents signature:
        # (batch, channels, height, width, num_frames, dtype, device, generator, latents)
        latents = pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            request.height,
            request.width,
            request.frame_count,
            torch.float32,
            device,
            generator,
            None,
        )

        return WanT2VSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=pipe.scheduler,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            do_cfg=do_cfg,
            seed=seed,
        )

    # -- forward_step --------------------------------------------------

    def forward_step(
        self,
        state: WanT2VSamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        """Wan T2V transformer forward + optional batched CFG.

        Returns noise_pred plus the un/conditional branches; the caller owns
        scheduler.step / SDE.

        Timestep shape convention (mirrors SD3 model):
        - rollouts: ``state.timesteps`` is 1-D ``[T]``; we expand a scalar to ``[B]``.
        - eval/training: collector packs per-sample timestep as ``[B]``; expand is a no-op.
        """
        t = state.timesteps[step_idx]
        bsz = state.latents.shape[0]
        td = self._transformer_dtype()

        latent_input = state.latents.to(td)
        timestep_batch = expand_batch_timestep(t, bsz).to(device=latent_input.device, dtype=td)
        prompt_embeds = state.prompt_embeds.to(td)
        negative_prompt_embeds = (
            None if state.negative_prompt_embeds is None else state.negative_prompt_embeds.to(td)
        )
        output = DiffusionBackboneCaller(
            self.transformer,
            WanDiffusionBackboneRunner(),
        )(
            DiffusionBackboneInput(
                hidden_states=latent_input,
                timestep=timestep_batch,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=state.guidance_scale,
                do_cfg=state.do_cfg,
                output_dtype=td,
            ),
        )
        return output.as_dict()

    # -- collector boundary --------------------------------------------

    def export_batch_context(self, state: WanT2VSamplingState) -> dict[str, Any]:
        """Project SamplingState -> RolloutBatch.context (shared metadata)."""
        return {
            "guidance_scale": state.guidance_scale,
            "cfg": state.do_cfg,
            "model_family": self.family,
        }

    def export_replay_tensors(self, state: WanT2VSamplingState) -> dict[str, Any]:
        """Project SamplingState into trajectory replay tensors."""
        return {
            "prompt_embeds": state.prompt_embeds,
            "negative_prompt_embeds": state.negative_prompt_embeds,
        }

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> WanT2VSamplingState:
        """Rebuild SamplingState for the eval forward path from a batch slice."""
        ts = replay_tensors["timesteps"]
        # Pack as [1, B] so forward_step's state.timesteps[0] returns [B].
        timesteps = pack_eval_timestep(ts, step_idx)
        return WanT2VSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=None,
            prompt_embeds=replay_tensors["prompt_embeds"],
            negative_prompt_embeds=replay_tensors.get("negative_prompt_embeds"),
            guidance_scale=batch_context["guidance_scale"],
            do_cfg=batch_context["cfg"] and batch_context["guidance_scale"] > 1.0,
            seed=0,
        )

    # -- decode_latents ------------------------------------------------

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode 5D latents -> video [B, C, T, H, W] via Wan VAE.

        Applies Wan-specific per-channel denormalization using the VAE's
        ``latents_mean`` / ``latents_std`` over the ``z_dim`` channel axis.
        """
        pipe = self.pipeline

        def _transform(chunk: torch.Tensor) -> torch.Tensor:
            x = chunk.to(pipe.vae.dtype)
            latents_mean = (
                torch.tensor(pipe.vae.config.latents_mean)
                .view(1, pipe.vae.config.z_dim, 1, 1, 1)
                .to(x.device, x.dtype)
            )
            latents_std = (
                1.0 / torch.tensor(pipe.vae.config.latents_std)
                .view(1, pipe.vae.config.z_dim, 1, 1, 1)
                .to(x.device, x.dtype)
            )
            return x / latents_std + latents_mean

        decoder = ChunkedLatentDecoder(
            LatentDecodeSpec(
                transform=LatentDecodeTransform(_transform),
                vae_decode=lambda chunk: pipe.vae.decode(
                    chunk,
                    return_dict=False,
                )[0],
                postprocess=lambda video: pipe.video_processor.postprocess_video(
                    video,
                    output_type="pt",
                ),
                output_layout="video_btchw",
                decode_batch_size=getattr(pipe, "decode_batch_size", None),
            ),
        )
        return decoder(latents)


class WanT2VReplayModel(WanT2VDiffusersModel):
    """Replay-only Wan model that owns no text encoder, VAE, or pipeline."""

    def __init__(self, *, transformer: Any, scheduler: Any, device: Any = None) -> None:
        DiffusionModelBase.__init__(self)
        self.transformer = transformer
        self._scheduler = scheduler
        self._device = device

    @property
    def pipeline(self) -> Any:
        raise RuntimeError("WanT2VReplayModel does not own a diffusers pipeline")

    def _set_transformer(self, transformer: Any) -> None:
        self.transformer = transformer

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
        raise RuntimeError("WanT2VReplayModel cannot encode prompts")

    def prepare_sampling(
        self,
        request: VideoGenerationRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> WanT2VSamplingState:
        del request, encoded, kwargs
        raise RuntimeError("WanT2VReplayModel cannot run rollout sampling")

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        del latents
        raise RuntimeError("WanT2VReplayModel cannot decode latents")


class WanI2VDiffusersModel(WanT2VDiffusersModel):
    """Diffusers-backed Wan 2.1 I2V model (14B 480P variant)."""

    family = "wan-diffusers-i2v"

    @classmethod
    def from_spec(cls, spec: Any) -> WanI2VDiffusersModel:
        """Load WanImageToVideoPipeline + freeze generation-only modules."""
        from diffusers import WanImageToVideoPipeline

        pipeline = WanImageToVideoPipeline.from_pretrained(
            spec.model_name_or_path,
            torch_dtype=spec.dtype,
        )
        _ensure_single_transformer_wan_i2v(pipeline)
        pipeline.set_progress_bar_config(disable=True)

        for module_name in ("vae", "text_encoder", "image_encoder"):
            module = getattr(pipeline, module_name, None)
            if module is not None:
                module.requires_grad_(False)

        memory_metadata = apply_vae_decode_memory(
            pipeline.vae,
            memory_config=spec.memory,
            owner="Wan I2V VAE",
        )

        # Offload flags ride in model_config (the whole cfg.model block) now,
        # not a curated spec.extra.
        extra = spec.model_config or {}
        # Two offload modes for single-GPU inference (Stage 1):
        #   * enable_sequential_cpu_offload — per-layer streaming,
        #     required when the transformer alone (~28 GB bf16) exceeds
        #     the card. Slow per step but fits on 32 GB GPUs.
        #   * enable_model_cpu_offload — per-module staging; only viable
        #     when transformer + activations + accelerate hooks comfortably
        #     fit, otherwise the .to(execution_device) call in pre_forward
        #     itself OOMs (observed on 32 GB GPUs).
        # Stage 0.5 spike (outputs/wan_i2v_spike/result_v2.json) confirms
        # sequential offload is the only single-card path for Wan I2V 14B
        # at 32 GB; multi-GPU sharding is the production fix.
        gpu_id = getattr(spec.device, "index", None) or 0
        if bool(extra.get("enable_sequential_cpu_offload", False)):
            pipeline.enable_sequential_cpu_offload(gpu_id=gpu_id)
        elif bool(extra.get("enable_model_cpu_offload", False)):
            pipeline.enable_model_cpu_offload(gpu_id=gpu_id)
        else:
            pipeline.vae.to(spec.device, dtype=torch.float32)
            pipeline.text_encoder.to(spec.device, dtype=spec.dtype)
            if getattr(pipeline, "image_encoder", None) is not None:
                pipeline.image_encoder.to(spec.device, dtype=spec.dtype)
        return cls(pipeline=pipeline, device=spec.device, memory_metadata=memory_metadata)

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode Wan I2V text and optional CLIP image conditioning."""

        max_seq = kwargs.get("max_sequence_length", 512) or 512
        guidance_scale = kwargs.get("guidance_scale", 5.0)
        do_cfg = guidance_scale > 1.0
        neg = negative_prompt if negative_prompt is not None else ""

        prompt_embeds, negative_prompt_embeds = self.pipeline.encode_prompt(
            prompt=prompt,
            negative_prompt=neg,
            do_classifier_free_guidance=do_cfg,
            num_videos_per_prompt=1,
            max_sequence_length=max_seq,
            device=self.device,
            dtype=self.transformer.dtype,
        )

        td = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(td)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(td)

        reference_image = kwargs.get("reference_image")
        image_embeds = kwargs.get("image_embeds")
        if image_embeds is None and reference_image is not None:
            image_embeds = self._encode_image_embeds(reference_image)

        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "image_embeds": image_embeds,
            "reference_image": reference_image,
        }

    def prepare_sampling(
        self,
        request: VideoGenerationRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> WanI2VSamplingState:
        """Build the per-request SamplingState for a Wan I2V denoise loop."""
        pipe = self.pipeline
        device = self.device

        prompt_embeds = encoded["prompt_embeds"]
        negative_prompt_embeds = encoded.get("negative_prompt_embeds")
        reference_image = kwargs.get("reference_image", encoded.get("reference_image"))
        if reference_image is None:
            raise ValueError("Wan I2V sampling requires a reference_image")

        guidance_scale = request.guidance_scale
        do_cfg = guidance_scale > 1.0

        pipe.scheduler.set_timesteps(request.num_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        batch_size = prompt_embeds.shape[0]
        seed = (
            request.seed if request.seed is not None
            else random.randint(0, sys.maxsize)
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        image = pipe.video_processor.preprocess(
            reference_image,
            height=request.height,
            width=request.width,
        ).to(device, dtype=torch.float32)

        num_channels_latents = int(pipe.vae.config.z_dim)
        with torch.no_grad():
            latents, condition = pipe.prepare_latents(
                image,
                batch_size,
                num_channels_latents,
                request.height,
                request.width,
                request.frame_count,
                torch.float32,
                device,
                generator,
                None,
                None,
            )

        image_embeds = encoded.get("image_embeds")
        if image_embeds is None:
            image_embeds = self._encode_image_embeds(reference_image)
        image_embeds = _align_optional_batch(image_embeds, batch_size)

        return WanI2VSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=pipe.scheduler,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image_embeds=image_embeds,
            condition=condition,
            guidance_scale=guidance_scale,
            do_cfg=do_cfg,
            seed=seed,
        )

    def forward_step(
        self,
        state: WanI2VSamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        """Wan I2V transformer forward + optional batched CFG."""

        t = state.timesteps[step_idx]
        bsz = state.latents.shape[0]
        td = self._transformer_dtype()

        latent_input = state.latents.to(td)
        timestep_batch = expand_batch_timestep(t, bsz).to(device=latent_input.device, dtype=td)
        prompt_embeds = state.prompt_embeds.to(td)
        negative_prompt_embeds = (
            None if state.negative_prompt_embeds is None else state.negative_prompt_embeds.to(td)
        )
        output = DiffusionBackboneCaller(
            self.transformer,
            WanI2VDiffusionBackboneRunner(),
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
                    "condition": state.condition.to(td),
                    "image_embeds": (
                        None if state.image_embeds is None else state.image_embeds.to(td)
                    ),
                },
            ),
        )
        return output.as_dict()

    def export_batch_context(self, state: WanI2VSamplingState) -> dict[str, Any]:
        """Project SamplingState -> RolloutBatch.context (shared metadata)."""
        return {
            "guidance_scale": state.guidance_scale,
            "cfg": state.do_cfg,
            "model_family": self.family,
            "conditioning": "reference_image",
        }

    def export_replay_tensors(self, state: WanI2VSamplingState) -> dict[str, Any]:
        """Project SamplingState into trajectory replay tensors."""
        return {
            "prompt_embeds": state.prompt_embeds,
            "negative_prompt_embeds": state.negative_prompt_embeds,
            "image_embeds": state.image_embeds,
            "condition": state.condition,
        }

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> WanI2VSamplingState:
        """Rebuild SamplingState for the eval forward path from a batch slice."""
        ts = replay_tensors["timesteps"]
        timesteps = pack_eval_timestep(ts, step_idx)
        return WanI2VSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=None,
            prompt_embeds=replay_tensors["prompt_embeds"],
            negative_prompt_embeds=replay_tensors.get("negative_prompt_embeds"),
            image_embeds=replay_tensors.get("image_embeds"),
            condition=replay_tensors["condition"],
            guidance_scale=batch_context["guidance_scale"],
            do_cfg=batch_context["cfg"] and batch_context["guidance_scale"] > 1.0,
            seed=0,
        )

    def _encode_image_embeds(self, reference_image: Any) -> torch.Tensor | None:
        if not _wan_i2v_uses_image_embeds(self.transformer):
            return None
        with torch.no_grad():
            image_embeds = self.pipeline.encode_image(reference_image, self.device)
        return image_embeds.to(self.transformer.dtype)


class WanI2VReplayModel(WanI2VDiffusersModel):
    """Replay-only Wan I2V model that owns no text, image, VAE, or pipeline modules."""

    def __init__(self, *, transformer: Any, scheduler: Any, device: Any = None) -> None:
        DiffusionModelBase.__init__(self)
        self.transformer = transformer
        self._scheduler = scheduler
        self._device = device

    @property
    def pipeline(self) -> Any:
        raise RuntimeError("WanI2VReplayModel does not own a diffusers pipeline")

    def _set_transformer(self, transformer: Any) -> None:
        self.transformer = transformer

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
        raise RuntimeError("WanI2VReplayModel cannot encode prompts")

    def prepare_sampling(
        self,
        request: VideoGenerationRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> WanI2VSamplingState:
        del request, encoded, kwargs
        raise RuntimeError("WanI2VReplayModel cannot run rollout sampling")

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        del latents
        raise RuntimeError("WanI2VReplayModel cannot decode latents")


def _align_optional_batch(value: torch.Tensor | None, batch_size: int) -> torch.Tensor | None:
    if value is None:
        return None
    if value.shape[:1] == (batch_size,):
        return value
    if value.shape[:1] != (1,):
        raise ValueError(
            "Wan I2V image_embeds batch dimension is incompatible with rollout "
            f"batch_size={batch_size}: {tuple(value.shape)}",
        )
    return value.repeat(batch_size, *([1] * (value.ndim - 1)))


def _wan_i2v_uses_image_embeds(transformer: Any) -> bool:
    config = _transformer_config(transformer)
    if isinstance(config, dict):
        return config.get("image_dim") is not None
    return getattr(config, "image_dim", None) is not None


def _transformer_config(transformer: Any) -> Any:
    config = getattr(transformer, "config", None)
    if isinstance(config, dict) and "image_dim" in config:
        return config
    if config is not None and hasattr(config, "image_dim"):
        return config
    base_model = getattr(transformer, "base_model", None)
    inner = getattr(base_model, "model", base_model)
    config = getattr(inner, "config", None)
    if config is not None:
        return config
    return getattr(transformer, "config", None)


def _ensure_single_transformer_wan_i2v(pipeline: Any) -> None:
    if _config_value(pipeline.config, "boundary_ratio") is not None:
        raise NotImplementedError(
            "Wan I2V RL currently supports single-transformer Wan 2.1 I2V; "
            "Wan 2.2 dual-stage I2V needs a separate replay contract.",
        )
    if bool(_config_value(pipeline.config, "expand_timesteps", False)):
        raise NotImplementedError(
            "Wan I2V RL does not yet support expand_timesteps pipelines.",
        )


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


__all__ = [
    "WanI2VDiffusersModel",
    "WanI2VReplayModel",
    "WanI2VSamplingState",
    "WanT2VDiffusersModel",
    "WanT2VReplayModel",
    "WanT2VSamplingState",
]
