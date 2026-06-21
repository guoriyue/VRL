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

import contextlib
import random
import sys
from collections.abc import Iterator, Mapping
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
from vrl.models.diffusion.wan_2_1.runner import (
    WanDiffusionBackboneRunner,
    WanI2VDiffusionBackboneRunner,
)
from vrl.models.utils import disable_adapter_on, load_weights_into


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
    guidance_scale_2: float | None = None
    boundary_ratio: float | None = None
    num_train_timesteps: int | None = None


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
    guidance_scale_2: float | None = None
    boundary_ratio: float | None = None
    num_train_timesteps: int | None = None


class WanT2VDiffusersModel(LoraModelMixin, DiffusionModelBase):
    """Diffusers-backed Wan 2.1 T2V model (1.3B variant)."""

    def __init__(
        self,
        *,
        pipeline: Any,
        device: Any = None,
        trainable_transformers: Any = None,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_pipeline", pipeline)
        self.transformer = pipeline.transformer
        self.transformer_2 = getattr(pipeline, "transformer_2", None)
        self._device = device
        self._boundary_ratio = _optional_float(
            _config_value(getattr(pipeline, "config", None), "boundary_ratio"),
            "boundary_ratio",
        )
        self._trainable_transformer_names = _normalize_trainable_transformers(
            trainable_transformers,
            dual_stage=self._boundary_ratio is not None,
        )
        for module in self._wan_transformers().values():
            module.requires_grad_(False)

    @property
    def pipeline(self) -> Any:
        return self._pipeline

    def _set_transformer(self, transformer: Any) -> None:
        self.transformer = transformer
        self.pipeline.transformer = transformer

    def _set_transformer_2(self, transformer: Any) -> None:
        self.transformer_2 = transformer
        self.pipeline.transformer_2 = transformer

    @property
    def device(self) -> Any:
        return self._device if self._device is not None else self.pipeline.device

    @property
    def boundary_ratio(self) -> float | None:
        return self._boundary_ratio

    # -- backend ownership (called by runtime, not by collectors) -------

    @classmethod
    def from_spec(cls, spec: Any) -> WanT2VDiffusersModel:
        """Load the diffusers WanPipeline + freeze non-trainable modules."""
        from diffusers import WanPipeline

        pipeline = WanPipeline.from_pretrained(
            spec.model_name_or_path,
            torch_dtype=spec.dtype,
        )
        _validate_wan_pipeline(pipeline, task="Wan T2V")
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        _apply_offload_mode(
            pipeline,
            spec,
            eager_module_dtypes={
                "vae": torch.float32,
                "text_encoder": spec.dtype,
            },
        )
        return cls(
            pipeline=pipeline,
            device=spec.device,
            trainable_transformers=(spec.model_config or {}).get("trainable_transformers"),
        )

    # Empty training adapters must initially preserve base Wan output.
    _lora_default_init_weights = True

    def _wan_transformers(self) -> dict[str, Any]:
        modules = {"transformer": self.transformer}
        if self.transformer_2 is not None:
            modules["transformer_2"] = self.transformer_2
        return modules

    def enable_full_finetune(self) -> None:
        for module in self.trainable_modules.values():
            module.requires_grad_(True)
            module.to(self.device)

    def apply_lora(self, spec: Any) -> None:
        """Attach LoRA to the configured Wan trainable transformer(s)."""

        from peft import LoraConfig, PeftModel, get_peft_model

        lora_path = spec.lora_path
        names = self._trainable_transformer_names
        if lora_path and len(names) != 1:
            raise ValueError(
                "model.lora.path can only resume one Wan transformer; set "
                "model.trainable_transformers to a single transformer name",
            )

        lora_config = spec.lora
        if not lora_path and lora_config is None:
            raise ValueError("LoRA runtime spec requires lora_config when lora_path is empty")

        for name in names:
            transformer = self._wan_transformers()[name]
            transformer.requires_grad_(False)
            transformer.to(self.device)
            if lora_path:
                wrapped = PeftModel.from_pretrained(
                    transformer,
                    lora_path,
                    is_trainable=True,
                )
                wrapped.set_adapter("default")
            else:
                assert lora_config is not None
                cfg = LoraConfig(
                    r=lora_config["rank"],
                    lora_alpha=lora_config["alpha"],
                    init_lora_weights=lora_config.get(
                        "init_lora_weights",
                        self._lora_default_init_weights,
                    ),
                    target_modules=lora_config["target_modules"],
                )
                wrapped = get_peft_model(transformer, cfg)
            self._set_wan_transformer(name, wrapped)

    def torch_compile_transformer(self, mode: str) -> None:
        for name, module in self.trainable_modules.items():
            self._set_wan_transformer(
                name,
                torch.compile(module, mode=mode, fullgraph=False),
            )

    def set_num_steps(self, n: int) -> None:
        self.pipeline.scheduler.set_timesteps(n, device=self.device)

    @property
    def trainable_modules(self) -> dict[str, Any]:
        modules = self._wan_transformers()
        return {name: modules[name] for name in self._trainable_transformer_names}

    @contextlib.contextmanager
    def disable_adapter(self) -> Iterator[None]:
        with contextlib.ExitStack() as stack:
            for module in self.trainable_modules.values():
                stack.enter_context(disable_adapter_on(module))
            yield

    def load_trainable_state(self, state_dict: Mapping[str, Any]) -> Any:
        state = dict(state_dict)
        modules = self.trainable_modules
        accepted_prefixes = tuple(f"{name}." for name in modules)
        unexpected = sorted(key for key in state if not key.startswith(accepted_prefixes))
        if unexpected:
            raise ValueError(
                f"{type(self).__name__}: unexpected trainable state keys {unexpected[:5]}",
            )
        results = {}
        for name, module in modules.items():
            prefix = f"{name}."
            module_state = {key: value for key, value in state.items() if key.startswith(prefix)}
            results[name] = load_weights_into(
                module,
                module_state,
                prefix=name,
                label=type(module).__name__,
            )
        return results

    @property
    def scheduler(self) -> Any:
        return self.pipeline.scheduler

    @property
    def raw_handle(self) -> Any:
        return self.pipeline

    def _set_wan_transformer(self, name: str, transformer: Any) -> None:
        if name == "transformer":
            self._set_transformer(transformer)
            return
        if name == "transformer_2":
            self._set_transformer_2(transformer)
            return
        raise ValueError(f"unknown Wan transformer name: {name!r}")

    def _transformer_for_timestep(
        self,
        state: WanT2VSamplingState | WanI2VSamplingState,
        timestep: torch.Tensor,
    ) -> tuple[Any, float]:
        if state.boundary_ratio is None:
            return self.transformer, state.guidance_scale
        if _uses_low_noise_transformer(
            timestep,
            boundary_ratio=state.boundary_ratio,
            num_train_timesteps=state.num_train_timesteps,
        ):
            transformer_2 = self.transformer_2
            if transformer_2 is None:
                raise RuntimeError("Wan dual-stage sampling requires transformer_2")
            return transformer_2, state.guidance_scale_2 or state.guidance_scale
        return self.transformer, state.guidance_scale

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
        guidance_scale_2 = _resolve_guidance_scale_2(request, guidance_scale, self._boundary_ratio)
        do_cfg = _uses_cfg(guidance_scale, guidance_scale_2)

        pipe.scheduler.set_timesteps(request.num_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        num_channels_latents = pipe.transformer.config.in_channels
        batch_size = prompt_embeds.shape[0]

        seed = request.seed if request.seed is not None else random.randint(0, sys.maxsize)
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
            guidance_scale_2=guidance_scale_2,
            boundary_ratio=self._boundary_ratio,
            num_train_timesteps=_scheduler_num_train_timesteps(pipe.scheduler),
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
        transformer, guidance_scale = self._transformer_for_timestep(state, t)
        td = _module_dtype(transformer)

        latent_input = state.latents.to(td)
        timestep_batch = expand_batch_timestep(t, bsz).to(device=latent_input.device, dtype=td)
        prompt_embeds = state.prompt_embeds.to(td)
        negative_prompt_embeds = (
            None if state.negative_prompt_embeds is None else state.negative_prompt_embeds.to(td)
        )
        output = DiffusionBackboneCaller(
            transformer,
            WanDiffusionBackboneRunner(),
        )(
            DiffusionBackboneInput(
                hidden_states=latent_input,
                timestep=timestep_batch,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
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
            "guidance_scale_2": state.guidance_scale_2,
            "cfg": state.do_cfg,
            "boundary_ratio": state.boundary_ratio,
            "num_train_timesteps": state.num_train_timesteps,
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
            do_cfg=batch_context["cfg"]
            and _uses_cfg(
                batch_context["guidance_scale"],
                batch_context.get("guidance_scale_2"),
            ),
            guidance_scale_2=batch_context.get("guidance_scale_2"),
            boundary_ratio=batch_context.get("boundary_ratio"),
            num_train_timesteps=batch_context.get("num_train_timesteps"),
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
            latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
                1, pipe.vae.config.z_dim, 1, 1, 1
            ).to(x.device, x.dtype)
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


class WanT2VReplayModel(ReplayRolloutStubs, WanT2VDiffusersModel):
    """Replay-only Wan model that owns no text encoder, VAE, or pipeline."""

    def __init__(
        self,
        *,
        transformer: Any,
        scheduler: Any,
        device: Any = None,
        transformer_2: Any = None,
        boundary_ratio: float | None = None,
        trainable_transformers: Any = None,
    ) -> None:
        DiffusionModelBase.__init__(self)
        self.transformer = transformer
        self.transformer_2 = transformer_2
        self._scheduler = scheduler
        self._device = device
        self._boundary_ratio = boundary_ratio
        self._trainable_transformer_names = _normalize_trainable_transformers(
            trainable_transformers,
            dual_stage=boundary_ratio is not None,
        )
        for module in self._wan_transformers().values():
            module.requires_grad_(False)

    @property
    def pipeline(self) -> Any:
        raise RuntimeError("WanT2VReplayModel does not own a diffusers pipeline")

    def _set_transformer(self, transformer: Any) -> None:
        self.transformer = transformer

    def _set_transformer_2(self, transformer: Any) -> None:
        self.transformer_2 = transformer

    @property
    def scheduler(self) -> Any:
        return self._scheduler

    @property
    def raw_handle(self) -> Any:
        return None


class WanI2VDiffusersModel(WanT2VDiffusersModel):
    """Diffusers-backed Wan 2.1 I2V model (14B 480P variant)."""

    @classmethod
    def from_spec(cls, spec: Any) -> WanI2VDiffusersModel:
        """Load WanImageToVideoPipeline + freeze generation-only modules."""
        from diffusers import WanImageToVideoPipeline

        pipeline = WanImageToVideoPipeline.from_pretrained(
            spec.model_name_or_path,
            torch_dtype=spec.dtype,
        )
        _validate_wan_pipeline(pipeline, task="Wan I2V")
        pipeline.set_progress_bar_config(disable=True)

        for module_name in ("vae", "text_encoder", "image_encoder"):
            module = getattr(pipeline, module_name, None)
            if module is not None:
                module.requires_grad_(False)

        _apply_offload_mode(
            pipeline,
            spec,
            eager_module_dtypes={
                "vae": torch.float32,
                "text_encoder": spec.dtype,
                "image_encoder": spec.dtype,
            },
        )
        return cls(
            pipeline=pipeline,
            device=spec.device,
            trainable_transformers=(spec.model_config or {}).get("trainable_transformers"),
        )

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
        guidance_scale_2 = _resolve_guidance_scale_2(request, guidance_scale, self._boundary_ratio)
        do_cfg = _uses_cfg(guidance_scale, guidance_scale_2)

        pipe.scheduler.set_timesteps(request.num_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        batch_size = prompt_embeds.shape[0]
        seed = request.seed if request.seed is not None else random.randint(0, sys.maxsize)
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
            guidance_scale_2=guidance_scale_2,
            boundary_ratio=self._boundary_ratio,
            num_train_timesteps=_scheduler_num_train_timesteps(pipe.scheduler),
        )

    def forward_step(
        self,
        state: WanI2VSamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        """Wan I2V transformer forward + optional batched CFG."""

        t = state.timesteps[step_idx]
        bsz = state.latents.shape[0]
        transformer, guidance_scale = self._transformer_for_timestep(state, t)
        td = _module_dtype(transformer)

        latent_input = state.latents.to(td)
        timestep_batch = expand_batch_timestep(t, bsz).to(device=latent_input.device, dtype=td)
        prompt_embeds = state.prompt_embeds.to(td)
        negative_prompt_embeds = (
            None if state.negative_prompt_embeds is None else state.negative_prompt_embeds.to(td)
        )
        output = DiffusionBackboneCaller(
            transformer,
            WanI2VDiffusionBackboneRunner(),
        )(
            DiffusionBackboneInput(
                hidden_states=latent_input,
                timestep=timestep_batch,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
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
            "guidance_scale_2": state.guidance_scale_2,
            "cfg": state.do_cfg,
            "conditioning": "reference_image",
            "boundary_ratio": state.boundary_ratio,
            "num_train_timesteps": state.num_train_timesteps,
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
            do_cfg=batch_context["cfg"]
            and _uses_cfg(
                batch_context["guidance_scale"],
                batch_context.get("guidance_scale_2"),
            ),
            guidance_scale_2=batch_context.get("guidance_scale_2"),
            boundary_ratio=batch_context.get("boundary_ratio"),
            num_train_timesteps=batch_context.get("num_train_timesteps"),
        )

    def _encode_image_embeds(self, reference_image: Any) -> torch.Tensor | None:
        if not _wan_i2v_uses_image_embeds(self.transformer):
            return None
        with torch.no_grad():
            image_embeds = self.pipeline.encode_image(reference_image, self.device)
        return image_embeds.to(self.transformer.dtype)


class WanI2VReplayModel(ReplayRolloutStubs, WanI2VDiffusersModel):
    """Replay-only Wan I2V model that owns no text, image, VAE, or pipeline modules."""

    def __init__(
        self,
        *,
        transformer: Any,
        scheduler: Any,
        device: Any = None,
        transformer_2: Any = None,
        boundary_ratio: float | None = None,
        trainable_transformers: Any = None,
    ) -> None:
        WanT2VReplayModel.__init__(
            self,
            transformer=transformer,
            scheduler=scheduler,
            device=device,
            transformer_2=transformer_2,
            boundary_ratio=boundary_ratio,
            trainable_transformers=trainable_transformers,
        )

    @property
    def pipeline(self) -> Any:
        raise RuntimeError("WanI2VReplayModel does not own a diffusers pipeline")

    def _set_transformer(self, transformer: Any) -> None:
        self.transformer = transformer

    def _set_transformer_2(self, transformer: Any) -> None:
        self.transformer_2 = transformer

    @property
    def scheduler(self) -> Any:
        return self._scheduler

    @property
    def raw_handle(self) -> Any:
        return None


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


def _validate_wan_pipeline(pipeline: Any, *, task: str) -> None:
    if bool(_config_value(pipeline.config, "expand_timesteps", False)):
        raise NotImplementedError(
            f"{task} RL does not yet support expand_timesteps pipelines.",
        )
    if (
        _config_value(pipeline.config, "boundary_ratio") is not None
        and getattr(pipeline, "transformer_2", None) is None
    ):
        raise ValueError(f"{task} dual-stage pipeline is missing transformer_2")


def _apply_offload_mode(
    pipeline: Any,
    spec: Any,
    *,
    eager_module_dtypes: Mapping[str, torch.dtype],
) -> None:
    """Apply model.offload_mode inside the Wan Diffusers loaders."""

    extra = spec.model_config or {}
    legacy = sorted(
        key
        for key in ("enable_model_cpu_offload", "enable_sequential_cpu_offload")
        if key in extra
    )
    if legacy:
        raise ValueError(
            f"removed Wan model config key(s): {', '.join('model.' + key for key in legacy)}; "
            "use model.offload_mode='none', 'model', or 'sequential'",
        )

    mode = extra.get("offload_mode", "none")
    if mode is None or mode == "":
        mode = "none"
    mode = str(mode)
    if mode not in {"none", "model", "sequential"}:
        raise ValueError(
            f"model.offload_mode must be one of ['none', 'model', 'sequential'], got {mode!r}",
        )

    # Diffusers exposes two mutually exclusive accelerate hooks here:
    # sequential streams per layer and is the 32 GB Wan I2V escape hatch; model
    # offload stages full modules and only works when the transformer fits.
    gpu_id = getattr(spec.device, "index", None) or 0
    if mode == "sequential":
        pipeline.enable_sequential_cpu_offload(gpu_id=gpu_id)
        return
    if mode == "model":
        pipeline.enable_model_cpu_offload(gpu_id=gpu_id)
        return

    for module_name, dtype in eager_module_dtypes.items():
        module = getattr(pipeline, module_name, None)
        if module is not None:
            module.to(spec.device, dtype=dtype)


def _resolve_guidance_scale_2(
    request: VideoGenerationRequest,
    guidance_scale: float,
    boundary_ratio: float | None,
) -> float | None:
    if boundary_ratio is None:
        return None
    raw = request.extra.get("guidance_scale_2") if request.extra else None
    if raw is None:
        return guidance_scale
    return float(raw)


def _uses_cfg(guidance_scale: float, guidance_scale_2: float | None) -> bool:
    return guidance_scale > 1.0 or (guidance_scale_2 is not None and guidance_scale_2 > 1.0)


def _uses_low_noise_transformer(
    timestep: torch.Tensor,
    *,
    boundary_ratio: float,
    num_train_timesteps: int | None,
) -> bool:
    if num_train_timesteps is None:
        raise ValueError("Wan dual-stage routing requires scheduler.config.num_train_timesteps")
    boundary_timestep = float(boundary_ratio) * int(num_train_timesteps)
    values = torch.as_tensor(timestep).detach().to(dtype=torch.float32)
    low_noise = values < boundary_timestep
    first = bool(low_noise.reshape(-1)[0].item())
    if bool((low_noise != first).any().item()):
        raise ValueError(
            "Wan dual-stage replay batch crosses the transformer boundary within "
            "one forward_step; batch samples must share the same denoise step",
        )
    return first


def _scheduler_num_train_timesteps(scheduler: Any) -> int | None:
    value = _config_value(getattr(scheduler, "config", None), "num_train_timesteps")
    return None if value is None else int(value)


def _module_dtype(module: Any) -> torch.dtype:
    dtype = getattr(module, "dtype", None)
    if dtype is not None:
        return dtype
    try:
        return next(module.parameters()).dtype
    except StopIteration as exc:
        raise RuntimeError(
            f"{type(module).__name__} has no parameters to infer dtype",
        ) from exc


def _optional_float(value: Any, field_name: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a float or null") from exc


def _normalize_trainable_transformers(value: Any, *, dual_stage: bool) -> tuple[str, ...]:
    if value is None or value == "":
        names = ("transformer_2",) if dual_stage else ("transformer",)
    elif isinstance(value, str):
        text = value.strip().lower()
        names = ("transformer", "transformer_2") if text in {"all", "both"} else (text,)
    else:
        names = tuple(str(item).strip().lower() for item in value)

    allowed = {"transformer", "transformer_2"} if dual_stage else {"transformer"}
    out: list[str] = []
    for name in names:
        if name not in allowed:
            raise ValueError(
                f"invalid Wan trainable transformer {name!r}; allowed={sorted(allowed)}",
            )
        if name not in out:
            out.append(name)
    if not out:
        raise ValueError("Wan trainable_transformers must not be empty")
    return tuple(out)


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
