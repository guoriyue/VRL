"""Cosmos Predict2.5 diffusers-backed model for DiffusionNFT RL."""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from typing import Any

import torch

from vrl.engine.diffusion.request import VideoGenerationRequest
from vrl.models.interfaces import ReplayRequest, ReplayResult, ReplaySegmentResult
from vrl.models.diffusion import DiffusionModelBase


class _NoOpCosmosSafetyChecker:
    """Avoid loading guardrail weights in internal optimization diagnostics."""

    def to(self, *_args: Any, **_kwargs: Any) -> _NoOpCosmosSafetyChecker:
        return self

    def check_text_safety(self, _prompt: str) -> bool:
        return True


def _load_pipeline_without_text_encoder(
    pipeline_cls: Any,
    spec: Any,
    *,
    revision: str | None,
) -> Any:
    from diffusers import AutoencoderKLWan, CosmosTransformer3DModel, UniPCMultistepScheduler

    load_kwargs: dict[str, Any] = {}
    if revision:
        load_kwargs["revision"] = revision

    transformer = CosmosTransformer3DModel.from_pretrained(
        spec.model_name_or_path,
        subfolder="transformer",
        torch_dtype=spec.dtype,
        **load_kwargs,
    )
    vae = AutoencoderKLWan.from_pretrained(
        spec.model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float32,
        **load_kwargs,
    )
    scheduler = UniPCMultistepScheduler.from_pretrained(
        spec.model_name_or_path,
        subfolder="scheduler",
        **load_kwargs,
    )
    return pipeline_cls(
        text_encoder=None,
        tokenizer=None,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        safety_checker=_NoOpCosmosSafetyChecker(),
    )


@dataclass
class CosmosPredict25SamplingState:
    latents: torch.Tensor
    timesteps: torch.Tensor
    scheduler: Any
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None
    guidance_scale: float
    do_cfg: bool
    cond_latent: torch.Tensor
    cond_mask: torch.Tensor
    cond_indicator: torch.Tensor
    padding_mask: torch.Tensor
    seed: int
    height: int
    width: int
    num_frames: int
    fps: int
    conditional_frame_timestep: float = 0.1


class CosmosPredict25Model(DiffusionModelBase):
    """Cosmos-Predict2.5 PredictBase model with DiffusionNFT training extras."""

    family = "cosmos-predict2.5-diffusers"

    def __init__(
        self,
        *,
        pipeline: Any,
        device: Any = None,
        synthetic_prompt_embeds: bool = False,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_pipeline", pipeline)
        self.transformer = pipeline.transformer
        self._device = device
        self.synthetic_prompt_embeds = bool(synthetic_prompt_embeds)

    @property
    def pipeline(self) -> Any:
        return self._pipeline

    @property
    def scheduler(self) -> Any:
        return self.pipeline.scheduler

    @property
    def backend_handle(self) -> Any:
        return self.pipeline

    @property
    def device(self) -> Any:
        return self._device if self._device is not None else self.pipeline.device

    @property
    def trainable_modules(self) -> dict[str, Any]:
        return {"transformer": self.transformer}

    def _set_transformer(self, transformer: Any) -> None:
        self.transformer = transformer
        self.pipeline.transformer = transformer

    @classmethod
    def from_spec(cls, spec: Any) -> CosmosPredict25Model:
        from diffusers import Cosmos2_5_PredictBasePipeline

        kwargs: dict[str, Any] = {"torch_dtype": spec.dtype}
        revision = (spec.extra or {}).get("model_revision")
        if revision:
            kwargs["revision"] = revision
        skip_text_encoder = bool((spec.extra or {}).get("skip_text_encoder", False))
        if skip_text_encoder:
            pipeline = _load_pipeline_without_text_encoder(
                Cosmos2_5_PredictBasePipeline,
                spec,
                revision=revision,
            )
        else:
            pipeline = Cosmos2_5_PredictBasePipeline.from_pretrained(
                spec.model_name_or_path,
                **kwargs,
            )
        torch.set_grad_enabled(True)
        pipeline.set_progress_bar_config(disable=True)
        if hasattr(pipeline, "vae"):
            pipeline.vae.requires_grad_(False)
            pipeline.vae.to(spec.device, dtype=torch.float32)
        if getattr(pipeline, "text_encoder", None) is not None:
            pipeline.text_encoder.requires_grad_(False)
            pipeline.text_encoder.to(spec.device, dtype=spec.dtype)
        return cls(
            pipeline=pipeline,
            device=spec.device,
            synthetic_prompt_embeds=skip_text_encoder,
        )

    def apply_lora(self, spec: Any) -> None:
        from peft import LoraConfig, PeftModel, get_peft_model

        self.pipeline.transformer.requires_grad_(False)
        self.pipeline.transformer.to(self.device)
        if spec.lora_path:
            transformer = PeftModel.from_pretrained(
                self.pipeline.transformer,
                spec.lora_path,
                is_trainable=True,
                adapter_name="default",
            )
        else:
            assert spec.lora_config is not None
            cfg = LoraConfig(
                r=spec.lora_config["rank"],
                lora_alpha=spec.lora_config["alpha"],
                init_lora_weights="gaussian",
                target_modules=spec.lora_config["target_modules"],
            )
            transformer = get_peft_model(self.pipeline.transformer, cfg, adapter_name="default")

        if "previous" not in getattr(transformer, "peft_config", {}):
            assert spec.lora_config is not None
            previous_cfg = LoraConfig(
                r=spec.lora_config["rank"],
                lora_alpha=spec.lora_config["alpha"],
                init_lora_weights="gaussian",
                target_modules=spec.lora_config["target_modules"],
            )
            transformer.add_adapter("previous", previous_cfg)
        _copy_adapter_weights(transformer, src="default", dst="previous")
        transformer.set_adapter("default")
        self._set_transformer(transformer)

    def sync_previous_policy_adapter(self, *, decay: float = 0.0) -> None:
        """Refresh the DiffusionNFT previous-policy adapter from the trainable adapter."""

        _copy_adapter_weights(
            self.transformer,
            src="default",
            dst="previous",
            decay=decay,
        )

    def enable_full_finetune(self) -> None:
        raise RuntimeError(
            "Cosmos Predict2.5 DiffusionNFT requires LoRA with default+previous "
            "adapters; set model.use_lora=true.",
        )

    def torch_compile_transformer(self, mode: str) -> None:
        self._set_transformer(torch.compile(self.pipeline.transformer, mode=mode, fullgraph=False))

    def set_num_steps(self, n: int) -> None:
        self.pipeline.scheduler.set_timesteps(n, device=self.device)

    def describe(self) -> dict[str, Any]:
        return {"family": self.family, "device": str(self.device)}

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        guidance_scale = float(kwargs.get("guidance_scale", 1.0))
        do_cfg = guidance_scale > 1.0
        max_sequence_length = int(kwargs.get("max_sequence_length", 512))
        if self.synthetic_prompt_embeds:
            prompt_embeds, negative_prompt_embeds = self._zero_prompt_embeds(
                prompt,
                do_cfg=do_cfg,
                max_sequence_length=max_sequence_length,
            )
        else:
            prompt_embeds, negative_prompt_embeds = self.pipeline.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_cfg,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=self.device,
                dtype=self.pipeline.transformer.dtype,
            )
        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds if do_cfg else None,
            "prompt_attention_mask": None,
            "pooled_prompt_embeds": None,
        }

    def _zero_prompt_embeds(
        self,
        prompt: str | list[str],
        *,
        do_cfg: bool,
        max_sequence_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        config = self.pipeline.transformer.config
        hidden_dim = int(
            getattr(config, "crossattn_proj_in_channels", None)
            if getattr(config, "use_crossattn_projection", False)
            else (
                getattr(config, "text_embed_dim", None)
                or getattr(config, "encoder_hidden_states_channels", 0)
            )
        )
        if hidden_dim <= 0:
            raise ValueError("Cosmos Predict2.5 transformer config lacks text embedding dim")
        prompt_embeds = torch.zeros(
            batch_size,
            max_sequence_length,
            hidden_dim,
            device=self.device,
            dtype=self.pipeline.transformer.dtype,
        )
        negative_prompt_embeds = torch.zeros_like(prompt_embeds) if do_cfg else None
        return prompt_embeds, negative_prompt_embeds

    def prepare_sampling(
        self,
        request: VideoGenerationRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> CosmosPredict25SamplingState:
        del kwargs
        pipe = self.pipeline
        device = self.device
        prompt_embeds = encoded["prompt_embeds"]
        guidance_scale = request.guidance_scale
        do_cfg = guidance_scale > 1.0
        pipe.scheduler.set_timesteps(request.num_steps, device=device)
        seed = request.seed if request.seed is not None else random.randint(0, sys.maxsize)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        batch_size = prompt_embeds.shape[0]
        num_channels_latents = pipe.transformer.config.in_channels - 1
        latents, cond_latent, cond_mask, cond_indicator = pipe.prepare_latents(
            video=None,
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=request.height,
            width=request.width,
            num_frames_in=0,
            num_frames_out=request.frame_count,
            do_classifier_free_guidance=do_cfg,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=None,
        )
        padding_mask = latents.new_zeros(
            1,
            1,
            request.height,
            request.width,
            dtype=prompt_embeds.dtype,
        )
        return CosmosPredict25SamplingState(
            latents=latents,
            timesteps=pipe.scheduler.timesteps,
            scheduler=pipe.scheduler,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=encoded.get("negative_prompt_embeds"),
            guidance_scale=guidance_scale,
            do_cfg=do_cfg,
            cond_latent=cond_latent,
            cond_mask=cond_mask,
            cond_indicator=cond_indicator,
            padding_mask=padding_mask,
            seed=seed,
            height=request.height,
            width=request.width,
            num_frames=request.frame_count,
            fps=request.fps or 16,
        )

    def forward_step(
        self,
        state: CosmosPredict25SamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        transformer_dtype = state.prompt_embeds.dtype
        state.scheduler.sigmas = state.scheduler.sigmas.to(state.latents.device)
        sigma_t = (
            torch.tensor(state.scheduler.sigmas[step_idx].item())
            .unsqueeze(0)
            .to(device=state.latents.device, dtype=transformer_dtype)
        )
        cond_timestep = torch.ones_like(state.cond_indicator) * state.conditional_frame_timestep
        in_latents = state.cond_mask * state.cond_latent + (1 - state.cond_mask) * state.latents
        in_latents = in_latents.to(transformer_dtype)
        in_timestep = state.cond_indicator * cond_timestep + (1 - state.cond_indicator) * sigma_t

        noise_pred_cond = self.transformer(
            hidden_states=in_latents,
            condition_mask=state.cond_mask.to(transformer_dtype),
            timestep=in_timestep.to(transformer_dtype),
            encoder_hidden_states=state.prompt_embeds,
            padding_mask=state.padding_mask,
            return_dict=False,
        )[0]
        gt_velocity = (state.latents - state.cond_latent) * state.cond_mask
        noise_pred_cond = gt_velocity + noise_pred_cond * (1 - state.cond_mask)

        if state.do_cfg:
            noise_pred_uncond = self.transformer(
                hidden_states=in_latents,
                condition_mask=state.cond_mask.to(transformer_dtype),
                timestep=in_timestep.to(transformer_dtype),
                encoder_hidden_states=state.negative_prompt_embeds,
                padding_mask=state.padding_mask,
                return_dict=False,
            )[0]
            noise_pred_uncond = gt_velocity + noise_pred_uncond * (1 - state.cond_mask)
            noise_pred = noise_pred_cond + state.guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
        else:
            noise_pred_uncond = torch.zeros_like(noise_pred_cond)
            noise_pred = noise_pred_cond
        return {
            "noise_pred": noise_pred,
            "noise_pred_cond": noise_pred_cond,
            "noise_pred_uncond": noise_pred_uncond,
        }

    def export_batch_context(self, state: CosmosPredict25SamplingState) -> dict[str, Any]:
        return {
            "guidance_scale": state.guidance_scale,
            "cfg": state.do_cfg,
            "model_family": self.family,
            "height": state.height,
            "width": state.width,
            "num_frames": state.num_frames,
            "fps": state.fps,
            "cond_mask": state.cond_mask,
            "cond_indicator": state.cond_indicator,
            "padding_mask": state.padding_mask,
            "conditional_frame_timestep": state.conditional_frame_timestep,
        }

    def export_replay_tensors(self, state: CosmosPredict25SamplingState) -> dict[str, Any]:
        return {
            "prompt_embeds": state.prompt_embeds,
            "negative_prompt_embeds": state.negative_prompt_embeds,
            "prompt_attention_mask": None,
            "pooled_prompt_embeds": None,
            "latents_clean": state.latents.detach(),
        }

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> CosmosPredict25SamplingState:
        del step_idx
        cond_latent = torch.zeros_like(latents)
        return CosmosPredict25SamplingState(
            latents=latents,
            timesteps=self.pipeline.scheduler.timesteps,
            scheduler=self.pipeline.scheduler,
            prompt_embeds=replay_tensors["prompt_embeds"],
            negative_prompt_embeds=replay_tensors.get("negative_prompt_embeds"),
            guidance_scale=batch_context["guidance_scale"],
            do_cfg=batch_context["cfg"] and batch_context["guidance_scale"] > 1.0,
            cond_latent=cond_latent,
            cond_mask=batch_context["cond_mask"],
            cond_indicator=batch_context["cond_indicator"],
            padding_mask=batch_context["padding_mask"],
            seed=0,
            height=batch_context["height"],
            width=batch_context["width"],
            num_frames=batch_context["num_frames"],
            fps=batch_context["fps"],
            conditional_frame_timestep=batch_context.get("conditional_frame_timestep", 0.1),
        )

    def replay_forward(
        self,
        batch: Any,
        timestep_idx: int,
        *,
        request: ReplayRequest | None = None,
    ) -> ReplayResult:
        del request
        from vrl.engine.trajectory import trajectory_replay_tensor_dict

        state = self.restore_eval_state(
            trajectory_replay_tensor_dict(batch, "denoise"),
            batch.context,
            batch.observations[:, timestep_idx],
            timestep_idx,
        )
        values = self.forward_step(state, timestep_idx)
        return ReplayResult(
            segments={
                "denoise": ReplaySegmentResult(
                    segment="denoise",
                    values=dict(values),
                ),
            },
        )

    def diffusion_nft_prepare_transformer_input(
        self,
        *,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor | None,
        pooled_prompt_embeds: torch.Tensor | None,
        timestep: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del prompt_attention_mask, pooled_prompt_embeds, kwargs
        batch = latents.shape[0]
        latent_frames = (num_frames - 1) // self.pipeline.vae_scale_factor_temporal + 1
        latent_h = height // self.pipeline.vae_scale_factor_spatial
        latent_w = width // self.pipeline.vae_scale_factor_spatial
        cond_mask = torch.zeros(
            (batch, 1, latent_frames, latent_h, latent_w),
            dtype=latents.dtype,
            device=latents.device,
        )
        padding_mask = latents.new_zeros(1, 1, height, width, dtype=latents.dtype)
        return {
            "hidden_states": latents,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "condition_mask": cond_mask,
            "padding_mask": padding_mask,
            "return_dict": False,
        }

    nft_prepare_transformer_input = diffusion_nft_prepare_transformer_input

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        pipe = self.pipeline
        latents_mean = pipe.latents_mean.to(latents.device, latents.dtype)
        latents_std = pipe.latents_std.to(latents.device, latents.dtype)
        x = latents * latents_std + latents_mean
        video = pipe.vae.decode(x.to(pipe.vae.dtype), return_dict=False)[0]
        video = pipe._match_num_frames(video, int((latents.shape[2] - 1) * pipe.vae_scale_factor_temporal + 1))
        video = pipe.video_processor.postprocess_video(video, output_type="pt")
        return video.permute(0, 2, 1, 3, 4)


__all__ = ["CosmosPredict25Model", "CosmosPredict25SamplingState"]


def _copy_adapter_weights(
    module: Any,
    *,
    src: str,
    dst: str,
    decay: float = 0.0,
) -> None:
    named = dict(module.named_parameters())
    copied = 0
    decay = float(decay)
    if not 0.0 <= decay <= 1.0:
        raise ValueError(f"adapter weight copy decay must be in [0, 1], got {decay}")
    for name, param in named.items():
        src_marker = f".{src}."
        if src_marker not in name:
            continue
        dst_name = name.replace(src_marker, f".{dst}.")
        dst_param = named.get(dst_name)
        if dst_param is None:
            continue
        if decay == 0.0:
            dst_param.data.copy_(param.data)
        else:
            dst_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)
        copied += 1
    if copied == 0:
        raise RuntimeError(
            f"failed to copy adapter weights from {src!r} to {dst!r}; "
            "no matching adapter parameters were found",
        )
