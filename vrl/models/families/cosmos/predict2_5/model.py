"""Cosmos Predict2.5 diffusers-backed text-to-world model."""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from typing import Any, Literal

import torch

from vrl.generation.types import DenoiseRequest
from vrl.models.families.cosmos import (
    CosmosReplayForward,
    NoOpCosmosSafetyChecker,
    no_safety_checker,
)
from vrl.models.interfaces.runtime import ModelBuild
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
    expand_tensor_to_batch,
    replay_tensor,
    shared_replay_tensor,
)


@dataclass(slots=True)
class CosmosPredict25DenoiseBackboneRunner(DenoiseBackboneRunnerBase):
    """Map Cosmos Predict2.5 transformer kwargs into the shared backbone contract."""

    cfg_mode = "separate_cfg"
    cfg_base = "cond"
    sigma: torch.Tensor
    # Compile-time constant of the reference pipeline, not a knob: nothing ever
    # produced another value, so it does not travel through SamplingState or
    # the replay batch_context.
    conditional_frame_timestep: float = 0.1

    def build_branch(
        self,
        request: DenoiseBackboneInput,
        branch: Literal["cond", "uncond"],
    ) -> DenoiseBranch:
        extra = request.extra
        embeds = request.prompt_embeds
        if branch == "uncond":
            embeds = request.negative_prompt_embeds
        hidden_states, timestep, gt_velocity = self._prepare_branch(
            latents=request.hidden_states,
            cond_latent=extra["cond_latent"],
            cond_mask=extra["cond_mask"],
            cond_indicator=extra["cond_indicator"],
        )
        return DenoiseBranch(
            hidden_states=hidden_states.to(extra["transformer_dtype"]),
            timestep=timestep.to(extra["transformer_dtype"]),
            encoder_hidden_states=embeds,
            extra_kwargs={
                "condition_mask": extra["cond_mask"].to(extra["transformer_dtype"]),
                "padding_mask": extra["padding_mask"],
            },
            metadata={"gt_velocity": gt_velocity},
        )

    def postprocess_branch(
        self,
        request: DenoiseBackboneInput,
        branch: DenoiseBranch,
        raw_output: torch.Tensor,
    ) -> torch.Tensor:
        return branch.metadata["gt_velocity"] + raw_output * (1 - request.extra["cond_mask"])

    def _prepare_branch(
        self,
        *,
        latents: torch.Tensor,
        cond_latent: torch.Tensor,
        cond_mask: torch.Tensor,
        cond_indicator: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        condition_timestep = torch.ones_like(cond_indicator) * float(
            self.conditional_frame_timestep,
        )
        hidden_states = cond_mask * cond_latent + (1 - cond_mask) * latents
        timestep = cond_indicator * condition_timestep + (1 - cond_indicator) * self.sigma
        gt_velocity = (latents - cond_latent) * cond_mask
        return hidden_states, timestep, gt_velocity


def _load_pipeline_without_text_encoder(
    pipeline_cls: Any,
    build: ModelBuild,
    *,
    revision: str | None,
) -> Any:
    from diffusers import AutoencoderKLWan, CosmosTransformer3DModel, UniPCMultistepScheduler

    load_kwargs: dict[str, Any] = {"revision": revision} if revision else {}

    transformer = CosmosTransformer3DModel.from_pretrained(
        build.model_name_or_path,
        subfolder="transformer",
        torch_dtype=build.parameter_dtype,
        **load_kwargs,
    )
    vae = AutoencoderKLWan.from_pretrained(
        build.model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float32,
        **load_kwargs,
    )
    scheduler = UniPCMultistepScheduler.from_pretrained(
        build.model_name_or_path,
        subfolder="scheduler",
        **load_kwargs,
    )
    return pipeline_cls(
        text_encoder=None,
        tokenizer=None,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        safety_checker=NoOpCosmosSafetyChecker(),
    )


@dataclass
class CosmosPredict25SamplingState(GuidedDenoiseSamplingStateBase):
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None
    do_cfg: bool
    cond_latent: torch.Tensor
    cond_mask: torch.Tensor
    cond_indicator: torch.Tensor
    padding_mask: torch.Tensor
    height: int
    width: int
    num_frames: int
    fps: int


class CosmosPredict25Model(CosmosReplayForward, DiffusersPipelineModelBase):
    """Cosmos-Predict2.5 PredictBase model behind the shared denoise runtime."""

    def __init__(
        self,
        *,
        pipeline: Any,
        device: Any = None,
        synthetic_prompt_embeds: bool = False,
    ) -> None:
        super().__init__(pipeline=pipeline, device=device)
        self.synthetic_prompt_embeds = bool(synthetic_prompt_embeds)

    @classmethod
    def from_build(cls, build: ModelBuild) -> CosmosPredict25Model:
        import diffusers.pipelines.cosmos.pipeline_cosmos2_5_predict as _predict_mod
        from diffusers import Cosmos2_5_PredictBasePipeline

        prompt_dtype, kwargs = cls._pipeline_load_dtypes(
            build,
            build.parameter_dtype,
        )
        revision = kwargs.get("revision")
        skip_text_encoder = bool((build.model_config or {}).get("skip_text_encoder", False))
        with torch.set_grad_enabled(torch.is_grad_enabled()):
            if skip_text_encoder:
                pipeline = _load_pipeline_without_text_encoder(
                    Cosmos2_5_PredictBasePipeline,
                    build,
                    revision=revision,
                )
            else:
                with no_safety_checker(_predict_mod):
                    pipeline = Cosmos2_5_PredictBasePipeline.from_pretrained(
                        build.model_name_or_path,
                        **kwargs,
                    )
        pipeline.set_progress_bar_config(disable=True)
        cls.freeze_pipeline_components(pipeline)
        cpu_resident = cls.place_pipeline_components(
            pipeline, build, prompt_encoder_dtype=prompt_dtype
        )
        model = cls(
            pipeline=pipeline,
            device=build.device,
            synthetic_prompt_embeds=skip_text_encoder,
        )
        model._cpu_resident = cpu_resident
        return model

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
        request: DenoiseRequest,
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
        # Slice the step's sigma on the device: a host round trip here would
        # synchronise the stream once per denoise step for a value that is
        # only ever consumed on the device.
        sigma_t = (
            state.scheduler.sigmas[step_idx]
            .detach()
            .to(torch.float32)
            .reshape(1)
            .to(device=state.latents.device, dtype=transformer_dtype)
        )
        output = DenoiseBackboneCaller(
            self.transformer,
            CosmosPredict25DenoiseBackboneRunner(sigma=sigma_t),
        )(
            DenoiseBackboneInput(
                hidden_states=state.latents,
                timestep=sigma_t,
                prompt_embeds=state.prompt_embeds,
                negative_prompt_embeds=state.negative_prompt_embeds,
                guidance_scale=state.guidance_scale,
                do_cfg=state.do_cfg,
                output_dtype=transformer_dtype,
                extra={
                    "cond_latent": state.cond_latent,
                    "cond_mask": state.cond_mask,
                    "cond_indicator": state.cond_indicator,
                    "padding_mask": state.padding_mask,
                    "transformer_dtype": transformer_dtype,
                },
            ),
        )
        return output.as_dict()

    def export_batch_context(self, state: CosmosPredict25SamplingState) -> dict[str, Any]:
        return {
            "guidance_scale": state.guidance_scale,
            "cfg": state.do_cfg,
            "height": state.height,
            "width": state.width,
            "num_frames": state.num_frames,
            "fps": state.fps,
        }

    def export_replay_tensors(self, state: CosmosPredict25SamplingState) -> dict[str, Any]:
        return {
            "prompt_embeds": state.prompt_embeds,
            "negative_prompt_embeds": state.negative_prompt_embeds,
            "prompt_attention_mask": None,
            "pooled_prompt_embeds": None,
            "latents_clean": state.latents.detach(),
            "cond_mask": expand_tensor_to_batch(
                state.cond_mask, state.latents.shape[0], materialize=True
            ),
            "cond_indicator": expand_tensor_to_batch(
                state.cond_indicator, state.latents.shape[0], materialize=True
            ),
            "padding_mask": expand_tensor_to_batch(
                state.padding_mask, state.latents.shape[0], materialize=True
            ),
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
            timesteps=self.scheduler.timesteps,
            scheduler=self.scheduler,
            prompt_embeds=replay_tensors["prompt_embeds"],
            negative_prompt_embeds=replay_tensors.get("negative_prompt_embeds"),
            guidance_scale=batch_context["guidance_scale"],
            do_cfg=batch_context["cfg"] and batch_context["guidance_scale"] > 1.0,
            cond_latent=cond_latent,
            cond_mask=replay_tensor(replay_tensors, batch_context, "cond_mask"),
            cond_indicator=replay_tensor(replay_tensors, batch_context, "cond_indicator"),
            padding_mask=shared_replay_tensor(
                replay_tensors,
                batch_context,
                "padding_mask",
            ),
            height=batch_context["height"],
            width=batch_context["width"],
            num_frames=batch_context["num_frames"],
            fps=batch_context["fps"],
        )

    def encode_video_to_latents(self, video: torch.Tensor) -> torch.Tensor:
        """Encode clean [0, 1] video into this checkpoint's latent domain.

        ``decode_latents`` unnormalizes with ``latent * latents_std +
        latents_mean`` before VAE decode. This is the exact inverse used by the
        offline SFT-target encoder; deterministic posterior modes keep shard
        generation reproducible.
        """

        pipe = self.pipeline
        x = (video.to(pipe.vae.dtype) * 2.0 - 1.0).to(self.device)
        with torch.no_grad():
            encoded = torch.cat(
                [pipe.vae.encode(sample.unsqueeze(0)).latent_dist.mode() for sample in x],
                dim=0,
            )
        latents_mean = pipe.latents_mean.to(encoded.device, encoded.dtype)
        latents_std = pipe.latents_std.to(encoded.device, encoded.dtype)
        return (encoded - latents_mean) / latents_std

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        pipe = self.pipeline
        latents_mean = pipe.latents_mean.to(latents.device, latents.dtype)
        latents_std = pipe.latents_std.to(latents.device, latents.dtype)
        frame_count = int((latents.shape[2] - 1) * pipe.vae_scale_factor_temporal + 1)
        decoder = ChunkedLatentDecoder(
            LatentDecodePlan(
                prepare_latents=lambda batch: (batch * latents_std + latents_mean).to(
                    pipe.vae.dtype,
                ),
                vae_decode=lambda batch: pipe.vae.decode(
                    batch,
                    return_dict=False,
                )[0],
                prepare_decoded=lambda video: pipe._match_num_frames(
                    video,
                    frame_count,
                ),
                postprocess=lambda video: pipe.video_processor.postprocess_video(
                    video,
                    output_type="pt",
                ),
                output_layout="video_btchw",
                decode_batch_size=getattr(pipe, "decode_batch_size", None),
            ),
        )
        return decoder(latents)


class CosmosPredict25ReplayModel(DiffusersReplayModelBase, CosmosPredict25Model):
    """Replay-only Cosmos Predict2.5 model without text encoder, VAE, or pipeline."""

    def __init__(self, *, transformer: Any, scheduler: Any, device: Any = None) -> None:
        DiffusersReplayModelBase.__init__(
            self,
            transformer=transformer,
            scheduler=scheduler,
            device=device,
        )
        self.synthetic_prompt_embeds = False

    # apply_lora is inherited from DenoiseModelBase: it walks
    # trainable_modules, which this replay model owns directly (no pipeline).
    # torch_compile_transformer is inherited from DenoiseModelBase: it calls
    # self._set_transformer, which the replay base owns.

    # set_num_steps is inherited from DiffusersPipelineModelBase: it reads
    # self.scheduler (overridden below to this replay model's own scheduler)
    # and UniPC has no dynamic shifting, so the base body is exact.

    # restore_eval_state is inherited from CosmosPredict25Model: it reads
    # ``self.scheduler``, which this replay model overrides to return its own
    # ``self._scheduler`` (the parent's property resolves to pipeline.scheduler).


__all__ = [
    "CosmosPredict25Model",
    "CosmosPredict25ReplayModel",
    "CosmosPredict25SamplingState",
]
