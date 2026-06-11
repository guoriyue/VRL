"""Cosmos Predict2 (Video2World) diffusers-backed model.

Diffusion implementation for the Cosmos Predict2 Video2World pipeline. The
generation helper flow is:

    encode_prompt -> prepare_sampling -> forward_step xN -> decode_latents

The collector owns the scheduler step / SDE step. ``forward_step`` does
only one transformer forward (with optional CFG branch) and returns noise
predictions; it does NOT mutate ``state.latents``.

Per-family :class:`CosmosPredict2SamplingState` is private to this file —
collector code MUST NOT introspect it beyond the documented attributes
(``latents``, ``timesteps``, ``scheduler``); cosmos-specific conditioning
fields are projected via ``export_*`` / ``restore_eval_state``.
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
    align_replay_tensor,
    broadcast_spatial_timestep,
    replay_tensor,
    shared_replay_tensor,
)
from vrl.models.diffusion.common.lora import LoraModelMixin
from vrl.models.diffusion.common.vae_decode_memory import apply_vae_decode_memory
from vrl.models.diffusion.cosmos.predict2.runner import (
    CosmosPredict2DiffusionBackboneRunner,
)
from vrl.models.diffusion.cosmos import CosmosReplayForward
from vrl.models.interfaces import ReplayRequest, ReplayResult, ReplaySegmentResult


@dataclass
class CosmosPredict2SamplingState:
    """Private Cosmos Predict2 sampling state. Collector MUST NOT introspect.

    Cosmos Predict2 Video2World needs the full conditioning bundle
    (``init_latents`` + cond/uncond masks + indicators + padding mask + fps)
    in the per-step transformer forward and in the SDE step. Keeping the
    bundle inside the model avoids leaking these knobs into the engine
    contract.
    """

    latents: torch.Tensor
    timesteps: torch.Tensor
    scheduler: Any
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None
    guidance_scale: float
    do_cfg: bool
    init_latents: torch.Tensor
    cond_mask: Any
    uncond_mask: Any
    padding_mask: Any
    cond_indicator: Any
    uncond_indicator: Any
    fps: int
    seed: int
    sigma_conditioning: float = 0.0001


class CosmosPredict2Model(CosmosReplayForward, LoraModelMixin, DiffusionModelBase):
    """Diffusers-backed Cosmos Predict2 Video2World model (RL path).

    The pipeline is constructed by the family runtime
    (:func:`vrl.models.diffusion.cosmos.predict2.runtime.build_cosmos_predict2_runtime_bundle`)
    and passed in. Scripts must NOT instantiate the diffusers pipeline
    directly.
    """

    family = "cosmos-predict2-diffusers-v2w"

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

    # -- backend ownership (called by runtime, not by collectors) -------

    @classmethod
    def from_spec(cls, spec: Any) -> CosmosPredict2Model:
        """Load Cosmos2VideoToWorldPipeline + freeze non-trainable modules.

        Patches the diffusers safety checker with a passthrough during load
        so RL training does not depend on the safety classifier weights.
        """
        import diffusers.pipelines.cosmos.pipeline_cosmos2_video2world as _v2w_mod
        from diffusers import Cosmos2VideoToWorldPipeline

        class _PassthroughSafetyChecker:
            def to(self, device: Any) -> _PassthroughSafetyChecker:
                return self

            def check_text_safety(self, prompt: str) -> bool:
                return True

            def check_video_safety(self, video: Any) -> Any:
                return video

        _orig = _v2w_mod.CosmosSafetyChecker
        _v2w_mod.CosmosSafetyChecker = _PassthroughSafetyChecker  # type: ignore[assignment]
        try:
            pipeline = Cosmos2VideoToWorldPipeline.from_pretrained(
                spec.model_name_or_path, torch_dtype=spec.dtype,
            )
        finally:
            _v2w_mod.CosmosSafetyChecker = _orig

        # diffusers from_pretrained disables grad globally — re-enable.
        torch.set_grad_enabled(True)

        pipeline.set_progress_bar_config(disable=True)
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        pipeline.vae.to(spec.device, dtype=torch.float32)
        memory_metadata = apply_vae_decode_memory(
            pipeline.vae,
            memory_config=getattr(spec, "memory", None),
            owner="Cosmos Predict2 VAE",
        )
        pipeline.text_encoder.to(spec.device, dtype=spec.dtype)
        return cls(
            pipeline=pipeline,
            device=spec.device,
            memory_metadata=memory_metadata,
        )

    def enable_full_finetune(self) -> None:
        self.pipeline.transformer.requires_grad_(True)
        self.pipeline.transformer.to(self.device)

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

    @property
    def device(self) -> Any:
        return self._device if self._device is not None else self.pipeline.device

    # -- encode_prompt -------------------------------------------------

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode prompt + optional negative for Cosmos Predict2.

        ``reference_image`` may be provided here so callers can plumb it
        through alongside text encoding; it's threaded into
        ``prepare_sampling`` via the returned dict so the collector can
        keep a single ``encoded`` handle.
        """
        guidance_scale = kwargs.get("guidance_scale", 7.0)
        do_cfg = guidance_scale > 1.0
        neg = negative_prompt if negative_prompt is not None else ""

        device = self.device
        encode_result = self.pipeline.encode_prompt(
            prompt=prompt,
            negative_prompt=neg or None,
            do_classifier_free_guidance=do_cfg,
            device=device,
        )
        prompt_embeds = encode_result[0]
        negative_prompt_embeds = encode_result[1] if do_cfg else None

        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "reference_image": kwargs.get("reference_image"),
        }

    # -- prepare_sampling ----------------------------------------------

    def prepare_sampling(
        self,
        request: VideoGenerationRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> CosmosPredict2SamplingState:
        """Build the per-request Cosmos Predict2 sampling state.

        Uses ``pipeline.prepare_latents(...)`` to materialize the
        Video2World 6-tuple (latents, init_latents, cond_indicator,
        uncond_indicator, cond_mask, uncond_mask) plus a zero
        ``padding_mask`` at pixel resolution that the transformer
        repeats internally.
        """
        pipe = self.pipeline
        device = self.device

        prompt_embeds = encoded["prompt_embeds"]
        negative_prompt_embeds = encoded.get("negative_prompt_embeds")
        reference_image = kwargs.get("reference_image", encoded.get("reference_image"))

        guidance_scale = request.guidance_scale
        do_cfg = guidance_scale > 1.0

        pipe.scheduler.set_timesteps(request.num_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        num_channels_latents = pipe.transformer.config.in_channels - 1
        batch_size = prompt_embeds.shape[0]

        # Reference image preprocessing for Video2World; fall back to zero
        # video conditioning when no reference is provided (degenerate but
        # enough for minimal validation paths).
        if reference_image is not None:
            video_input = pipe.video_processor.preprocess_video(
                reference_image,
                height=request.height,
                width=request.width,
            ).to(device, dtype=pipe.vae.dtype)
        else:
            video_input = torch.zeros(
                batch_size, 3, 1, request.height, request.width,
                device=device, dtype=pipe.vae.dtype,
            )

        seed = (
            request.seed if request.seed is not None
            else random.randint(0, sys.maxsize)
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        latents_result = pipe.prepare_latents(
            video=video_input,
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=request.height,
            width=request.width,
            num_frames=request.frame_count,
            do_classifier_free_guidance=do_cfg,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=None,
        )
        latents = latents_result[0]
        init_latents = latents_result[1]
        cond_indicator = latents_result[2]
        uncond_indicator = latents_result[3]
        cond_mask = latents_result[4]
        uncond_mask = latents_result[5]

        # Padding mask at pixel resolution; transformer repeats internally.
        padding_mask = latents.new_zeros(
            1, 1, request.height, request.width,
            dtype=prompt_embeds.dtype,
        )

        return CosmosPredict2SamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=pipe.scheduler,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            do_cfg=do_cfg,
            init_latents=init_latents,
            cond_mask=cond_mask,
            uncond_mask=uncond_mask,
            padding_mask=padding_mask,
            cond_indicator=cond_indicator,
            uncond_indicator=uncond_indicator,
            fps=request.fps or 16,
            seed=seed,
            sigma_conditioning=0.0001,
        )

    # -- forward_step --------------------------------------------------

    def forward_step(
        self,
        state: CosmosPredict2SamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        """One Cosmos Predict2 transformer forward + optional CFG.

        Mirrors the diffusers Cosmos2VideoToWorldPipeline denoising loop:
        sigma scaling (c_in/c_skip/c_out), spatial timestep blending via
        ``cond_indicator``, and post-transformer conditioning re-application.
        Returns ``{"noise_pred", "noise_pred_cond", "noise_pred_uncond"}``.
        NO scheduler step happens here.
        """
        batch_size = state.latents.shape[0]
        transformer_dtype = state.prompt_embeds.dtype

        # Sigma scaling coefficients from the scheduler.
        current_sigma = state.scheduler.sigmas[step_idx]
        current_t = current_sigma / (current_sigma + 1)

        # Spatial timestep tensor [B, 1, T, 1, 1]
        timestep = broadcast_spatial_timestep(
            current_t,
            batch_size=batch_size,
            frames=state.latents.size(2),
        )

        # Expand cond/uncond fields to runtime batch (prepare_latents may
        # return [1,...] but the collector may have stacked groups).
        cond_mask = state.cond_mask.expand(batch_size, -1, -1, -1, -1)
        uncond_mask = (
            state.uncond_mask.expand(batch_size, -1, -1, -1, -1)
            if state.uncond_mask is not None else None
        )
        # padding_mask kept at [1, 1, H, W] — transformer repeats internally.
        padding_mask = state.padding_mask
        init_latents = state.init_latents.expand(batch_size, -1, -1, -1, -1)
        cond_indicator = state.cond_indicator.expand(batch_size, -1, -1, -1, -1)
        uncond_indicator = (
            state.uncond_indicator.expand(batch_size, -1, -1, -1, -1)
            if state.uncond_indicator is not None else None
        )

        output = DiffusionBackboneCaller(
            self.transformer,
            CosmosPredict2DiffusionBackboneRunner(
                current_sigma=current_sigma,
                sigma_conditioning=state.sigma_conditioning,
            ),
        )(
            DiffusionBackboneInput(
                hidden_states=state.latents,
                timestep=timestep,
                prompt_embeds=state.prompt_embeds,
                negative_prompt_embeds=state.negative_prompt_embeds,
                guidance_scale=state.guidance_scale,
                do_cfg=state.do_cfg,
                output_dtype=transformer_dtype,
                extra={
                    "init_latents": init_latents,
                    "cond_mask": cond_mask,
                    "uncond_mask": uncond_mask,
                    "padding_mask": padding_mask,
                    "cond_indicator": cond_indicator,
                    "uncond_indicator": uncond_indicator,
                    "fps": state.fps,
                    "transformer_dtype": transformer_dtype,
                },
            ),
        )
        return output.as_dict()

    # -- collector boundary --------------------------------------------

    def export_batch_context(
        self,
        state: CosmosPredict2SamplingState,
    ) -> dict[str, Any]:
        """Project SamplingState -> RolloutBatch.context (shared metadata).

        Tensor conditioning is packed in replay tensors, not context, because
        trajectory context is intentionally JSON-safe metadata only.
        """
        return {
            "guidance_scale": state.guidance_scale,
            "cfg": state.do_cfg,
            "model_family": self.family,
            "fps": state.fps,
            "sigma_conditioning": state.sigma_conditioning,
        }

    def export_replay_tensors(
        self,
        state: CosmosPredict2SamplingState,
    ) -> dict[str, Any]:
        """Project SamplingState into trajectory replay tensors.

        ``init_latents`` is per-sample because Video2World conditioning
        depends on the reference image.
        """
        return {
            "prompt_embeds": state.prompt_embeds,
            "negative_prompt_embeds": state.negative_prompt_embeds,
            "init_latents": state.init_latents,
            "cond_mask": align_replay_tensor(state.cond_mask, state.latents.shape[0]),
            "uncond_mask": align_replay_tensor(
                state.uncond_mask,
                state.latents.shape[0],
            ),
            "padding_mask": align_replay_tensor(
                state.padding_mask,
                state.latents.shape[0],
            ),
            "cond_indicator": align_replay_tensor(
                state.cond_indicator,
                state.latents.shape[0],
            ),
            "uncond_indicator": align_replay_tensor(
                state.uncond_indicator,
                state.latents.shape[0],
            ),
        }

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> CosmosPredict2SamplingState:
        """Rebuild SamplingState for the eval forward path from a batch slice.

        IMPORTANT: Cosmos's ``forward_step`` indexes BOTH
        ``state.timesteps[step_idx]`` AND ``state.scheduler.sigmas[step_idx]``
        for sigma scaling, so the eval path passes through the actual
        ``step_idx`` (NOT 0 like sd3/wan). We hand back the model's
        own scheduler + its full timesteps array so that indexing stays
        consistent with the rollout-time scheduler state.
        """
        # ``step_idx`` is consumed by the caller's ``forward_step`` call,
        # not here — but we accept it in the signature for parity with
        # the base contract and to allow defensive checks if needed.
        del step_idx
        return CosmosPredict2SamplingState(
            latents=latents,
            timesteps=self.scheduler.timesteps,
            scheduler=self.scheduler,
            prompt_embeds=replay_tensors["prompt_embeds"],
            negative_prompt_embeds=replay_tensors.get("negative_prompt_embeds"),
            guidance_scale=batch_context["guidance_scale"],
            do_cfg=batch_context["cfg"] and batch_context["guidance_scale"] > 1.0,
            init_latents=replay_tensors["init_latents"],
            cond_mask=replay_tensor(replay_tensors, batch_context, "cond_mask"),
            uncond_mask=replay_tensor(replay_tensors, batch_context, "uncond_mask"),
            padding_mask=shared_replay_tensor(
                replay_tensors,
                batch_context,
                "padding_mask",
            ),
            cond_indicator=replay_tensor(replay_tensors, batch_context, "cond_indicator"),
            uncond_indicator=replay_tensor(
                replay_tensors,
                batch_context,
                "uncond_indicator",
            ),
            fps=batch_context["fps"],
            seed=0,
            sigma_conditioning=batch_context.get("sigma_conditioning", 0.0001),
        )

    # -- decode_latents ------------------------------------------------

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to video tensor [B, C, T, H, W] in [0, 1].

        Cosmos VAE expects latents un-normalized via ``sigma_data`` and
        the per-channel ``latents_mean`` / ``latents_std`` stats stored on
        ``vae.config``. We then pass through ``video_processor.postprocess_video``
        and permute to the [B, C, T, H, W] convention used elsewhere.
        """
        pipe = self.pipeline
        sigma_data = pipe.scheduler.config.sigma_data

        def _transform(chunk: torch.Tensor) -> torch.Tensor:
            x = chunk.to(pipe.vae.dtype)
            latents_mean = (
                torch.tensor(pipe.vae.config.latents_mean)
                .view(1, pipe.vae.config.z_dim, 1, 1, 1)
                .to(x.device, x.dtype)
            )
            latents_std = (
                torch.tensor(pipe.vae.config.latents_std)
                .view(1, pipe.vae.config.z_dim, 1, 1, 1)
                .to(x.device, x.dtype)
            )
            return x * latents_std / sigma_data + latents_mean

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


class CosmosPredict2ReplayModel(ReplayRolloutStubs, CosmosPredict2Model):
    """Replay-only Cosmos Predict2 model without pipeline-only modules."""

    def __init__(self, *, transformer: Any, scheduler: Any, device: Any = None) -> None:
        DiffusionModelBase.__init__(self)
        self.transformer = transformer
        self._scheduler = scheduler
        self._device = device

    @property
    def pipeline(self) -> Any:
        raise RuntimeError("CosmosPredict2ReplayModel does not own a diffusers pipeline")

    def _set_transformer(self, transformer: Any) -> None:
        self.transformer = transformer

    @property
    def scheduler(self) -> Any:
        return self._scheduler

    @property
    def backend_handle(self) -> Any:
        return None

    # restore_eval_state is inherited from CosmosPredict2Model: it reads
    # ``self.scheduler``, which this replay model overrides to return its own
    # ``self._scheduler`` (the parent's property resolves to pipeline.scheduler).


__all__ = [
    "CosmosPredict2Model",
    "CosmosPredict2ReplayModel",
    "CosmosPredict2SamplingState",
]
