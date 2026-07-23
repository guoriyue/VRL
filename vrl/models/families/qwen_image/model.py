"""Qwen-Image t2i diffusers-backed model.

Diffusion implementation for Alibaba Qwen-Image generation. The generation
helper flow mirrors every diffusion family:

    encode_prompt -> prepare_sampling -> forward_step xN -> decode_latents

Qwen-Image specifics vs SD3 (the reference family):
- Latents are PACKED ``[B, (h/2)*(w/2), C*4]`` like FLUX (stay packed through the
  denoise loop; only ``decode_latents`` unpacks). The unpacked latent carries a
  singleton temporal dim ``[B, C, 1, H, W]`` because the VAE is a video VAE.
- The text encoder is Qwen2.5-VL: ``encode_prompt`` returns a SEQUENCE embed plus
  an attention mask and NO pooled vector (unlike SD3's pooled CLIP).
- TRUE classifier-free guidance with a real unconditional branch; the runner runs
  ``separate_cfg`` (see runner.py for why batched packing is impossible).
- The transformer needs ``img_shapes`` (RoPE grid) and ``encoder_hidden_states_mask``;
  both are rebuilt deterministically on the replay path.
- The transformer time embedding multiplies its input by 1000 internally, so
  ``forward_step`` feeds ``t / 1000`` (matching diffusers' QwenImagePipeline).
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from typing import Any

import torch

from vrl.generation.types import VideoGenerationRequest
from vrl.models.interfaces.runtime import ModelBuild
from vrl.models.steps.denoise import (
    DiffusersPipelineModelBase,
    DiffusersReplayModelBase,
    GuidedDiffusionSamplingStateBase,
    diffusers_pipeline_dtypes,
)
from vrl.models.steps.denoise.common import (
    ChunkedLatentDecoder,
    DiffusionBackboneCaller,
    DiffusionBackboneInput,
    DiffusionBackboneRunnerBase,
    DiffusionBranch,
    LatentDecodePlan,
    expand_batch_timestep,
    pack_eval_timestep,
)
from vrl.models.steps.denoise.common.lora import LoraModelMixin
from vrl.models.steps.denoise.common.tensors import require_tensor


@dataclass
class QwenImageSamplingState(GuidedDiffusionSamplingStateBase):
    """Private Qwen-Image sampling state. Engine MUST NOT introspect."""

    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor | None
    negative_prompt_embeds: torch.Tensor | None
    negative_prompt_embeds_mask: torch.Tensor | None
    do_cfg: bool
    guidance_embeds: bool
    height: int
    width: int
    vae_scale_factor: int


class QwenImageModel(LoraModelMixin, DiffusersPipelineModelBase, DiffusionBackboneRunnerBase):
    """Diffusers-backed Qwen-Image t2i model.

    Implements the backbone-runner protocol itself. Qwen-Image does TRUE
    classifier-free guidance where cond/uncond prompts tokenize to DIFFERENT
    sequence lengths (their masks differ), so the two branches cannot be packed
    into one batched call — hence ``separate_cfg`` (two independent forwards)
    instead of sd3_5's ``batched_cfg``. ``finalize_noise_pred`` reproduces
    Qwen-Image's norm-preserving CFG combine.
    """

    cfg_mode = "separate_cfg"
    cfg_base = "uncond"

    def build_branch(
        self,
        request: DiffusionBackboneInput,
        branch: str,
    ) -> DiffusionBranch:
        """Map Qwen-Image transformer kwargs into the shared backbone contract."""
        if branch == "cond":
            embeds = request.prompt_embeds
            mask = request.extra.get("encoder_hidden_states_mask")
        else:
            embeds = require_tensor(
                request.negative_prompt_embeds,
                "negative_prompt_embeds",
            )
            mask = request.extra.get("negative_encoder_hidden_states_mask")
        extra_kwargs = {
            "encoder_hidden_states_mask": mask,
            "img_shapes": request.extra["img_shapes"],
        }
        if request.extra.get("guidance") is not None:
            extra_kwargs["guidance"] = request.extra["guidance"]
        return DiffusionBranch(
            hidden_states=request.hidden_states,
            timestep=request.timestep,
            encoder_hidden_states=embeds,
            extra_kwargs=extra_kwargs,
        )

    def finalize_noise_pred(
        self,
        request: DiffusionBackboneInput,
        combined: torch.Tensor,
        cond: torch.Tensor,
        uncond: torch.Tensor,
    ) -> torch.Tensor:
        """Qwen-Image norm-preserving CFG: rescale the combined pred to the cond norm.

        Mirrors diffusers QwenImagePipeline:
            comb = uncond + s * (cond - uncond)   # done by combine_cfg upstream
            noise = comb * (||cond|| / ||comb||)
        """
        del uncond
        if not request.do_cfg:
            return combined
        cond_norm = torch.norm(cond, dim=-1, keepdim=True)
        comb_norm = torch.norm(combined, dim=-1, keepdim=True)
        return combined * (cond_norm / comb_norm)

    def __init__(
        self,
        *,
        pipeline: Any,
        device: Any = None,
    ) -> None:
        super().__init__(pipeline=pipeline, device=device)
        # decode_latents only receives the packed latent tensor; prepare_sampling
        # records the spatial shape it must unpack to (single model instance runs
        # prepare -> denoise -> decode sequentially per chunk).
        self._decode_height = 1024
        self._decode_width = 1024

    # -- backend ownership (called by runtime, not by collectors) -------

    @classmethod
    def from_build(cls, build: ModelBuild) -> QwenImageModel:
        """Load the diffusers Qwen-Image pipeline + freeze non-trainable modules."""
        from diffusers import QwenImagePipeline

        model_dtype = build.parameter_dtype
        prompt_encoder_dtype, load_kwargs = diffusers_pipeline_dtypes(build, model_dtype)
        pipeline = QwenImagePipeline.from_pretrained(
            build.model_name_or_path,
            **load_kwargs,
        )
        pipeline.vae.requires_grad_(False)
        # Qwen-Image's Qwen2.5-VL text encoder is ~15 GB (bf16); with the ~20 GB+
        # transformer it cannot co-reside on a 32 GB card. The encoder feeds only the
        # one-shot encode_prompt, so park it on CPU (enable_model_cpu_offload
        # discipline); encode_prompt runs it there and moves embeds to the GPU.
        text_encoder = getattr(pipeline, "text_encoder", None)
        if text_encoder is not None:
            text_encoder.requires_grad_(False)
            text_encoder.to("cpu", dtype=prompt_encoder_dtype)
        pipeline.vae.to(build.device, dtype=torch.float32)
        return cls(
            pipeline=pipeline,
            device=build.device,
        )

    def _set_dynamic_timesteps(self, num_steps: int, image_seq_len: int, device: Any) -> Any:
        """Set Qwen-Image timesteps with the resolution-derived ``mu`` (diffusers parity)."""
        from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift

        scheduler = self.pipeline.scheduler
        config = scheduler.config
        if getattr(config, "use_dynamic_shifting", False):
            mu = calculate_shift(
                image_seq_len,
                config.get("base_image_seq_len", 256),
                config.get("max_image_seq_len", 4096),
                config.get("base_shift", 0.5),
                config.get("max_shift", 1.15),
            )
            scheduler.set_timesteps(num_steps, device=device, mu=mu)
        else:
            scheduler.set_timesteps(num_steps, device=device)
        return scheduler.timesteps

    # -- encode_prompt -------------------------------------------------

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode prompt via Qwen2.5-VL (sequence embeds + mask, no pooled).

        Returns the conditional embeds/mask and, when CFG is active and a negative
        prompt is supplied, the unconditional embeds/mask for the separate branch.
        """
        max_seq = kwargs.get("max_sequence_length", 1024)
        guidance_scale = kwargs.get("guidance_scale", 4.0)
        do_cfg = guidance_scale > 1.0 and negative_prompt is not None
        pipe = self.pipeline
        # The frozen Qwen2.5-VL encoder lives on CPU (see from_build); run encode
        # there, then move embeds onto the transformer device for the denoise forward.
        enc_device = self._encoder_device()
        td = self.transformer.dtype

        prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
            prompt=prompt,
            num_images_per_prompt=1,
            max_sequence_length=max_seq,
            device=enc_device,
        )
        result: dict[str, Any] = {
            "prompt_embeds": prompt_embeds.to(self.device, dtype=td),
            "prompt_embeds_mask": (
                None if prompt_embeds_mask is None else prompt_embeds_mask.to(self.device)
            ),
        }
        if do_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = pipe.encode_prompt(
                prompt=negative_prompt,
                num_images_per_prompt=1,
                max_sequence_length=max_seq,
                device=enc_device,
            )
            result["negative_prompt_embeds"] = negative_prompt_embeds.to(
                self.device,
                dtype=td,
            )
            result["negative_prompt_embeds_mask"] = (
                None
                if negative_prompt_embeds_mask is None
                else negative_prompt_embeds_mask.to(self.device)
            )
        return result

    # -- prepare_sampling ----------------------------------------------

    def prepare_sampling(
        self,
        request: VideoGenerationRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> QwenImageSamplingState:
        """Build the per-request packed-latent SamplingState for a denoise loop."""
        del kwargs
        pipe = self.pipeline
        device = self.device

        prompt_embeds = encoded["prompt_embeds"]
        prompt_embeds_mask = encoded.get("prompt_embeds_mask")
        negative_prompt_embeds = encoded.get("negative_prompt_embeds")
        negative_prompt_embeds_mask = encoded.get("negative_prompt_embeds_mask")

        seed = request.seed if request.seed is not None else random.randint(0, sys.maxsize)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        batch_size = prompt_embeds.shape[0]
        num_channels_latents = pipe.transformer.config.in_channels // 4
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

        # Dynamic-shifting timesteps depend on the packed image sequence length.
        timesteps = self._set_dynamic_timesteps(
            request.num_steps,
            latents.shape[1],
            device,
        )

        self._decode_height = int(request.height)
        self._decode_width = int(request.width)

        return QwenImageSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=pipe.scheduler,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            guidance_scale=request.guidance_scale,
            do_cfg=negative_prompt_embeds is not None,
            guidance_embeds=self._guidance_embeds,
            height=int(request.height),
            width=int(request.width),
            vae_scale_factor=int(pipe.vae_scale_factor),
        )

    # -- forward_step --------------------------------------------------

    def _img_shapes(self, state: QwenImageSamplingState, bsz: int) -> list:
        vsf = state.vae_scale_factor
        latent_h = state.height // vsf // 2
        latent_w = state.width // vsf // 2
        return [[(1, latent_h, latent_w)]] * bsz

    def forward_step(
        self,
        state: QwenImageSamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        """Qwen-Image transformer forward (true-CFG separate branches when do_cfg)."""
        t = state.timesteps[step_idx]
        bsz = state.latents.shape[0]
        td = self._transformer_dtype()

        latent_input = state.latents.to(td)
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
                extra={
                    "encoder_hidden_states_mask": state.prompt_embeds_mask,
                    "negative_encoder_hidden_states_mask": state.negative_prompt_embeds_mask,
                    "img_shapes": self._img_shapes(state, bsz),
                    "guidance": guidance,
                },
            ),
        )
        return output.as_dict()

    # -- collector boundary --------------------------------------------

    def export_batch_context(self, state: QwenImageSamplingState) -> dict[str, Any]:
        """Project Qwen-Image sampling state into trajectory context."""
        return {
            "guidance_scale": state.guidance_scale,
            "cfg": state.do_cfg,
            "guidance_embeds": state.guidance_embeds,
            "height": state.height,
            "width": state.width,
            "vae_scale_factor": state.vae_scale_factor,
        }

    def export_replay_tensors(self, state: QwenImageSamplingState) -> dict[str, Any]:
        """Project Qwen-Image sampling state into per-sample trajectory tensors.

        Masks / negative embeds are only stored when present, so ``restore`` reads
        them with ``.get`` and the no-CFG / all-ones-mask paths stay tensor-free.
        """
        tensors: dict[str, Any] = {"prompt_embeds": state.prompt_embeds}
        if state.prompt_embeds_mask is not None:
            tensors["prompt_embeds_mask"] = state.prompt_embeds_mask
        if state.negative_prompt_embeds is not None:
            tensors["negative_prompt_embeds"] = state.negative_prompt_embeds
        if state.negative_prompt_embeds_mask is not None:
            tensors["negative_prompt_embeds_mask"] = state.negative_prompt_embeds_mask
        return tensors

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> QwenImageSamplingState:
        """Rebuild QwenImageSamplingState from a batch slice for the eval forward path."""
        ts = replay_tensors["timesteps"]
        timesteps = pack_eval_timestep(ts, step_idx)
        return QwenImageSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=None,  # not needed for forward_step (no scheduler.step here)
            prompt_embeds=replay_tensors["prompt_embeds"],
            prompt_embeds_mask=replay_tensors.get("prompt_embeds_mask"),
            negative_prompt_embeds=replay_tensors.get("negative_prompt_embeds"),
            negative_prompt_embeds_mask=replay_tensors.get("negative_prompt_embeds_mask"),
            guidance_scale=batch_context["guidance_scale"],
            do_cfg=batch_context["cfg"]
            and replay_tensors.get("negative_prompt_embeds") is not None,
            guidance_embeds=batch_context["guidance_embeds"],
            height=batch_context["height"],
            width=batch_context["width"],
            vae_scale_factor=int(batch_context["vae_scale_factor"]),
        )

    # -- decode_latents ------------------------------------------------

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode packed latents → image via Qwen-Image video VAE.

        Unpack to ``[B, C, 1, H, W]``, denormalize with the VAE's per-channel
        mean/std, decode, then drop the singleton temporal frame.
        """
        pipe = self.pipeline
        vae = pipe.vae
        vae_scale_factor = pipe.vae_scale_factor
        height = self._decode_height
        width = self._decode_width
        z_dim = vae.config.z_dim
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1)
        latents_std = torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1)

        def _transform(chunk: torch.Tensor) -> torch.Tensor:
            unpacked = pipe._unpack_latents(chunk, height, width, vae_scale_factor)
            unpacked = unpacked.to(vae.dtype)
            mean = latents_mean.to(unpacked.device, unpacked.dtype)
            std = latents_std.to(unpacked.device, unpacked.dtype)
            return unpacked * std + mean

        decoder = ChunkedLatentDecoder(
            LatentDecodePlan(
                prepare_latents=_transform,
                vae_decode=lambda chunk: vae.decode(chunk, return_dict=False)[0],
                # Video VAE returns [B, C, 1, H, W]; drop the temporal frame.
                prepare_decoded=lambda decoded: decoded[:, :, 0],
                postprocess=lambda image: pipe.image_processor.postprocess(
                    image,
                    output_type="pt",
                ),
                output_layout="image_bchw",
                decode_batch_size=getattr(pipe, "decode_batch_size", None),
            ),
        )
        return decoder(latents)


class QwenImageReplayModel(DiffusersReplayModelBase, QwenImageModel):
    """Replay-only Qwen-Image model that owns no prompt encoders, VAE, or pipeline."""


__all__ = ["QwenImageModel", "QwenImageReplayModel", "QwenImageSamplingState"]
