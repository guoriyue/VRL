"""HunyuanImage-2.1 t2i diffusers-backed model.

Diffusion implementation for Tencent HunyuanImage-2.1 (17B dual/single-stream
MMDiT). The generation helper flow mirrors every diffusion family:

    encode_prompt -> prepare_sampling -> forward_step xN -> decode_latents

HunyuanImage specifics vs Qwen-Image (the reference t2i CFG family):
- TWO text streams: Qwen2.5-VL-7B produces the main sequence embeds + mask,
  a byT5 glyph encoder produces a second embed/mask pair (zero-filled by the
  pipeline when the prompt carries no quoted glyph text). The transformer
  merges both streams internally, so all four tensors ride each branch call.
- Both encoders are parked on CPU by ``from_build``: the 16.6 GB Qwen2.5-VL
  follows the Qwen-Image offload discipline, and the pipeline's
  ``encode_prompt`` runs BOTH encoders on a single ``device`` argument (see
  ``HunyuanImagePipeline.encode_prompt``), so the 0.9 GB byT5 shares the CPU
  encode device rather than splitting the one-device call.
- TRUE classifier-free guidance with separate branches: cond/uncond masks
  differ per branch (and the glyph stream may be zeros on one side only), so
  ``separate_cfg`` (two independent forwards), ``cfg_base = "uncond"``.
- Latents are plain 4D ``[B, 64, H/32, W/32]`` (32x spatial VAE, no packing);
  the transformer computes RoPE internally.
- The reference pipeline passes an EXPLICIT sigma ladder
  ``np.linspace(1.0, 0.0, steps+1)[:-1]`` into set_timesteps (FlowMatchEuler,
  shift=5.0 applied by the scheduler); ``prepare_sampling`` mirrors that.
- ``timestep`` is the raw scheduler t expanded to batch (no /1000).
- VAE decode is a plain ``latents / scaling_factor`` (no shift/mean/std),
  image output ``[B, C, H, W]``.

GUIDER DIVERGENCE (deliberate): the reference pipeline combines branches
through ``AdaptiveProjectedMixGuidance`` — classic CFG for the first
``adaptive_projected_guidance_start_step`` steps (default 5), then Adaptive
Projected Guidance (momentum-buffered, norm-thresholded, orthogonal-projected
update — see diffusers ``guiders/adaptive_projected_guidance_mix.py``). APG is
stateful across steps (momentum buffer) and non-linear, which breaks the
per-step stateless replay contract RL training needs. This wrapper therefore
implements the CLASSIC CFG combine ``uncond + s * (cond - uncond)`` on every
step; expect small sample-level divergence from ``HunyuanImagePipeline``
outputs after step 5 at the same seed.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from typing import Any

import torch

from vrl.generation.types import DenoiseRequest
from vrl.models.steps.denoise import (
    DiffusersPipelineModelBase,
    DiffusersReplayModelBase,
    GuidedDiffusionSamplingStateBase,
)
from vrl.models.steps.denoise.common import (
    DiffusionBackboneCaller,
    DiffusionBackboneInput,
    DiffusionBackboneRunnerBase,
    DiffusionBranch,
    VaeDecodeMixin,
    expand_batch_timestep,
    pack_eval_timestep,
)
from vrl.models.steps.denoise.common.lora import LoraModelMixin


@dataclass
class HunyuanImageSamplingState(GuidedDiffusionSamplingStateBase):
    """Private HunyuanImage sampling state. Engine MUST NOT introspect."""

    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor
    prompt_embeds_2: torch.Tensor
    prompt_embeds_mask_2: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None
    negative_prompt_embeds_mask: torch.Tensor | None
    negative_prompt_embeds_2: torch.Tensor | None
    negative_prompt_embeds_mask_2: torch.Tensor | None
    do_cfg: bool
    guidance_embeds: bool


class HunyuanImageModel(
    VaeDecodeMixin,
    LoraModelMixin,
    DiffusersPipelineModelBase,
    DiffusionBackboneRunnerBase,
):
    """Diffusers-backed HunyuanImage-2.1 t2i model (true CFG, dual text streams).

    Implements the backbone-runner protocol itself. Cond/uncond prompts carry
    branch-specific masks and glyph streams (the byT5 pair may be zeros on one
    branch only), so the two branches run as independent forwards
    (``separate_cfg``) combined with the classic CFG formula — see the module
    docstring for the deliberate divergence from the pipeline's APG guider.
    """

    cfg_mode = "separate_cfg"
    cfg_base = "uncond"

    # -- backend ownership (called by runtime, not by collectors) -------
    _pipeline_classname = "HunyuanImagePipeline"
    _frozen_encoder_names = ("text_encoder", "text_encoder_2")
    # The Qwen2.5-VL-7B encoder is ~16.6 GB bf16; with the ~35 GB 17B
    # transformer it cannot co-reside on a 32 GB card, so it is parked on
    # CPU (Qwen-Image discipline). The 0.9 GB byT5 glyph encoder joins it
    # because HunyuanImagePipeline.encode_prompt runs BOTH encoders on one
    # ``device`` argument — splitting devices would need reimplementing the
    # pipeline's private glyph extraction/zero-fill path. byT5 only runs
    # for quoted glyph text, so the CPU forward is a negligible one-shot.
    _prompt_encoder_on_cpu = True

    def build_branch(
        self,
        request: DiffusionBackboneInput,
        branch: str,
    ) -> DiffusionBranch:
        """Map HunyuanImage transformer kwargs into the shared backbone contract."""
        if branch == "cond":
            embeds = request.prompt_embeds
            mask = request.extra["encoder_attention_mask"]
            embeds_2 = request.extra["encoder_hidden_states_2"]
            mask_2 = request.extra["encoder_attention_mask_2"]
        else:
            embeds = request.negative_prompt_embeds
            mask = request.extra["negative_encoder_attention_mask"]
            embeds_2 = request.extra["negative_encoder_hidden_states_2"]
            mask_2 = request.extra["negative_encoder_attention_mask_2"]
        return DiffusionBranch(
            hidden_states=request.hidden_states,
            timestep=request.timestep,
            encoder_hidden_states=embeds,
            extra_kwargs={
                "encoder_attention_mask": mask,
                "encoder_hidden_states_2": embeds_2,
                "encoder_attention_mask_2": mask_2,
                "guidance": request.extra["guidance"],
            },
        )

    # -- encode_prompt -------------------------------------------------

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode prompt via Qwen2.5-VL (embeds + mask) and byT5 (glyph pair).

        The pipeline's ``encode_prompt`` has NO ``max_sequence_length`` knob —
        sequence lengths are fixed by ``tokenizer_max_length`` (1000) and
        ``tokenizer_2_max_length`` (128) on the pipeline — so the request-side
        ``max_sequence_length`` is ignored here. Returns the conditional
        quadruple and, when CFG is active and a negative prompt is supplied,
        the unconditional quadruple for the separate branch.
        """
        guidance_scale = kwargs.get("guidance_scale", 3.5)
        do_cfg = guidance_scale > 1.0 and negative_prompt is not None
        pipe = self.pipeline
        # Both frozen encoders live on CPU (see from_build); run encode there,
        # then move embeds onto the transformer device for the denoise forward.
        enc_device = self._encoder_device()
        td = self.transformer.dtype

        def _encode(text: str | list[str]) -> tuple[torch.Tensor, ...]:
            batch_size = len(text) if isinstance(text, list) else 1
            return pipe.encode_prompt(
                prompt=text,
                device=enc_device,
                batch_size=batch_size,
                num_images_per_prompt=1,
            )

        prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2 = _encode(prompt)
        result: dict[str, Any] = {
            "prompt_embeds": prompt_embeds.to(self.device, dtype=td),
            "prompt_embeds_mask": prompt_embeds_mask.to(self.device),
            "prompt_embeds_2": prompt_embeds_2.to(self.device, dtype=td),
            "prompt_embeds_mask_2": prompt_embeds_mask_2.to(self.device),
        }
        if do_cfg:
            (
                negative_prompt_embeds,
                negative_prompt_embeds_mask,
                negative_prompt_embeds_2,
                negative_prompt_embeds_mask_2,
            ) = _encode(negative_prompt)
            result["negative_prompt_embeds"] = negative_prompt_embeds.to(
                self.device,
                dtype=td,
            )
            result["negative_prompt_embeds_mask"] = negative_prompt_embeds_mask.to(
                self.device,
            )
            result["negative_prompt_embeds_2"] = negative_prompt_embeds_2.to(
                self.device,
                dtype=td,
            )
            result["negative_prompt_embeds_mask_2"] = negative_prompt_embeds_mask_2.to(
                self.device,
            )
        return result

    # -- prepare_sampling ----------------------------------------------

    def prepare_sampling(
        self,
        request: DenoiseRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> HunyuanImageSamplingState:
        """Build the per-request 4D-latent SamplingState for a denoise loop."""
        del kwargs
        import numpy as np

        pipe = self.pipeline
        device = self.device

        prompt_embeds = encoded["prompt_embeds"]
        negative_prompt_embeds = encoded.get("negative_prompt_embeds")

        # Mirror the reference pipeline's explicit sigma ladder (the scheduler
        # applies shift=5.0 on top).
        sigmas = np.linspace(1.0, 0.0, request.num_steps + 1)[:-1]
        pipe.scheduler.set_timesteps(sigmas=sigmas, device=device)
        timesteps = pipe.scheduler.timesteps

        seed = request.seed if request.seed is not None else random.randint(0, sys.maxsize)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        num_channels_latents = pipe.transformer.config.in_channels
        batch_size = prompt_embeds.shape[0]
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

        return HunyuanImageSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=pipe.scheduler,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=encoded["prompt_embeds_mask"],
            prompt_embeds_2=encoded["prompt_embeds_2"],
            prompt_embeds_mask_2=encoded["prompt_embeds_mask_2"],
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=encoded.get("negative_prompt_embeds_mask"),
            negative_prompt_embeds_2=encoded.get("negative_prompt_embeds_2"),
            negative_prompt_embeds_mask_2=encoded.get("negative_prompt_embeds_mask_2"),
            guidance_scale=request.guidance_scale,
            do_cfg=negative_prompt_embeds is not None,
            guidance_embeds=self._guidance_embeds,
        )

    # -- forward_step --------------------------------------------------

    def forward_step(
        self,
        state: HunyuanImageSamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        """HunyuanImage transformer forward (true-CFG separate branches when do_cfg)."""
        t = state.timesteps[step_idx]
        bsz = state.latents.shape[0]
        td = self._transformer_dtype()

        latent_input = state.latents.to(td)
        # Raw scheduler t expanded to batch (pipeline parity: no /1000).
        timestep_batch = expand_batch_timestep(t, bsz).to(
            device=latent_input.device,
            dtype=td,
        )
        guidance = None
        if state.guidance_embeds:
            # Distilled variants embed the scale x1000 (pipeline parity); the
            # 2.1 community base checkpoint has guidance_embeds=False -> None.
            guidance = torch.full(
                (bsz,),
                float(state.guidance_scale) * 1000.0,
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
                    "encoder_attention_mask": state.prompt_embeds_mask,
                    "encoder_hidden_states_2": state.prompt_embeds_2.to(td),
                    "encoder_attention_mask_2": state.prompt_embeds_mask_2,
                    "negative_encoder_attention_mask": state.negative_prompt_embeds_mask,
                    "negative_encoder_hidden_states_2": (
                        None
                        if state.negative_prompt_embeds_2 is None
                        else state.negative_prompt_embeds_2.to(td)
                    ),
                    "negative_encoder_attention_mask_2": state.negative_prompt_embeds_mask_2,
                    "guidance": guidance,
                },
            ),
        )
        return output.as_dict()

    # -- collector boundary --------------------------------------------

    def export_batch_context(self, state: HunyuanImageSamplingState) -> dict[str, Any]:
        """Project HunyuanImage sampling state into trajectory context."""
        return {
            "guidance_scale": state.guidance_scale,
            "cfg": state.do_cfg,
            "guidance_embeds": state.guidance_embeds,
        }

    def export_replay_tensors(self, state: HunyuanImageSamplingState) -> dict[str, Any]:
        """Project HunyuanImage sampling state into per-sample trajectory tensors.

        The conditional quadruple is always present (the transformer requires
        both masks); negative tensors are stored only when CFG is active, so
        ``restore`` reads them with ``.get`` and the no-CFG path stays
        tensor-free.
        """
        tensors: dict[str, Any] = {
            "prompt_embeds": state.prompt_embeds,
            "prompt_embeds_mask": state.prompt_embeds_mask,
            "prompt_embeds_2": state.prompt_embeds_2,
            "prompt_embeds_mask_2": state.prompt_embeds_mask_2,
        }
        if state.negative_prompt_embeds is not None:
            tensors["negative_prompt_embeds"] = state.negative_prompt_embeds
        if state.negative_prompt_embeds_mask is not None:
            tensors["negative_prompt_embeds_mask"] = state.negative_prompt_embeds_mask
        if state.negative_prompt_embeds_2 is not None:
            tensors["negative_prompt_embeds_2"] = state.negative_prompt_embeds_2
        if state.negative_prompt_embeds_mask_2 is not None:
            tensors["negative_prompt_embeds_mask_2"] = state.negative_prompt_embeds_mask_2
        return tensors

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> HunyuanImageSamplingState:
        """Rebuild HunyuanImageSamplingState from a batch slice for the eval path."""
        ts = replay_tensors["timesteps"]
        timesteps = pack_eval_timestep(ts, step_idx)
        return HunyuanImageSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=None,  # not needed for forward_step (no scheduler.step here)
            prompt_embeds=replay_tensors["prompt_embeds"],
            prompt_embeds_mask=replay_tensors["prompt_embeds_mask"],
            prompt_embeds_2=replay_tensors["prompt_embeds_2"],
            prompt_embeds_mask_2=replay_tensors["prompt_embeds_mask_2"],
            negative_prompt_embeds=replay_tensors.get("negative_prompt_embeds"),
            negative_prompt_embeds_mask=replay_tensors.get("negative_prompt_embeds_mask"),
            negative_prompt_embeds_2=replay_tensors.get("negative_prompt_embeds_2"),
            negative_prompt_embeds_mask_2=replay_tensors.get(
                "negative_prompt_embeds_mask_2",
            ),
            guidance_scale=batch_context["guidance_scale"],
            do_cfg=batch_context["cfg"]
            and replay_tensors.get("negative_prompt_embeds") is not None,
            guidance_embeds=batch_context["guidance_embeds"],
        )


class HunyuanImageReplayModel(DiffusersReplayModelBase, HunyuanImageModel):
    """Replay-only HunyuanImage model owning no prompt encoders, VAE, or pipeline."""


__all__ = [
    "HunyuanImageModel",
    "HunyuanImageReplayModel",
    "HunyuanImageSamplingState",
]
