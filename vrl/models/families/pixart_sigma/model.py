"""PixArt-Sigma t2i diffusers-backed model.

Diffusion implementation for PixArt-Sigma (DiT + SDXL KL-VAE + T5-XXL).
The generation helper flow mirrors every diffusion family:

    encode_prompt -> prepare_sampling -> forward_step xN -> decode_latents

PixArt-Sigma specifics vs SANA (the reference single-encoder t2i family):
- NOT flow-matching: an epsilon-prediction DDPM ladder. Rollout/replay
  log-probs run through ``sde_type="ddim"`` (``vrl/math/denoise/ddim.py``)
  — experiments MUST set ``rollout.sde.type: ddim``.
- The checkpoint ships a ``DPMSolverMultistepScheduler`` — a multi-step ODE
  solver with NO per-step Gaussian, unusable for RL log-probs. The family
  therefore constructs a ``DDIMScheduler`` from the shipped beta config in
  :func:`pixart_ddim_scheduler` (one construction site: rollout
  ``prepare_sampling`` and the replay model's ``prepare_replay`` both use it),
  with ``clip_sample=False`` so the DDIM eta-SDE transition stays Gaussian.
- LEARNED SIGMA: the transformer's ``out_channels`` is twice the latent
  channels (8 vs 4); the first half is the epsilon prediction and the second
  half the (unused) variance. ``postprocess_branch`` batches per branch BEFORE
  the CFG combine — the pipeline batches after, which is equivalent since
  channel-batch and the linear CFG combine commute.
- Single T5-XXL encoder (~9.5 GB bf16, parked on CPU) returning sequence
  embeds + attention masks; true batched CFG with uncond rows first. The mask
  threads through the transformer as ``encoder_attention_mask``.
- The transformer takes the raw integer timestep (no scaling/division) and
  requires ``added_cond_kwargs`` unconditionally when ``sample_size == 128``
  (512/1024-MS checkpoints set ``use_additional_conditions=False`` but the
  forward still wants the dict); we pass ``{"resolution": None,
  "aspect_ratio": None}`` on every call, exactly like the pipeline.
- ``encode_prompt`` passes ``clean_caption=False`` (True pulls in bs4/ftfy).
"""

from __future__ import annotations

import random
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from vrl.generation.types import VideoGenerationRequest
from vrl.models.interfaces.runtime import ModelBuild
from vrl.models.steps.denoise import (
    DiffusersPipelineModelBase,
    DiffusersReplayModelBase,
)
from vrl.models.steps.denoise.common import (
    DiffusionBackboneCaller,
    DiffusionBackboneInput,
    DiffusionBranch,
    EncoderAttentionMaskRunnerBase,
    MaskedPromptCollectorMixin,
    MaskedPromptSamplingState,
    VaeDecodeMixin,
    expand_batch_timestep,
)
from vrl.models.steps.denoise.common.lora import LoraModelMixin

# 512/1024-MS checkpoints train without micro-conditioning, but the
# transformer forward still requires the dict when config.sample_size == 128;
# the pipeline passes it unconditionally, so do we (both CFG branches share
# the same value, which the batched-CFG kwarg packer requires).
_ADDED_COND_KWARGS = {"resolution": None, "aspect_ratio": None}


def pixart_ddim_scheduler(scheduler_config: Any, num_steps: int, device: Any) -> Any:
    """Build the RL DDIM scheduler from PixArt-Sigma's shipped beta config.

    One construction site for both the rollout and replay paths: the
    checkpoint ships a DPM-Solver (multi-step ODE, no per-step Gaussian), so
    RL sampling/log-probs run a plain DDIM on the SAME trained beta ladder.
    ``clip_sample=False`` keeps the eta-SDE transition an unclipped Gaussian
    (``ddim_step_with_logprob`` fails loud otherwise); ``set_alpha_to_one``
    stays at its default (True) — the terminal step is the Dirac case the
    log-prob math already handles; ``timestep_spacing="leading"`` keeps
    integer timesteps on the trained ladder.
    """
    from diffusers import DDIMScheduler

    config = dict(scheduler_config)
    scheduler = DDIMScheduler(
        num_train_timesteps=int(config.get("num_train_timesteps", 1000)),
        beta_start=float(config.get("beta_start", 0.0001)),
        beta_end=float(config.get("beta_end", 0.02)),
        beta_schedule=str(config.get("beta_schedule", "linear")),
        trained_betas=config.get("trained_betas"),
        prediction_type="epsilon",
        clip_sample=False,
        timestep_spacing="leading",
    )
    scheduler.set_timesteps(int(num_steps), device=device)
    return scheduler


@dataclass
class PixArtSigmaSamplingState(MaskedPromptSamplingState):
    """Private PixArt-Sigma sampling state. Engine MUST NOT introspect."""


class PixArtSigmaModel(
    VaeDecodeMixin,
    MaskedPromptCollectorMixin,
    LoraModelMixin,
    DiffusersPipelineModelBase,
    EncoderAttentionMaskRunnerBase,
):
    """Diffusers-backed PixArt-Sigma t2i model (epsilon DDPM family).

    Implements the backbone-runner protocol itself. Both CFG branches pad to
    the same sequence length (T5 encode at ``max_sequence_length``), so they
    pack into one batched transformer call; the attention masks ride the
    batch as ``encoder_attention_mask``.
    """

    cfg_mode = "batched_cfg"
    cfg_base = "uncond"
    branch_extra_kwargs: ClassVar[Mapping[str, Any]] = {
        "added_cond_kwargs": _ADDED_COND_KWARGS,
    }
    sampling_state_cls = PixArtSigmaSamplingState

    # -- backend ownership (called by runtime, not by collectors) -------
    _pipeline_classname = "PixArtSigmaPipeline"
    _frozen_encoder_names = ("text_encoder",)
    # T5-XXL (~9.5 GB bf16); park on CPU (Qwen-Image discipline).
    _prompt_encoder_on_cpu = True

    def postprocess_branch(
        self,
        request: DiffusionBackboneInput,
        branch: DiffusionBranch,
        raw_output: torch.Tensor,
    ) -> torch.Tensor:
        """Batch the learned-sigma output down to the epsilon prediction.

        The transformer outputs ``2 * latent_channels`` (learned variance in
        the second half). Chunking per branch before the CFG combine is
        equivalent to the pipeline's batch-after-combine (both are linear).
        """
        del branch
        latent_channels = request.hidden_states.shape[1]
        if raw_output.shape[1] == 2 * latent_channels:
            return raw_output.chunk(2, dim=1)[0]
        return raw_output

    # -- encode_prompt -------------------------------------------------

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode prompt via T5-XXL (sequence embeds + attention mask, no pooled).

        ``clean_caption=False``: the True path imports bs4/ftfy, and RL
        datasets control their prompts anyway.
        """
        max_seq = kwargs.get("max_sequence_length", 300)
        guidance_scale = kwargs.get("guidance_scale", 4.5)
        do_cfg = guidance_scale > 1.0
        neg = negative_prompt if negative_prompt is not None else ""

        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=neg,
            num_images_per_prompt=1,
            device=self._encoder_device(),
            clean_caption=False,
            max_sequence_length=max_seq,
        )

        td = self.transformer.dtype
        result: dict[str, Any] = {
            "prompt_embeds": prompt_embeds.to(self.device, dtype=td),
            "prompt_attention_mask": (
                None if prompt_attention_mask is None else prompt_attention_mask.to(self.device)
            ),
        }
        if do_cfg and negative_prompt_embeds is not None:
            result["negative_prompt_embeds"] = negative_prompt_embeds.to(
                self.device,
                dtype=td,
            )
            result["negative_prompt_attention_mask"] = (
                None
                if negative_prompt_attention_mask is None
                else negative_prompt_attention_mask.to(self.device)
            )
        return result

    # -- prepare_sampling ----------------------------------------------

    def prepare_sampling(
        self,
        request: VideoGenerationRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> PixArtSigmaSamplingState:
        """Build the per-request SamplingState for a denoise loop.

        The shipped DPM-Solver is NOT used: the RL scheduler is a DDIM built
        from the same beta config (see :func:`pixart_ddim_scheduler`), so the
        ddim eta-SDE's clip_sample=False requirement holds by construction.
        """
        del kwargs
        pipe = self.pipeline
        device = self.device

        prompt_embeds = encoded["prompt_embeds"]
        prompt_attention_mask = encoded.get("prompt_attention_mask")
        negative_prompt_embeds = encoded.get("negative_prompt_embeds")
        negative_prompt_attention_mask = encoded.get("negative_prompt_attention_mask")

        scheduler = pixart_ddim_scheduler(
            pipe.scheduler.config,
            request.num_steps,
            device,
        )
        timesteps = scheduler.timesteps

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
        # No-op for DDIM (init_noise_sigma == 1.0); kept for pipeline parity.
        latents = latents * scheduler.init_noise_sigma

        do_cfg = request.guidance_scale > 1.0 and negative_prompt_embeds is not None

        return PixArtSigmaSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=scheduler,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            guidance_scale=request.guidance_scale,
            do_cfg=do_cfg,
        )

    # -- forward_step --------------------------------------------------

    def forward_step(
        self,
        state: PixArtSigmaSamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        """PixArt-Sigma transformer forward + optional batched CFG."""
        t = state.timesteps[step_idx]
        bsz = state.latents.shape[0]
        td = self._transformer_dtype()

        latent_input = state.latents.to(td)
        # Raw integer timestep — the sinusoidal proj consumes it directly.
        timestep_batch = expand_batch_timestep(t, bsz).to(
            device=latent_input.device,
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
                    "encoder_attention_mask": state.prompt_attention_mask,
                    "negative_encoder_attention_mask": state.negative_prompt_attention_mask,
                },
            ),
        )
        return output.as_dict()


class PixArtSigmaReplayModel(DiffusersReplayModelBase, PixArtSigmaModel):
    """Replay-only PixArt-Sigma model that owns no prompt encoder, VAE, or pipeline."""

    def prepare_replay(self, build: ModelBuild) -> None:
        """Standardize the replay scheduler onto the rollout's DDIM ladder.

        The generic replay loader builds the scheduler from the shipped config
        (DPM-Solver ODE — no per-step Gaussian), but the trajectory buffers and
        the ddim log-prob math live on the DDIM ladder the rollout used; the
        replay scheduler must carry the same timestep/alphas_cumprod table.
        """
        num_steps = build.num_steps
        # A scheduler without .config is a hand-injected test double — only
        # real diffusers schedulers carry the shipped beta config that needs
        # standardizing.
        config = getattr(self._scheduler, "config", None)
        if num_steps is not None and config is not None:
            self._scheduler = pixart_ddim_scheduler(
                config,
                int(num_steps),
                build.device,
            )


__all__ = [
    "PixArtSigmaModel",
    "PixArtSigmaReplayModel",
    "PixArtSigmaSamplingState",
    "pixart_ddim_scheduler",
]
