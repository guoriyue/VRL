"""PixArt-Sigma t2i diffusers-backed model.

Diffusion implementation for PixArt-Sigma (DiT + SDXL KL-VAE + T5-XXL).
The generation helper flow mirrors every diffusion family:

    encode_prompt -> prepare_sampling -> forward_step xN -> decode_latents

``PixArtSigmaPipelineModel`` runs that flow. ``PixArtSigmaModel`` — the
transformer, its scheduler and ``forward_step`` — is what replay runs; the
pipeline class adds the text encoder, latent preparation and VAE decode on top
of it.

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
  half the (unused) variance. ``forward_step`` keeps the first half after
  the CFG combine, like the pipeline.
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
from dataclasses import dataclass
from typing import Any

import torch

from vrl.generation.types import DenoiseRequest
from vrl.models.interfaces.runtime import ModelBuild
from vrl.models.steps.denoise import (
    DiffusersModelBase,
    DiffusersPipelineModelBase,
    GuidedDenoiseSamplingStateBase,
)
from vrl.models.steps.denoise.common import (
    VaeDecodeMixin,
    expand_batch_timestep,
    pack_eval_timestep,
    predict_with_cfg,
)

# 512/1024-MS checkpoints train without micro-conditioning, but the
# transformer forward still requires the dict when config.sample_size == 128;
# the pipeline passes it unconditionally, so do we (both CFG branches share
# the same value, which batched CFG requires of non-tensor kwargs).
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
class PixArtSigmaSamplingState(GuidedDenoiseSamplingStateBase):
    """Private PixArt-Sigma sampling state. Engine MUST NOT introspect.

    ``latents`` / ``timesteps`` / ``scheduler`` are the only fields the batch
    executor may touch.
    """

    prompt_embeds: torch.Tensor
    prompt_attention_mask: torch.Tensor | None
    negative_prompt_embeds: torch.Tensor | None
    negative_prompt_attention_mask: torch.Tensor | None
    do_cfg: bool


class PixArtSigmaModel(DiffusersModelBase):
    """PixArt-Sigma t2i transformer, scheduler and per-step math (epsilon DDPM): the model replay runs.

    Both CFG branches pad to the same sequence length (T5 encode at
    ``max_sequence_length``), so they pack into one batched transformer call;
    the attention masks ride the batch as ``encoder_attention_mask``.
    """

    def forward_step(
        self,
        state: PixArtSigmaSamplingState,
        step_idx: int,
    ) -> dict[str, Any]:
        """Transformer forward + batched CFG (raw integer timestep, no scaling)."""
        t = state.timesteps[step_idx]
        bsz = state.latents.shape[0]
        td = self._transformer_dtype()

        latent_input = state.latents.to(td)
        timestep_batch = expand_batch_timestep(t, bsz).to(device=latent_input.device)
        cond = {
            "hidden_states": latent_input,
            "timestep": timestep_batch,
            "encoder_hidden_states": state.prompt_embeds.to(td),
            "encoder_attention_mask": state.prompt_attention_mask,
            "added_cond_kwargs": _ADDED_COND_KWARGS,
        }
        uncond = None
        if state.do_cfg:
            uncond = {
                **cond,
                "encoder_hidden_states": state.negative_prompt_embeds.to(td),
                "encoder_attention_mask": state.negative_prompt_attention_mask,
            }
        noise_pred = predict_with_cfg(
            self.transformer,
            cond,
            uncond,
            guidance_scale=state.guidance_scale,
            batched=True,
            dtype=td,
        )
        # The transformer outputs ``2 * latent_channels`` (learned variance in
        # the second half); keep the epsilon half like the pipeline does.
        if noise_pred.shape[1] == 2 * latent_input.shape[1]:
            noise_pred = noise_pred.chunk(2, dim=1)[0]
        return {"noise_pred": noise_pred}

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> PixArtSigmaSamplingState:
        """Rebuild the sampling state from a batch slice for eval forward."""
        negative_prompt_embeds = replay_tensors.get("negative_prompt_embeds")
        return PixArtSigmaSamplingState(
            latents=latents,
            timesteps=pack_eval_timestep(replay_tensors["timesteps"], step_idx),
            # forward_step never calls scheduler.step, so replay needs none.
            scheduler=None,
            prompt_embeds=replay_tensors["prompt_embeds"],
            prompt_attention_mask=replay_tensors.get("prompt_attention_mask"),
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=replay_tensors.get("negative_prompt_attention_mask"),
            guidance_scale=batch_context["guidance_scale"],
            do_cfg=batch_context["cfg"] and negative_prompt_embeds is not None,
        )

    def prepare_replay(self, build: ModelBuild) -> None:
        """Standardize the replay scheduler onto the rollout's DDIM ladder.

        The generic replay loader builds the scheduler from the shipped config
        (DPM-Solver ODE — no per-step Gaussian), but the trajectory buffers and
        the ddim log-prob math live on the DDIM ladder the rollout used; the
        replay scheduler must carry the same timestep/alphas_cumprod table.
        """
        num_steps = build.num_steps
        if num_steps is not None:
            self._scheduler = pixart_ddim_scheduler(
                self._scheduler.config,
                num_steps,
                build.device,
            )


class PixArtSigmaPipelineModel(VaeDecodeMixin, DiffusersPipelineModelBase, PixArtSigmaModel):
    """PixArt-Sigma plus the diffusers pipeline that sampling needs.

    The pipeline brings prompt encoding, latent preparation and VAE decode;
    this class also exports the trajectory tensors replay reads back.
    """

    # -- encode_prompt -------------------------------------------------

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode the prompt via T5-XXL to sequence embeds + padding mask (no pooled)."""
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
            negative_prompt=neg,
            do_classifier_free_guidance=do_cfg,
            num_images_per_prompt=1,
            device=self._encoder_device(),
            max_sequence_length=max_seq,
            # The True path imports bs4/ftfy, and RL datasets control their
            # prompts anyway.
            clean_caption=False,
        )

        td = self.transformer.dtype
        result: dict[str, Any] = {
            "prompt_embeds": prompt_embeds.to(self.device, dtype=td),
            "prompt_attention_mask": (
                None if prompt_attention_mask is None else prompt_attention_mask.to(self.device)
            ),
        }
        if do_cfg and negative_prompt_embeds is not None:
            result["negative_prompt_embeds"] = negative_prompt_embeds.to(self.device, dtype=td)
            result["negative_prompt_attention_mask"] = (
                None
                if negative_prompt_attention_mask is None
                else negative_prompt_attention_mask.to(self.device)
            )
        return result

    # -- prepare_sampling ----------------------------------------------

    def prepare_sampling(
        self,
        request: DenoiseRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> PixArtSigmaSamplingState:
        """Build the per-request SamplingState for a denoise loop."""
        del kwargs
        pipe = self.pipeline
        device = self.device

        prompt_embeds = encoded["prompt_embeds"]
        negative_prompt_embeds = encoded.get("negative_prompt_embeds")

        # The shipped DPM-Solver is NOT used: the RL scheduler is a DDIM built
        # from the same beta config (see :func:`pixart_ddim_scheduler`), so the
        # ddim eta-SDE's clip_sample=False requirement holds by construction.
        scheduler = pixart_ddim_scheduler(pipe.scheduler.config, request.num_steps, device)

        seed = request.seed if request.seed is not None else random.randint(0, sys.maxsize)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        latents = pipe.prepare_latents(
            prompt_embeds.shape[0],
            pipe.transformer.config.in_channels,
            request.height,
            request.width,
            torch.float32,
            device,
            generator,
        )

        return PixArtSigmaSamplingState(
            latents=latents,
            timesteps=scheduler.timesteps,
            scheduler=scheduler,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=encoded.get("prompt_attention_mask"),
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=encoded.get("negative_prompt_attention_mask"),
            guidance_scale=request.guidance_scale,
            do_cfg=request.guidance_scale > 1.0 and negative_prompt_embeds is not None,
        )

    # -- trajectory boundary -------------------------------------------
    # Masks and negative embeds are exported only when present, so the no-CFG
    # path stays tensor-free and restore reads them back with ``.get``.

    def export_batch_context(self, state: PixArtSigmaSamplingState) -> dict[str, Any]:
        """Project sampling state into shared trajectory context."""
        return {"guidance_scale": state.guidance_scale, "cfg": state.do_cfg}

    def export_replay_tensors(self, state: PixArtSigmaSamplingState) -> dict[str, Any]:
        """Project sampling state into per-sample trajectory tensors."""
        tensors: dict[str, Any] = {
            "prompt_embeds": state.prompt_embeds,
            "latents_clean": state.latents.detach(),
        }
        for name in (
            "prompt_attention_mask",
            "negative_prompt_embeds",
            "negative_prompt_attention_mask",
        ):
            value = getattr(state, name)
            if value is not None:
                tensors[name] = value
        return tensors


__all__ = [
    "PixArtSigmaModel",
    "PixArtSigmaPipelineModel",
    "PixArtSigmaSamplingState",
    "pixart_ddim_scheduler",
]
