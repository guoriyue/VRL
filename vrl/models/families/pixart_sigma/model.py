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
  half the (unused) variance. ``postprocess_branch`` chunks per branch BEFORE
  the CFG combine — the pipeline chunks after, which is equivalent since
  channel-chunk and the linear CFG combine commute.
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
class PixArtSigmaSamplingState(GuidedDiffusionSamplingStateBase):
    """Private PixArt-Sigma sampling state. Engine MUST NOT introspect."""

    prompt_embeds: torch.Tensor
    prompt_attention_mask: torch.Tensor | None
    negative_prompt_embeds: torch.Tensor | None
    negative_prompt_attention_mask: torch.Tensor | None
    do_cfg: bool


class PixArtSigmaModel(LoraModelMixin, DiffusersPipelineModelBase, DiffusionBackboneRunnerBase):
    """Diffusers-backed PixArt-Sigma t2i model (epsilon DDPM family).

    Implements the backbone-runner protocol itself. Both CFG branches pad to
    the same sequence length (T5 encode at ``max_sequence_length``), so they
    pack into one batched transformer call; the attention masks ride the
    batch as ``encoder_attention_mask``.
    """

    cfg_mode = "batched_cfg"
    cfg_base = "uncond"

    def build_branch(
        self,
        request: DiffusionBackboneInput,
        branch: str,
    ) -> DiffusionBranch:
        """Map PixArt-Sigma transformer kwargs into the shared backbone contract."""
        if branch == "cond":
            embeds = request.prompt_embeds
            mask = request.extra.get("encoder_attention_mask")
        else:
            embeds = require_tensor(
                request.negative_prompt_embeds,
                "negative_prompt_embeds",
            )
            mask = request.extra.get("negative_encoder_attention_mask")
        return DiffusionBranch(
            hidden_states=request.hidden_states,
            timestep=request.timestep,
            encoder_hidden_states=embeds,
            extra_kwargs={
                "encoder_attention_mask": mask,
                "added_cond_kwargs": _ADDED_COND_KWARGS,
            },
        )

    def postprocess_branch(
        self,
        request: DiffusionBackboneInput,
        branch: DiffusionBranch,
        raw_output: torch.Tensor,
    ) -> torch.Tensor:
        """Chunk the learned-sigma output down to the epsilon prediction.

        The transformer outputs ``2 * latent_channels`` (learned variance in
        the second half). Chunking per branch before the CFG combine is
        equivalent to the pipeline's chunk-after-combine (both are linear).
        """
        del branch
        latent_channels = request.hidden_states.shape[1]
        if raw_output.shape[1] == 2 * latent_channels:
            return raw_output.chunk(2, dim=1)[0]
        return raw_output

    # -- backend ownership (called by runtime, not by collectors) -------

    @classmethod
    def from_build(cls, build: ModelBuild) -> PixArtSigmaModel:
        """Load the diffusers PixArt-Sigma pipeline + freeze non-trainable modules."""
        from diffusers import PixArtSigmaPipeline

        model_dtype = build.parameter_dtype
        prompt_encoder_dtype, load_kwargs = diffusers_pipeline_dtypes(build, model_dtype)
        pipeline = PixArtSigmaPipeline.from_pretrained(
            build.model_name_or_path,
            **load_kwargs,
        )
        pipeline.vae.requires_grad_(False)
        text_encoder = getattr(pipeline, "text_encoder", None)
        if text_encoder is not None:
            # T5-XXL (~9.5 GB bf16); park on CPU (Qwen-Image discipline).
            text_encoder.requires_grad_(False)
            text_encoder.to("cpu", dtype=prompt_encoder_dtype)
        pipeline.vae.to(build.device, dtype=torch.float32)
        return cls(
            pipeline=pipeline,
            device=build.device,
        )

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

    # -- collector boundary --------------------------------------------

    def export_batch_context(self, state: PixArtSigmaSamplingState) -> dict[str, Any]:
        """Project PixArt-Sigma sampling state into trajectory context."""
        return {
            "guidance_scale": state.guidance_scale,
            "cfg": state.do_cfg,
        }

    def export_replay_tensors(self, state: PixArtSigmaSamplingState) -> dict[str, Any]:
        """Project PixArt-Sigma sampling state into per-sample trajectory tensors.

        Masks / negative embeds are only stored when present, so ``restore``
        reads them with ``.get`` and the no-CFG path stays tensor-free.
        """
        tensors: dict[str, Any] = {"prompt_embeds": state.prompt_embeds}
        if state.prompt_attention_mask is not None:
            tensors["prompt_attention_mask"] = state.prompt_attention_mask
        if state.negative_prompt_embeds is not None:
            tensors["negative_prompt_embeds"] = state.negative_prompt_embeds
        if state.negative_prompt_attention_mask is not None:
            tensors["negative_prompt_attention_mask"] = state.negative_prompt_attention_mask
        return tensors

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> PixArtSigmaSamplingState:
        """Rebuild PixArtSigmaSamplingState from a batch slice for the eval path."""
        ts = replay_tensors["timesteps"]
        timesteps = pack_eval_timestep(ts, step_idx)
        return PixArtSigmaSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=None,  # not needed for forward_step (no scheduler.step here)
            prompt_embeds=replay_tensors["prompt_embeds"],
            prompt_attention_mask=replay_tensors.get("prompt_attention_mask"),
            negative_prompt_embeds=replay_tensors.get("negative_prompt_embeds"),
            negative_prompt_attention_mask=replay_tensors.get(
                "negative_prompt_attention_mask",
            ),
            guidance_scale=batch_context["guidance_scale"],
            do_cfg=batch_context["cfg"]
            and replay_tensors.get("negative_prompt_embeds") is not None,
        )

    # -- decode_latents ------------------------------------------------

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents -> image via the SDXL KL-VAE (4D, scaling only)."""
        pipe = self.pipeline
        vae = pipe.vae
        scaling_factor = vae.config.scaling_factor
        # SDXL VAE ships no shift_factor; read it defensively like sd3.
        shift_factor = getattr(vae.config, "shift_factor", 0.0) or 0.0
        decoder = ChunkedLatentDecoder(
            LatentDecodePlan(
                prepare_latents=lambda chunk: chunk.to(vae.dtype) / scaling_factor + shift_factor,
                vae_decode=lambda chunk: vae.decode(chunk, return_dict=False)[0],
                postprocess=lambda image: pipe.image_processor.postprocess(
                    image,
                    output_type="pt",
                ),
                output_layout="image_bchw",
                decode_batch_size=getattr(pipe, "decode_batch_size", None),
            ),
        )
        return decoder(latents)


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
