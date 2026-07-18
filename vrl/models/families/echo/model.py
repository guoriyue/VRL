"""JoyAI-Echo (LTX-2.3-derived) model on the VRL diffusion RL path.

Echo ships as a custom, non-diffusers, DMD-distilled audio-video generator. Its
inner ``LTXModel`` is nonetheless a plain **flow-matching velocity model**: with
``x_t = (1-sigma)*x0 + sigma*eps`` the velocity
``v = (x_t - x0)/sigma = eps - x0`` is exactly the
diffusers rectified-flow convention SD3/Flux/Wan already use. So we wrap Echo's
**video** transformer as a standard flow-matching policy and drive it through the
shared diffusion executor / GRPO replay machinery unchanged.

Out of scope here (Echo inference-only features, intentionally not on the RL
path): joint audio generation and the cross-shot ``PairedAudioVideoMemoryBank``.
This is single-shot, video-only — ``forward_step`` calls Echo with
``noisy_audio=None`` and no memory.

The released checkpoint is DMD-distilled for ~8-step sampling, but the velocity
field is defined at every sigma, so flow-matching SDE sampling for RL is well-posed
(the same regime flow-GRPO already runs).
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from typing import Any

import torch

from vrl.generation.types import VideoGenerationRequest
from vrl.models.interfaces.runtime import ModelBuild
from vrl.models.steps.denoise.base import (
    DiffusionModelBase,
    DiffusionSamplingStateBase,
    ReplayRolloutStubs,
)
from vrl.models.steps.denoise.common.lora import LoraModelMixin

# LTX-2 latent channel dimension (transformer in/out channels). The other latent
# grid factors (temporal/spatial compression) are read from Echo's own
# SpatioTemporalScaleFactors at runtime; this one architecture dim has no
# lighter Echo accessor that imports cleanly here, so it stays named.
_LATENT_CHANNELS = 128


def _resolve_echo_checkpoint(
    path_or_repo: str,
    *,
    revision: str | None = None,
) -> str:
    """Local path to the Echo safetensors, fetched into the HF cache if a repo id.

    Same convention as wan/sd3 configs (an HF repo id): the merged release lands
    in ``~/.cache/huggingface``, shared across clones, not the repo tree. An
    existing local file path is used as-is.
    """

    from pathlib import Path

    if Path(path_or_repo).is_file():
        return path_or_repo
    from huggingface_hub import hf_hub_download

    # The single merged release artifact (transformer + VAEs + vocoder).
    load_kwargs = {"revision": revision} if revision else {}
    return hf_hub_download(
        path_or_repo,
        "JoyAI-Echo-release.safetensors",
        **load_kwargs,
    )


def _resolve_gemma_dir(
    path_or_repo: str,
    *,
    revision: str | None = None,
) -> str:
    """Local dir of the Gemma encoder, fetched into the HF cache if a repo id."""

    from pathlib import Path

    if Path(path_or_repo).is_dir():
        return path_or_repo
    from huggingface_hub import snapshot_download

    load_kwargs = {"revision": revision} if revision else {}
    return snapshot_download(
        repo_id=path_or_repo,
        allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*", "*.txt"],
        **load_kwargs,
    )


@dataclass
class EchoSamplingState(DiffusionSamplingStateBase):
    """Private Echo sampling state. Engine MUST NOT introspect beyond the
    documented ``latents`` / ``timesteps`` / ``scheduler`` contract.

    ``latents`` are [B, F, C=128, H, W] (Echo layout). ``timesteps`` are
    flow-matching scheduler timesteps in ``[0, num_train]`` (the executor reads
    them per step); ``forward_step`` derives sigma = t/num_train. In replay
    ``scheduler`` is None (the evaluator owns its own scheduler).
    """

    video_context: torch.Tensor  # Gemma text context [B, S, 4096]
    attention_mask: Any  # text padding mask [B, S] or None
    num_train_timesteps: int  # sigma = t / num_train_timesteps


class EchoModel(LoraModelMixin, DiffusionModelBase):
    """Diffusers-free JoyAI-Echo video flow-matching policy."""

    # Echo replay keeps the velocity model in its native dtype; mirror the family
    # convention of casting the LoRA-wrapped transformer to the run dtype.
    def __init__(
        self,
        *,
        echo: Any,  # LTX2DiffusionWrapper
        text_encoder: Any,  # GemmaTextEncoderWrapper
        video_vae: Any,  # VideoVAEWrapper
        scheduler: Any,
        dtype: torch.dtype,
        device: Any = None,
    ) -> None:
        super().__init__()
        # The wrapper owns the X0Model; expose it as ``self.transformer`` so the
        # shared base (LoRA attach, weight sync, trainable_modules) operates on the
        # real nn.Module, and keep the two in sync in ``_set_transformer``.
        object.__setattr__(self, "_echo", echo)
        object.__setattr__(self, "_text_encoder", text_encoder)
        object.__setattr__(self, "_video_vae", video_vae)
        self.transformer = echo.model
        self._scheduler = scheduler
        self._dtype = dtype
        self._device = device

    @property
    def device(self) -> Any:
        return self._device

    def _set_transformer(self, transformer: Any) -> None:
        self.transformer = transformer
        self._echo.model = transformer

    def _lora_transformer(self) -> Any:
        return self.transformer

    def _lora_dtype(self, build: ModelBuild) -> Any:
        del build
        return self._dtype

    @property
    def trainable_modules(self) -> dict[str, Any]:
        return {"transformer": self.transformer}

    @property
    def scheduler(self) -> Any:
        return self._scheduler

    @property
    def raw_handle(self) -> Any:
        return self._echo

    def apply_full_finetune(self) -> None:
        self.transformer.requires_grad_(True)
        self.transformer.to(self.device)

    def enable_gradient_checkpointing(self) -> None:
        """Turn on activation checkpointing in the LTX velocity blocks.

        Echo's checkpointing flag lives on the inner ``LTXModel``
        (``set_gradient_checkpointing``), not the diffusers
        ``enable_gradient_checkpointing`` the generic helper expects. Walk the
        module tree so this also works after PEFT wraps the X0Model for LoRA.
        """

        for module in self.transformer.modules():
            setter = getattr(module, "set_gradient_checkpointing", None)
            if callable(setter):
                setter(True)
                return
        raise AttributeError(
            "Echo transformer exposes no set_gradient_checkpointing module",
        )

    def move_frozen_components(self, device: Any) -> None:
        """Move the frozen prompt encoder + VAE to ``device``.

        Echo owns no diffusers pipeline, so the base (pipeline-derived) version is
        a no-op for it. Implementing it lets Echo participate in the generic
        frozen-component offload the rollout lifecycle already drives — park on
        CPU while idle, restore on use. This is **device-driven**, not tied to any
        specific card's VRAM: the same code offloads on a 32GB card and stays put
        on a big one (the lifecycle decides). The trainable transformer is never
        touched here.
        """

        for wrapper in (self._text_encoder, self._video_vae):
            if wrapper is None:
                continue
            for name in ("text_encoder", "embeddings_processor", "encoder", "decoder", "vocoder"):
                module = getattr(wrapper, name, None)
                if isinstance(module, torch.nn.Module):
                    module.to(device)
            # The wrapper caches a device it moves its own inputs/outputs to.
            if hasattr(wrapper, "device"):
                wrapper.device = device

    def set_num_steps(self, n: int) -> None:
        self._scheduler.set_timesteps(int(n), device=self.device)

    # -- backend ownership ---------------------------------------------

    @classmethod
    def from_build(cls, build: ModelBuild) -> EchoModel:
        """Build Echo's text encoder + LTX transformer + video VAE from a build.

        ``build.model_name_or_path`` is the merged Echo safetensors checkpoint;
        ``build.model_config['gemma_path']`` is the separate Gemma-3-12B directory.
        """

        from diffusers import FlowMatchEulerDiscreteScheduler
        from ltx_distillation.models.ltx_wrapper import create_ltx2_wrapper
        from ltx_distillation.models.text_encoder_wrapper import (
            create_text_encoder_wrapper,
        )
        from ltx_distillation.models.vae_wrapper import create_vae_wrappers

        dtype = build.parameter_dtype
        device = torch.device(build.device) if build.device is not None else torch.device("cpu")
        model_cfg = build.model_config or {}
        gemma_ref = model_cfg.get("gemma_path")
        if not gemma_ref:
            raise ValueError(
                "Echo requires model.gemma_path (a Gemma-3-12B HF repo id or local dir)",
            )
        from vrl.models.loader import model_config_revision_kwargs, model_revision_kwargs

        checkpoint = _resolve_echo_checkpoint(
            build.model_name_or_path,
            **model_revision_kwargs(build),
        )
        gemma_path = _resolve_gemma_dir(
            gemma_ref,
            **model_config_revision_kwargs(build, "gemma_revision"),
        )

        # Sampling resolution sizes the wrapper's RoPE/patch grid; the rollout
        # request can override per call, but the wrapper wants a build-time
        # default. Read it from the sampling block when present.
        sampling = getattr(build, "sampling_config", None) or {}
        video_height = int(sampling.get("height", 512))
        video_width = int(sampling.get("width", 768))

        text_encoder = create_text_encoder_wrapper(
            checkpoint_path=checkpoint,
            gemma_path=gemma_path,
            device=device,
            dtype=dtype,
        )
        echo = create_ltx2_wrapper(
            checkpoint_path=checkpoint,
            gemma_path=gemma_path,
            device=device,
            dtype=dtype,
            video_height=video_height,
            video_width=video_width,
        )
        video_vae, _audio_vae = create_vae_wrappers(
            checkpoint_path=checkpoint,
            device=device,
            dtype=dtype,
            with_video_encoder=False,
        )
        # Echo ships a bespoke DMD few-step sampler, but for RL we drive its
        # velocity field with the standard flow-matching SDE (sigma schedule +
        # sde_step_with_logprob), same as the other diffusion families.
        # num_train_timesteps=1000 is Echo/LTX's train-time discretization:
        # forward_step derives sigma = t / num_train_timesteps, so this must match the
        # value the transformer's timestep embedding was trained on, not a free knob.
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        return cls(
            echo=echo,
            text_encoder=text_encoder,
            video_vae=video_vae,
            scheduler=scheduler,
            dtype=dtype,
            device=device,
        )

    # -- encode_prompt -------------------------------------------------

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode the prompt to Gemma video context (video-only: audio dropped).

        Returns batch-1 tensors; the executor repeats them to the chunk's sample
        count. ``negative_prompt`` is unused — Echo's DMD checkpoint bakes in
        guidance, so the RL policy runs without classifier-free guidance.
        """

        del negative_prompt
        text = prompt if isinstance(prompt, list) else [prompt]
        cond = self._text_encoder(text)
        return {
            "video_context": cond["video_context"],
            "attention_mask": cond.get("attention_mask"),
        }

    # -- prepare_sampling ----------------------------------------------

    def prepare_sampling(
        self,
        request: VideoGenerationRequest,
        encoded: dict[str, Any],
        **kwargs: Any,
    ) -> EchoSamplingState:
        """Build initial flow-matching noise latents + the sigma schedule."""

        del kwargs
        device = self.device
        video_context = encoded["video_context"]
        batch_size = int(video_context.shape[0])

        self._scheduler.set_timesteps(request.num_steps, device=device)
        timesteps = self._scheduler.timesteps

        # Latent grid compression read from Echo's own VAE scale factors (its
        # source of truth) rather than hand-copied 8/32 magic numbers.
        from ltx_core.types import SpatioTemporalScaleFactors

        sf = SpatioTemporalScaleFactors.default()
        if (request.frame_count - 1) % sf.time != 0:
            raise ValueError(
                f"Echo frame_count must be 1 + {sf.time}*k, got {request.frame_count}",
            )
        latent_shape = (
            batch_size,
            1 + (request.frame_count - 1) // sf.time,
            _LATENT_CHANNELS,
            request.height // sf.height,
            request.width // sf.width,
        )
        seed = request.seed if request.seed is not None else random.randint(0, sys.maxsize)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        latents = torch.randn(
            latent_shape,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )

        return EchoSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=self._scheduler,
            video_context=video_context,
            attention_mask=encoded.get("attention_mask"),
            num_train_timesteps=int(self._scheduler.config.num_train_timesteps),
            guidance_scale=request.guidance_scale,
        )

    # -- forward_step --------------------------------------------------

    def forward_step(self, state: EchoSamplingState, step_idx: int) -> dict[str, Any]:
        """One Echo velocity forward: predict x0, convert to flow-matching velocity.

        Echo returns x0; the executor/evaluator both consume a velocity
        ``noise_pred`` (rectified-flow ``v = (x_t - x0)/sigma``). sigma is derived from the
        flow-match timestep so rollout and replay agree without sharing scheduler
        state.
        """

        td = self._transformer_dtype()
        latents = state.latents.to(td)
        t = state.timesteps[step_idx]
        sigma = _sigma_from_timestep(t, state.num_train_timesteps, latents)
        # Echo's wrapper takes sigma directly (its Modality.sigma), broadcast per batch.
        echo_sigma = sigma.reshape(-1).to(td)
        if echo_sigma.numel() == 1:
            echo_sigma = echo_sigma.expand(latents.shape[0])

        # Move context to the transformer's device (not just dtype) so a
        # CPU-offloaded prompt encoder's embeds still land where the denoiser runs.
        conditional_dict = {
            "video_context": state.video_context.to(device=latents.device, dtype=td),
            "audio_context": None,
            "attention_mask": state.attention_mask,
        }
        x0, _audio = self._echo.forward(
            noisy_image_or_video=latents,
            conditional_dict=conditional_dict,
            timestep=echo_sigma,
            noisy_audio=None,
        )
        # v = (x_t - x0) / sigma, in fp32 for the downstream SDE log-prob math.
        sigma_b = sigma.float().reshape(-1, *[1] * (latents.dim() - 1))
        velocity = (latents.float() - x0.float()) / sigma_b
        return {"noise_pred": velocity}

    # -- collector boundary --------------------------------------------

    def export_batch_context(self, state: EchoSamplingState) -> dict[str, Any]:
        return {
            "guidance_scale": state.guidance_scale,
            "num_train_timesteps": state.num_train_timesteps,
        }

    def export_replay_tensors(self, state: EchoSamplingState) -> dict[str, Any]:
        return {
            "video_context": state.video_context,
            "attention_mask": state.attention_mask,
        }

    def restore_eval_state(
        self,
        replay_tensors: dict[str, Any],
        batch_context: dict[str, Any],
        latents: Any,
        step_idx: int,
    ) -> EchoSamplingState:
        """Rebuild state for the trainer replay forward (no scheduler.step here)."""

        ts = replay_tensors["timesteps"]
        # Pack as [1, B] so state.timesteps[0] is the per-sample timestep [B],
        # matching the eval-path convention (the executor calls forward_step with
        # step_idx=0 on the restored single-step state).
        if isinstance(ts, torch.Tensor) and ts.ndim >= 1:
            timesteps = ts[:, step_idx].reshape(1, -1) if ts.ndim > 1 else ts.reshape(1, -1)
        else:
            timesteps = torch.as_tensor(ts).reshape(1, -1)
        return EchoSamplingState(
            latents=latents,
            timesteps=timesteps,
            scheduler=None,
            video_context=replay_tensors["video_context"],
            attention_mask=replay_tensors.get("attention_mask"),
            num_train_timesteps=int(batch_context["num_train_timesteps"]),
            guidance_scale=batch_context.get("guidance_scale", 1.0),
        )

    # -- decode_latents ------------------------------------------------

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode video latents [B,F,C,H,W] → frames in [0, 1], layout [B,C,T,H,W]."""

        pixels = self._video_vae.decode(latents.to(self._dtype))
        pixels = pixels.float()
        # VAE emits [-1, 1]; the rollout wire/reward path expects [0, 1].
        return ((pixels + 1.0) / 2.0).clamp_(0.0, 1.0)


def _sigma_from_timestep(
    t: torch.Tensor,
    num_train_timesteps: int,
    like: torch.Tensor,
) -> torch.Tensor:
    """Flow-match timestep -> sigma in (0, 1]; guards sigma=0."""

    sigma = torch.as_tensor(t, device=like.device, dtype=torch.float32) / float(
        num_train_timesteps
    )
    return sigma.clamp_min(1e-6)


class EchoReplayModel(ReplayRolloutStubs, EchoModel):
    """Replay-only Echo model: the velocity transformer + a flow scheduler.

    Owns no text encoder, VAE, or sampling pipeline — the trainer only needs the
    velocity forward to recompute log-probs.
    """

    def __init__(
        self,
        *,
        echo: Any,
        scheduler: Any,
        dtype: torch.dtype,
        device: Any = None,
    ) -> None:
        DiffusionModelBase.__init__(self)
        object.__setattr__(self, "_echo", echo)
        object.__setattr__(self, "_text_encoder", None)
        object.__setattr__(self, "_video_vae", None)
        self.transformer = echo.model
        self._scheduler = scheduler
        self._dtype = dtype
        self._device = device

    @property
    def raw_handle(self) -> Any:
        return None


__all__ = ["EchoModel", "EchoReplayModel", "EchoSamplingState"]
