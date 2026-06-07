"""NextStep-1 wrapper for autoregressive image RL with continuous tokens.

Mirrors ``vrl.models.ar.janus_pro.JanusProModel`` but for StepFun's
continuous-token AR model. The shape contract is:

  * ``recompute_logprobs(...)`` →
        fresh_logprobs       [B, L]            # under current model

  * ``decode_image_tokens(...)`` → pixels [B, 3, H, W]

  * ``disable_adapter()`` — context manager for the LoRA-off ref pass

The "logits" abstraction does not apply: tokens are continuous so we
work with per-token Gaussian log-probs directly. The ``OnlineTrainer``
+ ``TokenGRPO`` pipeline is shape-agnostic (it only sees ``[B, L]``
log-prob tensors), so no trainer-side change is required.

UPSTREAM BINDING
================
This wrapper calls the upstream NextStep-1 package and its ``inference/``
scripts directly. The remaining binding-sensitive point is ``_init_kv``:
the KV-cache handle follows the upstream Qwen-style decoder cache type.

The flow head's velocity-call signature is handled in
``vrl.math.ar.flow_matching``.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from vrl.math.ar.flow_matching import (
    flow_logprob_at,
)
from vrl.models.dtypes import resolve_torch_dtype
from vrl.models.interfaces import ReplayRequest, ReplayResult, ReplaySegmentResult
from vrl.models.utils import count_trainable_params, disable_adapter_on, load_weights_into

logger = logging.getLogger(__name__)


# NextStep-1 image grid: 32x32 continuous patches at f8ch16 = 16-channel,
# 8x downsample VAE (per the model card). Override via config if you load
# a different checkpoint that uses a different grid.
NEXTSTEP_DEFAULT_TOKEN_NUM = 1024     # 32 x 32 patches per 256^2 image
NEXTSTEP_DEFAULT_TOKEN_DIM = 64       # latent_patch_size^2 * f8ch16 channels
NEXTSTEP_DEFAULT_PIXEL_SIZE = 256


@dataclass(slots=True)
class NextStep1Config:
    """Hyper-parameters for the NextStep-1 wrapper.

    Defaults target ``stepfun-ai/NextStep-1.1`` — the RL-post-trained
    14B variant — paired with the f8ch16 VAE tokenizer.
    """

    model_path: str = "stepfun-ai/NextStep-1.1"
    vae_path: str = "stepfun-ai/NextStep-1-f8ch16-Tokenizer"
    dtype: str = "bfloat16"
    device: str = "cuda"

    # LoRA — applied to the LLM trunk (the 14B AR transformer)
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    # NextStep-1's LLM is Qwen-derived; same names as Qwen-2 attention
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
    )
    lora_init: str = "gaussian"

    # Flow-head sampling — used by the AR runtime runner.
    num_flow_steps: int = 20             # K Euler steps inside the flow ODE
    noise_level: float = 1.0             # final-step Gaussian std multiplier
    cfg_scale: float = 4.5               # CFG strength on the velocity field

    # AR loop
    image_token_num: int = NEXTSTEP_DEFAULT_TOKEN_NUM
    token_dim: int = NEXTSTEP_DEFAULT_TOKEN_DIM
    image_size: int = NEXTSTEP_DEFAULT_PIXEL_SIZE

    # Frozen sub-modules
    freeze_vae: bool = True
    freeze_image_head: bool = False     # train the 157M flow head with LoRA-style updates

    # Memory
    gradient_checkpointing: bool = True


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class NextStep1Model(nn.Module):
    """Continuous-token AR T2I wrapper for the GRPO trainer.

    Composes:
      * ``self._pipeline``       : upstream ``NextStepPipeline`` (lazy-loaded)
      * ``self.language_model``  : the LLM trunk (LoRA target)
      * ``self.image_head``      : the 157M flow-matching MLP head
      * ``self.image_in_projector``: continuous-token → LLM-hidden projection
      * ``self.vae``             : f8ch16 VAE for decode
      * ``self.processor``       : tokenizer + chat-template
    """

    def __init__(self, config: NextStep1Config) -> None:
        super().__init__()
        self.config = config

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[config.dtype]
        self.dtype = torch_dtype
        self._device = torch.device(config.device)

        self._pipeline = self._load_pipeline()

        # Upstream NextStepModel inherits Qwen2Model + NextStepMixin —
        # i.e. the AR transformer trunk and the image head/projector live
        # on the same object. There is no separate ``.llm`` attribute.
        # We treat the whole NextStepModel as ``language_model`` so PEFT
        # can attach to its Qwen2 attention modules.
        self.language_model = self._pipeline.model
        self.image_head = self._pipeline.model.image_head
        self._image_in_projector = self._pipeline.model.image_in_projector
        self._image_out_projector = self._pipeline.model.image_out_projector
        self.vae = self._pipeline.vae
        self.processor = self._pipeline.tokenizer  # AutoTokenizer (naming aligns with Janus)
        self.config.token_dim = int(
            getattr(self.image_head, "input_dim", self.config.token_dim),
        )

        # Freeze what shouldn't be trained.
        if config.freeze_vae:
            for p in self.vae.parameters():
                p.requires_grad_(False)

        if config.use_lora:
            self._attach_lora()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_pipeline(self) -> Any:
        """Instantiate the upstream NextStep-1 pipeline.

        The upstream inference modules must already be importable in the
        runtime environment; this wrapper does not mutate ``sys.path``.
        """
        from gen_pipeline import NextStepPipeline  # type: ignore[import-not-found]

        return NextStepPipeline(
            model_name_or_path=self.config.model_path,
            vae_name_or_path=self.config.vae_path,
            device=str(self.device),
            dtype=self.dtype,
            enable_gradient_checkpointing=self.config.gradient_checkpointing,
        )

    def _attach_lora(self) -> None:
        from peft import LoraConfig, get_peft_model

        lora_cfg = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=list(self.config.lora_target_modules),
            init_lora_weights=self.config.lora_init,
        )
        # The whole NextStepModel becomes the PEFT-wrapped module. Since
        # the pipeline holds the model by reference, the upstream
        # ``decoding()`` path automatically sees the LoRA'd weights.
        self.language_model = get_peft_model(self.language_model, lora_cfg)
        self._pipeline.model = self.language_model

    # ------------------------------------------------------------------
    # Public: trainable param count
    # ------------------------------------------------------------------

    def trainable_param_count(self) -> int:
        return count_trainable_params(self)

    def load_trainable_state(self, state_dict: Mapping[str, Any]) -> Any:
        """Load only trainable NextStep parameters from a rollout sync state."""
        return load_weights_into(self, state_dict, prefix="model", label="NextStep1Model")

    @property
    def device(self) -> torch.device:
        """Device where the upstream NextStep pipeline is loaded."""
        return self._device

    # ------------------------------------------------------------------
    # Public: training-time log-prob recomputation
    # ------------------------------------------------------------------

    def recompute_logprobs(
        self,
        prompt_embeds: torch.Tensor,         # [B, L_text, D_hidden]
        uncond_embeds: torch.Tensor | None,
        prompt_mask: torch.Tensor,
        uncond_mask: torch.Tensor | None,
        tokens: torch.Tensor,                # [B, L_img, D_token]
        saved_noise: torch.Tensor,           # [B, L_img, D_token]
        *,
        cfg_scale: float | None = None,
        num_flow_steps: int | None = None,
        noise_level: float | None = None,
    ) -> torch.Tensor:
        """Re-compute fresh per-token log-probs under the current model.

        Returns ``[B, L_img]`` log-probs with grad through ``image_head``
        and (if LoRA is attached) through the LLM as well.
        """
        cfg = self.config
        cfg_scale = cfg_scale if cfg_scale is not None else cfg.cfg_scale
        num_flow_steps = num_flow_steps if num_flow_steps is not None else cfg.num_flow_steps
        noise_level = noise_level if noise_level is not None else cfg.noise_level

        B, L_img, _ = tokens.shape

        # Re-prime the LLM so its hidden states reflect the current LoRA'd
        # parameters. Same path as sampling but with grad enabled.
        kv_cond = self._init_kv(prompt_embeds, prompt_mask)
        kv_uncond = (
            self._init_kv(uncond_embeds, uncond_mask)
            if uncond_embeds is not None else None
        )
        c_cond = self._last_hidden(kv_cond)
        c_uncond = self._last_hidden(kv_uncond) if kv_uncond is not None else None

        out = torch.zeros(B, L_img, device=tokens.device, dtype=torch.float32)
        for j in range(L_img):
            lp = flow_logprob_at(
                self.image_head,
                cond=c_cond,
                target_token=tokens[:, j],
                saved_noise=saved_noise[:, j],
                num_flow_steps=num_flow_steps,
                noise_level=noise_level,
                cfg_uncond=c_uncond,
                cfg_scale=cfg_scale,
            )
            out[:, j] = lp.float()

            proj = self._image_in_projector(tokens[:, j])
            kv_cond, c_cond = self._step_llm(kv_cond, proj)
            if kv_uncond is not None:
                proj_u = self._image_in_projector(tokens[:, j])
                kv_uncond, c_uncond = self._step_llm(kv_uncond, proj_u)

        return out

    # ------------------------------------------------------------------
    # Replay forward — ReplayModel contract
    # ------------------------------------------------------------------

    def replay_forward(
        self,
        batch: Any,
        timestep_idx: int = 0,
        *,
        request: ReplayRequest | None = None,
    ) -> ReplayResult:
        """Re-run the AR loop and return per-token log-probs.

        Train-time replay for ``ContinuousTokenLogProbEvaluator``: reads prompt
        ids, CFG inputs, sampled continuous tokens, and saved noise from
        ``batch.trajectory``; returns log-probs under the current model.

        Differs from Janus's ``replay_forward`` (which returns logits) — for
        continuous tokens we go straight to log-probs since there is no
        codebook to softmax over.

        AR has no notion of "denoising step", so ``timestep_idx`` is ignored.

        Returns:
          ``ReplayResult`` with ``log_probs`` and ``tokens`` for ``image_tokens``.
        """
        del request, timestep_idx
        from vrl.trajectory import TrajectoryResolver

        resolver = TrajectoryResolver.from_batch(batch)
        replay = resolver.replay_tensor_dict("image_tokens")
        prompt_ids = replay["prompt_input_ids"]
        prompt_mask = replay["prompt_attention_mask"]
        uncond_ids = replay.get("uncond_input_ids")
        uncond_mask = replay.get("uncond_attention_mask")
        tokens = resolver.role_value("image_tokens", "action")
        saved_noise = replay["saved_noise"]

        embed = self.language_model.get_input_embeddings()
        prompt_embeds = embed(prompt_ids)
        uncond_embeds = embed(uncond_ids) if uncond_ids is not None else None

        log_probs = self.recompute_logprobs(
            prompt_embeds, uncond_embeds, prompt_mask, uncond_mask,
            tokens=tokens, saved_noise=saved_noise,
            cfg_scale=batch.context.get("cfg_scale"),
            num_flow_steps=batch.context.get("num_flow_steps"),
            noise_level=batch.context.get("noise_level"),
        )
        return ReplayResult(
            segments={
                "image_tokens": ReplaySegmentResult(
                    segment="image_tokens",
                    values={"log_probs": log_probs, "tokens": tokens},
                ),
            },
        )

    # ------------------------------------------------------------------
    # Public: decode tokens → pixels
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode_image_tokens(
        self,
        tokens: torch.Tensor,        # [B, L_img, D_token]
        image_size: int | None = None,
    ) -> torch.Tensor:
        """Continuous tokens → pixels in ``[-1, 1]`` via the f8ch16 VAE."""
        del image_size
        side = int(tokens.shape[1] ** 0.5)
        if side * side != tokens.shape[1]:
            raise ValueError(
                f"image_token_num must be a square grid, got {tokens.shape[1]}",
            )
        latent = self._pipeline.model.unpatchify(tokens, h=side, w=side)
        latent = (
            latent / self._pipeline.scaling_factor
        ) + self._pipeline.shift_factor
        decoded = self.vae.decode(latent.to(self.vae.dtype))
        pixels = decoded.sample if hasattr(decoded, "sample") else decoded[0]
        return pixels.to(torch.float32)

    # ------------------------------------------------------------------
    # Public: reference-model hook
    # ------------------------------------------------------------------

    def disable_adapter(self) -> contextlib.AbstractContextManager[None]:
        """Disable the LoRA adapter for a reference forward, or no-op when absent."""

        return disable_adapter_on(self.language_model)

    # ------------------------------------------------------------------
    # Internal: LLM step / KV plumbing
    # ------------------------------------------------------------------

    def _init_kv(
        self,
        embeds: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> Any:
        """Prime the LLM with text-prompt embeddings, return a KV-cache handle.

        TODO(nextstep-binding): the actual KV-cache type depends on the
        underlying ``transformers`` model class (Qwen-2). For HF this is a
        ``DynamicCache``. The pipeline's ``decoding()`` method already does
        this — we'll piggyback once we wire the binding.
        """
        out = self.language_model(
            inputs_embeds=embeds,
            attention_mask=mask,
            use_cache=True,
            output_hidden_states=True,
        )
        return {
            "past_key_values": out.past_key_values,
            "last_hidden": out.hidden_states[-1][:, -1],  # [B, D_hidden]
        }

    @staticmethod
    def _last_hidden(kv: Any) -> torch.Tensor:
        return kv["last_hidden"]

    def _lm_trunk(self) -> Any:
        """Return the Qwen-style decoder trunk, peeling PEFT when attached."""

        lm = self.language_model
        peft_inner = getattr(lm, "base_model", None)
        if (
            peft_inner is not None
            and hasattr(peft_inner, "model")
            and peft_inner.model is not lm
        ):
            return peft_inner.model
        return lm

    def _step_llm(
        self,
        kv: Any,
        new_embed: torch.Tensor,         # [B, D_hidden]
    ) -> tuple[Any, torch.Tensor]:
        """One-token LLM forward; returns updated kv + new last hidden."""
        out = self.language_model(
            inputs_embeds=new_embed.unsqueeze(1),
            past_key_values=kv["past_key_values"],
            use_cache=True,
            output_hidden_states=True,
        )
        kv2 = {
            "past_key_values": out.past_key_values,
            "last_hidden": out.hidden_states[-1][:, -1],
        }
        return kv2, kv2["last_hidden"]


class NextStep1ReplayModel(NextStep1Model):
    """Replay-only NextStep-1 wrapper without VAE, tokenizer, or pipeline."""

    def __init__(
        self,
        config: NextStep1Config,
        *,
        language_model: Any | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.dtype = resolve_torch_dtype(config.dtype)
        self._device = torch.device(config.device)

        self.language_model = (
            language_model
            if language_model is not None
            else _load_nextstep_replay_model(config)
        )
        self.image_head = self.language_model.image_head
        self._image_in_projector = self.language_model.image_in_projector
        self._image_out_projector = self.language_model.image_out_projector
        self.config.token_dim = int(
            getattr(self.image_head, "input_dim", self.config.token_dim),
        )

        if config.use_lora:
            self._attach_lora()

    def _attach_lora(self) -> None:
        from peft import LoraConfig, get_peft_model

        lora_cfg = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=list(self.config.lora_target_modules),
            init_lora_weights=self.config.lora_init,
        )
        self.language_model = get_peft_model(self.language_model, lora_cfg)

    @torch.no_grad()
    def decode_image_tokens(
        self,
        tokens: torch.Tensor,
        image_size: int | None = None,
    ) -> torch.Tensor:
        del tokens, image_size
        raise RuntimeError("NextStep1ReplayModel cannot decode image tokens")


def _load_nextstep_replay_model(config: NextStep1Config) -> Any:
    """Load the upstream NextStep model without the inference pipeline or VAE."""

    from nextstep_model import NextStep  # type: ignore[import-not-found]

    model = NextStep.from_pretrained(
        config.model_path,
        torch_dtype=resolve_torch_dtype(config.dtype),
        enable_gradient_checkpointing=config.gradient_checkpointing,
    )
    return model.to(device=config.device, dtype=resolve_torch_dtype(config.dtype)).eval()


__all__ = [
    "NEXTSTEP_DEFAULT_PIXEL_SIZE",
    "NEXTSTEP_DEFAULT_TOKEN_DIM",
    "NEXTSTEP_DEFAULT_TOKEN_NUM",
    "NextStep1Config",
    "NextStep1Model",
    "NextStep1ReplayModel",
]
