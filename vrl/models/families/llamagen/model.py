"""LlamaGen (t2i) wrapper for autoregressive text-to-image RL.

Mirrors ``vrl.models.families.janus_pro.JanusProModel`` for FoundationVision's
LlamaGen. The family-specific facts this file isolates:

  * The GPT is the vendored ``vendor/gpt.py`` Transformer (no HF integration;
    see the vendor header for why it cannot be mapped onto LlamaForCausalLM).
    Its output head IS the image vocab (16384) — there is no separate
    ``gen_head``; ``forward_image_logits`` slices the trunk's own logits.
  * Text conditioning is a 120-token prefix of frozen flan-t5-xl encoder
    features fed through the checkpoint's ``CaptionEmbedder`` MLP, left-padded
    exactly like upstream ``sample_t2i.py`` (flip mask, roll valid tokens to
    the end, zero the padded rows).
  * CFG's unconditional branch is the checkpoint's learned
    ``CaptionEmbedder.uncond_embedding`` (in T5 feature space), used with the
    *conditional* prompt's attention mask — upstream ``generate()`` passes
    ``torch.cat([emb_masks, emb_masks])``.
  * Image tokens decode through the vendored VQGAN (codebook 16384, 8-dim,
    ds16): a 16x16 grid -> 256 px for the stage1 checkpoint.

References
----------
Upstream sampling: https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/sample/sample_t2i.py
Upstream decode:   https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/models/generate.py
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

from vrl.models.dtypes import resolve_torch_dtype
from vrl.models.families.llamagen.config import (
    LLAMAGEN_CAPTION_DIM,
    LLAMAGEN_CAPTION_TOKEN_NUM,
    LLAMAGEN_DOWNSAMPLE_SIZE,
    LLAMAGEN_GPT_CKPT,
    LLAMAGEN_HF_REPO,
    LLAMAGEN_IMAGE_TOKEN_NUM,
    LLAMAGEN_T5_PATH,
    LLAMAGEN_VQ_CKPT,
    LlamaGenConfig,
)
from vrl.models.interfaces import (
    ReplayRequest,
    ReplayResult,
    ReplaySegmentResult,
    require_replay_segments,
    require_zero_replay_timestep,
)
from vrl.models.steps.token.base import ARModelBase, ARReplayRolloutStubs
from vrl.models.steps.token.lora import install_token_lora_adapter
from vrl.utils.logging import init_logger

logger = init_logger(__name__)

# LlamaGen t2i_XL_stage1_256 model architecture constant.
LLAMAGEN_IMAGE_VOCAB_SIZE = 16_384  # VQ-16 codebook size == GPT vocab size


class LlamaGenModel(ARModelBase):
    """Train-and-sample wrapper for LlamaGen text-to-image generation.

    Composes the vendored GPT (LoRA target), the frozen VQGAN decoder, and the
    frozen flan-t5-xl encoder (CPU-parked) in one ``nn.Module``.
    """

    def __init__(
        self,
        config: LlamaGenConfig | None = None,
        *,
        gpt: Any | None = None,
        vq_model: Any | None = None,
        t5_encoder: Any | None = None,
        t5_tokenizer: Any | None = None,
    ) -> None:
        """Construct the wrapper.

        The keyword modules let tests inject tiny-config vendored modules and
        T5 stubs without downloading checkpoints; production loading happens
        when they are ``None``.
        """
        super().__init__()
        self.config = config or LlamaGenConfig()

        if gpt is None:
            gpt = _load_llamagen_gpt(self.config)
        if vq_model is None and self._loads_vq():
            vq_model = _load_llamagen_vq(self.config)
        if t5_encoder is None:
            t5_encoder, loaded_tokenizer = _load_t5_encoder(self.config)
            if t5_tokenizer is None:
                t5_tokenizer = loaded_tokenizer
        if t5_tokenizer is None:
            raise ValueError("Must pass `t5_tokenizer` when `t5_encoder` is provided")

        # Freeze everything by default — LoRA re-enables only its adapters.
        for module in (gpt, vq_model, t5_encoder):
            if module is None:
                continue
            for p in module.parameters():
                p.requires_grad_(False)

        if self.config.use_lora:
            gpt = self._apply_lora(gpt)

        self.gpt = gpt
        # VQ decoder stays in float32 (upstream keeps it fp32 while the GPT
        # runs bf16); ``None`` on the replay subclass, which loads no decoder.
        self.vq = vq_model
        # CPU-parked frozen text encoder: prompts are encoded where T5 lives
        # and only the 120x2048 features move to the GPT device.
        self.t5_encoder = t5_encoder
        self._t5_tokenizer = t5_tokenizer

        trunk = self._gpt_trunk()
        for attr in ("cls_embedding", "tok_embeddings", "output", "layers"):
            if not hasattr(trunk, attr):
                raise RuntimeError(
                    f"Loaded model is missing `{attr}` — does not look like a "
                    "LlamaGen t2i Transformer."
                )
        if trunk.model_type != "t2i":
            raise RuntimeError(
                f"LlamaGen wrapper requires a t2i GPT; got model_type={trunk.model_type!r}"
            )

    def _loads_vq(self) -> bool:
        """Rollout models own the VQ decoder; the replay subclass skips it."""
        return True

    # ------------------------------------------------------------------
    # Sub-module accessors
    # ------------------------------------------------------------------

    def _gpt_trunk(self) -> nn.Module:
        """Return the unwrapped vendored Transformer (peels a PEFT wrap).

        PEFT replaces the target ``nn.Linear`` modules in-place, so calling
        the peeled trunk still routes through the LoRA layers (same pattern as
        Janus' ``_lm_trunk``).
        """
        m = self.gpt
        inner = getattr(m, "base_model", None)
        if inner is not None and hasattr(inner, "model") and inner.model is not m:
            return inner.model
        return m

    def _lm_trunk(self) -> nn.Module:
        """Seam hook: the AR language-model trunk.

        DOCUMENTED DEVIATION from the janus/nextstep contract: the vendored
        Transformer is NOT an HF-protocol module (no ``inputs_embeds`` /
        ``past_key_values`` forward), so the shared attention backends in
        ``vrl.nn.modules.ar_attention_backends`` cannot drive it.
        ``LlamaGenChunkExecutor`` overrides ``_ar_runner`` and the family
        runner drives the vendored static KV cache directly.
        """
        return self._gpt_trunk()

    @property
    def language_model(self) -> nn.Module:
        """The (possibly PEFT-wrapped) GPT — ``ARModelBase.disable_adapter`` target."""
        return self.gpt

    @property
    def vq_model(self) -> nn.Module:
        if self.vq is None:
            raise RuntimeError(f"{type(self).__name__} does not own a VQ decoder")
        return self.vq

    @property
    def t5_tokenizer(self) -> Any:
        return self._t5_tokenizer

    @property
    def device(self) -> torch.device:
        return next(self._gpt_trunk().parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self._gpt_trunk().parameters()).dtype

    def _apply_lora(self, gpt: Any) -> Any:
        gpt = install_token_lora_adapter(gpt, self.config)
        logger.info(
            "Applied LoRA (rank=%d, alpha=%d) to LlamaGen GPT targets %s.",
            self.config.lora_rank,
            self.config.lora_alpha,
            self.config.lora_target_modules,
        )
        return gpt

    # ------------------------------------------------------------------
    # Prompt conditioning — flan-t5-xl features, upstream left-pad layout
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_caption(
        self,
        input_ids: torch.Tensor,  # [B, cls_token_num] T5 token ids
        attention_mask: torch.Tensor,  # [B, cls_token_num]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """T5-encode caption ids into the GPT's conditioning layout.

        Reproduces upstream ``sample_t2i.py`` exactly: encode right-padded ids,
        flip the mask and roll each row's valid features to the end
        (left-padding), then zero the padded rows. Used by both rollout and
        trainer replay so the two sides see byte-identical conditioning.

        Returns ``(caption_embeds [B, T, caption_dim], emb_mask [B, T])`` on
        the GPT device, features in the GPT dtype.
        """
        if input_ids.shape[1] != self.config.cls_token_num:
            raise ValueError(
                f"caption ids must be padded to cls_token_num="
                f"{self.config.cls_token_num}; got length {input_ids.shape[1]}"
            )
        # Parameter-less test stubs fall back to the ids' device.
        t5_param = next(iter(self.t5_encoder.parameters()), None)
        t5_device = t5_param.device if t5_param is not None else input_ids.device
        outputs = self.t5_encoder(
            input_ids=input_ids.to(t5_device),
            attention_mask=attention_mask.to(t5_device),
        )
        embs = outputs.last_hidden_state.float()
        mask = attention_mask.to(embs.device)

        flipped_mask = torch.flip(mask, dims=[-1])
        rolled = []
        for row, valid in zip(embs, mask.sum(dim=-1).long().tolist(), strict=True):
            rolled.append(torch.cat([row[valid:], row[:valid]], dim=0))
        embs = torch.stack(rolled) * flipped_mask[:, :, None]
        return (
            embs.to(device=self.device, dtype=self.dtype),
            flipped_mask.to(device=self.device),
        )

    def uncond_caption_embeds(self, batch_size: int) -> torch.Tensor:
        """The learned null caption for CFG (T5 feature space).

        Upstream ``generate()``: ``cond_null = zeros_like(cond) +
        model.cls_embedding.uncond_embedding`` — the unconditional branch is
        the CaptionEmbedder's trained drop embedding, NOT an empty prompt.
        Its attention mask is the conditional prompt's mask (upstream passes
        ``cat([emb_masks, emb_masks])``); callers reuse the cond mask.
        """
        uncond = self._gpt_trunk().cls_embedding.uncond_embedding
        uncond = uncond[: self.config.cls_token_num]
        return (
            uncond.unsqueeze(0).expand(batch_size, -1, -1).to(device=self.device, dtype=self.dtype)
        )

    # ------------------------------------------------------------------
    # Train-time forward — image-token logits
    # ------------------------------------------------------------------

    def forward_image_logits(
        self,
        caption_embeds: torch.Tensor,  # [B, T, caption_dim] (from encode_caption)
        caption_mask: torch.Tensor,  # [B, T] (flipped/left-pad mask)
        image_token_ids: torch.Tensor,  # [B, L_img]
    ) -> torch.Tensor:
        """Teacher-forced forward returning per-position image-vocab logits.

        Feeds ``[caption prefix | image[:-1]]`` through the vendored trunk and
        slices the positions that *predict* each image token. The attention
        mask replicates upstream ``generate()``'s rollout mask (causal, padded
        caption columns masked out, diagonal restored), so replay logits match
        the KV-cache rollout logits position-for-position.

        Returns logits ``[B, L_img, vocab_size]`` (the trunk's ``output`` head
        IS the image vocab — no separate gen_head exists in LlamaGen).
        """
        trunk = self._gpt_trunk()
        # A leftover rollout KV cache would hijack the teacher-forced pass
        # (vendored Attention consults ``kv_cache`` whenever it is set).
        self._clear_kv_caches()

        L_img = image_token_ids.shape[1]
        T = caption_embeds.shape[1]
        if trunk.cls_token_num != T:
            raise ValueError(
                f"caption prefix length {T} != trunk cls_token_num {trunk.cls_token_num}"
            )
        total_len = T + L_img - 1
        mask = self._prefix_attention_mask(caption_mask, total_len=total_len)
        input_pos = torch.arange(total_len, device=image_token_ids.device)

        logits, _ = trunk(
            idx=image_token_ids[:, :-1],
            cond_idx=caption_embeds.to(self.dtype),
            input_pos=input_pos,
            mask=mask,
        )
        if trunk.training:
            # The vendored forward already sliced from cls_token_num - 1 in
            # train mode (all stochastic regularizers are built with p=0, so
            # train/eval forwards are numerically identical).
            return logits[:, :L_img]
        return logits[:, T - 1 : T - 1 + L_img]

    def _prefix_attention_mask(
        self,
        caption_mask: torch.Tensor,  # [B, T]
        *,
        total_len: int,
    ) -> torch.Tensor:
        """Rollout-equivalent attention mask for the teacher-forced pass.

        Mirrors upstream ``generate()``'s causal-mask patch: causal tril, all
        rows blocked from padded caption columns, then the diagonal restored
        (``causal_mask * (1 - eye) + eye``). Bool ``[B, 1, S, S]``.
        """
        B, T = caption_mask.shape
        device = caption_mask.device
        m = torch.tril(torch.ones(total_len, total_len, dtype=torch.bool, device=device))
        m = m.unsqueeze(0).expand(B, -1, -1).clone()
        m[:, :, :T] &= caption_mask[:, None, :].bool()
        m |= torch.eye(total_len, dtype=torch.bool, device=device)
        return m.unsqueeze(1)

    def _clear_kv_caches(self) -> None:
        """Detach the vendored per-layer static KV caches (rollout-only state)."""
        for block in self._gpt_trunk().layers:
            block.attention.kv_cache = None

    # ------------------------------------------------------------------
    # Replay forward — recompute logits at training time
    # ------------------------------------------------------------------

    def replay_forward(
        self,
        batch: Any,
        timestep_idx: int = 0,
        *,
        request: ReplayRequest | None = None,
    ) -> ReplayResult:
        """Single forward producing per-token logits over the image vocab.

        Reads T5 prompt ids/mask and sampled image tokens from
        ``batch.trajectory``, re-encodes the caption with the frozen T5 (so
        the conditioning is identical to rollout), and recomputes logits under
        the current model. AR has no denoising step, so only index zero is valid.
        """
        require_zero_replay_timestep(timestep_idx, owner=type(self).__name__)
        require_replay_segments(
            request,
            ("image_tokens",),
            owner=type(self).__name__,
        )
        from vrl.trajectory import TrajectoryResolver

        resolver = TrajectoryResolver.from_batch(batch)
        replay = resolver.replay_tensor_dict("image_tokens")
        prompt_ids = replay["prompt_input_ids"]
        prompt_mask = replay["prompt_attention_mask"]
        image_token_ids = resolver.role_value("image_tokens", "action")

        caption_embeds, caption_mask = self.encode_caption(prompt_ids, prompt_mask)
        logits = self.forward_image_logits(caption_embeds, caption_mask, image_token_ids)
        return ReplayResult(
            segments={
                "image_tokens": ReplaySegmentResult(
                    segment="image_tokens",
                    values={"logits": logits, "image_token_ids": image_token_ids},
                ),
            },
        )

    # ------------------------------------------------------------------
    # VQ decode — image tokens → pixels
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode_image_tokens(
        self,
        image_token_ids: torch.Tensor,  # [B, L_img]
        *,
        image_size: int | None = None,
    ) -> torch.Tensor:
        """Decode image-token grids to RGB pixels in ``[-1, 1]``.

        ``[B, L]`` ids -> ``decode_code`` with ``[B, e_dim, side, side]``
        (e_dim resolved from the live quantizer, mirroring the Janus lesson:
        never hardcode the codebook width). Returns
        ``[B, 3, side*downsample, side*downsample]``.
        """
        B, L = image_token_ids.shape
        side = int(L**0.5)
        if side * side != L:
            raise ValueError(f"expected square token grid, got L_img={L}")
        expected_size = side * self.config.downsample_size
        if image_size is not None and int(image_size) != expected_size:
            raise ValueError(
                f"image_size={image_size} inconsistent with {side}x{side} grid "
                f"at ds{self.config.downsample_size} (expected {expected_size})"
            )
        latent_dim = self._resolve_vq_codebook_dim()
        decoded = self.vq_model.decode_code(
            image_token_ids.to(device=self.device, dtype=torch.long),
            shape=[B, latent_dim, side, side],
        )
        return decoded.clamp(-1.0, 1.0)

    def _resolve_vq_codebook_dim(self) -> int:
        """Codebook entry width from the live quantizer (config as fallback)."""
        quant = getattr(self.vq_model, "quantize", None)
        emb = getattr(quant, "embedding", None) if quant is not None else None
        weight = getattr(emb, "weight", None) if emb is not None else None
        if weight is not None and weight.ndim == 2 and weight.shape[-1] > 0:
            return int(weight.shape[-1])
        return int(self.config.codebook_embed_dim)


class LlamaGenReplayModel(ARReplayRolloutStubs, LlamaGenModel):
    """Replay-only LlamaGen wrapper: GPT + T5 encoder, no VQ decoder.

    Trainer replay needs teacher-forced logits only; the VQ decoder is a
    rollout-side module, so it is never constructed here and
    ``decode_image_tokens`` raises via ``ARReplayRolloutStubs``.
    """

    def _loads_vq(self) -> bool:
        return False

    @property
    def vq_model(self) -> nn.Module:
        raise RuntimeError("LlamaGenReplayModel does not own a VQ decoder")


# ---------------------------------------------------------------------------
# Loaders — lazy imports so this module stays importable without the deps.
# ---------------------------------------------------------------------------


def _resolve_checkpoint_file(
    model_path: str,
    filename: str,
    *,
    revision: str | None,
) -> str:
    """Local dir join or HF hub download for one LlamaGen ``.pt`` file."""
    if os.path.isdir(model_path):
        path = os.path.join(model_path, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing LlamaGen checkpoint file: {path}")
        return path
    from huggingface_hub import hf_hub_download

    return hf_hub_download(model_path, filename=filename, revision=revision)


def _checkpoint_weights(checkpoint: Any, *, label: str) -> Any:
    """Pick the weight dict out of an upstream LlamaGen checkpoint payload."""
    if not isinstance(checkpoint, dict):
        return checkpoint
    # Same key preference order as upstream sample_t2i.py (plus "ema", which
    # some upstream tokenizer checkpoints use).
    for key in ("model", "module", "state_dict", "ema"):
        if key in checkpoint:
            return checkpoint[key]
    if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint
    raise RuntimeError(f"unrecognized {label} checkpoint layout: {sorted(checkpoint)[:5]}")


def _load_llamagen_gpt(config: LlamaGenConfig) -> nn.Module:
    from vrl.models.families.llamagen.vendor.gpt import GPT_models

    if config.gpt_model not in GPT_models:
        raise ValueError(
            f"unknown LlamaGen gpt_model {config.gpt_model!r}; known: {sorted(GPT_models)}"
        )
    # All stochastic-regularization probs are zeroed: RL replay must be
    # deterministic even if a trainer flips train() mode, and
    # class_dropout_prob only gates pretraining-time caption drop (the
    # uncond_embedding buffer exists and loads from the checkpoint regardless).
    gpt = GPT_models[config.gpt_model](
        block_size=int(config.image_token_num),
        cls_token_num=int(config.cls_token_num),
        model_type="t2i",
        caption_dim=int(config.caption_dim),
        vocab_size=LLAMAGEN_IMAGE_VOCAB_SIZE,
        token_dropout_p=0.0,
        attn_dropout_p=0.0,
        resid_dropout_p=0.0,
        ffn_dropout_p=0.0,
        drop_path_rate=0.0,
        class_dropout_prob=0.0,
    )

    ckpt_path = _resolve_checkpoint_file(
        config.model_path,
        config.gpt_ckpt,
        revision=config.revision,
    )
    # weights_only=False: upstream training checkpoints bundle non-tensor
    # metadata (argparse args) that strict weights-only loading rejects; the
    # file identity is pinned by config (repo + filename).
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = _checkpoint_weights(checkpoint, label="LlamaGen GPT")
    missing, unexpected = gpt.load_state_dict(state, strict=False)
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise RuntimeError(f"LlamaGen GPT checkpoint is missing keys: {preview}")
    if unexpected:
        logger.info("Ignoring %d non-module keys in GPT checkpoint", len(unexpected))
    del checkpoint, state

    dtype = resolve_torch_dtype(config.dtype)
    return gpt.to(device=config.device, dtype=dtype).eval()


def _load_llamagen_vq(config: LlamaGenConfig) -> nn.Module:
    from vrl.models.families.llamagen.vendor.vq_model import VQ_models

    vq_arch = f"VQ-{config.downsample_size}"
    if vq_arch not in VQ_models:
        raise ValueError(f"unknown LlamaGen VQ arch {vq_arch!r}; known: {sorted(VQ_models)}")
    vq = VQ_models[vq_arch](
        codebook_size=LLAMAGEN_IMAGE_VOCAB_SIZE,
        codebook_embed_dim=int(config.codebook_embed_dim),
    )
    ckpt_path = _resolve_checkpoint_file(
        config.model_path,
        config.vq_ckpt,
        revision=config.revision,
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    vq.load_state_dict(_checkpoint_weights(checkpoint, label="LlamaGen VQ"))
    del checkpoint
    # VQ decoder stays float32 (upstream keeps it fp32 next to a bf16 GPT).
    return vq.to(device=config.device, dtype=torch.float32).eval()


def _load_t5_encoder(config: LlamaGenConfig) -> tuple[nn.Module, Any]:
    """Frozen flan-t5-xl encoder + tokenizer, parked on CPU in float32.

    Prompt preprocessing note: upstream ``T5Embedder`` defaults to a heavy
    ftfy/BeautifulSoup caption-cleaning pipeline; this wrapper uses upstream's
    ``use_text_preprocessing=False`` branch (``lower().strip()``, applied by
    the executor) to avoid the extra dependencies.
    """
    from transformers import AutoTokenizer, T5EncoderModel

    revision_kwargs = {"revision": config.t5_revision} if config.t5_revision else {}
    tokenizer = AutoTokenizer.from_pretrained(config.t5_path, **revision_kwargs)
    encoder = T5EncoderModel.from_pretrained(
        config.t5_path,
        torch_dtype=torch.float32,
        **revision_kwargs,
    )
    return encoder.eval(), tokenizer


__all__ = [
    "LLAMAGEN_CAPTION_DIM",
    "LLAMAGEN_CAPTION_TOKEN_NUM",
    "LLAMAGEN_DOWNSAMPLE_SIZE",
    "LLAMAGEN_GPT_CKPT",
    "LLAMAGEN_HF_REPO",
    "LLAMAGEN_IMAGE_TOKEN_NUM",
    "LLAMAGEN_IMAGE_VOCAB_SIZE",
    "LLAMAGEN_T5_PATH",
    "LLAMAGEN_VQ_CKPT",
    "LlamaGenConfig",
    "LlamaGenModel",
    "LlamaGenReplayModel",
]
