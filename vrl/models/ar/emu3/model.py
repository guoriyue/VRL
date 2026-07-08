"""Emu3-Gen wrapper for autoregressive text-to-image RL.

Mirrors ``vrl.models.ar.janus_pro.JanusProModel`` for BAAI's Emu3 (the
HF-native ``Emu3ForConditionalGeneration``). The Janus-vs-Emu3 differences
this file isolates:

  * **No generation head.** Emu3 image-token logits are plain ``lm_head``
    outputs restricted to the image-token BPE columns — there is no separate
    ``gen_head`` projection. We restrict logits to a fixed *generation vocab*
    (image tokens + the four structural tokens, see below) so replay logits
    stay ``[B, L, V_gen]`` instead of the 184k-wide full vocab.

  * **Structural token grid.** Emu3 generates a ``height x (width + 1)`` BPE
    grid where column ``width`` of every row is a forced EOL token, followed
    by a forced EOF / EOI / EOS tail (upstream's ``prefix_allowed_tokens_fn``
    from the model card). ``emu3_forced_token_schedule`` /
    ``emu3_allowed_token_mask`` encode that constraint once; the rollout
    runner and train-time replay both apply it so log-probs stay in parity.

  * **Restricted-vocab action space.** Sampled token ids are indices into the
    generation vocab (``0..V_img-1`` image tokens, then EOL/EOF/EOI/EOS), not
    raw BPE ids. ``decode_image_tokens`` maps back to BPE ids and calls the
    upstream ``Emu3Model.decode_image_tokens(tokens, height, width)``, which
    expects the FULL structural sequence (``h*(w+1) + 3`` tokens: it strips
    the 3 tail tokens via ``[:, :-3]`` and the EOL column internally).

References
----------
HF modeling: ``transformers/models/emu3/modeling_emu3.py`` (Emu3Model.
decode_image_tokens, Emu3ImageVocabularyMapping) and the
``BAAI/Emu3-Gen-hf`` model-card ``prefix_allowed_tokens_fn``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from vrl.models.ar.base import ARModelBase, ARReplayRolloutStubs
from vrl.models.dtypes import resolve_torch_dtype
from vrl.models.interfaces import ReplayRequest, ReplayResult, ReplaySegmentResult
from vrl.models.utils import count_trainable_params
from vrl.utils.logging import init_logger

logger = init_logger(__name__)

# Structural special tokens around/inside the generated image grid. Names are
# checkpoint vocabulary keys (present in ``config.vocabulary_map`` of the
# HF-format checkpoints), NOT display strings.
EMU3_EOL_TOKEN = "<|extra_200|>"   # end-of-line: forced at column `width` of each row
EMU3_EOF_TOKEN = "<|extra_201|>"   # end-of-frame: forced right after the grid
EMU3_EOI_TOKEN = "<|image end|>"   # end-of-image: forced after EOF
# Tail after the h*(w+1) grid: EOF, EOI, EOS.
EMU3_TAIL_TOKEN_NUM = 3
# Structural columns appended after the image tokens in the generation vocab:
# EOL, EOF, EOI, EOS (in that order).
EMU3_STRUCTURAL_TOKEN_NUM = 4


@dataclass(slots=True)
class Emu3Config:
    """Hyper-parameters for the Emu3 wrapper (defaults target Emu3-Gen ~9B)."""

    model_path: str = "BAAI/Emu3-Gen-hf"
    dtype: str = "bfloat16"           # "bfloat16" | "float16" | "float32"

    # LoRA — Emu3's text model uses LLaMA-style projection names.
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
    )
    lora_init: str = "gaussian"       # PEFT ``init_lora_weights``

    # Generation defaults — used by the AR runtime runner.
    guidance_scale: float = 3.0
    temperature: float = 1.0
    # Target pixel area for the generated image; the processor derives the
    # latent grid (height, width) from it: 262144 = 512x512 -> a 64x64 grid.
    image_area: int = 262_144
    ratio: str = "1:1"

    # Misc
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Structural-constraint schedule (pure functions — shared by runner + replay)
# ---------------------------------------------------------------------------


def emu3_grid_token_num(height: int, width: int) -> int:
    """Total generated sequence length for a ``height x width`` latent grid.

    Each of the ``height`` rows carries ``width`` image tokens plus a forced
    EOL, followed by the forced EOF/EOI/EOS tail.
    """

    if height < 1 or width < 1:
        raise ValueError(f"latent grid dims must be >= 1, got {height}x{width}")
    return height * (width + 1) + EMU3_TAIL_TOKEN_NUM


def emu3_forced_token_schedule(
    height: int,
    width: int,
    num_image_tokens: int,
) -> torch.Tensor:
    """Per-position forced generation-vocab index; ``-1`` marks free positions.

    Mirrors the official ``prefix_allowed_tokens_fn`` (Emu3-Gen model card),
    re-based to 0-indexed generated positions ``j``:

      * ``j % (width+1) == width`` inside the grid -> EOL
      * ``j == height*(width+1)``                  -> EOF
      * ``j == height*(width+1) + 1``              -> EOI
      * ``j == height*(width+1) + 2``              -> EOS
      * otherwise the position is free (image-token vocab only).

    Forced values are *generation-vocab* indices: the structural columns sit
    right after the ``num_image_tokens`` image columns, ordered
    EOL, EOF, EOI, EOS.
    """

    if num_image_tokens < 1:
        raise ValueError("num_image_tokens must be >= 1")
    total = emu3_grid_token_num(height, width)
    eol = num_image_tokens
    eof = num_image_tokens + 1
    eoi = num_image_tokens + 2
    eos = num_image_tokens + 3
    forced = torch.full((total,), -1, dtype=torch.long)
    positions = torch.arange(total)
    grid_len = height * (width + 1)
    forced[(positions % (width + 1) == width) & (positions < grid_len)] = eol
    forced[grid_len] = eof
    forced[grid_len + 1] = eoi
    forced[grid_len + 2] = eos
    return forced


def emu3_allowed_token_mask(
    forced_gen_index: torch.Tensor,
    num_image_tokens: int,
) -> torch.Tensor:
    """Boolean ``[L, V_gen]`` mask of legal generation-vocab columns per position.

    Free positions (``forced == -1``) allow only the image-token columns;
    forced positions allow exactly the forced structural column. The SAME mask
    feeds both rollout sampling and replay logits, so the old/new log-prob
    distributions cannot drift apart structurally.
    """

    if forced_gen_index.ndim != 1:
        raise ValueError("forced_gen_index must be a 1-D position schedule")
    vocab = num_image_tokens + EMU3_STRUCTURAL_TOKEN_NUM
    length = forced_gen_index.shape[0]
    allowed = torch.zeros(length, vocab, dtype=torch.bool, device=forced_gen_index.device)
    free = forced_gen_index < 0
    allowed[free, :num_image_tokens] = True
    forced_rows = (~free).nonzero(as_tuple=True)[0]
    allowed[forced_rows, forced_gen_index[forced_rows]] = True
    return allowed


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class Emu3Model(ARModelBase):
    """Train-and-sample wrapper for Emu3 text-to-image generation.

    Keeps the LoRA-wrapped text model + frozen lm_head / VQ decoder in a
    single ``nn.Module`` so it integrates cleanly with the rollout sync path.
    """

    def __init__(
        self,
        config: Emu3Config | None = None,
        *,
        emu3: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        """Construct the wrapper.

        Args:
          config: hyper-parameters. ``None`` -> defaults for Emu3-Gen-hf.
          emu3: optional pre-loaded ``Emu3ForConditionalGeneration`` (saves
            the ~18 GB checkpoint download in tests). When ``None``, we load
            from ``config.model_path``.
          processor: optional pre-loaded ``Emu3Processor``.
        """
        super().__init__()
        self.config = config or Emu3Config()

        if emu3 is None:
            emu3, processor = _load_emu3_from_pretrained(self.config)
        elif processor is None:
            raise ValueError("Must pass `processor` when `emu3` is provided")

        self._processor = processor

        # Freeze everything by default — LoRA wrap re-enables only attention
        # projections in the text model.
        for p in emu3.parameters():
            p.requires_grad_(False)

        self.emu3 = emu3
        self._require_surface(("model", "lm_head"))
        self._require_inner_surface(("text_model", "vocabulary_mapping", "vqmodel"))

        if self.config.use_lora:
            self._apply_lora()

        self._init_gen_vocab()

    # ------------------------------------------------------------------
    # Sub-module accessors
    # ------------------------------------------------------------------

    def _require_surface(self, attrs: tuple[str, ...]) -> None:
        for attr in attrs:
            if not hasattr(self.emu3, attr):
                raise RuntimeError(
                    f"Loaded model is missing `{attr}` — does not look like "
                    "an Emu3ForConditionalGeneration checkpoint."
                )

    def _require_inner_surface(self, attrs: tuple[str, ...]) -> None:
        for attr in attrs:
            if not hasattr(self.emu3.model, attr):
                raise RuntimeError(
                    f"Loaded model is missing `model.{attr}` — does not look "
                    "like an Emu3ForConditionalGeneration checkpoint."
                )

    @property
    def processor(self) -> Any:
        return self._processor

    @property
    def language_model(self) -> nn.Module:
        """The Emu3TextModel trunk (PEFT-wrapped when LoRA is attached)."""

        return self.emu3.model.text_model

    @property
    def vocabulary_mapping(self) -> Any:
        return self.emu3.model.vocabulary_mapping

    @property
    def vq_model(self) -> nn.Module:
        return self.emu3.model.vqmodel

    @property
    def text_config(self) -> Any:
        return self.emu3.config.text_config

    @property
    def device(self) -> torch.device:
        return next(self.emu3.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.emu3.parameters()).dtype

    def trainable_param_count(self) -> int:
        return count_trainable_params(self.emu3)

    def _lm_trunk(self) -> nn.Module:
        """Return the Emu3TextModel trunk that emits ``last_hidden_state``.

        Unlike Janus (whose ``language_model`` is a ForCausalLM that projects
        to text logits), Emu3's text model IS the bare trunk — there is no
        lm_head inside it. We still peel the PEFT wrapper so attention
        backends drive the raw HF module directly.
        """
        lm = self.language_model
        peft_inner = getattr(lm, "base_model", None)
        if (
            peft_inner is not None
            and hasattr(peft_inner, "model")
            and peft_inner.model is not lm
        ):
            return peft_inner.model
        return lm

    # ------------------------------------------------------------------
    # LoRA / reference-policy helpers
    # ------------------------------------------------------------------

    def _apply_lora(self) -> None:
        """Attach a PEFT LoRA adapter to the text-model trunk."""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "PEFT is required for use_lora=True. pip install peft>=0.12"
            ) from e

        lora_cfg = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            init_lora_weights=self.config.lora_init,
            target_modules=list(self.config.lora_target_modules),
            bias="none",
        )
        # Wrap ONLY the text model — lm_head / VQ stay frozen. Note the
        # top-level ``emu3.text_model`` is a read-only property; the real
        # submodule lives at ``emu3.model.text_model``.
        self.emu3.model.text_model = get_peft_model(
            self.emu3.model.text_model, lora_cfg,
        )
        logger.info(
            "Applied LoRA (rank=%d, alpha=%d) to the Emu3 text model.",
            self.config.lora_rank, self.config.lora_alpha,
        )

    @property
    def has_lora_adapter(self) -> bool:
        """True iff this wrapper carries a real PEFT adapter we can disable."""
        lm = self.language_model
        return hasattr(lm, "disable_adapter") and callable(lm.disable_adapter)

    # ------------------------------------------------------------------
    # Generation vocab — image tokens + EOL/EOF/EOI/EOS structural columns
    # ------------------------------------------------------------------

    def _init_gen_vocab(self) -> None:
        mapping = self.vocabulary_mapping
        image_tokens = list(mapping.image_tokens)
        if not image_tokens:
            raise RuntimeError(
                "Emu3 vocabulary_map carries no `<|visual token ...|>` entries",
            )
        self.image_vocab_size = len(image_tokens)
        eol = mapping.eol_token_id
        if eol is None:
            raise RuntimeError(
                f"Emu3 vocabulary_map is missing the EOL token {EMU3_EOL_TOKEN!r}",
            )
        bpe_ids = [
            *image_tokens,
            int(eol),
            self._structural_token_id(EMU3_EOF_TOKEN),
            self._structural_token_id(EMU3_EOI_TOKEN),
            self._eos_token_id(),
        ]
        # Plain attribute (NOT a buffer): the mapping is derived config data,
        # not a weight — keeping it out of state_dict keeps the trainable-sync
        # payload clean. Lazily re-homed to the compute device on first use.
        self._gen_vocab_bpe_ids = torch.tensor(
            bpe_ids, dtype=torch.long, device=self.device,
        )

    def _structural_token_id(self, token: str) -> int:
        """Resolve a structural token id from the checkpoint vocabulary map.

        The HF-format checkpoints ship the FULL tokenizer vocab in
        ``config.vocabulary_map`` (verified on BAAI/Emu3-Gen-hf), so the map
        alone suffices — the replay model has no tokenizer. The tokenizer
        fallback covers hypothetical slimmed-down maps on the rollout path.
        """
        value = self.vocabulary_mapping.vocab_map.get(token)
        if value is not None:
            return int(value)
        tokenizer = getattr(self._processor_or_none(), "tokenizer", None)
        if tokenizer is not None:
            resolved = tokenizer.convert_tokens_to_ids(token)
            if resolved is not None:
                return int(resolved)
        raise RuntimeError(
            f"Could not resolve Emu3 structural token {token!r} from the "
            "checkpoint vocabulary_map or tokenizer.",
        )

    def _eos_token_id(self) -> int:
        eos = getattr(self.text_config, "eos_token_id", None)
        if isinstance(eos, (list, tuple)):
            eos = eos[0] if eos else None
        if eos is None:
            tokenizer = getattr(self._processor_or_none(), "tokenizer", None)
            eos = getattr(tokenizer, "eos_token_id", None)
        if eos is None:
            raise RuntimeError("Could not resolve the Emu3 EOS token id")
        return int(eos)

    def _processor_or_none(self) -> Any:
        # The replay subclass raises on `.processor`; id resolution must not.
        return getattr(self, "_processor", None)

    def _gen_ids_on(self, device: torch.device) -> torch.Tensor:
        ids = self._gen_vocab_bpe_ids
        if ids.device != device:
            ids = ids.to(device)
            self._gen_vocab_bpe_ids = ids
        return ids

    def image_gen_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden states to generation-vocab logits.

        Emu3's image logits are plain ``lm_head`` outputs (no gen_head);
        restricting to the generation-vocab columns is what keeps replay
        logits ``[B, L, V_gen]`` instead of the 184k full vocab.
        """
        logits = self.emu3.lm_head(hidden)
        return logits.index_select(-1, self._gen_ids_on(logits.device))

    def embed_gen_tokens(self, gen_token_ids: torch.Tensor) -> torch.Tensor:
        """Embed generation-vocab token ids via the text-model embedding."""
        bpe_ids = self._gen_ids_on(gen_token_ids.device)[gen_token_ids]
        return self.language_model.get_input_embeddings()(bpe_ids)

    # ------------------------------------------------------------------
    # Prompt encoding — generation-mode processor call (cond + uncond)
    # ------------------------------------------------------------------

    def encode_generation_prompts(
        self,
        prompts: list[str],
        *,
        max_text_length: int,
        image_area: int | None = None,
        ratio: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        """Tokenize prompts in Emu3 image-generation mode.

        Uses ``processor(..., return_for_image_generation=True)`` which
        appends the ``<|image start|>{h}*{w}<|image token|>`` grid prefix and
        returns the latent grid dims. Prompts are LEFT-padded to
        ``max_text_length`` (the runner reads the prefill hidden at position
        -1, so the last position must be the real image-wrapper token).

        Returns ``(input_ids, attention_mask, (height, width))`` on the model
        device.
        """
        inputs = self.processor(
            text=list(prompts),
            return_for_image_generation=True,
            image_area=int(image_area if image_area is not None else self.config.image_area),
            ratio=str(ratio if ratio is not None else self.config.ratio),
            padding="max_length",
            max_length=int(max_text_length),
            return_tensors="pt",
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        if ids.shape[1] > max_text_length:
            raise ValueError(
                f"Emu3 generation prompt is {ids.shape[1]} tokens, which exceeds "
                f"max_text_length={max_text_length}. Truncation would cut the "
                "image-generation suffix, so raise instead.",
            )
        height, width = (int(dim) for dim in inputs["image_sizes"][0])
        return ids.to(self.device), mask.to(self.device), (height, width)

    # ------------------------------------------------------------------
    # Train-time forward — generation-vocab logits (teacher-forced)
    # ------------------------------------------------------------------

    def forward_image_logits(
        self,
        prompt_inputs_embeds: torch.Tensor,    # [B, L_text, H]
        prompt_attention_mask: torch.Tensor,   # [B, L_text]
        image_token_ids: torch.Tensor,         # [B, L_img] generation-vocab ids
    ) -> torch.Tensor:
        """One teacher-forced pass returning per-position gen-vocab logits.

        Layout convention matches Janus: text first, then the structural
        image sequence; logits are read at the positions that *predict* each
        token (the position immediately before it). Returns raw (unmasked)
        logits ``[B, L_img, V_gen]`` — ``replay_forward`` applies the
        structural mask because the mask needs the latent grid dims.
        """
        B, L_img = image_token_ids.shape
        img_embeds = self.embed_gen_tokens(image_token_ids)  # [B, L_img, H]

        # Concat: [text | image[:-1]] — image[-1] doesn't predict anything new
        inputs_embeds = torch.cat(
            [prompt_inputs_embeds, img_embeds[:, :-1, :]], dim=1,
        )
        L_text = prompt_inputs_embeds.shape[1]
        attn = torch.cat(
            [
                prompt_attention_mask,
                torch.ones(
                    B, L_img - 1,
                    dtype=prompt_attention_mask.dtype,
                    device=prompt_attention_mask.device,
                ),
            ],
            dim=1,
        )

        outputs = self._lm_trunk()(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            use_cache=False,
        )
        hidden = outputs.last_hidden_state  # [B, L_text + L_img - 1, H]
        gen_hidden = hidden[:, L_text - 1 : L_text - 1 + L_img, :]
        return self.image_gen_logits(gen_hidden)

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
        """Single forward producing per-token logits over the generation vocab.

        Mirrors Janus' replay contract: read prompt ids/masks and sampled
        tokens from ``batch.trajectory``, recompute logits under the current
        model, return ``{"logits", "image_token_ids"}``. The structural mask
        is applied here (with grid dims from the rollout context) so the
        evaluator's plain ``log_softmax`` reproduces the rollout's masked
        conditional distribution — the old/new log-prob parity invariant.

        AR has no notion of "denoising step", so ``timestep_idx`` is ignored.
        """
        del timestep_idx, request
        from vrl.trajectory import TrajectoryResolver

        resolver = TrajectoryResolver.from_batch(batch)
        replay = resolver.replay_tensor_dict("image_tokens")
        prompt_ids = replay["prompt_input_ids"]
        prompt_mask = replay["prompt_attention_mask"]
        image_token_ids = resolver.role_value("image_tokens", "action")

        embed = self.language_model.get_input_embeddings()
        prompt_embeds = embed(prompt_ids)
        logits = self.forward_image_logits(
            prompt_embeds, prompt_mask, image_token_ids,
        )  # [B, L_img, V_gen]

        height, width = self._replay_grid_dims(batch, int(image_token_ids.shape[1]))
        forced = emu3_forced_token_schedule(
            height, width, self.image_vocab_size,
        ).to(logits.device)
        allowed = emu3_allowed_token_mask(forced, self.image_vocab_size)
        logits = logits.masked_fill(~allowed.unsqueeze(0), float("-inf"))
        return ReplayResult(
            segments={
                "image_tokens": ReplaySegmentResult(
                    segment="image_tokens",
                    values={"logits": logits, "image_token_ids": image_token_ids},
                ),
            },
        )

    def _replay_grid_dims(self, batch: Any, token_count: int) -> tuple[int, int]:
        """Read the latent grid dims the rollout generated with.

        The executor stores ``image_height``/``image_width`` in the trajectory
        context (they are NOT derivable from the token count alone for
        non-square ratios).
        """
        context = getattr(batch, "context", None) or {}
        trajectory = getattr(batch, "trajectory", None)
        if not context and trajectory is not None:
            context = getattr(trajectory, "context", None) or {}
        height = context.get("image_height")
        width = context.get("image_width")
        if height is None or width is None:
            raise RuntimeError(
                "Emu3 replay requires image_height/image_width in the rollout "
                "context to rebuild the structural token mask.",
            )
        height, width = int(height), int(width)
        expected = emu3_grid_token_num(height, width)
        if expected != token_count:
            raise RuntimeError(
                f"Emu3 replay token count {token_count} does not match the "
                f"{height}x{width} grid (expected {expected}).",
            )
        return height, width

    # ------------------------------------------------------------------
    # VQ decode — generation-vocab tokens -> pixels
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode_image_tokens(
        self,
        image_token_ids: torch.Tensor,    # [B, L_total] generation-vocab ids
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Decode the full structural token sequence to RGB pixels in [-1, 1].

        Maps generation-vocab ids back to BPE ids and delegates to the
        upstream ``Emu3Model.decode_image_tokens``, which expects the tail
        (EOF/EOI/EOS) present: it slices ``[:, :-3]``, reshapes to
        ``(-1, height, width + 1)``, and strips the EOL column internally.
        """
        _, L = image_token_ids.shape
        expected = emu3_grid_token_num(height, width)
        if expected != L:
            raise ValueError(
                f"decode_image_tokens expects the full structural sequence of "
                f"{expected} tokens for a {height}x{width} grid, got {L}",
            )
        bpe_ids = self._gen_ids_on(image_token_ids.device)[image_token_ids]
        image = self.emu3.model.decode_image_tokens(
            bpe_ids, height=height, width=width,
        )
        return image.clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# Replay-only model — text model + lm_head, no VQ decoder / processor
# ---------------------------------------------------------------------------


class Emu3ReplayCore(nn.Module):
    """Minimal Emu3 module set needed for trainer replay.

    ``Emu3ForConditionalGeneration.__init__`` constructs the VQVAE; replay
    only needs teacher-forced generation-vocab logits, so this core keeps the
    text trunk + lm_head + vocabulary mapping. The submodule paths mirror the
    HF layout (``model.text_model.*`` / ``lm_head.*``) so the checkpoint
    state dict loads by name (verified against BAAI/Emu3-Gen-hf's
    ``model.safetensors.index.json``).
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        from transformers.models.emu3.modeling_emu3 import (
            Emu3ImageVocabularyMapping,
            Emu3TextModel,
        )

        self.config = config
        inner = nn.Module()
        inner.text_model = Emu3TextModel._from_config(config.text_config)
        inner.vocabulary_mapping = Emu3ImageVocabularyMapping(config.vocabulary_map)
        self.model = inner
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )


class Emu3ReplayModel(ARReplayRolloutStubs, Emu3Model):
    """Replay-only Emu3 wrapper without processor or VQ decoder."""

    def __init__(
        self,
        config: Emu3Config | None = None,
        *,
        emu3: Any | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config or Emu3Config()
        if emu3 is None:
            emu3 = _load_emu3_replay_core_from_pretrained(self.config)

        for p in emu3.parameters():
            p.requires_grad_(False)

        self.emu3 = emu3
        self._require_surface(("model", "lm_head"))
        self._require_inner_surface(("text_model", "vocabulary_mapping"))
        if hasattr(self.emu3.model, "vqmodel"):
            raise RuntimeError(
                "Emu3 replay model unexpectedly loaded the VQ decoder",
            )

        if self.config.use_lora:
            self._apply_lora()

        self._init_gen_vocab()

    @property
    def processor(self) -> Any:
        raise RuntimeError("Emu3ReplayModel does not own an Emu3Processor")

    @property
    def vq_model(self) -> nn.Module:
        raise RuntimeError("Emu3ReplayModel does not own a VQ decoder")

    @property
    def text_config(self) -> Any:
        return self.emu3.config.text_config


# ---------------------------------------------------------------------------
# Loaders — lazy imports so this module imports without model downloads.
# ---------------------------------------------------------------------------


def _load_emu3_from_pretrained(config: Emu3Config) -> tuple[Any, Any]:
    """Load ``Emu3ForConditionalGeneration`` + ``Emu3Processor`` from disk/HF."""

    from transformers import Emu3ForConditionalGeneration, Emu3Processor

    dtype = resolve_torch_dtype(config.dtype)
    processor = Emu3Processor.from_pretrained(config.model_path)
    # The prefill runner reads the last-position hidden state as "next token
    # context", so padded prompts must be LEFT-padded (GPT2 tokenizers default
    # to right padding).
    processor.tokenizer.padding_side = "left"
    emu3 = Emu3ForConditionalGeneration.from_pretrained(
        config.model_path,
        dtype=dtype,
    )
    emu3 = emu3.to(device=config.device).eval()
    return emu3, processor


def _load_emu3_replay_core_from_pretrained(config: Emu3Config) -> Emu3ReplayCore:
    """Load Emu3 replay modules without constructing the VQ decoder.

    Same load shape as the janus_pro replay loader (config-init the minimal
    module set, then strict-check a ``strict=False`` checkpoint load against
    the core's own keys).
    """

    from transformers import AutoConfig

    dtype = resolve_torch_dtype(config.dtype)
    model_config = AutoConfig.from_pretrained(config.model_path)
    core = Emu3ReplayCore(model_config)
    checkpoint_dir = _resolve_hf_checkpoint_dir(config.model_path)
    missing_keys = _load_emu3_replay_checkpoint(core, checkpoint_dir)
    missing_replay = sorted(set(missing_keys) & set(core.state_dict()))
    if missing_replay:
        preview = ", ".join(missing_replay[:5])
        suffix = " ..." if len(missing_replay) > 5 else ""
        raise RuntimeError(f"Emu3 replay checkpoint is missing keys: {preview}{suffix}")
    return core.to(device=config.device, dtype=dtype).eval()


def _resolve_hf_checkpoint_dir(model_path: str) -> str:
    import os

    if os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download

    return snapshot_download(model_path)


def _load_emu3_replay_checkpoint(model: nn.Module, checkpoint_dir: str) -> list[str]:
    import os

    from transformers.modeling_utils import load_sharded_checkpoint, load_state_dict

    for name in (
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
    ):
        if os.path.exists(os.path.join(checkpoint_dir, name)):
            result = load_sharded_checkpoint(model, checkpoint_dir, strict=False)
            return list(getattr(result, "missing_keys", result[0] if result else []))

    for name in (
        "model.safetensors",
        "pytorch_model.bin",
    ):
        path = os.path.join(checkpoint_dir, name)
        if os.path.exists(path):
            state = load_state_dict(path)
            missing, _unexpected = model.load_state_dict(state, strict=False)
            return list(missing)

    raise FileNotFoundError(
        f"No supported Emu3 checkpoint file found in {checkpoint_dir}",
    )


__all__ = [
    "EMU3_EOF_TOKEN",
    "EMU3_EOI_TOKEN",
    "EMU3_EOL_TOKEN",
    "EMU3_STRUCTURAL_TOKEN_NUM",
    "EMU3_TAIL_TOKEN_NUM",
    "Emu3Config",
    "Emu3Model",
    "Emu3ReplayCore",
    "Emu3ReplayModel",
    "emu3_allowed_token_mask",
    "emu3_forced_token_schedule",
    "emu3_grid_token_num",
]
