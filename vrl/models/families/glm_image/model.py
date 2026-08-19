"""GLM-Image wrapper for autoregressive text-to-image RL.

GLM-Image (zai-org/GLM-Image) is a HYBRID: a 9B AR model
(``GlmImageForConditionalGeneration``, checkpoint subfolder
``vision_language_encoder``) generates discrete "prior" image tokens, then a
FROZEN 7B diffusion DiT (``GlmImageTransformer2DModel`` + VAE + ByT5 glyph
encoder) decodes them to pixels. RL trains ONLY the AR section (LoRA); the
diffusion decode is pure postprocess — this family's "VQ decode" happens to be
a flow-matching pipeline segment (``diffusers`` ``GlmImagePipeline``).

The Emu3-vs-GLM differences this file isolates:

  * **Codebook-prefix generation vocab.** ``lm_head`` projects to
    ``vision_vocab_size`` (16512) only — image codebook ids 0..16383 plus the
    image_start/image_end specials. Image token ids double as text-vocab
    embedding rows (``embed_tokens`` rows 0..16383), so no BPE remap exists:
    restricting logits is a plain ``[..., :codebook]`` slice and embedding a
    sampled token is a direct ``embed_tokens`` lookup.

  * **No structural tokens inside the raster.** The AR emits a plain raster:
    ``prev_h*prev_w`` preview-grid tokens, then ``token_h*token_w`` large-grid
    tokens, then one EOS (= image_end, 16385). Every generated position is a
    free codebook draw; there is no EOL column or forced tail. We deliberately
    do NOT sample the trailing EOS: the decode segment consumes only the large
    grid, and our loop's fixed length (``glm_image_token_num``) already
    terminates the sequence (the reference stops on max_new_tokens /
    eos_token_id=16385 — see ``GlmImagePipeline.generate_prior_tokens``,
    diffusers pipeline_glm_image.py lines 234-243 and 439-443).

  * **3-axis mrope decode positions.** Generated image tokens carry 2D
    spatial position ids per grid (temporal constant, height/width raster),
    precomputed by the reference ``GlmImageModel.get_rope_index``
    (modeling_glm_image.py lines 1084-1123). ``glm_image_decode_position_schedule``
    reproduces that schedule as a pure function; rollout and replay both use
    it so log-probs stay in parity.

  * **No AR-side CFG.** The reference AR generation is plain
    ``generate(..., do_sample=True)`` with temperature=0.9 / top_p=0.75 from
    the checkpoint ``generation_config.json`` — no guidance logits processor.
    The rollout runner is single-sequence; the only CFG in this family lives
    inside the frozen DiT decode (``decode_guidance_scale``).

References
----------
HF modeling: ``transformers/models/glm_image/modeling_glm_image.py``
(GlmImageForConditionalGeneration, GlmImageTextModel, get_rope_index).
Processor: ``transformers/models/glm_image/processing_glm_image.py``
(_build_prompt_with_target_shape). Decode:
``diffusers/pipelines/glm_image/pipeline_glm_image.py`` (__call__ with
``prior_token_ids``, _upsample_token_ids).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from vrl.models.dtypes import resolve_torch_dtype
from vrl.models.families.glm_image.config import GlmImageConfig
from vrl.models.interfaces import (
    ReplayRequest,
    ReplayResult,
    replay_context_image_size,
    single_segment_result,
)
from vrl.models.steps.token.base import (
    ARModelBase,
    ARReplayCore,
    ARReplayRolloutStubs,
)
from vrl.models.steps.token.lora import install_token_lora_adapter
from vrl.models.steps.token.vocab_head import VocabHeadSplit, head_replay_values
from vrl.utils.logging import init_logger

logger = init_logger(__name__)

# Latent downsample factor: image pixels per AR token cell edge. Checkpoint
# architecture constant (32x total downsampling, processing_glm_image.py
# ``factor = 32``).
GLM_IMAGE_LATENT_FACTOR = 32

# Checkpoint subfolders inside the zai-org/GLM-Image repo (model_index.json).
GLM_IMAGE_AR_SUBFOLDER = "vision_language_encoder"
GLM_IMAGE_PROCESSOR_SUBFOLDER = "processor"


# ---------------------------------------------------------------------------
# Grid / position-schedule math (pure functions — shared by runner + replay)
# ---------------------------------------------------------------------------


def glm_image_grid_dims(height: int, width: int) -> tuple[int, int, int, int]:
    """Return ``(token_h, token_w, prev_h, prev_w)`` for a pixel-size target.

    Mirrors ``GlmImageProcessor._build_prompt_with_target_shape``
    (processing_glm_image.py lines 218-225): the large grid is the 32x
    downsample; the preview grid preserves the aspect ratio at ~16x16 cells.
    Rejects non-multiple-of-32 sizes instead of silently flooring, so replay
    grid dims recomputed from the stored pixel size can never drift from what
    the rollout actually generated with.
    """

    factor = GLM_IMAGE_LATENT_FACTOR
    if height < factor or width < factor or height % factor or width % factor:
        raise ValueError(
            f"GLM-Image image size must be positive multiples of {factor}, got {height}x{width}",
        )
    token_h = height // factor
    token_w = width // factor
    ratio = token_h / token_w
    prev_h = int(math.sqrt(ratio) * (factor // 2))
    prev_w = int(math.sqrt(1 / ratio) * (factor // 2))
    return token_h, token_w, prev_h, prev_w


def glm_image_token_num(height: int, width: int) -> int:
    """Total generated AR sequence length for a pixel-size target.

    Generation order (reference ``get_rope_index`` decode schedule walks the
    grid list back-to-front): preview raster first, then the large raster.
    The reference generates one extra EOS/image_end token after the large
    grid (``max_new_tokens = sum(grid_sizes) + 1``); we do not sample it —
    the decode segment never consumes it and our fixed-length loop already
    terminates the sequence.
    """

    token_h, token_w, prev_h, prev_w = glm_image_grid_dims(height, width)
    return prev_h * prev_w + token_h * token_w


def glm_image_decode_position_schedule(
    prompt_len: int,
    grids: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    """Per-generated-token 3-axis mrope position ids, ``[3, L_decode + 1]``.

    Pure reimplementation of the decode-stage schedule that the reference
    ``GlmImageModel.get_rope_index`` caches in ``_cached_decode_position_ids``
    (modeling_glm_image.py lines 1084-1123), verified equal in
    tests/models/families/glm_image/test_token_schedule.py:

      * grids are consumed in REVERSE order (t2i grids are
        ``[(token_h, token_w), (prev_h, prev_w)]``, so the preview raster is
        generated first);
      * within a grid: temporal axis constant at the grid base position,
        height/width axes are ``base + row`` / ``base + col`` rasters;
      * the base advances by ``max(h, w)`` per grid;
      * one trailing end-marker entry (all three axes at the final base) for
        the EOS token — kept for reference parity, unused by our loop.

    The schedule is translation-equivariant in ``prompt_len``
    (``schedule(n) == schedule(0) + n``), which the runner exploits for
    per-row valid-length offsets.
    """

    if prompt_len < 0:
        raise ValueError("prompt_len must be >= 0")
    if not grids:
        raise ValueError("grids must be non-empty")
    temporal: list[torch.Tensor] = []
    heights: list[torch.Tensor] = []
    widths: list[torch.Tensor] = []
    pos = int(prompt_len)
    for h, w in reversed(grids):
        if h < 1 or w < 1:
            raise ValueError(f"grid dims must be >= 1, got {h}x{w}")
        h_idx = torch.arange(h).unsqueeze(1).expand(h, w).flatten()
        w_idx = torch.arange(w).unsqueeze(0).expand(h, w).flatten()
        temporal.append(torch.full((h * w,), pos, dtype=torch.long))
        heights.append(pos + h_idx)
        widths.append(pos + w_idx)
        pos += max(h, w)
    end = torch.tensor([pos], dtype=torch.long)
    temporal.append(end)
    heights.append(end.clone())
    widths.append(end.clone())
    return torch.stack(
        [torch.cat(temporal), torch.cat(heights), torch.cat(widths)],
    )


def glm_image_prefill_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """3-axis position ids for a text-only (t2i) prompt, ``[3, B, L]``.

    Mirrors the reference ``get_rope_index`` prefill branch for a prompt with
    no complete images (modeling_glm_image.py lines 1011-1082): valid tokens
    get ``arange`` text positions replicated on all three axes; padding slots
    keep the init value 1 (attention-masked, never read).
    """

    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be [B, L]")
    batch, length = attention_mask.shape
    position_ids = torch.ones(
        3,
        batch,
        length,
        dtype=torch.long,
        device=attention_mask.device,
    )
    for row in range(batch):
        valid = attention_mask[row] == 1
        count = int(valid.sum())
        text_positions = torch.arange(
            count,
            dtype=torch.long,
            device=attention_mask.device,
        )
        position_ids[:, row, valid] = text_positions.unsqueeze(0).expand(3, -1)
    return position_ids


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class GlmImageModel(ARModelBase):
    """Train-and-sample wrapper for GLM-Image text-to-image generation.

    Owns the LoRA-wrapped 9B AR model; the frozen DiT decode stack (glyph
    text_encoder + transformer + VAE + scheduler) is loaded lazily on the
    first ``decode_image_tokens`` call and parked on CPU between decodes.
    """

    checkpoint_description = "a GlmImageForConditionalGeneration checkpoint"

    def __init__(
        self,
        config: GlmImageConfig | None = None,
        *,
        glm: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        """Construct the wrapper.

        Args:
          config: hyper-parameters. ``None`` -> defaults for zai-org/GLM-Image.
          glm: optional pre-loaded ``GlmImageForConditionalGeneration`` (saves
            the ~19 GB checkpoint download in tests). When ``None``, we load
            from ``config.model_path`` subfolder ``vision_language_encoder``.
          processor: optional pre-loaded ``GlmImageProcessor``.
        """
        super().__init__()
        self.config = config or GlmImageConfig()

        if glm is None:
            glm, processor = _load_glm_image_from_pretrained(self.config)
        elif processor is None:
            raise ValueError("Must pass `processor` when `glm` is provided")

        self._processor = processor
        # Lazy frozen decode stack (GlmImagePipeline without the AR model).
        self._decode_pipeline: Any | None = None

        # Freeze everything by default — LoRA wrap re-enables only attention
        # projections in the text trunk.
        for p in glm.parameters():
            p.requires_grad_(False)

        self.glm = glm
        self._require_surface(("model", "lm_head"))
        self._require_inner_surface(("language_model",))

        if self.config.use_lora:
            self._apply_lora()

        self._init_gen_vocab()

    # ------------------------------------------------------------------
    # Sub-module accessors
    # ------------------------------------------------------------------

    def _require_surface(self, attrs: tuple[str, ...]) -> None:
        self._require_module_attrs(self.glm, attrs)

    def _require_inner_surface(self, attrs: tuple[str, ...]) -> None:
        self._require_module_attrs(self.glm.model, attrs, prefix="model.")

    @property
    def processor(self) -> Any:
        return self._processor

    @property
    def language_model(self) -> nn.Module:
        """The GlmImageTextModel trunk (PEFT-wrapped when LoRA is attached)."""

        return self.glm.model.language_model

    @property
    def text_config(self) -> Any:
        return self.glm.config.text_config

    @property
    def device(self) -> torch.device:
        return next(self.glm.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.glm.parameters()).dtype

    # ------------------------------------------------------------------
    # LoRA / reference-policy helpers
    # ------------------------------------------------------------------

    def _apply_lora(self) -> None:
        """Attach a PEFT LoRA adapter to the text-model trunk."""
        # Wrap ONLY the text trunk — lm_head / vision tower / VQ encoder stay
        # frozen. ``glm.model.language_model`` is the real submodule path.
        self.glm.model.language_model = install_token_lora_adapter(
            self.glm.model.language_model,
            self.config,
        )
        logger.info(
            "Applied LoRA (rank=%d, alpha=%d) to the GLM-Image text trunk.",
            self.config.lora_rank,
            self.config.lora_alpha,
        )

    # ------------------------------------------------------------------
    # Generation vocab — the image codebook prefix of the vision vocab
    # ------------------------------------------------------------------

    def _init_gen_vocab(self) -> None:
        """Derive the codebook width from the checkpoint config.

        ``lm_head`` is ``hidden -> vision_vocab_size`` (16512); the legal
        sampling vocabulary is the VQ codebook prefix (``vq_config.
        num_embeddings`` = 16384 = ``image_start_token_id``). Codebook ids
        are ALSO text-vocab embedding rows 0..codebook-1, so gen-vocab ids
        need no remapping anywhere.
        """
        vq_config = getattr(self.glm.config, "vq_config", None)
        codebook = getattr(vq_config, "num_embeddings", None)
        if codebook is None:
            raise RuntimeError(
                "GLM-Image config carries no vq_config.num_embeddings — "
                "cannot derive the image codebook size.",
            )
        vision_vocab = int(self.text_config.vision_vocab_size)
        if int(codebook) >= vision_vocab:
            raise RuntimeError(
                f"GLM-Image codebook size {codebook} must be smaller than "
                f"vision_vocab_size {vision_vocab}",
            )
        self.image_vocab_size = int(codebook)

    def image_gen_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden states to codebook logits ``[..., V_codebook]``.

        The codebook occupies the contiguous prefix of the vision vocab, so
        the restriction is a slice — image_start/image_end and the reserved
        tail columns are never sampleable inside the raster (the reference
        relies on training statistics for this; slicing is the emu3-style
        hard structural guarantee, and replay applies the same slice so
        old/new log-prob distributions cannot drift).
        """
        logits = self.glm.lm_head(hidden)
        return logits[..., : self.image_vocab_size]

    def embed_gen_tokens(self, gen_token_ids: torch.Tensor) -> torch.Tensor:
        """Embed codebook token ids via the text-model embedding (identity map)."""
        return self.language_model.get_input_embeddings()(gen_token_ids)

    # ------------------------------------------------------------------
    # Prompt encoding — t2i chat-template processor call (single branch)
    # ------------------------------------------------------------------

    def encode_generation_prompts(
        self,
        prompts: list[str],
        *,
        max_text_length: int,
        image_height: int | None = None,
        image_width: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int, int]]:
        """Tokenize prompts in GLM-Image t2i generation mode.

        Uses ``processor.apply_chat_template`` exactly as the reference
        pipeline does (pipeline_glm_image.py lines 359-376): the processor
        appends ``<sop>{token_h} {token_w}<eop><sop>{prev_h} {prev_w}<eop>``
        plus the trailing image_start token. Prompts are LEFT-padded to
        ``max_text_length`` (the tokenizer's configured padding side; the
        runner reads the prefill hidden at position -1, which must be the
        real image_start token).

        Returns ``(input_ids, attention_mask, (token_h, token_w, prev_h,
        prev_w))`` on the model device.
        """
        height = int(image_height if image_height is not None else self.config.image_height)
        width = int(image_width if image_width is not None else self.config.image_width)
        token_h, token_w, prev_h, prev_w = glm_image_grid_dims(height, width)

        messages = [
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}] for prompt in prompts
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding="max_length",
            max_length=int(max_text_length),
            target_h=height,
            target_w=width,
            return_dict=True,
            return_tensors="pt",
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        if ids.shape[1] > max_text_length:
            raise ValueError(
                f"GLM-Image generation prompt is {ids.shape[1]} tokens, which "
                f"exceeds max_text_length={max_text_length}. Truncation would "
                "cut the grid suffix, so raise instead.",
            )
        grid_thw = inputs.get("image_grid_thw")
        if grid_thw is not None:
            expected = [[1, token_h, token_w], [1, prev_h, prev_w]]
            got = grid_thw[:2].tolist()
            if got != expected:
                raise RuntimeError(
                    f"GLM-Image processor grids {got} != derived grids "
                    f"{expected} — grid math drifted from the processor.",
                )
        return (
            ids.to(self.device),
            mask.to(self.device),
            (token_h, token_w, prev_h, prev_w),
        )

    # ------------------------------------------------------------------
    # Train-time forward — codebook logits (teacher-forced)
    # ------------------------------------------------------------------

    def _image_gen_hidden(
        self,
        prompt_inputs_embeds: torch.Tensor,  # [B, L_text, H]
        prompt_attention_mask: torch.Tensor,  # [B, L_text]
        image_token_ids: torch.Tensor,  # [B, L_img] codebook ids
        *,
        grids: tuple[tuple[int, int], ...],
    ) -> torch.Tensor:
        """One teacher-forced trunk pass returning hidden states per position.

        Layout matches Emu3 (text first, then the generated raster; logits
        read at the positions that *predict* each token), with the GLM
        addition of explicit 3-axis mrope position ids: text positions for
        the prompt, then the decode schedule offset by each row's valid
        prompt length. Returns the ``lm_head`` input, ``[B, L_img, H]``.
        """
        B, L_img = image_token_ids.shape
        img_embeds = self.embed_gen_tokens(image_token_ids)  # [B, L_img, H]

        # Concat: [text | image[:-1]] — image[-1] doesn't predict anything new
        inputs_embeds = torch.cat(
            [prompt_inputs_embeds, img_embeds[:, :-1, :]],
            dim=1,
        )
        L_text = prompt_inputs_embeds.shape[1]
        attn = torch.cat(
            [
                prompt_attention_mask,
                torch.ones(
                    B,
                    L_img - 1,
                    dtype=prompt_attention_mask.dtype,
                    device=prompt_attention_mask.device,
                ),
            ],
            dim=1,
        )

        device = inputs_embeds.device
        prompt_positions = glm_image_prefill_position_ids(prompt_attention_mask)
        # schedule(valid_len) == schedule(0) + valid_len (pure translation).
        base_schedule = glm_image_decode_position_schedule(0, grids).to(device)
        valid_lens = prompt_attention_mask.sum(dim=1).to(
            device=device,
            dtype=torch.long,
        )
        decode_positions = (
            base_schedule[:, : L_img - 1].unsqueeze(1)  # [3, 1, L_img-1]
            + valid_lens.view(1, B, 1)  # [1, B, 1]
        )
        position_ids = torch.cat([prompt_positions, decode_positions], dim=-1)

        outputs = self._lm_trunk()(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            position_ids=position_ids,
            use_cache=False,
        )
        hidden = outputs.last_hidden_state  # [B, L_text + L_img - 1, H]
        return hidden[:, L_text - 1 : L_text - 1 + L_img, :]

    def forward_image_logits(
        self,
        prompt_inputs_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        image_token_ids: torch.Tensor,
        *,
        grids: tuple[tuple[int, int], ...],
    ) -> torch.Tensor:
        """Teacher-forced codebook logits; see ``_image_gen_hidden``."""

        gen_hidden = self._image_gen_hidden(
            prompt_inputs_embeds,
            prompt_attention_mask,
            image_token_ids,
            grids=grids,
        )
        return self.image_gen_logits(gen_hidden)

    def vocab_head_split(self) -> VocabHeadSplit | None:
        """The codebook restriction as a weight slice of a plain lm_head.

        ``image_gen_logits`` is ``lm_head(hidden)[..., :V_codebook]`` — a row
        slice of the projection, so the split's weight/bias are the same
        slice (views, no copy). Exact-type check for the same reason as
        janus: a LoRA-wrapped lm_head must fall back to eager logits.
        """

        lm_head = self.glm.lm_head
        if type(lm_head) is not nn.Linear:
            return None
        return VocabHeadSplit(
            prefix=lambda h: h,
            weight=lm_head.weight[: self.image_vocab_size],
            bias=lm_head.bias[: self.image_vocab_size] if lm_head.bias is not None else None,
        )

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
        """Single forward producing per-token logits over the codebook.

        Mirrors the Emu3 replay contract: read prompt ids/masks and sampled
        tokens from ``batch.trajectory``, recompute logits under the current
        model, return ``{"logits", "image_token_ids"}``. Every generated
        position is a free codebook draw (no structural mask), so the
        codebook slice inside ``image_gen_logits`` is the whole parity story:
        the evaluator's plain ``log_softmax`` reproduces the rollout's
        restricted conditional distribution.

        AR has no notion of a denoising step, so only index zero is valid.
        """
        replay, image_token_ids = self._resolve_image_token_replay(
            batch,
            timestep_idx,
            request,
        )
        prompt_ids = replay["prompt_input_ids"]
        prompt_mask = replay["prompt_attention_mask"]

        grids = self._replay_grids(batch, int(image_token_ids.shape[1]))

        embed = self.language_model.get_input_embeddings()
        prompt_embeds = embed(prompt_ids)
        gen_hidden = self._image_gen_hidden(
            prompt_embeds,
            prompt_mask,
            image_token_ids,
            grids=grids,
        )
        values = head_replay_values(
            gen_hidden,
            self.vocab_head_split(),
            lambda: self.image_gen_logits(gen_hidden),
        )
        return single_segment_result(
            "image_tokens",
            {**values, "image_token_ids": image_token_ids},
        )

    def _replay_grids(
        self,
        batch: Any,
        token_count: int,
    ) -> tuple[tuple[int, int], ...]:
        """Rebuild the (large, preview) grids the rollout generated with.

        The executor stores the pixel ``image_height``/``image_width`` in the
        trajectory context; the grid math is deterministic from them (and NOT
        from the token count alone for non-square sizes).
        """
        height, width = replay_context_image_size(
            batch,
            token_count=token_count,
            expected_token_num=glm_image_token_num,
            owner="GLM-Image",
        )
        token_h, token_w, prev_h, prev_w = glm_image_grid_dims(height, width)
        return ((token_h, token_w), (prev_h, prev_w))

    # ------------------------------------------------------------------
    # Decode — AR prior tokens -> pixels via the frozen DiT segment
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode_image_tokens(
        self,
        image_token_ids: torch.Tensor,  # [B, L_total] codebook ids
        *,
        height: int,
        width: int,
        prompts: list[str],
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Decode the generated raster to RGB pixels in [-1, 1].

        Drops the preview grid, nearest-upsamples the large grid 2x (the DiT
        conditions on d16 token cells while the AR emits d32 —
        pipeline_glm_image.py ``_upsample_token_ids``), and runs the frozen
        ``GlmImagePipeline`` denoise+VAE segment via its public
        ``prior_token_ids`` input (pipeline lines 742/822-836), with
        ``prompt`` supplying the ByT5 glyph embeddings. The AR model plays no
        part — this is this family's "VQ decode".
        """
        token_h, token_w, prev_h, prev_w = glm_image_grid_dims(height, width)
        B, L = image_token_ids.shape
        expected = prev_h * prev_w + token_h * token_w
        if expected != L:
            raise ValueError(
                f"decode_image_tokens expects {expected} tokens "
                f"(preview {prev_h}x{prev_w} + large {token_h}x{token_w}) for "
                f"a {height}x{width} target, got {L}",
            )
        if len(prompts) != B:
            raise ValueError(
                f"decode_image_tokens needs one prompt per row for glyph "
                f"embeddings: got {len(prompts)} prompts for {B} rows",
            )
        large = image_token_ids[:, prev_h * prev_w :]
        prior_token_ids = torch.cat(
            [_upsample_token_ids(large[row], token_h, token_w) for row in range(B)],
            dim=0,
        )

        pipe = self._require_decode_pipeline()
        run_device = torch.device(self.config.device)
        ar_device = self.device
        offload_ar = self.config.decode_offload_ar and run_device.type == "cuda"
        try:
            if offload_ar:
                # 9B AR (18G) + DiT/VAE/ByT5 (~15G) exceed a 32GB card; park
                # the AR on CPU for the duration of the frozen decode.
                self.glm.to("cpu")
                torch.cuda.empty_cache()
            pipe.to(run_device)
            output = pipe(
                prompt=list(prompts),
                prior_token_ids=prior_token_ids.to(run_device),
                height=int(height),
                width=int(width),
                num_inference_steps=int(
                    num_inference_steps
                    if num_inference_steps is not None
                    else self.config.decode_num_inference_steps
                ),
                guidance_scale=float(
                    guidance_scale
                    if guidance_scale is not None
                    else self.config.decode_guidance_scale
                ),
                generator=generator,
                output_type="pt",
            )
        finally:
            pipe.to("cpu")
            if offload_ar:
                torch.cuda.empty_cache()
                self.glm.to(ar_device)
        # VaeImageProcessor.postprocess("pt") returns [0, 1]; the rollout
        # output contract is [-1, 1] like the other AR families.
        images = output.images.to(torch.float32)
        return (images * 2.0 - 1.0).clamp(-1.0, 1.0)

    def _require_decode_pipeline(self) -> Any:
        if self._decode_pipeline is None:
            self._decode_pipeline = _load_glm_image_decode_pipeline(self.config)
        return self._decode_pipeline


def _upsample_token_ids(token_ids: torch.Tensor, token_h: int, token_w: int) -> torch.Tensor:
    """Nearest-upsample a flat d32 raster to the DiT's d16 grid, ``[1, 4*H*W]``.

    Byte-for-byte the reference ``GlmImagePipeline._upsample_token_ids``
    (pipeline_glm_image.py lines 254-261); reimplemented so the replay-side
    module tree never imports diffusers.
    """
    token_ids = token_ids.view(1, 1, token_h, token_w)
    token_ids = torch.nn.functional.interpolate(
        token_ids.float(),
        scale_factor=2,
        mode="nearest",
    ).to(dtype=torch.long)
    return token_ids.view(1, -1)


# ---------------------------------------------------------------------------
# Replay-only model — text trunk + lm_head, no vision tower / DiT / VAE
# ---------------------------------------------------------------------------


class GlmImageReplayCore(ARReplayCore):
    """Minimal GLM-Image module set needed for trainer replay.

    ``GlmImageForConditionalGeneration.__init__`` constructs the ViT vision
    tower and VQ encoder; replay only needs teacher-forced codebook logits,
    so this core keeps the text trunk + lm_head. The submodule paths mirror
    the HF layout (``model.language_model.*`` / ``lm_head.*``) so the
    checkpoint state dict loads by name (verified against zai-org/GLM-Image's
    ``vision_language_encoder/model.safetensors.index.json``).
    """

    checkpoint_subfolder = GLM_IMAGE_AR_SUBFOLDER

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        from transformers.models.glm_image.modeling_glm_image import (
            GlmImageTextModel,
        )

        inner = nn.Module()
        inner.language_model = GlmImageTextModel._from_config(config.text_config)
        self.model = inner
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vision_vocab_size,
            bias=False,
        )


class GlmImageReplayModel(ARReplayRolloutStubs, GlmImageModel):
    """Replay-only GLM-Image wrapper without processor, vision tower, or DiT."""

    def __init__(
        self,
        config: GlmImageConfig | None = None,
        *,
        glm: Any | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config or GlmImageConfig()
        if glm is None:
            glm = GlmImageReplayCore.from_pretrained(
                self.config.model_path,
                device=self.config.device,
                dtype=resolve_torch_dtype(self.config.dtype),
                revision=self.config.revision,
            )

        for p in glm.parameters():
            p.requires_grad_(False)

        self.glm = glm
        self._decode_pipeline = None
        self._require_surface(("model", "lm_head"))
        self._require_inner_surface(("language_model",))
        if hasattr(self.glm.model, "visual") or hasattr(self.glm.model, "vqmodel"):
            raise RuntimeError(
                "GLM-Image replay model unexpectedly loaded generation-only "
                "modules (vision tower / VQ encoder)",
            )

        if self.config.use_lora:
            self._apply_lora()

        self._init_gen_vocab()

    @property
    def processor(self) -> Any:
        raise RuntimeError("GlmImageReplayModel does not own a GlmImageProcessor")

    def _require_decode_pipeline(self) -> Any:
        raise RuntimeError("GlmImageReplayModel does not own the DiT decode stack")


# ---------------------------------------------------------------------------
# Loaders — lazy imports so this module imports without model downloads.
# ---------------------------------------------------------------------------


def _load_glm_image_from_pretrained(config: GlmImageConfig) -> tuple[Any, Any]:
    """Load the AR model + processor from the hybrid checkpoint's subfolders."""

    from transformers import GlmImageForConditionalGeneration, GlmImageProcessor

    dtype = resolve_torch_dtype(config.dtype)
    processor = GlmImageProcessor.from_pretrained(
        config.model_path,
        subfolder=GLM_IMAGE_PROCESSOR_SUBFOLDER,
        revision=config.revision,
    )
    # The prefill runner reads the last-position hidden state as "next token
    # context"; the checkpoint tokenizer already configures left padding, but
    # enforce it in case a local copy drifted.
    processor.tokenizer.padding_side = "left"
    glm = GlmImageForConditionalGeneration.from_pretrained(
        config.model_path,
        subfolder=GLM_IMAGE_AR_SUBFOLDER,
        dtype=dtype,
        revision=config.revision,
    )
    glm = glm.to(device=config.device).eval()
    return glm, processor


def _load_glm_image_decode_pipeline(config: GlmImageConfig) -> Any:
    """Assemble the frozen DiT decode segment on CPU.

    Builds ``GlmImagePipeline`` from the checkpoint's non-AR components; the
    9B ``vision_language_encoder`` is registered as ``None`` because
    ``__call__`` with ``prior_token_ids`` never touches it
    (pipeline_glm_image.py lines 822-836). Components stay on CPU;
    ``decode_image_tokens`` moves them to GPU per decode.
    """

    from diffusers import (
        AutoencoderKL,
        FlowMatchEulerDiscreteScheduler,
        GlmImagePipeline,
        GlmImageTransformer2DModel,
    )
    from transformers import AutoTokenizer, GlmImageProcessor, T5EncoderModel

    dtype = resolve_torch_dtype(config.dtype)
    path = config.model_path
    pipe = GlmImagePipeline(
        tokenizer=AutoTokenizer.from_pretrained(
            path,
            subfolder="tokenizer",
            revision=config.revision,
        ),
        processor=GlmImageProcessor.from_pretrained(
            path,
            subfolder=GLM_IMAGE_PROCESSOR_SUBFOLDER,
            revision=config.revision,
        ),
        text_encoder=T5EncoderModel.from_pretrained(
            path,
            subfolder="text_encoder",
            dtype=dtype,
            revision=config.revision,
        ),
        vision_language_encoder=None,
        # diffusers models take ``torch_dtype`` (transformers 5 renamed its
        # kwarg to ``dtype``; diffusers 0.37 has not).
        vae=AutoencoderKL.from_pretrained(
            path,
            subfolder="vae",
            torch_dtype=dtype,
            revision=config.revision,
        ),
        transformer=GlmImageTransformer2DModel.from_pretrained(
            path,
            subfolder="transformer",
            torch_dtype=dtype,
            revision=config.revision,
        ),
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(
            path,
            subfolder="scheduler",
            revision=config.revision,
        ),
    )
    for module in (pipe.text_encoder, pipe.vae, pipe.transformer):
        module.requires_grad_(False)
        module.eval()
    return pipe


__all__ = [
    "GLM_IMAGE_AR_SUBFOLDER",
    "GLM_IMAGE_LATENT_FACTOR",
    "GLM_IMAGE_PROCESSOR_SUBFOLDER",
    "GlmImageConfig",
    "GlmImageModel",
    "GlmImageReplayCore",
    "GlmImageReplayModel",
    "glm_image_decode_position_schedule",
    "glm_image_grid_dims",
    "glm_image_prefill_position_ids",
    "glm_image_token_num",
]
