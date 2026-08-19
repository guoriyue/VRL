"""Janus-Pro-1B wrapper for autoregressive text-to-image RL.

This file isolates every Janus-specific detail (image-token vocab range,
``gen_head`` projection, CFG sampling, VQ decode) behind a small surface
that the generic GRPO trainer can call:

  * ``JanusProModel.forward_image_logits(...)``
        Train-time forward — returns logits over the *image* vocab for
        each image-token position. Used by the evaluator to recompute
        new log-probs under the current model.

  * ``JanusProModel.decode_image_tokens(...)``
        Decode 24x24 image tokens → pixels via the frozen VQ model.

  * ``JanusProModel.disable_adapter()``
        Context manager that turns LoRA off so the same module can serve
        as the reference model (DPO-style ``disable_adapter`` trick).

Why a custom forward instead of stock ``forward()``?
====================================================
Janus' generation path is *not* the same as its understanding-path
``forward``. For T2I we must:
  1. Embed text-prompt tokens with ``language_model.get_input_embeddings()``
  2. Embed previously-sampled image tokens with ``prepare_gen_img_embeds``
  3. Run the language-model trunk
  4. Project the *image-token* hidden states with ``gen_head``
     (NOT ``language_model.lm_head`` — that produces text logits!)

Doing this wrong silently optimises against text logits and trains nothing.

References
----------
DeepSeek's reference implementation:
  https://github.com/deepseek-ai/Janus/blob/main/generation_inference.py
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vrl.math.token.logprob import require_positive_temperature
from vrl.models.dtypes import resolve_torch_dtype
from vrl.models.families.janus_pro import JANUS_R1_SEGMENTS
from vrl.models.families.janus_pro.config import (
    JANUS_IMAGE_TOKEN_NUM,
    JanusProConfig,
)
from vrl.models.interfaces import (
    ReplayRequest,
    ReplayResult,
    ReplaySegmentResult,
    require_replay_segments,
    require_zero_replay_timestep,
    single_segment_result,
)
from vrl.models.peft_adapter import peel_peft
from vrl.models.steps.token.base import (
    ARModelBase,
    ARReplayCore,
    ARReplayRolloutStubs,
)
from vrl.models.steps.token.lora import install_token_lora_adapter
from vrl.models.steps.token.vocab_head import VocabHeadSplit, head_replay_values
from vrl.trajectory import role_tensor
from vrl.utils.logging import init_logger

logger = init_logger(__name__)

# Janus-Pro-1B image-tokenizer model constants (from deepseek-ai/Janus config)
JANUS_IMAGE_VOCAB_SIZE = 16_384  # gen_vision_model codebook size
JANUS_IMAGE_PATCH_SIZE = 16  # decoder upsample factor → 384 px
# Derived: sqrt(576 tokens) = 24-wide latent grid x 16 px/patch = 384 px.
JANUS_IMAGE_PIXEL_SIZE = int(JANUS_IMAGE_TOKEN_NUM**0.5) * JANUS_IMAGE_PATCH_SIZE
# Byte-sensitive model-protocol prompts: the R1 self-correction loop feeds these
# to the model verbatim. The special tokens (<end_of_image>, <begin_of_image>) and
# the FULLWIDTH VERTICAL LINE (U+FF5C) inside the end-of-sentence token must survive
# byte-for-byte. Keep in Python — do NOT move to YAML, where an editor/loader may
# normalize that ambiguous character (RUF001, suppressed below) or the leading
# newline and silently break the R1 loop.
JANUS_R1_SELFCHECK_PROMPT = "<end_of_image>\nLet me think Does this image match the prompt..."
JANUS_R1_REGEN_PROMPT = "<｜end▁of▁sentence｜>\nNext, I will draw a new image<begin_of_image>"  # noqa: RUF001


# ---------------------------------------------------------------------------
# Functional helper: project hidden states to image-token logits
# ---------------------------------------------------------------------------


def image_token_logits_from_hidden(
    mmgpt: nn.Module,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Apply Janus' generation head to hidden states.

    Args:
      mmgpt: a ``MultiModalityCausalLM`` instance (or LoRA-wrapped peer).
      hidden_states: trunk output at *image-token* positions, shape
        ``[B, L_img, hidden_size]``.

    Returns:
      Logits over the image vocabulary, shape
      ``[B, L_img, JANUS_IMAGE_VOCAB_SIZE]``.
    """
    # ``gen_head`` lives on the underlying mmgpt; PEFT wrapping preserves it.
    # ``peel_peft`` documents why hasattr(base_model) alone is not the key.
    base = peel_peft(mmgpt)
    return base.gen_head(hidden_states)


def image_token_head_split(mmgpt: nn.Module) -> VocabHeadSplit | None:
    """Split ``gen_head`` at its final vocab Linear when its shape allows.

    DeepSeek's ``vision_head`` is ``vision_head(vision_activation(
    output_mlp_projector(h)))`` with a plain final Linear over the 16384-token
    image vocab. Returns None for any other structure (callers fall back to
    eager logits), which also fail-safes a future checkpoint that LoRA-wraps
    or reshapes the head — ``VocabHeadSplit.from_linear`` owns that guard.
    """

    head = peel_peft(mmgpt).gen_head
    split = VocabHeadSplit.from_linear(head)
    if split is not None:
        return split
    proj = getattr(head, "output_mlp_projector", None)
    act = getattr(head, "vision_activation", None)
    if callable(proj) and callable(act):
        return VocabHeadSplit.from_linear(
            getattr(head, "vision_head", None),
            prefix=lambda h: act(proj(h)),
        )
    return None


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class JanusProModel(ARModelBase):
    """Train-and-sample wrapper for Janus-Pro text-to-image generation.

    Keeps the LoRA-wrapped language model + frozen vq / vision / aligner
    in a single ``nn.Module`` so it integrates cleanly with FSDP / EMA.
    """

    checkpoint_description = "a Janus MultiModalityCausalLM checkpoint"

    def __init__(
        self,
        config: JanusProConfig | None = None,
        *,
        mmgpt: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        """Construct the wrapper.

        Args:
          config: hyper-parameters. ``None`` → defaults for Janus-Pro-1B.
          mmgpt: optional pre-loaded ``MultiModalityCausalLM`` (saves the
            ~3 GB checkpoint download in tests). When ``None``, we load
            from ``config.model_path``.
          processor: optional pre-loaded ``VLChatProcessor``.
        """
        super().__init__()
        self.config = config or JanusProConfig()

        if mmgpt is None:
            mmgpt, processor = _load_janus_from_pretrained(self.config)
        elif processor is None:
            raise ValueError("Must pass `processor` when `mmgpt` is provided")

        self._processor = processor

        # Freeze everything by default — LoRA wrap re-enables only attention
        # projections in the language model.
        for p in mmgpt.parameters():
            p.requires_grad_(False)

        if self.config.use_lora:
            mmgpt = self._apply_lora(mmgpt)

        self.mmgpt = mmgpt

        # Sanity: confirm gen_head + gen_vision_model exist
        self._require_module_attrs(
            self._base(),
            ("gen_head", "gen_vision_model", "language_model"),
        )

    # ------------------------------------------------------------------
    # Sub-module accessors
    # ------------------------------------------------------------------

    def _base(self) -> nn.Module:
        """Return the unwrapped MultiModalityCausalLM (peels a PEFT wrap)."""
        return peel_peft(self.mmgpt)

    def _lm_trunk(self) -> nn.Module:
        """Return the LlamaModel trunk that emits ``last_hidden_state``.

        One hop deeper than the shared default, because Janus' layering
        depends on whether LoRA is attached:
          * No LoRA:  ``language_model`` is ``LlamaForCausalLM``; its
            ``.model`` is the ``LlamaModel`` trunk.
          * With LoRA: ``language_model`` is a PEFT ``LoraModel`` wrapping
            ``LlamaForCausalLM``; the trunk is two hops in via
            ``base_model.model.model``.

        Calling ``LlamaForCausalLM`` directly returns a ``CausalLMOutputWithPast``
        which has ``.logits`` over the text vocab — the *wrong* projection
        for image-token generation. We need the raw hidden states, so we
        unwrap all the way down to ``LlamaModel``.
        """
        # The base peels PEFT and yields LlamaForCausalLM (or the bare lm
        # without LoRA); the extra ``.model`` hop reaches the trunk beneath it.
        cls_lm = super()._lm_trunk()
        return cls_lm.model if hasattr(cls_lm, "model") else cls_lm

    @property
    def processor(self) -> Any:
        return self._processor

    @property
    def language_model(self) -> nn.Module:
        return self._base().language_model

    @property
    def vq_model(self) -> nn.Module:
        return self._base().gen_vision_model

    @property
    def device(self) -> torch.device:
        return next(self.mmgpt.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.mmgpt.parameters()).dtype

    # ------------------------------------------------------------------
    # LoRA / reference-policy helpers
    # ------------------------------------------------------------------

    def _apply_lora(self, mmgpt: Any) -> Any:
        """Attach a PEFT LoRA adapter to the language-model trunk."""
        mmgpt.language_model = install_token_lora_adapter(
            mmgpt.language_model,
            self.config,
            task_type="CAUSAL_LM",
        )
        logger.info(
            "Applied LoRA (rank=%d, alpha=%d) to Janus language model.",
            self.config.lora_rank,
            self.config.lora_alpha,
        )
        return mmgpt

    # ------------------------------------------------------------------
    # Train-time forward — image-token logits
    # ------------------------------------------------------------------

    def _image_gen_hidden(
        self,
        prompt_inputs_embeds: torch.Tensor,  # [B, L_text, H]
        prompt_attention_mask: torch.Tensor,  # [B, L_text]
        image_token_ids: torch.Tensor,  # [B, L_img]
    ) -> torch.Tensor:
        """One trunk pass returning hidden states at image-token positions.

        Layout convention: text comes first, then image tokens. We feed
        the *teacher-forced* sequence and extract logits at positions
        that *predict* each image token (i.e. the position immediately
        before it).

        Args:
          prompt_inputs_embeds: text embeddings (already passed through
            ``language_model.get_input_embeddings()``). Shape
            ``[B, L_text, hidden_size]``.
          prompt_attention_mask: 1/0 mask for the text part, ``[B, L_text]``.
          image_token_ids: previously-sampled image tokens to score, shape
            ``[B, L_img]``. ``L_img`` is typically
            ``JANUS_IMAGE_TOKEN_NUM`` (576).

        Returns:
          Trunk hidden states at positions that *predict* each image token,
          shape ``[B, L_img, hidden_size]`` — the ``gen_head`` input.
        """
        base = self._base()
        B, L_img = image_token_ids.shape

        # Embed image tokens via Janus' generation embedder.
        img_embeds = base.prepare_gen_img_embeds(image_token_ids)  # [B, L_img, H]

        # Concat: [text | image[:-1]]  — image[-1] doesn't predict anything new
        inputs_embeds = torch.cat([prompt_inputs_embeds, img_embeds[:, :-1, :]], dim=1)
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

        outputs = self._lm_trunk()(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            use_cache=False,
            output_hidden_states=False,
        )
        hidden = outputs.last_hidden_state  # [B, L_text + L_img - 1, H]

        # Positions that *predict* image_token_ids[:, 0..L_img-1]
        # are L_text - 1, L_text, ..., L_text + L_img - 2.
        return hidden[:, L_text - 1 : L_text - 1 + L_img, :]

    def forward_image_logits(
        self,
        prompt_inputs_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        image_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced image-vocab logits; see ``_image_gen_hidden``."""

        gen_hidden = self._image_gen_hidden(
            prompt_inputs_embeds,
            prompt_attention_mask,
            image_token_ids,
        )
        return image_token_logits_from_hidden(self.mmgpt, gen_hidden)

    def vocab_head_split(self) -> VocabHeadSplit | None:
        """Janus' gen_head split (DeepSeek projector+GELU+Linear structure)."""

        return image_token_head_split(self.mmgpt)

    def _image_token_replay_values(
        self,
        prompt_inputs_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        image_token_ids: torch.Tensor,
    ) -> dict[str, Any]:
        """Replay payload for the image segment, fused-head form when possible.

        The eager form materializes ``[B, L_img, 16384]`` logits and keeps
        them alive for backward; the split form (janus' head is not a LoRA
        target — it adapts q/k/v/o only) hands the projection's input and
        weight to the ReplaySegmentResult contract instead.
        """

        gen_hidden = self._image_gen_hidden(
            prompt_inputs_embeds,
            prompt_attention_mask,
            image_token_ids,
        )
        return head_replay_values(
            gen_hidden,
            self.vocab_head_split(),
            lambda: image_token_logits_from_hidden(self.mmgpt, gen_hidden),
        )

    def forward_text_logits(
        self,
        prompt_inputs_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        text_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced text-vocab logits for an R1 reflection segment."""

        B, L_txt = text_token_ids.shape
        text_embeds = self.language_model.get_input_embeddings()(text_token_ids)
        inputs_embeds = torch.cat(
            [prompt_inputs_embeds, text_embeds[:, :-1, :]],
            dim=1,
        )
        attn = torch.cat(
            [
                prompt_attention_mask,
                torch.ones(
                    B,
                    L_txt - 1,
                    dtype=prompt_attention_mask.dtype,
                    device=prompt_attention_mask.device,
                ),
            ],
            dim=1,
        )
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            use_cache=False,
        )
        L_ctx = prompt_inputs_embeds.shape[1]
        return outputs.logits[:, L_ctx - 1 : L_ctx - 1 + L_txt, :]

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

        Train-time replay: read prompt ids, prompt masks, and sampled image
        tokens from ``batch.trajectory`` and recompute logits under the current
        model.

        AR has no notion of a denoising step, so only index zero is valid.

        See ``vrl/models/interfaces/replay.py::ReplayModel`` for the shared
        trainer replay protocol.

        Returns:
          ``ReplayResult`` with one or more segment payloads.
        """
        require_zero_replay_timestep(timestep_idx, owner=type(self).__name__)
        requested_segments = None if request is None else request.segment_names
        if requested_segments == ("image_tokens",):
            requested_segments = None
        if requested_segments is not None:
            require_replay_segments(
                request,
                JANUS_R1_SEGMENTS,
                owner=type(self).__name__,
            )
            segments = {
                name: ReplaySegmentResult(
                    segment=name,
                    values=self.replay_r1_segment(
                        self._r1_segment_payload_from_trajectory(batch, name),
                    ),
                )
                for name in requested_segments
            }
            return ReplayResult(segments=segments)

        replay, image_token_ids = self._resolve_image_token_replay(
            batch,
            timestep_idx,
            request,
        )
        prompt_ids = replay["prompt_input_ids"]
        prompt_mask = replay["prompt_attention_mask"]

        embed = self.language_model.get_input_embeddings()
        prompt_embeds = embed(prompt_ids)
        values = self._image_token_replay_values(
            prompt_embeds,
            prompt_mask,
            image_token_ids,
        )
        return single_segment_result(
            "image_tokens",
            {**values, "image_token_ids": image_token_ids},
        )

    def _r1_segment_payload_from_trajectory(
        self,
        batch: Any,
        segment_name: str,
    ) -> dict[str, Any]:
        trajectory = getattr(batch, "trajectory", None)
        if trajectory is None or segment_name not in trajectory.segments:
            raise RuntimeError(
                f"Janus-Pro-R1 replay requires trajectory segment {segment_name!r}",
            )
        segment = trajectory.segments[segment_name]
        payload: dict[str, Any] = {
            "name": segment.name,
            "token_ids": role_tensor(segment, "action").value,
            "visual": bool(segment.metadata.get("visual", segment.modality == "image")),
            "modality": segment.modality,
        }
        for key in ("prompt_embeds", "attention_mask", "prompt_attention_mask"):
            tensor = segment.tensors.get(key)
            if tensor is not None:
                payload[key] = tensor.value
        if "attention_mask" not in payload and "prompt_attention_mask" in payload:
            payload["attention_mask"] = payload["prompt_attention_mask"]
        return payload

    def replay_r1_segment(self, segment: dict[str, Any]) -> dict[str, Any]:
        """Replay one Janus-Pro-R1 segment from packed rollout extras."""

        token_ids = segment["token_ids"]
        prompt_embeds = segment["prompt_embeds"]
        attention_mask = segment["attention_mask"]
        if bool(segment.get("visual", True)):
            values = self._image_token_replay_values(
                prompt_embeds,
                attention_mask,
                token_ids,
            )
            return {**values, "token_ids": token_ids}
        logits = self.forward_text_logits(
            prompt_embeds,
            attention_mask,
            token_ids,
        )
        return {"logits": logits, "token_ids": token_ids}

    @torch.no_grad()
    def generate_with_refine(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        *,
        guidance_scale: float,
        temperature: float,
        image_token_num: int,
        max_reflect_len: int,
        uncond_input_ids: torch.Tensor | None = None,
        uncond_attention_mask: torch.Tensor | None = None,
        image_size: int = JANUS_IMAGE_PIXEL_SIZE,
        refine_mode: str | None = None,
        image_sampler: Callable[..., tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> dict[str, Any]:
        """Run Janus-Pro-R1-style first image, self-check, and regeneration.

        Each returned segment includes the prefix embeddings and attention
        mask needed to replay that segment's sampled tokens. Image sampling is
        injected by the runtime runner; VQ decode deliberately reuses
        ``decode_image_tokens`` so plain Janus-Pro decoding remains the source
        of truth.
        """

        if max_reflect_len < 1:
            raise ValueError("max_reflect_len must be >= 1")
        temperature = require_positive_temperature(temperature)

        mode = (refine_mode or "selfcheck").lower()
        if mode not in {"selfcheck", "always"}:
            raise ValueError("refine_mode must be one of: 'selfcheck', 'always'")

        prompt_input_ids = prompt_input_ids.to(self.device)
        prompt_attention_mask = prompt_attention_mask.to(self.device)
        if uncond_input_ids is None:
            # Direct callers may only have the required minimal signature.
            # The executor passes a real empty-prompt context, which is the
            # correct CFG path. This fallback keeps the API runnable.
            uncond_input_ids = prompt_input_ids
        else:
            uncond_input_ids = uncond_input_ids.to(self.device)
        if uncond_attention_mask is None:
            uncond_attention_mask = prompt_attention_mask
        else:
            uncond_attention_mask = uncond_attention_mask.to(self.device)

        cond_embeds = self._embed_text_ids(prompt_input_ids)
        uncond_embeds = self._embed_text_ids(uncond_input_ids)
        pad_token_id = self._pad_token_id()
        yes_token_id = self._last_token_id("Yes")
        no_token_id = self._last_token_id("No")
        eos_token_id = self._eos_token_id()

        if image_sampler is None:
            raise ValueError(
                "generate_with_refine requires image_sampler; "
                "runtime generation must pass JanusProARModelRunner sampling",
            )
        sample_image = image_sampler
        initial_ids, initial_logps = sample_image(
            cond_embeds,
            uncond_embeds,
            prompt_attention_mask,
            uncond_attention_mask,
            guidance_scale=guidance_scale,
            temperature=temperature,
            image_token_num=image_token_num,
        )
        initial_image = self.decode_image_tokens(initial_ids, image_size=image_size)
        image_embeds = self._base().prepare_gen_img_embeds(initial_ids)
        image_mask = torch.ones(
            image_embeds.shape[:2],
            dtype=prompt_attention_mask.dtype,
            device=prompt_attention_mask.device,
        )

        selfcheck_prefix_ids = self._repeat_text_ids(
            JANUS_R1_SELFCHECK_PROMPT,
            batch_size=prompt_input_ids.shape[0],
        )
        selfcheck_prefix_embeds = self._embed_text_ids(selfcheck_prefix_ids)
        selfcheck_prefix_mask = torch.ones(
            selfcheck_prefix_ids.shape,
            dtype=prompt_attention_mask.dtype,
            device=prompt_attention_mask.device,
        )
        selfcheck_prompt_embeds = torch.cat(
            [cond_embeds, image_embeds, selfcheck_prefix_embeds],
            dim=1,
        )
        selfcheck_prompt_mask = torch.cat(
            [prompt_attention_mask, image_mask, selfcheck_prefix_mask],
            dim=1,
        )

        text_ids, text_logps, text_mask, selfcheck = self._sample_selfcheck_text(
            selfcheck_prompt_embeds,
            selfcheck_prompt_mask,
            max_new_tokens=max_reflect_len,
            temperature=float(temperature),
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )

        regen_prefix_ids = self._repeat_text_ids(
            JANUS_R1_REGEN_PROMPT,
            batch_size=prompt_input_ids.shape[0],
        )
        regen_prefix_embeds = self._embed_text_ids(regen_prefix_ids)
        regen_prefix_mask = torch.ones(
            regen_prefix_ids.shape,
            dtype=prompt_attention_mask.dtype,
            device=prompt_attention_mask.device,
        )
        text_embeds = self._embed_text_ids(text_ids)
        text_attention_mask = text_mask.to(dtype=prompt_attention_mask.dtype)
        final_cond_embeds = torch.cat(
            [
                cond_embeds,
                image_embeds,
                selfcheck_prefix_embeds,
                text_embeds,
                regen_prefix_embeds,
            ],
            dim=1,
        )
        final_cond_mask = torch.cat(
            [
                prompt_attention_mask,
                image_mask,
                selfcheck_prefix_mask,
                text_attention_mask,
                regen_prefix_mask,
            ],
            dim=1,
        )
        final_uncond_embeds = torch.cat(
            [
                uncond_embeds,
                image_embeds,
                selfcheck_prefix_embeds,
                text_embeds,
                regen_prefix_embeds,
            ],
            dim=1,
        )
        final_uncond_mask = torch.cat(
            [
                uncond_attention_mask,
                image_mask,
                selfcheck_prefix_mask,
                text_attention_mask,
                regen_prefix_mask,
            ],
            dim=1,
        )

        refined_ids, refined_logps = sample_image(
            final_cond_embeds,
            final_uncond_embeds,
            final_cond_mask,
            final_uncond_mask,
            guidance_scale=guidance_scale,
            temperature=temperature,
            image_token_num=image_token_num,
        )
        refined_image = self.decode_image_tokens(
            refined_ids,
            image_size=image_size,
        )

        if mode == "always":
            use_refined = torch.ones_like(selfcheck, dtype=torch.bool)
        else:
            # Reference R1 semantics: "Yes" accepts the first image;
            # "No" asks the model to use the regenerated image.
            use_refined = ~selfcheck

        final_ids = torch.where(use_refined.unsqueeze(1), refined_ids, initial_ids)
        final_logps = torch.where(
            use_refined.unsqueeze(1),
            refined_logps,
            initial_logps,
        )
        final_image = torch.where(
            use_refined.view(-1, 1, 1, 1),
            refined_image,
            initial_image,
        )
        initial_prompt_embeds_pad, initial_prompt_mask_pad = self._left_pad_replay_context(
            cond_embeds,
            prompt_attention_mask,
            target_length=final_cond_embeds.shape[1],
            pad_token_id=pad_token_id,
        )
        final_prompt_embeds = torch.where(
            use_refined.view(-1, 1, 1),
            final_cond_embeds,
            initial_prompt_embeds_pad,
        )
        final_prompt_mask = torch.where(
            use_refined.unsqueeze(1),
            final_cond_mask,
            initial_prompt_mask_pad,
        )

        ones_initial = torch.ones_like(initial_logps)
        ones_final = torch.ones_like(final_logps)
        segments = {
            "initial_image": {
                "name": "initial_image",
                "token_ids": initial_ids,
                "token_log_probs": initial_logps,
                "token_mask": ones_initial,
                "prompt_embeds": cond_embeds,
                "attention_mask": prompt_attention_mask,
                "prompt_attention_mask": prompt_attention_mask,
                "visual": True,
                "cfg": True,
            },
            "selfcheck_text": {
                "name": "selfcheck_text",
                "token_ids": text_ids,
                "token_log_probs": text_logps,
                "token_mask": text_mask.to(dtype=text_logps.dtype),
                "prompt_embeds": selfcheck_prompt_embeds,
                "attention_mask": selfcheck_prompt_mask,
                "prompt_attention_mask": selfcheck_prompt_mask,
                "visual": False,
                "cfg": False,
            },
            "final_image": {
                "name": "final_image",
                "token_ids": final_ids,
                "token_log_probs": final_logps,
                "token_mask": ones_final,
                "prompt_embeds": final_prompt_embeds,
                "attention_mask": final_prompt_mask,
                "prompt_attention_mask": final_prompt_mask,
                "visual": True,
                "cfg": True,
            },
        }
        return {
            "initial_image": initial_image,
            "final_image": final_image,
            "selfcheck": selfcheck,
            "segments": segments,
            "context": {
                "temperature": float(temperature),
                # Display/provenance-only: OnlineTrainer persists these R1
                # policy choices in its first-step ``rollout_context`` record.
                "guidance_scale": float(guidance_scale),
                "refine_mode": mode,
            },
        }

    def _pad_token_id(self) -> int:
        tokenizer = self.processor.tokenizer
        for attr in ("pad_id", "pad_token_id"):
            value = getattr(self.processor, attr, None)
            if value is not None:
                return int(value)
            value = getattr(tokenizer, attr, None)
            if value is not None:
                return int(value)
        return 0

    def _eos_token_id(self) -> int:
        tokenizer = self.processor.tokenizer
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None:
            return int(eos)
        return self._last_token_id("<｜end▁of▁sentence｜>")  # noqa: RUF001

    def _last_token_id(self, text: str) -> int:
        ids = self._encode_text_ids(text)
        if not ids:
            raise RuntimeError(f"tokenizer produced no ids for {text!r}")
        return int(ids[-1])

    def _embed_text_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.get_input_embeddings()(token_ids)

    def _repeat_text_ids(
        self,
        text: str,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        ids = self._encode_text_ids(text)
        if not ids:
            ids = [self._pad_token_id()]
        tensor = torch.tensor(ids, dtype=torch.long, device=self.device)
        return tensor.unsqueeze(0).expand(batch_size, -1).contiguous()

    def _encode_text_ids(self, text: str) -> list[int]:
        tokenizer = self.processor.tokenizer
        encode = getattr(tokenizer, "encode", None)
        if callable(encode):
            ids = encode(text)
            if isinstance(ids, torch.Tensor):
                ids = ids.detach().cpu().reshape(-1).tolist()
            if ids and self._looks_like_bos(ids[0]):
                ids = ids[1:]
            return [int(x) for x in ids]
        vocab_size = getattr(tokenizer, "vocab_size", 256)
        return [ord(ch) % int(vocab_size) for ch in text]

    def _looks_like_bos(self, token_id: int) -> bool:
        tokenizer = self.processor.tokenizer
        bos = getattr(tokenizer, "bos_token_id", None)
        if bos is not None:
            return int(token_id) == int(bos)
        return int(token_id) == 1

    def _left_pad_replay_context(
        self,
        prompt_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        target_length: int,
        pad_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pad_len = int(target_length) - int(prompt_embeds.shape[1])
        if pad_len < 0:
            raise ValueError("target_length must be >= prompt context length")
        if pad_len == 0:
            return prompt_embeds, attention_mask
        pad_ids = torch.full(
            (prompt_embeds.shape[0], pad_len),
            int(pad_token_id),
            dtype=torch.long,
            device=prompt_embeds.device,
        )
        pad_embeds = self._embed_text_ids(pad_ids)
        pad_mask = torch.zeros(
            prompt_embeds.shape[0],
            pad_len,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        return (
            torch.cat([pad_embeds, prompt_embeds], dim=1),
            torch.cat([pad_mask, attention_mask], dim=1),
        )

    @torch.no_grad()
    def _sample_selfcheck_text(
        self,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float,
        yes_token_id: int,
        no_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = prompt_embeds.shape[0]
        device = prompt_embeds.device
        temp = require_positive_temperature(temperature)

        token_ids = torch.full(
            (batch_size, max_new_tokens),
            int(pad_token_id),
            dtype=torch.long,
            device=device,
        )
        log_probs = torch.zeros(batch_size, max_new_tokens, dtype=torch.float32, device=device)
        mask = torch.zeros(batch_size, max_new_tokens, dtype=torch.float32, device=device)
        context_embeds = prompt_embeds
        context_mask = prompt_attention_mask
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        embed = self.language_model.get_input_embeddings()

        for pos in range(max_new_tokens):
            outputs = self.language_model(
                inputs_embeds=context_embeds,
                attention_mask=context_mask,
                use_cache=False,
            )
            logits = outputs.logits[:, -1, :].float()
            if pos == 0:
                allowed = torch.tensor(
                    [yes_token_id, no_token_id],
                    dtype=torch.long,
                    device=device,
                )
                restricted = torch.full_like(logits, float("-inf"))
                restricted[:, allowed] = logits[:, allowed]
                logits = restricted
            log_probs_all = F.log_softmax(logits / temp, dim=-1)
            probs = torch.exp(log_probs_all)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            next_token = torch.where(
                finished,
                torch.full_like(next_token, int(pad_token_id)),
                next_token,
            )
            log_probs[:, pos] = log_probs_all.gather(
                -1,
                next_token.unsqueeze(-1),
            ).squeeze(-1)
            token_ids[:, pos] = next_token
            active = ~finished
            is_eos = next_token == int(eos_token_id)
            is_pad = next_token == int(pad_token_id)
            keep = active & ~is_eos & ~is_pad
            mask[:, pos] = keep.to(dtype=mask.dtype)
            finished = finished | is_eos | is_pad
            next_embeds = embed(next_token).unsqueeze(1)
            context_embeds = torch.cat([context_embeds, next_embeds], dim=1)
            context_mask = torch.cat(
                [
                    context_mask,
                    keep.to(dtype=context_mask.dtype).unsqueeze(1),
                ],
                dim=1,
            )
            if bool(finished.all()):
                break

        selfcheck = token_ids[:, 0] == int(yes_token_id)
        return token_ids, log_probs, mask, selfcheck

    # ------------------------------------------------------------------
    # VQ decode — image tokens → pixels
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode_image_tokens(
        self,
        image_token_ids: torch.Tensor,  # [B, L_img]
        *,
        image_size: int = JANUS_IMAGE_PIXEL_SIZE,
    ) -> torch.Tensor:
        """Decode image-token grids to RGB pixels in [-1, 1].

        Returns shape ``[B, 3, image_size, image_size]``.
        """
        if not isinstance(image_token_ids, torch.Tensor):
            raise TypeError(
                "Janus image_token_ids must be a torch.Tensor; "
                f"got {type(image_token_ids).__name__}",
            )
        if image_token_ids.ndim != 2:
            raise ValueError(
                "Janus image_token_ids must have shape [batch, tokens]; "
                f"got {tuple(image_token_ids.shape)}",
            )
        if not isinstance(image_size, int) or isinstance(image_size, bool):
            raise TypeError(f"Janus image_size must be an int; got {type(image_size).__name__}")
        B, L = image_token_ids.shape
        side = int(L**0.5)
        if side * side != L:
            raise ValueError(f"expected square image-token grid, got L_img={L}")
        expected_image_size = side * JANUS_IMAGE_PATCH_SIZE
        if image_size != expected_image_size:
            raise ValueError(
                "Janus image_size does not match the image-token grid: "
                f"requested {image_size}, expected {expected_image_size} "
                f"from {side}x{side} tokens",
            )
        # Janus' decode_code feeds shape[1] to the quantizer as the codebook-entry
        # dim, and it differs across Janus-Pro variants — resolve it from the LIVE
        # quantizer. Do NOT "align to upstream" by hardcoding `8` (upstream
        # generation_inference.py): any variant whose codebook width != 8 would then
        # reshape to garbage silently. Locked by
        # tests/models/families/janus_pro/test_upstream_reconcile_contracts.py.
        latent_channels = self._resolve_vq_latent_channels()
        decoded = self.vq_model.decode_code(
            image_token_ids.to(torch.int32),
            shape=[B, latent_channels, side, side],
        )
        return decoded.clamp(-1.0, 1.0)

    def _resolve_vq_latent_channels(self) -> int:
        """Resolve the VQ decoder's latent-channel dimension.

        ``decode_code`` feeds ``shape[1]`` to ``get_codebook_entry``, which
        uses it as the *codebook-entry* dim, NOT the encoder z_channels dim.
        On Janus-Pro-1B these differ: ``config.z_channels=256`` (encoder
        hidden) but the quantizer codebook is 8-dim — using 256 produces a
        silent reshape explosion.

        Resolution is intentionally strict: use the explicit override or the
        live quantizer embedding shape. If neither is available, fail instead
        of guessing a checkpoint-specific constant.
        """
        override = self.config.vq_latent_channels
        if override is not None:
            if not isinstance(override, int) or override <= 0:
                raise RuntimeError(
                    f"vq_latent_channels override must be a positive int; got {override!r}"
                )
            return override

        # Live probe of the quantizer — authoritative on any Janus variant.
        quant = getattr(self.vq_model, "quantize", None)
        emb = getattr(quant, "embedding", None) if quant is not None else None
        if emb is not None and hasattr(emb, "weight"):
            w = emb.weight
            if w.ndim == 2 and w.shape[-1] > 0:
                return int(w.shape[-1])

        raise RuntimeError(
            "Could not resolve Janus VQ latent channels. Set "
            "JanusProConfig.vq_latent_channels or provide a VQ model with "
            "quantize.embedding.weight."
        )


class JanusProReplayCore(ARReplayCore):
    """Minimal Janus module set needed for trainer replay.

    The full upstream class constructs vision encoders and the VQ decoder in
    ``__init__``. Replay only needs text/image-token logits, so this core keeps
    the generation embedding/projection path and the language model only.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)

        from janus.models.modeling_vlm import model_name_to_cls
        from transformers import LlamaForCausalLM

        gen_vision_config = config.gen_vision_config
        gen_aligner_config = config.gen_aligner_config
        gen_head_config = config.gen_head_config

        gen_aligner_cls = model_name_to_cls(gen_aligner_config.cls)
        self.gen_aligner = gen_aligner_cls(gen_aligner_config.params)

        gen_head_cls = model_name_to_cls(gen_head_config.cls)
        self.gen_head = gen_head_cls(gen_head_config.params)

        self.gen_embed = nn.Embedding(
            int(gen_vision_config.params.image_token_size),
            int(gen_vision_config.params.n_embed),
        )
        self.language_model = LlamaForCausalLM(config.language_config)

    def prepare_gen_img_embeds(self, image_ids: torch.LongTensor) -> torch.Tensor:
        return self.gen_aligner(self.gen_embed(image_ids))


class JanusProReplayModel(ARReplayRolloutStubs, JanusProModel):
    """Replay-only Janus wrapper without processor, vision tower, or VQ decoder."""

    def __init__(
        self,
        config: JanusProConfig | None = None,
        *,
        mmgpt: Any | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config or JanusProConfig()
        if mmgpt is None:
            mmgpt = _load_janus_replay_core_from_pretrained(self.config)

        for p in mmgpt.parameters():
            p.requires_grad_(False)

        if self.config.use_lora:
            mmgpt = self._apply_lora(mmgpt)

        self.mmgpt = mmgpt
        base = self._base()
        for attr in ("gen_head", "language_model", "prepare_gen_img_embeds"):
            if not hasattr(base, attr):
                raise RuntimeError(
                    f"Loaded Janus replay model is missing required `{attr}`",
                )
        forbidden = [
            attr for attr in ("gen_vision_model", "vision_model", "aligner") if hasattr(base, attr)
        ]
        if forbidden:
            raise RuntimeError(
                f"Janus replay model unexpectedly loaded generation-only modules: {forbidden}",
            )

    @property
    def processor(self) -> Any:
        raise RuntimeError("JanusProReplayModel does not own a VLChatProcessor")

    @property
    def vq_model(self) -> nn.Module:
        raise RuntimeError("JanusProReplayModel does not own a VQ decoder")


# ---------------------------------------------------------------------------
# Loader — lazy import so this module is importable without the janus pkg.
# ---------------------------------------------------------------------------


def _load_janus_from_pretrained(config: JanusProConfig) -> tuple[Any, Any]:
    """Load ``MultiModalityCausalLM`` + ``VLChatProcessor`` from disk/HF.

    The ``janus`` package (``deepseek-ai/Janus`` on GitHub, NOT PyPI)
    must be installed:

        git clone https://github.com/deepseek-ai/Janus
        cd Janus && pip install -e .
    """
    try:
        from janus.models import MultiModalityCausalLM, VLChatProcessor
    except ImportError as e:
        raise ImportError(
            "Cannot import deepseek-ai/Janus. Install via:\n"
            "  git clone https://github.com/deepseek-ai/Janus\n"
            "  cd Janus && pip install -e .\n"
            "(The PyPI package called 'janus' is unrelated — it's an "
            "asyncio queue library.)"
        ) from e

    from transformers import AutoModelForCausalLM

    dtype = resolve_torch_dtype(config.dtype)

    processor = VLChatProcessor.from_pretrained(
        config.model_path,
        revision=config.revision,
    )
    mmgpt = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=dtype,
        revision=config.revision,
    )
    if not isinstance(mmgpt, MultiModalityCausalLM):
        raise TypeError(
            f"Loaded model {type(mmgpt).__name__} is not MultiModalityCausalLM",
        )
    mmgpt = mmgpt.to(device=config.device, dtype=dtype).eval()
    return mmgpt, processor


def _load_janus_replay_core_from_pretrained(config: JanusProConfig) -> JanusProReplayCore:
    """Load Janus replay modules without constructing VQ or vision modules.

    The upstream-package pre-flight lives here rather than in
    ``JanusProReplayCore.__init__`` so a caller that hands the wrapper an
    already-built core never needs the ``janus`` package installed.
    """

    try:
        import janus.models.modeling_vlm  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Cannot import deepseek-ai/Janus. Install via:\n"
            "  git clone https://github.com/deepseek-ai/Janus\n"
            "  cd Janus && pip install -e ."
        ) from e

    return JanusProReplayCore.from_pretrained(
        config.model_path,
        device=config.device,
        dtype=resolve_torch_dtype(config.dtype),
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
    )


__all__ = [
    "JANUS_IMAGE_PATCH_SIZE",
    "JANUS_IMAGE_PIXEL_SIZE",
    "JANUS_IMAGE_TOKEN_NUM",
    "JANUS_IMAGE_VOCAB_SIZE",
    "JANUS_R1_SEGMENTS",
    "JanusProConfig",
    "JanusProModel",
    "JanusProReplayCore",
    "JanusProReplayModel",
]
