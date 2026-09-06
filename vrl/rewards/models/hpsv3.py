"""Repo-owned HPSv3 reward model loading and inference.

Portions of this module are adapted from MizzenAI/HPSv3 under the MIT license:
the Qwen2-VL reward-head model (``hpsv3/model/qwen2vl_trainer.py``,
``Qwen2VLRewardModelBT``), its inference batch preparation
(``hpsv3/inference.py``), and the checkpoint key remap for post-4.52
transformers layouts. VRL owns this narrowed implementation so reward scoring
does not depend on an HPSv3 checkout: the upstream package pins deepspeed/trl,
and its ``TrainingConfig`` subclasses ``transformers.TrainingArguments`` whose
construction touches accelerate global state (upstream issues #6/#2) — a
landmine next to an RL trainer's own ``Accelerator``.

HPSv3 is an *image* preference model: Qwen2-VL-7B with a ranknet head pooled
at one ``<|Reward|>`` special token, emitting ``(mu, sigma)`` per image (we
score with ``mu``; scores are unbounded, typically -1..+12, higher is better —
upstream issue #9). This wrapper scores a *video* by scoring frames
independently and aggregating: descending sort, mean of the top
``top_frame_fraction`` of frame scores — the semantics Flash-GRPO's reward
server applies (``flow_grpo/rewards.py``, ``video_hpsv3_remote``: sort
reverse, average the top 30%). Temporal coherence is deliberately NOT in this
reward; pair it with a motion-aware component if motion quality matters.

The loader supports exactly one checkpoint family — ``MizzenAI/HPSv3``'s
``HPSv3.safetensors`` over the ``Qwen/Qwen2-VL-7B-Instruct`` base with the
``HPSv3_7B.yaml`` architecture (ranknet head, special-token pooling,
``output_dim=2``) — mirroring how the Kling loader pins KlingTeam/VideoReward.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration

from vrl.models.dtypes import resolve_torch_dtype
from vrl.rewards.assets.hpsv3_prompts import (
    HPSV3_REWARD_SPECIAL_TOKEN,
    build_hpsv3_frame_prompt,
)
from vrl.rewards.inference import RewardInferenceArtifact
from vrl.rewards.models.hub import resolve_model_root
from vrl.utils.logging import init_logger, kv

logger = init_logger(__name__)

_DEFAULT_REWARD_MODEL = "MizzenAI/HPSv3"
_DEFAULT_BASE_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
_CHECKPOINT_FILENAME = "HPSv3.safetensors"
# HPSv3_7B.yaml pins min_pixels == max_pixels == 256 * 28 * 28; inference uses
# the same fixed budget, so every frame is resized to this token count.
_FRAME_PIXELS = 256 * 28 * 28
# HPSv3_7B.yaml: output_dim 2 (mu, sigma), loss_type "uncertainty".
_OUTPUT_DIM = 2
_DEFAULT_TOP_FRAME_FRACTION = 0.3
_DEFAULT_FRAMES_PER_FORWARD = 8


class HPSv3Qwen2VLRewardModel(Qwen2VLForConditionalGeneration):
    """Qwen2-VL backbone with HPSv3's ranknet head pooled at ``<|Reward|>``."""

    def __init__(
        self,
        config: Any,
        special_token_ids: list[int] | None = None,
        # Transformers forwards loading kwargs (use_cache, ...) into the model
        # constructor; they describe generation behavior this reward head never
        # uses, so absorb them (same guard as the Kling reward model).
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__(config)
        hidden_size = config.get_text_config().hidden_size
        # HPSv3_7B.yaml rm_head_type "ranknet" with rm_head_kwargs null — the
        # upstream default ranknet stack. Sequential indices (0/3/5) are part
        # of the checkpoint's key namespace; do not restructure.
        self.rm_head = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(1024, 16),
            nn.ReLU(),
            nn.Linear(16, _OUTPUT_DIM),
        )
        # Upstream keeps the head in float32 while the backbone runs bf16.
        self.rm_head.to(torch.float32)
        self.special_token_ids = special_token_ids

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        kwargs.pop("return_dict", None)
        if input_ids is None or self.special_token_ids is None:
            raise ValueError("HPSv3 special-token pooling requires input_ids")
        # Replicate the upstream forward exactly: merge image embeddings by
        # hand, then call the backbone with input_ids=None and no position_ids.
        # That call path never reaches Qwen2-VL's M-RoPE index computation, so
        # the checkpoint was TRAINED under plain sequential positions — routing
        # pixel_values through the modern merged forward instead (which does
        # compute M-RoPE) shifts scores by up to ~0.9 and flips near-tie
        # rankings vs the upstream inferencer on identical inputs (measured).
        # Faithfulness to the training-time quirk beats architectural hygiene.
        inputs_embeds = self.model.language_model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.model.visual.dtype)
            image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)
            # Transformers 5.x returns BaseModelOutputWithPooling from the vision
            # tower; the merged patch embeddings this scatter needs are its
            # pooler_output (`merger(hidden_states)`), not last_hidden_state.
            # Older versions returned that tensor directly.
            image_embeds = getattr(image_embeds, "pooler_output", image_embeds)
            image_mask = (
                (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs[0]
        logits = self.rm_head(hidden_states.to(torch.float32))
        special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for special_token_id in self.special_token_ids:
            special_token_mask = special_token_mask | (input_ids == special_token_id)
        pooled_logits = logits[special_token_mask, ...]
        return {"logits": pooled_logits.view(input_ids.shape[0], -1)}


class HPSv3Model:
    """Load HPSv3 and score one (prompt, video) pair per call, frame by frame."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        self.worker_config = dict(worker_config)
        self.model_root = resolve_model_root(
            self.worker_config,
            default_model=_DEFAULT_REWARD_MODEL,
            family="HPSv3",
        )
        self.checkpoint_path = self.model_root / _CHECKPOINT_FILENAME
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"HPSv3 model root must contain {_CHECKPOINT_FILENAME}: {self.model_root}",
            )
        base_model_path = str(self.worker_config.get("base_model_path", "")).strip()
        self.base_model = base_model_path or _DEFAULT_BASE_MODEL
        self.dtype = resolve_torch_dtype(str(self.worker_config.get("dtype", "bfloat16")))
        self.device = str(self.worker_config.get("device", "cuda:0"))
        self.local_files_only = bool(self.worker_config.get("local_files_only", False))
        self.top_frame_fraction = float(
            self.worker_config.get("top_frame_fraction", _DEFAULT_TOP_FRAME_FRACTION),
        )
        if not 0.0 < self.top_frame_fraction <= 1.0:
            raise ValueError(
                f"HPSv3 top_frame_fraction must be in (0, 1], got {self.top_frame_fraction}",
            )
        self.frames_per_forward = int(
            self.worker_config.get("frames_per_forward", _DEFAULT_FRAMES_PER_FORWARD),
        )
        if self.frames_per_forward < 1:
            raise ValueError(
                f"HPSv3 frames_per_forward must be >= 1, got {self.frames_per_forward}",
            )
        # None scores every frame (Flash-GRPO parity); an int subsamples
        # uniformly to bound scoring cost on long videos.
        num_frames = self.worker_config.get("num_frames")
        self.num_frames = None if num_frames is None else int(num_frames)
        disable_flash_attn2 = self.worker_config.get("disable_flash_attn2")
        if disable_flash_attn2 is None:
            disable_flash_attn2 = importlib.util.find_spec("flash_attn") is None
        self.attn_implementation = "sdpa" if bool(disable_flash_attn2) else "flash_attention_2"

        logger.info(
            "loading HPSv3 %s",
            kv(
                root=self.model_root,
                base_model=self.base_model,
                device=self.device,
                dtype=self.dtype,
                attn=self.attn_implementation,
                top_frame_fraction=self.top_frame_fraction,
            ),
        )
        model, processor = self._build_model_and_processor()
        self._load_checkpoint(model)
        model.eval()
        self.model = model.to(self.device)
        self.processor = processor
        logger.info("loaded HPSv3 model %s", kv(device=self.device))

    def _build_model_and_processor(self) -> tuple[HPSv3Qwen2VLRewardModel, Any]:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            self.base_model,
            padding_side="right",
            local_files_only=self.local_files_only,
        )
        processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": [HPSV3_REWARD_SPECIAL_TOKEN]},
        )
        special_token_ids = processor.tokenizer.convert_tokens_to_ids(
            [HPSV3_REWARD_SPECIAL_TOKEN],
        )
        model = HPSv3Qwen2VLRewardModel.from_pretrained(
            self.base_model,
            special_token_ids=special_token_ids,
            torch_dtype=self.dtype,
            attn_implementation=self.attn_implementation,
            local_files_only=self.local_files_only,
        )
        model.resize_token_embeddings(len(processor.tokenizer))
        # from_pretrained's torch_dtype cast covers freshly-initialized modules
        # too; restore the head's deliberate float32 before checkpoint load.
        model.rm_head.to(torch.float32)
        return model, processor

    def _load_checkpoint(self, model: HPSv3Qwen2VLRewardModel) -> None:
        import safetensors.torch

        state_dict = safetensors.torch.load_file(str(self.checkpoint_path), device="cpu")
        state_dict = _remap_qwen2vl_state_dict(state_dict, model.state_dict())
        model.load_state_dict(state_dict, strict=True)

    def __call__(self, artifact: RewardInferenceArtifact) -> dict[str, float]:
        prompt, video_path = artifact.require_prompt_and_video_path(family="HPSv3")
        return self._score_video(video_path, prompt)

    def _score_video(self, video_path: str, prompt: str) -> dict[str, float]:
        from vrl.utils.media import read_video_frames

        frames = read_video_frames(video_path, self.num_frames)
        images = _frames_to_pil(frames)
        mu_scores: list[float] = []
        for start in range(0, len(images), self.frames_per_forward):
            chunk = images[start : start + self.frames_per_forward]
            batch = self._prepare_batch(chunk, prompt)
            with torch.no_grad():
                logits = self.model(return_dict=True, **batch)["logits"]
            # Column 0 is mu; column 1 is the uncertainty head's sigma.
            mu_scores.extend(float(row[0].item()) for row in logits)
        return _aggregate_frame_scores(mu_scores, self.top_frame_fraction)

    def _prepare_batch(self, images: list[Any], prompt: str) -> Mapping[str, Any]:
        from qwen_vl_utils import process_vision_info

        text = build_hpsv3_frame_prompt(prompt)
        message_list = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                            "min_pixels": _FRAME_PIXELS,
                            "max_pixels": _FRAME_PIXELS,
                        },
                        {"type": "text", "text": text},
                    ],
                },
            ]
            for image in images
        ]
        image_inputs, _ = process_vision_info(message_list)
        batch = self.processor(
            text=self.processor.apply_chat_template(
                message_list,
                tokenize=False,
                add_generation_prompt=True,
            ),
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        return batch.to(self.device)


def _frames_to_pil(frames: torch.Tensor) -> list[Any]:
    """Convert a ``[T,H,W,3]`` float ``[0,1]`` frame stack to PIL images."""

    from PIL import Image

    array = (frames.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()
    return [Image.fromarray(frame) for frame in array]


def _aggregate_frame_scores(scores: list[float], top_fraction: float) -> dict[str, float]:
    """Per-frame mu scores -> the public score keys.

    ``top_frame_mean`` is the Flash-GRPO objective: descending sort, mean of
    the top ``top_fraction`` of frames (at least one). ``frame_mean`` and
    ``frame_min`` are whole-video diagnostics: a rising ``top_frame_mean``
    with a falling ``frame_min`` is the "best frames improve while worst
    frames rot" reward-hacking signature.
    """

    if not scores:
        raise ValueError("HPSv3 scored no frames")
    ordered = sorted(scores, reverse=True)
    top_count = max(1, int(len(ordered) * top_fraction))
    top = ordered[:top_count]
    return {
        "top_frame_mean": sum(top) / len(top),
        "frame_mean": sum(scores) / len(scores),
        "frame_min": min(scores),
    }


def _remap_qwen2vl_state_dict(
    state_dict: dict[str, torch.Tensor],
    model_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Rename pre-4.52 Qwen2-VL checkpoint keys to the nested current layout.

    The published checkpoint predates the transformers rename that moved
    ``model.*`` under ``model.language_model.*`` and ``visual.*`` under
    ``model.visual.*`` (upstream issues #15/#29). No-op when either side
    already agrees.
    """

    target_is_nested = any(key.startswith("model.language_model.") for key in model_state)
    source_is_nested = any("language_model" in key for key in state_dict)
    if not target_is_nested or source_is_nested:
        return state_dict
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("visual."):
            remapped[f"model.{key}"] = value
        elif key.startswith("model."):
            remapped[f"model.language_model.{key[len('model.') :]}"] = value
        else:
            remapped[key] = value
    return remapped


__all__ = ["HPSv3Model", "HPSv3Qwen2VLRewardModel"]
