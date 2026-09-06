"""Cosmos3 reasoner reward model for the in-process reward runtime.

Uses the **reasoner / understanding tower** of NVIDIA Cosmos3 (``cosmos3_omni``)
— a Qwen3-VL VLM — as a generative judge over a generated robot/physical-AI
video. Cosmos3's generator tower is a separate diffusion model (not this file);
here the reasoner reads the video + task instruction and prints four physical-AI
axes (1-5 integers), which we parse:

* ``task_success``           — does the action complete the instructed goal.
* ``contact_realism``        — grasps/contacts/forces are physically believable.
* ``temporal_consistency``   — object identity and motion are stable over frames.
* ``physical_plausibility``  — no clipping, teleporting, or physics violations.

This is the "domain judge" the robotic-data-factory sprint asks for (Kling
``overall_reward`` is video-quality, not task/contact-aware). It mirrors
``videoscore2.py``: a generative Qwen-VL judge, greedily decoded once, parsed
with a fixed-format regex. Public score keys are exactly the four axes plus
``overall`` (their mean) — the free-text wording never leaks as a score key.

CHECKPOINT NOTE — the raw ``nvidia/Cosmos3-Nano`` checkpoint is a **flat-key
unified** omni checkpoint (reasoner LLM/visual keys + ``*_moe_gen`` generator-
tower keys in one flat namespace). ``Qwen3VLForConditionalGeneration`` expects
the **nested** Qwen3-VL key layout, so the unified checkpoint does NOT load
HF-direct. ``worker_config.checkpoint_layout`` selects how the reasoner weights
are obtained:

* ``remapped`` — ``model_path`` points at a pre-remapped, reasoner-only Qwen3-VL
  checkpoint (produced offline by the vLLM ``cosmos3.py`` WeightsMapper: flat->
  nested, drop the generator/audio/action towers). Loaded cleanly here.
* unset / anything else — ``__init__`` raises with the remap prerequisite rather
  than attempting a doomed raw ``from_pretrained`` that mismatches every key.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from vrl.models.dtypes import resolve_torch_dtype
from vrl.rewards.assets.video_judge_prompts import (
    COSMOS3_SCORE_REGEX,
    COSMOS3_SYSTEM_PROMPT,
    COSMOS3_USER_TEMPLATE,
)
from vrl.rewards.inference import RewardInferenceArtifact
from vrl.utils.logging import init_logger, kv

logger = init_logger(__name__)

_DEFAULT_FPS = 2.0
_DEFAULT_MAX_NEW_TOKENS = 1024


class Cosmos3ReasonerRewardModel:
    """Load the Cosmos3 reasoner (Qwen3-VL) and judge one (prompt, video) pair."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        self.worker_config = dict(worker_config)
        # Generator tower is a separate diffusion model; only the reasoner judges
        # here, and the unified checkpoint will not load HF-direct (see module
        # docstring). Require an explicit, supported layout.
        self.checkpoint_layout = (
            str(
                self.worker_config.get("checkpoint_layout", ""),
            )
            .strip()
            .lower()
        )
        self.dtype = resolve_torch_dtype(str(self.worker_config.get("dtype", "bfloat16")))
        self.device = str(self.worker_config.get("device", "cuda:0"))
        self.fps = float(self.worker_config.get("fps", _DEFAULT_FPS))
        self.max_new_tokens = int(
            self.worker_config.get("max_new_tokens", _DEFAULT_MAX_NEW_TOKENS),
        )
        max_frame_pixels = self.worker_config.get("max_frame_pixels")
        self.max_frame_pixels = None if max_frame_pixels is None else int(max_frame_pixels)
        self.local_files_only = bool(self.worker_config.get("local_files_only", False))
        # 16B reasoner: allow sharding / attention backend without a code change.
        self.attn_implementation = self.worker_config.get("attn_implementation")
        self.device_map = self.worker_config.get("device_map")

        if self.checkpoint_layout != "remapped":
            raise ValueError(
                "Cosmos3 reasoner judge requires worker_config.checkpoint_layout='remapped' "
                "with model_path pointing at a pre-remapped, reasoner-only Qwen3-VL checkpoint. "
                "The raw nvidia/Cosmos3-Nano unified checkpoint is flat-key (reasoner + "
                "*_moe_gen generator tower in one namespace) and does NOT load under "
                "Qwen3VLForConditionalGeneration. Produce the remapped checkpoint offline via "
                "the vLLM cosmos3.py WeightsMapper (flat->nested, drop generator/audio/action "
                f"towers). Got checkpoint_layout={self.checkpoint_layout!r}.",
            )
        # 'remapped' means an offline-produced LOCAL dir; reward_model_name's
        # snapshot_download fallback would silently pull the raw flat-key unified
        # repo (which does NOT load), failing late after a multi-GB download. Require
        # model_path so the unfilled-config case fails fast with an actionable error.
        if not str(self.worker_config.get("model_path", "")).strip():
            raise ValueError(
                "Cosmos3 reasoner checkpoint_layout='remapped' requires a non-empty "
                "worker_config.model_path pointing at the offline pre-remapped, reasoner-only "
                "Qwen3-VL checkpoint dir. Refusing to snapshot_download reward_model_name "
                "(the raw unified nvidia/Cosmos3-Nano checkpoint is flat-key and will not load).",
            )

        # The ``remapped`` layout's ``model_path`` (validated non-empty above) is
        # the offline-produced reasoner-only Qwen3-VL dir. Resolve it directly with
        # an existence check; there is no snapshot_download fallback (the raw
        # unified repo does not load — see the guard above).
        model_path = str(self.worker_config.get("model_path", "")).strip()
        self.model_root = Path(model_path).expanduser().resolve()
        if not self.model_root.exists():
            raise FileNotFoundError(
                f"Cosmos3 reasoner model_path missing: {self.model_root}",
            )
        logger.info(
            "loading Cosmos3 reasoner judge %s",
            kv(
                root=self.model_root,
                device=self.device,
                dtype=self.dtype,
                fps=self.fps,
                layout=self.checkpoint_layout,
            ),
        )

        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            str(self.model_root),
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        load_kwargs: dict[str, Any] = {
            "torch_dtype": self.dtype,
            "local_files_only": self.local_files_only,
        }
        if self.attn_implementation:
            load_kwargs["attn_implementation"] = str(self.attn_implementation)
        if self.device_map is not None:
            load_kwargs["device_map"] = self.device_map
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(self.model_root),
            **load_kwargs,
        )
        model.eval()
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", None) or processor
        # device_map handles placement itself; otherwise pin to a single device.
        self.model = model if self.device_map is not None else model.to(self.device)

    def __call__(
        self,
        artifact: RewardInferenceArtifact,
    ) -> dict[str, float]:
        prompt, video_path = artifact.require_prompt_and_video_path(
            family="Cosmos3 reasoner judge",
        )
        return self._score_video(video_path, prompt)

    def _score_video(self, video_path: str, prompt: str) -> dict[str, float]:
        from qwen_vl_utils import process_vision_info

        video_content: dict[str, Any] = {
            "type": "video",
            "video": f"file://{video_path}",
            "fps": self.fps,
        }
        if self.max_frame_pixels is not None:
            video_content["max_pixels"] = self.max_frame_pixels
        messages = [
            {"role": "system", "content": COSMOS3_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    video_content,
                    {
                        "type": "text",
                        "text": COSMOS3_USER_TEMPLATE.format(prompt=prompt),
                    },
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
            )
        prompt_len = int(inputs["input_ids"].shape[1])
        generated_ids = generated.sequences[0][prompt_len:].tolist()
        decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        parsed = _parse_integer_scores(decoded)
        if parsed is None:
            raise ValueError(
                "Cosmos3 reasoner produced no parseable score line; "
                f"output head was: {decoded[:200]!r}",
            )
        return _normalize_scores(*parsed)


def _parse_integer_scores(text: str) -> tuple[int, int, int, int] | None:
    """Extract the four 1-5 integer axes from the judge's generated text."""

    match = COSMOS3_SCORE_REGEX.search(text)
    if match is None:
        return None
    scores = tuple(int(match.group(i)) for i in (1, 2, 3, 4))
    if any(not (1 <= value <= 5) for value in scores):
        return None
    return scores  # type: ignore[return-value]


def _normalize_scores(
    task_success: float,
    contact_realism: float,
    temporal_consistency: float,
    physical_plausibility: float,
) -> dict[str, float]:
    """Map the four axes to the public score keys plus their mean ``overall``.

    This dict is the public scoring contract: only the documented keys, so a
    config cannot select an undocumented upstream key as ``score_key``.
    """

    task_success = float(task_success)
    contact_realism = float(contact_realism)
    temporal_consistency = float(temporal_consistency)
    physical_plausibility = float(physical_plausibility)
    return {
        "task_success": task_success,
        "contact_realism": contact_realism,
        "temporal_consistency": temporal_consistency,
        "physical_plausibility": physical_plausibility,
        "overall": (task_success + contact_realism + temporal_consistency + physical_plausibility)
        / 4.0,
    }


__all__ = ["Cosmos3ReasonerRewardModel"]
