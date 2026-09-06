"""Shared prelude for generative Qwen-VL video judges.

VideoScore2 and the Cosmos3 reasoner are the same shape: load a Qwen-VL
processor + model, build one chat turn holding the video and a rubric, run the
processor, generate one judgement, decode it, and parse a fixed-format score
line. Everything up to the parse lives here once; a judge supplies its prompts,
its model loader, its generation recipe, and its parser.

Placement: ``worker_config.device_map`` lets transformers shard a large judge;
inputs then follow ``self.model.device`` (the first shard), which is also the
single pinned device when ``device_map`` is unset.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from vrl.models.dtypes import resolve_torch_dtype
from vrl.rewards.inference import RewardInferenceArtifact
from vrl.utils.logging import init_logger, kv

logger = init_logger(__name__)

_DEFAULT_FPS = 2.0
_DEFAULT_MAX_NEW_TOKENS = 1024


class QwenVLVideoJudge:
    """Load a Qwen-VL judge and score one (prompt, video) pair per call."""

    family: str = "Qwen-VL judge"

    def __init__(self, worker_config: Mapping[str, Any], *, model_root: Path) -> None:
        self.worker_config = dict(worker_config)
        self.model_root = model_root
        self.dtype = resolve_torch_dtype(str(self.worker_config.get("dtype", "bfloat16")))
        self.device = str(self.worker_config.get("device", "cuda:0"))
        self.fps = float(self.worker_config.get("fps", _DEFAULT_FPS))
        self.max_new_tokens = int(
            self.worker_config.get("max_new_tokens", _DEFAULT_MAX_NEW_TOKENS),
        )
        # Optional per-frame pixel bounds for qwen smart_resize, matching the
        # Kling reward's knobs; None keeps the processor default. qwen-vl-utils
        # asserts max >= min, and its video default min is well above a small
        # max, so a caller who bounds one side must bound the other too.
        max_frame_pixels = self.worker_config.get("max_frame_pixels")
        min_frame_pixels = self.worker_config.get("min_frame_pixels")
        self.max_frame_pixels = None if max_frame_pixels is None else int(max_frame_pixels)
        self.min_frame_pixels = None if min_frame_pixels is None else int(min_frame_pixels)
        if (
            self.max_frame_pixels is not None
            and self.min_frame_pixels is not None
            and self.min_frame_pixels > self.max_frame_pixels
        ):
            raise ValueError(
                f"{self.family}: min_frame_pixels={self.min_frame_pixels} exceeds "
                f"max_frame_pixels={self.max_frame_pixels}",
            )
        self.local_files_only = bool(self.worker_config.get("local_files_only", False))
        # Large judges: allow sharding / attention backend without a code change.
        self.attn_implementation = self.worker_config.get("attn_implementation")
        self.device_map = self.worker_config.get("device_map")

        logger.info(
            "loading %s %s",
            self.family,
            kv(root=self.model_root, device=self.device, dtype=self.dtype, fps=self.fps),
        )
        from transformers import AutoProcessor

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
        model = self._load_model(load_kwargs)
        model.eval()
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", None) or processor
        # device_map handles placement itself; otherwise pin to a single device.
        self.model = model if self.device_map is not None else model.to(self.device)

    def _load_model(self, load_kwargs: dict[str, Any]) -> Any:
        """Build the judge model from ``self.model_root`` with the shared load kwargs."""

        raise NotImplementedError

    def _messages(self, video_path: str, prompt: str) -> list[dict[str, Any]]:
        """The chat turns for one judgement (system rubric + video + user text)."""

        raise NotImplementedError

    def _parse(self, decoded: str, generated: Any, generated_ids: list[int]) -> dict[str, float]:
        """Public scores from the judge's decoded text (and, optionally, its logits)."""

        raise NotImplementedError

    def _generate_kwargs(self) -> dict[str, Any]:
        """The decoding recipe; greedy by default."""

        return {"do_sample": False}

    def __call__(self, artifact: RewardInferenceArtifact) -> dict[str, float]:
        prompt, video_path = artifact.require_prompt_and_video_path(family=self.family)
        return self._score_video(video_path, prompt)

    def _video_content(self, video_path: str) -> dict[str, Any]:
        content: dict[str, Any] = {
            "type": "video",
            "video": f"file://{video_path}",
            "fps": self.fps,
        }
        if self.max_frame_pixels is not None:
            content["max_pixels"] = self.max_frame_pixels
        if self.min_frame_pixels is not None:
            content["min_pixels"] = self.min_frame_pixels
        return content

    def _judge_inputs(self, video_path: str, prompt: str) -> tuple[Any, list[dict[str, Any]]]:
        """Processor output for one judgement, on the model's device."""

        from qwen_vl_utils import process_vision_info

        messages = self._messages(video_path, prompt)
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
        return inputs, messages

    def _score_video(self, video_path: str, prompt: str) -> dict[str, float]:
        inputs, _ = self._judge_inputs(video_path, prompt)
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                **self._generate_kwargs(),
            )
        prompt_len = int(inputs["input_ids"].shape[1])
        generated_ids = generated.sequences[0][prompt_len:].tolist()
        decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return self._parse(decoded, generated, generated_ids)


__all__ = ["QwenVLVideoJudge"]
