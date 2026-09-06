"""UnifiedReward-2.0 video reward model for the in-process reward runtime.

Wraps ``CodeGoat24/UnifiedReward-2.0-qwen-7b`` - a Qwen2.5-VL-7B general
image/video reward - for pointwise video scoring. Following the upstream
``UnifiedReward-2.0-inference`` pointwise script, we sample N frames, pass them
as images, and ask for three axes the model is trained to emit (each a 1-5
float, so the signal is already continuous - no token-probability trick needed):

* ``alignment`` - how well the video matches the caption.
* ``physics``   - gravity / movement / collision / interaction plausibility.
* ``style``     - visual appeal independent of the caption.

This is the sprint's "second judge / rubric reranker": it reads a natural-language
question, so a custom rubric (e.g. "same character, same dress, plausible skirt
motion") can steer its attention via ``worker_config.rubric_path`` - a YAML asset,
never a hardcoded business-prompt constant in this module. The rubric only changes
the framing text; the three-axis output grammar (which we parse) is fixed.

Public score keys: ``alignment`` / ``physics`` / ``style`` / ``overall`` (mean).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from vrl.models.dtypes import resolve_torch_dtype
from vrl.rewards.assets.video_judge_prompts import (
    UNIFIED_REWARD_VIDEO_AXIS_PATTERNS,
    UNIFIED_REWARD_VIDEO_PROBLEM_TEMPLATE,
)
from vrl.rewards.inference import RewardInferenceArtifact
from vrl.rewards.models.hub import resolve_model_root
from vrl.utils.logging import init_logger, kv

logger = init_logger(__name__)

_DEFAULT_REWARD_MODEL = "CodeGoat24/UnifiedReward-2.0-qwen-7b"
_DEFAULT_NUM_FRAMES = 16
_DEFAULT_MAX_NEW_TOKENS = 256


class UnifiedRewardVideoModel:
    """Load UnifiedReward-2.0 and pointwise-score one (prompt, video) per call."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        self.worker_config = dict(worker_config)
        self.model_root = resolve_model_root(
            self.worker_config,
            default_model=_DEFAULT_REWARD_MODEL,
            family="UnifiedReward-2.0",
        )
        self.dtype = resolve_torch_dtype(str(self.worker_config.get("dtype", "bfloat16")))
        self.device = str(self.worker_config.get("device", "cuda:0"))
        self.num_frames = int(self.worker_config.get("num_frames", _DEFAULT_NUM_FRAMES))
        self.max_new_tokens = int(
            self.worker_config.get("max_new_tokens", _DEFAULT_MAX_NEW_TOKENS),
        )
        self.local_files_only = bool(self.worker_config.get("local_files_only", False))
        self.problem_template = _load_rubric(
            str(self.worker_config.get("rubric_path", "")).strip(),
        )

        logger.info(
            "loading UnifiedReward-2.0 %s",
            kv(root=self.model_root, device=self.device, dtype=self.dtype, frames=self.num_frames),
        )

        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            str(self.model_root),
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            str(self.model_root),
            torch_dtype=self.dtype,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        model.eval()
        self.model = model.to(self.device)

    def __call__(
        self,
        artifact: RewardInferenceArtifact,
    ) -> dict[str, float]:
        # The rubric is caption-conditioned, so an empty prompt must fail fast
        # (drift fix: this raise was silently missing here).
        prompt, video_path = artifact.require_prompt_and_video_path(
            family="UnifiedReward-2.0",
        )
        frames = _sample_frames(video_path, self.num_frames)
        if not frames:
            raise ValueError(
                f"UnifiedReward-2.0 extracted no frames from {video_path!r}",
            )
        problem = self.problem_template.format(prompt=prompt)
        messages = [
            {
                "role": "user",
                "content": [
                    *({"type": "image", "image": frame} for frame in frames),
                    {"type": "text", "text": problem},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            images=frames,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        prompt_len = int(inputs["input_ids"].shape[1])
        decoded = self.processor.tokenizer.decode(
            generated[0][prompt_len:],
            skip_special_tokens=True,
        )
        return _parse_axis_scores(decoded)


def _parse_axis_scores(text: str) -> dict[str, float]:
    """Parse declared score ranges and normalize every axis to the public 1-5 scale."""

    scores: dict[str, float] = {}
    for axis, pattern in UNIFIED_REWARD_VIDEO_AXIS_PATTERNS.items():
        match = pattern.search(text)
        if match is None:
            raise ValueError(
                f"UnifiedReward-2.0 output missing {axis!r} score; head was: {text[:200]!r}",
            )
        lower, upper, value = (float(group) for group in match.groups())
        if lower >= upper:
            raise ValueError(
                f"UnifiedReward-2.0 output has invalid {axis!r} score range "
                f"{lower:g}-{upper:g}; head was: {text[:200]!r}",
            )
        if not lower <= value <= upper:
            raise ValueError(
                f"UnifiedReward-2.0 output has out-of-range {axis!r} score "
                f"{value:g} for declared range {lower:g}-{upper:g}; head was: {text[:200]!r}",
            )
        # The model occasionally emits a self-consistent 1-10 label despite the
        # upstream 1-5 prompt. Respecting the declared range preserves ordering
        # while keeping the reward contract comparable across generations.
        scores[axis] = 1.0 + 4.0 * (value - lower) / (upper - lower)
    scores["overall"] = sum(scores[a] for a in ("alignment", "physics", "style")) / 3.0
    return scores


def _load_rubric(rubric_path: str) -> str:
    """Return the problem template: a YAML ``problem_template`` override or the default."""

    if not rubric_path:
        return UNIFIED_REWARD_VIDEO_PROBLEM_TEMPLATE
    from omegaconf import OmegaConf

    rubric = OmegaConf.load(Path(rubric_path).expanduser().resolve())
    template = str(OmegaConf.select(rubric, "problem_template", default="")).strip()
    if not template:
        raise ValueError(f"rubric {rubric_path!r} has no non-empty 'problem_template'")
    if "{prompt}" not in template:
        raise ValueError(f"rubric {rubric_path!r} problem_template must contain a {{prompt}} slot")
    return template


def _sample_frames(video_path: str, num_frames: int) -> list[Any]:
    """Uniformly sample ``num_frames`` RGB PIL frames (upstream cv2 sampling)."""

    import cv2
    from PIL import Image

    capture = cv2.VideoCapture(video_path)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        return []
    wanted = {int(i * total / num_frames) for i in range(num_frames)}
    frames: list[Any] = []
    index = 0
    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        index += 1
        if len(frames) >= num_frames:
            break
    capture.release()
    return frames


__all__ = ["UnifiedRewardVideoModel"]
