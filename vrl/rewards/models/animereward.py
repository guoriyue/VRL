"""AnimeReward visual-quality head as a model-backed RewardModel.

The quality dimension of AnimeReward (arXiv 2504.10044, released inside
``IndexTeam/Index-anisora``): an Idefics2-8B sequence-classification regression
head trained on anime video frames to score clearness, resolution, brightness,
and color. Returns ``{"animereward_quality": x}``; drive it with
``score_key="animereward_quality"``.

Two deliberate deviations from the paper's video setting, both forced by scoring
STILLS from an image model:

- The head consumes ``MAX_NUM_FRAMES`` sampled frames. A still is replicated
  into a static clip, which is the closest in-distribution encoding of "this
  image, held". Measured on anima output this still separates cleanly:
  on-prompt anime 0.696 +/- 0.027 vs off-prompt collapse 0.470 +/- 0.081
  (Cohen's d = 3.7), where LAION-aesthetic and PickScore both scored the two
  sets identically.
- The five video-only dimensions (smoothness, motion, image-video consistency,
  and cross-frame character identity) are undefined for a single still and are
  not loaded.

The head is Mantis's Idefics2 fork, which targets transformers 4.x, while this
repo runs 5.x. It therefore loads only inside the standalone reward service run
from its own pinned environment (see ``configs/reward_service/animereward.yaml``),
mirroring the VBench precedent in ``[tool.uv] conflicts``. Importing this module
under the repo's transformers is fine; ``_load_module`` is where the pin bites.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.rewards.models.base import TorchRewardModel
from vrl.utils.media import to_uint8

# Verbatim from the release's reward_infer.py quality branch. The head is a
# regressor -- the "output a number from [1,2,3,4,5]" rubric is the training
# prompt, not a parsed instruction -- so this text must not be paraphrased.
REGRESSION_QUERY_PROMPT = """
Suppose you are an expert in judging and evaluating the quality of AI-generated videos,
please watch the following frames of a given video and see the text prompt for generating the video,
then give scores from 7 different dimensions:
(1) visual quality: the quality of the video in terms of clearness, resolution, brightness, and color
(2) object consistency, the consistency of objects or humans in video
please judge whether the video is consistent with the text prompt.
For each dimension, output a number from [1,2,3,4,5],
in which '1' means 'Bad', '2' means 'Poor', '3' means 'Fair', '4' means 'Good', '5' means 'Perfect'.
"""

# reward_infer.py samples this many frames regardless of the 24-frame checkpoint name.
MAX_NUM_FRAMES = 8

# The head regresses onto a 0-100 scale; /100 puts it in [0, 1] like pickscore.
_SCORE_SCALE = 100.0


class AnimeRewardQualityModel(TorchRewardModel):
    """AnimeReward visual-quality regressor (Idefics2-8B), normalised to ~[0, 1]."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        super().__init__(worker_config)
        self.model_name = str(self.worker_config.get("model_name", "")).strip()
        if not self.model_name:
            raise ValueError(
                "AnimeRewardQualityModel requires model_name (the quality "
                "checkpoint-final directory)",
            )
        self.num_frames = int(self.worker_config.get("num_frames", MAX_NUM_FRAMES))
        if self.num_frames < 1:
            raise ValueError("AnimeRewardQualityModel num_frames must be >= 1")
        self._processor: Any = None

    def _load_module(self) -> Any:
        # Mantis's fork, not transformers' Idefics2: the released checkpoint
        # declares architectures=["Idefics2ForSequenceClassification"], which
        # only exists there.
        from mantis.models.idefics2 import Idefics2ForSequenceClassification
        from transformers import AutoProcessor

        # AutoProcessor owns CPU tokenization/image transforms only; the returned
        # module is the CUDA state the pool's build frame must capture.
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        return (
            Idefics2ForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
            )
            .eval()
            .to(self.device)
        )

    def _as_frames(self, media: Any) -> list[Any]:
        """Normalise any supported media payload to a list of PIL frames."""

        import numpy as np
        import torch
        from PIL import Image

        if isinstance(media, torch.Tensor):
            arr = to_uint8(media).cpu().numpy()
            if arr.ndim == 3:
                arr = arr[None]
            if arr.ndim == 4:
                arr = arr.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            elif arr.ndim == 5:
                arr = arr[0].transpose(0, 2, 3, 1)
            frames = [Image.fromarray(a) for a in arr]
        elif isinstance(media, np.ndarray):
            frames = (
                [Image.fromarray(media)]
                if media.ndim == 3
                else [Image.fromarray(a) for a in media]
            )
        elif isinstance(media, Image.Image):
            frames = [media]
        else:
            raise TypeError(f"AnimeRewardQualityModel cannot score media of type {type(media)!r}")
        return [frame.convert("RGB") for frame in frames]

    def score_media(self, *, media: Any, prompt: str) -> Mapping[str, float]:
        import torch

        frames = self._as_frames(media)
        if not frames:
            raise ValueError("AnimeRewardQualityModel received an empty media payload")
        # A still (or a clip shorter than the head's window) is held for the
        # whole window rather than zero-padded, keeping every frame in-distribution.
        if len(frames) < self.num_frames:
            frames = [frames[len(frames) * i // self.num_frames] for i in range(self.num_frames)]
        elif len(frames) > self.num_frames:
            step = len(frames) / self.num_frames
            frames = [frames[int(i * step)] for i in range(self.num_frames)]

        module = self._module_for_inference()
        eval_prompt = REGRESSION_QUERY_PROMPT + f"\ntext prompt: {prompt}\n"
        content: list[dict[str, Any]] = [{"type": "image"} for _ in frames]
        content.append({"type": "text", "text": eval_prompt})
        text = self._processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
        )
        inputs = self._processor(text=text, images=frames, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = module(**inputs).logits
        return {"animereward_quality": float(logits[0].item()) / _SCORE_SCALE}


__all__ = ["MAX_NUM_FRAMES", "REGRESSION_QUERY_PROMPT", "AnimeRewardQualityModel"]
