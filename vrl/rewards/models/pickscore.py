"""PickScore as a model-backed RewardModel (CLIP ViT-H/14 preference model).

Scoring logic ported from the in-process PickScoreReward; loading goes through
TorchRewardModel so device/dtype/lazy-load is shared. Returns ``{"pickscore": x}``;
drive it with ``score_key="pickscore"``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.rewards.models.base import TorchRewardModel
from vrl.utils.media import to_uint8


class PickScoreRewardModel(TorchRewardModel):
    """PickScore v1 reward model. Scores normalised by /26 to roughly [0, 1]."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        super().__init__(worker_config)
        self.processor_name = str(
            self.worker_config.get(
                "processor_name", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            ),
        )
        self.model_name = str(
            self.worker_config.get("model_name", "yuvalkirstain/PickScore_v1"),
        )
        self._processor: Any = None
        self._model: Any = None

    def _load(self) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self._processor = CLIPProcessor.from_pretrained(self.processor_name)
        self._model = (
            CLIPModel.from_pretrained(self.model_name).eval().to(self.device, dtype=self.dtype)
        )

    def score_media(self, *, media: Any, prompt: str, request: Any) -> Mapping[str, float]:
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
                mid = arr.shape[1] // 2
                arr = arr[:, mid].transpose(0, 2, 3, 1)
            images = [Image.fromarray(a) for a in arr]
        elif isinstance(media, np.ndarray):
            images = [Image.fromarray(media)] if media.ndim == 3 else [Image.fromarray(a) for a in media]
        elif isinstance(media, Image.Image):
            images = [media]
        else:
            return {"pickscore": 0.0}

        score = self._score(prompt, images)
        return {"pickscore": float(score)}

    def _score(self, prompt: str, images: list[Any]) -> float:
        import torch

        with torch.no_grad():
            image_inputs = self._processor(
                images=images, padding=True, truncation=True, max_length=77, return_tensors="pt",
            )
            image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
            text_inputs = self._processor(
                text=[prompt] * len(images),
                padding=True, truncation=True, max_length=77, return_tensors="pt",
            )
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            image_embs = self._model.get_image_features(**image_inputs).pooler_output
            image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)
            text_embs = self._model.get_text_features(**text_inputs).pooler_output
            text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
            logit_scale = self._model.logit_scale.exp()
            scores = logit_scale * (text_embs @ image_embs.T)
            return float((scores.diag() / 26).mean().item())


def pickscore_reward_model(worker_config: Mapping[str, Any]) -> PickScoreRewardModel:
    """RewardModel factory for the reward registry / InProcessRewardRuntime."""

    return PickScoreRewardModel(worker_config)


__all__ = ["PickScoreRewardModel", "pickscore_reward_model"]
