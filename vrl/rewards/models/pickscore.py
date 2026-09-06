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
                "processor_name",
                "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            ),
        )
        self.model_name = str(
            self.worker_config.get("model_name", "yuvalkirstain/PickScore_v1"),
        )
        self.processor_revision = (
            str(self.worker_config.get("processor_revision", "") or "").strip() or None
        )
        self.model_revision = (
            str(self.worker_config.get("model_revision", "") or "").strip() or None
        )
        self._processor: Any = None

    def _load_module(self) -> Any:
        from transformers import CLIPModel, CLIPProcessor

        processor_kwargs = {"revision": self.processor_revision} if self.processor_revision else {}
        model_kwargs = {"revision": self.model_revision} if self.model_revision else {}
        # CLIPProcessor owns CPU tokenization/image transforms only; the returned
        # CLIP module is the CUDA state the pool's build frame must capture.
        self._processor = CLIPProcessor.from_pretrained(
            self.processor_name,
            **processor_kwargs,
        )
        return (
            CLIPModel.from_pretrained(self.model_name, **model_kwargs)
            .eval()
            .to(self.device, dtype=self.dtype)
        )

    def score_media(self, *, media: Any, prompt: str) -> Mapping[str, float]:
        import numpy as np
        import torch
        from PIL import Image

        if isinstance(media, torch.Tensor):
            arr = to_uint8(media).cpu().numpy()
            if arr.ndim == 5:
                # Score the middle frame. The canonical video layout is [B,C,T,H,W];
                # [B,T,C,H,W] wins the ambiguous case, matching nsfw_safety.
                frame_axis = 1 if arr.shape[2] in (1, 3, 4) else 2
                arr = arr.take(arr.shape[frame_axis] // 2, axis=frame_axis)
            if arr.ndim == 3:
                arr = arr[None]
            arr = arr.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(a) for a in arr]
        elif isinstance(media, np.ndarray):
            images = (
                [Image.fromarray(media)]
                if media.ndim == 3
                else [Image.fromarray(a) for a in media]
            )
        elif isinstance(media, Image.Image):
            images = [media]
        else:
            return {"pickscore": 0.0}

        score = self._score(prompt, images)
        return {"pickscore": float(score)}

    def _score(self, prompt: str, images: list[Any]) -> float:
        import torch

        model = self._module_for_inference()
        with torch.no_grad():
            image_inputs = self._processor(
                images=images,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
            text_inputs = self._processor(
                text=[prompt] * len(images),
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            image_embs = model.get_image_features(**image_inputs).pooler_output
            image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)
            text_embs = model.get_text_features(**text_inputs).pooler_output
            text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp()
            scores = logit_scale * (text_embs @ image_embs.T)
            return float((scores.diag() / 26).mean().item())


__all__ = ["PickScoreRewardModel"]
