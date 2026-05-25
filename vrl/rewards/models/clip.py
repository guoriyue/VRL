"""CLIP text-image similarity reward as a model-backed TorchRewardModel."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.rewards.models.base import TorchRewardModel


class CLIPScoreRewardModel(TorchRewardModel):
    """CLIP cosine similarity / 30; returns ``{"clipscore": mean_score}``."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        super().__init__(worker_config)
        self.model_name = str(
            self.worker_config.get("model_name", "openai/clip-vit-large-patch14"),
        )
        self._scorer: Any = None

    def _load(self) -> None:
        import torch
        import torch.nn as nn
        import torchvision.transforms as T
        from transformers import AutoImageProcessor, CLIPModel, CLIPProcessor

        def _get_size(size: Any) -> tuple[int, int] | int:
            if isinstance(size, int):
                return (size, size)
            if "height" in size and "width" in size:
                return (size["height"], size["width"])
            if "shortest_edge" in size:
                return size["shortest_edge"]
            raise ValueError(f"Invalid size: {size}")

        def _get_transform(processor: AutoImageProcessor) -> T.Compose:
            cfg = processor.to_dict()
            resize = (
                T.Resize(_get_size(cfg.get("size"))) if cfg.get("do_resize") else nn.Identity()
            )
            crop = (
                T.CenterCrop(_get_size(cfg.get("crop_size")))
                if cfg.get("do_center_crop")
                else nn.Identity()
            )
            normalise = (
                T.Normalize(mean=processor.image_mean, std=processor.image_std)
                if cfg.get("do_normalize")
                else nn.Identity()
            )
            return T.Compose([resize, crop, normalise])

        model_name = self.model_name
        device = self.device

        class _ClipScorer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.device = device
                self.model = CLIPModel.from_pretrained(model_name).to(device)
                self.processor = CLIPProcessor.from_pretrained(model_name)
                self.tform = _get_transform(self.processor.image_processor)
                self.eval()

            @torch.no_grad()
            def __call__(self, pixels: torch.Tensor, prompts: list[str]) -> torch.Tensor:
                texts = self.processor(
                    text=prompts, padding="max_length", truncation=True, return_tensors="pt",
                ).to(self.device)
                pixels = self.tform(pixels.to(dtype=pixels.dtype)).to(self.device)
                outputs = self.model(pixel_values=pixels, **texts)
                return outputs.logits_per_image.diagonal() / 30

        self._scorer = _ClipScorer()

    def score_media(self, *, media: Any, prompt: str, request: Any) -> Mapping[str, float]:
        import numpy as np
        import torch

        output = media
        if isinstance(output, torch.Tensor):
            pixels = output
            if pixels.ndim == 5:
                pixels = pixels[:, pixels.shape[1] // 2]  # middle frame
            if pixels.max() <= 1.0:
                pass  # already [0, 1]
            else:
                pixels = pixels.float() / 255.0
        elif isinstance(output, np.ndarray):
            pixels = torch.from_numpy(output.transpose(0, 3, 1, 2)).float() / 255.0
        else:
            return {"clipscore": 0.0}

        if pixels.ndim == 3:
            pixels = pixels.unsqueeze(0)

        scores = self._scorer(pixels, [prompt] * pixels.shape[0])
        return {"clipscore": float(scores.mean().item())}


def clip_score_reward_model(worker_config: Mapping[str, Any]) -> CLIPScoreRewardModel:
    return CLIPScoreRewardModel(worker_config)


__all__ = ["CLIPScoreRewardModel", "clip_score_reward_model"]
