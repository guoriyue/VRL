"""Aesthetic reward as a model-backed TorchRewardModel."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.rewards.models.base import TorchRewardModel
from vrl.utils.media import to_uint8


class AestheticRewardModel(TorchRewardModel):
    """Aesthetic predictor; returns ``{"aesthetic": mean_score}``."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        super().__init__(worker_config)
        self.model_name = str(
            self.worker_config.get("model_name", "openai/clip-vit-large-patch14"),
        )
        self._scorer: Any = None

    def _load(self) -> None:
        import torch
        import torch.nn as nn
        from transformers import CLIPModel, CLIPProcessor

        model_name = self.model_name

        class _MLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(768, 1024),
                    nn.Dropout(0.2),
                    nn.Linear(1024, 128),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.Dropout(0.1),
                    nn.Linear(64, 16),
                    nn.Linear(16, 1),
                )

            @torch.no_grad()
            def forward(self, embed: torch.Tensor) -> torch.Tensor:
                return self.layers(embed)

        class _AestheticScorer(nn.Module):
            def __init__(self, dtype: Any) -> None:
                super().__init__()
                self.clip = CLIPModel.from_pretrained(model_name)
                self.processor = CLIPProcessor.from_pretrained(model_name)
                self.mlp = _MLP()
                from importlib import resources

                weights_path = resources.files("vrl.rewards.assets").joinpath(
                    "sac+logos+ava1-l14-linearMSE.pth",
                )
                state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
                self.mlp.load_state_dict(state_dict)
                self.dtype = dtype
                self.eval()

            @torch.no_grad()
            def __call__(self, images: Any) -> torch.Tensor:
                device = next(self.parameters()).device
                inputs = self.processor(images=images, return_tensors="pt")
                inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
                # Transformers 5 returns its projected embedding inside the
                # feature output; the LAION head requires that 768-d projection,
                # not the unprojected vision hidden state.
                embed = self.clip.get_image_features(**inputs).pooler_output
                embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
                return self.mlp(embed).squeeze(1)

        self._scorer = _AestheticScorer(self.dtype).to(self.device)

    def score_media(self, *, media: Any, prompt: str, request: Any) -> Mapping[str, float]:
        import torch

        output = media
        if isinstance(output, torch.Tensor):
            images_raw = to_uint8(output)
            if images_raw.ndim == 4 and images_raw.shape[0] > 4:
                b = images_raw.shape[0]
                indices = [b // 4, b // 2, 3 * b // 4]
                images = [images_raw[i].cpu().numpy().transpose(1, 2, 0) for i in indices]
            elif images_raw.ndim == 4 and images_raw.shape[0] <= 4:
                t = images_raw.shape[1]
                indices = [t // 4, t // 2, 3 * t // 4]
                images = [images_raw[:, i, :, :].cpu().numpy().transpose(1, 2, 0) for i in indices]
            elif images_raw.ndim == 3:
                images = [images_raw.cpu().numpy().transpose(1, 2, 0)]
            else:
                t = images_raw.shape[0]
                indices = [t // 4, t // 2, 3 * t // 4]
                images = [images_raw[i].cpu().numpy().transpose(1, 2, 0) for i in indices]
        else:
            images = [output]

        scores = self._scorer(images)
        return {"aesthetic": float(scores.mean().item())}


def aesthetic_reward_model(worker_config: Mapping[str, Any]) -> AestheticRewardModel:
    return AestheticRewardModel(worker_config)


__all__ = ["AestheticRewardModel", "aesthetic_reward_model"]
