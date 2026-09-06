"""Aesthetic reward as a model-backed TorchRewardModel.

The scorer is the LAION aesthetic predictor: a small MLP head trained on
SAC/LAION-Logos/AVA ratings over CLIP ViT-L/14 image embeddings. The ``_MLP``
class below deliberately re-declares that head's layer stack — the packaged
``sac+logos+ava1-l14-linearMSE.pth`` state dict (shipped in
``vrl.rewards.assets`` so scoring needs no external download) only loads into
this exact architecture. Returns ``{"aesthetic": score}``, the mean over up to
three evenly spaced frames.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.rewards.models.base import TorchRewardModel
from vrl.rewards.models.media import evenly_spaced_frames, pil_frames_from_media


class AestheticRewardModel(TorchRewardModel):
    """Aesthetic predictor; returns ``{"aesthetic": mean_score}``."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        super().__init__(worker_config)
        self.model_name = str(
            self.worker_config.get("model_name", "openai/clip-vit-large-patch14"),
        )
        self.model_revision = (
            str(self.worker_config.get("model_revision", "") or "").strip() or None
        )

    def _load_module(self) -> Any:
        import torch
        import torch.nn as nn
        from transformers import CLIPModel, CLIPProcessor

        model_name = self.model_name
        load_kwargs = {"revision": self.model_revision} if self.model_revision else {}

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
                self.clip = CLIPModel.from_pretrained(model_name, **load_kwargs)
                self.processor = CLIPProcessor.from_pretrained(model_name, **load_kwargs)
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

        return _AestheticScorer(self.dtype).to(self.device)

    def score_media(self, *, media: Any, prompt: str) -> Mapping[str, float]:
        # A video contributes three evenly spaced frames; an image, itself.
        images = [
            frame
            for frames in pil_frames_from_media(media)
            for frame in evenly_spaced_frames(frames, 3)
        ]
        scores = self._module_for_inference()(images)
        return {"aesthetic": float(scores.mean().item())}


__all__ = ["AestheticRewardModel"]
