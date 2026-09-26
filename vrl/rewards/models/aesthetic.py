"""SigLIP Aesthetic Predictor V2.5 behind the shared torch reward lifecycle.

Uses the upstream model/processor and a packaged, immutable V2.5 head. An
image scores itself; a video scores up to three evenly spaced frames. Scores
are raw predictor outputs, not probabilities and not comparable to old LAION
CLIP-head scores. Install the ``reward`` extra for the upstream implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vrl.rewards.models.base import TorchRewardModel
from vrl.rewards.models.media import evenly_spaced_frames, pil_frames_from_media


class AestheticRewardModel(TorchRewardModel):
    """V2.5 image aesthetics; returns ``{"aesthetic": mean_score}``."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        super().__init__(worker_config)
        self.model_name = str(
            self.worker_config.get("model_name", "google/siglip-so400m-patch14-384"),
        )
        self.model_revision = (
            str(self.worker_config.get("model_revision", "") or "").strip() or None
        )

    def _load_module(self) -> Any:
        from importlib import resources

        from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip

        load_kwargs = {"revision": self.model_revision} if self.model_revision else {}
        asset = resources.files("vrl.rewards.assets").joinpath("aesthetic_predictor_v2_5.pth")
        with resources.as_file(asset) as path:
            model, self._processor = convert_v2_5_from_siglip(
                predictor_name_or_path=str(path),
                encoder_model_name=self.model_name,
                **load_kwargs,
            )
        return model.to(device=self.device, dtype=self.dtype).eval()

    def score_media(self, *, media: Any, prompt: str) -> Mapping[str, float]:
        import torch

        model = self._module_for_inference()
        images = [
            frame
            for frames in pil_frames_from_media(media)
            for frame in evenly_spaced_frames(frames, 3)
        ]
        pixels = self._processor(images=images, return_tensors="pt").pixel_values
        with torch.inference_mode():
            scores = model(pixels.to(device=self.device, dtype=self.dtype)).logits
        return {"aesthetic": float(scores.float().mean().item())}


__all__ = ["AestheticRewardModel"]
