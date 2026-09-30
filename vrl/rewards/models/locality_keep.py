"""Locality keep: pixel agreement with a declared source outside declared boxes.

Metadata ``locality_keep = {"source": path, "boxes": [[x0, y0, x1, y1], ...]}``
names the image every state of an edit sequence must match away from the
regions an edit may touch (absolute pixel boxes on the source's canvas). The
axes are plain measurements against that source over the outside pixels:

* ``locality_psnr``: peak signal-to-noise ratio in dB (capped at 60 for an
  identical image);
* ``locality_hf_ratio``: Laplacian-variance ratio candidate / source, 1.0 when
  the fine texture is unchanged, above 1 when texture was invented, below 1
  when it was smoothed away;
* ``locality_keep``: ``min(1, locality_psnr / 40) * min(hf, 1 / hf)``, the
  bounded default key. The texture factor is what makes it safe as a reward:
  on real editor outputs (already ~20 dB from their source) a mild blur
  *raises* PSNR, so PSNR alone would pay for smoothing; the symmetric
  high-frequency factor charges blur and grain alike (constructed-control
  probe, agentic ``locality_keep_probe``: blur and grain then rank below the
  untouched output in 100% of cases, 10 and 2 group-sigmas apart).

Known blind spots of the key on that probe: a small object painted over
(0.3% of the pixels) and a slight colour-tone drift move it by less than the
spread inside one candidate group. This is a model-free measurement, not a
judgement of the edit itself; it does not know which changes inside the
boxes were asked for.

On untiled manga outputs the legacy key can also reward added grain when
high-frequency energy starts below the source. ``locality_detail_keep`` is an
opt-in alternative: 1 / (1 + weighted penalty), with normalized pixel MSE,
relative source-Laplacian MAE and absolute log high-frequency energy ratio.
It requires explicit ``worker_config.detail_penalty_weights`` for those three
features. Weights must be frozen after damaged-output calibration; there is
no universal claim that this proxy detects every loss of detail.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from vrl.rewards.inference import RewardInferenceArtifact

PSNR_CAP = 60.0
PSNR_UNIT = 40.0


class LocalityKeepRewardModel:
    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        self._sources: dict[str, np.ndarray] = {}
        self._detail_weights = worker_config.get("detail_penalty_weights")
        if self._detail_weights is not None:
            keys = {"normalized_mse", "relative_laplacian_mae", "absolute_log_hf_ratio"}
            if not isinstance(self._detail_weights, Mapping) or set(self._detail_weights) != keys:
                raise ValueError(f"detail_penalty_weights must declare exactly {sorted(keys)}")
            self._detail_weights = {
                name: float(weight) for name, weight in self._detail_weights.items()
            }
            if any(
                not np.isfinite(weight) or weight < 0 for weight in self._detail_weights.values()
            ):
                raise ValueError("detail penalty weights must be finite and nonnegative")
            if not any(self._detail_weights.values()):
                raise ValueError("at least one detail penalty weight must be positive")

    def _source(self, path: str) -> np.ndarray:
        if path not in self._sources:
            from PIL import Image

            with Image.open(path) as image:
                self._sources[path] = np.asarray(image.convert("RGB"), dtype=np.float64)
        return self._sources[path]

    def __call__(self, artifact: RewardInferenceArtifact) -> dict[str, float]:
        import cv2
        import torch
        from PIL import Image

        from vrl.utils.media import to_pil_image

        spec = artifact.metadata.get("locality_keep")
        if not isinstance(spec, Mapping) or not spec.get("source"):
            raise ValueError("locality_keep needs metadata locality_keep with a source path")
        boxes = spec.get("boxes", [])
        if not isinstance(boxes, (list, tuple)) or any(len(box) != 4 for box in boxes):
            raise ValueError("locality_keep boxes must be [x0, y0, x1, y1] lists")
        source = self._source(str(spec["source"]))
        if artifact.path and not artifact.path.endswith(".pt"):
            with Image.open(artifact.path) as image:
                media = image.copy()
        else:
            media = artifact.as_media()
        if isinstance(media, torch.Tensor) and media.ndim == 4 and media.shape[1] == 1:
            media = media[:, 0]
        candidate = np.asarray(to_pil_image(media).convert("RGB"), dtype=np.float64)
        if candidate.shape != source.shape:
            raise ValueError(
                f"locality_keep candidate {candidate.shape[:2]} does not match the source "
                f"{source.shape[:2]}"
            )
        height, width = source.shape[:2]
        outside = np.ones((height, width), dtype=bool)
        for x0, y0, x1, y1 in boxes:
            outside[
                max(0, int(y0)) : min(height, int(y1)), max(0, int(x0)) : min(width, int(x1))
            ] = False
        if not outside.any():
            raise ValueError("locality_keep boxes cover the whole canvas")
        mse = float(np.mean(((candidate - source) ** 2)[outside]))
        psnr = PSNR_CAP if mse == 0 else min(PSNR_CAP, 10 * np.log10(255.0**2 / mse))
        gray = lambda a: cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)  # noqa: E731
        # Fill the boxes with the source so their edges do not leak into the outside statistic.
        filled = np.where(outside[..., None], candidate, source)
        lap_candidate = cv2.Laplacian(gray(filled), cv2.CV_64F)[outside]
        lap_source = cv2.Laplacian(gray(source), cv2.CV_64F)[outside]
        lap_c, lap_s = lap_candidate.var(), lap_source.var()
        hf_ratio = float(lap_c / lap_s) if lap_s > 0 else 1.0
        texture = min(hf_ratio, 1.0 / hf_ratio) if hf_ratio > 0 else 0.0
        scores = {
            "locality_keep": float(min(1.0, max(0.0, psnr / PSNR_UNIT)) * texture),
            "locality_psnr": float(psnr),
            "locality_hf_ratio": hf_ratio,
        }
        if self._detail_weights is not None:
            # Variance alone can reward adding noise to an already smoothed
            # output. Error against the actual source's high-frequency pattern
            # charges that noise; the energy gap separately charges smoothing.
            # Calibrate both terms on real damaged-output controls before use.
            features = {
                "normalized_mse": mse / 255.0**2,
                "relative_laplacian_mae": float(
                    np.abs(lap_candidate - lap_source).mean() / (np.abs(lap_source).mean() + 1e-8)
                ),
                "absolute_log_hf_ratio": float(abs(np.log((lap_c + 1e-8) / (lap_s + 1e-8)))),
            }
            penalty = sum(self._detail_weights[name] * value for name, value in features.items())
            scores.update({f"locality/{name}": value for name, value in features.items()})
            scores["locality_detail_keep"] = 1.0 / (1.0 + penalty)
        return scores


__all__ = ["LocalityKeepRewardModel"]
