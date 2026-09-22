"""Continuous color witnesses shared by localized-edit training and diagnostics.

Regions and color vocabulary belong in experiment assets, not workflow code.
The worst protected region bounds locality; correct regions cannot average away
an incorrect one. These sparse witnesses are not full segmentation verifiers.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image


def membership(pixels: np.ndarray, box: list[float], *, hue: float, config: dict) -> float:
    height, width = pixels.shape[:2]
    x0, y0, x1, y1 = [
        round(v * n) for v, n in zip(box, [width, height, width, height], strict=True)
    ]
    patch = pixels[y0:y1, x0:x1]
    if not patch.size:
        raise ValueError(f"Empty witness patch: {box}")
    hsv = np.asarray(
        Image.fromarray((patch * 255).round().astype(np.uint8)).convert("HSV"), dtype=np.float32
    )
    hsv[..., 0] *= 360.0 / 255.0
    hsv[..., 1:] /= 255.0
    difference = np.abs(hsv[..., 0] - hue)
    difference = np.minimum(difference, 360 - difference)
    hue_score = np.exp(-0.5 * (difference / config["hue_sigma_degrees"]) ** 2)
    saturation = np.clip(
        (hsv[..., 1] - config["saturation_start"])
        / (config["saturation_full"] - config["saturation_start"]),
        0,
        1,
    )
    return float((hue_score * saturation).mean())


def color_locality(
    source: Image.Image, edited: Image.Image, rule: dict, config: dict
) -> dict[str, float]:
    source_pixels = (
        np.asarray(
            source.convert("RGB").resize(edited.size, Image.Resampling.LANCZOS), dtype=np.float32
        )
        / 255.0
    )
    edited_pixels = np.asarray(edited.convert("RGB"), dtype=np.float32) / 255.0
    hue = config["color_hue_degrees"][rule["color"]]
    completion = min(
        membership(edited_pixels, box, hue=hue, config=config) for box in rule["target"]
    )
    protected = {}
    for label, box in rule["protected"].items():
        before = membership(source_pixels, box, hue=hue, config=config)
        after = membership(edited_pixels, box, hue=hue, config=config)
        protected[f"preservation_{label}"] = 1.0 - max(0.0, after - before)
    preservation = min(protected.values())
    return {
        "completion": completion,
        "worst_preservation": preservation,
        "locality": min(completion, preservation),
        **protected,
    }


def combine_locality(
    evidence: dict[str, float], quality: float, *, quality_weight: float
) -> dict[str, float]:
    if not math.isfinite(quality) or not math.isfinite(quality_weight) or quality_weight < 0:
        raise ValueError(
            "Locality reward needs finite quality and a nonnegative calibrated weight"
        )
    bounded = math.tanh(quality)
    return {
        **evidence,
        "bounded_quality": bounded,
        "editreward_locality": evidence["locality"] + quality_weight * bounded,
    }
