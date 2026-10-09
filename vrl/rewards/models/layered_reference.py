"""Layer decomposition scored against known complete RGBA layers.

Metadata ``layered_reference = {"layers": <directory>}`` names a directory of
``layer_*.png`` RGBA frames, back to front, that are the complete truth for
the prompt's source image (a rendered fixture whose hidden content is known).
The candidate is the policy's ``[layers, 4, H, W]`` RGBA stack. Every axis
comes from ``vrl.rewards.models.layered_score.score_layers``, which
matches truth objects to candidate layers injectively and measures them:

* ``layered_hidden_completion``: one minus the premultiplied RGBA error over
  the truth pixels hidden behind objects in front (amodal completion);
* ``layered_object_match``: product over truth objects of their match quality
  (alpha IoU times visible agreement) -- merged or missing objects zero it;
* ``layered_visible_retention``: one minus the error on the visible pixels;
* ``layered_edit_holes``: one minus the composite error after hiding or moving
  each object, truth versus candidate (holes and residues under an edit);
* ``layered_nonredundancy``, ``layered_reconstruction``: surplus alpha mass
  outside the matched layers, and the recomposed-image error;
* ``layered_reference``: the product of those, the aggregate default key;
* ``layered_composite_coverage``: fraction of the truth's opaque pixels the
  candidate composite covers opaquely (holes after recomposition);
* ``layered_inferable_completion``: one minus the premultiplied RGBA error on
  the hidden pixels an ``inferable_mask`` marks as deducible from what is
  visible (continuation or symmetry), scored against the truth;
* ``layered_unknowable_plausibility``: on the hidden pixels an
  ``unknowable_mask`` marks as not deducible, no truth pixels are consulted:
  the matched layer must continue the object there (alpha) with colors near
  its own visible colors (continuity).

Optional metadata ``layered_reference = {"layers": dir, "inferable_mask": png,
"unknowable_mask": png}``; without the masks the last two axes are 1.0.

Verifiable against known layers, so gate 2 of the reward qualification uses
an oracle set; nothing here is qualified for a training key until the gates
in ``.claude/skills/reward/qualification.md`` have been run on the policy's
own outputs. One rendered scene is not a general completeness judge.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from vrl.rewards.inference import RewardInferenceArtifact

METADATA_KEY = "layered_reference"
SCORE_KEYS = (
    "layered_reference",
    "layered_hidden_completion",
    "layered_object_match",
    "layered_visible_retention",
    "layered_edit_holes",
    "layered_nonredundancy",
    "layered_reconstruction",
    "layered_composite_coverage",
    "layered_inferable_completion",
    "layered_unknowable_plausibility",
)


def reference_layers_directory(metadata: Mapping[str, Any]) -> str:
    """The truth-layer directory a sample's metadata names, validated."""

    spec = metadata.get(METADATA_KEY)
    if not isinstance(spec, Mapping) or not isinstance(spec.get("layers"), str):
        raise ValueError(
            f"{METADATA_KEY} needs metadata {METADATA_KEY} = {{'layers': <directory>}}"
        )
    layers = spec["layers"].strip()
    if not layers:
        raise ValueError(f"{METADATA_KEY}.layers must be a non-empty directory path")
    return layers


class LayeredReferenceRewardModel:
    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        self._data_root = worker_config.get("data_root")
        self._references: dict[str, np.ndarray] = {}

    def _reference(self, directory: str) -> np.ndarray:
        if directory not in self._references:
            from PIL import Image

            root = Path(directory)
            if not root.is_absolute() and self._data_root:
                root = Path(self._data_root) / root
            files = sorted(root.glob("layer_*.png"))
            if not files:
                raise ValueError(f"{METADATA_KEY}.layers {root} holds no layer_*.png frames")
            frames = []
            for file in files:
                with Image.open(file) as image:
                    frames.append(np.asarray(image.convert("RGBA"), dtype=np.float64) / 255.0)
            stack = np.stack(frames)
            if stack.ndim != 4 or stack.shape[-1] != 4:
                raise ValueError(f"{METADATA_KEY}.layers {root} is not an RGBA frame stack")
            self._references[directory] = stack
        return self._references[directory]

    @staticmethod
    def _candidate(artifact: RewardInferenceArtifact) -> np.ndarray:
        import torch

        media = artifact.as_media()
        if not isinstance(media, torch.Tensor) or media.ndim != 4 or media.shape[1] != 4:
            raise TypeError(
                f"{METADATA_KEY} scores a [layers, 4, H, W] RGBA stack, got "
                f"{type(media).__name__} {tuple(getattr(media, 'shape', ()))}"
            )
        stack = media.detach().float().cpu()
        if stack.numel() and float(stack.max()) > 1.0:
            stack = stack / 255.0
        return stack.clamp(0.0, 1.0).permute(0, 2, 3, 1).numpy().astype(np.float64)

    def _mask(self, metadata: Mapping[str, Any], key: str) -> np.ndarray | None:
        path = metadata.get(METADATA_KEY, {}).get(key)
        if path is None:
            return None
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"{METADATA_KEY}.{key} must be a path string")
        cache_key = f"{key}:{path}"
        if cache_key not in self._references:
            from PIL import Image

            file = Path(path)
            if not file.is_absolute() and self._data_root:
                file = Path(self._data_root) / file
            with Image.open(file) as image:
                self._references[cache_key] = np.asarray(image.convert("L")) > 127
        return self._references[cache_key]

    def __call__(self, artifact: RewardInferenceArtifact) -> dict[str, float]:
        from vrl.rewards.models.layered_score import composite, score_layers

        reference = self._reference(reference_layers_directory(artifact.metadata))
        candidate = self._candidate(artifact)
        if candidate.shape[1:3] != reference.shape[1:3]:
            raise ValueError(
                f"candidate layers are {candidate.shape[1:3]} but the reference "
                f"layers are {reference.shape[1:3]}; render the truth at the training geometry"
            )
        result = score_layers(reference, candidate)
        proposal = result["reward_proposal"]["components"]
        inferable = self._mask(artifact.metadata, "inferable_mask")
        unknowable = self._mask(artifact.metadata, "unknowable_mask")
        axes = region_axes(reference, candidate, result["assignment"], inferable, unknowable)
        truth_opaque = composite(reference)[..., 3] >= 0.99
        covered = composite(candidate)[..., 3] >= 0.99
        coverage = (
            float((covered & truth_opaque).sum() / truth_opaque.sum())
            if truth_opaque.any()
            else 1.0
        )
        return {
            "layered_reference": float(result["reward_proposal"]["value"]),
            "layered_hidden_completion": float(proposal["hidden_quality"]),
            "layered_object_match": float(proposal["all_object_quality"]),
            "layered_visible_retention": float(proposal["visible_quality"]),
            "layered_edit_holes": float(proposal["edit_quality"]),
            "layered_nonredundancy": float(proposal["nonredundancy"]),
            "layered_reconstruction": float(proposal["reconstruction_quality"]),
            "layered_composite_coverage": coverage,
            **axes,
        }


def region_axes(reference, candidate, assignment, inferable, unknowable) -> dict[str, float]:
    """Inferable-region completion (against truth) and unknowable-region plausibility (no truth).

    Both walk the truth objects (index 0 is the background and is skipped) with
    their matched candidate layer. Inferable pixels are the object's hidden
    pixels inside ``inferable``; the axis is one minus the pixel-weighted
    premultiplied RGBA error there. Unknowable pixels are the object's hidden
    pixels inside ``unknowable``; the axis is the matched layer's mean alpha
    there times a color-continuity factor against that layer's own visible
    region, so memorized truth pixels earn nothing extra.
    """
    from vrl.rewards.models.layered_score import premultiplied

    inferable_error, inferable_mass = 0.0, 0.0
    plausibility, unknowable_mass = 0.0, 0.0
    blank = np.zeros_like(reference[0])
    for i in range(1, len(reference)):
        predicted_index = assignment[i]
        predicted = blank if predicted_index is None else candidate[predicted_index]
        transmission = np.ones(reference.shape[1:3])
        for above in reference[i + 1 :]:
            transmission *= 1 - above[..., 3]
        hidden = reference[i, ..., 3] * (1 - transmission)
        visible = reference[i, ..., 3] * transmission
        residual = np.abs(premultiplied(reference[i]) - premultiplied(predicted)).mean(axis=-1)
        if inferable is not None:
            weight = hidden * inferable
            inferable_error += float((residual * weight).sum())
            inferable_mass += float(weight.sum())
        if unknowable is not None:
            weight = hidden * unknowable
            mass = float(weight.sum())
            if mass > 0:
                alpha = float((predicted[..., 3] * weight).sum() / mass)
                visible_px = (visible > 0.5) & (predicted[..., 3] > 0.5)
                region_px = (weight > 0.5) & (predicted[..., 3] > 0.5)
                if visible_px.any() and region_px.any():
                    gap = float(
                        np.abs(
                            predicted[region_px][:, :3].mean(axis=0)
                            - predicted[visible_px][:, :3].mean(axis=0)
                        ).mean()
                    )
                    continuity = max(0.0, 1.0 - 2.0 * gap)
                else:
                    continuity = 0.0 if region_px.any() else 1.0
                plausibility += alpha * continuity * mass
                unknowable_mass += mass
    return {
        "layered_inferable_completion": 1.0 - inferable_error / inferable_mass
        if inferable_mass
        else 1.0,
        "layered_unknowable_plausibility": plausibility / unknowable_mass
        if unknowable_mass
        else 1.0,
    }


__all__ = [
    "METADATA_KEY",
    "SCORE_KEYS",
    "LayeredReferenceRewardModel",
    "reference_layers_directory",
    "region_axes",
]
