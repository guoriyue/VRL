"""Layer decomposition scored against known complete RGBA layers.

Metadata ``layered_reference = {"layers": <directory>}`` names a directory of
``layer_*.png`` RGBA frames, back to front, that are the complete truth for
the prompt's source image (a rendered fixture whose hidden content is known).
The candidate is the policy's ``[layers, 4, H, W]`` RGBA stack. Every axis
comes from ``vrl.scripts.eval.layered_reference_score.score_layers``, which
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
* ``layered_reference``: the product of those, the aggregate default key.

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

    def __call__(self, artifact: RewardInferenceArtifact) -> dict[str, float]:
        from vrl.scripts.eval.layered_reference_score import score_layers

        reference = self._reference(reference_layers_directory(artifact.metadata))
        candidate = self._candidate(artifact)
        if candidate.shape[1:3] != reference.shape[1:3]:
            raise ValueError(
                f"candidate layers are {candidate.shape[1:3]} but the reference "
                f"layers are {reference.shape[1:3]}; render the truth at the training geometry"
            )
        result = score_layers(reference, candidate)
        proposal = result["reward_proposal"]["components"]
        return {
            "layered_reference": float(result["reward_proposal"]["value"]),
            "layered_hidden_completion": float(proposal["hidden_quality"]),
            "layered_object_match": float(proposal["all_object_quality"]),
            "layered_visible_retention": float(proposal["visible_quality"]),
            "layered_edit_holes": float(proposal["edit_quality"]),
            "layered_nonredundancy": float(proposal["nonredundancy"]),
            "layered_reconstruction": float(proposal["reconstruction_quality"]),
        }


__all__ = [
    "METADATA_KEY",
    "SCORE_KEYS",
    "LayeredReferenceRewardModel",
    "reference_layers_directory",
]
