"""DINOv2 perceptual target-similarity reward for robot-domain Video2World.

Drop-in successor to ``target_video_similarity`` (the retired 64x64 pixel-L1
reward, see ``SPRINT_future_reward`` S1). It keeps the exact same metadata
contract (``target_video`` / ``target_image`` from the manifest) and the same
sequence / final-frame / temporal weighting, but replaces the reward-hackable
per-frame feature: instead of flattened low-res RGB compared by L1, each frame is
embedded by a frozen DINOv2 ViT and compared by cosine similarity.

Why this is not pixel-L1 in disguise: a DINOv2 embedding is a learned semantic
feature, so the conditional-mean ("blur the prediction to the temporal average")
solution that maximizes pixel L1 no longer maximizes cosine similarity -- a blurry
or frozen frame lands off the real-image manifold and its embedding drifts away
from the target's. Measured on the SPRINT_future_reward adversarial battery:
blur / temporal-mean scored >0.15 cosine below the exact match. Discrimination
regressions are now caught by a human reviewing rollouts (the automated probe
was removed).

This is a *perceptual anchor* that still scores against a single recorded future, so
it penalizes plausible divergent rollouts -- the weaker "imitate one recording"
regime (SPRINT_future_reward S6). For instruction-grounded action-following, the
sprint designs an inverse-dynamics reward in S3 (not shipped).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from vrl.rewards.base import decode_artifact_frames
from vrl.utils.artifacts import default_data_root, resolve_artifact_path
from vrl.utils.media import align_frame_counts, read_image_as_frames, read_video_frames

# ImageNet statistics DINOv2 was trained with; frames are normalized with these
# before the ViT. DINOv2 patch size is 14, so the resize target must be a
# multiple of 14 (224 = 16 * 14).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_DINO_INPUT_SIZE = 224


class TargetDinoSimilarityModel:
    """Compare generated frames against target media via frozen DINOv2 embeddings."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        cfg = dict(worker_config)
        self.data_root = str(cfg.get("data_root") or default_data_root())
        self.num_frames = max(1, int(cfg.get("num_frames", 16)))
        # Temporal weight is high on purpose: per-frame cosine alone cannot separate
        # a frozen / temporal-mean clip from the real one (same scene -> ~0.76 cosine),
        # so the order-sensitive temporal term carries equal weight to appearance.
        # These defaults pass the SPRINT S4 discrimination probe (gap_ratio ~0.30,
        # temporal-mean ~0.25 cosine below exact); a lower temporal weight fails it.
        self.sequence_weight = float(cfg.get("sequence_weight", 0.4))
        self.final_weight = float(cfg.get("final_weight", 0.2))
        self.temporal_weight = float(cfg.get("temporal_weight", 0.4))
        self.allow_absolute_paths = bool(cfg.get("allow_absolute_paths", True))
        self.hub_repo = str(cfg.get("dino_hub_repo", "facebookresearch/dinov2"))
        self.hub_model = str(cfg.get("dino_hub_model", "dinov2_vits14"))
        requested = str(cfg.get("device", "")).strip()
        self.device = requested or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: torch.nn.Module | None = None

    def _encoder(self) -> torch.nn.Module:
        # Lazy so importing the module (e.g. for the registry) never pulls weights.
        if self._model is None:
            model = torch.hub.load(self.hub_repo, self.hub_model, verbose=False)
            model.eval().to(self.device)
            for param in model.parameters():
                param.requires_grad_(False)
            self._model = model
        return self._model

    def prepare_for_inference(self) -> None:
        """Materialize DINO weights inside the owning reward memory pool."""

        self._encoder()

    def move_to(self, device: str) -> None:
        """Move all materialized DINO state for shared-GPU phase handoff."""

        if self._model is not None:
            self._model = self._model.to(device)
        self.device = str(device)

    def __call__(self, *, artifact: Any, request: Any) -> Mapping[str, float]:
        del request
        metadata = dict(getattr(artifact, "metadata", {}) or {})
        target_video = str(metadata.get("target_video", "") or "").strip()
        target_image = str(metadata.get("target_image", "") or "").strip()
        if not target_video and not target_image:
            raise ValueError(
                f"artifact {artifact.artifact_id!r} has no target_video or target_image metadata",
            )

        generated = decode_artifact_frames(artifact, self.num_frames)
        target = (
            read_video_frames(self._resolve(target_video), self.num_frames)
            if target_video
            else read_image_as_frames(self._resolve(target_image))
        )
        if target.shape[0] == 1 and generated.shape[0] > 1:
            target = target.expand(generated.shape[0], -1, -1, -1)
        generated, target = align_frame_counts(generated, target)

        gen_embed = self._embed(generated)
        tgt_embed = self._embed(target)
        sequence = _mean_cosine(gen_embed, tgt_embed)
        final = _mean_cosine(gen_embed[-1:], tgt_embed[-1:])
        temporal = _temporal_cosine(gen_embed, tgt_embed)
        total_weight = self.sequence_weight + self.final_weight + self.temporal_weight
        if total_weight <= 0:
            raise ValueError("target_dino_similarity weights must sum to a positive value")
        overall = (
            self.sequence_weight * sequence
            + self.final_weight * final
            + self.temporal_weight * temporal
        ) / total_weight
        return {
            "target_dino_similarity": float(overall),
            "target_dino_sequence_similarity": float(sequence),
            "target_dino_final_frame_similarity": float(final),
            "target_dino_temporal_similarity": float(temporal),
        }

    @torch.no_grad()
    def _embed(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode ``[T,H,W,3]`` float frames into L2-normalized ``[T,D]`` DINOv2 features."""

        nchw = frames.permute(0, 3, 1, 2).contiguous().to(self.device)
        nchw = torch.nn.functional.interpolate(
            nchw,
            size=(_DINO_INPUT_SIZE, _DINO_INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        mean = torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        normalized = (nchw - mean) / std
        features = self._encoder()(normalized)
        return torch.nn.functional.normalize(features.float(), dim=-1)

    def _resolve(self, raw_path: str) -> Path:
        return resolve_artifact_path(
            raw_path,
            data_root=self.data_root,
            allow_absolute=self.allow_absolute_paths,
        )


def _mean_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-frame cosine similarity of two L2-normalized ``[T,D]`` stacks."""

    return float((a * b).sum(dim=-1).mean().item())


def _temporal_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine of consecutive-frame embedding deltas -- order-sensitive motion term.

    Per-frame cosine is order-blind (shuffle / reverse look identical frame-wise),
    so the temporal term compares the *direction of change* between adjacent
    embeddings; a shuffled or reversed clip has different deltas and scores lower.
    """

    if a.shape[0] < 2 or b.shape[0] < 2:
        return _mean_cosine(a, b)
    da = torch.nn.functional.normalize(a[1:] - a[:-1], dim=-1)
    db = torch.nn.functional.normalize(b[1:] - b[:-1], dim=-1)
    return float((da * db).sum(dim=-1).mean().item())


__all__ = ["TargetDinoSimilarityModel"]
