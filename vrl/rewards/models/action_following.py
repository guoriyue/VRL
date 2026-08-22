"""Vision-only inverse-dynamics reward: does the generated future obey the commanded actions?

SPRINT_idm_action_following_reward, phases 2-3. Pixel and perceptual rewards
score against ONE recorded future, so every pixel exploit (static first frame,
temporal blur, shuffle, reverse) lands within a couple percent of a perfect
clip. This reward scores in the low-dimensional action space instead: a small
IDM regresses the 7-DOF DROID action (cartesian 6 + gripper 1) from each
adjacent frame pair, and the score is agreement with the episode's COMMANDED
actions read from ``artifact.metadata["target_actions"]`` — never from pixels.
Static clips predict ~zero motion against non-zero commands, blur destroys the
end-effector's coherent delta-pose, reversal flips the sign, and a real clip of
the WRONG instruction matches the wrong command sequence (the decisive test).

Architecture follows EVA (arXiv:2603.17808): conv backbone + per-channel
spatial-softmax keypoints + MLP, well under 1M parameters. Spatial-softmax
extracts 2D feature positions, which is what recovering an end-effector delta
needs; global pooling would collapse exactly the spatial signal the action
lives in.

The checkpoint carries its own action normalization (per-dim mean/std of the
training labels) and input size; the reward stays a pure consumer. Trained by
``vrl/scripts/rewards/train_droid_idm.py``; the discrimination gate
(``vrl/scripts/rewards/idm_discrimination_probe.py``) must pass per the sprint
before this signal drives GRPO.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSoftmax(nn.Module):
    """Per-channel softmax over H*W returning expected (x, y) coordinates."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = features.shape
        weights = F.softmax(features.reshape(batch, channels, height * width), dim=-1)
        ys = torch.linspace(-1.0, 1.0, height, device=features.device)
        xs = torch.linspace(-1.0, 1.0, width, device=features.device)
        grid_y = ys.view(1, 1, height, 1).expand(1, 1, height, width).reshape(1, 1, -1)
        grid_x = xs.view(1, 1, 1, width).expand(1, 1, height, width).reshape(1, 1, -1)
        expected_y = (weights * grid_y).sum(dim=-1)
        expected_x = (weights * grid_x).sum(dim=-1)
        return torch.cat([expected_x, expected_y], dim=-1)  # [B, 2C]


class FramePairIDM(nn.Module):
    """(o_i, o_{i+1}) -> action_i. Frames enter channel-concatenated (6ch)."""

    def __init__(self, *, action_dim: int = 7, width: int = 64) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, width, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.keypoints = SpatialSoftmax()
        self.head = nn.Sequential(
            nn.Linear(2 * width, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, action_dim),
        )

    def forward(self, frame_pairs: torch.Tensor) -> torch.Tensor:
        """frame_pairs: [B, 6, H, W] in [0,1] -> normalized actions [B, action_dim]."""

        return self.head(self.keypoints(self.encoder(frame_pairs)))


def load_idm_checkpoint(path: str, *, device: str = "cpu") -> dict[str, Any]:
    """Load a train_droid_idm.py checkpoint: model + its normalization contract."""

    payload = torch.load(path, map_location=device, weights_only=True)
    model = FramePairIDM(
        action_dim=int(payload["action_dim"]),
        width=int(payload["width"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return {
        "model": model,
        "action_mean": payload["action_mean"].to(device),
        "action_std": payload["action_std"].to(device),
        "image_size": int(payload["image_size"]),
    }


def frame_pairs_from_clip(frames: torch.Tensor, *, image_size: int) -> torch.Tensor:
    """[T,H,W,3] float clip -> [T-1, 6, S, S] channel-concatenated pairs."""

    if frames.ndim != 4 or frames.shape[0] < 2:
        raise ValueError(f"need [T>=2,H,W,3] frames, got {tuple(frames.shape)}")
    chw = frames.permute(0, 3, 1, 2)
    resized = F.interpolate(
        chw, size=(image_size, image_size), mode="bilinear", align_corners=False
    )
    return torch.cat([resized[:-1], resized[1:]], dim=1)


@torch.no_grad()
def score_action_following(
    frames: torch.Tensor,
    target_actions: torch.Tensor,
    idm: Mapping[str, Any],
) -> dict[str, float]:
    """Per-clip agreement between IDM-recovered and commanded actions.

    ``match = exp(-mse_z)`` per pair in the checkpoint's normalized action
    space, averaged over pairs — 1.0 is perfect recovery, and a clip whose
    recovered motion is unrelated to the commands decays toward 0. Splits:
    ``ee_pose_match`` (first 6 dims) and ``gripper_match`` (last dim) localize
    WHAT disagrees; ``action_match``/``overall`` are the full-vector score.
    """

    model: FramePairIDM = idm["model"]
    mean: torch.Tensor = idm["action_mean"]
    std: torch.Tensor = idm["action_std"]
    device = next(model.parameters()).device

    pairs = frame_pairs_from_clip(frames, image_size=idm["image_size"]).to(device)
    count = min(pairs.shape[0], target_actions.shape[0])
    if count < 1:
        raise ValueError("clip and target_actions share no scorable steps")
    predicted_z = model(pairs[:count])
    target_z = (target_actions[:count].to(device).float() - mean) / std

    def match(dims: slice) -> float:
        err = (predicted_z[:, dims] - target_z[:, dims]).pow(2).mean(dim=-1)
        return float(torch.exp(-err).mean().item())

    overall = match(slice(None))
    return {
        "action_match": overall,
        "ee_pose_match": match(slice(0, model.action_dim - 1)),
        "gripper_match": match(slice(model.action_dim - 1, model.action_dim)),
        "overall": overall,
    }


class ActionFollowingIDMModel:
    """Disk-artifact reward worker: decode clip, read commands, score agreement."""

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        cfg = dict(worker_config)
        checkpoint = str(cfg.get("checkpoint", "") or "")
        if not checkpoint:
            raise ValueError(
                "action_following requires 'checkpoint' (train with "
                "vrl/scripts/rewards/train_droid_idm.py)",
            )
        device = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self._idm = load_idm_checkpoint(checkpoint, device=device)

    def __call__(self, artifact: Any, request: Any = None) -> dict[str, float]:
        from vrl.rewards.models.media import decode_artifact_frames

        raw = artifact.metadata.get("target_actions")
        if raw is None:
            raise ValueError(
                f"artifact {artifact.artifact_id} carries no metadata['target_actions'] — "
                "the manifest row must supply the commanded action sequence",
            )
        frames = decode_artifact_frames(artifact)
        return score_action_following(frames, torch.as_tensor(raw, dtype=torch.float32), self._idm)


__all__ = [
    "ActionFollowingIDMModel",
    "FramePairIDM",
    "SpatialSoftmax",
    "frame_pairs_from_clip",
    "load_idm_checkpoint",
    "score_action_following",
]
