"""RAFT optical-flow motion-dynamics reward (VBench Dynamic Degree).

A pure *quality guard*, not a target-matching reward: it measures how much real
motion the generated clip contains, with no reference to the target. The score is
the VBench "Dynamic Degree" statistic -- the mean of the top-5% optical-flow
magnitudes across consecutive frames (RAFT-small), normalized by the frame
diagonal and scaled to ``[0,1]``.

Its single job is to put a hard floor under static / blurred / temporal-mean
collapse: a frozen first frame or a time-averaged blur has flow ~= 0 and so scores
~= 0, while a clip with genuine motion scores high. EVA reports that long GRPO
runs relapse into static collapse even with a good main reward, so this term stays
in the blend as a cheap anti-collapse guard.

It is deliberately *not* a temporal-order judge: flow magnitude is direction- and
order-agnostic, so a reversed or shuffled clip can score as high as (or higher
than) the real one. That is fine -- order is the job of the IDM / DINOv2 temporal
terms; this term only kills the no-motion hacks.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from vrl.rewards.base import decode_artifact_frames


class MotionDynamicsModel:
    """Score generated-video motion magnitude with RAFT-small optical flow.

    Scores from decoded frames rather than ``artifact.as_media()``, so it cannot
    use ``TorchRewardModel`` and hand-rolls the lazy-module parking contract
    documented in ``vrl.rewards.models.base``.
    """

    def __init__(self, worker_config: Mapping[str, Any]) -> None:
        cfg = dict(worker_config)
        self.num_frames = max(2, int(cfg.get("num_frames", 16)))
        # RAFT needs H,W divisible by 8; 256 keeps small motions resolvable cheaply.
        self.flow_size = int(cfg.get("flow_size", 256))
        if self.flow_size % 8 != 0:
            raise ValueError(
                f"motion_dynamics flow_size must be divisible by 8, got {self.flow_size}"
            )
        self.top_fraction = float(cfg.get("top_fraction", 0.05))
        # Maps normalized top-5% flow (fraction of frame diagonal) into a [0,1]
        # reward. Domain-tunable: DROID exterior-cam motion is small (~0.007 of the
        # diagonal/step), so scale=50 lands real motion ~0.37 while static collapses
        # stay ~0.03 -- enough separation for the anti-collapse guard to bite in the
        # blend. Discrimination (static -> floor) is scale-invariant.
        self.magnitude_scale = float(cfg.get("magnitude_scale", 50.0))
        requested = str(cfg.get("device", "")).strip()
        self.device = requested or ("cuda" if torch.cuda.is_available() else "cpu")
        self._module: torch.nn.Module | None = None

    def _load_module(self) -> torch.nn.Module:
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

        model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False)
        model.eval().to(self.device)
        for param in model.parameters():
            param.requires_grad_(False)
        return model

    def _module_for_inference(self) -> torch.nn.Module:
        if self._module is None:
            self._module = self._load_module()
        return self._module

    def prepare_for_inference(self) -> None:
        """Materialize lazy model state inside the runtime's owning memory pool."""

        self._module_for_inference()

    def __call__(self, *, artifact: Any, request: Any) -> Mapping[str, float]:
        del request
        frames = decode_artifact_frames(artifact, self.num_frames)
        if frames.shape[0] < 2:
            return {"motion_dynamics": 0.0}
        raw = self._dynamic_degree(frames)
        return {"motion_dynamics": float(min(1.0, max(0.0, raw * self.magnitude_scale)))}

    @torch.no_grad()
    def _dynamic_degree(self, frames: torch.Tensor) -> float:
        """Mean of the top-``top_fraction`` per-pixel flow magnitudes, in diagonal units."""

        nchw = frames.permute(0, 3, 1, 2).contiguous().to(self.device)
        nchw = torch.nn.functional.interpolate(
            nchw,
            size=(self.flow_size, self.flow_size),
            mode="bilinear",
            align_corners=False,
        )
        # RAFT expects images mapped to [-1, 1].
        images = nchw * 2.0 - 1.0
        flows = self._module_for_inference()(images[:-1], images[1:])[-1]  # [T-1, 2, H, W]
        magnitude = torch.linalg.vector_norm(flows, dim=1).flatten()  # pixels
        diagonal = float((self.flow_size**2 + self.flow_size**2) ** 0.5)
        magnitude = magnitude / diagonal
        k = max(1, int(magnitude.numel() * self.top_fraction))
        top = torch.topk(magnitude, k).values
        return float(top.mean().item())


__all__ = ["MotionDynamicsModel"]
