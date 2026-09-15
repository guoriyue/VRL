"""Worker-side reward artifact materialization.

The rollout worker already holds the decoded media; writing the reward file
here (mp4 or ``.pt``) means the driver receives only path + digest per sample.
That removes two driver-process costs that compete with the launch-bound
trainer for the interpreter: the multi-MB-per-sample tensor transfer and
libx264 encoding (docs/sprints/SPRINT_miles_diffusion_parity_program.md, A).
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vrl.generation.types import RewardArtifactSpec
from vrl.rewards.types import MaterializedArtifact
from vrl.utils.artifacts import sha256_file


def materialize_reward_artifacts(
    media: Any,
    specs: Sequence[RewardArtifactSpec],
) -> dict[str, list[MaterializedArtifact]]:
    """Write ``media`` (batch-first decoded tensor) once per spec, one file per sample.

    Video media is ``[B,C,T,H,W]``, image media ``[B,C,H,W]``; uint8 or float
    in [0, 1]. Files are named by a fresh materialization id so a retried
    batch never overwrites a file the driver may still be scoring.
    """

    import torch

    if not specs:
        return {}
    if not isinstance(media, torch.Tensor):
        raise TypeError(
            "reward artifact materialization requires the decoded media tensor; "
            f"got {type(media).__name__}",
        )
    if media.ndim not in {4, 5}:
        raise ValueError(
            f"reward artifact media must be [B,C,H,W] or [B,C,T,H,W], got {tuple(media.shape)}",
        )
    batch = media.detach()
    out: dict[str, list[MaterializedArtifact]] = {}
    for spec in specs:
        expected_ndim = 5 if spec.media_type == "video" else 4
        if batch.ndim != expected_ndim:
            raise ValueError(
                f"reward artifact {spec.name!r} expects {spec.media_type} media "
                f"({expected_ndim} dims), got {tuple(batch.shape)}",
            )
        root = Path(spec.root)
        root.mkdir(parents=True, exist_ok=True)
        suffix = "mp4" if spec.artifact_format == "mp4" else "pt"
        files: list[MaterializedArtifact] = []
        for sample in batch:
            path = root / f"{uuid.uuid4().hex}.{suffix}"
            if spec.artifact_format == "mp4":
                from vrl.utils.media import write_mp4

                write_mp4(sample, path, fps=float(spec.fps) if spec.fps is not None else 8.0)
            else:
                # Reward models take unit-range floats (their to_uint8 multiplies
                # by 255); media crosses the wire as uint8, so restore k/255
                # here exactly as the driver did before it handed samples over.
                tensor = sample.cpu()
                if tensor.dtype == torch.uint8:
                    tensor = tensor.float() / 255.0
                torch.save(tensor, path)
            files.append(
                MaterializedArtifact(
                    path=str(path.resolve()),
                    size_bytes=path.stat().st_size,
                    sha256=sha256_file(path),
                ),
            )
        out[spec.name] = files
    return out


__all__ = ["materialize_reward_artifacts"]
