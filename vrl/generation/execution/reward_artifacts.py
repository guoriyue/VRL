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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from vrl.generation.types import RewardArtifactSpec
from vrl.rewards.types import MaterializedArtifact
from vrl.utils.artifacts import sha256_file

# Files of one batch are written concurrently. A single mp4 write is bound by
# the frame-by-frame pipe into the encoder process plus the digest pass, both
# of which release the GIL, so independent samples overlap almost linearly
# (4 x 480p/33f: 1.58s -> 0.53s; 4 x 704p/93f: 5.74s -> 2.30s on 48 cores).
# The GPU worker sits idle for exactly this span between batches. Capped so
# four co-located workers do not fan out into hundreds of encoder threads.
_MAX_PARALLEL_SAMPLE_WRITES = 4


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
    jobs: list[tuple[RewardArtifactSpec, Any]] = []
    for spec in specs:
        expected_ndim = 5 if spec.media_type == "video" else 4
        if batch.ndim != expected_ndim:
            raise ValueError(
                f"reward artifact {spec.name!r} expects {spec.media_type} media "
                f"({expected_ndim} dims), got {tuple(batch.shape)}",
            )
        Path(spec.root).mkdir(parents=True, exist_ok=True)
        jobs.extend((spec, sample) for sample in batch)

    if len(jobs) < 2:
        written = [_write_sample_artifact(spec, sample) for spec, sample in jobs]
    else:
        with ThreadPoolExecutor(
            max_workers=min(len(jobs), _MAX_PARALLEL_SAMPLE_WRITES),
        ) as pool:
            written = list(pool.map(lambda job: _write_sample_artifact(*job), jobs))

    # ``jobs`` is spec-major, sample-minor, and ``pool.map`` preserves order.
    out: dict[str, list[MaterializedArtifact]] = {}
    for (spec, _), artifact in zip(jobs, written, strict=True):
        out.setdefault(spec.name, []).append(artifact)
    return out


def _write_sample_artifact(spec: RewardArtifactSpec, sample: Any) -> MaterializedArtifact:
    """Write one sample's file for ``spec`` and return its path, size and digest."""

    import torch

    suffix = "mp4" if spec.artifact_format == "mp4" else "pt"
    path = Path(spec.root) / f"{uuid.uuid4().hex}.{suffix}"
    if spec.artifact_format == "mp4":
        from vrl.utils.media import write_mp4

        write_mp4(sample, path, fps=float(spec.fps) if spec.fps is not None else 8.0)
    else:
        torch.save(sample.cpu(), path)
    return MaterializedArtifact(
        path=str(path.resolve()),
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
    )


__all__ = ["materialize_reward_artifacts"]
