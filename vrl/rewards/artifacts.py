"""Reward artifact stores: media ownership from build to terminal state.

Owns the artifact vocabulary (``MediaType``, ``ArtifactFormat``) and the
store seam ``RewardFunction`` scores through: ``RewardArtifactStore`` is the
protocol, ``InMemoryRewardArtifactStore`` the default transport, and
``DiskRewardArtifactStore`` the file-backed one. Disk materialization exists
because the reward service's only artifact transport is shared filesystem
paths: the wire format rejects in-memory media, so disk rewards must write
stable files (with the size/sha256 integrity fields the server re-verifies)
before scoring. The disk store tracks which paths it owns so the
terminal-state seam in base.py can either delete a call's materializations
(``release``) or transfer them out of store ownership (``retain``) when they
are explicitly kept or a remote request's fate is unknown — a file a live
remote scorer might still read is never deleted. torch loads lazily inside
the disk writer so this module stays importable in torch-free processes. No
generation-side dual: generation results travel in-process through the Ray
layer.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, get_args, runtime_checkable

from vrl.rewards.inference import RewardInferenceArtifact
from vrl.rewards.types import RewardSample
from vrl.utils.artifacts import sha256_file

if TYPE_CHECKING:
    import torch

# Valid artifact media kinds. The Literal is the single source of truth; the
# store constructor derives its validation set from it.
MediaType = Literal["image", "video"]

# On-disk artifact container: mp4 = real video container decord can read,
# tensor = torch.save .pt. media_type (image/video) is a separate axis.
ArtifactFormat = Literal["tensor", "mp4"]


@runtime_checkable
class RewardArtifactStore(Protocol):
    """Owner of one scoring call's media artifacts, from build to terminal state.

    ``materialize`` turns samples into scoreable artifacts. Exactly one of
    ``release`` (delete the call's materializations) or ``retain`` (transfer
    them to the debug/output owner) runs after scoring reaches a terminal
    state. The in-memory store no-ops the terminal half; the disk store
    deletes or keeps real files.
    """

    def materialize(
        self,
        samples: list[RewardSample],
    ) -> list[RewardInferenceArtifact]: ...

    def release(self, artifacts: list[RewardInferenceArtifact]) -> None: ...

    def retain(self, artifacts: list[RewardInferenceArtifact]) -> None: ...


class InMemoryRewardArtifactStore:
    """Default store: media rides the request in-memory, nothing to clean up."""

    def materialize(
        self,
        samples: list[RewardSample],
    ) -> list[RewardInferenceArtifact]:
        artifacts: list[RewardInferenceArtifact] = []
        for sample in samples:
            artifacts.append(
                RewardInferenceArtifact(
                    artifact_id=f"{sample.sample_id}:in-memory",
                    path="",
                    media=sample.output,
                    prompt=str(sample.prompt),
                    metadata=dict(sample.metadata or {}),
                ),
            )
        return artifacts

    def release(self, artifacts: list[RewardInferenceArtifact]) -> None:
        return None

    def retain(self, artifacts: list[RewardInferenceArtifact]) -> None:
        return None


class DiskRewardArtifactStore:
    """Driver-side writer for stable reward media artifacts."""

    def __init__(
        self,
        root: str | Path,
        *,
        media_type: MediaType = "video",
        artifact_format: ArtifactFormat = "tensor",
    ) -> None:
        if media_type not in get_args(MediaType):
            raise ValueError(
                f"media_type must be one of {', '.join(get_args(MediaType))}",
            )
        if artifact_format not in get_args(ArtifactFormat):
            raise ValueError(
                f"artifact_format must be one of {', '.join(get_args(ArtifactFormat))}",
            )
        if artifact_format == "mp4" and media_type != "video":
            raise ValueError("artifact_format=mp4 requires media_type=video")
        self.root = Path(root)
        self.media_type = media_type
        self.artifact_format = artifact_format
        self._owned_paths: set[Path] = set()
        self.root.mkdir(parents=True, exist_ok=True)

    def materialize(self, samples: list[RewardSample]) -> list[RewardInferenceArtifact]:
        artifacts: list[RewardInferenceArtifact] = []
        try:
            for sample in samples:
                artifacts.append(self._write_one(sample))
        except BaseException:
            self.release(artifacts)
            raise
        return artifacts

    def release(self, artifacts: list[RewardInferenceArtifact]) -> None:
        """Delete materializations owned by this store; safe to retry."""

        errors: list[OSError] = []
        for artifact in artifacts:
            if not artifact.path:
                continue
            path = Path(artifact.path)
            if path not in self._owned_paths:
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError as error:
                errors.append(error)
            else:
                self._owned_paths.discard(path)
        if errors:
            raise OSError(
                f"failed to release {len(errors)} reward artifacts",
            ) from errors[0]

    def retain(self, artifacts: list[RewardInferenceArtifact]) -> None:
        """Transfer retained debug artifacts out of the store's ownership."""

        for artifact in artifacts:
            if artifact.path:
                self._owned_paths.discard(Path(artifact.path))

    def _write_one(self, sample: RewardSample) -> RewardInferenceArtifact:
        import torch

        output = sample.output
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                "video reward artifact materialization requires tensor sample output",
            )
        tensor = output.detach().cpu()
        _validate_media_shape(tensor, self.media_type)

        metadata = dict(sample.metadata or {})
        materialization_id = uuid.uuid4().hex
        artifact_id = f"{sample.sample_id}:{materialization_id}"
        suffix = "mp4" if self.artifact_format == "mp4" else "pt"
        path = (self.root / f"{materialization_id}.{suffix}").resolve()
        self._owned_paths.add(path)
        try:
            if self.artifact_format == "mp4":
                from vrl.utils.media import write_mp4

                # fps is an mp4 encoding parameter only; reading it up front
                # would let junk fps metadata break tensor materialization.
                write_mp4(tensor, path, fps=_fps(metadata))
            else:
                torch.save(tensor, path)
            size_bytes = path.stat().st_size
            # Per-request audit trails are the opt-in debug_dir JSONLs owned by
            # InferenceRewardFunction._write_debug; the store writes media only.
            artifact = RewardInferenceArtifact(
                artifact_id=artifact_id,
                path=str(path),
                prompt=str(sample.prompt),
                size_bytes=size_bytes,
                sha256=sha256_file(path),
                metadata=_artifact_provenance(metadata),
            )
        except BaseException:
            path.unlink(missing_ok=True)
            self._owned_paths.discard(path)
            raise
        return artifact


def _validate_media_shape(tensor: torch.Tensor, media_type: str) -> None:
    if tensor.numel() == 0:
        raise ValueError("video reward artifact tensor must be non-empty")
    if media_type == "image" and tensor.ndim not in {3, 4}:
        raise ValueError(
            "image reward artifact expects [C,H,W] or [B,C,H,W] tensor, "
            f"got shape={tuple(tensor.shape)}",
        )
    if media_type == "video" and tensor.ndim not in {4, 5}:
        raise ValueError(
            "video reward artifact expects [C,T,H,W] or [B,C,T,H,W] tensor, "
            f"got shape={tuple(tensor.shape)}",
        )


def _fps(metadata: dict[str, Any]) -> float:
    value = metadata.get("video_fps", metadata.get("fps", 8.0))
    return float(value) if value is not None else 8.0


def _artifact_provenance(metadata: dict[str, Any]) -> dict[str, Any]:
    """Scalar metadata crosses the artifact wire; rich values stay driver-side.

    The rule replaces a hand-curated key list: the disk artifact rides a JSON
    wire to the reward service, so only JSON-scalar provenance (task type,
    reference/target paths, source ids) can cross — tensors, PIL images, and
    nested payloads (e.g. geneval dicts) are in-memory-transport data by
    nature. A predicate cannot forget a newly added provenance key.
    """

    return {
        key: value
        for key, value in metadata.items()
        if isinstance(value, (str, int, float, bool)) and str(value).strip()
    }


__all__ = [
    "ArtifactFormat",
    "DiskRewardArtifactStore",
    "InMemoryRewardArtifactStore",
    "MediaType",
    "RewardArtifactStore",
]
