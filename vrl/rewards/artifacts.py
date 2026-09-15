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
from typing import Any, Literal, Protocol, get_args, runtime_checkable

from vrl.rewards.inference import RewardInferenceArtifact
from vrl.rewards.types import RewardSample
from vrl.utils.artifacts import sha256_file

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
                    sample_id=sample.sample_id,
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
        name: str = "",
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
        # Reward component this store serves: the key under which the rollout
        # worker delivers pre-written files (RewardSample.artifacts).
        self.name = str(name)
        self._owned_paths: set[Path] = set()
        self.root.mkdir(parents=True, exist_ok=True)

    def spec(self) -> dict[str, Any]:
        """What a rollout worker needs to write this store's files itself."""

        return {
            "name": self.name,
            "root": str(self.root.resolve()),
            "media_type": self.media_type,
            "artifact_format": self.artifact_format,
        }

    def materialize(self, samples: list[RewardSample]) -> list[RewardInferenceArtifact]:
        artifacts: list[RewardInferenceArtifact] = []
        try:
            for sample in samples:
                delivered = sample.artifacts.get(self.name) if self.name else None
                artifacts.append(
                    self._write_one(sample)
                    if delivered is None
                    else self._adopt(sample, delivered)
                )
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

    def _adopt(self, sample: RewardSample, delivered: Any) -> RewardInferenceArtifact:
        """Take ownership of a file the rollout worker wrote for this store."""

        path = Path(delivered.path)
        if not path.is_absolute() or not path.exists():
            raise FileNotFoundError(
                f"worker-materialized reward artifact for {self.name!r} is missing: {path}",
            )
        expected_root = self.root.resolve()
        if expected_root not in path.resolve().parents:
            raise ValueError(
                f"worker-materialized reward artifact {path} lies outside this store's "
                f"root {expected_root}",
            )
        self._owned_paths.add(path.resolve())
        return RewardInferenceArtifact(
            artifact_id=f"{sample.sample_id}:{path.stem}",
            sample_id=sample.sample_id,
            path=str(path.resolve()),
            prompt=str(sample.prompt),
            size_bytes=int(delivered.size_bytes),
            sha256=str(delivered.sha256),
            metadata=_artifact_provenance(dict(sample.metadata or {})),
        )

    def _write_one(self, sample: RewardSample) -> RewardInferenceArtifact:
        import torch

        output = sample.output
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                f"{self.media_type} reward artifact materialization requires tensor sample output",
            )
        if output.numel() == 0:
            raise ValueError(f"{self.media_type} reward artifact tensor must be non-empty")
        if self.media_type == "image" and output.ndim not in {3, 4}:
            raise ValueError(
                "image reward artifact expects [C,H,W] or [B,C,H,W] tensor, "
                f"got shape={tuple(output.shape)}",
            )
        if self.media_type == "video" and output.ndim not in {4, 5}:
            raise ValueError(
                "video reward artifact expects [C,T,H,W] or [B,C,T,H,W] tensor, "
                f"got shape={tuple(output.shape)}",
            )

        tensor = output.detach().cpu()
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
                fps = metadata.get("video_fps", metadata.get("fps", 8.0))
                write_mp4(tensor, path, fps=float(fps) if fps is not None else 8.0)
            else:
                torch.save(tensor, path)
            size_bytes = path.stat().st_size
            # Per-request audit trails are the opt-in debug_dir JSONLs owned by
            # InferenceRewardFunction._write_debug; the store writes media only.
            artifact = RewardInferenceArtifact(
                artifact_id=artifact_id,
                sample_id=sample.sample_id,
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
