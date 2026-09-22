"""Standalone scoring of existing media with immutable, resumable evidence.

Uses the same scorer transports as training without constructing a generator,
trainer, or Ray cluster. Each successful sample is atomically persisted; resume
requires identical inputs and scorer provenance. Calibration is deliberately
separate from raw measurements.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from vrl.config.base import ConfigBase
from vrl.config.reward_inference import RewardInferenceConfig
from vrl.rewards.inference import (
    RewardInferenceArtifact,
    RewardInferenceRequest,
    RewardInferenceResult,
)
from vrl.utils.artifacts import sha256_file


class ScoringConfig(ConfigBase):
    """Frozen evaluation recipe, independent of training configuration."""

    name: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    preprocessing_revision: str = Field(min_length=1)
    rubric_revision: str = Field(min_length=1)
    inference: RewardInferenceConfig = Field(
        default_factory=lambda: RewardInferenceConfig(kind="in_process")
    )
    worker_config: dict[str, Any] = Field(default_factory=dict)
    media_mode: Literal["file", "tensor"] = "file"
    batch_size: int = Field(default=1, gt=0, strict=True)

    @model_validator(mode="after")
    def validate_backend(self) -> ScoringConfig:
        if self.inference.kind == "ray":
            raise ValueError("standalone scoring uses in_process or operator-owned HTTP")
        if self.inference.kind == "http":
            if self.worker_config:
                raise ValueError("HTTP scoring configuration belongs to the service")
            if not self.inference.expected_model_version:
                raise ValueError("HTTP evaluation requires expected_model_version")
        elif not self.worker_config.get("model_factory"):
            raise ValueError("local scoring requires worker_config.model_factory")
        return self


class MediaRow(ConfigBase):
    """One existing sample; paths are relative to its manifest directory."""

    sample_id: str = Field(min_length=1)
    prompt_id: str = Field(min_length=1)
    prompt: str
    path: Path
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Auxiliary files (reference image, mask, etc.) enter metadata by key, and
    # their content digests participate in input identity just like the output.
    assets: dict[str, Path] = Field(default_factory=dict)


def fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Publish a complete record without leaving truncated success evidence."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_media_manifest(path: Path) -> list[MediaRow]:
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = MediaRow.model_validate_json(line)
            if row.sample_id in seen:
                raise ValueError(f"duplicate sample_id at {path}:{number}: {row.sample_id!r}")
            seen.add(row.sample_id)
            if set(row.assets) & set(row.metadata):
                raise ValueError(f"asset keys collide with metadata for {row.sample_id!r}")
            row.path = (path.parent / row.path.expanduser()).resolve(strict=True)
            row.assets = {
                key: (path.parent / asset.expanduser()).resolve(strict=True)
                for key, asset in row.assets.items()
            }
            if not all(item.is_file() for item in (row.path, *row.assets.values())):
                raise ValueError(f"media and assets must be files: {row.sample_id!r}")
            rows.append(row)
    if not rows:
        raise ValueError("media manifest is empty")
    return rows


def _input_record(row: MediaRow) -> dict[str, Any]:
    return {
        **row.model_dump(mode="json"),
        "sha256": sha256_file(row.path),
        "asset_sha256": {key: sha256_file(path) for key, path in row.assets.items()},
    }


def _artifact(row: MediaRow, record: dict[str, Any], media_mode: str) -> RewardInferenceArtifact:
    media = None
    if media_mode == "tensor":
        from vrl.rewards.models.media import decode_artifact_frames

        source = RewardInferenceArtifact(
            artifact_id=row.sample_id, sample_id=row.sample_id, path=str(row.path)
        )
        # Explicit RGB decode is part of this mode. Alpha-aware scorers must use
        # file mode until the transport has an explicit RGBA contract.
        frames = decode_artifact_frames(source)
        media = frames.permute(3, 0, 1, 2).contiguous()
    return RewardInferenceArtifact(
        artifact_id=row.sample_id,
        sample_id=row.sample_id,
        path=str(row.path) if media is None else "",
        prompt=row.prompt,
        metadata={**row.metadata, **{key: str(path) for key, path in row.assets.items()}},
        size_bytes=row.path.stat().st_size if media is None else None,
        sha256=record["sha256"] if media is None else None,
        media=media,
    )


async def score_manifest(
    manifest: Path,
    config: ScoringConfig,
    output_dir: Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Score and persist all axes, failing visibly on invalid or failed work.

    Resume reuses only successful rows. Failed batches are recorded and cause
    this call to raise; rerunning with resume retries them. File-mode HTTP needs
    shared paths allowed by the service. Auxiliary assets always need shared
    paths for remote execution, even with uploaded tensor media.
    """
    from vrl.rewards.runtime import build_reward_scorer

    rows = load_media_manifest(manifest.resolve())
    inputs = [_input_record(row) for row in rows]
    provenance = {
        "schema": "vrl.reward-evaluation.v1",
        "config": config.model_dump(mode="json"),
        "inputs": inputs,
    }
    run_id = fingerprint(provenance)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".writer.lock"
    # No stale-lock auto-recovery: another process may still own remote work.
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    scorer = None
    try:
        os.close(descriptor)
        provenance_path = output_dir / "provenance.json"
        if provenance_path.exists():
            if not resume:
                raise FileExistsError("evaluation exists; use resume or a new output directory")
            saved = json.loads(provenance_path.read_text())
            if saved != {"run_id": run_id, **provenance}:
                raise ValueError("resume inputs or scorer configuration changed")
        else:
            if any(path != lock_path for path in output_dir.iterdir()):
                raise ValueError("output directory has files but no evaluation provenance")
            atomic_json(provenance_path, {"run_id": run_id, **provenance})
        records_dir = output_dir / "samples"
        records_dir.mkdir(exist_ok=True)
        pending = []
        for row, record in zip(rows, inputs, strict=True):
            record_path = records_dir / f"{fingerprint(row.sample_id)}.json"
            if resume and record_path.exists():
                saved = json.loads(record_path.read_text())
                if saved.get("run_id") != run_id or saved.get("input") != record:
                    raise ValueError(f"incompatible cached sample: {row.sample_id}")
                if saved.get("status") == "success":
                    result = RewardInferenceResult(**saved["result"])
                    if result.artifact_id != row.sample_id or not result.scores:
                        raise ValueError(f"invalid cached result: {row.sample_id}")
                    pending.append((row, record, record_path, True))
                    continue
            pending.append((row, record, record_path, False))
        todo = [item for item in pending if not item[3]]
        if todo:
            scorer = build_reward_scorer(config.worker_config, inference=config.inference)
        for start in range(0, len(todo), config.batch_size):
            batch = todo[start : start + config.batch_size]
            try:
                for row, record, _, _ in batch:
                    if _input_record(row) != record:
                        raise ValueError(f"input changed during evaluation: {row.sample_id}")
                request = RewardInferenceRequest(
                    request_id=f"eval-{uuid.uuid4().hex}",
                    artifacts=tuple(
                        _artifact(row, record, config.media_mode) for row, record, _, _ in batch
                    ),
                )
                results = request.validate_and_order_results(
                    await asyncio.wait_for(scorer.score_batch(request), config.inference.timeout_s)
                )
                if any(not result.scores for result in results):
                    raise ValueError("scorer returned no score axes")
                for (row, record, path, _), result in zip(batch, results, strict=True):
                    if _input_record(row) != record:
                        raise ValueError(f"input changed during scoring: {row.sample_id}")
                    atomic_json(
                        path,
                        {
                            "run_id": run_id,
                            "input": record,
                            "status": "success",
                            "result": asdict(result),
                        },
                    )
            except Exception as error:
                for _, record, path, _ in batch:
                    atomic_json(
                        path,
                        {
                            "run_id": run_id,
                            "input": record,
                            "status": "error",
                            "error": {"type": type(error).__name__, "message": str(error)},
                        },
                    )
                raise
        summary = {
            "run_id": run_id,
            "samples": len(rows),
            "scored": len(todo),
            "reused": len(rows) - len(todo),
        }
        atomic_json(output_dir / "summary.json", summary)
        return summary
    finally:
        try:
            if scorer is not None:
                await scorer.shutdown()
        finally:
            lock_path.unlink(missing_ok=True)
