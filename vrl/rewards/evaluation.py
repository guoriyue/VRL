"""Score existing media into a resumable scoring run, and read one back.

An ``Evaluation`` is the unit every offline reward tool consumes: the scoring
recipe, one input record per sample (with content digests) and one result or
error per sample. ``Evaluation.score`` produces it with the same scorer
transports as training, without a generator, trainer or Ray cluster; each
sample is persisted as it completes, so a killed run resumes from what it
finished. ``Evaluation.load`` reads a persisted run.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import asdict, dataclass, field
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
from vrl.utils.json_files import canonical_json_sha256, write_json

EVALUATION_SCHEMA = "vrl.reward-evaluation.v1"


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
    # A producer may note the digest it wrote; the run records its own digest.
    sha256: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Auxiliary files (reference image, mask, ...) enter reward metadata by key.
    assets: dict[str, Path] = Field(default_factory=dict)


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
            row.path = (path.parent / row.path.expanduser()).resolve(strict=True)
            row.assets = {
                key: (path.parent / asset.expanduser()).resolve(strict=True)
                for key, asset in row.assets.items()
            }
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


def _asset_metadata(assets: dict[str, Path]) -> dict[str, Any]:
    """Manifest assets as reward metadata; ``reference_image`` becomes the one-element list."""

    metadata: dict[str, Any] = {key: str(path) for key, path in assets.items()}
    reference = metadata.pop("reference_image", None)
    if reference is not None:
        metadata["reference_images"] = [reference]
    return metadata


def _artifact(row: MediaRow, record: dict[str, Any], media_mode: str) -> RewardInferenceArtifact:
    media = None
    if media_mode == "tensor":
        from vrl.rewards.models.media import decode_artifact_frames

        source = RewardInferenceArtifact(
            artifact_id=row.sample_id, sample_id=row.sample_id, path=str(row.path)
        )
        # Explicit RGB decode is part of this mode; alpha-aware scorers use file mode.
        media = decode_artifact_frames(source).permute(3, 0, 1, 2).contiguous()
    return RewardInferenceArtifact(
        artifact_id=row.sample_id,
        sample_id=row.sample_id,
        path=str(row.path) if media is None else "",
        prompt=row.prompt,
        metadata={**row.metadata, **_asset_metadata(row.assets)},
        size_bytes=row.path.stat().st_size if media is None else None,
        sha256=record["sha256"] if media is None else None,
        media=media,
    )


def _record_path(directory: Path, sample_id: str) -> Path:
    return directory / "samples" / f"{canonical_json_sha256(sample_id, allow_nan=False)}.json"


@dataclass
class Evaluation:
    """One scoring run: its recipe, and one input plus result (or error) per sample.

    ``records`` maps sample id to ``{"input", "status", "result" | "error"}`` with
    status ``success``, ``error`` or ``missing``. ``run_id`` identifies the recipe
    and inputs, not one inference attempt.
    """

    run_id: str
    config: dict[str, Any]
    records: dict[str, dict[str, Any]]
    schema: str = EVALUATION_SCHEMA
    # Scoring runs joined from several scorers keep their sources here.
    source_run_ids: dict[str, str] = field(default_factory=dict)
    # Filled by ``score``: how many samples were scored now versus reused.
    summary: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, directory: Path) -> Evaluation:
        """Read a persisted run; samples without a record are ``missing``."""

        provenance = json.loads((directory / "provenance.json").read_text())
        records = {}
        for item in provenance["inputs"]:
            path = _record_path(directory, item["sample_id"])
            if path.exists():
                record = json.loads(path.read_text())
                record.pop("run_id", None)
                records[item["sample_id"]] = record
            else:
                records[item["sample_id"]] = {"input": item, "status": "missing"}
        return cls(provenance["run_id"], provenance["config"], records)

    @classmethod
    async def score(
        cls, manifest: Path, config: ScoringConfig, output_dir: Path, *, resume: bool = False
    ) -> Evaluation:
        """Score every manifest row and persist the run under ``output_dir``.

        ``resume`` reuses samples the same run already scored successfully and
        retries the rest; the inputs and recipe must be the ones the run was
        started with. A failed batch is recorded per sample and re-raised.
        """

        from vrl.rewards.runtime import build_reward_scorer

        rows = load_media_manifest(manifest.resolve())
        inputs = [_input_record(row) for row in rows]
        provenance = {"schema": EVALUATION_SCHEMA, "config": config.model_dump(mode="json")}
        provenance = {
            **provenance,
            "inputs": inputs,
            "run_id": canonical_json_sha256({**provenance, "inputs": inputs}, allow_nan=False),
        }
        run_id = provenance["run_id"]
        output_dir.mkdir(parents=True, exist_ok=True)
        provenance_path = output_dir / "provenance.json"
        if provenance_path.exists():
            if not resume:
                raise FileExistsError("evaluation exists; use resume or a new output directory")
            if json.loads(provenance_path.read_text()) != provenance:
                raise ValueError("resume inputs or scorer configuration changed")
        else:
            write_json(provenance_path, provenance)
        (output_dir / "samples").mkdir(exist_ok=True)
        todo = []
        for row, record in zip(rows, inputs, strict=True):
            path = _record_path(output_dir, row.sample_id)
            if resume and path.exists():
                saved = json.loads(path.read_text())
                if saved.get("run_id") == run_id and saved.get("status") == "success":
                    continue
            todo.append((row, record, path))
        scorer = None
        try:
            if todo:
                scorer = build_reward_scorer(config.worker_config, inference=config.inference)
            for start in range(0, len(todo), config.batch_size):
                batch = todo[start : start + config.batch_size]
                try:
                    request = RewardInferenceRequest(
                        request_id=f"eval-{uuid.uuid4().hex}",
                        artifacts=tuple(
                            _artifact(row, record, config.media_mode) for row, record, _ in batch
                        ),
                    )
                    results = request.validate_and_order_results(
                        await asyncio.wait_for(
                            scorer.score_batch(request), config.inference.timeout_s
                        )
                    )
                    if any(not result.scores for result in results):
                        raise ValueError("scorer returned no score axes")
                    for (_, record, path), result in zip(batch, results, strict=True):
                        write_json(
                            path,
                            {
                                "run_id": run_id,
                                "input": record,
                                "status": "success",
                                "result": asdict(result),
                            },
                        )
                except Exception as error:
                    for _, record, path in batch:
                        write_json(
                            path,
                            {
                                "run_id": run_id,
                                "input": record,
                                "status": "error",
                                "error": {"type": type(error).__name__, "message": str(error)},
                            },
                        )
                    raise
        finally:
            if scorer is not None:
                await scorer.shutdown()
        summary = {"samples": len(rows), "scored": len(todo), "reused": len(rows) - len(todo)}
        write_json(output_dir / "summary.json", {"run_id": run_id, **summary})
        evaluation = cls.load(output_dir)
        evaluation.summary = summary
        return evaluation

    def results(self) -> dict[str, RewardInferenceResult]:
        """Successful samples as typed results."""

        return {
            sample_id: RewardInferenceResult(**row["result"])
            for sample_id, row in self.records.items()
            if row["status"] == "success"
        }


__all__ = [
    "EVALUATION_SCHEMA",
    "Evaluation",
    "MediaRow",
    "ScoringConfig",
    "load_media_manifest",
]
