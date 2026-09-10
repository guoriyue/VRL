"""Immutable launch evidence for reproducible training investigations.

This records what the trainer process actually observed. It is not a model
support registry or a claim that the run completed, learned, or was deterministic.
Metrics, checkpoint contents and the final run verdict are separate evidence.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from vrl.models.checkpoint_identity import local_checkpoint_content

# These are file/protocol and environment boundaries, not algorithm vocabulary.
RUN_EVIDENCE_SCHEMA = "vrl.run-evidence/v1"
RUN_ARTIFACTS_SCHEMA = "vrl.run-artifacts/v1"
_RUNTIME_ENVIRONMENT_KEYS = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_LAUNCH_BLOCKING",
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTHONHASHSEED",
)


def _code_identity(repository: Path) -> dict[str, Any]:
    def git(*args: str) -> bytes:
        return subprocess.check_output(
            ["git", "-C", str(repository), *args],
            stderr=subprocess.DEVNULL,
            timeout=10,
        )

    try:
        commit = git("rev-parse", "HEAD").decode().strip()
        status = git("status", "--porcelain=v1", "--untracked-files=normal")
        diff = git("diff", "--binary", "HEAD")
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "reason": type(error).__name__}
    return {
        "available": True,
        "commit": commit,
        "dirty": bool(status),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        # Do not pretend a commit plus a dirty flag reproduces uncommitted code.
        "status": status.decode(errors="replace").splitlines(),
    }


def _runtime_identity() -> dict[str, Any]:
    packages = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }
    devices = []
    driver: dict[str, Any] = {"available": False}
    if torch.cuda.is_available():
        try:
            versions = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL,
                timeout=10,
                text=True,
            )
            driver = {"available": True, "versions": sorted(set(versions.splitlines()))}
        except (OSError, subprocess.SubprocessError) as error:
            driver["reason"] = type(error).__name__
        for index in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": device.name,
                    "memory_bytes": device.total_memory,
                    "compute_capability": [device.major, device.minor],
                }
            )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": dict(sorted(packages.items())),
        "torch_cuda": torch.version.cuda,
        "torch_build": torch.__config__.show(),
        "driver": driver,
        "cudnn": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "environment": {
            name: os.environ[name] for name in _RUNTIME_ENVIRONMENT_KEYS if name in os.environ
        },
        "device_scope": "trainer-process-visible-devices",
        "devices": devices,
    }


def _configured_data_files(config: Any) -> dict[str, Any]:
    """Hash declared manifests/reports, not the media files referenced by rows."""

    data = config.get("data") if isinstance(config, dict) else None
    if not isinstance(data, dict):
        return {}
    manifest = data.get("manifest")
    paths = list(manifest) if isinstance(manifest, dict) else ([manifest] if manifest else [])
    paths.extend(data[name] for name in ("eval_manifest", "source_report") if data.get(name))
    records = {}
    for name in dict.fromkeys(paths):
        path = Path(name).expanduser().resolve()
        try:
            with path.open("rb") as handle:
                before = os.fstat(handle.fileno())
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
                after = os.fstat(handle.fileno())
            records[name] = {
                "available": True,
                "sha256": digest,
                "bytes": after.st_size,
                "changed_during_read": (before.st_size, before.st_mtime_ns)
                != (after.st_size, after.st_mtime_ns),
            }
        except OSError as error:
            records[name] = {"available": False, "reason": type(error).__name__}
    return records


def write_run_evidence(
    cfg: Any,
    output_dir: str | Path,
    *,
    model_identity: dict[str, Any],
    resumed: bool,
    provided_examples: bool = False,
) -> Path:
    """Publish a distinct, immutable launch record, including on every resume.

    The caller has resolved/materialized the model and owns rank-local IO. The
    record deliberately contains no inferred evidence grade or success flag.
    A dirty checkout or missing source information remains visible to auditors.
    """

    if not isinstance(model_identity, dict) or not model_identity:
        raise ValueError("run evidence requires the resolved model identity")
    config = OmegaConf.to_container(cfg, resolve=True)
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    record = {
        "schema": RUN_EVIDENCE_SCHEMA,
        "launch_id": uuid.uuid4().hex,
        "captured_at": datetime.now(UTC).isoformat(),
        "phase": "before-training-loop",
        "resumed": bool(resumed),
        "config": config,
        "config_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "model_identity": model_identity,
        "provided_examples": bool(provided_examples),
        "configured_data_files": _configured_data_files(config),
        "code": _code_identity(Path(__file__).resolve().parents[2]),
        "runtime": _runtime_identity(),
    }
    directory = Path(output_dir) / "run_evidence"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{record['launch_id']}.json"
    _publish_record(destination, record)
    return destination


def _publish_record(destination: Path, record: dict[str, Any]) -> None:
    """Share atomic, non-overwriting publication across evidence records."""

    directory = destination.parent
    encoded = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # Atomic publication without overwriting a previous attempt, even if
        # a launch identifier collides.
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_launch(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("schema") != RUN_EVIDENCE_SCHEMA:
        raise ValueError(f"unsupported launch evidence: {path}")
    if record.get("launch_id") != path.stem:
        raise ValueError(f"launch identifier does not match evidence filename: {path}")
    canonical = json.dumps(
        record["config"], sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    if hashlib.sha256(canonical.encode()).hexdigest() != record.get("config_sha256"):
        raise ValueError(f"launch config digest mismatch: {path}")
    return record


def seal_run_artifacts(launch_path: str | Path) -> Path:
    """Bind final online-loop artifacts to this launch, before runtime cleanup.

    Hash complete checkpoint contents without deserializing tensors. This is an
    observation of bytes on disk, not a success verdict or recipe quality grade.
    Resuming in the same output directory can invalidate the old observation;
    archive the directory before resume to preserve independently verifiable runs.
    """

    launch_path = Path(launch_path)
    if launch_path.parent.name != "run_evidence" or launch_path.suffix != ".json":
        raise ValueError("launch evidence must be run_evidence/<launch_id>.json")
    launch = _read_launch(launch_path)
    output_dir = launch_path.parent.parent
    paths = {
        "launch": launch_path,
        "metrics": output_dir / "metrics.csv",
        "final_checkpoint": output_dir / "checkpoint-final",
    }
    artifacts = {
        role: {
            "path": path.relative_to(output_dir).as_posix(),
            "content": asdict(local_checkpoint_content(path)),
        }
        for role, path in paths.items()
    }
    # Reject malformed/empty artifacts; a directory standing in for metrics or
    # an empty checkpoint must not acquire a completion-looking receipt.
    if artifacts["metrics"]["content"]["kind"] != "file":
        raise ValueError("metrics artifact must be a file")
    checkpoint = artifacts["final_checkpoint"]["content"]
    if checkpoint["kind"] != "tree" or checkpoint["files"] == 0:
        raise ValueError("final checkpoint artifact must be a nonempty directory")
    record = {
        "schema": RUN_ARTIFACTS_SCHEMA,
        "launch_id": launch["launch_id"],
        "captured_at": datetime.now(UTC).isoformat(),
        "phase": "after-training-loop-before-cleanup",
        "artifacts": artifacts,
    }
    destination = launch_path.with_suffix(".artifacts.json")
    _publish_record(destination, record)
    return destination


def verify_run_artifacts(seal_path: str | Path) -> dict[str, Any]:
    """Fail on missing/replaced artifacts; never load checkpoint pickle payloads.

    The receipt itself needs an external trusted hash/archive for authenticity.
    Matching self-contained hashes establishes consistency, not authorship.
    """

    seal_path = Path(seal_path)
    if seal_path.parent.name != "run_evidence" or not seal_path.name.endswith(".artifacts.json"):
        raise ValueError("artifact evidence must be run_evidence/<launch_id>.artifacts.json")
    record = json.loads(seal_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("schema") != RUN_ARTIFACTS_SCHEMA:
        raise ValueError(f"unsupported artifact evidence: {seal_path}")
    launch_path = seal_path.with_name(seal_path.name.removesuffix(".artifacts.json") + ".json")
    launch = _read_launch(launch_path)
    if record.get("launch_id") != launch["launch_id"]:
        raise ValueError("artifact receipt belongs to a different launch")
    expected = {
        "launch": f"run_evidence/{launch_path.name}",
        "metrics": "metrics.csv",
        "final_checkpoint": "checkpoint-final",
    }
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(expected):
        raise ValueError("artifact receipt must bind launch, metrics, and final checkpoint")
    for role, relative in expected.items():
        artifact = artifacts[role]
        if not isinstance(artifact, dict) or artifact.get("path") != relative:
            raise ValueError(f"unexpected {role} artifact path")
        observed = asdict(local_checkpoint_content(seal_path.parent.parent / relative))
        if observed != artifact.get("content"):
            raise ValueError(f"{role} artifact content mismatch")
    return record
