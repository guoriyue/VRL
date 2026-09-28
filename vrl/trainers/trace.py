"""Immutable launch evidence for reproducible training investigations.

This records what the trainer process actually observed. It is not a model
support registry or a claim that the run completed, learned, or was deterministic.
Metrics, checkpoint contents and the final run result are separate evidence.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from vrl.models.precision import float32_precision_state
from vrl.utils.json_files import canonical_json_sha256, write_json

# These are file/protocol and environment boundaries, not algorithm vocabulary.
RUN_EVIDENCE_SCHEMA = "vrl.run-evidence/v1"
_RUNTIME_ENVIRONMENT_KEYS = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_LAUNCH_BLOCKING",
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTHONHASHSEED",
    "TORCHINDUCTOR_EMULATE_PRECISION_CASTS",
)


class TrainingRunTrace:
    """Own one launch's evidence record path."""

    def __init__(self, launch_path: Path) -> None:
        self.launch_path = launch_path

    @classmethod
    def capture(
        cls,
        cfg: Any,
        output_dir: str | Path,
        *,
        model_identity: dict[str, Any],
        resumed: bool,
        provided_examples: bool = False,
    ) -> TrainingRunTrace:
        """Publish a distinct, immutable launch record, including on every resume.

        The caller has resolved/materialized the model and owns rank-local IO. The
        record deliberately contains no inferred evidence grade or success flag.
        A dirty checkout or missing source information remains visible to auditors.
        """

        if not isinstance(model_identity, dict) or not model_identity:
            raise ValueError("run evidence requires the resolved model identity")
        config = OmegaConf.to_container(cfg, resolve=True)
        record = {
            "schema": RUN_EVIDENCE_SCHEMA,
            "launch_id": uuid.uuid4().hex,
            "captured_at": datetime.now(UTC).isoformat(),
            "phase": "before-training-loop",
            "resumed": bool(resumed),
            "config": config,
            "config_sha256": canonical_json_sha256(config, allow_nan=False),
            "model_identity": model_identity,
            "provided_examples": bool(provided_examples),
            "configured_data_files": cls._configured_data_files(config),
            "code": cls._git_snapshot(Path(__file__).resolve().parents[2]),
            "runtime": cls._runtime_snapshot(),
        }
        directory = Path(output_dir) / "run_evidence"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{record['launch_id']}.json"
        write_json(destination, record, overwrite=False)
        return cls(destination)

    @staticmethod
    def _git_snapshot(repository: Path) -> dict[str, Any]:
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

    @staticmethod
    def _runtime_snapshot() -> dict[str, Any]:
        packages = {}
        for distribution in importlib.metadata.distributions():
            metadata = distribution.metadata
            name = metadata["Name"]
            if name:
                packages[name] = metadata["Version"]
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
            "float32_precision": float32_precision_state(),
            "matmul_reduced_precision_reduction": {
                "fp16": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                "bf16": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            },
            "environment": {
                name: os.environ[name] for name in _RUNTIME_ENVIRONMENT_KEYS if name in os.environ
            },
            "device_scope": "trainer-process-visible-devices",
            "devices": devices,
        }

    @staticmethod
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
