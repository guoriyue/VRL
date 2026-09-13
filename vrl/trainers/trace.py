"""Immutable launch evidence for reproducible training investigations.

This records what the trainer process actually observed. It is not a model
support registry or a claim that the run completed, learned, or was deterministic.
Metrics, checkpoint contents and the final run result are separate evidence.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from omegaconf import OmegaConf

from vrl.models.checkpoint_identity import LocalCheckpointContent
from vrl.models.precision import float32_precision_state
from vrl.utils.json_files import write_json

if TYPE_CHECKING:
    from vrl.scripts.eval.image_checkpoint_eval import EvaluationArchive

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
    "TORCHINDUCTOR_EMULATE_PRECISION_CASTS",
)


class TrainingRunTrace:
    """Own one launch's evidence paths and its artifact/verification lifecycle.

    Verification rereads records from disk so an existing object cannot hide
    modified evidence. Loading accepts either a launch record or its artifact receipt.
    """

    def __init__(self, launch_path: Path) -> None:
        self.launch_path = launch_path
        self.output_dir = launch_path.parent.parent
        self.artifacts_path = launch_path.with_suffix(".artifacts.json")

    @classmethod
    def load(cls, path: str | Path) -> TrainingRunTrace:
        path = Path(path)
        if path.name.endswith(".artifacts.json"):
            path = path.with_name(path.name.removesuffix(".artifacts.json") + ".json")
        if path.parent.name != "run_evidence" or path.suffix != ".json":
            raise ValueError("launch evidence must be run_evidence/<launch_id>.json")
        trace = cls(path)
        trace._read_launch()
        return trace

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
            "configured_data_files": cls._configured_data_files(config),
            "code": cls._git_snapshot(Path(__file__).resolve().parents[2]),
            "runtime": cls._runtime_snapshot(),
        }
        directory = Path(output_dir) / "run_evidence"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{record['launch_id']}.json"
        write_json(destination, record, overwrite=False)
        return cls.load(destination)

    def _read_launch(self) -> dict[str, Any]:
        path = self.launch_path
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

    def seal_artifacts(self) -> Path:
        """Bind final online-loop artifacts to this launch, before runtime cleanup.

        Hash complete checkpoint contents without deserializing tensors. This is an
        observation of bytes on disk, not a success result or recipe quality grade.
        Resuming in the same output directory can invalidate the old observation;
        archive the directory before resume to preserve independently verifiable runs.
        """

        launch_path = self.launch_path
        launch = self._read_launch()
        output_dir = self.output_dir
        paths = {
            "launch": launch_path,
            "metrics": output_dir / "metrics.csv",
            "final_checkpoint": output_dir / "checkpoint-final",
        }
        full_precision = output_dir / "metrics.full_precision.csv"
        if full_precision.exists():
            paths["full_precision_metrics"] = full_precision
        artifacts = {
            role: {
                "path": path.relative_to(output_dir).as_posix(),
                "content": asdict(LocalCheckpointContent.from_path(path)),
            }
            for role, path in paths.items()
        }
        # Reject malformed/empty artifacts; a directory standing in for metrics or
        # an empty checkpoint must not acquire a completion-looking receipt.
        for role in ("metrics", "full_precision_metrics"):
            if role in artifacts and artifacts[role]["content"]["kind"] != "file":
                raise ValueError(f"{role} artifact must be a file")
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
        destination = self.artifacts_path
        write_json(destination, record, overwrite=False)
        return destination

    def verify_artifacts(self) -> dict[str, Any]:
        """Fail on missing/replaced artifacts; never load checkpoint pickle payloads.

        The receipt itself needs an external trusted hash/archive for authenticity.
        Matching self-contained hashes establishes consistency, not authorship.
        """

        seal_path = self.artifacts_path
        record = json.loads(seal_path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or record.get("schema") != RUN_ARTIFACTS_SCHEMA:
            raise ValueError(f"unsupported artifact evidence: {seal_path}")
        launch_path = self.launch_path
        launch = self._read_launch()
        if record.get("launch_id") != launch["launch_id"]:
            raise ValueError("artifact receipt belongs to a different launch")
        expected = {
            "launch": f"run_evidence/{launch_path.name}",
            "metrics": "metrics.csv",
            "final_checkpoint": "checkpoint-final",
        }
        artifacts = record.get("artifacts")
        if isinstance(artifacts, dict) and "full_precision_metrics" in artifacts:
            expected["full_precision_metrics"] = "metrics.full_precision.csv"
        if not isinstance(artifacts, dict) or set(artifacts) != set(expected):
            raise ValueError("artifact receipt must bind launch, metrics, and final checkpoint")
        for role, relative in expected.items():
            artifact = artifacts[role]
            if not isinstance(artifact, dict) or artifact.get("path") != relative:
                raise ValueError(f"unexpected {role} artifact path")
            observed = asdict(LocalCheckpointContent.from_path(self.output_dir / relative))
            if observed != artifact.get("content"):
                raise ValueError(f"{role} artifact content mismatch")
        return record

    def verify_completion(self, result_path: str | Path) -> dict[str, Any]:
        """Check artifact integrity and the supplied successful process outcome.

        There is no cross-process attempt identity: this does not establish that
        the supplied result and artifacts came from the same execution.
        """

        self.verify_artifacts()
        launch = self._read_launch()
        result = json.loads(Path(result_path).read_text(encoding="utf-8"))
        if not isinstance(result, dict) or result.get("schema_version") != 1:
            raise ValueError("unsupported run result")
        if result.get("status") != "success":
            raise ValueError("run attempt did not complete successfully")
        if (
            type(result.get("supervisor_exit_code")) is not int
            or result["supervisor_exit_code"] != 0
        ):
            raise ValueError("supervisor did not observe a successful process exit")
        world_size = int(launch["runtime"].get("environment", {}).get("WORLD_SIZE", "1"))
        if world_size > 1:
            ranks = result.get("rank_results")
            if result.get("world_size") != world_size or not isinstance(ranks, list):
                raise ValueError("distributed completion requires an aggregate result")
            if len(ranks) != world_size:
                raise ValueError("distributed completion is missing rank results")
            seen = set()
            for rank in ranks:
                if (
                    not isinstance(rank, dict)
                    or type(rank.get("rank")) is not int
                    or rank["rank"] not in range(world_size)
                    or rank["rank"] in seen
                    or rank.get("world_size") != world_size
                    or rank.get("status") != "success"
                ):
                    raise ValueError("distributed completion contains an invalid rank result")
                seen.add(rank["rank"])
        elif "rank" in result or "rank_results" in result or result.get("world_size", 1) != 1:
            raise ValueError("single-process launch has a distributed result")
        return result

    def verify_evaluation(
        self,
        result_path: str | Path,
        archive: EvaluationArchive,
    ) -> dict[str, Any]:
        """Associate a completed image evaluation with the exact final trained state.

        The caller supplies the expected EvaluationPlan through its archive. Never
        reconstruct the intended protocol from the report being checked. Checkpoint
        labels and paths are presentation: content and model identity establish the
        association. This does not certify held-out data independence or learning.
        """

        from vrl.trainers.checkpointing import TRAINING_CHECKPOINT_NAME
        from vrl.utils.artifacts import sha256_file

        self.verify_completion(result_path)
        launch = self._read_launch()
        report = archive.verify_report()
        protocol = report["protocol"]
        if protocol["model_identity"] != launch["model_identity"]:
            raise ValueError("evaluation model identity differs from the training launch")
        final_checkpoint = self.output_dir / "checkpoint-final" / TRAINING_CHECKPOINT_NAME
        digest = sha256_file(final_checkpoint)
        labels = [
            target["label"]
            for target in protocol["targets"]
            if target["path"] and target["checkpoint_sha256"] == digest
        ]
        if not labels:
            raise ValueError("evaluation does not contain the final training checkpoint content")
        canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return {
            "launch_id": launch["launch_id"],
            "checkpoint_sha256": digest,
            "checkpoint_labels": labels,
            "evaluation_protocol_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "evaluation_content": asdict(LocalCheckpointContent.from_path(archive.directory)),
        }

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
