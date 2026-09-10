"""AdamW moments streamed through bounded, parameter-aligned disk buckets.

Model parameters/FP32 masters and gradients remain owned by the existing trainer.
Scratch files are not resumable checkpoints: state_dict exports portable moments.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import torch

from vrl.utils.artifacts import sha256_file


class DiskStreamingAdamW(torch.optim.Optimizer):
    """Keep moments on disk between steps; fail terminally after a partial update."""

    def __init__(self, parameters, *, directory, bucket_bytes, lr, betas, eps, weight_decay):
        self._locked = False
        inner = torch.optim.AdamW(
            parameters,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            foreach=False,
            fused=False,
        )
        super().__init__(inner.param_groups, inner.defaults)
        self._inner = inner
        self.param_groups = inner.param_groups
        self.state = inner.state
        self._parameters = tuple(p for group in self.param_groups for p in group["params"])
        if len({id(p) for p in self._parameters}) != len(self._parameters):
            raise ValueError("disk AdamW requires unique parameters")
        if type(bucket_bytes) is not int or bucket_bytes < 1:
            raise ValueError("optimizer bucket_bytes must be a positive integer")
        self._buckets = []
        self._initialized_parameters: set[int] = set()
        self._validate_groups(self.param_groups)
        index = 0
        for group_index, group in enumerate(self.param_groups):
            bucket, size = [], 0
            for parameter in group["params"]:
                if (
                    parameter.dtype != torch.float32
                    or parameter.layout != torch.strided
                    or parameter.is_meta
                    or hasattr(parameter, "placements")
                ):
                    raise NotImplementedError(
                        "disk AdamW requires plain FP32 parameters/masters; no FSDP shards"
                    )
                required = parameter.numel() * 8 + 4  # two FP32 moments plus scalar step
                if required > bucket_bytes:
                    raise ValueError(
                        f"parameter {index} needs {required} moment bytes; increase bucket_bytes"
                    )
                if bucket and size + required > bucket_bytes:
                    self._buckets.append((group_index, tuple(bucket)))
                    bucket, size = [], 0
                bucket.append(index)
                size += required
                index += 1
            if bucket:
                self._buckets.append((group_index, tuple(bucket)))
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        self._workspace = tempfile.TemporaryDirectory(prefix="adamw-", dir=root)
        self._current: Path | None = None
        self._hashes: dict[str, str] = {}
        self._generation = 0
        self._failed = False
        self._group_parameters = tuple(tuple(g["params"]) for g in self.param_groups)
        self._locked = True

    def add_param_group(self, group):
        if self._locked:
            raise RuntimeError("disk AdamW does not support changing parameter layout")
        return super().add_param_group(group)

    @staticmethod
    def _validate_groups(groups):
        for group in groups:
            if any(
                group.get(key, False)
                for key in ("amsgrad", "capturable", "differentiable", "fused", "foreach")
            ):
                raise ValueError("disk AdamW requires non-fused, non-capturable standard AdamW")
            if not group.get("decoupled_weight_decay", True):
                raise ValueError("disk AdamW requires decoupled weight decay")
            if isinstance(group["lr"], torch.Tensor):
                raise ValueError("disk AdamW requires a scalar learning rate")
            # Delegate numerical hyperparameter validation to the native owner.
            torch.optim.AdamW(
                [torch.empty(0)],
                lr=group["lr"],
                betas=group["betas"],
                eps=group["eps"],
                weight_decay=group["weight_decay"],
            )

    def _require_intact(self):
        if len(self.param_groups) != len(self._group_parameters) or any(
            len(group["params"]) != len(expected)
            or any(p is not q for p, q in zip(group["params"], expected, strict=True))
            for group, expected in zip(self.param_groups, self._group_parameters, strict=True)
        ):
            raise RuntimeError("disk AdamW parameter layout changed after construction")
        if self._failed:
            raise RuntimeError(
                "disk AdamW failed during update; restore a checkpoint into a new optimizer"
            )

    def _parameter_layout(self):
        return {
            "schema": 1,
            "parameters": [
                {"shape": list(p.shape), "dtype": str(p.dtype)} for p in self._parameters
            ],
            "buckets": [[group, list(indices)] for group, indices in self._buckets],
        }

    def _read_bucket(self, index, *, mmap=False):
        if self._current is None:
            return {}
        name = f"bucket-{index}.pt"
        path = self._current / name
        if sha256_file(path) != self._hashes[name]:
            raise ValueError(f"optimizer bucket checksum mismatch: {name}")
        return torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)

    @staticmethod
    def _write_bucket(directory, index, state):
        path = directory / f"bucket-{index}.pt"
        with path.open("xb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        return path.name, sha256_file(path)

    def _commit_generation(self, staging, hashes):
        destination = Path(self._workspace.name) / f"generation-{self._generation + 1}"
        staging.rename(destination)
        previous = self._current
        self._current, self._hashes = destination, hashes
        self._generation += 1
        if previous is not None:
            shutil.rmtree(previous)

    def step(self, closure=None):
        self._require_intact()
        self._validate_groups(self.param_groups)
        if closure is not None:
            raise NotImplementedError("disk AdamW does not support closures")
        if any(p.grad is not None and p.grad.is_sparse for p in self._parameters):
            raise RuntimeError("AdamW does not support sparse gradients")
        staging = Path(tempfile.mkdtemp(prefix="pending-", dir=self._workspace.name))
        hashes = {}
        initialized = set()
        try:
            for bucket_index, (group_index, indices) in enumerate(self._buckets):
                saved = self._read_bucket(bucket_index)
                parameters = [self._parameters[i] for i in indices]
                for index, parameter in zip(indices, parameters, strict=True):
                    if index in saved:
                        self.state[parameter] = {
                            key: value.to(parameter.device) if key != "step" else value
                            for key, value in saved[index].items()
                        }
                    elif parameter.grad is not None:
                        # Explicit scalar dtype keeps the bucket accounting independent
                        # of the process-wide default dtype.
                        self.state[parameter] = {
                            "step": torch.zeros((), dtype=torch.float32, device="cpu"),
                            "exp_avg": torch.zeros_like(parameter),
                            "exp_avg_sq": torch.zeros_like(parameter),
                        }
                del saved
                self._inner.param_groups = [
                    {**self.param_groups[group_index], "params": parameters}
                ]
                self._inner.step()
                exported = {
                    index: {
                        key: value.detach().cpu() for key, value in self.state[parameter].items()
                    }
                    for index, parameter in zip(indices, parameters, strict=True)
                    if parameter in self.state
                }
                name, digest = self._write_bucket(staging, bucket_index, exported)
                hashes[name] = digest
                initialized.update(exported)
                self.state.clear()
                del exported
            self._commit_generation(staging, hashes)
            self._initialized_parameters = initialized
        except BaseException:
            self._failed = True
            raise
        finally:
            self.state.clear()
            self._inner.param_groups = self.param_groups
            if staging.exists():
                shutil.rmtree(staging)

    def state_dict(self):
        self._require_intact()
        result = super().state_dict()
        # File-backed views avoid allocating all moments merely to export a
        # checkpoint. Serialization may still fault all pages through host RAM.
        result["state"] = {}
        for index in range(len(self._buckets)):
            result["state"].update(self._read_bucket(index, mmap=True))
        result["disk_streaming"] = {
            "layout": self._parameter_layout(),
            "initialized_parameters": sorted(self._initialized_parameters),
        }
        return result

    def load_state_dict(self, state_dict):
        self._require_intact()
        metadata = state_dict.get("disk_streaming", {})
        if metadata.get("layout") != self._parameter_layout():
            raise ValueError("disk AdamW checkpoint parameter/bucket layout differs")
        groups = state_dict["param_groups"]
        if len(groups) != len(self.param_groups):
            raise ValueError("disk AdamW checkpoint parameter groups differ")
        expected = super().state_dict()["param_groups"]
        if any(a["params"] != b["params"] for a, b in zip(groups, expected, strict=True)):
            raise ValueError("disk AdamW checkpoint parameter order differs")
        self._validate_groups(groups)
        state = state_dict["state"]
        if sorted(state) != metadata.get("initialized_parameters"):
            raise ValueError("disk AdamW checkpoint is missing initialized parameter state")
        if set(state) - set(range(len(self._parameters))):
            raise ValueError("disk AdamW checkpoint contains unknown parameter state")
        for index, values in state.items():
            if set(values) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("disk AdamW checkpoint has invalid moment fields")
            if any(
                values[key].shape != self._parameters[index].shape
                or values[key].dtype != torch.float32
                for key in ("exp_avg", "exp_avg_sq")
            ):
                raise ValueError("disk AdamW checkpoint moment shape/dtype differs")
            if (
                values["step"].shape != torch.Size([])
                or values["step"].dtype != torch.float32
                or not torch.isfinite(values["step"]).all()
                or values["step"].item() < 1
                or values["step"].item() != int(values["step"].item())
            ):
                raise ValueError("disk AdamW checkpoint has invalid step")
        staging = Path(tempfile.mkdtemp(prefix="restore-", dir=self._workspace.name))
        try:
            hashes = {}
            for bucket_index, (_, indices) in enumerate(self._buckets):
                bucket = {
                    index: {key: value.detach().cpu() for key, value in state[index].items()}
                    for index in indices
                    if index in state
                }
                name, digest = self._write_bucket(staging, bucket_index, bucket)
                hashes[name] = digest
            self._commit_generation(staging, hashes)
            self._initialized_parameters = set(state)
            for target, saved in zip(self.param_groups, groups, strict=True):
                target.update({key: value for key, value in saved.items() if key != "params"})
        except BaseException:
            self._failed = True
            raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
