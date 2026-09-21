"""The ``disk`` parking destination: frozen CPU shards as reclaimable file mappings.

Owns directory validation, mappings and temporary-file lifetime. It does not
move models between devices (that is ``ModelParking``, the ``move``
mechanism, which owns one of these when given a ``parking_directory``) or
choose when a role yields the GPU.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class FrozenDiskStore:
    """Replace frozen CPU storage with reclaimable file mappings.

    The caller refreshes framework-specific shard views after store(), and
    restores the model before cleanup(). Mappings preserve parameter objects.
    """

    # Linux filesystem classification and mount-table path, not user settings.
    _RAM_BACKED_FILESYSTEMS = frozenset({"tmpfs", "ramfs", "devtmpfs"})
    _MOUNTS = "/proc/mounts"

    def __init__(self, directory: str) -> None:
        self._directory = directory
        self._directories = []

    def store(self, parameters: Iterable[Any]) -> None:
        """Disk-backed frozen shards; bytes and parameter aliases stay intact.

        Anonymous CPU copies of every parked role can exceed host RAM. Shared
        file mappings let Linux reclaim inactive frozen shards while generation
        owns the GPUs. This is storage relocation only.
        """
        import tempfile
        from pathlib import Path

        import torch

        root = self._directory
        self._require_disk_backed_directory()
        directory = tempfile.TemporaryDirectory(prefix="trainer-", dir=root)
        self._directories.append(directory)
        with torch.no_grad():
            for index, parameter in enumerate(parameters):
                if parameter.requires_grad:
                    continue
                local = getattr(parameter, "_local_tensor", parameter)
                if local.device.type != "cpu" or not local.is_contiguous():
                    raise RuntimeError("disk parking requires contiguous CPU frozen shards")
                if local.numel() == 0:
                    continue
                mapped = torch.from_file(
                    str(Path(directory.name) / f"{index}.bin"),
                    shared=True,
                    size=local.numel(),
                    dtype=local.dtype,
                ).reshape(local.shape)
                mapped.copy_(local)
                if hasattr(parameter, "_local_tensor"):
                    parameter._local_tensor = mapped
                else:
                    parameter.data = mapped

    def _require_disk_backed_directory(self) -> None:
        """Refuse a parking directory that is missing or not a real disk.

        The longest mount point that prefixes the directory decides; hosts
        without ``/proc/mounts`` are not checked.
        """
        import os

        directory = self._directory
        assert directory is not None
        if not os.path.isdir(directory):
            raise ValueError(f"parking directory does not exist: {directory}")
        try:
            with open(self._MOUNTS, encoding="utf-8") as handle:
                entries = [line.split() for line in handle]
        except OSError:
            return
        resolved = os.path.realpath(directory)
        best: tuple[str, str] | None = None
        for entry in entries:
            if len(entry) < 3:
                continue
            mount_point, fstype = entry[1], entry[2]
            covers = resolved == mount_point or resolved.startswith(mount_point.rstrip("/") + "/")
            if covers and (best is None or len(mount_point) > len(best[0])):
                best = (mount_point, fstype)
        if best is not None and best[1] in self._RAM_BACKED_FILESYSTEMS:
            raise ValueError(
                f"parking directory {directory} is on a {best[1]} mount ({best[0]}); "
                "disk parking needs node-local disk such as NVMe, or the files stay in RAM",
            )

    def release_unused_host_memory(self) -> None:
        """Return discarded anonymous copies to the host allocator when possible."""
        if not self._directories:
            return
        import ctypes
        import gc

        gc.collect()
        trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
        if trim is not None:
            trim.argtypes = [ctypes.c_size_t]
            trim.restype = ctypes.c_int
            trim(0)

    def cleanup(self) -> None:
        """Remove backing files after model restoration; safe to repeat."""
        for directory in self._directories:
            directory.cleanup()
        self._directories.clear()
