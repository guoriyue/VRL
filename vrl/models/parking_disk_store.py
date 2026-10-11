"""The ``disk`` host storage of parking: parked CPU storage as reclaimable file mappings.

``ModelParking`` (the ``move`` mechanism) hands every tensor it has moved to
CPU to the store it was given. Owns directory validation, mappings and
temporary-file lifetime. It does not move tensors between devices or choose
when a role yields the GPU.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class ParkingDiskStore:
    """Replace parked CPU storage with reclaimable file mappings.

    The caller refreshes framework-specific shard views after store(), and
    restores the tensors before cleanup(). Mappings preserve parameter objects.
    """

    # Linux filesystem classification and mount-table path, not user settings.
    _RAM_BACKED_FILESYSTEMS = frozenset({"tmpfs", "ramfs", "devtmpfs"})
    _MOUNTS = "/proc/mounts"

    def __init__(self, directory: str) -> None:
        self._directory = directory
        self._directories = []

    def store(self, tensors: Iterable[Any]) -> None:
        """Copy each tensor from its device into a CPU file mapping, in place.

        Bytes and parameter aliases stay intact; the source storage is freed
        tensor by tensor. Anonymous CPU copies of every parked role can exceed
        host RAM. Shared file mappings let Linux reclaim inactive pages while
        another role owns the GPUs. A mapping is ordinary writable memory, so
        this is storage relocation only.
        """
        import tempfile
        from pathlib import Path

        import torch

        root = self._directory
        self._require_disk_backed_directory()
        directory = tempfile.TemporaryDirectory(prefix="parked-", dir=root)
        self._directories.append(directory)
        with torch.no_grad():
            for index, parameter in enumerate(tensors):
                local = getattr(parameter, "_local_tensor", parameter)
                if not local.is_contiguous():
                    raise RuntimeError("disk parking requires contiguous shards")
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
