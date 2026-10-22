"""Make host tensors DMA sources: pinned where they are, or in one shared tmpfs copy.

A copy from pageable host memory is bounded by the host's own memcpy (about
17 GB/s here, however many threads do it), a copy from pinned memory by the
link (about 50 GB/s). Pinning a tensor in place is one ``cudaHostRegister``
over its pages, except for the way a checkpoint loader leaves its weights: as
private mappings of the shard files. Registering such a page makes the kernel
copy it into the process (copy-on-write), so two processes streaming one
checkpoint would each hold a private 40 GB, and a shared or read-only mapping
of a disk file is refused by the driver. Those tensors are copied once into one
file on tmpfs, keyed by their content, which every process on the host maps,
registers and reads from: one pinned physical copy, found again by later runs.
"""

from __future__ import annotations

import bisect
import concurrent.futures
import fcntl
import hashlib
import logging
import os
from collections.abc import Iterable
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

PAGE_BYTES = 4096
DEFAULT_SHARE_DIRECTORY = Path("/dev/shm/vrl-host-streaming")
MEMORY_FILESYSTEMS = frozenset({"tmpfs", "ramfs", "hugetlbfs"})

NamedTensors = list[tuple[str, torch.Tensor]]


class HostPinning:
    """Pins host tensors for one process: in place, or through the shared copy under ``share_directory``."""

    def __init__(self, share_directory: Path | str = DEFAULT_SHARE_DIRECTORY) -> None:
        self.share_directory = Path(share_directory)
        with open("/proc/mounts") as mounts:
            rows = [line.split() for line in mounts]
        self._mounts = [(fields[1], fields[2]) for fields in rows if len(fields) >= 3]

    def pin(self, named: Iterable[tuple[str, torch.Tensor]]) -> int:
        """Pin every CPU tensor of ``named`` in place or in the shared copy; return bytes pinned.

        Tensors already pinned or not on the CPU are left alone. File-backed
        tensors are re-pointed at the shared copy (``Tensor.data``), so the
        objects their modules hold keep their identity and names. What cannot
        be pinned stays pageable with a warning, slower but not wrong: a share
        directory that is not tmpfs or is too small, or pages the driver
        refuses to register.
        """

        mappings = self._disk_file_mappings()
        anonymous: list[torch.Tensor] = []
        file_backed: NamedTensors = []
        for name, tensor in named:
            if tensor.device.type != "cpu" or tensor.is_pinned():
                continue
            pointer = tensor.untyped_storage().data_ptr()
            index = bisect.bisect_right(mappings, (pointer, float("inf"))) - 1
            if index >= 0 and mappings[index][0] <= pointer < mappings[index][1]:
                file_backed.append((name, tensor))
            else:
                anonymous.append(tensor)
        pinned = 0
        try:
            if anonymous:
                pinned += self._register(anonymous)
            if file_backed:
                pinned += self._shared_copy(file_backed)
        except RuntimeError as error:
            logger.warning("host pinning: %s; the remaining weights stay pageable", error)
        return pinned

    @staticmethod
    def _register(tensors: Iterable[torch.Tensor]) -> int:
        """``cudaHostRegister`` the pages behind ``tensors``, each page once; return the bytes pinned."""

        ranges = sorted(
            (
                tensor.untyped_storage().data_ptr() // PAGE_BYTES * PAGE_BYTES,
                -(
                    -(tensor.untyped_storage().data_ptr() + tensor.untyped_storage().nbytes())
                    // PAGE_BYTES
                )
                * PAGE_BYTES,
            )
            for tensor in tensors
        )
        merged: list[tuple[int, int]] = []
        for start, stop in ranges:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
            else:
                merged.append((start, stop))
        torch.cuda.init()
        cudart = torch.cuda.cudart()
        pinned = 0
        for start, stop in merged:
            code = int(cudart.cudaHostRegister(start, stop - start, 0))
            if code != 0:
                raise RuntimeError(
                    f"cudaHostRegister refused {(stop - start) / 2**20:.0f} MiB of host memory: "
                    f"{cudart.cudaGetErrorString(cudart.cudaError(code))}"
                )
            pinned += stop - start
        return pinned

    def _on_memory_filesystem(self, path: str | Path) -> bool:
        """Whether ``path`` lives on a memory filesystem (pinnable, shared between processes)."""

        path = str(path)
        deepest = ""
        fstype = ""
        for point, kind in self._mounts:
            if (path == point or path.startswith(point.rstrip("/") + "/")) and len(point) > len(
                deepest
            ):
                deepest, fstype = point, kind
        return fstype in MEMORY_FILESYSTEMS

    def _disk_file_mappings(self) -> list[tuple[int, int]]:
        """Sorted address ranges of this process's mappings of files on disk."""

        ranges: list[tuple[int, int]] = []
        with open("/proc/self/maps") as maps:
            for line in maps:
                fields = line.split()
                if len(fields) < 6 or not fields[5].startswith("/"):
                    continue
                if self._on_memory_filesystem(fields[5]):
                    continue
                start, stop = (int(part, 16) for part in fields[0].split("-"))
                ranges.append((start, stop))
        ranges.sort()
        return ranges

    def _shared_copy(self, named: NamedTensors) -> int:
        """Point ``named`` at a registered tmpfs copy of their bytes; return the bytes pinned.

        The copy is keyed by a digest of every tensor's name, dtype, shape and
        full content (not a sample: a false hit would silently train on
        another checkpoint's weights), written under a lock and published by
        rename, so concurrent processes produce one file and later runs find
        it.
        """

        directory = self.share_directory
        offsets = []
        total = 0
        for _, tensor in named:
            offsets.append(total)
            total += -(-tensor.numel() * tensor.element_size() // PAGE_BYTES) * PAGE_BYTES
        directory.mkdir(parents=True, exist_ok=True)
        if not self._on_memory_filesystem(directory):
            logger.warning(
                "host pinning: %s is not tmpfs, so the driver cannot pin a file there; "
                "%.1f GiB of file-backed weights stay pageable",
                directory,
                total / 2**30,
            )
            return 0

        def as_bytes(tensor: torch.Tensor) -> memoryview:
            return memoryview(tensor.detach().contiguous().view(torch.uint8).numpy())

        def digest(item: tuple[str, torch.Tensor]) -> bytes:
            name, tensor = item
            hasher = hashlib.blake2b(digest_size=16)
            hasher.update(f"{name}|{tensor.dtype}|{tuple(tensor.shape)}|".encode())
            hasher.update(as_bytes(tensor))
            return hasher.digest()

        # Hashing is memory-bound and hashlib releases the GIL, so threads help.
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            digests = b"".join(pool.map(digest, named))
        path = directory / f"{hashlib.blake2b(digests, digest_size=16).hexdigest()}.bin"

        if not path.exists():
            stats = os.statvfs(directory)
            if stats.f_bavail * stats.f_frsize < total:
                logger.warning(
                    "host pinning: %s has %.1f GiB free, %.1f GiB needed; "
                    "file-backed weights stay pageable",
                    directory,
                    stats.f_bavail * stats.f_frsize / 2**30,
                    total / 2**30,
                )
                return 0
            with open(path.with_suffix(".lock"), "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if not path.exists():
                    partial = path.with_suffix(f".{os.getpid()}.partial")
                    with open(partial, "wb") as out:
                        for (_, tensor), offset in zip(named, offsets, strict=True):
                            out.seek(offset)
                            out.write(as_bytes(tensor))
                        out.truncate(total)
                    os.replace(partial, path)

        shared = torch.from_file(str(path), shared=True, size=total, dtype=torch.uint8)
        pinned = self._register([shared])
        for (_, tensor), offset in zip(named, offsets, strict=True):
            stop = offset + tensor.numel() * tensor.element_size()
            tensor.data = shared[offset:stop].view(tensor.dtype).view(tensor.shape)
        return pinned


__all__ = ["DEFAULT_SHARE_DIRECTORY", "HostPinning"]
