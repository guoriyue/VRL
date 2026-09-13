"""Load the locked NVIDIA runtime libraries before torch can look elsewhere.

torch's wheel finds its CUDA libraries through an RPATH relative to a flat
site-packages (``$ORIGIN/../../nvidia/<lib>/lib``). rules_python keeps every
wheel in its own directory, so that RPATH misses and the dynamic linker falls
back to the host's ldconfig cache; a machine with a system CUDA install then
supplies ``libcudart.so.<major>`` from ``/usr/local/cuda``, which is neither
declared nor guaranteed to match. Loading the pinned libraries first (with
``RTLD_GLOBAL``, so later sonames resolve against them) makes the lock win.
This is torch's own ``_preload_cuda_deps`` strategy, applied unconditionally.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

_LIB_DIRS = ("nvidia/*/lib", "nvidia/cu*/lib")
# torch's own preload order (torch/__init__.py::_preload_cuda_deps): a library
# must come after the ones it links, or the linker fills the gap from the host.
_ORDERED_PREFIXES = (
    "libcublasLt.so",
    "libcublas.so",
    "libcudnn.so",
    "libnvrtc.so",
    "libnvrtc-builtins.so",
    "libcudart.so",
    "libcupti.so",
    "libcufft.so",
    "libcurand.so",
    "libnvJitLink.so",
    "libcusparse.so",
    "libcusparseLt.so",
    "libcusolver.so",
    "libnccl.so",
    "libnvshmem_host.so",
    "libcufile.so",
)
_PRELOADED: list[str] = []


def preload() -> list[str]:
    """Load every NVIDIA shared library found on ``sys.path``; return their paths."""

    if _PRELOADED or os.environ.get("VRL_SKIP_NVIDIA_PRELOAD"):
        return list(_PRELOADED)
    candidates: list[Path] = []
    for entry in sys.path:
        root = Path(entry)
        for pattern in _LIB_DIRS:
            for lib_dir in root.glob(pattern):
                candidates.extend(sorted(p for p in lib_dir.glob("lib*.so*") if p.is_file()))
    # Only the runtime libraries torch itself preloads. Others in those
    # directories are interposers or tools (libnvblas hijacks BLAS symbols).
    pending = sorted(
        (path for path in _dedupe(candidates) if path.name.startswith(_ORDERED_PREFIXES)),
        key=_load_rank,
    )
    # Keep retrying the ones that fail until a pass makes no progress.
    while pending:
        remaining: list[Path] = []
        for path in pending:
            try:
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                remaining.append(path)
            else:
                _PRELOADED.append(str(path))
        if len(remaining) == len(pending):
            break
        pending = remaining
    return list(_PRELOADED)


def _load_rank(path: Path) -> tuple[int, str]:
    for index, prefix in enumerate(_ORDERED_PREFIXES):
        if path.name.startswith(prefix):
            return (index, path.name)
    return (len(_ORDERED_PREFIXES), path.name)


def _dedupe(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = path.name
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


__all__ = ["preload"]
