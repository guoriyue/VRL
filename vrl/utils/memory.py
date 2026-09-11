"""Host-memory instrumentation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from vrl.utils.logging import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class HostMemorySnapshot:
    """Current process RSS plus system memory totals in MiB."""

    rss_mb: float | None
    available_mb: float | None
    total_mb: float | None

    @classmethod
    def capture(cls) -> HostMemorySnapshot:
        """Capture Linux host memory without adding a psutil dependency."""

        return cls(
            rss_mb=_read_proc_field_mb("/proc/self/status", "VmRSS"),
            available_mb=_read_proc_field_mb("/proc/meminfo", "MemAvailable"),
            total_mb=_read_proc_field_mb("/proc/meminfo", "MemTotal"),
        )

    def __str__(self) -> str:
        """Format host-memory values with unknown fields omitted."""

        parts: list[str] = []
        if self.rss_mb is not None:
            parts.append(f"rss={self.rss_mb:.1f}MiB")
        if self.available_mb is not None:
            parts.append(f"available={self.available_mb:.1f}MiB")
        if self.total_mb is not None:
            parts.append(f"total={self.total_mb:.1f}MiB")
        used = self.used_fraction
        if used is not None:
            parts.append(f"used={used:.3f}")
        return " ".join(parts) if parts else "unavailable"

    @property
    def used_fraction(self) -> float | None:
        if self.available_mb is None or self.total_mb in (None, 0):
            return None
        return 1.0 - (self.available_mb / self.total_mb)


def log_host_memory(label: str, *, log: logging.Logger | None = None) -> HostMemorySnapshot:
    """Log a compact host-memory snapshot and return it for tests/hooks."""

    snapshot = HostMemorySnapshot.capture()
    target = log or logger
    target.info("host_memory[%s]: %s", label, snapshot)
    return snapshot


def _read_proc_field_mb(path: str, field: str) -> float | None:
    """Read ``field:`` (kB) from a /proc table and return it in MiB."""
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith(f"{field}:"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    return None
                return float(parts[1]) / 1024.0
    except OSError:
        return None
    return None


__all__ = [
    "HostMemorySnapshot",
    "log_host_memory",
]
