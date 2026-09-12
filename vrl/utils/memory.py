"""Host-memory instrumentation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from vrl.utils.logging import init_logger

_logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class HostMemorySnapshot:
    """Current process RSS plus system memory totals in MiB."""

    rss_mb: float | None
    available_mb: float | None
    total_mb: float | None

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


class HostMemoryMonitor:
    """Read host memory and log measurements through one configured logger."""

    def __init__(
        self,
        *,
        logger: logging.Logger | None = None,
        proc_root: str | Path = "/proc",
    ) -> None:
        self.logger = logger if logger is not None else _logger
        self.proc_root = Path(proc_root)

    def capture(self) -> HostMemorySnapshot:
        """Read process RSS and system totals, opening each proc table once."""

        process = self._read_fields_mb(self.proc_root / "self/status", ("VmRSS",))
        system = self._read_fields_mb(self.proc_root / "meminfo", ("MemAvailable", "MemTotal"))
        return HostMemorySnapshot(
            rss_mb=process.get("VmRSS"),
            available_mb=system.get("MemAvailable"),
            total_mb=system.get("MemTotal"),
        )

    def log(self, label: str) -> HostMemorySnapshot:
        """Capture, log, and return the same measurement."""

        snapshot = self.capture()
        self.logger.info("host_memory[%s]: %s", label, snapshot)
        return snapshot

    @staticmethod
    def _read_fields_mb(path: Path, fields: tuple[str, ...]) -> dict[str, float]:
        """Read requested kB fields as MiB; unavailable fields stay absent."""

        values: dict[str, float] = {}
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    name, separator, raw = line.partition(":")
                    if not separator or name not in fields:
                        continue
                    parts = raw.split()
                    if parts:
                        values[name] = float(parts[0]) / 1024.0
        except OSError:
            return {}
        return values


__all__ = ["HostMemoryMonitor", "HostMemorySnapshot"]
