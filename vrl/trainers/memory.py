"""Host-memory instrumentation and colocated rollout guards."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from vrl.models.replay_loading import bundle_loads_full_generation_modules
from vrl.rollouts.runtime.config import RolloutBackendConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HostMemorySnapshot:
    """Current process RSS plus system memory totals in MiB."""

    rss_mb: float | None
    available_mb: float | None
    total_mb: float | None

    @property
    def used_fraction(self) -> float | None:
        if self.available_mb is None or self.total_mb in (None, 0):
            return None
        return 1.0 - (self.available_mb / self.total_mb)


def capture_host_memory() -> HostMemorySnapshot:
    """Capture Linux host memory without adding a psutil dependency."""

    return HostMemorySnapshot(
        rss_mb=_read_proc_status_mb("VmRSS"),
        available_mb=_read_meminfo_mb("MemAvailable"),
        total_mb=_read_meminfo_mb("MemTotal"),
    )


def log_host_memory(label: str, *, log: logging.Logger | None = None) -> HostMemorySnapshot:
    """Log a compact host-memory snapshot and return it for tests/hooks."""

    snapshot = capture_host_memory()
    target = log or logger
    target.info("host_memory[%s]: %s", label, format_host_memory(snapshot))
    return snapshot


def format_host_memory(snapshot: HostMemorySnapshot) -> str:
    """Format host-memory values with unknown fields omitted."""

    parts: list[str] = []
    if snapshot.rss_mb is not None:
        parts.append(f"rss={snapshot.rss_mb:.1f}MiB")
    if snapshot.available_mb is not None:
        parts.append(f"available={snapshot.available_mb:.1f}MiB")
    if snapshot.total_mb is not None:
        parts.append(f"total={snapshot.total_mb:.1f}MiB")
    used = snapshot.used_fraction
    if used is not None:
        parts.append(f"used={used:.3f}")
    return " ".join(parts) if parts else "unavailable"


def validate_colocated_replay_memory(
    *,
    bundle: Any,
    rollout_config: RolloutBackendConfig,
    strict: bool | None = None,
    log: logging.Logger | None = None,
) -> None:
    """Warn or fail when trainer and Ray worker both own full generation state.

    This guard does not implement a family-specific minimal replay loader. It
    makes the risk explicit and provides a strict mode for CI or future recipes
    once those loaders exist.
    """

    if not _is_colocated_gpu_rollout(rollout_config):
        return
    if not bundle_loads_full_generation_modules(bundle):
        return

    message = (
        "trainer bundle declares loads_full_generation_modules=true while "
        "colocated Ray rollout is enabled; host RAM can contain the trainer "
        "generation model plus a rollout worker generation model. Implement a "
        "family-specific minimal replay loader before enabling strict guard."
    )
    if strict is None:
        strict = _env_flag("VRL_STRICT_REPLAY_MEMORY_GUARD")
    if strict:
        raise ValueError(message)
    (log or logger).warning(message)


def _is_colocated_gpu_rollout(config: RolloutBackendConfig) -> bool:
    return bool(
        config.allow_driver_gpu_overlap
        and config.num_workers >= 1
        and config.gpus_per_worker > 0
    )


def _read_proc_status_mb(field: str) -> float | None:
    path = "/proc/self/status"
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


def _read_meminfo_mb(field: str) -> float | None:
    path = "/proc/meminfo"
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


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


__all__ = [
    "HostMemorySnapshot",
    "capture_host_memory",
    "format_host_memory",
    "log_host_memory",
    "validate_colocated_replay_memory",
]
