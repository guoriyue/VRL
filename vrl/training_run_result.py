"""Per-process training outcomes shared by launchers and supervisors."""

from __future__ import annotations

import logging
import os
import signal
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from vrl.utils.json_files import write_json

if TYPE_CHECKING:
    from vrl.config.schema import RootConfig

TRAINING_RUN_RESULT_NAME = "training_run_result.json"


class TrainingRunResultWriter:
    """Own the destination and rank identity for one process's outcome records."""

    def __init__(
        self, output_dir: str | Path | None, *, environ: Mapping[str, str] | None = None
    ) -> None:
        self.output_dir = None if output_dir is None else Path(output_dir)
        environment = os.environ if environ is None else environ
        self.file_name, self.rank, self.world_size = self._resolve_file(environment)

    @classmethod
    def from_root(cls, root: RootConfig) -> TrainingRunResultWriter:
        output_dir = str(
            (root.trainer.output_dir if root.trainer is not None else None) or ""
        ).strip()
        return cls(output_dir or None)

    @staticmethod
    def rank_file_name(rank: int) -> str:
        """Return the per-rank result file name owned by one torchrun worker."""

        if rank < 0:
            raise ValueError(f"rank must be non-negative, got {rank}")
        return f"training_run_result.rank-{rank}.json"

    @classmethod
    def _resolve_file(
        cls,
        environ: Mapping[str, str],
    ) -> tuple[str, int | None, int | None]:
        """Pick this process's result file name: per-rank under torchrun, else the aggregate.

        Malformed ``RANK``/``WORLD_SIZE`` fall back to the single-process name rather
        than raising, so a broken environment never masks the trainer's own failure.
        """

        rank_raw = environ.get("RANK")
        world_size_raw = environ.get("WORLD_SIZE")
        try:
            rank = int(rank_raw) if rank_raw is not None else None
            world_size = int(world_size_raw) if world_size_raw is not None else None
        except ValueError:
            return TRAINING_RUN_RESULT_NAME, None, None
        if rank is None or world_size is None or world_size <= 1 or rank < 0 or rank >= world_size:
            return TRAINING_RUN_RESULT_NAME, None, None
        return cls.rank_file_name(rank), rank, world_size

    def write(
        self,
        *,
        error: BaseException | None = None,
        received_signal: signal.Signals | None = None,
    ) -> None:
        """Publish this run's outcome as an explicit machine-readable contract.

        A supervisor must never guess the failure cause from the exit code alone:
        the result names the error class so restart policy ("same class twice ->
        stop") is decided on facts. A missing result file (SIGKILL, OOM-killed
        interpreter) is itself informative: the run died without unwinding. Every
        multi-rank worker owns a separate file; the supervisor publishes the aggregate
        only after torchrun has joined all workers.
        """

        if self.output_dir is None:
            return
        if error is not None:
            from vrl.runtime_errors import root_failure_cause

            root_error = root_failure_cause(error)
            result = {
                "status": "failed",
                # Cleanup wrappers keep the complete outer message, while restart
                # policy keys on the stable operation root that actually failed.
                "error_class": type(root_error).__name__,
                "error_message": str(error)[:2000],
            }
        elif received_signal is not None:
            result = {
                "status": "terminated",
                "signal": int(received_signal),
                "signal_name": received_signal.name,
            }
        else:
            result = {"status": "success"}
        result["schema_version"] = 1
        if self.rank is not None and self.world_size is not None:
            result["rank"] = self.rank
            result["world_size"] = self.world_size
        try:
            write_json(self.output_dir / self.file_name, result)
        except OSError:
            logging.getLogger(__name__).warning(
                "failed to write run result to %s",
                self.output_dir,
                exc_info=True,
            )
