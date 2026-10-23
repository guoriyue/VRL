"""Repo-owned run supervisor: bounded restart with checkpoint resume.

Replaces the out-of-repo ``run-until-success`` shell guardian. The contract
differences that make this one trustworthy:

- **Explicit results, not exit-code guessing.** ``vrl-train`` publishes
  ``training_run_result.json``. A missing result means the child died without
  unwinding (SIGKILL, OOM kill) and is treated as an infrastructure failure.
- **Process-group ownership.** The child runs in its own session; stop means
  SIGTERM to the whole group, a bounded grace period for the child's own
  cleanup (Ray teardown, checkpoint flush), then SIGKILL to the group. No
  orphaned workers.
- **Checkpoint-verified resume.** Restarts resume from the latest COMPLETE
  checkpoint (atomic-publish + size-verified, see
  ``vrl.trainers.checkpointing``); a fresh start happens only when no
  trustworthy checkpoint exists.
- **Same-cause circuit breaker.** Two consecutive failures with the same error
  class stop the loop instead of burning GPU re-hitting a deterministic bug.

The child is one ``vrl-train`` process. A DDP/FSDP recipe needs torchrun's
rank environment, which this supervisor does not provide.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vrl.training_run_result import TRAINING_RUN_RESULT_NAME

logger = logging.getLogger(__name__)

# Consecutive failures with one error class that stop the restart loop.
_SAME_CAUSE_LIMIT = 2


@dataclass(frozen=True, slots=True)
class AttemptOutcome:
    """One child run's result, joined from exit code and result file."""

    exit_code: int
    result: dict[str, Any] | None

    @property
    def succeeded(self) -> bool:
        return (
            self.exit_code == 0
            and self.result is not None
            and self.result.get("status") == "success"
        )

    @property
    def failure_class(self) -> str:
        """Stable category key for the same-cause circuit breaker."""

        if self.result is None:
            # Died without unwinding (SIGKILL/OOM) — infrastructure-shaped.
            return f"no-result-exit-{self.exit_code}"
        kind = str(self.result.get("status"))
        if kind == "failed":
            return str(self.result.get("error_class", "unknown-error"))
        if kind == "terminated":
            return f"signal-{self.result.get('signal_name', self.result.get('signal'))}"
        return f"result-{kind}-exit-{self.exit_code}"


@dataclass
class RunSupervisor:
    """Restart a training command until success, a stop, or the breaker trips."""

    command: list[str]
    output_dir: Path
    max_attempts: int = 0  # 0 = unbounded; the circuit breaker still applies
    term_grace_seconds: float = 60.0
    backoff_seconds: float = 5.0
    sleep: Any = time.sleep  # injectable for tests

    _child: subprocess.Popen | None = field(default=None, init=False, repr=False)
    _stop_requested: bool = field(default=False, init=False, repr=False)
    _group_cleanup_lock: Any = field(default_factory=threading.RLock, init=False, repr=False)

    def run(self) -> int:
        attempts = 0
        last_class: str | None = None
        same_class_count = 0
        while True:
            attempts += 1
            resume = self._resume_override()
            outcome = self._run_attempt(resume)
            if self._stop_requested:
                logger.info("supervisor stopped by operator signal; not restarting")
                return outcome.exit_code
            if outcome.succeeded:
                logger.info("run succeeded after %d attempt(s)", attempts)
                return 0
            cause = outcome.failure_class
            same_class_count = same_class_count + 1 if cause == last_class else 1
            last_class = cause
            logger.warning(
                "attempt %d failed (exit=%d, cause=%s, consecutive same-cause=%d)",
                attempts,
                outcome.exit_code,
                cause,
                same_class_count,
            )
            if same_class_count >= _SAME_CAUSE_LIMIT:
                logger.error(
                    "circuit breaker: %d consecutive failures with cause %s; stopping",
                    same_class_count,
                    cause,
                )
                return outcome.exit_code or 1
            if self.max_attempts and attempts >= self.max_attempts:
                logger.error("attempt budget exhausted (%d); stopping", attempts)
                return outcome.exit_code or 1
            self.sleep(self.backoff_seconds)

    def _resume_override(self) -> list[str]:
        """Dotlist overrides resuming from the latest complete checkpoint.

        Also clears ``model.lora.path``: a warm-start adapter is only for the
        FIRST attempt; once a checkpoint exists it is the resume source of
        truth and the recipe rejects the combination.
        """

        from vrl.trainers.checkpointing import find_latest_complete_checkpoint

        latest = find_latest_complete_checkpoint(self.output_dir)
        if latest is None:
            return []
        logger.info("resuming from latest complete checkpoint: %s", latest)
        return [f"trainer.resume_from={latest}", "model.lora.path="]

    def _run_attempt(self, extra_overrides: list[str]) -> AttemptOutcome:
        result_path = self.output_dir / TRAINING_RUN_RESULT_NAME
        # A result left by the previous attempt must not stand in for this one.
        result_path.unlink(missing_ok=True)
        # start_new_session puts the child in its own process group so stop
        # can signal every descendant (Ray drivers fork local raylets).
        self._child = subprocess.Popen(
            [*self.command, *extra_overrides],
            start_new_session=True,
        )
        try:
            exit_code = self._child.wait()
        except BaseException as error:
            try:
                self.request_stop()
                self._child.wait()
            except BaseException as cleanup_error:
                error.add_note(f"supervisor child cleanup also failed: {cleanup_error!r}")
            raise
        finally:
            if self._child.poll() is not None:
                # The leader's exit does not prove its workers stopped. Drain
                # the owned session before another attempt can allocate a GPU.
                self._drain_child_group(self._child)
                self._child = None
        try:
            result = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            result = None
        return AttemptOutcome(
            exit_code=exit_code,
            result=result if isinstance(result, dict) else None,
        )

    def request_stop(self, signum: int = signal.SIGTERM) -> None:
        """Operator stop: signal the child's whole group, bounded escalation."""

        self._stop_requested = True
        child = self._child
        if child is None:
            return
        self._drain_child_group(child, signum=signum)

    @staticmethod
    def _group_has_live_members(group: int) -> bool:
        """Ignore exited zombies while checking the process group we created."""
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            return False
        proc = Path("/proc")
        if not proc.is_dir():
            # On systems without procfs, retain the conservative killpg result.
            return True
        for entry in proc.iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                # comm can contain spaces and closing parentheses. The fields
                # after its final ')' start with state, ppid, and process group.
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                if fields[0] != "Z" and int(fields[2]) == group:
                    return True
            except (FileNotFoundError, ProcessLookupError):
                continue
            except (PermissionError, IndexError, ValueError):
                # Never declare cleanup complete when membership is unknown.
                try:
                    if os.getpgid(int(entry.name)) == group:
                        return True
                except ProcessLookupError:
                    continue
        return False

    def _drain_child_group(self, child: subprocess.Popen, *, signum: int = signal.SIGTERM) -> None:
        """Bound cleanup independently of whether the session leader has exited."""
        group = child.pid  # start_new_session=True establishes this ownership.
        # A signal handler may reenter on this thread; tests also stop from a
        # second thread. Keep one cleanup owner without deadlocking that handler.
        with self._group_cleanup_lock:
            if not self._group_has_live_members(group):
                return
            with contextlib.suppress(ProcessLookupError):
                os.killpg(group, signum)
            deadline = time.monotonic() + self.term_grace_seconds
            while self._group_has_live_members(group):
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
            else:
                return
            logger.warning("child group %d survived grace period; SIGKILL", group)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(group, signal.SIGKILL)
            kill_deadline = time.monotonic() + 5.0
            while self._group_has_live_members(group):
                if time.monotonic() >= kill_deadline:
                    raise RuntimeError(
                        f"owned child group {group} still has live workers after SIGKILL"
                    )
                time.sleep(0.05)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Supervise a vrl-train run: restart on failure, resume from "
        "the latest complete checkpoint, stop after two consecutive same-cause failures.",
    )
    parser.add_argument("--config", required=True, help="Bundled config name or YAML path.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="Total attempt budget (0 = unbounded; the same-cause breaker still applies).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Ordered +group=option presets and dotlist overrides forwarded to vrl-train.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = build_parser().parse_args(argv)

    from vrl.config.loading import load_config
    from vrl.config.schema import parse_config

    root = parse_config(load_config(args.config, overrides=args.overrides))
    output_dir = str((root.trainer.output_dir if root.trainer is not None else None) or "").strip()
    if not output_dir:
        raise SystemExit("supervise requires trainer.output_dir in the resolved config")

    supervisor = RunSupervisor(
        command=[
            sys.executable,
            "-m",
            "vrl.scripts.train",
            "--config",
            args.config,
            *args.overrides,
        ],
        output_dir=Path(output_dir),
        max_attempts=args.max_attempts,
    )

    def _forward_stop(signum: int, frame: Any) -> None:
        del frame
        supervisor.request_stop(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _forward_stop)

    raise SystemExit(supervisor.run())


if __name__ == "__main__":
    main()
