"""Repo-owned run supervisor: bounded restart with checkpoint resume.

Replaces the out-of-repo ``run-until-success`` shell guardian. The contract
differences that make this one trustworthy:

- **Explicit verdicts, not exit-code guessing.** ``vrl-train`` publishes
  ``run_verdict.json`` (success / failed+error class / terminated+signal);
  a missing verdict means the process died without unwinding (SIGKILL, OOM
  kill) and is treated as an infrastructure failure.
- **Process-group ownership.** The child runs in its own session; stop means
  SIGTERM to the whole group, a bounded grace period for the child's own
  cleanup (Ray teardown, checkpoint flush), then SIGKILL to the group. No
  orphaned Ray workers.
- **Checkpoint-verified resume.** Restarts resume from the latest COMPLETE
  checkpoint (atomic-publish + size-verified, see
  ``vrl.trainers.checkpointing``); a fresh start happens only when no
  trustworthy checkpoint exists.
- **Same-cause circuit breaker.** Consecutive failures with the same error
  class stop the loop instead of burning GPU re-hitting a deterministic bug.
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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RUN_VERDICT_NAME = "run_verdict.json"


@dataclass(frozen=True, slots=True)
class AttemptOutcome:
    """One child run's result, joined from exit code and verdict file."""

    exit_code: int
    verdict: dict[str, Any] | None

    @property
    def succeeded(self) -> bool:
        return (
            self.exit_code == 0
            and self.verdict is not None
            and self.verdict.get("verdict") == "success"
        )

    @property
    def failure_class(self) -> str:
        """Stable category key for the same-cause circuit breaker."""

        if self.verdict is None:
            # Died without unwinding (SIGKILL/OOM) — infrastructure-shaped.
            return f"no-verdict-exit-{self.exit_code}"
        kind = str(self.verdict.get("verdict"))
        if kind == "failed":
            return str(self.verdict.get("error_class", "unknown-error"))
        if kind == "terminated":
            return f"signal-{self.verdict.get('signal_name', self.verdict.get('signal'))}"
        return f"verdict-{kind}-exit-{self.exit_code}"


@dataclass
class RunSupervisor:
    """Restart a training command until success, a stop, or the breaker trips."""

    command: list[str]
    output_dir: Path
    max_attempts: int = 0  # 0 = unbounded; the circuit breaker still applies
    same_cause_limit: int = 2
    term_grace_seconds: float = 60.0
    backoff_seconds: float = 5.0
    sleep: Any = time.sleep  # injectable for tests

    _child: subprocess.Popen | None = field(default=None, init=False, repr=False)
    _stop_requested: bool = field(default=False, init=False, repr=False)

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
            if same_class_count >= self.same_cause_limit:
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
        verdict_path = self.output_dir / RUN_VERDICT_NAME
        verdict_path.unlink(missing_ok=True)
        # start_new_session puts the child in its own process group so stop
        # can signal every descendant (Ray drivers fork local raylets).
        self._child = subprocess.Popen(
            [*self.command, *extra_overrides],
            start_new_session=True,
        )
        try:
            exit_code = self._child.wait()
        finally:
            self._child = None
        return AttemptOutcome(exit_code=exit_code, verdict=self._read_verdict(verdict_path))

    def _read_verdict(self, path: Path) -> dict[str, Any] | None:
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        return raw if isinstance(raw, dict) else None

    def request_stop(self, signum: int = signal.SIGTERM) -> None:
        """Operator stop: signal the child's whole group, bounded escalation."""

        self._stop_requested = True
        child = self._child
        if child is None or child.poll() is not None:
            return
        group = child.pid  # start_new_session=True makes pid == pgid
        try:
            os.killpg(group, signum)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + self.term_grace_seconds
        while time.monotonic() < deadline:
            if child.poll() is not None:
                return
            time.sleep(0.2)
        logger.warning("child group %d survived grace period; SIGKILL", group)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(group, signal.SIGKILL)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Supervise a vrl-train run: restart on failure, resume from "
        "the latest complete checkpoint, stop on repeated same-cause failures.",
    )
    parser.add_argument("--config", required=True, help="Bundled config name or YAML path.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="Total attempt budget (0 = unbounded; the same-cause breaker still applies).",
    )
    parser.add_argument(
        "--same-cause-limit",
        type=int,
        default=2,
        help="Stop after this many CONSECUTIVE failures with the same error class.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides forwarded to vrl-train.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = build_parser().parse_args(argv)

    from vrl.config.loading import load_config
    from vrl.utils.config import cfg_path

    cfg = load_config(args.config, overrides=args.overrides)
    output_dir = str(cfg_path(cfg, "trainer.output_dir", "") or "").strip()
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
        same_cause_limit=args.same_cause_limit,
    )

    def _forward_stop(signum: int, frame: Any) -> None:
        del frame
        supervisor.request_stop(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _forward_stop)

    raise SystemExit(supervisor.run())


if __name__ == "__main__":
    main()
