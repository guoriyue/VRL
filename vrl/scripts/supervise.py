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
- **Optional metrics health gate.** When explicitly enabled, consecutive new
  non-finite or out-of-bound training rows stop the child group without a
  restart and publish a machine-readable health verdict.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from vrl.trainers.metrics_io import online_metric_columns

logger = logging.getLogger(__name__)

RUN_VERDICT_NAME = "run_verdict.json"
HEALTH_VERDICT_NAME = "health_verdict.json"
_REQUIRED_HEALTH_METRICS = (
    "loss",
    "reward_mean",
    "reward_std",
    "grad_norm",
    "pre_update_logprob_abs_diff_max",
)
_CONTINUOUS_HEALTH_METRICS = (
    "continuous_stale_versions",
    "continuous_producer_errors",
)


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


@dataclass(frozen=True, slots=True)
class HealthGateConfig:
    """Thresholds for the opt-in metrics health gate.

    Presence of this config is what enables the gate (``RunSupervisor.health``),
    so "enabled but unconfigured" and "thresholds set but disabled" cannot be
    expressed. ``continuous`` is derived from the resolved training schedule;
    an operational stale-policy threshold cannot silently opt a strict run into
    a different metrics protocol.
    """

    poll_seconds: float = 5.0
    failure_limit: int = 3
    max_pre_update_logprob_diff: float = 0.01
    continuous: bool = False
    max_stale_policy_versions: int | None = None
    max_stale_logprob_diff: float = 0.05
    min_reward_std: float = 1e-4
    min_grad_norm: float = 1e-8

    def __post_init__(self) -> None:
        required_columns = set(_REQUIRED_HEALTH_METRICS)
        if self.continuous:
            required_columns.update(_CONTINUOUS_HEALTH_METRICS)
        missing_columns = sorted(required_columns - set(online_metric_columns()))
        if missing_columns:
            raise AssertionError(
                "health metrics are absent from the online CSV protocol: "
                + ", ".join(missing_columns),
            )
        if self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be > 0")
        if self.failure_limit < 1:
            raise ValueError("failure_limit must be >= 1")
        if self.max_stale_policy_versions is not None and self.max_stale_policy_versions < 0:
            raise ValueError("max_stale_policy_versions must be >= 0")
        if self.continuous and self.max_stale_policy_versions is None:
            raise ValueError(
                "continuous health checks require max_stale_policy_versions",
            )
        if not self.continuous and self.max_stale_policy_versions is not None:
            raise ValueError(
                "max_stale_policy_versions requires a continuous training schedule",
            )
        for name in (
            "max_pre_update_logprob_diff",
            "max_stale_logprob_diff",
            "min_reward_std",
            "min_grad_norm",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and >= 0")


class MetricsHealthGate:
    """Judge new metrics.csv rows and hold the gate's verdict state.

    Owns the row baseline, the consecutive-unhealthy streak, and the
    per-attempt producer-error watermark. Process control stays with
    ``RunSupervisor``: the gate never signals the child; it reads the metrics
    file, reports whether it tripped, and publishes ``health_verdict.json``.
    """

    def __init__(self, config: HealthGateConfig, output_dir: Path) -> None:
        self.config = config
        self.output_dir = output_dir
        self.tripped = False
        self._unhealthy_rows = 0
        self._producer_errors = 0
        self._seen_rows: list[tuple[tuple[str, str | None], ...]] = []

    def baseline_existing_rows(self) -> None:
        """Treat rows already on disk as history, not judgments to make."""

        (self.output_dir / HEALTH_VERDICT_NAME).unlink(missing_ok=True)
        self._seen_rows = self._read_complete_rows()[1]

    def start_attempt(self) -> None:
        # The producer counter and row streak belong to one child lifecycle.
        # Historical rows are only a file-observation baseline; carrying their
        # counter forward would hide the restarted producer's first new error.
        self._producer_errors = 0
        self._unhealthy_rows = 0

    def judge_new_rows(self) -> bool:
        """Judge rows appended since the last call; True the moment we trip."""

        rows, signatures = self._read_complete_rows()
        common_prefix = 0
        for previous, current in zip(self._seen_rows, signatures, strict=False):
            if previous != current:
                break
            common_prefix += 1

        # A fresh attempt may truncate metrics.csv before writing a new header.
        # Comparing row signatures, instead of remembering only a row count,
        # makes the replacement visible without re-judging an unchanged prefix
        # (an intact file keeps common_prefix == len(seen), so this slice is
        # exactly the appended rows).
        new_rows = rows[common_prefix:]
        self._seen_rows = signatures

        for row in new_rows:
            reasons, parsed = self._metric_health_reasons(row)
            reasons.extend(self._track_producer_errors(parsed))
            if not reasons:
                self._unhealthy_rows = 0
                continue
            self._unhealthy_rows += 1
            logger.warning(
                "unhealthy metrics row epoch=%s (%d/%d): %s",
                row.get("epoch"),
                self._unhealthy_rows,
                self.config.failure_limit,
                "; ".join(reasons),
            )
            if self._unhealthy_rows >= self.config.failure_limit:
                self.tripped = True
                self._write_verdict(row, parsed, reasons)
                return True
        return False

    def _track_producer_errors(self, parsed: dict[str, float]) -> list[str]:
        """Update the per-attempt producer-error watermark; report any change."""

        value = parsed.get("continuous_producer_errors")
        if value is None or value < 0 or not value.is_integer():
            return []
        current = int(value)
        previous = self._producer_errors
        self._producer_errors = current
        if current == previous:
            return []
        if current < previous:
            return [
                f"continuous_producer_errors decreased unexpectedly from {previous} to {current}",
            ]
        return [
            f"continuous_producer_errors increased from {previous} to {current}",
        ]

    def _read_complete_rows(
        self,
    ) -> tuple[list[dict[str, str | None]], list[tuple[tuple[str, str | None], ...]]]:
        try:
            text = (self.output_dir / "metrics.csv").read_text(encoding="utf-8")
        except OSError:
            return [], []
        if not text.endswith("\n"):
            text = text.rpartition("\n")[0]
        lines = text.splitlines()
        if len(lines) < 2:
            return [], []
        rows = list(csv.DictReader(lines))
        signatures = [tuple(row.items()) for row in rows]
        return rows, signatures

    def _metric_health_reasons(
        self,
        row: dict[str, str | None],
    ) -> tuple[list[str], dict[str, float]]:
        reasons: list[str] = []
        parsed: dict[str, float] = {}
        required_metrics = list(_REQUIRED_HEALTH_METRICS)
        if self.config.continuous:
            required_metrics.extend(_CONTINUOUS_HEALTH_METRICS)
        for name in required_metrics:
            raw = row.get(name)
            if raw is None or not raw.strip():
                reasons.append(f"metric {name} is missing")
                continue
            try:
                value = float(raw)
            except ValueError:
                reasons.append(f"metric {name} is not numeric: {raw!r}")
                continue
            if not math.isfinite(value):
                reasons.append(f"metric {name} is not finite: {raw}")
                continue
            parsed[name] = value

        stale_versions = parsed.get("continuous_stale_versions")
        producer_errors = parsed.get("continuous_producer_errors")
        if stale_versions is not None:
            if stale_versions < 0 or not stale_versions.is_integer():
                reasons.append(
                    "continuous_stale_versions must be a non-negative integer "
                    f"(got {stale_versions:g})",
                )
            elif (
                self.config.max_stale_policy_versions is not None
                and stale_versions > self.config.max_stale_policy_versions
            ):
                reasons.append(
                    "continuous_stale_versions "
                    f"{stale_versions:g} exceeds maximum "
                    f"{self.config.max_stale_policy_versions}",
                )
        if producer_errors is not None and (
            producer_errors < 0 or not producer_errors.is_integer()
        ):
            reasons.append(
                "continuous_producer_errors must be a non-negative integer "
                f"(got {producer_errors:g})",
            )

        parity = parsed.get("pre_update_logprob_abs_diff_max")
        parity_limit = self.config.max_pre_update_logprob_diff
        if stale_versions is not None and stale_versions > 0:
            # With bounded async rollout this metric includes intentional policy
            # drift, not only rollout/replay numerical parity. Keep a separate,
            # looser importance-ratio safety bound instead of misclassifying every
            # valid stale row as a parity failure.
            parity_limit = self.config.max_stale_logprob_diff
        if parity is not None and parity > parity_limit:
            reasons.append(
                f"pre_update_logprob_abs_diff_max {parity:g} exceeds maximum {parity_limit:g}",
            )
        reward_std = parsed.get("reward_std")
        if reward_std is not None and reward_std < self.config.min_reward_std:
            reasons.append(
                f"reward_std {reward_std:g} is below minimum {self.config.min_reward_std:g}",
            )
        grad_norm = parsed.get("grad_norm")
        if grad_norm is not None and grad_norm < self.config.min_grad_norm:
            reasons.append(
                f"grad_norm {grad_norm:g} is below minimum {self.config.min_grad_norm:g}",
            )
        return reasons, parsed

    def _write_verdict(
        self,
        row: dict[str, str | None],
        parsed: dict[str, float],
        reasons: list[str],
    ) -> None:
        epoch_raw = row.get("epoch")
        try:
            epoch_number = float(epoch_raw) if epoch_raw is not None else None
            epoch: int | float | str | None = (
                int(epoch_number)
                if epoch_number is not None and epoch_number.is_integer()
                else epoch_number
            )
        except ValueError:
            epoch = epoch_raw
        verdict = {
            "verdict": "failed",
            "source": "metrics_health_gate",
            "epoch": epoch,
            "consecutive_unhealthy_rows": self._unhealthy_rows,
            "failure_limit": self.config.failure_limit,
            "reasons": reasons,
            "metrics": parsed,
            "thresholds": {
                "max_pre_update_logprob_diff": self.config.max_pre_update_logprob_diff,
                "continuous": self.config.continuous,
                "max_stale_policy_versions": self.config.max_stale_policy_versions,
                "max_stale_logprob_diff": self.config.max_stale_logprob_diff,
                "min_reward_std": self.config.min_reward_std,
                "min_grad_norm": self.config.min_grad_norm,
            },
        }
        path = self.output_dir / HEALTH_VERDICT_NAME
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)


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
    health: HealthGateConfig | None = None  # presence enables the metrics gate

    _child: subprocess.Popen | None = field(default=None, init=False, repr=False)
    _stop_requested: bool = field(default=False, init=False, repr=False)
    _health_gate: MetricsHealthGate | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_attempts < 0:
            raise ValueError("max_attempts must be >= 0")
        if self.same_cause_limit < 1:
            raise ValueError("same_cause_limit must be >= 1")
        for name in ("term_grace_seconds", "backoff_seconds"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0")
        if self.health is not None:
            self._health_gate = MetricsHealthGate(self.health, self.output_dir)

    def run(self) -> int:
        if self._health_gate is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._health_gate.baseline_existing_rows()

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
            if self._health_gate is not None and self._health_gate.tripped:
                logger.error("metrics health gate tripped; not restarting")
                return outcome.exit_code or 1
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
        if self._health_gate is not None:
            self._health_gate.start_attempt()
        # start_new_session puts the child in its own process group so stop
        # can signal every descendant (Ray drivers fork local raylets).
        self._child = subprocess.Popen(
            [*self.command, *extra_overrides],
            start_new_session=True,
        )
        try:
            if self._health_gate is None:
                exit_code = self._child.wait()
            else:
                exit_code = self._wait_with_health_checks(self._child, self._health_gate)
        finally:
            self._child = None
        return AttemptOutcome(exit_code=exit_code, verdict=self._read_verdict(verdict_path))

    def _wait_with_health_checks(
        self,
        child: subprocess.Popen,
        gate: MetricsHealthGate,
    ) -> int:
        health_stop_deadline: float | None = None
        while True:
            try:
                exit_code = child.wait(timeout=gate.config.poll_seconds)
            except subprocess.TimeoutExpired:
                if not gate.tripped and gate.judge_new_rows():
                    self._signal_child_group(child, signal.SIGTERM)
                    health_stop_deadline = time.monotonic() + self.term_grace_seconds
                elif (
                    gate.tripped
                    and health_stop_deadline is not None
                    and time.monotonic() >= health_stop_deadline
                ):
                    logger.warning(
                        "child group %d survived health-gate grace period; SIGKILL",
                        child.pid,
                    )
                    self._signal_child_group(child, signal.SIGKILL)
                    health_stop_deadline = None
                continue

            # A short child may publish a final row and exit between polling
            # intervals. Judge that row before accepting its success verdict.
            if not gate.tripped:
                gate.judge_new_rows()
            return exit_code

    @staticmethod
    def _signal_child_group(child: subprocess.Popen, signum: int) -> bool:
        if child.poll() is not None:
            return False
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            return False
        return True

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
        if not self._signal_child_group(child, signum):
            return
        deadline = time.monotonic() + self.term_grace_seconds
        while time.monotonic() < deadline:
            if child.poll() is not None:
                return
            time.sleep(0.2)
        logger.warning("child group %d survived grace period; SIGKILL", group)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(group, signal.SIGKILL)


def _bounded_number(cast: type, minimum: float, *, exclusive: bool = False) -> Any:
    """argparse ``type=`` callable enforcing a finite lower-bounded number."""

    def parse(raw: str) -> Any:
        value = cast(raw)
        out_of_range = value <= minimum if exclusive else value < minimum
        if not math.isfinite(value) or out_of_range:
            bound = f"{'>' if exclusive else '>='} {minimum:g}"
            raise argparse.ArgumentTypeError(f"must be finite and {bound}")
        return value

    return parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Supervise a vrl-train run: restart on failure, resume from "
        "the latest complete checkpoint, stop on repeated same-cause failures.",
    )
    parser.add_argument("--config", required=True, help="Bundled config name or YAML path.")
    supervisor_defaults = {item.name: item.default for item in fields(RunSupervisor)}
    parser.add_argument(
        "--max-attempts",
        type=_bounded_number(int, 0),
        default=supervisor_defaults["max_attempts"],
        help="Total attempt budget (0 = unbounded; the same-cause breaker still applies).",
    )
    parser.add_argument(
        "--same-cause-limit",
        type=_bounded_number(int, 1),
        default=supervisor_defaults["same_cause_limit"],
        help="Stop after this many CONSECUTIVE failures with the same error class.",
    )
    parser.add_argument(
        "--health-metrics",
        action="store_true",
        help="Stop without restarting when consecutive new metrics.csv rows are unhealthy.",
    )
    # Threshold defaults live on HealthGateConfig; the CLI only mirrors them so
    # the two layers cannot drift apart.
    health_defaults = HealthGateConfig()
    parser.add_argument(
        "--health-poll-seconds",
        type=_bounded_number(float, 0, exclusive=True),
        default=health_defaults.poll_seconds,
        help="Seconds between metrics.csv health checks "
        f"(default: {health_defaults.poll_seconds:g}).",
    )
    parser.add_argument(
        "--health-failure-limit",
        type=_bounded_number(int, 1),
        default=health_defaults.failure_limit,
        help="Stop after this many consecutive unhealthy rows "
        f"(default: {health_defaults.failure_limit}).",
    )
    parser.add_argument(
        "--health-max-pre-update-logprob-diff",
        type=_bounded_number(float, 0),
        default=None,
        help="Maximum healthy pre-update log-probability difference. When omitted, "
        "derive trainer.debug.max_abs_logprob_diff from the resolved training config.",
    )
    parser.add_argument(
        "--health-max-stale-policy-versions",
        type=_bounded_number(int, 0),
        default=health_defaults.max_stale_policy_versions,
        help="Operational stale-version limit for a continuous training schedule. "
        "It may be stricter, but not wider, than the training schedule.",
    )
    parser.add_argument(
        "--health-max-stale-logprob-diff",
        type=_bounded_number(float, 0),
        default=health_defaults.max_stale_logprob_diff,
        help="Maximum healthy log-probability drift for stale rollouts "
        f"(default: {health_defaults.max_stale_logprob_diff:g}).",
    )
    parser.add_argument(
        "--health-min-reward-std",
        type=_bounded_number(float, 0),
        default=health_defaults.min_reward_std,
        help="Minimum healthy reward standard deviation "
        f"(default: {health_defaults.min_reward_std:g}).",
    )
    parser.add_argument(
        "--health-min-grad-norm",
        type=_bounded_number(float, 0),
        default=health_defaults.min_grad_norm,
        help=f"Minimum healthy gradient norm (default: {health_defaults.min_grad_norm:g}).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides forwarded to vrl-train.",
    )
    return parser


def _build_health_gate_config(
    args: argparse.Namespace,
    cfg: Any,
) -> HealthGateConfig | None:
    """Project the resolved training schedule into the supervisor health contract."""

    if not args.health_metrics:
        return None

    from vrl.rollouts.orchestration.types import RolloutScheduleMode
    from vrl.trainers.core.types import RolloutOrchestrationConfig
    from vrl.utils.config import cfg_path

    schedule_mode = RolloutScheduleMode(
        str(
            cfg_path(
                cfg,
                "trainer.rollout_orchestration.schedule_mode",
                RolloutOrchestrationConfig().schedule_mode,
            ),
        ),
    )
    continuous = schedule_mode is RolloutScheduleMode.CONTINUOUS
    stale_limit = args.health_max_stale_policy_versions
    if continuous:
        training_stale_limit = cfg_path(
            cfg,
            "trainer.rollout_orchestration.continuous.max_stale_policy_versions",
            None,
        )
        if training_stale_limit is None:
            raise ValueError(
                "continuous training config is missing "
                "trainer.rollout_orchestration.continuous.max_stale_policy_versions",
            )
        training_stale_limit = int(training_stale_limit)
        if stale_limit is None:
            stale_limit = training_stale_limit
        if stale_limit > training_stale_limit:
            raise ValueError(
                "health max_stale_policy_versions cannot exceed the training "
                f"schedule limit ({stale_limit} > {training_stale_limit})",
            )
    elif stale_limit is not None:
        raise ValueError(
            "--health-max-stale-policy-versions requires "
            "trainer.rollout_orchestration.schedule_mode=continuous",
        )

    parity_limit = args.health_max_pre_update_logprob_diff
    if parity_limit is None:
        parity_limit = float(
            cfg_path(
                cfg,
                "trainer.debug.max_abs_logprob_diff",
                HealthGateConfig().max_pre_update_logprob_diff,
            ),
        )

    return HealthGateConfig(
        poll_seconds=args.health_poll_seconds,
        failure_limit=args.health_failure_limit,
        max_pre_update_logprob_diff=parity_limit,
        continuous=continuous,
        max_stale_policy_versions=stale_limit,
        max_stale_logprob_diff=args.health_max_stale_logprob_diff,
        min_reward_std=args.health_min_reward_std,
        min_grad_norm=args.health_min_grad_norm,
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    from vrl.config.loading import load_config
    from vrl.utils.config import cfg_path

    cfg = load_config(args.config, overrides=args.overrides)
    output_dir = str(cfg_path(cfg, "trainer.output_dir", "") or "").strip()
    if not output_dir:
        raise SystemExit("supervise requires trainer.output_dir in the resolved config")

    try:
        health = _build_health_gate_config(args, cfg)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
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
        health=health,
    )

    def _forward_stop(signum: int, frame: Any) -> None:
        del frame
        supervisor.request_stop(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _forward_stop)

    raise SystemExit(supervisor.run())


if __name__ == "__main__":
    main()
