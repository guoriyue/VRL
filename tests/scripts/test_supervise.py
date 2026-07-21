"""Run supervisor tests — real subprocesses, real process groups, no fakes."""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import pytest

from vrl.scripts.supervise import (
    HEALTH_VERDICT_NAME,
    HealthGateConfig,
    MetricsHealthGate,
    RunSupervisor,
    build_parser,
)

_METRICS_HEADER = "epoch,loss,reward_mean,reward_std,grad_norm,pre_update_logprob_abs_diff_max\n"


def _metric_row(
    epoch: int,
    *,
    loss: str = "1.0",
    reward_mean: str = "2.0",
    reward_std: str = "0.2",
    grad_norm: str = "0.1",
    parity: str = "0.001",
) -> str:
    return f"{epoch},{loss},{reward_mean},{reward_std},{grad_norm},{parity}\n"


def _continuous_metric_row(
    epoch: int,
    *,
    parity: str = "0.02",
    stale_versions: str = "1",
    producer_errors: str = "0",
) -> dict[str, str]:
    return {
        "epoch": str(epoch),
        "loss": "1.0",
        "reward_mean": "2.0",
        "reward_std": "0.2",
        "grad_norm": "0.1",
        "pre_update_logprob_abs_diff_max": parity,
        "continuous_stale_versions": stale_versions,
        "continuous_producer_errors": producer_errors,
    }


def _continuous_metric_csv_row(
    epoch: int,
    *,
    producer_errors: str = "0",
) -> str:
    row = _continuous_metric_row(epoch, producer_errors=producer_errors)
    return ",".join(row.values()) + "\n"


def _write_continuous_metrics(
    out: Path,
    rows: list[dict[str, str]],
) -> None:
    header = ",".join(rows[0]) + "\n"
    body = "".join(",".join(row.values()) + "\n" for row in rows)
    (out / "metrics.csv").write_text(header + body)


def _child_script(tmp_path: Path, body: str) -> list[str]:
    script = tmp_path / "child.py"
    script.write_text(body)
    return [sys.executable, str(script)]


def _write_verdict_snippet(out_dir: Path) -> str:
    return (
        "import json, pathlib\n"
        f"out = pathlib.Path({str(out_dir)!r})\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
    )


def test_success_first_attempt(tmp_path) -> None:
    out = tmp_path / "run"
    command = _child_script(
        tmp_path,
        _write_verdict_snippet(out)
        + "(out / 'run_verdict.json').write_text(json.dumps({'verdict': 'success'}))\n",
    )
    supervisor = RunSupervisor(command=command, output_dir=out, sleep=lambda _: None)
    assert supervisor.run() == 0


def test_same_cause_circuit_breaker_stops(tmp_path) -> None:
    out = tmp_path / "run"
    attempts_file = tmp_path / "attempts"
    command = _child_script(
        tmp_path,
        _write_verdict_snippet(out)
        + f"attempts = pathlib.Path({str(attempts_file)!r})\n"
        + "n = int(attempts.read_text()) + 1 if attempts.exists() else 1\n"
        + "attempts.write_text(str(n))\n"
        + "(out / 'run_verdict.json').write_text(json.dumps("
        "{'verdict': 'failed', 'error_class': 'ValueError'}))\n" + "raise SystemExit(1)\n",
    )
    supervisor = RunSupervisor(
        command=command,
        output_dir=out,
        same_cause_limit=2,
        sleep=lambda _: None,
    )
    assert supervisor.run() == 1
    assert attempts_file.read_text() == "2"  # stopped at the breaker, not later


def test_transient_failure_then_success_restarts(tmp_path) -> None:
    out = tmp_path / "run"
    marker = tmp_path / "failed_once"
    command = _child_script(
        tmp_path,
        _write_verdict_snippet(out)
        + f"marker = pathlib.Path({str(marker)!r})\n"
        + "if not marker.exists():\n"
        + "    marker.write_text('1')\n"
        + "    (out / 'run_verdict.json').write_text(json.dumps("
        "{'verdict': 'terminated', 'signal': 15, 'signal_name': 'SIGTERM'}))\n"
        + "    raise SystemExit(143)\n"
        + "(out / 'run_verdict.json').write_text(json.dumps({'verdict': 'success'}))\n",
    )
    supervisor = RunSupervisor(command=command, output_dir=out, sleep=lambda _: None)
    assert supervisor.run() == 0


def test_missing_verdict_is_a_distinct_failure_class(tmp_path) -> None:
    out = tmp_path / "run"
    # Child dies without unwinding: no verdict file is ever written.
    command = _child_script(tmp_path, "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n")
    supervisor = RunSupervisor(
        command=command,
        output_dir=out,
        same_cause_limit=2,
        sleep=lambda _: None,
    )
    exit_code = supervisor.run()
    assert exit_code != 0  # breaker tripped on consecutive no-verdict deaths


def test_restart_resumes_from_latest_complete_checkpoint(tmp_path) -> None:
    import torch.nn as nn

    from vrl.trainers.checkpointing import save_training_checkpoint

    out = tmp_path / "run"
    out.mkdir(parents=True)

    class _Trainer:
        def state_dict(self):
            return {"step": 4, "global_step": 4}

    class _Bundle:
        def __init__(self) -> None:
            self.trainable_modules = {"module": nn.Linear(1, 1, bias=False)}

    save_training_checkpoint(
        out / "checkpoint-4",
        trainer=_Trainer(),
        bundle=_Bundle(),
        family="unit",
        progress={"next_epoch": 4, "global_step": 4},
        rng_state={},
    )

    attempts_file = tmp_path / "attempts"
    argv_log = tmp_path / "argv.log"
    command = _child_script(
        tmp_path,
        "import sys\n"
        + _write_verdict_snippet(out)
        + f"attempts = pathlib.Path({str(attempts_file)!r})\n"
        + "n = int(attempts.read_text()) + 1 if attempts.exists() else 1\n"
        + "attempts.write_text(str(n))\n"
        + "if n == 1:\n"
        + "    (out / 'run_verdict.json').write_text(json.dumps("
        + "{'verdict': 'failed', 'error_class': 'TransientRuntimeError'}))\n"
        + "    raise SystemExit(1)\n"
        + f"pathlib.Path({str(argv_log)!r}).write_text('\\n'.join(sys.argv[1:]))\n"
        + "(out / 'run_verdict.json').write_text(json.dumps({'verdict': 'success'}))\n",
    )
    supervisor = RunSupervisor(command=command, output_dir=out, sleep=lambda _: None)
    assert supervisor.run() == 0

    assert attempts_file.read_text() == "2"
    argv = argv_log.read_text().splitlines()
    assert f"trainer.resume_from={out / 'checkpoint-4'}" in argv
    assert "model.lora.path=" in argv  # warm-start adapter cleared on resume


def test_stop_kills_whole_child_process_group(tmp_path) -> None:
    out = tmp_path / "run"
    grandchild_pid_file = tmp_path / "grandchild.pid"
    # Child spawns a grandchild in the SAME group, then sleeps forever.
    command = _child_script(
        tmp_path,
        "import pathlib, subprocess, sys, time\n"
        + "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
        + f"pathlib.Path({str(grandchild_pid_file)!r}).write_text(str(p.pid))\n"
        + "time.sleep(600)\n",
    )
    supervisor = RunSupervisor(
        command=command,
        output_dir=out,
        term_grace_seconds=3.0,
        sleep=lambda _: None,
    )
    result: list[int] = []
    runner = threading.Thread(target=lambda: result.append(supervisor.run()))
    runner.start()
    deadline = time.monotonic() + 10
    while not grandchild_pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert grandchild_pid_file.exists(), "child never started its grandchild"
    grandchild_pid = int(grandchild_pid_file.read_text())

    supervisor.request_stop(signal.SIGTERM)
    runner.join(timeout=10)
    assert not runner.is_alive()
    assert result and result[0] != 0

    # The grandchild must be gone too (whole process group was signaled).
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        os.kill(grandchild_pid, signal.SIGKILL)
        raise AssertionError("grandchild survived supervisor stop")


def test_verdict_file_contract_matches_train_cli() -> None:
    """The supervisor reads the same file name vrl-train writes."""
    from vrl.scripts import supervise, train

    assert supervise.RUN_VERDICT_NAME == train.RUN_VERDICT_NAME


def test_metrics_health_gate_is_disabled_by_default() -> None:
    args = build_parser().parse_args(["--config", "unit"])

    assert args.health_metrics is False


def test_metrics_health_gate_cli_thresholds_are_configurable() -> None:
    args = build_parser().parse_args(
        [
            "--config",
            "unit",
            "--health-metrics",
            "--health-poll-seconds",
            "2.5",
            "--health-failure-limit",
            "4",
            "--health-max-pre-update-logprob-diff",
            "0.02",
            "--health-max-stale-policy-versions",
            "1",
            "--health-max-stale-logprob-diff",
            "0.08",
            "--health-min-reward-std",
            "0.003",
            "--health-min-grad-norm",
            "0.0002",
        ],
    )

    assert args.health_metrics is True
    assert args.health_poll_seconds == 2.5
    assert args.health_failure_limit == 4
    assert args.health_max_pre_update_logprob_diff == 0.02
    assert args.health_max_stale_policy_versions == 1
    assert args.health_max_stale_logprob_diff == 0.08
    assert args.health_min_reward_std == 0.003
    assert args.health_min_grad_norm == 0.0002


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--health-poll-seconds", "0"),
        ("--health-failure-limit", "0"),
        ("--health-max-pre-update-logprob-diff", "nan"),
        ("--health-max-stale-policy-versions", "-1"),
        ("--health-max-stale-logprob-diff", "nan"),
        ("--health-min-reward-std", "-1"),
        ("--health-min-grad-norm", "inf"),
    ],
)
def test_metrics_health_gate_cli_rejects_invalid_thresholds(option, value) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--config", "unit", option, value])


def test_metrics_health_gate_allows_missing_and_header_only_files(tmp_path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    (out / "metrics.csv").write_text(_METRICS_HEADER)
    command = _child_script(
        tmp_path,
        _write_verdict_snippet(out)
        + "(out / 'run_verdict.json').write_text(json.dumps({'verdict': 'success'}))\n",
    )
    supervisor = RunSupervisor(
        command=command,
        output_dir=out,
        health=HealthGateConfig(poll_seconds=0.01),
        sleep=lambda _: None,
    )

    assert supervisor.run() == 0
    assert not (out / HEALTH_VERDICT_NAME).exists()


def test_continuous_health_gate_uses_stale_drift_bound() -> None:
    gate = MetricsHealthGate(
        HealthGateConfig(
            max_pre_update_logprob_diff=0.01,
            max_stale_policy_versions=1,
            max_stale_logprob_diff=0.05,
        ),
        Path("unused"),
    )

    stale_reasons, _ = gate._metric_health_reasons(
        _continuous_metric_row(0, parity="0.02", stale_versions="1"),
    )
    on_policy_reasons, _ = gate._metric_health_reasons(
        _continuous_metric_row(1, parity="0.02", stale_versions="0"),
    )

    assert stale_reasons == []
    assert any("exceeds maximum 0.01" in reason for reason in on_policy_reasons)


@pytest.mark.parametrize(
    ("stale_versions", "producer_errors", "expected"),
    [
        ("2", "0", "continuous_stale_versions 2 exceeds maximum 1"),
        ("0.5", "0", "must be a non-negative integer"),
        ("1", "0.5", "continuous_producer_errors must be a non-negative integer"),
        ("1", "-1", "continuous_producer_errors must be a non-negative integer"),
    ],
)
def test_continuous_health_gate_rejects_scheduler_failures(
    stale_versions,
    producer_errors,
    expected,
) -> None:
    gate = MetricsHealthGate(
        HealthGateConfig(max_stale_policy_versions=1),
        Path("unused"),
    )

    reasons, _ = gate._metric_health_reasons(
        _continuous_metric_row(
            0,
            stale_versions=stale_versions,
            producer_errors=producer_errors,
        ),
    )

    assert any(expected in reason for reason in reasons)


def test_continuous_health_gate_only_flags_new_producer_errors(tmp_path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    gate = MetricsHealthGate(
        HealthGateConfig(max_stale_policy_versions=1, failure_limit=3),
        out,
    )

    rows = [_continuous_metric_row(0, producer_errors="0")]
    _write_continuous_metrics(out, rows)
    assert gate.judge_new_rows() is False

    rows.append(_continuous_metric_row(1, producer_errors="1"))
    _write_continuous_metrics(out, rows)
    assert gate.judge_new_rows() is False
    assert gate._unhealthy_rows == 1

    rows.append(_continuous_metric_row(2, producer_errors="1"))
    _write_continuous_metrics(out, rows)
    assert gate.judge_new_rows() is False
    assert gate._unhealthy_rows == 0


def test_continuous_health_gate_trips_on_consecutive_producer_error_increases(
    tmp_path,
) -> None:
    out = tmp_path / "run"
    out.mkdir()
    gate = MetricsHealthGate(
        HealthGateConfig(max_stale_policy_versions=1, failure_limit=3),
        out,
    )

    for epoch, producer_errors in enumerate(("1", "2", "3")):
        rows = [_continuous_metric_row(i, producer_errors=str(i + 1)) for i in range(epoch + 1)]
        _write_continuous_metrics(out, rows)
        tripped = gate.judge_new_rows()
        assert tripped is (producer_errors == "3")

    verdict = json.loads((out / HEALTH_VERDICT_NAME).read_text())
    assert verdict["epoch"] == 2
    assert verdict["reasons"] == [
        "continuous_producer_errors increased from 2 to 3",
    ]


def test_continuous_health_gate_baselines_existing_producer_errors(tmp_path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    rows = [_continuous_metric_row(0, producer_errors="1")]
    _write_continuous_metrics(out, rows)
    command = _child_script(
        tmp_path,
        _write_verdict_snippet(out)
        + "with (out / 'metrics.csv').open('a') as handle:\n"
        + f"    handle.write({_continuous_metric_csv_row(1, producer_errors='0')!r})\n"
        + "(out / 'run_verdict.json').write_text(json.dumps({'verdict': 'success'}))\n",
    )
    supervisor = RunSupervisor(
        command=command,
        output_dir=out,
        health=HealthGateConfig(
            poll_seconds=0.01,
            max_stale_policy_versions=1,
            failure_limit=1,
        ),
        sleep=lambda _: None,
    )

    assert supervisor.run() == 0
    assert not (out / HEALTH_VERDICT_NAME).exists()


def test_metrics_health_gate_resets_failure_streak_after_healthy_row(tmp_path) -> None:
    out = tmp_path / "run"
    metrics = _METRICS_HEADER + _metric_row(0, reward_std="0")
    metrics += _metric_row(1)
    metrics += _metric_row(2, grad_norm="0")
    command = _child_script(
        tmp_path,
        _write_verdict_snippet(out)
        + f"(out / 'metrics.csv').write_text({metrics!r})\n"
        + "(out / 'run_verdict.json').write_text(json.dumps({'verdict': 'success'}))\n",
    )
    supervisor = RunSupervisor(
        command=command,
        output_dir=out,
        health=HealthGateConfig(poll_seconds=0.01, failure_limit=2),
        sleep=lambda _: None,
    )

    assert supervisor.run() == 0
    assert not (out / HEALTH_VERDICT_NAME).exists()


def test_metrics_health_gate_ignores_old_rows_and_partial_append(tmp_path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    old_metrics = _METRICS_HEADER
    old_metrics += _metric_row(0, reward_std="0")
    old_metrics += _metric_row(1, reward_std="0")
    (out / "metrics.csv").write_text(old_metrics)
    command = _child_script(
        tmp_path,
        _write_verdict_snippet(out)
        + "with (out / 'metrics.csv').open('a') as handle:\n"
        + "    handle.write('2,1.0,2.0,0')\n"
        + "    handle.flush()\n"
        + "    import time; time.sleep(0.08)\n"
        + "    handle.write(',0.1,0.001\\n')\n"
        + "    handle.flush()\n"
        + "(out / 'run_verdict.json').write_text(json.dumps({'verdict': 'success'}))\n",
    )
    supervisor = RunSupervisor(
        command=command,
        output_dir=out,
        health=HealthGateConfig(poll_seconds=0.01, failure_limit=2),
        sleep=lambda _: None,
    )

    assert supervisor.run() == 0
    assert not (out / HEALTH_VERDICT_NAME).exists()


def test_metrics_health_gate_only_checks_replaced_suffix_after_truncation(tmp_path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    retained_row = _metric_row(0)
    (out / "metrics.csv").write_text(_METRICS_HEADER + retained_row + _metric_row(1))
    command = _child_script(
        tmp_path,
        "import pathlib, time\n"
        + f"out = pathlib.Path({str(out)!r})\n"
        + f"(out / 'metrics.csv').write_text({_METRICS_HEADER + retained_row!r})\n"
        + "time.sleep(0.04)\n"
        + "with (out / 'metrics.csv').open('a') as handle:\n"
        + f"    handle.write({_metric_row(1, reward_std='0')!r})\n"
        + "    handle.flush()\n"
        + "time.sleep(600)\n",
    )
    supervisor = RunSupervisor(
        command=command,
        output_dir=out,
        health=HealthGateConfig(poll_seconds=0.01, failure_limit=1),
        term_grace_seconds=0.2,
        sleep=lambda _: None,
    )

    assert supervisor.run() != 0
    verdict = json.loads((out / HEALTH_VERDICT_NAME).read_text())
    assert verdict["epoch"] == 1


def test_metrics_health_gate_stops_child_group_and_writes_verdict(tmp_path) -> None:
    out = tmp_path / "run"
    grandchild_pid_file = tmp_path / "health-grandchild.pid"
    unhealthy_rows = _metric_row(
        0,
        loss="nan",
        reward_mean="inf",
        reward_std="0",
        grad_norm="0",
        parity="0.02",
    ) + _metric_row(
        1,
        loss="nan",
        reward_mean="inf",
        reward_std="0",
        grad_norm="0",
        parity="0.02",
    )
    command = _child_script(
        tmp_path,
        "import pathlib, subprocess, sys, time\n"
        + f"out = pathlib.Path({str(out)!r})\n"
        + "out.mkdir(parents=True, exist_ok=True)\n"
        + "grandchild = subprocess.Popen([sys.executable, '-c', "
        + "'import time; time.sleep(600)'])\n"
        + f"pathlib.Path({str(grandchild_pid_file)!r}).write_text(str(grandchild.pid))\n"
        + f"rows = {unhealthy_rows!r}.splitlines(keepends=True)\n"
        + "with (out / 'metrics.csv').open('w') as handle:\n"
        + f"    handle.write({_METRICS_HEADER!r})\n"
        + "    handle.flush()\n"
        + "    for row in rows:\n"
        + "        handle.write(row)\n"
        + "        handle.flush()\n"
        + "        time.sleep(0.05)\n"
        + "time.sleep(600)\n",
    )
    supervisor = RunSupervisor(
        command=command,
        output_dir=out,
        health=HealthGateConfig(poll_seconds=0.01, failure_limit=2),
        term_grace_seconds=0.2,
        sleep=lambda _: None,
    )

    assert supervisor.run() != 0
    verdict = json.loads((out / HEALTH_VERDICT_NAME).read_text())
    assert verdict["verdict"] == "failed"
    assert verdict["source"] == "metrics_health_gate"
    assert verdict["epoch"] == 1
    assert verdict["consecutive_unhealthy_rows"] == 2
    assert any("loss is not finite" in reason for reason in verdict["reasons"])
    assert any("reward_mean is not finite" in reason for reason in verdict["reasons"])
    assert any("exceeds maximum" in reason for reason in verdict["reasons"])
    assert any("reward_std" in reason for reason in verdict["reasons"])
    assert any("grad_norm" in reason for reason in verdict["reasons"])

    grandchild_pid = int(grandchild_pid_file.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        os.kill(grandchild_pid, signal.SIGKILL)
        raise AssertionError("grandchild survived metrics health-gate stop")
