"""Run supervisor tests — real subprocesses, real process groups, no fakes."""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from pathlib import Path

from vrl.scripts.supervise import RunSupervisor


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

    argv_log = tmp_path / "argv.log"
    command = _child_script(
        tmp_path,
        "import sys\n"
        + _write_verdict_snippet(out)
        + f"pathlib.Path({str(argv_log)!r}).write_text('\\n'.join(sys.argv[1:]))\n"
        + "(out / 'run_verdict.json').write_text(json.dumps({'verdict': 'success'}))\n",
    )
    supervisor = RunSupervisor(command=command, output_dir=out, sleep=lambda _: None)
    assert supervisor.run() == 0

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
