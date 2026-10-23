"""Run supervisor tests — real subprocesses, real process groups, no fakes."""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

from vrl.scripts.supervise import RunSupervisor
from vrl.training_run_result import TrainingRunResultWriter


def _child_script(tmp_path: Path, body: str) -> list[str]:
    script = tmp_path / "child.py"
    script.write_text(body)
    return [sys.executable, str(script)]


def _write_result_snippet(out_dir: Path) -> str:
    return (
        "import json, os, pathlib\n"
        f"out = pathlib.Path({str(out_dir)!r})\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
    )


def test_success_first_attempt(tmp_path) -> None:
    out = tmp_path / "run"
    command = _child_script(
        tmp_path,
        _write_result_snippet(out)
        + "(out / 'training_run_result.json').write_text(json.dumps({'status': 'success'}))\n",
    )
    supervisor = RunSupervisor(command=command, output_dir=out, sleep=lambda _: None)
    assert supervisor.run() == 0


def test_same_cause_circuit_breaker_stops(tmp_path) -> None:
    out = tmp_path / "run"
    attempts_file = tmp_path / "attempts"
    command = _child_script(
        tmp_path,
        _write_result_snippet(out)
        + f"attempts = pathlib.Path({str(attempts_file)!r})\n"
        + "n = int(attempts.read_text()) + 1 if attempts.exists() else 1\n"
        + "attempts.write_text(str(n))\n"
        + "(out / 'training_run_result.json').write_text(json.dumps("
        "{'status': 'failed', 'error_class': 'ValueError'}))\n" + "raise SystemExit(1)\n",
    )
    supervisor = RunSupervisor(command=command, output_dir=out, sleep=lambda _: None)
    assert supervisor.run() == 1
    assert attempts_file.read_text() == "2"  # stopped at the breaker, not later


def test_transient_failure_then_success_restarts(tmp_path) -> None:
    out = tmp_path / "run"
    marker = tmp_path / "failed_once"
    command = _child_script(
        tmp_path,
        _write_result_snippet(out)
        + f"marker = pathlib.Path({str(marker)!r})\n"
        + "if not marker.exists():\n"
        + "    marker.write_text('1')\n"
        + "    (out / 'training_run_result.json').write_text(json.dumps("
        "{'status': 'terminated', 'signal': 15, 'signal_name': 'SIGTERM'}))\n"
        + "    raise SystemExit(143)\n"
        + "(out / 'training_run_result.json').write_text(json.dumps({'status': 'success'}))\n",
    )
    supervisor = RunSupervisor(command=command, output_dir=out, sleep=lambda _: None)
    assert supervisor.run() == 0


def test_missing_result_is_a_distinct_failure_class(tmp_path) -> None:
    out = tmp_path / "run"
    # Child dies without unwinding: no result file is ever written.
    command = _child_script(tmp_path, "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n")
    supervisor = RunSupervisor(command=command, output_dir=out, sleep=lambda _: None)
    exit_code = supervisor.run()
    assert exit_code != 0  # breaker tripped on consecutive no-result deaths


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
        model_identity={"schema": "test"},
    )

    attempts_file = tmp_path / "attempts"
    argv_log = tmp_path / "argv.log"
    command = _child_script(
        tmp_path,
        "import sys\n"
        + _write_result_snippet(out)
        + f"attempts = pathlib.Path({str(attempts_file)!r})\n"
        + "n = int(attempts.read_text()) + 1 if attempts.exists() else 1\n"
        + "attempts.write_text(str(n))\n"
        + "if n == 1:\n"
        + "    (out / 'training_run_result.json').write_text(json.dumps("
        + "{'status': 'failed', 'error_class': 'TransientRuntimeError'}))\n"
        + "    raise SystemExit(1)\n"
        + f"pathlib.Path({str(argv_log)!r}).write_text('\\n'.join(sys.argv[1:]))\n"
        + "(out / 'training_run_result.json').write_text(json.dumps({'status': 'success'}))\n",
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
    # The worker publishes readiness only after installing its TERM handler.
    worker = (
        "import os,pathlib,signal,time; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"pathlib.Path({str(grandchild_pid_file)!r}).write_text(str(os.getpid())); "
        "time.sleep(600)"
    )
    command = _child_script(
        tmp_path,
        "import subprocess, sys, time\n"
        + f"subprocess.Popen([sys.executable, '-c', {worker!r}])\n"
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


def test_train_result_uses_cleanup_wrapper_root_class(tmp_path) -> None:
    from vrl.ray.operation_deadline import RayOperationTimeout
    from vrl.rollouts.collector.core import (
        PromptCollectionCleanupError,
    )
    from vrl.rollouts.orchestration.rollout_runtime import RolloutPhaseCleanupError

    root = RayOperationTimeout("rollout.generation.batch", 1.0)
    collection = PromptCollectionCleanupError(
        root,
        [RuntimeError("reward cleanup failed")],
    )
    wrapped = RolloutPhaseCleanupError(
        collection,
        RuntimeError("offload cleanup failed"),
    )
    out = tmp_path / "run"

    TrainingRunResultWriter(str(out), environ={}).write(error=wrapped)

    result = json.loads((out / "training_run_result.json").read_text())
    assert result["error_class"] == "RayOperationTimeout"
    assert "offload cleanup failed" in result["error_message"]
    assert "RayOperationTimeout" in result["error_message"]


def test_supervisor_clears_previous_success_before_new_process(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "training_run_result.json").write_text('{"status": "success"}')
    command = _child_script(tmp_path, "pass\n")
    outcome = RunSupervisor(command=command, output_dir=out)._run_attempt([])
    assert outcome.exit_code == 0
    assert outcome.result is None
    assert not outcome.succeeded


def test_supervisor_rejects_success_result_with_nonzero_exit(tmp_path):
    out = tmp_path / "run"
    command = _child_script(
        tmp_path,
        "from vrl.training_run_result import TrainingRunResultWriter\n"
        f"TrainingRunResultWriter({str(out)!r}).write()\n"
        "raise SystemExit(3)\n",
    )
    outcome = RunSupervisor(command=command, output_dir=out)._run_attempt([])
    assert outcome.exit_code == 3
    assert outcome.result["status"] == "success"
    assert not outcome.succeeded
    assert outcome.failure_class == "result-success-exit-3"
