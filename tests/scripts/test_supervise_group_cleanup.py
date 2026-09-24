"""A crashed group leader must not leave live workers behind the next attempt."""

import contextlib
import json
import os
import signal
import sys

from vrl.scripts.supervise import RunSupervisor


def test_restart_cleans_term_resistant_workers_after_leader_sigkill(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    worker_pid = tmp_path / "worker.pid"
    child = tmp_path / "child.py"
    worker = (
        "import os,pathlib,signal,time; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"pathlib.Path({str(worker_pid)!r}).write_text(str(os.getpid())); "
        "time.sleep(30)"
    )
    child.write_text(
        "import json,os,pathlib,signal,subprocess,sys,time\n"
        f"pid_file=pathlib.Path({str(worker_pid)!r})\n"
        "if not pid_file.exists():\n"
        f"    subprocess.Popen([sys.executable,'-c',{worker!r}])\n"
        "    while not pid_file.exists():time.sleep(.01)\n"
        "    os.kill(os.getpid(),signal.SIGKILL)\n"
        "pid=int(pid_file.read_text())\n"
        "stat=pathlib.Path(f'/proc/{pid}/stat')\n"
        "live=stat.exists() and stat.read_text().rsplit(')',1)[1].split()[0]!='Z'\n"
        f"pathlib.Path({str(output / 'training_run_result.json')!r}).write_text(json.dumps({{'status':'failed' if live else 'success','error_class':'LeakedWorker'}}))\n"
        "sys.exit(1 if live else 0)\n"
    )
    supervisor = RunSupervisor(
        command=[sys.executable, str(child)],
        output_dir=output,
        max_attempts=2,
        term_grace_seconds=0.1,
        backoff_seconds=0,
    )
    try:
        assert supervisor.run() == 0
        assert json.loads((output / "training_run_result.json").read_text())["status"] == "success"
    finally:
        if worker_pid.exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(worker_pid.read_text()), signal.SIGKILL)
