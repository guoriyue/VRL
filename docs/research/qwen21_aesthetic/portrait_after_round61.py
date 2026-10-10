"""One-time, identity-pinned handoff after a complete round61 checkpoint."""

import json
import subprocess
import sys
import time
from pathlib import Path

from continue_window2_to240 import verify_checkpoint

REPO = Path("/home/ubuntu/VRL")
ROOT = Path("/mnt/nvme/outputs/qwen21_aesthetic_lora")
OUT = ROOT / "alice_portrait_probe"
UNIT = "qwen21-aesthetic-eval43-resume.service"
INVOCATION = "ab6c9c50861f4da09ddb1f7c75954f15"
CHECKPOINT = ROOT / "window2_trial/train_to_120_after_eval43/checkpoint-61"


def state():
    raw = subprocess.check_output(
        [
            "systemctl",
            "--user",
            "show",
            UNIT,
            "-p",
            "MainPID",
            "-p",
            "ActiveState",
            "-p",
            "InvocationID",
        ],
        text=True,
    )
    return dict(line.split("=", 1) for line in raw.splitlines())


def record(phase, **details):
    value = dict(time=time.time(), phase=phase, **details)
    temporary = OUT / "handoff_status.tmp"
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(OUT / "handoff_status.json")
    print(json.dumps(value), flush=True)


def require_same_training():
    current = state()
    assert current == dict(MainPID="779817", ActiveState="active", InvocationID=INVOCATION), (
        current
    )


def main():
    require_same_training()
    assert not (OUT / "generated").exists()
    report = ROOT / "window2_trial/pickscore_15_60/vs_lora15/summary.json"
    delta = json.loads(report.read_text())["arms"]["lora60"]["scores"]["pickscore"][
        "paired_delta_from_base"
    ]
    assert delta["count"] == 64 and delta["bootstrap_95ci"][1] < 0
    record(
        "waiting_for_checkpoint61",
        unit=UNIT,
        invocation=INVOCATION,
        reason="Round60 native PickScore regressed vs15; run the frozen user-requested portrait diagnostic.",
        pickscore_delta=delta,
    )
    deadline = time.monotonic() + 7200
    while not (CHECKPOINT / "checkpoint_meta.json").exists():
        require_same_training()
        if time.monotonic() > deadline:
            raise TimeoutError("Observer expired; leave existing training untouched")
        time.sleep(15)
    evidence = verify_checkpoint(CHECKPOINT, 61)
    (OUT / "resume_checkpoint61.json").write_text(json.dumps(evidence, indent=2) + "\n")
    require_same_training()
    record("stopping_after_saved_checkpoint", checkpoint=evidence)
    subprocess.run(["systemctl", "--user", "stop", UNIT], check=True)
    current = state()
    assert current["MainPID"] == "0" and current["ActiveState"] == "inactive", current
    record("portrait_evaluation", checkpoint=evidence)
    subprocess.run(
        [
            sys.executable,
            str(REPO / "docs/research/qwen21_aesthetic/evaluate_portrait_probe.py"),
            "--execute",
        ],
        cwd=REPO,
        check=True,
    )
    record("portrait_complete_pending_training_decision", checkpoint=evidence)


if __name__ == "__main__":
    main()
