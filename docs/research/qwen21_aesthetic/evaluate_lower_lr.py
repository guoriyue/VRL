"""Verify paired images and independently score the completed lower-LR trial."""

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/ubuntu/VRL")
OLD = Path("/mnt/nvme/outputs/qwen21_aesthetic_lora/window2_trial")
ROOT = OLD.parent / "window2_lr3e5_from15"
UNIT = "qwen21-aesthetic-lr3e5-from15.service"
INVOCATION = "9088fbfa11eb4de197f85f20015a5522"
PYTHON = "/mnt/nvme/venv-main/bin/python"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    sys.path.insert(0, str(REPO))
    deadline = time.monotonic() + 18 * 3600
    while True:
        raw = subprocess.check_output(
            [
                "systemctl",
                "--user",
                "show",
                UNIT,
                "-p",
                "ActiveState",
                "-p",
                "MainPID",
                "-p",
                "InvocationID",
                "-p",
                "Result",
            ],
            text=True,
        )
        s = dict(line.split("=", 1) for line in raw.splitlines())
        if s["ActiveState"] not in ("active", "activating", "deactivating"):
            assert (
                s["ActiveState"] == "inactive" and s["Result"] == "success" and s["MainPID"] == "0"
            ), s
            assert s["InvocationID"] in ("", INVOCATION), s
            break
        assert s["InvocationID"] == INVOCATION and s["MainPID"] == "881682", s
        if time.monotonic() > deadline:
            raise TimeoutError("Observation deadline: training remains untouched")
        time.sleep(30)
    campaign = module("campaign", OLD / "run.py")
    campaign.require_success(ROOT / "train_to_30")
    integrity = module("integrity", OLD.parent / "fresh_test256_window15/evaluate.py")
    continuation = module(
        "continuation", REPO / "docs/research/qwen21_aesthetic/continue_window2_to240.py"
    )
    checkpoint = continuation.verify_checkpoint(ROOT / "train_to_30/checkpoint-final", 30)
    import torch

    saved = torch.load(
        ROOT / "train_to_30/checkpoint-final/checkpoint.pt", map_location="cpu", weights_only=False
    )
    assert all(g["lr"] == 3e-5 for g in saved["trainer"]["optimizer"]["param_groups"])
    del saved
    paired_rows = []
    for gpu in range(4):
        prompts = [
            json.loads(s) for s in (ROOT / f"validation_gpu{gpu}.jsonl").read_text().splitlines()
        ]
        assert len(prompts) == 16
        baseline_hashes = None
        for step, folder in [(15, OLD), (30, ROOT)]:
            rows = [
                json.loads(s)
                for s in (folder / f"eval_step{step}/gpu{gpu}/report/scores.jsonl")
                .read_text()
                .splitlines()
            ]
            integrity.validate_shard(rows, prompts, f"lora{step}", 20260926 + 16 * gpu)
            hashes = {
                r["prompt_index"]: r["image_sha256"]
                for r in rows
                if r["checkpoint_label"] == "base"
            }
            if baseline_hashes is None:
                baseline_hashes = hashes
            else:
                assert baseline_hashes == hashes, "Native base outputs drifted between evaluations"
            for row in rows:
                if row["checkpoint_label"] == "base":
                    continue
                row["prompt_index"] += gpu * 16
                paired_rows.append(row)
    from vrl.scripts.eval.score_report import write_curve_report

    write_curve_report(
        paired_rows,
        ROOT / "aesthetic_30_vs_15",
        base_label="lora15",
        bootstrap_resamples=10000,
        seed=260925,
    )
    (ROOT / "paired_integrity.json").write_text(
        json.dumps(
            dict(valid=True, prompts=64, native_base_hashes_identical=True, checkpoint=checkpoint),
            indent=2,
        )
        + "\n"
    )
    link = ROOT / "eval_step15"
    if link.exists():
        assert link.resolve() == (OLD / "eval_step15").resolve()
    else:
        link.symlink_to(OLD / "eval_step15", target_is_directory=True)
    subprocess.run(
        [
            PYTHON,
            str(REPO / "docs/research/qwen21_aesthetic/cross_eval_pickscore.py"),
            "--root",
            str(ROOT),
            "--steps",
            "15",
            "30",
            "--output",
            str(ROOT / "pickscore_15_30"),
            "--reuse",
            str(OLD / "pickscore_5_15"),
        ],
        cwd=REPO,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(REPO)),
        check=True,
    )
    (ROOT / "independent_eval_complete.json").write_text(
        json.dumps(
            dict(
                time=time.time(), checkpoint=checkpoint, status="complete_pending_quality_review"
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
