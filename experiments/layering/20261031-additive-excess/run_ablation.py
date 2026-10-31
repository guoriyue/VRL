"""Execute the annotation-only pure-RL pair and fixed matched evaluations."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).parent
REPO = Path("/mnt/nvme/VRL-layering-excess-20261031")
EVAL_REPO = Path("/home/ubuntu/VRL-layering")
PREVIOUS = Path("/mnt/nvme/layering_steps30_curriculum_20261030")
PY = "/mnt/nvme/venv-layering/bin/python"
TORCHRUN = "/mnt/nvme/venv-layering/bin/torchrun"
CHECKPOINT = Path("/mnt/nvme/layering/checkpoint-28-fsdp4-adam-reset-20261029")
BASE = Path("/mnt/nvme/layering_baseline_sampling30_20261030/baseline28")
FRESH = Path("/mnt/nvme/layering_fresh_complete_review_20261027/eval_rows.jsonl")


def stage(name, **details):
    value = {"stage": name, "pid": os.getpid(), "time": time.time(), **details}
    temporary = ROOT / "stage.tmp"
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(ROOT / "stage.json")
    print(json.dumps(value), flush=True)


def run(command, logfile):
    isolated = "vrl.scripts.train" in command or str(ROOT / "preflight.py") in command
    repo = REPO if isolated else EVAL_REPO
    environment = {**os.environ, "PYTHONPATH": f"{repo}:/mnt/nvme/layering_reward_audit_20261028"}
    with logfile.open("w") as log:
        subprocess.run(
            command,
            cwd=repo,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config = {"commands": {"excess": json.loads((ROOT / "protocol.json").read_text())["command"]}}
    if not args.run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "arms": list(config["commands"]),
                    "updates_per_arm": 7,
                    "eval_cases_each": 49,
                    "automatic_longtrain": False,
                }
            )
        )
        return
    os.environ.update(
        HF_HOME="/mnt/nvme/hf",
        HF_HUB_OFFLINE="1",
        OMP_NUM_THREADS="4",
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
        CUDA_VISIBLE_DEVICES="0,1,2,3",
        PYTORCH_ALLOC_CONF="expandable_segments:True",
        TMPDIR="/mnt/nvme/tmp",
        PYTHONPATH=f"{REPO}:/mnt/nvme/layering_reward_audit_20261028",
    )
    with (
        (ROOT / "runner.lock").open("a") as own,
        Path("/mnt/nvme/layering_dense_20261025/runner.lock").open("a") as gpu,
    ):
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stage("waiting_for_gpu_lock")
        fcntl.flock(gpu, fcntl.LOCK_EX)
        stage("verifying_frozen_inputs")
        frozen = json.loads((ROOT / "source_hashes.json").read_text())
        for filename, digest in frozen.items():
            with Path(filename).open("rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != digest:
                    raise RuntimeError(f"Frozen input changed: {filename}")
        if (ROOT / "preflight").exists() or (ROOT / "train").exists():
            raise RuntimeError("Existing run output requires explicit recovery")
        (ROOT / "logs").mkdir(exist_ok=True)
        stage("production_preflight")
        run(
            [
                TORCHRUN,
                "--standalone",
                "--nproc-per-node=4",
                str(ROOT / "preflight.py"),
                "--adapter",
                "/mnt/nvme/layering/adapters/v3c_28_live",
                "--output",
                str(ROOT / "preflight"),
                "--manifest",
                str(ROOT / "preflight_row.jsonl"),
                "--config",
                "experiment/qwen_image_layered/online_grpo_dense_control",
                "--seed-per-rank",
            ],
            ROOT / "logs/preflight.log",
        )
        for rank in range(4):
            result = json.loads((ROOT / f"preflight/rank_{rank}/result.json").read_text())
            assert result["passed"] and result["layers"] == 5 and result["num_steps"] == 30
            assert all(result["conditioning_exact"].values())
            assert result["reward_artifact_exact"] and result["complete_truth"]
        assert len(list(BASE.glob("*/seed4401/main_scores.json"))) == 34
        assert Path("/mnt/nvme/layering_fresh_baseline_sampling30_20261030/report.json").exists()
        for arm in ["excess"]:
            folder = ROOT
            if list((folder / "train").glob("checkpoint-*")) or list(
                (folder / "rollouts").glob("*/scores.json")
            ):
                raise RuntimeError(
                    f"Refusing to overwrite existing arm {arm}; explicit recovery needed"
                )
            (folder / "logs").mkdir(exist_ok=True)
            (folder / "trainer-parking").mkdir(exist_ok=True)
            stage("training", arm=arm, start_step=28, target_step=35)
            run(config["commands"][arm], folder / "logs/train35.log")
            checkpoint, adapter = folder / "train/checkpoint-35", folder / "adapter35"
            stage("exporting_live_adapter", arm=arm)
            run(
                [
                    PY,
                    str(EVAL_REPO / "outputs/layered_grpo/eval/export_live_adapter.py"),
                    str(checkpoint),
                    str(adapter),
                ],
                folder / "logs/export35.log",
            )
            run(
                [
                    PY,
                    "-m",
                    "tools.layering_verify_live_adapter",
                    "--checkpoint",
                    str(checkpoint),
                    "--baseline",
                    str(CHECKPOINT),
                    "--adapter",
                    str(adapter),
                    "--output",
                    str(folder / "verify35.json"),
                ],
                folder / "logs/verify35.log",
            )
            verified = json.loads((folder / "verify35.json").read_text())
            assert (
                verified["owned_export_exact"]
                and verified["weights_finite"]
                and verified["changed_tensors"] > 0
            )
            assert verified["adam_step_max"] == 7
            assert all(g["eps"] == 1e-12 and g["lr"] == 5e-5 for g in verified["optimizer_groups"])
            for cohort, rows in [("main", PREVIOUS / "eval_rows.jsonl"), ("fresh", FRESH)]:
                stage("evaluating", arm=arm, cohort=cohort)
                output = folder / f"eval_{cohort}"
                run(
                    [
                        TORCHRUN,
                        "--standalone",
                        "--nproc-per-node=4",
                        str(PREVIOUS / "evaluate.py"),
                        "--steps",
                        "30",
                        "--rows",
                        str(rows),
                        "--adapter",
                        str(adapter),
                        "--output",
                        str(output),
                        "--reward-kwargs",
                        str(PREVIOUS / "reward_kwargs.json"),
                    ],
                    folder / f"logs/eval_{cohort}.log",
                )
                for row in map(json.loads, rows.read_text().splitlines()):
                    sample = output / row["id"] / "seed4401"
                    score = json.loads((sample / "main_scores.json").read_text())
                    assert (
                        score["steps"] == 30
                        and score["seed"] == 4401
                        and score["adapter"] == str(adapter)
                    )
                    assert len(list(sample.glob("layer_*.png"))) == row["layers"]
            run(
                [
                    PY,
                    "-m",
                    "tools.layering_natural_report",
                    "--rows",
                    str(PREVIOUS / "eval_expanded.jsonl"),
                    "--output",
                    str(folder / "main_report.json"),
                    str(BASE),
                    str(folder / "eval_main"),
                ],
                folder / "logs/report_main.log",
            )
            run(
                [
                    PY,
                    str(PREVIOUS / "report_fresh.py"),
                    "--arm",
                    str(folder / "eval_fresh"),
                    "--output",
                    str(folder / "fresh_report.json"),
                ],
                folder / "logs/report_fresh.log",
            )
            stage("arm_complete_pending_paired_review", arm=arm)
        stage("ablation_complete_pending_quality_review")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        stage("failed", error=str(error))
        raise
