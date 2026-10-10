"""Bounded four-GPU training/evaluation campaign; failures require diagnosis."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/mnt/nvme/outputs/qwen21_aesthetic_lora")
REPO = Path("/home/ubuntu/VRL")
PYTHON = "/mnt/nvme/venv-main/bin/python"
UNIT = "qwen21-aesthetic-lora.service"


def status(phase, **details):
    record = dict(time=time.time(), phase=phase, **details)
    temporary = ROOT / "campaign_status.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(ROOT / "campaign_status.json")
    print(json.dumps(record), flush=True)


def require_success(run):
    for rank in range(4):
        result = json.loads((run / f"training_run_result.rank-{rank}.json").read_text())
        if result["status"] != "success":
            raise RuntimeError(f"Rank {rank} failed: {result}")
    if not (run / "checkpoint-final" / "checkpoint.pt").is_file():
        raise RuntimeError(f"Missing final checkpoint: {run}")


def evaluate(run, step, limit, *, build_gallery=True):
    status("evaluating", step=step, prompts_per_gpu=limit or 16)
    processes = []
    handles = []
    try:
        for gpu in range(4):
            output = ROOT / f"eval_step{step}" / f"gpu{gpu}"
            output.parent.mkdir(exist_ok=True)
            handle = (output.parent / f"gpu{gpu}.log").open("w")
            handles.append(handle)
            command = [
                PYTHON,
                "-m",
                "vrl.scripts.eval.image_checkpoint_eval",
                "--run-dir",
                str(run),
                "--checkpoint",
                f"lora{step}={run}/checkpoint-final",
                "--manifest",
                str(ROOT / f"validation_gpu{gpu}.jsonl"),
                "--limit",
                str(limit),
                "--samples-per-prompt",
                "1",
                "--width",
                "1024",
                "--height",
                "1024",
                "--num-steps",
                "40",
                "--seed",
                str(20260926 + gpu * 16),
                "--blind-seed",
                "260925",
                "--eval-policy-override",
                "model.memory.cpu_resident=[]",
                "--device",
                "cuda:0",
                "--reward-device",
                "cuda:0",
                "--output-dir",
                str(output),
            ]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            processes.append(
                subprocess.Popen(
                    command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT
                )
            )
        deadline = time.monotonic() + 3600
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError(f"Evaluation failed at step {step}; inspect GPU logs")
            if time.monotonic() > deadline:
                raise TimeoutError("Native evaluation exceeded one hour")
            time.sleep(5)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError(f"Evaluation failed at step {step}")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for handle in handles:
            handle.close()
    from vrl.scripts.eval.score_report import write_curve_report

    rows = []
    for gpu in range(4):
        path = ROOT / f"eval_step{step}" / f"gpu{gpu}" / "report" / "scores.jsonl"
        for line in path.read_text().splitlines():
            row = json.loads(line)
            row["prompt_index"] += gpu * 16
            rows.append(row)
    summary = write_curve_report(
        rows, ROOT / f"eval_step{step}" / "aggregate", bootstrap_resamples=10000, seed=260925
    )
    delta = summary["arms"][f"lora{step}"]["scores"]["aesthetic"]["paired_delta_from_base"]
    status("evaluation_complete", step=step, aesthetic_delta=delta)
    if build_gallery:
        subprocess.run(
            [PYTHON, str(REPO / "docs/research/qwen21_aesthetic/build_lora_gallery.py")],
            cwd=REPO,
            check=True,
        )
    if delta["clear_decrease"] and delta["mean"] < -0.1:
        raise RuntimeError("Held-out aesthetic proxy regressed; revise before continuing")


def main():
    sys.path.insert(0, str(REPO))
    os.environ.update(
        HF_HOME="/mnt/nvme/hf",
        HF_HUB_OFFLINE="1",
        PYTHONPATH=str(REPO),
        OMP_NUM_THREADS="4",
        TOKENIZERS_PARALLELISM="false",
    )
    status("waiting_for_acceptance")
    deadline = time.monotonic() + 3600
    while True:
        state = subprocess.check_output(
            ["systemctl", "--user", "show", UNIT, "-p", "ActiveState", "--value"], text=True
        ).strip()
        if state not in ("active", "activating", "deactivating"):
            break
        if time.monotonic() > deadline:
            raise TimeoutError("Acceptance has not finished in one hour; inspect service")
        time.sleep(10)
    result = subprocess.check_output(
        ["systemctl", "--user", "show", UNIT, "-p", "Result", "--value"], text=True
    ).strip()
    if result != "success":
        raise RuntimeError(f"Acceptance service failed: {result}")
    run = ROOT / "acceptance"
    require_success(run)
    evaluate(run, 2, 4)
    # Bounded stages leave reviewable checkpoints. No automatic blind retries,
    # no unbounded optimization of an unverified aesthetic proxy.
    for step in (5, 10, 20):
        previous = run
        run = ROOT / f"train_to_{step}"
        run.mkdir(exist_ok=False)
        status("training", target_step=step, resume=str(previous / "checkpoint-final"))
        command = [
            str(ROOT / "launch.sh"),
            f"trainer.resume_from={previous}/checkpoint-final",
            f"trainer.output_dir={run}",
            f"trainer.total_epochs={step}",
            "trainer.save_freq=5",
            "rollout.n_samples_per_prompt=8",
            "rollout.samples_per_generation_batch=2",
            "actor.training_microbatch_size=2",
        ]
        with (run / "service.log").open("w") as log:
            subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        require_success(run)
        evaluate(run, step, 0 if step == 20 else 4)
    status("awaiting_quality_review", checkpoint=str(run / "checkpoint-final"))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        status("failed", error=repr(error))
        raise
