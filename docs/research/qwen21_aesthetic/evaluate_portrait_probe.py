"""Run the frozen Alice probe when GPUs are free; never interrupt training."""

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path("/home/ubuntu/VRL")
ROOT = Path("/mnt/nvme/outputs/qwen21_aesthetic_lora")
PROBE = ROOT / "alice_portrait_probe"
RUN = ROOT / "window2_trial/train_to_60_after_eval43"


def prepare():
    protocol = json.loads((PROBE / "protocol.json").read_text())
    manifest = PROBE / "prompts.jsonl"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == protocol["manifest_sha256"]
    rows = [json.loads(s) for s in manifest.read_text().splitlines()]
    assert len(rows) == 12
    jobs = []
    for i, row in enumerate(rows):
        assert row["seeds"] == list(range(2026092700, 2026092704))
        path = PROBE / f"prompt_{i:02d}.jsonl"
        content = json.dumps({"id": row["id"], "prompt": row["prompt"]}) + "\n"
        if path.exists():
            assert path.read_text() == content
        else:
            path.write_text(content)
        output = PROBE / f"generated/prompt_{i:02d}"
        command = [
            sys.executable,
            "-m",
            "vrl.scripts.eval.image_checkpoint_eval",
            "--run-dir",
            str(RUN),
            "--checkpoint",
            f"lora43={ROOT}/window2_trial/train_to_60/checkpoint-43",
            "--checkpoint",
            f"lora60={RUN}/checkpoint-60",
            "--manifest",
            str(path),
            "--samples-per-prompt",
            "4",
            "--width",
            "1024",
            "--height",
            "1024",
            "--num-steps",
            "40",
            "--seed",
            "2026092700",
            "--blind-seed",
            "260927",
            "--eval-policy-override",
            "model.memory.cpu_resident=[]",
            "--device",
            "cuda:0",
            "--reward-device",
            "cuda:0",
            "--output-dir",
            str(output),
        ]
        jobs.append(
            dict(index=i, gpu=i % 4, prompt=row["prompt"], output=str(output), command=command)
        )
    (PROBE / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
    return jobs


def run_lane(gpu, jobs):
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        HF_HOME="/mnt/nvme/hf",
        HF_HUB_OFFLINE="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        PYTHONPATH=str(REPO),
    )
    for job in jobs:
        if job["gpu"] != gpu:
            continue
        with (PROBE / f"prompt_{job['index']:02d}.log").open("x") as log:
            subprocess.run(
                job["command"], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )


def verify(jobs):
    from PIL import Image

    rows = []
    for job in jobs:
        records = [
            json.loads(s)
            for s in (Path(job["output"]) / "report/scores.jsonl").read_text().splitlines()
        ]
        expected = {(arm, s) for arm in ("base", "lora43", "lora60") for s in range(4)}
        seen = set()
        for row in records:
            key = row["checkpoint_label"], row["sample_index"]
            assert key in expected and key not in seen
            seen.add(key)
            assert row["prompt_index"] == 0 and row["prompt"] == job["prompt"]
            assert row["seed"] == 2026092700 + row["sample_index"]
            p = Path(row["image_path"])
            assert hashlib.sha256(p.read_bytes()).hexdigest() == row["image_sha256"]
            with Image.open(p) as im:
                assert im.size == (1024, 1024)
            rows.append(dict(row, prompt_index=job["index"]))
        assert seen == expected
    (PROBE / "scores.verified.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = summarize(rows)
    (PROBE / "paired_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(dict(validated_images=len(rows), prompts=12, seeds_per_prompt=4)))


def summarize(rows):
    """Average seeds within prompts before bootstrapping the paired differences."""
    cells = {}
    for row in rows:
        key = row["prompt_index"], row["sample_index"], row["checkpoint_label"]
        assert key not in cells, f"Duplicate cell: {key}"
        value = float(row["r_aesthetic"])
        assert math.isfinite(value), f"Nonfinite score: {key}"
        cells[key] = value
    arms = ("base", "lora43", "lora60")
    assert set(cells) == {(p, s, a) for p in range(12) for s in range(4) for a in arms}
    comparisons = {}
    for candidate, reference in (("lora43", "base"), ("lora60", "base"), ("lora60", "lora43")):
        deltas = [
            statistics.mean(cells[p, s, candidate] - cells[p, s, reference] for s in range(4))
            for p in range(12)
        ]
        rng = random.Random(260927)
        draws = sorted(statistics.mean(rng.choices(deltas, k=12)) for _ in range(20000))
        comparisons[f"{candidate}_minus_{reference}"] = dict(
            mean_delta=statistics.mean(deltas),
            prompt_deltas=deltas,
            prompt_wins=sum(d > 0 for d in deltas),
            prompt_ties=sum(d == 0 for d in deltas),
            prompt_bootstrap_ci95=[draws[499], draws[19499]],
        )
    return dict(
        metric="r_aesthetic",
        prompts=12,
        seeds_per_prompt=4,
        bootstrap_unit="prompt (four seeds averaged)",
        bootstrap_replicates=20000,
        bootstrap_seed=260927,
        comparisons=comparisons,
        interpretation="Targeted descriptive diagnostic, not a representative held-out test. "
        "Prompts share two subjects and six styles; intervals are conditional on this "
        "small selected set, unadjusted for multiple comparisons, and do not establish "
        "human preference or broad aesthetic improvement.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    jobs = prepare()
    if not args.execute:
        print("Prepared 12 jobs (144 images). No GPU work launched. Round60 checkpoint required.")
        return
    assert (RUN / "checkpoint-60/checkpoint_meta.json").is_file()
    busy = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if busy:
        raise RuntimeError(
            "GPUs have active compute processes; leave training untouched and retry at an idle milestone."
        )
    assert not (PROBE / "generated").exists(), (
        "Existing evaluation: inspect/resume explicitly, never overwrite."
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda gpu: run_lane(gpu, jobs), range(4)))
    verify(jobs)


if __name__ == "__main__":
    main()
