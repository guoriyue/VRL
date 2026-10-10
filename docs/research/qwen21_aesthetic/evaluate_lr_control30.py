"""Queued native evaluation of the original-LR round30 control; no training mutation."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/ubuntu/VRL")
OLD = Path("/mnt/nvme/outputs/qwen21_aesthetic_lora/window2_trial")
LOW = OLD.parent / "window2_lr3e5_from15"
OUT = LOW / "lr_control30"
SOURCE = OLD / "train_to_60/checkpoint-30"
PYTHON = "/mnt/nvme/venv-main/bin/python"
WAIT = "qwen21-aesthetic-lr3e5-independent-eval.service"
INVOCATION = "b59acbb48f574e47b4dcd7a6919aab07"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def state(unit):
    raw = subprocess.check_output(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "-p",
            "MainPID",
            "-p",
            "ActiveState",
            "-p",
            "Result",
            "-p",
            "InvocationID",
        ],
        text=True,
    )
    return dict(line.split("=", 1) for line in raw.splitlines())


def status(phase, **kwargs):
    row = dict(time=time.time(), phase=phase, **kwargs)
    tmp = OUT / "status.tmp"
    tmp.write_text(json.dumps(row, indent=2) + "\n")
    tmp.replace(OUT / "status.json")
    print(json.dumps(row), flush=True)


def prepare():
    import torch

    verifier = module(
        "checkpoint_verifier", REPO / "docs/research/qwen21_aesthetic/continue_window2_to240.py"
    )
    evidence = verifier.verify_checkpoint(SOURCE, 30)
    saved = torch.load(SOURCE / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert all(g["lr"] == 1e-4 for g in saved["trainer"]["optimizer"]["param_groups"])
    OUT.mkdir(exist_ok=False)
    view = OUT / "original_lr_run_view"
    view.mkdir()
    (view / "checkpoint-final").symlink_to(SOURCE, target_is_directory=True)
    config = OLD / "train_to_60/resolved_config.yaml"
    (view / "resolved_config.yaml").write_bytes(config.read_bytes())
    manifests = {}
    for gpu in range(4):
        name = f"validation_gpu{gpu}.jsonl"
        assert (OLD / name).read_bytes() == (LOW / name).read_bytes()
        (OUT / name).write_bytes((OLD / name).read_bytes())
        manifests[name] = digest(OUT / name)
    protocol = dict(
        created_at=time.time(),
        original_checkpoint=evidence,
        original_lr=1e-4,
        lower_lr=3e-5,
        completed_round=30,
        optimizer_steps=60,
        original_config_sha256=digest(config),
        manifest_sha256=manifests,
        sampling=dict(
            width=1024,
            height=1024,
            steps=40,
            samples_per_prompt=1,
            seed=20260926,
            prompts=64,
            sampler="native",
        ),
        comparison="Original LR round30 versus lower LR round30, with base and round15 references.",
        limitation="Reused fixed validation set, one training seed; not a fresh confirmatory test or human aesthetic proof.",
        wait_unit=WAIT,
        wait_invocation=INVOCATION,
    )
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    status("prepared_waiting_for_lower_lr_evaluation")


def run():
    protocol = json.loads((OUT / "protocol.json").read_text())
    deadline = time.monotonic() + 20 * 3600
    status("waiting", unit=WAIT, invocation=INVOCATION)
    while True:
        s = state(WAIT)
        if s["ActiveState"] not in ("active", "activating", "deactivating"):
            assert (
                s["ActiveState"] == "inactive" and s["Result"] == "success" and s["MainPID"] == "0"
            ), s
            assert s["InvocationID"] in ("", INVOCATION), s
            break
        assert s["InvocationID"] == INVOCATION and s["MainPID"] == "886109", s
        if time.monotonic() > deadline:
            raise TimeoutError("Observer expired; existing jobs remain untouched")
        time.sleep(30)
    s = state("qwen21-aesthetic-lr3e5-from15.service")
    assert s["ActiveState"] == "inactive" and s["Result"] == "success" and s["MainPID"] == "0", s
    assert s["InvocationID"] in ("", "9088fbfa11eb4de197f85f20015a5522"), s
    done = json.loads((LOW / "independent_eval_complete.json").read_text())
    assert done["status"] == "complete_pending_quality_review"
    assert done["checkpoint"]["completed_round"] == 30
    assert digest(SOURCE / "checkpoint.pt") == protocol["original_checkpoint"]["sha256"]
    assert (
        digest(OUT / "original_lr_run_view/resolved_config.yaml")
        == protocol["original_config_sha256"]
    )
    for name, sha in protocol["manifest_sha256"].items():
        assert digest(OUT / name) == sha and digest(LOW / name) == sha
    busy = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    assert not busy, f"GPU process present; refusing overlap: {busy}"
    assert not (OUT / "eval_step30").exists(), "Existing control evaluation needs review"
    campaign = module("control_campaign", OLD / "run.py")
    campaign.ROOT = OUT
    campaign.status = status
    campaign.evaluate(OUT / "original_lr_run_view", 30, 0, build_gallery=False)
    integrity = module("control_integrity", OLD.parent / "fresh_test256_window15/evaluate.py")
    paired = []
    for gpu in range(4):
        prompts = [
            json.loads(x) for x in (OUT / f"validation_gpu{gpu}.jsonl").read_text().splitlines()
        ]
        assert len(prompts) == 16
        base_hashes = None
        for root, label in ((OUT, "original_lr30"), (LOW, "lower_lr30")):
            rows = [
                json.loads(x)
                for x in (root / f"eval_step30/gpu{gpu}/report/scores.jsonl")
                .read_text()
                .splitlines()
            ]
            integrity.validate_shard(rows, prompts, "lora30", 20260926 + gpu * 16)
            hashes = {
                r["prompt_index"]: r["image_sha256"]
                for r in rows
                if r["checkpoint_label"] == "base"
            }
            if base_hashes is None:
                base_hashes = hashes
            else:
                assert base_hashes == hashes, (
                    "Native base images differ; sampler comparison invalid"
                )
            for row in rows:
                if row["checkpoint_label"] == "base":
                    continue
                row["checkpoint_label"] = label
                row["prompt_index"] += gpu * 16
                paired.append(row)
    from vrl.scripts.eval.score_report import write_curve_report

    write_curve_report(
        paired,
        OUT / "aesthetic_lower_vs_original",
        base_label="original_lr30",
        bootstrap_resamples=10000,
        seed=260925,
    )
    (OUT / "paired_integrity.json").write_text(
        json.dumps(
            dict(
                valid=True,
                prompts=64,
                base_hashes_identical=True,
                original_checkpoint=protocol["original_checkpoint"],
                lower_checkpoint=done["checkpoint"],
            ),
            indent=2,
        )
        + "\n"
    )
    status("independent_pickscore")
    subprocess.run(
        [
            PYTHON,
            str(REPO / "docs/research/qwen21_aesthetic/cross_eval_pickscore.py"),
            "--root",
            str(OUT),
            "--steps",
            "30",
            "--output",
            str(OUT / "pickscore"),
            "--reuse",
            str(LOW / "pickscore_15_30"),
        ],
        cwd=REPO,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        check=True,
    )
    paired = []
    for path, label in (
        (OUT / "pickscore/raw_scores.jsonl", "original_lr30"),
        (LOW / "pickscore_15_30/raw_scores.jsonl", "lower_lr30"),
    ):
        selected = [
            json.loads(x)
            for x in path.read_text().splitlines()
            if json.loads(x)["checkpoint_label"] == "lora30"
        ]
        assert sorted(r["prompt_index"] for r in selected) == list(range(64))
        for row in selected:
            row["checkpoint_label"] = label
        paired.extend(selected)
    write_curve_report(
        paired,
        OUT / "pickscore_lower_vs_original",
        base_label="original_lr30",
        bootstrap_resamples=10000,
        seed=260925,
    )
    status("complete_pending_quality_review")


if __name__ == "__main__":
    sys.path.insert(0, str(REPO))
    os.environ.update(
        HF_HOME="/mnt/nvme/hf",
        HF_HUB_OFFLINE="1",
        PYTHONPATH=str(REPO),
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        TOKENIZERS_PARALLELISM="false",
    )
    if sys.argv[1:] == ["--prepare"]:
        prepare()
    elif sys.argv[1:] == ["--run"]:
        run()
    else:
        raise SystemExit("Use --prepare once, then --run once")
