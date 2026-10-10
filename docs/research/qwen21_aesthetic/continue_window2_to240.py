"""Resume the same window2 lineage to 240 optimizer updates, after queued evaluation."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/ubuntu/VRL")
ROOT = Path("/mnt/nvme/outputs/qwen21_aesthetic_lora/window2_trial")
OUT = ROOT / "extension240"
PYTHON = "/mnt/nvme/venv-main/bin/python"
WAIT_UNIT = "qwen21-fresh256-window15.service"
WAIT_INVOCATION = "e4f6d5279f37473083f8f1b6ee2a67e5"
MILESTONES = (60, 120)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def status(phase, **details):
    data = dict(time=time.time(), phase=phase, **details)
    tmp = OUT / "status.tmp"
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(OUT / "status.json")
    print(json.dumps(data), flush=True)


def overrides(source, target):
    original = json.loads((ROOT / "experiment.json").read_text())["overrides"]
    retained = [
        v
        for v in original
        if not v.startswith(
            ("trainer.total_epochs=", "trainer.output_dir=", "trainer.resume_from=")
        )
    ]
    return retained + [
        f"trainer.output_dir={ROOT / f'train_to_{target}'}",
        f"trainer.total_epochs={target}",
        f"trainer.resume_from={source}",
    ]


def verify_checkpoint(path, completed):
    meta = json.loads((path / "checkpoint_meta.json").read_text())
    assert meta["completed_epoch"] == completed and meta["global_step"] == completed * 2, meta
    payload = path / "checkpoint.pt"
    assert payload.stat().st_size == meta["checkpoint_file_bytes"]
    import torch
    from safetensors.torch import load_file

    adapter = load_file(str(path / "lora_weights/adapter_model.safetensors"))
    assert sum(t.numel() for t in adapter.values()) == 33554432
    assert all(torch.isfinite(t).all().item() for t in adapter.values())
    return dict(
        checkpoint=str(path),
        sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),
        completed_round=completed,
        optimizer_steps=completed * 2,
    )


def wait_for_evaluation():
    deadline = time.monotonic() + 12 * 3600
    status("waiting_for_round15_evaluation", unit=WAIT_UNIT, invocation=WAIT_INVOCATION)
    while True:
        raw = subprocess.check_output(
            [
                "systemctl",
                "--user",
                "show",
                WAIT_UNIT,
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
        state = dict(line.split("=", 1) for line in raw.splitlines())
        if state["ActiveState"] not in ("active", "activating", "deactivating"):
            # Successful inactive units may clear InvocationID in systemd.
            assert state["ActiveState"] == "inactive" and state["Result"] == "success", state
            assert state["MainPID"] == "0", state
            assert state["InvocationID"] in ("", WAIT_INVOCATION), state
            break
        assert state["InvocationID"] == WAIT_INVOCATION, state
        assert state["MainPID"] == "518522", state
        if time.monotonic() > deadline:
            raise TimeoutError("Observer deadline; existing work remains untouched")
        time.sleep(45)
    prior = json.loads((ROOT.parent / "fresh_test256_window15/campaign_status.json").read_text())
    assert prior["phase"] == "complete_pending_quality_review", prior
    assert prior["checkpoint"] == str(ROOT / "train_to_15/checkpoint-final"), prior
    test_root = ROOT.parent / "fresh_test256_window15"
    for report, metric in [
        ("eval_step15/aggregate/summary.json", "aesthetic"),
        ("pickscore/vs_base/summary.json", "pickscore"),
    ]:
        result = json.loads((test_root / report).read_text())
        assert result["arms"]["lora15"]["scores"][metric]["paired_delta_from_base"]["count"] == 256


def main():
    sys.path.insert(0, str(REPO))
    os.environ.update(
        HF_HOME="/mnt/nvme/hf",
        HF_HUB_OFFLINE="1",
        PYTHONPATH=str(REPO),
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        TOKENIZERS_PARALLELISM="false",
    )
    OUT.mkdir(exist_ok=True)
    campaign = load_module("window_campaign", ROOT / "run.py")
    campaign.status = status
    integrity = load_module("pair_integrity", ROOT.parent / "fresh_test256_window15/evaluate.py")
    if "--check-only" in sys.argv:
        for start, target in ((15, 60), (60, 120)):
            args = overrides(ROOT / f"train_to_{start}/checkpoint-final", target)
            assert len(args) == len(set(x.split("=", 1)[0] for x in args))
            assert f"trainer.total_epochs={target}" in args
            assert "actor.ppo_epochs=2" in args
            print(
                json.dumps(
                    dict(start=start, target=target, optimizer_steps=target * 2, overrides=args)
                )
            )
        return
    wait_for_evaluation()
    source = ROOT / "train_to_15/checkpoint-final"
    start = 15
    for target in MILESTONES:
        campaign.require_success(source.parent)
        evidence = verify_checkpoint(source, start)
        run = ROOT / f"train_to_{target}"
        run.mkdir(exist_ok=False)
        args = overrides(source, target)
        (run / "continuation_protocol.json").write_text(
            json.dumps(
                dict(
                    source=evidence,
                    target_round=target,
                    target_optimizer_steps=target * 2,
                    initialization="Full checkpoint including optimizer and RNG; no base reset",
                    overrides=args,
                    validation="Existing 64 native prompts, then independent CPU PickScore",
                    holdout="Fresh256 belongs only to preselected round15; not reused as a fresh test",
                ),
                indent=2,
            )
            + "\n"
        )
        status("training", source=str(source), start_round=start, target_round=target)
        with (run / "service.log").open("x") as log:
            subprocess.run(
                [str(ROOT.parent / "launch.sh"), *args],
                cwd=REPO,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        campaign.require_success(run)
        verify_checkpoint(run / "checkpoint-final", target)
        assert not (ROOT / f"eval_step{target}").exists(), "Existing evaluation requires review"
        campaign.evaluate(run, target, 0, build_gallery=False)
        for gpu in range(4):
            scores = ROOT / f"eval_step{target}/gpu{gpu}/report/scores.jsonl"
            rows = [json.loads(s) for s in scores.read_text().splitlines()]
            prompts = [
                json.loads(s)
                for s in (ROOT / f"validation_gpu{gpu}.jsonl").read_text().splitlines()
            ]
            integrity.validate_shard(rows, prompts, f"lora{target}", 20260926 + gpu * 16)
        status("independent_pickscore", round=target)
        subprocess.run(
            [
                PYTHON,
                str(REPO / "docs/research/qwen21_aesthetic/cross_eval_pickscore.py"),
                "--root",
                str(ROOT),
                "--steps",
                "15",
                str(target),
                "--output",
                str(ROOT / f"pickscore_15_{target}"),
                "--reuse",
                str(ROOT / "pickscore_5_15"),
            ],
            cwd=REPO,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
            check=True,
        )
        source, start = run / "checkpoint-final", target
    status("complete_pending_quality_review", checkpoint=str(source), optimizer_steps=240)


if __name__ == "__main__":
    main()
