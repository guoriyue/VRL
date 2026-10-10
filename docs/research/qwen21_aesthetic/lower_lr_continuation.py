"""Auditable lower-LR continuation, preserving weights, Adam moments and RNG."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path("/home/ubuntu/VRL")
OLD = Path("/mnt/nvme/outputs/qwen21_aesthetic_lora/window2_trial")
ROOT = OLD.parent / "window2_lr3e5_from15"
SOURCE = OLD / "train_to_15/checkpoint-final"
LR = 3e-5
TARGET = 30


def equal(a, b):
    assert type(a) is type(b), (type(a), type(b))
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        import numpy as np

        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b)
        else:
            assert a == b


def prepare():
    ROOT.mkdir(exist_ok=False)
    derived = ROOT / "resume15_lr3e5"
    shutil.copytree(SOURCE, derived)
    source_file = SOURCE / "checkpoint.pt"
    state = torch.load(source_file, map_location="cpu", weights_only=False)
    assert state["progress"]["completed_epoch"] == 15
    assert state["progress"]["global_step"] == 30
    groups = state["trainer"]["optimizer"]["param_groups"]
    original_lrs = [g["lr"] for g in groups]
    assert all(x == 1e-4 for x in original_lrs)
    for g in groups:
        g["lr"] = LR
    torch.save(state, derived / "checkpoint.pt")
    restored = torch.load(derived / "checkpoint.pt", map_location="cpu", weights_only=False)
    equal(state, restored)
    for g, lr in zip(restored["trainer"]["optimizer"]["param_groups"], original_lrs):
        g["lr"] = lr
    equal(torch.load(source_file, map_location="cpu", weights_only=False), restored)
    meta = json.loads((derived / "checkpoint_meta.json").read_text())
    meta["checkpoint_file_bytes"] = (derived / "checkpoint.pt").stat().st_size
    (derived / "checkpoint_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    for gpu in range(4):
        shutil.copy2(OLD / f"validation_gpu{gpu}.jsonl", ROOT)
    original = json.loads((OLD / "experiment.json").read_text())["overrides"]
    overrides = [
        v
        for v in original
        if not v.startswith(
            (
                "trainer.total_epochs=",
                "trainer.output_dir=",
                "trainer.resume_from=",
                "actor.optim.lr=",
            )
        )
    ]
    overrides += [
        f"actor.optim.lr={LR}",
        f"trainer.total_epochs={TARGET}",
        f"trainer.output_dir={ROOT / 'train_to_30'}",
        f"trainer.resume_from={derived}",
    ]
    protocol = dict(
        source=str(SOURCE),
        source_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
        derived_sha256=hashlib.sha256((derived / "checkpoint.pt").read_bytes()).hexdigest(),
        only_checkpoint_change="optimizer.param_groups[*].lr: 0.0001 -> 0.00003; full recursive equality verified after undoing only that change",
        hypothesis="Smaller updates may preserve early independent preference gains; aesthetic benefit unproven.",
        start_round=15,
        target_round=TARGET,
        additional_optimizer_steps=30,
        validation="All original 64 fixed prompts and seeds, native 40-step 1024px; independent PickScore and paired comparison with round15. Reused validation, not fresh test.",
        overrides=overrides,
    )
    (ROOT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(protocol), flush=True)


def run():
    sys.path.insert(0, str(REPO))
    os.environ.update(
        HF_HOME="/mnt/nvme/hf",
        HF_HUB_OFFLINE="1",
        PYTHONPATH=str(REPO),
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        TOKENIZERS_PARALLELISM="false",
    )
    protocol = json.loads((ROOT / "protocol.json").read_text())
    out = ROOT / "train_to_30"
    out.mkdir(exist_ok=False)
    with (out / "service.log").open("x") as log:
        subprocess.run(
            [str(OLD.parent / "launch.sh"), *protocol["overrides"]],
            cwd=REPO,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    spec = importlib.util.spec_from_file_location("campaign", OLD / "run.py")
    campaign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campaign)
    campaign.ROOT = ROOT
    campaign.require_success(out)
    campaign.evaluate(out, TARGET, 0, build_gallery=False)
    campaign.status(
        "native_evaluation_complete_pending_independent_pickscore",
        checkpoint=str(out / "checkpoint-final"),
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["--prepare"]:
        prepare()
    elif sys.argv[1:] == ["--run"]:
        run()
    else:
        raise SystemExit("Use --prepare once, then --run")
