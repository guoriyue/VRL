"""User-requested native evaluation at round43, then full-state continuation."""

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/ubuntu/VRL")
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location(
    "extension", REPO / "docs/research/qwen21_aesthetic/continue_window2_to240.py"
)
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)
ROOT = ext.ROOT
OUT = ROOT / "early_eval43"


def status(phase, **details):
    data = dict(time=time.time(), phase=phase, **details)
    tmp = OUT / "status.tmp"
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(OUT / "status.json")
    print(json.dumps(data), flush=True)


def evaluate(campaign, integrity, run, step):
    assert not (ROOT / f"eval_step{step}").exists()
    campaign.evaluate(run, step, 0, build_gallery=False)
    results = []
    for gpu in range(4):
        scores = ROOT / f"eval_step{step}/gpu{gpu}/report/scores.jsonl"
        rows = [json.loads(s) for s in scores.read_text().splitlines()]
        prompts = [
            json.loads(s) for s in (ROOT / f"validation_gpu{gpu}.jsonl").read_text().splitlines()
        ]
        integrity.validate_shard(rows, prompts, f"lora{step}", 20260926 + gpu * 16)
        results.append(dict(gpu=gpu, valid=True, pairs=len(prompts), images=len(rows)))
    (ROOT / f"eval_step{step}/pair_integrity.json").write_text(
        json.dumps(dict(shards=results), indent=2) + "\n"
    )
    status("independent_pickscore", round=step)
    subprocess.run(
        [
            ext.PYTHON,
            str(REPO / "docs/research/qwen21_aesthetic/cross_eval_pickscore.py"),
            "--root",
            str(ROOT),
            "--steps",
            "15",
            str(step),
            "--output",
            str(ROOT / f"pickscore_15_{step}"),
            "--reuse",
            str(ROOT / "pickscore_5_15"),
        ],
        cwd=REPO,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        check=True,
    )


def main():
    os.environ.update(
        HF_HOME="/mnt/nvme/hf",
        HF_HUB_OFFLINE="1",
        PYTHONPATH=str(REPO),
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        TOKENIZERS_PARALLELISM="false",
    )
    OUT.mkdir(exist_ok=True)
    old = subprocess.check_output(
        [
            "systemctl",
            "--user",
            "show",
            "qwen21-aesthetic-eval47-resume.service",
            "-p",
            "MainPID",
            "-p",
            "ActiveState",
        ],
        text=True,
    )
    fields = dict(line.split("=", 1) for line in old.splitlines())
    assert fields["MainPID"] == "0" and fields["ActiveState"] in ("inactive", "failed"), fields
    source = ROOT / "train_to_60/checkpoint-47"
    evidence = ext.verify_checkpoint(source, 47)
    assert (
        evidence["sha256"]
        == json.loads((ROOT / "checkpoint47_acceptance.json").read_text())["sha256"]
    )
    wrapper = OUT / "evaluation_run"
    wrapper.mkdir(exist_ok=False)
    (wrapper / "resolved_config.yaml").symlink_to(source.parent / "resolved_config.yaml")
    eval_source = ROOT / "train_to_60/checkpoint-43"
    ext.verify_checkpoint(eval_source, 43)
    (wrapper / "checkpoint-final").symlink_to(eval_source, target_is_directory=True)
    (OUT / "protocol.json").write_text(
        json.dumps(
            dict(
                source=evidence,
                reason="User requested evaluation now; uncheckpointed round48 interrupted",
                evaluation="Existing fixed64 native prompts, paired seeds, 1024x1024, 40 steps; aesthetic and CPU PickScore",
                interpretation="Repeated validation, not a new unseen test; compare base and round15",
                continuation="Full optimizer and RNG resume from47 to60 then120, no hyperparameter changes",
            ),
            indent=2,
        )
        + "\n"
    )
    campaign = ext.load_module("window_campaign", ROOT / "run.py")
    campaign.status = status
    integrity = ext.load_module(
        "pair_integrity", ROOT.parent / "fresh_test256_window15/evaluate.py"
    )
    evaluate(campaign, integrity, wrapper, 43)
    start = 47
    for target in (60, 120):
        evidence = ext.verify_checkpoint(source, start)
        run = ROOT / f"train_to_{target}_after_eval43"
        run.mkdir(exist_ok=False)
        args = [
            v for v in ext.overrides(source, target) if not v.startswith("trainer.output_dir=")
        ]
        args.append(f"trainer.output_dir={run}")
        assert len(args) == len(set(x.split("=", 1)[0] for x in args))
        (run / "continuation_protocol.json").write_text(
            json.dumps(
                dict(
                    source=evidence,
                    overrides=args,
                    target_round=target,
                    target_optimizer_steps=target * 2,
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
        ext.verify_checkpoint(run / "checkpoint-final", target)
        evaluate(campaign, integrity, run, target)
        source, start = run / "checkpoint-final", target
    status("complete_pending_quality_review", checkpoint=str(source), optimizer_steps=240)


if __name__ == "__main__":
    main()
