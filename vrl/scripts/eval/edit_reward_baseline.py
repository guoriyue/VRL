"""Score the same natural editing candidates with unmodified upstream rewards."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward", choices=["editscore", "editreward"], required=True)
    parser.add_argument("--out", default="outputs/qwen_image_21_reward_study")
    parser.add_argument("--upstream-root", default="/tmp/vrl-edit-research")
    parser.add_argument("--split", choices=["development", "heldout"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--judge-seed", type=int, default=42)
    parser.add_argument("--view", choices=["full", "task_crop"], default="full")
    args = parser.parse_args()
    torch.set_num_threads(8)
    out = Path(args.out).resolve()
    root = Path(args.upstream_root)
    sys.path[:0] = [str(root / "EditScore"), str(root / "EditReward")]
    report = json.loads((out / "report.json").read_text())
    cases = list(report["cases"].values())
    if args.split:
        cases = [case for case in cases if case["split"] == args.split]
    if args.limit:
        cases = cases[: args.limit]
    rules = (
        json.loads((out / "region_checks.json").read_text())["tasks"]
        if args.view == "task_crop"
        else {}
    )
    if args.view == "task_crop":
        cases = [
            case
            for case in cases
            if "review_crop" in rules.get(case.get("evaluation_task_id", case["task_id"]), {})
        ]
    suffix = "" if args.view == "full" else "_task_crop"
    score_path = out / f"{args.reward}{suffix}_seed{args.judge_seed}.jsonl"
    completed = (
        {json.loads(line)["name"] for line in score_path.read_text().splitlines()}
        if score_path.exists()
        else set()
    )
    cases = [case for case in cases if case["name"] not in completed]
    if not cases:
        return
    from huggingface_hub import snapshot_download

    base_revision = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
    base_model = snapshot_download(
        "Qwen/Qwen2.5-VL-7B-Instruct", revision=base_revision, local_files_only=True
    )
    if args.reward == "editscore":
        from editscore import EditScore

        adapter = snapshot_download(
            "EditScore/EditScore-7B",
            revision="dcce052accfc75f2d48279c889a71755ecdf737b",
            local_files_only=True,
        )
        scorer = EditScore(
            backbone="qwen25vl",
            model_name_or_path=base_model,
            lora_path=adapter,
            score_range=25,
            temperature=0.7,
            num_pass=1,
            reduction="average_last",
            seed=args.judge_seed,
        )
        scorer.model.model.eval()
        model_id = "EditScore/EditScore-7B"
    else:
        import yaml
        from EditReward import EditRewardInferencer

        config = yaml.safe_load(
            (root / "EditReward/EditReward/config/EditReward-Qwen2.5-7B-VL.yaml").read_text()
        )
        config["output_dir"] = str(out / "reward_runtime")
        config["disable_flash_attn2"] = True
        config["model_name_or_path"] = base_model
        config_path = out / "editreward_inference.yaml"
        config_path.write_text(yaml.safe_dump(config))
        checkpoint = snapshot_download(
            "TIGER-Lab/EditReward-Qwen2.5-VL-7B",
            revision="51b92ab5246295637c4ab3bd71e54a26f0a5189d",
            local_files_only=True,
        )
        scorer = EditRewardInferencer(
            config_path=str(config_path),
            checkpoint_path=checkpoint,
            device="cuda",
            reward_dim="overall_detail",
            rm_head_type="ranknet_multi_head",
        )
        model_id = "TIGER-Lab/EditReward-Qwen2.5-VL-7B"
    provenance = {
        "model": model_id,
        "judge_seed": args.judge_seed,
        "upstream_commit": subprocess.check_output(
            [
                "git",
                "-C",
                str(root / ("EditScore" if args.reward == "editscore" else "EditReward")),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip(),
        "torch": torch.__version__,
        "device": "cuda",
        "baseline": args.view == "full",
        "view": args.view,
        "base_model_revision": base_revision,
        "settings": {
            "editscore": {
                "score_range": 25,
                "temperature": 0.7,
                "num_pass": 1,
                "reduction": "average_last",
            },
            "editreward": {
                "pooling_strategy": "mean",
                "attention": "sdpa",
                "image_pixels": 200704,
            },
        }[args.reward],
    }
    import transformers

    provenance["transformers"] = transformers.__version__
    (out / f"{args.reward}{suffix}_seed{args.judge_seed}_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    inputs = out / "reward_inputs"
    inputs.mkdir(exist_ok=True)
    with torch.inference_mode():
        for case in cases:
            print(f"START {args.reward} {case['name']}", flush=True)
            started = time.monotonic()
            src_path = out / f"{Path(case['source']).stem}_input.png"
            if not src_path.exists():
                src_path = out / case["source"]
            with Image.open(src_path) as original:
                rgba = original.convert("RGBA")
                source = Image.alpha_composite(
                    Image.new("RGBA", rgba.size, "white"), rgba
                ).convert("RGB")
            with Image.open(out / case["output"]) as im:
                rgba = im.convert("RGBA")
                edited = Image.alpha_composite(
                    Image.new("RGBA", rgba.size, "white"), rgba
                ).convert("RGB")
            source = source.resize(edited.size, Image.Resampling.LANCZOS)
            judge_prompt = case.get("evaluation_prompt", case["prompt"])
            if args.view == "task_crop":
                rule = rules[case.get("evaluation_task_id", case["task_id"])]
                w, h = source.size
                box = tuple(
                    round(v * n) for v, n in zip(rule["review_crop"], [w, h, w, h], strict=True)
                )
                source, edited = source.crop(box), edited.crop(box)
                judge_prompt = rule["crop_instruction"]
            if args.reward == "editscore":
                raw = scorer.evaluate([source, edited], judge_prompt)
                raw = {
                    k: float(v) if isinstance(v, (float, int)) or hasattr(v, "item") else v
                    for k, v in raw.items()
                }
                score = float(raw["overall"])
            else:
                target_path = inputs / f"{case['name']}{suffix}.png"
                edited.save(target_path)
                src_path = inputs / f"{case['name']}{suffix}_source.png"
                source.save(src_path)
                values = (
                    scorer.reward(
                        prompts=[judge_prompt],
                        image_src=[str(src_path)],
                        image_paths=[str(target_path)],
                    )[0]
                    .float()
                    .cpu()
                    .tolist()
                )
                raw = {"mean": values[0], "log_sigma": values[1], "pooling_strategy": "mean"}
                score = float(values[0])
            if not math.isfinite(score):
                raise ValueError(f"Non-finite reward for {case['name']}: {raw}")
            row = {
                "name": case["name"],
                "task_id": case["task_id"],
                "split": case["split"],
                "score": score,
                "details": raw,
                "seconds": time.monotonic() - started,
            }
            with score_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(
                f"DONE {args.reward} {case['name']} score={score:.4f} seconds={row['seconds']:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
