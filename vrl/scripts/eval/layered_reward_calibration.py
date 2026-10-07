"""Calibrate a frozen visual judge on full layers and known failure controls.

Scores are provisional until controls and candidate rankings are reviewed.
This tool neither changes model layers nor performs an optimizer update.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from vrl.scripts.eval.layered_candidate_views import prepare_views


def prepare_controls(source, layers, directory):
    """Retain known-invalid decompositions for falsifying reward behavior."""
    empty = Image.new("RGBA", source.size)
    whole = source.convert("RGBA")
    controls = {
        "empty": [empty.copy() for _ in layers],
        "whole_frame": [whole] + [empty.copy() for _ in layers[1:]],
        "duplicates": [whole.copy() for _ in layers],
    }
    fragments = []
    for index in range(len(layers)):
        fragment = Image.new("RGBA", source.size)
        left = index * source.width // len(layers)
        right = (index + 1) * source.width // len(layers)
        fragment.paste(whole.crop((left, 0, right, source.height)), (left, 0))
        fragments.append(fragment)
    controls["vertical_fragments"] = fragments
    hole = [layer.copy() for layer in layers]
    # Remove underlying pixels behind the strongest upper foreground layer.
    # This preserves the original foreground but breaks exposed lower content.
    foreground = max(range(1, len(layers)), key=lambda i: np.array(layers[i])[..., 3].sum())
    covered = np.array(layers[foreground])[..., 3] > 127
    for index in range(foreground):
        pixels = np.array(hole[index])
        pixels[covered, 3] = 0
        hole[index] = Image.fromarray(pixels)
    controls["background_hole"] = hole
    rows = []
    for name, control in controls.items():
        target = directory / name
        prepare_views(source, control, target)
        rows.append({"kind": "control", "control": name, "view": str(target / "review.png")})
    return rows


def parse_scores(answer, weights):
    start, end = answer.find("{"), answer.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Judge did not return a JSON object")
    scores = json.loads(answer[start : end + 1])
    for key in weights:
        value = scores.get(key)
        if type(value) is not int or not 0 <= value <= 4:
            raise ValueError(f"Invalid judge score for {key}: {value!r}")
    if not isinstance(scores.get("evidence"), str) or not scores["evidence"].strip():
        raise ValueError("Missing visual evidence")
    reward = sum(scores[key] * weight for key, weight in weights.items())
    return scores, reward / (4 * sum(weights.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--rubric",
        type=Path,
        default=Path("vrl/config/presets/reward_rubrics/layered_editability.yaml"),
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root / "plan.json").read_text())
    rubric = yaml.safe_load(args.rubric.read_text())
    output = args.output_dir.resolve() if args.output_dir else root / "reward_calibration"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in plan["cases"]:
        if case["split"] != "train":
            continue
        source = Image.open(root / "conditions" / case["name"] / "source.png").convert("RGB")
        for candidate in sorted((root / "rollouts" / case["name"]).glob("candidate*")):
            paths = [candidate / f"layer_{index}.png" for index in range(plan["layers"])]
            if not all(path.is_file() for path in paths):
                continue
            layers = [Image.open(path).convert("RGBA") for path in paths]
            prepare_views(source, layers, candidate / "review")
            rows.append(
                {
                    "case": case["name"],
                    "candidate": candidate.name,
                    "kind": "candidate",
                    "view": str(candidate / "review/review.png"),
                }
            )
            if candidate.name == "candidate00":
                rows.extend(
                    {"case": case["name"], **row}
                    for row in prepare_controls(source, layers, output / case["name"])
                )
    if not rows:
        raise RuntimeError("No complete training candidates available")
    (output / "inputs.json").write_text(json.dumps(rows, indent=2))
    if args.prepare_only:
        print(json.dumps({"prepared": len(rows), "output": str(output)}), flush=True)
        return
    result_path = output / "scores.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    from transformers import AutoModelForImageTextToText, AutoProcessor

    record = {
        "model": str(args.model.resolve()),
        "max_new_tokens": args.max_new_tokens,
        "enable_thinking": False,
        "rubric_sha256": hashlib.sha256(args.rubric.read_bytes()).hexdigest(),
        "rubric": rubric,
        "status": "running",
        "validated_for_optimization": False,
        "scope": "Provisional frozen VLM scores, not human review or training success.",
        "rows": [],
    }
    with Path("/tmp/vrl-preservation-gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        model = (
            AutoModelForImageTextToText.from_pretrained(
                args.model, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
            )
            .eval()
            .to("cuda")
        )
        processor = AutoProcessor.from_pretrained(
            args.model, local_files_only=True, max_pixels=1280 * 1280
        )
        for row in rows:
            view = Image.open(row["view"]).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": rubric["prompt"]}],
                }
            ]
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = processor(text=[prompt], images=[view], return_tensors="pt").to("cuda")
            with torch.inference_mode():
                tokens = model.generate(
                    **inputs, do_sample=False, max_new_tokens=args.max_new_tokens
                )
            answer = processor.batch_decode(
                tokens[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
            )[0]
            result = {
                **row,
                "view_sha256": hashlib.sha256(Path(row["view"]).read_bytes()).hexdigest(),
                "answer": answer,
            }
            try:
                scores, reward = parse_scores(answer, rubric["weights"])
                result.update(scores=scores, reward=reward, valid=True)
            except (ValueError, TypeError) as error:
                result.update(valid=False, error=str(error))
            record["rows"].append(result)
            result_path.write_text(json.dumps(record, indent=2))
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in result.items()
                        if key not in ("answer", "scores")
                    }
                ),
                flush=True,
            )
        record["status"] = "complete"
        result_path.write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
