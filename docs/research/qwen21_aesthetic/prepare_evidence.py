"""Prepare supplementary toolkit evidence after the blind labels are locked."""

import hashlib
import json
import statistics
from pathlib import Path

root = Path("outputs/qwen21_aesthetic_qualification")
labels_path = root / "blind/labels.jsonl"
assert (
    hashlib.sha256(labels_path.read_bytes()).hexdigest()
    == (root / "blind/labels.sha256").read_text().strip()
)
keys = {
    r["id"]: r for r in map(json.loads, (root / "selection_key.jsonl").read_text().splitlines())
}
labels = [json.loads(line) for line in labels_path.read_text().splitlines()]
out = []
for label in labels:
    if label["appeal"] == "tie":
        continue
    for side, arm in keys[label["id"]]["arms"].items():
        out.append(
            {
                "sample_id": arm["sample_id"],
                "labels": {"pair_preferred": side == label["appeal"]},
                "note": label["note"],
                "judge": "GPT-6 blind visual review",
            }
        )
(root / "outcome_labels.jsonl").write_text("".join(json.dumps(row) + "\n" for row in out))
contrast = [
    {
        "name": "pair_preferred_vs_rejected_pooled_diagnostic",
        "positive": {"pair_preferred": [True]},
        "negative": {"pair_preferred": [False]},
    }
]
(root / "contrasts.json").write_text(json.dumps(contrast, indent=2) + "\n")
stress = json.loads((root / "stress.json").read_text())
stress["transforms"] = {
    k: v
    for k, v in stress["transforms"].items()
    if k in ["blur_rgb", "noise_rgb", "black_rgb", "checkerboard_rgb"]
}
stress["scope_note"] = (
    "RGB task. Opaque-alpha transform is an identity control, not damage. Alpha probes are preserved in stress.json."
)
(root / "stress_rgb.json").write_text(json.dumps(stress, indent=2) + "\n")
groups = {}
for path in (root / "scores/samples").glob("*.json"):
    row = json.loads(path.read_text())
    assert row["status"] == "success"
    groups.setdefault(row["input"]["prompt_id"], []).append(row["result"]["scores"]["aesthetic"])
assert len(groups) == 64 and all(len(v) == 8 for v in groups.values())
spread = {
    "sampler": "512x512, 10-step Flow-GRPO SDE, noise 0.7, 8 seeds per prompt",
    "prompt_count": len(groups),
    "zero_spread_prompts": sum(max(v) == min(v) for v in groups.values()),
    "mean_within_prompt_std": statistics.mean(statistics.pstdev(v) for v in groups.values()),
    "mean_within_prompt_range": statistics.mean(max(v) - min(v) for v in groups.values()),
    "note": "Continuous spread measured directly. No calibrated binary aesthetic-success threshold exists; no fabricated success-band gate.",
    "per_prompt": {
        k: {"std": statistics.pstdev(v), "range": max(v) - min(v)}
        for k, v in sorted(groups.items())
    },
}
(root / "continuous_spread.json").write_text(json.dumps(spread, indent=2) + "\n")
