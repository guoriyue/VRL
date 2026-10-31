"""Prepare paired annotation arms with identical images and layer budgets."""

import copy
import hashlib
import json
import random
from pathlib import Path

ROOT = Path("/mnt/nvme/layering_dense_20261025")
PREVIOUS = Path("/mnt/nvme/layering_natural_20261025")


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def main():
    dense = read_rows(PREVIOUS / "dense_coco/training_expanded.jsonl")
    old = {
        row["metadata"]["coco_id"]: row
        for row in read_rows(Path("data/external/coco/layered_val2017/train_mixed_v3.jsonl"))
        if row["metadata"].get("source") == "coco_val2017"
    }
    natural = read_rows(PREVIOUS / "data/train.jsonl")
    heldout = read_rows(PREVIOUS / "dense_coco/heldout_expanded.jsonl")
    assert not {r["metadata"]["coco_id"] for r in dense} & {
        r["metadata"]["coco_id"] for r in heldout
    }
    random.Random(20261025).shuffle(dense)
    arms = {"expanded": [], "legacy": []}
    for index, row in enumerate(dense):
        legacy = copy.deepcopy(row)
        original = old[row["metadata"]["coco_id"]]
        assert original["reference_image"] == row["reference_image"]
        legacy["metadata"] = {
            **original["metadata"],
            "complete_layer_truth": False,
            "annotation_version": "legacy-1.5-percent-filter",
        }
        arms["expanded"].append(row)
        arms["legacy"].append(legacy)
        if index % 3 == 2:
            for rows in arms.values():
                rows.append(copy.deepcopy(natural[(index // 3) % len(natural)]))
    for arm, rows in arms.items():
        for row in rows:
            assert Path(row["reference_image"]).is_file()
            assert Path(row["metadata"]["instances"]).is_file()
            assert row["request_overrides"]["layers"] <= 11
        write_rows(ROOT / arm / "train.jsonl", rows)
    for a, b in zip(arms["expanded"], arms["legacy"], strict=True):
        assert a["reference_image"] == b["reference_image"]
        assert a["request_overrides"] == b["request_overrides"]
        assert a["prompt"] == b["prompt"]
    evaluation = read_rows(PREVIOUS / "eval_rows.jsonl")
    write_rows(ROOT / "eval_rows.jsonl", evaluation)
    lookup = {row["id"]: row for row in heldout}
    write_rows(ROOT / "eval_expanded.jsonl", [lookup.get(r["id"], r) for r in evaluation])
    (ROOT / "reward_kwargs.json").write_text((PREVIOUS / "reward_kwargs.json").read_text())
    protocol = {
        "method": "pure GRPO, paired annotation ablation; no SFT",
        "start": "/mnt/nvme/layering/checkpoint-28-fsdp4-localcuda",
        "optimizer": "identical inherited Adam state; reset comparison remains separate",
        "arms": ["expanded", "legacy"],
        "only_between_arm_change": "COCO visible instance annotations",
        "coco_unique_photos": len(dense),
        "controlled_unique_variants": len(natural),
        "rows_per_arm": len(arms["expanded"]),
        "sampling": "sequential_window, three COCO rows then one controlled row; fixed shuffle seed 20261025",
        "initial_updates_per_arm": 8,
        "checkpoints": [32, 36],
        "samples_per_prompt": 4,
        "global_prompts_per_update": 4,
        "evaluation_seed": 4401,
        "training_rng": "restored from identical checkpoint-28 RNG states in both arms",
        "evaluation": "Identical 34 scenes at original layer budgets; rescore both original and expanded masks",
        "stop": "Stop an arm on >=3 fewer original complete COCO objects or >=3 additional merged layers",
        "promotion": "No automatic promotion. Require expanded object gain, no split/merge/removal regression, visual review, capacity-aware evaluation and independent seeds.",
        "limitations": [
            "Initial screen sees only 24 COCO prompt draws per arm, not an epoch over 2410 images",
            "Controlled full truth still covers only four training source images",
            "Most expanded held-out annotations exceed original layer budgets",
            "SAM augmentation, wider full-object data and Adam-reset control remain outstanding",
        ],
        "scheduled_rows": [
            {
                "step": epoch + 1,
                "rows": [
                    {
                        "reference_image": row["reference_image"],
                        "layers": row["request_overrides"]["layers"],
                    }
                    for row in arms["expanded"][epoch * 4 : epoch * 4 + 4]
                ],
            }
            for epoch in range(28, 36)
        ],
    }
    for path in ROOT.glob("*/train.jsonl"):
        protocol.setdefault("manifest_sha256", {})[str(path)] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    (ROOT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
