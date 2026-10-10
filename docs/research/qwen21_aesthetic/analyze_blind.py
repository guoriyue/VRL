"""Unblind only after independent labels have been saved and hashed."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, default=Path("outputs/qwen21_aesthetic_qualification"))
a = p.parse_args()
root = a.root
label_path = root / "blind" / "labels.jsonl"
labels = {r["id"]: r for r in map(json.loads, label_path.read_text().splitlines())}
assert len(labels) == 64
# Persist the label digest before loading selection identities into analysis.
lock = root / "blind" / "labels.sha256"
digest = hashlib.sha256(label_path.read_bytes()).hexdigest()
if lock.exists():
    assert lock.read_text().strip() == digest
else:
    lock.write_text(digest + "\n")
keys = {
    r["id"]: r for r in map(json.loads, (root / "selection_key.jsonl").read_text().splitlines())
}
assert len(keys) == 64 and labels.keys() == keys.keys()
rows = []
for pid, key in sorted(keys.items()):
    label = labels[pid]
    winner = label["appeal"]
    assert winner in ["A", "B", "tie"]
    selected = next(s for s, v in key["arms"].items() if v["arm"] == "selected")
    other = "B" if selected == "A" else "A"
    assert all(label[f"adherence_{s}"] in [0, 1, 2, 3] for s in ["A", "B"])
    if key["identical"]:
        assert winner == "tie" and label["adherence_A"] == label["adherence_B"], (
            "Identical images must tie"
        )
    rows.append(
        {
            "id": pid,
            "utility": 0.5 if winner == "tie" else float(winner == selected),
            "adherence_delta": label[f"adherence_{selected}"] - label[f"adherence_{other}"],
            "identical": key["identical"],
            "selected_side": selected,
            **label,
        }
    )
rng = np.random.default_rng(2026092504)
indices = rng.integers(0, 64, size=(10000, 64))
report = {
    "labels_sha256": digest,
    "judge": "GPT-6 assistant visual review, blinded to reward and selection identity; not human labels",
    "n": 64,
}
for field in ["utility", "adherence_delta"]:
    values = np.array([r[field] for r in rows])
    boot = values[indices].mean(axis=1)
    report[field] = {
        "mean": float(values.mean()),
        "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
    }
report["wins"] = sum(r["utility"] == 1 for r in rows)
report["ties"] = sum(r["utility"] == 0.5 for r in rows)
report["losses"] = sum(r["utility"] == 0 for r in rows)
report["identical_selections"] = sum(r["identical"] for r in rows)
report["preference_gate"] = report["utility"]["ci95"][0] > 0.5
report["adherence_gate"] = report["adherence_delta"]["ci95"][0] >= -0.15
report["user_training_gate_pass"] = report["preference_gate"] and report["adherence_gate"]
report["categories"] = {}
for category in range(8):
    group = rows[category * 8 : (category + 1) * 8]
    report["categories"][str(category)] = {
        f: float(np.mean([r[f] for r in group])) for f in ["utility", "adherence_delta"]
    }
(root / "blind_verdict.json").write_text(json.dumps(report, indent=2) + "\n")
(root / "unblinded_labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
print(json.dumps(report, indent=2))
