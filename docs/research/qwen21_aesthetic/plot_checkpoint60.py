"""Plot the completed 60-round line and fixed native validation checkpoints."""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/nvme/work/aesthetic_plot_deps")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path("/mnt/nvme/outputs/qwen21_aesthetic_lora/window2_trial")
output = Path("/mnt/nvme/work/qwen_aesthetic_figures/training_and_validation_round60.png")
rows = []
for run in ["train_to_5", "train_to_15", "train_to_60", "train_to_60_after_eval43"]:
    rows.extend(csv.DictReader((root / run / "metrics.full_precision.csv").open()))
rows = [row for row in rows if int(row["epoch"]) < 60]
x = np.array([int(row["epoch"]) + 1 for row in rows])
assert list(x) == list(range(1, 61)), "Missing or duplicate training rounds"
y = np.array([float(row["reward_mean"]) for row in rows])
assert np.isfinite(y).all()
steps = [5, 15, 43, 47, 60]
fig, axes = plt.subplots(2, 2, figsize=(14, 8), layout="constrained")
ax = axes[0, 0]
ax.plot(x, y, alpha=0.5, label="Per-round reward (different prompts)")
ax.plot(x[4:], np.convolve(y, np.ones(5) / 5, mode="valid"), label="5-round mean")
best = int(np.argmax(y))
ax.scatter([x[best]], [y[best]], color="red")
ax.set(
    title=f"Highest training reward: round {x[best]}",
    xlabel="Round",
    ylabel="Aesthetic predictor score",
)
ax.legend()
ax = axes[0, 1]
for key, label in [
    ("loss", "Total loss"),
    ("policy_loss", "Policy loss"),
    ("weighted_kl_loss", "Weighted KL"),
]:
    ax.plot(x, [float(row[key]) * 1e6 for row in rows], label=label)
ax.set(
    title="Loss diagnostics; not a quality ranking",
    xlabel="Round",
    ylabel="Logged loss x 1,000,000",
)
ax.legend()
for ax, metric in [(axes[1, 0], "aesthetic"), (axes[1, 1], "pickscore")]:
    for step in steps:
        if metric == "aesthetic":
            path = root / f"eval_step{step}/aggregate/summary.json"
        else:
            directory = "pickscore" if step == 5 else f"pickscore_15_{47 if step == 15 else step}"
            path = root / directory / "vs_base/summary.json"
        delta = json.loads(path.read_text())["arms"][f"lora{step}"]["scores"][metric][
            "paired_delta_from_base"
        ]
        mean = delta["mean"]
        lo, hi = delta["bootstrap_95ci"]
        ax.errorbar(step, mean, yerr=[[mean - lo], [hi - mean]], fmt="o", capsize=5)
    ax.axhline(0, color="grey", linestyle="--")
    ax.set(
        title=f"Fixed64 native validation: {metric}",
        xlabel="Checkpoint round",
        ylabel="Paired gain vs base; 95% bootstrap CI",
        xticks=steps,
        xlim=(0, 66),
    )
fig.suptitle(
    "Qwen-Image-2.1 LoRA RL | rounds 1-60\nRepeated validation set; score gains do not establish human preference",
    fontsize=15,
)
fig.savefig(output, dpi=150)
plt.close(fig)
print(output)
