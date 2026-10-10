"""Decide at the 1000-update checkpoint whether the weighted arm continues.

Compares the quick held-out means (6 scenes, one seed) of the 500- and
1000-update checkpoints with the parent on object match and inferable
completion; the arm continues only when the better checkpoint beats the parent
by more than 0.02 on at least one of them. The rule was fixed before training.
"""

import json
import sys
from pathlib import Path

parent, c500, c1000, out = (Path(p) for p in sys.argv[1:5])
means = {
    name: json.loads((path / "scores.json").read_text())["means"]
    for name, path in (("parent", parent), ("500", c500), ("1000", c1000))
}
keys = ("layered_object_match", "layered_inferable_completion")
best = max(("500", "1000"), key=lambda n: sum(means[n][k] for k in keys))
gains = {k: means[best][k] - means["parent"][k] for k in keys}
decision = {
    "rule": "continue if max(ckpt500, ckpt1000) beats parent by > 0.02 on object match or inferable completion (6 held-out scenes, seed 4401)",
    "means": means,
    "best": best,
    "gains": gains,
    "continue": any(g > 0.02 for g in gains.values()),
}
out.write_text(json.dumps(decision, indent=2) + "\n")
print(json.dumps(decision))
sys.exit(0 if decision["continue"] else 3)
