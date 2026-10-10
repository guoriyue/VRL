import argparse
import json
import math
import time
from pathlib import Path

import torch
from PIL import Image

from vrl.rewards.models.aesthetic import AestheticRewardModel

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=Path, default=Path("/mnt/nvme/work/smoke.jpg"))
parser.add_argument("--out", type=Path, default=Path("outputs/qwen21_aesthetic_qualification"))
args = parser.parse_args()
torch.set_num_threads(4)
out = args.out
out.mkdir(parents=True, exist_ok=True)
config = {
    "device": "cuda:0",
    "dtype": "float32",
    "model_revision": "9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
}
model = AestheticRewardModel(config)
source = args.image
image = Image.open(source).convert("RGB")
start = time.monotonic()
scores = [model.score_media(media=image, prompt="")["aesthetic"] for _ in range(3)]
report = {
    "config": config,
    "image": str(source),
    "scores": scores,
    "finite": all(math.isfinite(s) for s in scores),
    "max_range": max(scores) - min(scores),
    "elapsed_s": time.monotonic() - start,
}
report["pass"] = report["finite"] and report["max_range"] <= 1e-6
(out / "scorer_smoke.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
assert report["pass"]
