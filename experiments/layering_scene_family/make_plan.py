"""Rewrite the probe plan's machine-specific paths for a new host.

The plan pins the Qwen-Image-Layered snapshot, the official Stable-Layers adapter
revision, the upstream ``decompose.py`` (by SHA-256) and the LoRA layout. Only two
fields hold local paths; everything the trainer checks (base revision, upstream hash)
is kept, so caches built on another machine stay valid.
"""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="plan.json from the data bundle")
    parser.add_argument(
        "--base-path",
        type=Path,
        required=True,
        help="local Qwen/Qwen-Image-Layered snapshot at the plan's base_revision",
    )
    parser.add_argument(
        "--upstream", type=Path, required=True, help="decompose.py from the data bundle"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    digest = hashlib.sha256(args.upstream.read_bytes()).hexdigest()
    assert digest == plan["upstream_sha256"], f"decompose.py hash {digest} does not match the plan"
    assert (args.base_path / "model_index.json").exists(), (
        f"{args.base_path} is not a diffusers snapshot"
    )
    plan["base_path"], plan["upstream"] = (
        str(args.base_path.resolve()),
        str(args.upstream.resolve()),
    )
    args.out.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "base_revision": plan["base_revision"]}))


if __name__ == "__main__":
    main()
