"""Generate the randomized scene family: 200 training scenes and 24 held-out scenes.

Each scene must carry at least one inferable and one unknowable occlusion
(>= 150 px each); seeds whose render lacks either are recorded as rejected and
the next seed is tried. Held-out scenes avoid every object-class combination
present in the training split and draw from the reserved parameter ranges.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BLENDER = "/opt/blender/blender"


def render(seed, split, out, avoid=None, samples=16, renderer="render_scene_family.py"):
    cmd = [
        BLENDER,
        "--background",
        "--python",
        str(HERE / renderer),
        "--",
        "--out",
        str(out),
        "--seed",
        str(seed),
        "--split",
        split,
        "--samples",
        str(samples),
        "--device",
        "GPU",
    ]
    if avoid:
        cmd += ["--avoid-combos", str(avoid)]
    started = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    ok = proc.returncode == 0 and (out / "occlusion.json").exists()
    return ok, time.perf_counter() - started, proc.stdout[-2000:] + proc.stderr[-2000:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE / "dataset")
    parser.add_argument("--train", type=int, default=200)
    parser.add_argument("--heldout", type=int, default=24)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--renderer", default="render_scene_family.py")
    parser.add_argument("--train-seed0", type=int, default=1000)
    parser.add_argument("--heldout-seed0", type=int, default=5000)
    args = parser.parse_args()
    root = args.root
    card = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "samples": args.samples,
        "resolution": 640,
        "splits": {},
        "rejected": {},
        "reserved_heldout_ranges": {
            "leg_style": "tapered cylinder legs (train: box, cylinder)",
            "cabinet_doors": "double doors with door_ratio 0.6-0.8 (train: single drawer handle)",
            "lamp_shade": "cylinder shade (train: cone)",
            "bookshelf_shelves": "4-5 shelves (train: 2-3)",
            "wall_hue": "0.55-0.65 (train: 0.05-0.2, 0.3-0.5, 0.7-0.95)",
            "chair_hue": "0.0-0.1 (train: 0.5-0.75)",
            "note": "each reserved range is drawn with probability 0.7 per held-out object; scene-level object-class combinations seen in training are excluded from the held-out split",
        },
        "occlusion_rule": None,
        "object_classes": [
            "cabinet",
            "table",
            "chair",
            "lamp",
            "plant",
            "crate",
            "bookshelf",
            "sofa",
            "stool",
        ]
        + (
            [
                "bed",
                "wardrobe",
                "desk",
                "armchair",
                "tv_stand",
                "side_table",
                "rug",
                "pendant",
                "painting",
            ]
            if "v2" in args.renderer
            else []
        ),
        "objects_per_scene": "2-7" if "v2" in args.renderer else "2-6",
    }
    train_combos = set()
    card["renderer"] = args.renderer
    for split, count, seed0 in (
        ("train", args.train, args.train_seed0),
        ("heldout", args.heldout, args.heldout_seed0),
    ):
        avoid_path = None
        if split == "heldout":
            avoid_path = root / "train_combos.json"
            avoid_path.write_text(json.dumps(sorted(train_combos)))
        accepted, rejected, seed = [], [], seed0
        while len(accepted) < count:
            out = root / split / f"scene_{seed}"
            ok, seconds, log = render(seed, split, out, avoid_path, args.samples, args.renderer)
            if ok:
                occ = json.loads((out / "occlusion.json").read_text())
                scene = json.loads((out / "scene.json").read_text())
                combo = tuple(sorted(o["class"] for o in scene["objects"]))
                if (
                    occ["has_inferable_occlusion"]
                    and occ["has_unknowable_occlusion"]
                    and (split == "train" or combo not in train_combos)
                ):
                    accepted.append(
                        {
                            "seed": seed,
                            "dir": str(out.relative_to(root)),
                            "objects": list(combo),
                            "inferable_pixels": occ["inferable_pixels"],
                            "unknowable_pixels": occ["unknowable_pixels"],
                            "seconds": round(seconds, 1),
                        }
                    )
                    if split == "train":
                        train_combos.add(combo)
                    card["occlusion_rule"] = occ["rule"]
                else:
                    rejected.append(
                        {
                            "seed": seed,
                            "reason": "missing inferable/unknowable occlusion or combo seen in training",
                            "inferable": occ["inferable_pixels"],
                            "unknowable": occ["unknowable_pixels"],
                        }
                    )
                    subprocess.run(["rm", "-rf", str(out)])
            else:
                rejected.append({"seed": seed, "reason": "render failed", "log_tail": log[-500:]})
                subprocess.run(["rm", "-rf", str(out)])
            seed += 1
            print(
                json.dumps(
                    {
                        "split": split,
                        "accepted": len(accepted),
                        "rejected": len(rejected),
                        "seed": seed - 1,
                    }
                ),
                flush=True,
            )
        card["splits"][split] = accepted
        card["rejected"][split] = rejected
        (root / "dataset_card.json").write_text(json.dumps(card, indent=2) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
