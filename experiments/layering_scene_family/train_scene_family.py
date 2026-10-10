"""Multi-scene supervised candidate support over the randomized scene family. Not RL.

Each step draws one cached training scene, noises its target latents at a
schedule position inside ``--schedule-indices`` and penalizes the velocity error
under that scene's token weights (``weighted`` arm: inferable hidden tokens
raised; ``control`` arm: all ones). Starts from the official adapter, saves
``adapter_<step>.pt`` every ``--checkpoint-every`` updates and ``adapter.pt`` at
the end. Run with the layered_rl interpreter; takes the GPU lock.
"""

import argparse
import fcntl
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from vrl.math.denoise.flow_matching import diffusion_pretraining_pair
from vrl.scripts.generation.qwen_layered_rl_probe import (
    load_transformer,
    load_upstream,
    predict_velocity,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("weighted", "control"), required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--schedule-indices", type=int, nargs=2, default=(0, 48))
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="previous output dir: continue from its adapter.pt and optimizer.pt",
    )
    parser.add_argument(
        "--extra-empty-max",
        type=int,
        default=0,
        help="append 0..K fully transparent spare layers per step (target = encoded empty frame)",
    )
    parser.add_argument("--empty-latent", type=Path, default=None)
    parser.add_argument(
        "--extra-dataset",
        type=Path,
        action="append",
        default=[],
        help="more cached datasets (their train/ split) mixed in uniformly by scene",
    )
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    upstream_path = Path(plan["upstream"])
    assert hashlib.sha256(upstream_path.read_bytes()).hexdigest() == plan["upstream_sha256"]
    low, high = args.schedule_indices
    assert 0 <= low < high <= plan["steps"]
    scenes = sorted((args.dataset / "train").glob("scene_*/cache.pt"))[: args.max_scenes]
    for extra in args.extra_dataset:
        scenes += sorted((extra / "train").glob("*/cache.pt"))
    empty = None
    if args.extra_empty_max:
        assert args.empty_latent is not None, "--extra-empty-max needs --empty-latent"
        empty = torch.load(args.empty_latent, map_location="cpu", weights_only=True)[
            "latent"
        ].float()
    assert scenes, "no cached training scenes"
    caches = [torch.load(path, map_location="cpu", weights_only=True) for path in scenes]
    assert all(c["base_revision"] == plan["base_revision"] for c in caches)
    weight_key = "weights_weighted" if args.arm == "weighted" else "weights_control"
    args.output.mkdir(parents=True, exist_ok=False)
    resume_steps = 0
    if args.resume_from is not None:
        previous = json.loads((args.resume_from / "result.json").read_text())
        resume_steps = int(previous.get("total_updates", previous["supervised_updates"]))
    config = {
        "kind": "multi_scene_supervised_flow_matching_support",
        "resume_from": None if args.resume_from is None else str(args.resume_from),
        "resumed_at_update": resume_steps,
        "arm": args.arm,
        "rl_updates": 0,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "schedule_indices": [low, high],
        "checkpoint_every": args.checkpoint_every,
        "scenes": len(caches),
        "extra_datasets": [str(e) for e in args.extra_dataset],
        "extra_empty_max": args.extra_empty_max,
        "scene_dirs": [str(p.parent.name) for p in scenes],
        "initial_adapter_sha256": hashlib.sha256(args.initial_adapter.read_bytes()).hexdigest(),
        "dataset_card_sha256": hashlib.sha256(
            (args.dataset / "dataset_card.json").read_bytes()
        ).hexdigest(),
        "scope": "Supervised candidate support on the randomized scene family; no RL; held-out evaluation decides.",
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    with Path(os.environ.get("VRL_GPU_LOCK", "/tmp/vrl-preservation-gpu0.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        upstream = load_upstream(upstream_path)
        model = load_transformer(plan)
        parameters = {
            name: value for name, value in model.named_parameters() if value.requires_grad
        }
        initial = torch.load(args.initial_adapter, weights_only=True, map_location="cpu")
        assert initial.keys() == parameters.keys()
        start_weights = (
            initial
            if args.resume_from is None
            else torch.load(args.resume_from / "adapter.pt", weights_only=True, map_location="cpu")
        )
        with torch.no_grad():
            for name, value in parameters.items():
                value.copy_(start_weights[name])
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            plan["base_path"], subfolder="scheduler", local_files_only=True
        )
        scheduler.set_timesteps(
            sigmas=np.linspace(1, 0, plan["steps"] + 1)[:-1],
            device="cuda",
            mu=(caches[0]["condition"].shape[1] / 256) ** 0.5,
        )
        optimizer = torch.optim.AdamW(parameters.values(), lr=args.learning_rate, weight_decay=0)
        if args.resume_from is not None:
            optimizer.load_state_dict(
                torch.load(
                    args.resume_from / "optimizer.pt", weights_only=True, map_location="cuda"
                )
            )
        # A resumed segment draws from a seed offset by its start update so scene and noise draws do not repeat.
        rng = random.Random(args.seed + resume_steps)
        generator = torch.Generator(device="cuda").manual_seed(args.seed + resume_steps)
        logs = []
        started = time.perf_counter()
        for step in range(args.steps):
            cache = caches[rng.randrange(len(caches))]
            condition = {
                k: (v.cuda() if isinstance(v, torch.Tensor) else v)
                for k, v in cache.items()
                if k in ("condition", "prompt", "mask", "layers", "width", "height")
            }
            clean = cache["target_latents"].cuda().float()
            weights = cache.get(weight_key, cache.get("weights_control"))
            weights = (
                (torch.ones(1, clean.shape[1], 1) if weights is None else weights).cuda().float()
            )
            spare = rng.randint(0, args.extra_empty_max) if args.extra_empty_max else 0
            if spare:
                # Spare layers on top of the stack must stay empty: the target is a transparent frame.
                clean = torch.cat([clean] + [empty.cuda()] * spare, dim=1)
                weights = torch.cat(
                    [weights, torch.ones(1, spare * empty.shape[1], 1, device="cuda")], dim=1
                )
                weights = weights / weights.mean()
                condition["layers"] = int(cache["layers"]) + spare
            optimizer.zero_grad(set_to_none=True)
            index = int(torch.randint(low, high, (1,), generator=generator, device="cuda"))
            timestep = scheduler.timesteps[index]
            noise = torch.randn(clean.shape, generator=generator, device="cuda")
            noisy, velocity_target = diffusion_pretraining_pair(
                scheduler, clean, noise, timestep[None]
            )
            prediction = predict_velocity(model, noisy, timestep, condition, upstream)
            error = (prediction - velocity_target).square()
            loss = (error * weights).mean()
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                list(parameters.values()), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            logs.append(
                {
                    "step": resume_steps + step + 1,
                    "scene": cache["source_sha256"][:8],
                    "layers": int(condition["layers"]),
                    "spare": spare,
                    "schedule_index": index,
                    "sigma": float(scheduler.sigmas[index]),
                    "training_loss": float(loss.detach()),
                    "mse": float(error.detach().mean()),
                    "gradient_norm": float(norm),
                    "elapsed": round(time.perf_counter() - started, 1),
                }
            )
            print(json.dumps(logs[-1]), flush=True)
            if (step + 1) % 25 == 0 or step + 1 == args.steps:
                (args.output / "training.json").write_text(json.dumps(logs) + "\n")
            del prediction, loss, error, noisy, velocity_target, clean, weights
            if (
                args.checkpoint_every
                and (step + 1) % args.checkpoint_every == 0
                and step + 1 < args.steps
            ):
                torch.save(
                    {name: value.detach().cpu().clone() for name, value in parameters.items()},
                    args.output / f"adapter_{resume_steps + step + 1}.pt",
                )
        saved = {name: value.detach().cpu().clone() for name, value in parameters.items()}
        torch.save(saved, args.output / "adapter.pt")
        torch.save(optimizer.state_dict(), args.output / "optimizer.pt")
        differences = {
            name: float((value - initial[name]).abs().max()) for name, value in saved.items()
        }
        result = {
            **config,
            "supervised_updates": args.steps,
            "total_updates": resume_steps + args.steps,
            "changed_tensors": sum(v > 0 for v in differences.values()),
            "max_parameter_difference": max(differences.values()),
            "adapter_sha256": hashlib.sha256(
                (args.output / "adapter.pt").read_bytes()
            ).hexdigest(),
            "seconds": round(time.perf_counter() - started, 1),
            "quality_improvement_established": False,
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
