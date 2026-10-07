"""Small known-layer supervised warm-start diagnostic, explicitly not RL.

Reuse the layered policy and scheduler to test whether complete target layers
can shift candidate support before collecting a fresh reward-driven RL batch.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
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


def run_probe(root, target_path, output, *, steps, learning_rate, seed):
    if steps < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Positive update count and finite positive learning rate required")
    plan = json.loads((root / "plan.json").read_text())
    cases = [case for case in plan["cases"] if case["split"] == "train"]
    if len(cases) != 1:
        raise ValueError("This support diagnostic requires exactly one training fixture")
    case = cases[0]
    target = torch.load(target_path, map_location="cpu", weights_only=True)
    condition = torch.load(
        root / "conditions" / case["name"] / "condition.pt", map_location="cpu", weights_only=True
    )
    assert target["source_sha256"] == condition["source_sha256"] == case["sha256"]
    assert target["base_revision"] == plan["base_revision"]
    expected_tokens = (
        (plan["layers"] + 1) * (condition["height"] // 16) * (condition["width"] // 16)
    )
    assert target["latents"].shape == (1, expected_tokens, 64)
    assert torch.isfinite(target["latents"]).all()
    upstream_path = Path(plan["upstream"])
    assert hashlib.sha256(upstream_path.read_bytes()).hexdigest() == plan["upstream_sha256"]
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "kind": "supervised_flow_matching_support_probe",
        "rl_updates": 0,
        "steps": steps,
        "learning_rate": learning_rate,
        "seed": seed,
        "source_root": str(root),
        "target_path": str(target_path),
        "target_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
        "initial_adapter_sha256": hashlib.sha256(
            (root / "initial_adapter.pt").read_bytes()
        ).hexdigest(),
        "scope": "Single-fixture supervised diagnostic; not RL or held-out image generalization.",
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    upstream = load_upstream(upstream_path)
    model = load_transformer(plan)
    parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    initial = torch.load(root / "initial_adapter.pt", weights_only=True, map_location="cpu")
    assert initial.keys() == parameters.keys()
    with torch.no_grad():
        for name, value in parameters.items():
            value.copy_(initial[name])
    clean = target["latents"].cuda().float()
    condition = {
        key: value.cuda() if isinstance(value, torch.Tensor) else value
        for key, value in condition.items()
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        plan["base_path"], subfolder="scheduler", local_files_only=True
    )
    scheduler.set_timesteps(
        sigmas=np.linspace(1, 0, plan["steps"] + 1)[:-1],
        device="cuda",
        mu=(condition["condition"].shape[1] / 256) ** 0.5,
    )
    # Fixed unseen noise realizations test denoising fit, not image generalization.
    evaluation = []
    with torch.no_grad():
        for index in (10, 25, 40):
            generator = torch.Generator(device="cuda").manual_seed(seed + 10000 + index)
            noise = torch.randn(clean.shape, generator=generator, device="cuda")
            timestep = scheduler.timesteps[index]
            noisy, velocity_target = diffusion_pretraining_pair(
                scheduler, clean, noise, timestep[None]
            )
            prediction = predict_velocity(model, noisy, timestep, condition, upstream)
            evaluation.append(
                {
                    "index": index,
                    "before_mse": float((prediction - velocity_target).square().mean()),
                }
            )
    optimizer = torch.optim.AdamW(parameters.values(), lr=learning_rate, weight_decay=0)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    logs = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        index = int(torch.randint(1, plan["steps"], (1,), generator=generator, device="cuda"))
        timestep = scheduler.timesteps[index]
        noise = torch.randn(clean.shape, generator=generator, device="cuda")
        noisy, velocity_target = diffusion_pretraining_pair(
            scheduler, clean, noise, timestep[None]
        )
        prediction = predict_velocity(model, noisy, timestep, condition, upstream)
        loss = (prediction - velocity_target).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        gradients = [value.grad for value in parameters.values() if value.grad is not None]
        assert gradients and all(torch.isfinite(value).all() for value in gradients)
        norm = torch.nn.utils.clip_grad_norm_(
            list(parameters.values()), 1.0, error_if_nonfinite=True
        )
        assert float(norm) > 0
        optimizer.step()
        logs.append(
            {
                "step": step + 1,
                "schedule_index": index,
                "sigma": float(scheduler.sigmas[index]),
                "mse": float(loss.detach()),
                "gradient_norm": float(norm),
                "gradient_tensors": len(gradients),
            }
        )
        (output / "training.json").write_text(json.dumps(logs, indent=2) + "\n")
        print(json.dumps(logs[-1]), flush=True)
        del prediction, loss
    saved = {name: value.detach().cpu().clone() for name, value in parameters.items()}
    assert all(torch.isfinite(value).all() for value in saved.values())
    differences = {
        name: float((value - initial[name]).abs().max()) for name, value in saved.items()
    }
    assert max(differences.values()) > 0
    torch.save(saved, output / "adapter.pt")
    torch.save(optimizer.state_dict(), output / "optimizer.pt")
    with torch.no_grad():
        for row in evaluation:
            generator = torch.Generator(device="cuda").manual_seed(seed + 10000 + row["index"])
            noise = torch.randn(clean.shape, generator=generator, device="cuda")
            timestep = scheduler.timesteps[row["index"]]
            noisy, velocity_target = diffusion_pretraining_pair(
                scheduler, clean, noise, timestep[None]
            )
            prediction = predict_velocity(model, noisy, timestep, condition, upstream)
            row["after_mse"] = float((prediction - velocity_target).square().mean())
        for value in parameters.values():
            value.zero_()
        restored = torch.load(output / "adapter.pt", map_location="cpu", weights_only=True)
        for name, value in parameters.items():
            value.copy_(restored[name])
        reloaded = predict_velocity(model, noisy, timestep, condition, upstream)
    assert torch.equal(prediction, reloaded)
    result = {
        **config,
        "supervised_updates": steps,
        "changed_tensors": sum(value > 0 for value in differences.values()),
        "max_parameter_difference": max(differences.values()),
        "adapter_sha256": hashlib.sha256((output / "adapter.pt").read_bytes()).hexdigest(),
        "reload_velocity_exact": True,
        "fixed_noise_denoising": evaluation,
        "quality_improvement_established": False,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--target-latents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    with Path("/tmp/vrl-preservation-gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_probe(
            args.root.resolve(),
            args.target_latents.resolve(),
            args.output.resolve(),
            steps=args.steps,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
