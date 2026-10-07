"""Evaluate frozen base, official adapter and trained LoRA with identical Heun steps."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vrl.scripts.eval.layered_candidate_views import prepare_views
from vrl.scripts.generation.qwen_layered_rl_probe import (
    FrozenCPUBlock,
    load_transformer,
    load_upstream,
    predict_velocity,
)


def evaluate(root, *, arm, cases, seed, checkpoint=None, official_adapter=None):
    from diffusers import (
        AutoencoderKLQwenImage,
        FlowMatchEulerDiscreteScheduler,
        QwenImageTransformer2DModel,
    )

    plan = json.loads((root / "plan.json").read_text())
    assert plan["quantization"] == "bf16_cpu"
    upstream_path = Path(plan["upstream"])
    assert hashlib.sha256(upstream_path.read_bytes()).hexdigest() == plan["upstream_sha256"]
    upstream = load_upstream(upstream_path)
    selected = [case for case in plan["cases"] if case["name"] in cases]
    assert set(cases) == {case["name"] for case in selected}
    if arm == "stable":
        from peft import PeftModel

        assert official_adapter is not None
        assert plan["official_adapter_revision"] in str(official_adapter.resolve())
        base = QwenImageTransformer2DModel.from_pretrained(
            plan["base_path"],
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        model = PeftModel.from_pretrained(base, official_adapter).eval()
        model.requires_grad_(False)
        for name, child in base.named_children():
            if name != "transformer_blocks":
                child.to("cuda")
        base.transformer_blocks = torch.nn.ModuleList(
            [FrozenCPUBlock(block) for block in base.transformer_blocks]
        )
        weight_path = official_adapter / "adapter_model.safetensors"
    else:
        model = load_transformer(plan)
        weight_path = checkpoint
        if arm == "base" and weight_path is None:
            weight_path = root / "initial_adapter.pt"
        assert weight_path is not None
        saved = torch.load(weight_path, weights_only=True)
        parameters = {
            name: value for name, value in model.named_parameters() if value.requires_grad
        }
        assert saved.keys() == parameters.keys()
        if arm == "base":
            assert all(
                torch.count_nonzero(value) == 0
                for name, value in saved.items()
                if "lora_B" in name
            )
        with torch.no_grad():
            for name, value in parameters.items():
                value.copy_(saved[name])
        model.requires_grad_(False)
    model.eval()
    weight_sha = hashlib.sha256(weight_path.read_bytes()).hexdigest()
    vae = (
        AutoencoderKLQwenImage.from_pretrained(
            plan["base_path"], subfolder="vae", torch_dtype=torch.bfloat16, local_files_only=True
        )
        .eval()
        .cuda()
    )
    vae.requires_grad_(False)
    vae.enable_slicing()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        plan["base_path"], subfolder="scheduler", local_files_only=True
    )
    for case in selected:
        directory = root / "evaluation" / case["name"] / arm / f"seed{seed}"
        directory.mkdir(parents=True, exist_ok=False)
        condition_path = root / "conditions" / case["name"] / "condition.pt"
        condition = torch.load(condition_path, weights_only=True)
        assert condition["source_sha256"] == case["sha256"]
        condition = {
            key: value.cuda() if isinstance(value, torch.Tensor) else value
            for key, value in condition.items()
        }
        scheduler.set_timesteps(
            sigmas=np.linspace(1, 0, plan["steps"] + 1)[:-1],
            device="cuda",
            mu=(condition["condition"].shape[1] / 256) ** 0.5,
        )
        generator = torch.Generator(device="cuda").manual_seed(seed)
        latent = torch.randn(
            1,
            (plan["layers"] + 1) * condition["condition"].shape[1],
            model.config.in_channels,
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
        with torch.no_grad():
            for index, timestep in enumerate(scheduler.timesteps):
                dt = scheduler.sigmas[index + 1] - scheduler.sigmas[index]
                velocity = predict_velocity(model, latent, timestep, condition, upstream)
                midpoint = latent + dt * velocity
                if scheduler.sigmas[index + 1] > 0:
                    corrected = predict_velocity(
                        model, midpoint, scheduler.sigmas[index + 1] * 1000, condition, upstream
                    )
                    latent = latent + dt * (velocity + corrected) / 2
                else:
                    latent = midpoint
                if index % 10 == 0:
                    print(
                        json.dumps({"case": case["name"], "arm": arm, "step": index}), flush=True
                    )
            decoded = upstream.decode_layers(
                vae, latent, condition["height"], condition["width"], plan["layers"]
            )
        layers = [upstream.layer_to_pil_rgba(decoded[0, index]) for index in range(plan["layers"])]
        for index, layer in enumerate(layers):
            layer.save(directory / f"layer_{index}.png")
        source = Image.open(root / "conditions" / case["name"] / "source.png").convert("RGB")
        source.save(directory / "source.png")
        diagnostics = prepare_views(source, layers, directory / "review")
        record = {
            "arm": arm,
            "case": case["name"],
            "split": case["split"],
            "seed": seed,
            "sampler": "Heun; fp32 states and initial noise; bf16 model; CFG1",
            "steps": plan["steps"],
            "layers": plan["layers"],
            "weight_sha256": weight_sha,
            "condition_sha256": hashlib.sha256(condition_path.read_bytes()).hexdigest(),
            "upstream_sha256": plan["upstream_sha256"],
            "diagnostics": diagnostics,
            "scope": "Paired evaluation protocol; not a stochastic RL trajectory or proof of improvement.",
        }
        (directory / "record.json").write_text(json.dumps(record, indent=2))
        print(json.dumps({"saved_evaluation": str(directory)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=("base", "stable", "tuned"), required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Trained adapter, or an explicit zero-B adapter for the frozen base arm.",
    )
    parser.add_argument("--official-adapter", type=Path)
    args = parser.parse_args()
    with Path("/tmp/vrl-preservation-gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evaluate(
            args.root.resolve(),
            arm=args.arm,
            cases=args.cases,
            seed=args.seed,
            checkpoint=args.checkpoint,
            official_adapter=args.official_adapter,
        )


if __name__ == "__main__":
    main()
