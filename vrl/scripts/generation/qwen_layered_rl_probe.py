"""Exercise layered-image conditioning, stochastic rollout and LoRA replay.

The gradient probe is a numerical diagnostic, not a reward-trained checkpoint.
Collection records actual float32 SDE actions and mean element log densities.
Frozen quantized backbone weights are an explicit memory adaptation; they are not
claimed numerically equivalent to the released bf16 baselines.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vrl.math.denoise.flow_matching import sde_step_with_logprob


def noise_schedule(spec, count):
    """Resolve and validate positive stochastic noise for every recorded action."""
    values = spec.get("noise_levels", [spec["noise_level"]] * count)
    if len(values) != count or any(
        not math.isfinite(value) or not 0 < value <= 1 for value in values
    ):
        raise ValueError("Noise schedule requires one finite positive value <= 1 per step")
    return [float(value) for value in values]


def recorded_noise_levels(trajectory, plan):
    """Bind replay noise to the recorded behavior and the planned absolute steps."""
    count = len(trajectory["timesteps"])
    actual = noise_schedule(trajectory, count)
    start = trajectory.get("schedule_start_index", 0)
    expected = noise_schedule(plan, plan.get("steps", count))[start : start + count]
    if actual != expected:
        raise ValueError("Recorded noise schedule disagrees with collection plan")
    return actual


def recorded_schedule(trajectory):
    """Resolve local action indices against the original sampling schedule."""
    local_times = trajectory["timesteps"]
    count = len(local_times)
    if trajectory.get("teacher_initialized") or trajectory.get("schedule_start_index", 0):
        required = ("full_sigmas", "full_timesteps", "absolute_step_indices")
        if not all(key in trajectory for key in required):
            raise ValueError("Restarted trajectories require the complete sampling schedule")
        sigmas, times, indices = (trajectory[key] for key in required)
        start = trajectory["schedule_start_index"]
        expected = torch.arange(start, start + count, device=indices.device)
        if not torch.equal(indices, expected) or start < 0 or start + count > len(times):
            raise ValueError("Invalid restarted trajectory step mapping")
        if len(sigmas) != len(times) + 1:
            raise ValueError("Incomplete original sigma schedule")
        if not torch.equal(times[indices], local_times) or not torch.equal(
            sigmas[start : start + count + 1], trajectory["sigmas"]
        ):
            raise ValueError("Local trajectory schedule disagrees with original schedule")
        return sigmas, times, indices
    return trajectory["sigmas"], local_times, torch.arange(count)


class FrozenCPUBlock(torch.nn.Module):
    """Materialize frozen weights per checkpointed block; retain GPU LoRA leaves.

    functional_call leaves the CPU master parameters untouched. Non-reentrant
    checkpointing discards forward intermediates and recreates one block's
    device weights during backward, avoiding unsafe Parameter.data swapping.
    """

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, *args, **kwargs):
        state = {
            name: parameter if parameter.requires_grad else parameter.to("cuda")
            for name, parameter in self.block.named_parameters()
        }
        state.update({name: buffer.to("cuda") for name, buffer in self.block.named_buffers()})
        return torch.func.functional_call(self.block, state, args, kwargs)


def load_upstream(path):
    """Load the pinned release's packing, VAE and RoPE conventions."""
    spec = importlib.util.spec_from_file_location("layered_release", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cache_conditions(plan, root, upstream):
    from diffusers import QwenImageLayeredPipeline

    pipe = QwenImageLayeredPipeline.from_pretrained(
        plan["base_path"], transformer=None, torch_dtype=torch.bfloat16, local_files_only=True
    )
    pipe.text_encoder.requires_grad_(False).eval().to("cuda")
    prompts = {}
    with torch.no_grad():
        for case in plan["cases"]:
            text = case.get("prompt", plan["prompt"])
            if text not in prompts:
                prompt, mask = pipe.encode_prompt(text, device=torch.device("cuda"))
                prompts[text] = (prompt.cpu(), mask.cpu() if mask is not None else None)
    pipe.text_encoder.to("cpu")
    pipe.text_encoder = None
    gc.collect()
    torch.cuda.empty_cache()
    pipe.vae.requires_grad_(False).eval().to("cuda")
    pipe.vae.enable_slicing()
    for case in plan["cases"]:
        text = case.get("prompt", plan["prompt"])
        prompt, mask = prompts[text]
        source = Path(case["source"])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == case["sha256"]
        input_mode = case.get("input_mode", "RGB")
        if input_mode not in ("RGB", "RGBA"):
            raise ValueError(f"Unsupported conditioning input mode: {input_mode}")
        image = Image.open(source).convert(input_mode)
        width, height = upstream.compute_aspect_resize(*image.size, plan["size"])
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1
        with torch.no_grad():
            condition = upstream.encode_condition_image(pipe.vae, pixels[None].cuda()).cpu()
        directory = root / "conditions" / case["name"]
        directory.mkdir(parents=True, exist_ok=False)
        image.save(directory / "source.png")
        torch.save(
            {
                "condition": condition,
                "prompt": prompt,
                "mask": mask,
                "prompt_text": text,
                "width": width,
                "height": height,
                "layers": plan["layers"],
                "source_sha256": case["sha256"],
                "input_mode": input_mode,
            },
            directory / "condition.pt",
        )
        print(
            json.dumps({"cached": case["name"], "condition_shape": list(condition.shape)}),
            flush=True,
        )


def load_transformer(plan):
    from diffusers import BitsAndBytesConfig, QwenImageTransformer2DModel
    from peft import LoraConfig

    torch.manual_seed(plan["adapter_seed"])
    if plan.get("quantization", "nf4") == "bf16_cpu":
        quantization = None
    elif plan.get("quantization", "nf4") == "int8":
        quantization = BitsAndBytesConfig(load_in_8bit=True)
    elif plan.get("quantization", "nf4") == "nf4":
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        raise ValueError("Unsupported quantization policy")
    model = QwenImageTransformer2DModel.from_pretrained(
        plan["base_path"],
        subfolder="transformer",
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"} if quantization is None else {"": 0},
        quantization_config=quantization,
    )
    model.requires_grad_(False)
    model.add_adapter(LoraConfig(**plan["lora"]))
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert "lora_" in name, name
            param.data = param.data.to(device="cuda", dtype=torch.float32)
    if quantization is None:
        for name, child in model.named_children():
            if name != "transformer_blocks":
                child.to("cuda")
        model.transformer_blocks = torch.nn.ModuleList(
            [FrozenCPUBlock(block) for block in model.transformer_blocks]
        )
    model.enable_gradient_checkpointing()
    return model.eval()


def predict_velocity(model, latent, timestep, condition, upstream):
    """Keep the condition tokens outside the generated action sequence."""
    inputs = torch.cat([latent.to(torch.bfloat16), condition["condition"]], dim=1)
    shapes = upstream.build_img_shapes(
        condition["layers"], condition["height"] // 16, condition["width"] // 16
    )
    return upstream.transformer_forward(
        model,
        inputs,
        timestep.expand(latent.shape[0]).to(torch.bfloat16),
        condition["prompt"],
        shapes,
        condition["mask"],
        torch.zeros(latent.shape[0], dtype=torch.long, device=latent.device),
    )[:, : latent.shape[1]].float()


def compare_precision(plan, root, upstream):
    """Compare recorded NF4 velocities with the original bf16 backbone."""
    from diffusers import QwenImageTransformer2DModel

    model = QwenImageTransformer2DModel.from_pretrained(
        plan["base_path"],
        subfolder="transformer",
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).eval()
    model.requires_grad_(False)
    model.enable_group_offload(
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        num_blocks_per_group=1,
        use_stream=True,
        low_cpu_mem_usage=True,
    )
    case = next(case for case in plan["cases"] if case["split"] == "train")
    condition = torch.load(root / "conditions" / case["name"] / "condition.pt", weights_only=True)
    condition = {
        key: value.cuda() if isinstance(value, torch.Tensor) else value
        for key, value in condition.items()
    }
    trajectory = torch.load(
        root / "rollouts" / case["name"] / "candidate00/trajectory.pt", weights_only=True
    )
    rows = []
    for index in (0, plan["steps"] // 2, plan["steps"] - 1):
        with torch.no_grad():
            actual = predict_velocity(
                model,
                trajectory["observations"][index].cuda(),
                trajectory["timesteps"][index].cuda(),
                condition,
                upstream,
            ).cpu()
        recorded = trajectory["old_velocity"][index]
        row = {
            "step": index,
            "bf16_rms": float(actual.square().mean().sqrt()),
            "nf4_rms": float(recorded.square().mean().sqrt()),
            "difference_rms": float((actual - recorded).square().mean().sqrt()),
            "relative_rms_error": float(
                (actual - recorded).square().mean().sqrt() / actual.square().mean().sqrt()
            ),
            "cosine": float(
                torch.nn.functional.cosine_similarity(actual.flatten(), recorded.flatten(), dim=0)
            ),
        }
        rows.append(row)
        (root / "precision_comparison.json").write_text(
            json.dumps(
                {
                    "rows": rows,
                    "scope": "Same recorded states, condition and zero-LoRA policy; bf16 versus NF4 velocity only, not a full image quality comparison.",
                },
                indent=2,
            )
        )
        print(json.dumps(row), flush=True)


def run_model(plan, root, upstream, *, stage):
    from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler

    noise_levels = noise_schedule(plan, plan["steps"])
    model = load_transformer(plan)
    trainable = {name: param for name, param in model.named_parameters() if param.requires_grad}
    adapter_path = root / "initial_adapter.pt"
    if adapter_path.exists():
        saved = torch.load(adapter_path, weights_only=True)
        assert saved.keys() == trainable.keys()
        with torch.no_grad():
            for name, parameter in trainable.items():
                parameter.copy_(saved[name])
    else:
        torch.save({name: param.detach().cpu() for name, param in trainable.items()}, adapter_path)
    adapter_sha256 = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
    (root / "model_record.json").write_text(
        json.dumps(
            {
                "quantization": plan.get("quantization", "nf4") + " frozen base; fp32 LoRA",
                "trainable_parameters": sum(p.numel() for p in trainable.values()),
                "trainable_tensors": len(trainable),
                "initial_adapter_sha256": adapter_sha256,
                "source_optimizer_updates": plan.get("source_optimizer_updates", 0),
                "source_supervised_updates": plan.get("source_supervised_updates", 0),
                "rl_update_completed": False,
                "scope": "This collection/probe does not update parameters; source policy may already be trained.",
            },
            indent=2,
        )
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        plan["base_path"], subfolder="scheduler", local_files_only=True
    )
    vae = None
    if stage == "collect":
        vae = (
            AutoencoderKLQwenImage.from_pretrained(
                plan["base_path"],
                subfolder="vae",
                local_files_only=True,
                torch_dtype=torch.bfloat16,
            )
            .eval()
            .cuda()
        )
        vae.requires_grad_(False)
        vae.enable_slicing()
    for case in plan["cases"]:
        if case["split"] != "train":
            continue
        condition = torch.load(
            root / "conditions" / case["name"] / "condition.pt", weights_only=True
        )
        condition = {
            key: value.cuda() if isinstance(value, torch.Tensor) else value
            for key, value in condition.items()
        }
        scheduler.set_timesteps(
            sigmas=np.linspace(1, 0, plan["steps"] + 1)[:-1],
            device="cuda",
            mu=(condition["condition"].shape[1] / 256) ** 0.5,
        )
        shape = (
            1,
            (plan["layers"] + 1) * (condition["height"] // 16) * (condition["width"] // 16),
            model.config.in_channels,
        )
        for candidate, seed in enumerate(plan["candidate_seeds"]):
            generator = torch.Generator(device="cuda").manual_seed(seed)
            latent = torch.randn(shape, generator=generator, device="cuda", dtype=torch.float32)
            observations, actions, densities, velocities = [], [], [], []
            for index, timestep in enumerate(scheduler.timesteps):
                with torch.no_grad():
                    velocity = predict_velocity(model, latent, timestep, condition, upstream)
                    transition = sde_step_with_logprob(
                        scheduler,
                        velocity,
                        timestep[None],
                        latent,
                        generator=generator,
                        noise_level=noise_levels[index],
                        step_index=index,
                    )
                if stage == "probe":
                    replay_velocity = predict_velocity(
                        model, latent, timestep, condition, upstream
                    )
                    replay = sde_step_with_logprob(
                        scheduler,
                        replay_velocity,
                        timestep[None],
                        latent,
                        prev_sample=transition.prev_sample,
                        noise_level=noise_levels[index],
                        step_index=index,
                    )
                    parity = float((replay.log_prob.detach() - transition.log_prob).abs().max())
                    assert parity < 1e-5, parity
                    (-replay.log_prob.sum()).backward()
                    gradients = [p.grad for p in trainable.values() if p.grad is not None]
                    assert gradients and all(torch.isfinite(g).all() for g in gradients)
                    norm = float(
                        torch.stack([g.float().square().sum() for g in gradients]).sum().sqrt()
                    )
                    assert norm > 0
                    record = {
                        "case": case["name"],
                        "latent_shape": list(shape),
                        "logprob_parity_max": parity,
                        "velocity_parity_max": float(
                            (velocity - replay_velocity.detach()).abs().max()
                        ),
                        "gradient_norm": norm,
                        "gradient_tensors": len(gradients),
                        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                        "rl_update_completed": False,
                        "scope": "Real layered-model SDE replay and backward diagnostic only; no reward or optimizer update.",
                    }
                    (root / "probe.json").write_text(json.dumps(record, indent=2))
                    print(json.dumps(record), flush=True)
                    return
                observations.append(latent.cpu())
                actions.append(transition.prev_sample.cpu())
                densities.append(transition.log_prob.cpu())
                velocities.append(velocity.cpu())
                latent = transition.prev_sample
                if index % 10 == 0:
                    print(
                        json.dumps({"case": case["name"], "candidate": candidate, "step": index}),
                        flush=True,
                    )
            directory = root / "rollouts" / case["name"] / f"candidate{candidate:02d}"
            directory.mkdir(parents=True, exist_ok=False)
            torch.save(
                {
                    "observations": torch.stack(observations),
                    "actions": torch.stack(actions),
                    "old_log_density_mean": torch.stack(densities),
                    "old_velocity": torch.stack(velocities),
                    "sigmas": scheduler.sigmas.cpu(),
                    "timesteps": scheduler.timesteps.cpu(),
                    "seed": seed,
                    "noise_level": plan["noise_level"],
                    "noise_levels": noise_levels,
                    "condition_case": case["name"],
                    "initial_adapter_seed": plan["adapter_seed"],
                    "behavior_adapter_sha256": adapter_sha256,
                    "source_supervised_updates": plan.get("source_supervised_updates", 0),
                    "action_dtype": "float32",
                },
                directory / "trajectory.pt",
            )
            with torch.no_grad():
                layers = upstream.decode_layers(
                    vae, latent, condition["height"], condition["width"], plan["layers"]
                )
                for i in range(plan["layers"]):
                    upstream.layer_to_pil_rgba(layers[0, i]).save(directory / f"layer_{i}.png")
                upstream.tensor_to_pil(upstream.composite_layers(layers)[0]).save(
                    directory / "composite.png"
                )
            print(json.dumps({"saved_candidate": str(directory)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("cache", "probe", "collect", "precision"), required=True
    )
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = args.plan.resolve().parent
    upstream_path = Path(plan["upstream"])
    assert hashlib.sha256(upstream_path.read_bytes()).hexdigest() == plan["upstream_sha256"]
    upstream = load_upstream(upstream_path)
    with Path("/tmp/vrl-preservation-gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (root / (args.stage + "_process.json")).write_text(
            json.dumps({"pid": os.getpid(), "stage": args.stage})
        )
        if args.stage == "cache":
            cache_conditions(plan, root, upstream)
        elif args.stage == "precision":
            compare_precision(plan, root, upstream)
        else:
            run_model(plan, root, upstream, stage=args.stage)


if __name__ == "__main__":
    main()
