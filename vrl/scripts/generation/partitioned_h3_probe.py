"""One local-checkpoint H3 request with explicit disjoint component placement."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from vrl.config.precision import RolePrecision
from vrl.generation.execution.sample_batches import GenerationSampleBatch
from vrl.generation.types import GenerationRequest
from vrl.models.families.minimax_h3.partitioned_generation import (
    H3GenerationPlacement,
    build_partitioned_h3_generation_runtime_bundle,
)
from vrl.models.families.minimax_h3.runtime import MiniMaxH3BatchExecutor
from vrl.models.interfaces.runtime import ModelBuild, RolloutBuildOptions
from vrl.utils.media import write_mp4


def layer_owners(count: int, devices: list[int]) -> tuple[int, ...]:
    if count < len(devices) or not devices or len(set(devices)) != len(devices):
        raise ValueError("Need distinct devices and at least one layer per device")
    if any(device < 0 for device in devices):
        raise ValueError("CUDA indices must be nonnegative")
    return tuple(
        devices[min(index * len(devices) // count, len(devices) - 1)] for index in range(count)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--policy-devices", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--policy-root", type=int, default=0)
    parser.add_argument("--encoder-devices", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--video-vae-device", type=int)
    parser.add_argument("--keep-encoder-during-decode", action="store_true")
    for name in ("steps", "height", "width", "frames"):
        parser.add_argument(f"--{name}", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder-layer", type=int, default=50)
    parser.add_argument("--max-text-tokens", type=int, default=512)
    parser.add_argument("--save-trajectory", action="store_true")
    args = parser.parse_args()
    path = args.path.resolve(strict=True)
    if not path.is_dir() or args.output.exists():
        parser.error("Use an existing local checkpoint directory and a new output directory")
    if min(args.steps, args.height, args.width, args.frames, args.max_text_tokens) < 1:
        parser.error("Geometry, steps and text length must be positive")
    with (path / "transformer/config.json").open() as handle:
        transformer_config = json.load(handle)
    with (path / "text_encoder/config.json").open() as handle:
        encoder_config = json.load(handle)
    encoder_layers = encoder_config["text_config"]["num_hidden_layers"]
    if not 0 <= args.encoder_layer < encoder_layers:
        parser.error("Selected encoder hidden state must precede its final layer")
    placement = H3GenerationPlacement(
        layer_owners(transformer_config["num_layers"], args.policy_devices),
        args.encoder_devices[0],
        layer_owners(encoder_layers, args.encoder_devices),
        args.encoder_devices[0] if args.video_vae_device is None else args.video_vae_device,
        args.encoder_devices[-1],
        park_encoder_for_decode=not args.keep_encoder_during_decode,
    )
    preset_path = Path(__file__).resolve().parents[2] / "config/presets/model/minimax_h3/h3.yaml"
    preset = OmegaConf.to_container(OmegaConf.load(preset_path)["model"], resolve=True)
    build = ModelBuild(
        model_name_or_path=str(path),
        revision=None,
        device=f"cuda:{args.policy_root}",
        parameter_dtype=torch.bfloat16,
        family="minimax_h3",
        precision=RolePrecision("bf16", "ieee", outer_autocast=False),
        model_config={key: preset[key] for key in ("use_lora", "lora", "torch_compile")},
        rollout=RolloutBuildOptions(prompt_encoder_dtype=torch.bfloat16),
        generation_memory=preset.get("memory"),
        sampling_config={"num_steps": args.steps},
    )
    placement.validate(build)
    devices = sorted(set([args.policy_root, *args.policy_devices, *args.encoder_devices]))
    if max(devices) >= torch.cuda.device_count():
        parser.error("Placement refers to a CUDA device that is not visible")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    for device in devices:
        # Initialize each allocator before resetting its peak counters.
        torch.empty(1, dtype=torch.uint8, device=torch.device("cuda", device))
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    bundle = build_partitioned_h3_generation_runtime_bundle(build, placement)
    for device in devices:
        torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - start
    load_peaks = {str(d): torch.cuda.max_memory_allocated(d) for d in devices}
    bundle.model.pipeline.text_encoder_layer = args.encoder_layer
    request = GenerationRequest(
        request_id="partitioned-h3-probe",
        family="minimax_h3",
        task="t2v",
        inputs=[args.prompt],
        samples_per_prompt=1,
        sampling={
            "num_steps": args.steps,
            "height": args.height,
            "width": args.width,
            "num_frames": args.frames,
            "fps": 24,
            "guidance_scale": 1.0,
            "max_sequence_length": args.max_text_tokens,
            "seed": args.seed,
        },
    )
    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    result = MiniMaxH3BatchExecutor(bundle.model).forward_batch(
        request, GenerationSampleBatch(prompt_index=0, sample_start=0, sample_count=1)
    )
    for device in devices:
        torch.cuda.synchronize(device)
    generation_seconds = time.perf_counter() - start
    for tensor in (result.video, result.observations, result.actions, result.log_probs):
        if not torch.isfinite(tensor).all():
            raise RuntimeError("Nonfinite generation result")
    write_mp4(result.video[0], args.output / "video.mp4", fps=24)
    if args.save_trajectory:
        torch.save(result, args.output / "trajectory.pt")
    report = {
        "checkpoint": str(path),
        "prompt": args.prompt,
        "sampling": request.sampling,
        "encoder_layer": args.encoder_layer,
        "lora": build.lora,
        "generation_memory": preset.get("memory"),
        "placement": vars(placement),
        "policy_root": args.policy_root,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "stage_seconds": result.stage_durations,
        "load_peak_allocated_bytes": load_peaks,
        "post_executor_peak_allocated_bytes": {
            str(d): torch.cuda.max_memory_allocated(d) for d in devices
        },
        "executor_memory": result.memory,
        "peak_scope": "Executor resets current-device peaks at stage boundaries; post-executor counters are not guaranteed whole-generation high-water marks.",
        "video_shape": list(result.video.shape),
        "finite": True,
        "scope": "One request; no quality, learning, replay parity or speedup claim.",
    }
    with (args.output / "result.json").open("x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
