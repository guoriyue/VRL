"""Check real visual-controller likelihood replay and adapter gradients, not RL gain."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import torch

from agentic.controller import CategoricalController
from agentic.episode import (
    Action,
    Artifact,
    Observation,
    PolicyStamp,
    Task,
)
from vrl.utils.json_files import write_json


async def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    checkpoint = Path(args.path).resolve(strict=True)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    controller = CategoricalController.from_qwen_checkpoint(
        checkpoint,
        policy=PolicyStamp("qwen3-vl-controller", args.revision, 0),
        replay_dir=output / "replay",
        device=torch.device(args.device),
        max_image_pixels=65536,
        temperature=args.temperature,
        observation_mode=getattr(args, "controller_observation", "rgb"),
    )
    model = controller.model
    task = Task(
        "controller-probe",
        args.instruction,
        Artifact.from_path(Path(args.source)),
        (
            Action("edit", "edit", args.instruction),
            Action("stop", "stop"),
        ),
    )
    current = Artifact.from_path(Path(args.current or args.source))
    observation = Observation(0, 2, current)
    await controller.activate()
    try:
        decision = await controller.decide(task, observation, seed=args.seed)
        before = controller.replay_log_prob(task, observation, decision)
        replay_error = abs(before.item() - decision.old_log_prob)
        if replay_error > 1e-5:
            raise RuntimeError(f"controller likelihood replay error {replay_error}")
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        parameter_before = [parameter.detach().cpu().clone() for parameter in trainable]
        optimizer = torch.optim.AdamW(trainable, lr=1e-5)
        (-before).backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not torch.isfinite(norm) or norm <= 0:
            raise RuntimeError(f"invalid controller gradient norm {norm}")
        optimizer.step()
        parameter_delta = max(
            (parameter.detach().cpu() - previous).abs().max().item()
            for parameter, previous in zip(trainable, parameter_before, strict=True)
        )
        if parameter_delta <= 0:
            raise RuntimeError("controller optimizer did not change any trainable parameters")
        optimizer.zero_grad(set_to_none=True)
        controller.advance_policy()
        with torch.no_grad():
            after = controller.replay_log_prob(task, observation, decision).item()
        model.save_pretrained(output / "adapter")
        write_json(
            output / "report.json",
            {
                "kind": "likelihood-gradient-probe-not-rl-training",
                "settings": vars(args),
                "action": decision.action,
                "old_log_prob": decision.old_log_prob,
                "replay_log_prob": before.item(),
                "replay_abs_error": replay_error,
                "gradient_norm": norm.item(),
                "post_update_log_prob": after,
                "max_abs_parameter_delta": parameter_delta,
                "trainable_parameters": sum(parameter.numel() for parameter in trainable),
                "replay": decision.replay,
            },
        )
        print(f"controller replay error={replay_error}; gradient norm={norm.item()}", flush=True)
    finally:
        await controller.park()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--current")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--controller-observation", choices=("rgb", "rgba"), default="rgb")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
