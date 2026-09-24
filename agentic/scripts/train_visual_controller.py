"""Train bounded visual control from real editing episodes with a frozen Qwen editor.

Input is a JSONL manifest of explicit visual tasks. Every optimizer update collects
fresh independent episodes per task, verifies likelihood replay, and checkpoints
the controller, optimizer and RNG. This does not jointly train the diffusion tool.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import torch

from agentic.episode import (
    Episode,
    Task,
)
from agentic.scripts.visual_episode import (
    add_visual_session_arguments,
    open_visual_session,
    visual_editor_sampling,
)
from agentic.trainer import ControllerTrainer
from vrl.run import OnlineRunConfig
from vrl.utils.json_files import write_json
from vrl.utils.validation import require_int


async def run(args: argparse.Namespace) -> None:
    require_int(args.updates, path="updates", minimum=1)
    require_int(args.episodes_per_task, path="episodes_per_task", minimum=2)
    require_int(args.max_tool_calls, path="max_tool_calls", minimum=1)
    sampling = visual_editor_sampling(args)
    torch.set_num_threads(8)
    OnlineRunConfig(
        total_epochs=args.updates, seed=args.seed, deterministic=args.deterministic
    ).initialize_process_rng()
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    tasks = []
    manifest = Path(args.tasks).expanduser().resolve()
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        tasks.append(Task.from_manifest_record(raw, base_dir=manifest.parent))
    if not tasks or len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("visual task manifest must contain unique non-empty task IDs")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "metrics").mkdir()
    contract = {
        "tasks": [task.as_record() for task in tasks],
        "generator_revision": args.revision,
        "reward_model": args.reward_model,
        "reward_revision": args.reward_revision,
        "resolution": args.resolution,
        "steps": args.steps,
        "output_mode": args.output_mode,
        "max_tool_calls": args.max_tool_calls,
        "tool_cost": args.tool_cost,
        "gamma": args.gamma,
        "episodes_per_task": args.episodes_per_task,
        "seed": args.seed,
        "deterministic": args.deterministic,
    }
    if getattr(args, "height", None) is not None:
        contract.update(output_height=sampling["height"], output_width=sampling["width"])
    write_json(output / "settings.json", vars(args))
    write_json(output / "contract.json", contract)
    async with open_visual_session(args, output=output) as (controller, tool, judge):
        contract["editor_policy"] = asdict(tool.policy_stamp)
        if getattr(args, "reward_config", None):
            contract["judge_revision"] = judge.revision
        write_json(output / "contract.json", contract)
        trainer = ControllerTrainer(controller, learning_rate=args.learning_rate)
        progress = {"next_update": 0, "episode_cursor": 0}
        if args.resume:
            progress = trainer.load_checkpoint(Path(args.resume))
        if progress["next_update"] >= args.updates:
            raise ValueError("resume checkpoint has already reached requested updates")
        trainer.save_checkpoint(
            output / "checkpoint-start.pt", contract=contract, progress=progress
        )
        for update in range(progress["next_update"], args.updates):
            traces = []
            for task in tasks:
                for _ in range(args.episodes_per_task):
                    cursor = progress["episode_cursor"]
                    trace = await Episode(
                        max_tool_calls=args.max_tool_calls,
                        tool_cost=args.tool_cost,
                        gamma=args.gamma,
                    ).run(
                        task,
                        controller,
                        tool,
                        judge,
                        output_dir=output / "episodes" / f"{cursor:08d}",
                        seed=args.seed + cursor * (2 * args.max_tool_calls + 1),
                    )
                    traces.append(trace)
                    progress["episode_cursor"] += 1
            metrics = await trainer.update(traces)
            metrics.update(
                update=update,
                mean_return=sum(trace["discounted_return"] for trace in traces) / len(traces),
                mean_tool_calls=sum(trace["tool_calls"] for trace in traces) / len(traces),
                stop_fraction=sum(trace["termination"] == "stop" for trace in traces)
                / len(traces),
            )
            write_json(output / "metrics" / f"{update:08d}.json", metrics)
            progress["next_update"] = update + 1
            trainer.save_checkpoint(
                output / f"checkpoint-{update + 1}.pt", contract=contract, progress=progress
            )
            print(json.dumps(metrics), flush=True)
        controller.model.save_pretrained(output / "controller_adapter")
        write_json(
            output / "result.json",
            {
                "status": "completed",
                "kind": "controller-only-rl",
                "policy": asdict(controller.policy_stamp),
                "progress": progress,
                "capability_improvement_verified": False,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_visual_session_arguments(parser)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--resume")
    parser.add_argument("--deterministic", action="store_true")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
