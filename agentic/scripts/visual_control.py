"""Compare a frozen visual controller to explicit editor-budget baselines.

Uses local Qwen editing and an operator-owned reward service. Keep evaluation
sources disjoint from the controller's training tasks; the script does not
check this. This is not a training command.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Literal

import torch

from agentic.episode import (
    Controller,
    Decision,
    Editor,
    Episode,
    Judge,
    Observation,
    PolicyStamp,
    Task,
)
from agentic.scripts.visual_episode import add_visual_session_arguments, open_visual_session
from agentic.trainer import ControllerTrainer
from vrl.run import OnlineRunConfig
from vrl.utils.json_files import write_json
from vrl.utils.validation import require_int


class BaselineController:
    """Protocol adapter for stop, one fixed edit, or seeded uniform actions."""

    def __init__(self, mode: Literal["stop", "fixed", "random"]) -> None:
        if mode not in {"stop", "fixed", "random"}:
            raise ValueError("unknown visual baseline")
        self.mode = mode
        self.policy_stamp = PolicyStamp(f"baseline-{mode}", "v1", 0)

    async def activate(self) -> None:
        pass

    async def park(self) -> None:
        pass

    async def decide(self, task: Task, observation: Observation, *, seed: int) -> Decision:
        stop = next(action for action in task.actions if action.kind == "stop")
        log_prob = 0.0
        if self.mode == "random":
            action = random.Random(seed).choice(task.actions)
            log_prob = -math.log(len(task.actions))
        elif self.mode == "fixed" and observation.step == 0:
            edits = [action for action in task.actions if action.kind == "edit"]
            if not edits:
                raise ValueError("fixed baseline requires an edit action")
            action = edits[0]
        else:
            action = stop
        return Decision(
            action.name,
            self.policy_stamp,
            log_prob,
            observation.digest(task),
            {"kind": "evaluation-baseline", "mode": self.mode, "seed": seed},
        )


async def compare_visual_policies(
    task: Task,
    controller: Controller,
    tool: Editor,
    judge: Judge,
    *,
    output_dir: Path,
    seed: int,
    max_tool_calls: int,
    tool_cost: float,
) -> dict:
    """Run four policies plus same-budget best-of-N, preserving every trace.

    Equal maximum editor calls are not equal FLOPs: controller decisions and reward
    selection have separate overhead. Report actual calls, decisions, judge calls
    and wall time. A role failure aborts the comparison without activating another
    role, since failed parking may leave an unsafe memory owner. Partial completed
    methods remain persisted; they are never averaged as a complete comparison.
    """
    require_int(seed, path="evaluation seed", minimum=0)
    require_int(max_tool_calls, path="max_tool_calls", minimum=1)
    if not math.isfinite(tool_cost) or tool_cost < 0:
        raise ValueError("tool_cost must be finite and non-negative")
    if not any(action.kind == "edit" for action in task.actions):
        raise ValueError("comparison requires an edit action")
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "vrl.visual-comparison.v1",
        "status": "running",
        "task_id": task.task_id,
        "source_sha256": task.source.sha256,
        "seed": seed,
        "max_tool_calls": max_tool_calls,
        "tool_cost": tool_cost,
        "methods": {},
        "capability_improvement_verified": False,
    }
    report_path = output_dir / "comparison.json"
    write_json(report_path, report)
    methods = {
        "stop": BaselineController("stop"),
        "fixed": BaselineController("fixed"),
        "random": BaselineController("random"),
        "controller": controller,
    }
    try:
        for name, policy in methods.items():
            start = time.monotonic()
            trace = await Episode(max_tool_calls=max_tool_calls, tool_cost=tool_cost).run(
                task, policy, tool, judge, output_dir=output_dir / name, seed=seed
            )
            report["methods"][name] = {
                "final_artifact": trace["final_artifact"],
                "final_score": trace["final_score"],
                "net_return": trace["discounted_return"],
                "tool_calls": trace["tool_calls"],
                "controller_decisions": len(trace["steps"]) if name == "controller" else 0,
                "judge_calls": 1,
                "seconds": time.monotonic() - start,
                "trace_paths": [str(output_dir / name / "episode.json")],
            }
            write_json(report_path, report)
        start = time.monotonic()
        candidates = []
        paths = []
        for index in range(max_tool_calls):
            directory = output_dir / "best_of_n" / f"{index:04d}"
            candidates.append(
                await Episode(max_tool_calls=1, tool_cost=tool_cost).run(
                    task,
                    methods["fixed"],
                    tool,
                    judge,
                    output_dir=directory,
                    seed=seed + index * (2 * max_tool_calls + 1),
                )
            )
            paths.append(str(directory / "episode.json"))
        best = max(range(len(candidates)), key=lambda i: candidates[i]["final_score"]["total"])
        selected = candidates[best]
        calls = sum(trace["tool_calls"] for trace in candidates)
        report["methods"]["best_of_n"] = {
            "final_artifact": selected["final_artifact"],
            "final_score": selected["final_score"],
            "net_return": selected["final_score"]["total"] - tool_cost * calls,
            "tool_calls": calls,
            "controller_decisions": 0,
            "judge_calls": len(candidates),
            "seconds": time.monotonic() - start,
            "trace_paths": paths,
            "selected_index": best,
            "selection_axis": "final_score.total",
            "candidate_count": len(candidates),
        }
        report["status"] = "success"
        write_json(report_path, report)
        return report
    except BaseException as error:
        report.update(status="error", error={"type": type(error).__name__, "message": str(error)})
        write_json(report_path, report)
        raise


async def run(args: argparse.Namespace) -> None:
    require_int(args.repeats, path="repeats", minimum=1)
    require_int(args.max_tool_calls, path="max_tool_calls", minimum=1)
    manifest = Path(args.tasks).resolve()
    tasks, groups, source_assignments = [], {}, {}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        task = Task.from_manifest_record(raw, base_dir=manifest.parent)
        if task.task_id in groups:
            raise ValueError("evaluation task IDs must be unique")
        group = raw.get("source_group", task.source.sha256)
        if not isinstance(group, str) or not group:
            raise ValueError("source_group must be a non-empty string")
        if source_assignments.setdefault(task.source.sha256, group) != group:
            raise ValueError("the same source cannot belong to different evaluation groups")
        tasks.append(task)
        groups[task.task_id] = group
    if not tasks:
        raise ValueError("evaluation needs at least one task")
    checkpoint = checkpoint_payload = None
    if args.controller_checkpoint:
        checkpoint = Path(args.controller_checkpoint).resolve()
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "settings.json",
        {
            **vars(args),
            "source_groups": groups,
            "controller_checkpoint": str(checkpoint) if checkpoint is not None else None,
        },
    )
    torch.set_num_threads(8)
    OnlineRunConfig(total_epochs=1, seed=args.seed, deterministic=True).initialize_process_rng()
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    records, media_rows = [], []
    async with open_visual_session(args, output=output) as (controller, tool, judge):
        if checkpoint_payload is not None:
            trainer = ControllerTrainer(controller, **checkpoint_payload["config"])
            trainer.load_checkpoint(checkpoint)
            OnlineRunConfig(
                total_epochs=1, seed=args.seed, deterministic=True
            ).initialize_process_rng()
        for index, task in enumerate(tasks):
            for repeat in range(args.repeats):
                seed = (
                    args.seed
                    + (index * args.repeats + repeat) * (2 * args.max_tool_calls + 1) ** 2
                )
                report = await compare_visual_policies(
                    task,
                    controller,
                    tool,
                    judge,
                    output_dir=output / "comparisons" / f"{index:04d}-{repeat:04d}",
                    seed=seed,
                    max_tool_calls=args.max_tool_calls,
                    tool_cost=args.tool_cost,
                )
                report["source_group"] = groups[task.task_id]
                records.append(report)
                for method, result in report["methods"].items():
                    media_rows.append(
                        {
                            "sample_id": f"{index:04d}-{repeat:04d}-{method}",
                            "prompt_id": task.task_id,
                            "prompt": task.instruction,
                            "path": result["final_artifact"]["path"],
                            "assets": {
                                "reference_image": task.source.path,
                                **{name: asset.path for name, asset in task.reward_assets.items()},
                            },
                            "metadata": {
                                "method": method,
                                "seed": seed,
                                "source_group": groups[task.task_id],
                                "tool_calls": result["tool_calls"],
                            },
                        }
                    )
                print(
                    json.dumps(
                        {
                            "task_id": task.task_id,
                            "seed": seed,
                            "methods": {k: v["net_return"] for k, v in report["methods"].items()},
                        }
                    ),
                    flush=True,
                )
    summaries = {}
    for method in records[0]["methods"]:
        grouped = defaultdict(list)
        for record in records:
            grouped[record["source_group"]].append(record["methods"][method])
        summaries[method] = {
            metric: statistics.fmean(
                statistics.fmean(row[metric] for row in rows) for rows in grouped.values()
            )
            for metric in (
                "net_return",
                "tool_calls",
                "controller_decisions",
                "judge_calls",
                "seconds",
            )
        }
    (output / "selected_media.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in media_rows)
    )
    write_json(
        output / "result.json",
        {
            "status": "completed",
            "source_group_count": len(set(groups.values())),
            "comparisons": len(records),
            "source_balanced_means": summaries,
            "capability_improvement_verified": False,
            "limitations": [
                "Selection and reporting share the optimized judge; independent rescoring is required.",
                "Editor-call budgets match; controller/reward overhead is reported separately.",
                "Means are descriptive; no significance claim or near-duplicate detection.",
            ],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_visual_session_arguments(parser)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--controller-checkpoint")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
