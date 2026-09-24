"""Run a declared sequence of Qwen edits and report preservation after every step."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import torch
from pydantic import ConfigDict, Field

from agentic.episode import (
    Decision,
    Editor,
    Episode,
    Judge,
    Observation,
    PolicyStamp,
    Task,
)
from agentic.export import export_episode_media
from agentic.scripts.visual_episode import add_visual_session_arguments, open_visual_session
from vrl.config.base import ConfigBase
from vrl.rewards.sequences import (
    EditSequenceSpec,
    SequenceRequirement,
    sequence_report,
)
from vrl.run import OnlineRunConfig
from vrl.utils.json_files import canonical_json_sha256, write_json


class SequencePlan(ConfigBase):
    """An explicit action schedule and when each final-task requirement becomes due."""

    model_config = ConfigDict(frozen=True)

    sequence_id: str = Field(min_length=1)
    actions: tuple[str, ...] = Field(min_length=1)
    requirements: tuple[SequenceRequirement, ...] = Field(min_length=1)

    def validate_task(self, task: Task) -> EditSequenceSpec:
        edits = {action.name for action in task.actions if action.kind == "edit"}
        if any(name not in edits for name in self.actions):
            raise ValueError("sequence actions must select existing edit actions")
        return EditSequenceSpec(
            sequence_id=self.sequence_id,
            samples=[f"{task.task_id}:state:{i:04d}" for i in range(len(self.actions) + 1)],
            requirements=list(self.requirements),
        )


class OrderedController:
    """Deterministic protocol adapter; never treat its zero log-probs as learned policy data."""

    def __init__(self, actions: tuple[str, ...]) -> None:
        if not actions or any(not isinstance(action, str) or not action for action in actions):
            raise ValueError("ordered controller needs nonempty edit action names")
        self.actions = tuple(actions)
        self.policy_stamp = PolicyStamp(
            "scripted-edit-sequence", canonical_json_sha256(self.actions, allow_nan=False), 0
        )

    async def activate(self) -> None:
        pass

    async def park(self) -> None:
        pass

    async def decide(self, task: Task, observation: Observation, *, seed: int) -> Decision:
        if not 0 <= observation.step <= len(self.actions):
            raise ValueError("ordered controller received a step outside its plan")
        expected_previous = self.actions[observation.step - 1] if observation.step else None
        if observation.previous_action != expected_previous:
            raise ValueError("ordered controller received a different action history")
        if observation.step < len(self.actions):
            action = self.actions[observation.step]
            if not any(item.name == action and item.kind == "edit" for item in task.actions):
                raise ValueError("ordered controller action is not an available edit")
        else:
            action = next(item.name for item in task.actions if item.kind == "stop")
        return Decision(
            action,
            self.policy_stamp,
            0.0,
            observation.digest(task),
            {"kind": "scripted-sequence-baseline", "actions": list(self.actions), "seed": seed},
        )


async def run_visual_sequence(
    task: Task,
    plan: SequencePlan,
    tool: Editor,
    judge: Judge,
    *,
    output_dir: Path,
    seed: int,
    tool_cost: float = 0.0,
    timeout_s: float = 1800.0,
) -> dict[str, Any]:
    """Execute every planned edit on its predecessor and audit the recorded scores.

    Requirements use one fixed final task specification; `active_from` delays a
    check until its planned stage. A preservation report never changes terminal
    rewards or pays for the same achievement multiple times. Independent rescoring
    consumes the exported media manifest and the saved sequence specification.
    """
    spec = plan.validate_task(task)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "plan.json", plan.model_dump(mode="json"))
    trace = await Episode(
        max_tool_calls=len(plan.actions), tool_cost=tool_cost, timeout_s=timeout_s
    ).run(
        task,
        OrderedController(plan.actions),
        tool,
        judge,
        output_dir=output_dir / "episode",
        seed=seed,
    )
    exported = export_episode_media(trace, output_dir / "media")
    rows = [
        json.loads(line) for line in (output_dir / "media/media.jsonl").read_text().splitlines()
    ]
    scores = [step["observation"]["score"] for step in trace["steps"]] + [trace["final_score"]]
    records = {
        row["sample_id"]: {
            "input": row,
            "status": "success",
            "result": {"scores": value["components"]},
        }
        for row, value in zip(rows, scores, strict=True)
    }
    observations = {
        "run_id": canonical_json_sha256(trace, allow_nan=False),
        "config": {"kind": "recorded-episode-judge", "revision": judge.revision},
        "records": records,
    }
    report = sequence_report(observations, spec)
    # The sequence digest covers measurements/requirements. Execution identity
    # is retained separately because that report alone does not prove lineage.
    result = {
        "schema": "vrl.visual-sequence-execution.v1",
        "plan_id": canonical_json_sha256(plan.model_dump(mode="json"), allow_nan=False),
        "episode_id": canonical_json_sha256(trace, allow_nan=False),
        "export_id": exported["export_id"],
        "controller_kind": "scripted-baseline",
        "evidence_kind": "recorded-judge-observations",
        "tool_calls": trace["tool_calls"],
        "sequence_report": report,
    }
    write_json(output_dir / "sequence_spec.json", spec.model_dump(mode="json"))
    write_json(output_dir / "judge_observations.json", observations)
    write_json(output_dir / "report.json", result)
    return result


async def run(args: argparse.Namespace) -> None:
    manifest = Path(args.task).expanduser().resolve()
    task = Task.from_manifest_record(json.loads(manifest.read_text()), base_dir=manifest.parent)
    plan = SequencePlan.model_validate_json(Path(args.plan).read_text())
    plan.validate_task(task)
    if len(plan.actions) > args.max_tool_calls:
        raise ValueError("sequence exceeds --max-tool-calls; declare its full budget explicitly")
    torch.set_num_threads(8)
    OnlineRunConfig(total_epochs=1, seed=args.seed, deterministic=True).initialize_process_rng()
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "settings.json", vars(args))
    # A scripted baseline needs no extra controller model allocation. The editor
    # still loads its own required text encoder through the normal family path.
    async with open_visual_session(
        args, output=output, controller=OrderedController(plan.actions)
    ) as (_, tool, judge):
        report = await run_visual_sequence(
            task,
            plan,
            tool,
            judge,
            output_dir=output / "sequence",
            seed=args.seed,
            tool_cost=args.tool_cost,
            timeout_s=args.timeout_s,
        )
    print(
        json.dumps(
            {
                "episode_id": report["episode_id"],
                "tool_calls": report["tool_calls"],
                "final_requirements_met": report["sequence_report"]["final_requirements_met"],
                "coverage_complete": report["sequence_report"]["coverage_complete"],
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_visual_session_arguments(parser)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--plan", required=True, help="Ordered actions and requirement activation JSON"
    )
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
