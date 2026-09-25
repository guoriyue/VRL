"""Bounded visual editing episodes: task, observation and decision records, and the loop.

An edit is one semantic action; its denoising steps are not controller
decisions. The controller, editor and judge keep separate identities so a
trace says which policy produced each decision and image. This module owns
scheduling and credit only; models and optimizers live elsewhere.

The same loop serves every training mode: a learned controller with a frozen
editor, a declared schedule with an editor trained by ``vrl`` (edit chains),
and, later, both. The judge scores every state once, after the last edit, so
an editor that scores its own sample groups in one batched reward call fits
the same contract as a per-image judge.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from vrl.utils.artifacts import sha256_file
from vrl.utils.json_files import canonical_json_sha256, write_json
from vrl.utils.validation import require_int

_TASK_FIELDS = frozenset(
    {
        "task_id",
        "instruction",
        "source",
        "actions",
        "requirement",
        "context_images",
        "source_group",
        "reward_assets",
    }
)


@dataclass(frozen=True, slots=True)
class PolicyStamp:
    """Which policy, which weights, which update: a role's on-policy identity."""

    name: str
    revision: str
    version: int


@dataclass(frozen=True, slots=True)
class Artifact:
    """An image on disk, named by content so a trace can refer to it unambiguously."""

    path: str
    sha256: str

    @classmethod
    def from_path(cls, path: str | Path) -> Artifact:
        resolved = Path(path).expanduser().resolve(strict=True)
        return cls(str(resolved), sha256_file(resolved))


@dataclass(frozen=True, slots=True)
class Action:
    name: str
    kind: Literal["edit", "stop"]
    instruction: str = ""

    def __post_init__(self) -> None:
        if not self.name or self.kind not in {"edit", "stop"}:
            raise ValueError("action needs a name and an edit/stop kind")
        if (self.kind == "edit") != bool(self.instruction.strip()):
            raise ValueError("an edit action carries an instruction; stop carries none")


@dataclass(frozen=True, slots=True)
class Task:
    """One task row: what the controller sees, what the editor may do, what the judge gets."""

    task_id: str
    instruction: str
    source: Artifact
    actions: tuple[Action, ...]
    # Extra conditions in words, shown to the controller and passed to rewards.
    requirement: str = ""
    # Named images the controller sees after the original and current ones.
    context_images: dict[str, Artifact] = field(default_factory=dict)
    # Verifier-only files (targets, masks). Never shown to the controller or editor.
    reward_assets: dict[str, Artifact] = field(default_factory=dict)

    @classmethod
    def from_manifest_record(cls, record: dict[str, Any], *, base_dir: Path) -> Task:
        """Paths resolve relative to the declaring task file; unknown fields fail."""

        unknown = set(record) - _TASK_FIELDS
        if unknown:
            raise ValueError(f"unknown visual task fields: {sorted(unknown)}")

        def images(name: str) -> dict[str, Artifact]:
            return {
                key: Artifact.from_path(base_dir / Path(path).expanduser())
                for key, path in record.get(name, {}).items()
            }

        return cls(
            record["task_id"],
            record["instruction"],
            Artifact.from_path(base_dir / Path(record["source"]).expanduser()),
            tuple(Action(**action) for action in record["actions"]),
            record.get("requirement", ""),
            images("context_images"),
            images("reward_assets"),
        )

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Task:
        """Rebuild a task from its persisted trace form."""

        return cls(
            record["task_id"],
            record["instruction"],
            Artifact(**record["source"]),
            tuple(Action(**action) for action in record["actions"]),
            record.get("requirement", ""),
            {name: Artifact(**item) for name, item in record.get("context_images", {}).items()},
            {name: Artifact(**item) for name, item in record.get("reward_assets", {}).items()},
        )

    def as_record(self) -> dict[str, Any]:
        """Persisted form; empty optional fields are omitted so older traces keep their identity."""

        record = asdict(self)
        for name in ("requirement", "context_images", "reward_assets"):
            if not record[name]:
                record.pop(name)
        return record

    def __post_init__(self) -> None:
        if not self.task_id or not self.instruction.strip():
            raise ValueError("visual task ID and instruction are required")
        names = [action.name for action in self.actions]
        if not names or len(names) != len(set(names)):
            raise ValueError("visual action names must be non-empty and unique")
        if sum(action.kind == "stop" for action in self.actions) != 1:
            raise ValueError("a visual task requires exactly one stop action")


@dataclass(frozen=True, slots=True)
class Score:
    total: float
    components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Observation:
    step: int
    remaining_tool_calls: int
    current: Artifact
    previous_action: str | None = None

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Observation:
        return cls(
            record["step"],
            record["remaining_tool_calls"],
            Artifact(**record["current"]),
            record["previous_action"],
        )

    def digest(self, task: Task) -> str:
        """Bind a decision's replay to the exact task, action vocabulary and observation."""

        return canonical_json_sha256(
            {"task": task.as_record(), "observation": asdict(self)}, allow_nan=False
        )


@dataclass(frozen=True, slots=True)
class Decision:
    action: str
    policy: PolicyStamp
    old_log_prob: float
    observation_digest: str
    # Controller-owned replay context (token IDs, a tensor file reference), never a model.
    replay: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Decision:
        return cls(
            record["action"],
            PolicyStamp(**record["policy"]),
            record["old_log_prob"],
            record["observation_digest"],
            record["replay"],
        )


class Controller(Protocol):
    @property
    def policy_stamp(self) -> PolicyStamp: ...
    async def activate(self) -> None: ...
    async def park(self) -> None: ...
    async def decide(self, task: Task, observation: Observation, *, seed: int) -> Decision: ...


class Editor(Protocol):
    @property
    def policy_stamp(self) -> PolicyStamp: ...
    async def activate(self) -> None: ...
    async def park(self) -> None: ...
    async def edit(
        self, task: Task, observation: Observation, action: Action, *, seed: int, output_dir: Path
    ) -> Artifact: ...


class Judge(Protocol):
    """Scores every state of an episode once, after the last edit.

    ``artifacts`` is the source followed by each edit output. A state the judge
    cannot score is ``None``; the final state must be scored.
    """

    @property
    def revision(self) -> str: ...
    async def activate(self) -> None: ...
    async def park(self) -> None: ...
    async def score(self, task: Task, artifacts: list[Artifact]) -> list[Score | None]: ...


class Episode:
    """Budget and credit rules for one bounded episode; ``run`` executes and persists it."""

    schema = "vrl.visual-episode.v2"

    def __init__(
        self,
        *,
        max_tool_calls: int = 2,
        tool_cost: float = 0.0,
        gamma: float = 1.0,
        timeout_s: float = 600.0,
    ) -> None:
        require_int(max_tool_calls, path="max_tool_calls", minimum=1)
        if not math.isfinite(tool_cost) or tool_cost < 0 or not 0 <= gamma <= 1 or timeout_s <= 0:
            raise ValueError("tool_cost must be >= 0, gamma in [0, 1] and timeout_s > 0")
        self.max_tool_calls = max_tool_calls
        self.tool_cost = tool_cost
        self.gamma = gamma
        self.timeout_s = timeout_s

    @staticmethod
    def returns(rewards: list[float], *, gamma: float = 1.0) -> list[float]:
        """Finite-horizon return-to-go: a decision is paid only for what follows it."""

        if not 0 <= gamma <= 1 or not all(math.isfinite(value) for value in rewards):
            raise ValueError("gamma must be in [0, 1] and rewards finite")
        returns = [0.0] * len(rewards)
        value = 0.0
        for index in range(len(rewards) - 1, -1, -1):
            value = rewards[index] + gamma * value
            returns[index] = value
        return returns

    async def run(
        self,
        task: Task,
        controller: Controller,
        editor: Editor,
        judge: Judge,
        *,
        output_dir: Path,
        seed: int,
    ) -> dict[str, Any]:
        """Run one finite episode with sequential resource handoffs.

        Each role parks even if its operation raises; a failed park aborts the
        episode and the next role is never activated. An error trace is never a
        successful training episode.
        """

        output_dir.mkdir(parents=True, exist_ok=False)
        return await _EpisodeRun(
            self, task, controller, editor, judge, output_dir=output_dir, seed=seed
        ).execute()

    @classmethod
    def training_steps(
        cls, trace: dict[str, Any], *, policy: PolicyStamp
    ) -> tuple[Task, list[tuple[Observation, Decision, float]]]:
        """What the controller trainer consumes from one successful on-policy trace."""

        if trace.get("schema") != cls.schema or trace.get("status") != "success":
            raise ValueError("controller training needs a successful visual episode")
        if trace["controller_policy"] != asdict(policy):
            raise ValueError("controller training requires the current on-policy version")
        task = Task.from_record(trace["task"])
        steps = [
            (
                Observation.from_record(step["observation"]),
                Decision.from_record(step["decision"]),
                step["return_to_go"],
            )
            for step in trace["steps"]
        ]
        return task, steps


class _EpisodeRun:
    """The mutable state of one ``Episode.run``: roles, the trace, and its file."""

    def __init__(
        self,
        rules: Episode,
        task: Task,
        controller: Controller,
        editor: Editor,
        judge: Judge,
        *,
        output_dir: Path,
        seed: int,
    ) -> None:
        self.rules, self.task = rules, task
        self.controller, self.editor, self.judge = controller, editor, judge
        self.output_dir, self.seed = output_dir, seed
        self.path = output_dir / "episode.json"
        self.trace: dict[str, Any] = {
            "schema": Episode.schema,
            "status": "running",
            "task": task.as_record(),
            "seed": seed,
            "controller_policy": asdict(controller.policy_stamp),
            "tool_policy": asdict(editor.policy_stamp),
            "judge_revision": judge.revision,
            "max_tool_calls": rules.max_tool_calls,
            "tool_cost": rules.tool_cost,
            "gamma": rules.gamma,
            "timeout_s": rules.timeout_s,
            "steps": [],
        }

    def save(self) -> None:
        write_json(self.path, self.trace, allow_nan=False)

    async def in_role(self, role: Any, operation: Any, *args: Any, **kwargs: Any) -> Any:
        """Hold one role's resources for one operation; it parks even on failure."""

        try:
            await role.activate()
            return await operation(*args, **kwargs)
        finally:
            await role.park()

    async def execute(self) -> dict[str, Any]:
        rules, task, trace = self.rules, self.task, self.trace
        actions = {action.name: action for action in task.actions}
        self.save()
        try:
            async with asyncio.timeout(rules.timeout_s):
                # Every role starts parked, so the first activation finds a free GPU.
                for role in (self.controller, self.editor, self.judge):
                    await role.park()
                states = [task.source]
                previous_action = None
                while True:
                    calls = len(states) - 1
                    observation = Observation(
                        len(trace["steps"]),
                        rules.max_tool_calls - calls,
                        states[-1],
                        previous_action,
                    )
                    decision = await self.in_role(
                        self.controller,
                        self.controller.decide,
                        task,
                        observation,
                        seed=self.seed + len(trace["steps"]) * 2,
                    )
                    if decision.observation_digest != observation.digest(task):
                        raise ValueError("controller decision does not match its observation")
                    if decision.action not in actions:
                        raise ValueError(
                            f"controller selected an unavailable action: {decision.action}"
                        )
                    action = actions[decision.action]
                    step: dict[str, Any] = {
                        "observation": asdict(observation),
                        "decision": asdict(decision),
                        "reward": 0.0,
                        "status": "pending",
                    }
                    trace["steps"].append(step)
                    self.save()
                    if action.kind == "stop":
                        trace["termination"] = "stop"
                        step["status"] = "complete"
                        break
                    edited = await self.in_role(
                        self.editor,
                        self.editor.edit,
                        task,
                        observation,
                        action,
                        seed=self.seed + observation.step * 2 + 1,
                        output_dir=self.output_dir,
                    )
                    states.append(edited)
                    step.update(
                        tool_result={
                            "artifact": asdict(edited),
                            "policy": asdict(self.editor.policy_stamp),
                        },
                        reward=-rules.tool_cost,
                        status="complete",
                    )
                    previous_action = action.name
                    self.save()
                    if len(states) - 1 == rules.max_tool_calls:
                        trace["termination"] = "tool_budget"
                        break
                scores = await self.in_role(self.judge, self.judge.score, task, list(states))
                if len(scores) != len(states) or scores[-1] is None:
                    raise ValueError("judge must score every state and the final one")
                # The terminal objective is paid exactly once, including on an immediate stop.
                trace["steps"][-1]["reward"] += scores[-1].total
                returns = Episode.returns(
                    [step["reward"] for step in trace["steps"]], gamma=rules.gamma
                )
                for step, value in zip(trace["steps"], returns, strict=True):
                    step["return_to_go"] = value
                trace.update(
                    status="success",
                    tool_calls=len(states) - 1,
                    state_scores=[None if s is None else asdict(s) for s in scores],
                    final_artifact=asdict(states[-1]),
                    final_score=asdict(scores[-1]),
                    discounted_return=returns[0],
                )
                self.save()
                return trace
        except BaseException as error:
            if trace["steps"] and trace["steps"][-1]["status"] == "pending":
                trace["steps"][-1]["status"] = "error"
            trace.update(
                status="error", error={"type": type(error).__name__, "message": str(error)}
            )
            self.save()
            raise
