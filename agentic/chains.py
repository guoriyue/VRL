"""Edit chains: a declared multi-step editing schedule, run and trained step by step.

A chain is one source image and an ordered list of editing instructions. Step
``k`` edits the output of step ``k-1`` (the source for step 0), and every state
is scored once after the last step. ``run_chain`` is the one loop; the editor
and judge roles decide what it is for:

* evaluation: ``roles.LocalEditor`` (one image per step) with ``roles.RewardJudge``;
* training: ``GroupEditor``, which generates a sample group per step through
  ``vrl``'s rollout collector, draws the next state uniformly, scores every
  group in one reward call, and hands the groups to ``vrl``'s one-shot trainer.

Each step is scored against its parent, so the editor is paid for the
requested edit and for keeping earlier edits. No credit flows backwards. ``vrl``
sees a chain only through its ``OwnedCollection`` seam.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import torch

from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.stats import RolloutStats
from vrl.trainers.data.artifacts import resolve_prompt_example_references
from vrl.trainers.data.prompts import PromptExample, prompt_example_from_row
from vrl.utils.artifacts import sha256_file
from vrl.utils.json_files import write_json
from vrl.utils.media import write_png
from vrl.utils.media_reference import MediaReference

if TYPE_CHECKING:
    from vrl.rollouts.collector.core import RolloutCollector

RUN_SCHEMA = "vrl.edit-chain-run.v1"

# Reward metadata every step carries; chain and step metadata may not set them.
CHAIN_METADATA_KEYS = frozenset(
    {
        "prompt_id",
        "chain_id",
        "chain_step",
        "chain_length",
        "chain_source",
        "chain_parent_sample_id",
    }
)

_ROW_FIELDS = frozenset(
    {"chain_id", "source", "steps", "requirement", "reward_assets", "metadata"}
)


@dataclass(frozen=True, slots=True)
class PolicyStamp:
    """Which policy, which weights, which update: a role's identity in a run record."""

    name: str
    revision: str
    version: int


@dataclass(frozen=True, slots=True)
class Artifact:
    """An image on disk, named by content so a record can refer to it unambiguously."""

    path: str
    sha256: str

    @classmethod
    def from_path(cls, path: str | Path) -> Artifact:
        resolved = Path(path).expanduser().resolve(strict=True)
        return cls(str(resolved), sha256_file(resolved))


@dataclass(frozen=True, slots=True)
class Score:
    total: float
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class EditChain:
    """A declared editing schedule over one source image."""

    chain_id: str
    source: Artifact
    # Step prompts and their reward-only fields; conditioning images are the
    # chain's business, so steps declare none.
    steps: list[PromptExample]
    # Where training runs of this chain (record, drawn parent images) are written.
    media_dir: Path
    # Extra conditions in words, passed to rewards beside the instructions.
    requirement: str = ""
    # Verifier-only files (targets, masks), passed to rewards by path.
    reward_assets: dict[str, Artifact] = field(default_factory=dict)
    # Reward metadata shared by every step; a step's own metadata wins.
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chain_id:
            raise ValueError("edit chain needs a chain_id")
        if not self.steps:
            raise ValueError(f"edit chain {self.chain_id!r} declares no steps")
        for index, step in enumerate(self.steps):
            if step.reference_images or step.reference_video:
                raise ValueError(
                    f"edit chain {self.chain_id!r} step {index} declares conditioning media; "
                    "the chain supplies each step's reference image"
                )
        owners = [("chain", self.metadata)] + [
            (f"step {index}", step.metadata) for index, step in enumerate(self.steps)
        ]
        for owner, metadata in owners:
            reserved = CHAIN_METADATA_KEYS & set(metadata)
            if reserved:
                raise ValueError(
                    f"edit chain {self.chain_id!r} {owner} metadata sets collector-owned keys "
                    f"{sorted(reserved)}"
                )

    @property
    def instruction(self) -> str:
        """The whole task in words: every step's instruction in order."""

        return " ".join(step.prompt for step in self.steps)

    def record(self) -> dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "instruction": self.instruction,
            "requirement": self.requirement,
            "source": asdict(self.source),
            "steps": [{"prompt": step.prompt, "metadata": step.metadata} for step in self.steps],
            "reward_assets": {name: asdict(asset) for name, asset in self.reward_assets.items()},
            "metadata": self.metadata,
        }

    def step_example(
        self, index: int, parent_image: str, *, parent_sample_id: str | None
    ) -> PromptExample:
        """The one-shot prompt for step ``index`` conditioned on ``parent_image``."""

        step = self.steps[index]
        metadata = {
            **self.metadata,
            **step.metadata,
            "prompt_id": f"{self.chain_id}:{index}",
            "chain_id": self.chain_id,
            "chain_step": index,
            "chain_length": len(self.steps),
            "chain_source": self.source.path,
            "chain_parent_sample_id": parent_sample_id,
        }
        return replace(step, reference_images=[parent_image], metadata=metadata)

    async def collect(
        self,
        collector: RolloutCollector,
        *,
        group_size: int,
        runtime_debug: bool,
        policy_version: int | None,
        stats: RolloutStats,
    ) -> list[RolloutBatch]:
        """Run the chain once for training and return its per-step sample groups."""

        editor = GroupEditor(
            collector,
            group_size=group_size,
            runtime_debug=runtime_debug,
            policy_version=policy_version,
            stats=stats,
        )
        await run_chain(
            self, editor, editor, output_dir=self.media_dir / self.chain_id / uuid.uuid4().hex
        )
        return editor.batches


class Editor(Protocol):
    @property
    def policy_stamp(self) -> PolicyStamp: ...
    async def activate(self) -> None: ...
    async def park(self) -> None: ...
    async def edit(
        self, chain: EditChain, index: int, current: Artifact, *, seed: int, output_dir: Path
    ) -> Artifact: ...


class Judge(Protocol):
    """Scores every state of a run once, after the last step.

    ``artifacts`` is the source followed by each step's output. A state the
    judge cannot score is ``None``; the final state must be scored.
    """

    @property
    def revision(self) -> str: ...
    async def activate(self) -> None: ...
    async def park(self) -> None: ...
    async def score(self, chain: EditChain, artifacts: list[Artifact]) -> list[Score | None]: ...


async def run_chain(
    chain: EditChain,
    editor: Editor,
    judge: Judge,
    *,
    output_dir: Path,
    seed: int = 0,
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    """Edit every step on its predecessor, then score all states; persist ``run.json``.

    Roles hold the GPU one at a time: each activates for its operation and parks
    even if it raises, and a failed park ends the run. An error record never
    trains.
    """

    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "run.json"
    trace: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "status": "running",
        "chain": chain.record(),
        "seed": seed,
        "editor_policy": asdict(editor.policy_stamp),
        "judge_revision": judge.revision,
        "steps": [],
    }

    def save() -> None:
        write_json(path, trace, allow_nan=False)

    async def in_role(role: Any, operation: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            await role.activate()
            return await operation(*args, **kwargs)
        finally:
            await role.park()

    save()
    try:
        async with asyncio.timeout(timeout_s):
            for role in (editor, judge):
                await role.park()
            states = [chain.source]
            for index, step in enumerate(chain.steps):
                record = {"index": index, "prompt": step.prompt, "status": "pending"}
                trace["steps"].append(record)
                save()
                edited = await in_role(
                    editor,
                    editor.edit,
                    chain,
                    index,
                    states[-1],
                    seed=seed + index,
                    output_dir=output_dir,
                )
                states.append(edited)
                record.update(artifact=asdict(edited), status="complete")
                save()
            scores = await in_role(judge, judge.score, chain, list(states))
            if len(scores) != len(states) or scores[-1] is None:
                raise ValueError("judge must score every state and the final one")
            trace.update(
                status="success",
                state_scores=[None if score is None else asdict(score) for score in scores],
                final_artifact=asdict(states[-1]),
                final_score=asdict(scores[-1]),
            )
            save()
            return trace
    except BaseException as error:
        if trace["steps"] and trace["steps"][-1]["status"] == "pending":
            trace["steps"][-1]["status"] = "error"
        trace.update(status="error", error={"type": type(error).__name__, "message": str(error)})
        save()
        raise


def sample_image(output: Any, index: int) -> torch.Tensor:
    """One decoded image from a generation output: a tensor batch or boxed references."""

    media = output[index]
    if isinstance(media, MediaReference):
        media = media.resolve()
    if not isinstance(media, torch.Tensor):
        raise TypeError(f"edit chain parent must be an image tensor, got {type(media).__name__}")
    if media.ndim == 4 and media.shape[1] == 1:
        # A one-frame [C, T, H, W] image family output.
        media = media[:, 0]
    if media.ndim != 3:
        raise ValueError(f"edit chain parent must be one image, got shape {tuple(media.shape)}")
    return media


class GroupEditor:
    """Editor and judge of a training run, over ``vrl``'s rollout collector.

    ``edit`` generates one sample group per step through the collector and
    draws the next state uniformly from the driver RNG (checkpointed with the
    run). ``score`` scores every group in one reward call at the end, keeps
    the trainer batches, and reports the drawn sample's score for each state by
    position; the source is not scored. The collector's rollout schedule owns
    GPU phases, so activate/park do nothing here.
    """

    revision = "vrl-rollout-collector"

    def __init__(
        self,
        collector: RolloutCollector,
        *,
        group_size: int,
        runtime_debug: bool,
        policy_version: int | None,
        stats: RolloutStats,
    ) -> None:
        self.collector = collector
        self.group_size = group_size
        self.runtime_debug = runtime_debug
        self.stats = stats
        self.policy_stamp = PolicyStamp("editor", "vrl-rollout-runtime", policy_version or 0)
        self.batches: list[RolloutBatch] = []
        self._generated: list[Any] = []
        self._drawn: list[int] = []  # the sample index drawn at each step
        self._parent_sample_id: str | None = None
        self._started = time.perf_counter()

    async def activate(self) -> None:
        pass

    async def park(self) -> None:
        pass

    async def edit(
        self, chain: EditChain, index: int, current: Artifact, *, seed: int, output_dir: Path
    ) -> Artifact:
        from vrl.rollouts.collector.core import RolloutGenerationResult

        example = chain.step_example(index, current.path, parent_sample_id=self._parent_sample_id)
        request = self.collector.request_builder.build(
            [example.generation_input()],
            group_size=self.group_size,
            metadata=example.reward_metadata(),
            request_overrides=dict(example.request_overrides or {}),
            runtime_debug=self.runtime_debug,
            policy_version=self.policy_stamp.version or None,
            reward_media_refs=True,
        )
        started = time.perf_counter()
        unscored = await self.collector.generate_rollout(request)
        self._generated.append(
            RolloutGenerationResult(unscored, [index], started, time.perf_counter())
        )
        rows = unscored.output.sample_rows
        if len(rows) != self.group_size:
            raise RuntimeError(
                f"edit chain step returned {len(rows)} samples for a group of {self.group_size}"
            )
        chosen = random.randrange(self.group_size)
        path = output_dir / f"step{index:02d}.png"
        write_png(sample_image(unscored.output.output, chosen), path)
        self._drawn.append(chosen)
        self._parent_sample_id = rows[chosen].sample_id
        return Artifact.from_path(path)

    async def score(self, chain: EditChain, artifacts: list[Artifact]) -> list[Score | None]:
        reward_started = time.perf_counter()
        batches = self.collector.assemble_training_batches(
            await self.collector.evaluate_rollout([group.unscored for group in self._generated])
        )
        reward_wall = time.perf_counter() - reward_started
        self.batches = self.collector.finish_scored_prompt_groups(
            self._generated, batches, self.stats
        )
        self.stats.add_phases(
            {
                "collect.wall": time.perf_counter() - self._started,
                "collect.generation_wall": sum(
                    group.completed_at - group.started_at for group in self._generated
                ),
                "collect.reward_wall": reward_wall,
                "collect.generation_reward_overlap": 0.0,
            }
        )
        self.stats.add_counter("collect.group_count", len(self.batches))
        self.stats.add_counter(
            "collect.sample_count", sum(int(batch.rewards.shape[0]) for batch in self.batches)
        )
        # States are positional: the source, then one drawn sample per step.
        if len(artifacts) != len(self._drawn) + 1:
            raise ValueError("chain judge expects the source followed by one state per step")
        scores: list[Score | None] = [None]
        for batch, sample in zip(self.batches, self._drawn, strict=True):
            scores.append(
                Score(
                    float(batch.rewards[sample].item()),
                    {
                        name: float(values[sample].item())
                        for name, values in batch.extras.get("reward_components", {}).items()
                    },
                )
            )
        return scores


def load_edit_chains(path: str | Path, *, media_dir: Path) -> list[EditChain]:
    """Parse an edit-chain JSONL manifest.

    Each row is ``{"source": image, "steps": [...], "chain_id"?, "requirement"?,
    "reward_assets"?: {name: path}, "metadata"?}``. A step is an instruction
    string or a prompt-manifest row without conditioning media. Paths resolve
    relative to the manifest; the source and reward assets must exist. Unknown
    row keys are rejected: a chain has two metadata owners, so a silent merge
    could not say which one a stray key meant.
    """

    manifest = Path(path).expanduser().resolve()

    def image(text: Any, context: str, what: str) -> Artifact:
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{context}: {what} must be an image path")
        resolved = (manifest.parent / Path(text).expanduser()).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"{context}: {what} does not exist: {resolved}")
        return Artifact.from_path(resolved)

    chains: list[EditChain] = []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        context = f"{manifest}:{line_number}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{context}: invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{context}: edit chain rows must be objects")
        unknown = set(row) - _ROW_FIELDS
        if unknown:
            raise ValueError(f"{context}: unknown edit chain fields {sorted(unknown)}")
        raw_steps = row.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError(f"{context}: steps must be a non-empty list")
        steps = []
        for index, raw in enumerate(raw_steps):
            if isinstance(raw, str):
                raw = {"prompt": raw}
            if not isinstance(raw, dict):
                raise ValueError(f"{context}: step {index} must be a string or an object")
            step = prompt_example_from_row(raw, context=f"{context} step {index}")
            steps.append(
                resolve_prompt_example_references(
                    step, data_root=manifest.parent, allow_absolute=True
                )
            )
        for name in ("metadata", "reward_assets"):
            if not isinstance(row.get(name, {}), dict):
                raise ValueError(f"{context}: {name} must be an object")
        requirement = row.get("requirement", "")
        if not isinstance(requirement, str):
            raise ValueError(f"{context}: requirement must be a string")
        chain_id = row.get("chain_id", f"{manifest.stem}:{line_number}")
        if not isinstance(chain_id, str) or not chain_id:
            raise ValueError(f"{context}: chain_id must be a non-empty string")
        chains.append(
            EditChain(
                chain_id,
                image(row.get("source"), context, "source"),
                steps,
                Path(media_dir),
                requirement,
                {
                    name: image(value, context, f"reward asset {name}")
                    for name, value in row.get("reward_assets", {}).items()
                },
                dict(row.get("metadata", {})),
            )
        )
    if len({chain.chain_id for chain in chains}) != len(chains):
        raise ValueError(f"{manifest}: chain_id values must be unique")
    return chains


__all__ = [
    "CHAIN_METADATA_KEYS",
    "RUN_SCHEMA",
    "Artifact",
    "EditChain",
    "Editor",
    "GroupEditor",
    "Judge",
    "PolicyStamp",
    "Score",
    "load_edit_chains",
    "run_chain",
    "sample_image",
]
