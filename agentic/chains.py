"""Edit chains: editor-only training on a declared multi-step schedule.

A chain is one source image and an ordered list of editing instructions. It
runs through the same ``Episode`` loop as controller training: the schedule
is an ``OrderedController``, and ``GroupEditor`` plays both the editor and the
judge over ``vrl``'s rollout collector. Each step generates a group of samples
through the collector, draws one uniformly as the next state, and the whole
episode is scored in one batched reward call at the end. The resulting
per-group batches train the editor with ``vrl``'s one-shot trainer.

Each step is scored against its parent, so the editor is paid for the
requested edit and for keeping earlier edits. No terminal return flows
backwards: per-step credit under a fixed schedule. ``vrl`` sees the chain only
through ``OwnedCollection``.
"""

from __future__ import annotations

import json
import random
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from agentic.episode import Action, Artifact, Episode, Observation, PolicyStamp, Score, Task
from agentic.roles import OrderedController
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.stats import RolloutStats
from vrl.trainers.data.artifacts import resolve_prompt_example_references
from vrl.trainers.data.prompts import PromptExample, prompt_example_from_row
from vrl.utils.media import write_png
from vrl.utils.media_reference import MediaReference

if TYPE_CHECKING:
    from vrl.rollouts.collector.core import RolloutCollector

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

_ROW_FIELDS = frozenset({"chain_id", "source", "steps", "metadata"})


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


@dataclass
class EditChain:
    """A declared editing schedule over one source image."""

    chain_id: str
    # Resolved path of the image step 0 edits.
    source: str
    # Step prompts and their reward-only fields; conditioning images are the
    # chain's business, so steps declare none.
    steps: list[PromptExample]
    # Where each chain episode (trace, drawn parent images) is written.
    media_dir: Path
    # Reward metadata shared by every step; a step's own metadata wins.
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chain_id or not self.source:
            raise ValueError("edit chain needs a chain_id and a source image")
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

    def step_names(self) -> list[str]:
        return [f"step-{index:02d}" for index in range(len(self.steps))]

    def task(self) -> Task:
        """The episode task: one edit action per step, in order, then stop."""

        return Task(
            self.chain_id,
            " ".join(step.prompt for step in self.steps),
            Artifact.from_path(self.source),
            (
                *(
                    Action(name, "edit", step.prompt)
                    for name, step in zip(self.step_names(), self.steps, strict=True)
                ),
                Action("stop", "stop"),
            ),
        )

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
            "chain_source": self.source,
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
        """Run the chain as one episode and return its per-step training groups."""

        editor = GroupEditor(
            collector,
            self,
            group_size=group_size,
            runtime_debug=runtime_debug,
            policy_version=policy_version,
            stats=stats,
        )
        await Episode(max_tool_calls=len(self.steps)).run(
            self.task(),
            OrderedController(tuple(self.step_names())),
            editor,
            editor,
            output_dir=self.media_dir / self.chain_id / uuid.uuid4().hex,
            seed=0,
        )
        return editor.batches


class GroupEditor:
    """Editor and judge of a chain episode, over the rollout collector.

    ``edit`` generates one sample group per step through the collector and
    draws the next state uniformly from the driver RNG (checkpointed with the
    run). ``score`` scores every group in one reward call at the end of the
    episode, keeps the trainer batches, and reports the drawn sample's score
    for each edited state by position; the source is not scored. The collector's rollout
    schedule owns GPU phases, so activate/park do nothing here.
    """

    revision = "vrl-rollout-collector"

    def __init__(
        self,
        collector: RolloutCollector,
        chain: EditChain,
        *,
        group_size: int,
        runtime_debug: bool,
        policy_version: int | None,
        stats: RolloutStats,
    ) -> None:
        self.collector = collector
        self.chain = chain
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
        self, task: Task, observation: Observation, action: Action, *, seed: int, output_dir: Path
    ) -> Artifact:
        from vrl.rollouts.collector.core import RolloutGenerationResult

        step = observation.step
        example = self.chain.step_example(
            step, observation.current.path, parent_sample_id=self._parent_sample_id
        )
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
            RolloutGenerationResult(unscored, [step], started, time.perf_counter())
        )
        rows = unscored.output.sample_rows
        if len(rows) != self.group_size:
            raise RuntimeError(
                f"edit chain step returned {len(rows)} samples for a group of {self.group_size}"
            )
        chosen = random.randrange(self.group_size)
        path = output_dir / f"step{step:02d}.png"
        write_png(sample_image(unscored.output.output, chosen), path)
        self._drawn.append(chosen)
        self._parent_sample_id = rows[chosen].sample_id
        return Artifact.from_path(path)

    async def score(self, task: Task, artifacts: list[Artifact]) -> list[Score | None]:
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

    Each row is ``{"source": image, "steps": [...], "chain_id"?: str, "metadata"?: {}}``.
    A step is an instruction string or a prompt-manifest row without conditioning
    media; its reward-only paths resolve relative to the manifest. ``source``
    must exist. Unknown row keys are rejected: a chain has two metadata owners,
    so a silent merge could not say which one a stray key meant.
    """

    manifest = Path(path).expanduser().resolve()
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
        source_text = row.get("source")
        if not isinstance(source_text, str) or not source_text.strip():
            raise ValueError(f"{context}: source must be an image path")
        source = (manifest.parent / Path(source_text).expanduser()).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"{context}: source image does not exist: {source}")
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
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"{context}: metadata must be an object")
        chain_id = row.get("chain_id", f"{manifest.stem}:{line_number}")
        if not isinstance(chain_id, str) or not chain_id:
            raise ValueError(f"{context}: chain_id must be a non-empty string")
        chains.append(EditChain(chain_id, str(source), steps, Path(media_dir), dict(metadata)))
    if len({chain.chain_id for chain in chains}) != len(chains):
        raise ValueError(f"{manifest}: chain_id values must be unique")
    return chains


__all__ = ["CHAIN_METADATA_KEYS", "EditChain", "GroupEditor", "load_edit_chains", "sample_image"]
