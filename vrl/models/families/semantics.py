"""Typed temporal/action semantics exposed by a model-family entry.

These fields classify the executable policy variant selected by the registry,
not every component stored in a checkpoint.  A hybrid checkpoint may contain a
causal reasoner, a token prior, and a frozen denoise renderer while exposing
only one of those stages as the RL policy. Generation-only entries retain the
same routing vocabulary but omit a replay recipe in the family registry;
semantics alone never imply trainability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

GenerationRegime = Literal["full_sequence", "chunk_autoregressive"]


@dataclass(frozen=True, slots=True)
class PolicySemantics:
    """Temporal organization of the selected executable denoise stage.

    ``full_sequence`` updates all output positions together in each policy
    step; ``chunk_autoregressive`` advances one temporal chunk from earlier
    batches. Every registered policy is a continuous denoise step over a
    denoise-layout trajectory, so the regime is the only routing fact.
    """

    generation_regime: GenerationRegime

    def __post_init__(self) -> None:
        if self.generation_regime not in get_args(GenerationRegime):
            raise ValueError(
                f"unsupported policy generation_regime: {self.generation_regime!r}",
            )


# ── Family task vocabulary ──────────────────────────────────────────────────
# Owner of the task-token vocabulary carried by ``ModelFamilyEntry.task`` and
# ``GenerationRequest.task``. The video-task set and the task->task_type
# projection both derive from this one place, so a new token is a single edit
# here — a new video token can no longer silently fall through a hand-maintained
# set to the wrong "image" default (the skew this centralizes away). This is NOT
# a form-4 registry re-derivation: the registry stores only each family's task
# literal, never these modality / task_type mappings.
Task = Literal["t2i", "t2v", "t2w", "i2v", "v2w"]

# i2v is image-conditioned but still emits video frames, so it scores as video.
VIDEO_TASKS: frozenset[str] = frozenset({"t2v", "i2v", "v2w", "t2w"})

_TASK_TYPE_BY_TASK: dict[str, str] = {
    "t2i": "text_to_image",
    "t2v": "text_to_video",
    "t2w": "text_to_video",
    "i2v": "image_to_video",
    "v2w": "video2world",
}


def task_modality(task: str) -> str:
    """Reward/output modality of a family task token: ``"video"`` or ``"image"``."""

    return "video" if task in VIDEO_TASKS else "image"


def task_type_for(task: str) -> str | None:
    """Map a family task token to its ``PromptExample.task_type``, or ``None``."""

    return _TASK_TYPE_BY_TASK.get(task)


__all__ = [
    "VIDEO_TASKS",
    "GenerationRegime",
    "PolicySemantics",
    "Task",
    "task_modality",
    "task_type_for",
]
