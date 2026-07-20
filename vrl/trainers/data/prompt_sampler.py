"""Prompt batch selection for online training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class PromptSamplingStrategy(str, Enum):
    """Supported prompt batch selection strategies."""

    RANDOM_WITHOUT_REPLACEMENT = "random_without_replacement"
    SEQUENTIAL_WINDOW = "sequential_window"


@dataclass(frozen=True, slots=True, kw_only=True)
class PromptBatchSampler:
    """Select one rank-local prompt batch from a shared global draw.

    Every distributed rank must use an identically seeded or restored generator.
    Each call consumes the same global draw on every rank, then returns only that
    rank's slice. This is deliberately not a ``torch.utils.data.BatchSampler``:
    online training requests one batch per optimizer update and also needs a
    side-effect-free preview of the next draw.
    """

    generator: torch.Generator
    num_examples: int
    prompts_per_rank: int
    strategy: PromptSamplingStrategy | str
    num_replicas: int = 1
    rank: int = 0

    def __post_init__(self) -> None:
        if self.num_examples < 1:
            raise ValueError("prompt manifest must contain at least one example")
        if self.prompts_per_rank < 1:
            raise ValueError("prompts_per_rank must be >= 1")
        if self.num_replicas < 1:
            raise ValueError("num_replicas must be >= 1")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(
                f"rank must be in [0, {self.num_replicas}), got {self.rank}",
            )
        try:
            strategy = PromptSamplingStrategy(self.strategy)
        except ValueError as exc:
            expected = " / ".join(strategy.value for strategy in PromptSamplingStrategy)
            raise ValueError(
                f"unknown prompt sampling strategy {self.strategy!r}; expected {expected}",
            ) from exc
        object.__setattr__(self, "strategy", strategy)

        global_batch_size = self.prompts_per_rank * self.num_replicas
        if (
            self.strategy is PromptSamplingStrategy.RANDOM_WITHOUT_REPLACEMENT
            and self.num_examples < global_batch_size
        ):
            raise ValueError(
                "random_without_replacement requires at least one example per "
                f"global batch slot ({self.num_examples} < {global_batch_size})",
            )

    def sample(self, *, epoch: int = 0) -> list[int]:
        """Consume the generator and return this rank's prompt indices."""

        return self._sample_with(self.generator, epoch=epoch)

    def preview(self, *, epoch: int = 0) -> list[int]:
        """Return this rank's next indices without consuming the generator."""

        preview_generator = torch.Generator(device=self.generator.device)
        preview_generator.set_state(self.generator.get_state().clone())
        return self._sample_with(preview_generator, epoch=epoch)

    def _sample_with(self, generator: torch.Generator, *, epoch: int) -> list[int]:
        global_batch_size = self.prompts_per_rank * self.num_replicas
        if self.strategy is PromptSamplingStrategy.RANDOM_WITHOUT_REPLACEMENT:
            global_indices = torch.randperm(
                self.num_examples,
                generator=generator,
            )[:global_batch_size].tolist()
        else:
            start = (max(0, int(epoch)) * global_batch_size) % self.num_examples
            global_indices = [
                (start + offset) % self.num_examples for offset in range(global_batch_size)
            ]

        shard_start = self.rank * self.prompts_per_rank
        return global_indices[shard_start : shard_start + self.prompts_per_rank]


__all__ = ["PromptBatchSampler", "PromptSamplingStrategy"]
