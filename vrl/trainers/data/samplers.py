"""Training samplers."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, Sampler


class DistributedKRepeatSampler(Sampler):
    """Sampler that repeats each prompt K times across all GPUs.

    For GRPO to work, we need K samples per prompt in each batch so we
    can compute per-prompt advantages.  This sampler:
    1. Selects M = (num_replicas * batch_size) / K unique prompts
    2. Repeats each K times
    3. Shuffles deterministically (synced across ranks via seed)
    4. Splits to each rank

    Yields lists of indices (one batch per iteration), infinitely.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        k: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, (
            f"k ({k}) must divide num_replicas*batch_size ({self.total_samples})"
        )
        self.m = self.total_samples // self.k  # unique prompts per iteration
        self.epoch = 0

    def __iter__(self):  # type: ignore[override]
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()
            repeated = [idx for idx in indices for _ in range(self.k)]

            shuffled_order = torch.randperm(len(repeated), generator=g).tolist()
            shuffled = [repeated[i] for i in shuffled_order]

            per_rank = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                per_rank.append(shuffled[start : start + self.batch_size])

            yield per_rank[self.rank]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


__all__ = ["DistributedKRepeatSampler"]
