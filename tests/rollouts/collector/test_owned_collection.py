"""A prompt item that owns its collection runs through the collector unchanged by it."""

from __future__ import annotations

import pytest
import torch

from tests.rollouts.collector._helpers import PromptCollectionFake
from vrl.rollouts.batch import RolloutBatch
from vrl.rollouts.stats import RolloutStats


class _Owned:
    def __init__(self, groups: int) -> None:
        self.groups = groups
        self.seen: list[dict] = []

    async def collect(self, collector, *, group_size, runtime_debug, policy_version, stats):
        self.seen.append({"collector": collector, "group_size": group_size})
        return [
            RolloutBatch(
                rewards=torch.zeros(group_size),
                group_ids=torch.full((group_size,), index, dtype=torch.long),
                extras={},
                context={},
                trajectory=None,
            )
            for index in range(self.groups)
        ]


@pytest.mark.asyncio
async def test_owned_items_collect_themselves_and_get_disjoint_group_ids() -> None:
    fake = PromptCollectionFake()
    first, second = _Owned(2), _Owned(1)
    batches = await fake.prepare_training_batches(
        prompts=[first, second],
        group_size=3,
        runtime_debug=False,
        policy_version=7,
        stats=RolloutStats(),
    )
    assert first.seen[0]["collector"] is fake and first.seen[0]["group_size"] == 3
    assert [batch.group_ids.tolist() for batch in batches] == [[0] * 3, [1] * 3, [2] * 3]


@pytest.mark.asyncio
async def test_owned_items_do_not_mix_with_plain_prompts() -> None:
    fake = PromptCollectionFake()
    with pytest.raises(ValueError, match="cannot share"):
        await fake.prepare_training_batches(
            prompts=[_Owned(1), "plain"],
            group_size=1,
            runtime_debug=False,
            policy_version=None,
            stats=RolloutStats(),
        )
    with pytest.raises(ValueError, match="prepare_training_batches"):
        list(
            fake.build_generation_requests(
                prompts=[_Owned(1)], group_size=1, runtime_debug=False, policy_version=None
            )
        )
