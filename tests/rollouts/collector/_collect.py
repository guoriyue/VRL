"""Test-side one-shot collect helper.

Production drives ``collect_unscored`` + ``score_rollouts`` separately
(``collect_prompt_batches``); this convenience wrapper exists only for tests
that want a single scored batch.
"""

from __future__ import annotations

from typing import Any

from vrl.rollouts.batch import RolloutBatch


async def collect_scored(collector: Any, inputs: list[Any], **kwargs: Any) -> RolloutBatch:
    unscored = await collector.collect_unscored(inputs, **kwargs)
    return (await collector.score_rollouts([unscored]))[0]
