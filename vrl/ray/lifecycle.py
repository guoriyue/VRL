"""Generic Ray actor and placement-group lifecycle helpers."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def kill_actors(ray: Any, actors: list[Any]) -> list[tuple[Any, Exception]]:
    """Best-effort kill actors and return failures to the resource owner.

    Framework adapters log locally, while callers that own lifecycle truth can
    retain failed handles and refuse to report cleanup as complete.
    """

    failures: list[tuple[Any, Exception]] = []
    for actor in actors:
        try:
            ray.kill(actor, no_restart=True)
        except Exception as error:
            failures.append((actor, error))
            logger.warning("Failed to kill owned Ray actor %r", actor, exc_info=True)
    return failures


def remove_placement_group(placement_group: Any) -> Exception | None:
    """Best-effort removal, returning an error to the resource owner."""

    if placement_group is None:
        return None
    try:
        from ray.util import remove_placement_group as _remove_placement_group

        _remove_placement_group(placement_group)
    except Exception as error:
        logger.warning(
            "Failed to remove owned Ray placement group %r",
            placement_group,
            exc_info=True,
        )
        return error
    return None


__all__ = ["kill_actors", "remove_placement_group"]
