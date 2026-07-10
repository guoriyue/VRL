"""Generic Ray actor and placement-group lifecycle helpers."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def kill_actors(ray: Any, actors: list[Any]) -> None:
    """Best-effort kill for Ray actors."""

    for actor in actors:
        try:
            ray.kill(actor, no_restart=True)
        except Exception:
            logger.warning("Failed to kill owned Ray actor %r", actor, exc_info=True)


def remove_placement_group(placement_group: Any) -> None:
    """Best-effort removal for a Ray placement group."""

    if placement_group is None:
        return
    try:
        from ray.util import remove_placement_group as _remove_placement_group

        _remove_placement_group(placement_group)
    except Exception:
        logger.warning(
            "Failed to remove owned Ray placement group %r",
            placement_group,
            exc_info=True,
        )


__all__ = ["kill_actors", "remove_placement_group"]
