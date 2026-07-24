"""Generic Ray actor and placement-group lifecycle helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable
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


def kill_and_retain(
    ray: Any,
    items: list[Any],
    get_actor: Callable[[Any], Any],
) -> tuple[list[Any], list[tuple[Any, Exception]]]:
    """Kill each item's actor and retain only the items whose kill FAILED.

    The two Ray lifecycle owners (``RayActorGroup.shutdown`` and the generation
    runtime shutdown) both own lifecycle truth: they drop handles for actors that
    died and keep the ones they could not kill so cleanup is not falsely reported
    complete. ``get_actor`` maps one owned item to its Ray actor handle (or
    ``None`` when already released). Returns ``(surviving, failures)`` where
    ``surviving`` are the still-owned items whose actor kill raised and
    ``failures`` are the ``(actor, error)`` pairs from :func:`kill_actors`.
    """

    actors = [actor for item in items if (actor := get_actor(item)) is not None]
    failures = kill_actors(ray, actors)
    failed_actor_ids = {id(actor) for actor, _ in failures}
    surviving = [
        item
        for item in items
        if (actor := get_actor(item)) is not None and id(actor) in failed_actor_ids
    ]
    return surviving, failures


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


__all__ = ["kill_actors", "kill_and_retain", "remove_placement_group"]
