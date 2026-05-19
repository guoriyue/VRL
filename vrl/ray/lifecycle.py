"""Generic Ray actor and placement-group lifecycle helpers."""

from __future__ import annotations

import contextlib
from typing import Any


def kill_actors(ray: Any, actors: list[Any]) -> None:
    """Best-effort kill for Ray actors."""

    for actor in actors:
        with contextlib.suppress(Exception):
            ray.kill(actor, no_restart=True)


def remove_placement_group(placement_group: Any) -> None:
    """Best-effort removal for a Ray placement group."""

    if placement_group is None:
        return
    with contextlib.suppress(Exception):
        from ray.util import remove_placement_group as _remove_placement_group

        _remove_placement_group(placement_group)


__all__ = ["kill_actors", "remove_placement_group"]
