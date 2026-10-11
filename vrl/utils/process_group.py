"""This process's default torch.distributed group and its one owner.

torch keeps one default process group per OS process. In VRL two kinds of
process create one: a trainer rank (the group of all trainer ranks) and a rank
of a multi-GPU generation engine (the group of that engine's ranks). Both are
the same lifecycle, so it lives here once; the trainer and the engine only say
how their ranks meet and what they build on top.
"""

from __future__ import annotations

from typing import Any, Self


class DefaultProcessGroup:
    """This process's default torch.distributed group, owned only by this object.

    Creating refuses a group that already exists: a second owner would disagree
    about its lifetime. ``shutdown`` destroys the group together with every
    subgroup built on it, keeps the handle if that fails so a retry can
    destroy it, and does nothing once it succeeded. Subclasses construct
    through ``_create``.
    """

    def __init__(self) -> None:
        # The default group this object created; None once destroyed.
        self._world: Any = None

    @classmethod
    def _create(cls, *, backend: str, rank: int, world_size: int, **init_kwargs: Any) -> Self:
        import torch.distributed as dist

        if dist.is_initialized():
            raise RuntimeError(
                "a torch.distributed process group is already initialized; "
                "the object that creates it is its only owner",
            )
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size, **init_kwargs)
        owner = cls()
        owner._world = dist.group.WORLD
        return owner

    @property
    def world(self) -> Any:
        """The default group handle, or ``None`` when this object holds no group."""

        return self._world

    def shutdown(self) -> None:
        """Destroy the group and its subgroups; safe to repeat."""

        import torch.distributed as dist

        world = self._world
        if world is None:
            return
        if dist.is_initialized():
            # Destroying the default group destroys every subgroup with it.
            dist.destroy_process_group(world)
        self._world = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.shutdown()


__all__ = ["DefaultProcessGroup"]
