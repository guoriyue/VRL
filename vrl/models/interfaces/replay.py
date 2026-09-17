"""Trainer replay and runtime model contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, cast, runtime_checkable

from vrl.utils.validation import require_int


@dataclass(frozen=True, slots=True)
class ReplayRequest:
    """Options for replaying a recorded trajectory."""

    segment_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.segment_names is None:
            return
        if isinstance(self.segment_names, (str, bytes)):
            raise ValueError("ReplayRequest.segment_names must be a sequence of names")
        segment_names = tuple(self.segment_names)
        for name in segment_names:
            if not isinstance(name, str) or not name:
                raise ValueError("ReplayRequest.segment_names must contain non-empty strings")
        object.__setattr__(self, "segment_names", segment_names)


@dataclass(slots=True)
class ReplaySegmentResult:
    """Current replay result for one trajectory segment."""

    segment: str
    values: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.segment, str) or not self.segment:
            raise ValueError("ReplaySegmentResult.segment must be a non-empty string")

    def require_value(self, key: str) -> Any:
        """Return a named replay payload or fail with the available payload keys."""

        if key not in self.values:
            raise KeyError(
                f"ReplaySegmentResult for segment {self.segment!r} "
                f"missing required key {key!r}; got {sorted(self.values)}",
            )
        return self.values[key]


@dataclass(slots=True)
class ReplayResult:
    """Current replay result for a rollout batch."""

    segments: dict[str, ReplaySegmentResult]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("ReplayResult.segments must be non-empty")
        for key, segment in self.segments.items():
            if not isinstance(segment, ReplaySegmentResult):
                raise TypeError(f"ReplayResult segment {key!r} must be a ReplaySegmentResult")
            if key != segment.segment:
                raise ValueError(
                    f"ReplayResult segment key {key!r} must match "
                    f"ReplaySegmentResult.segment={segment.segment!r}",
                )

    @classmethod
    def from_segment(cls, name: str, values: dict[str, Any]) -> ReplayResult:
        """Build one segment with its mapping key and segment name aligned."""

        return cls(segments={name: ReplaySegmentResult(segment=name, values=values)})

    def require_segment(self, segment_name: str) -> ReplaySegmentResult:
        """Return a replay segment result or fail with a helpful segment list."""

        try:
            return self.segments[segment_name]
        except KeyError as err:
            raise KeyError(
                f"ReplayResult missing segment {segment_name!r}; got {sorted(self.segments)}",
            ) from err


@runtime_checkable
class ReplayModel(Protocol):
    """Minimal model contract consumed by replay evaluators."""

    def replay_forward(
        self,
        batch: Any,
        timestep_idx: int = 0,
        *,
        request: ReplayRequest | None = None,
    ) -> ReplayResult:
        """Replay recorded trajectory actions under the current model."""
        ...

    def disable_adapter(self) -> AbstractContextManager[None]:
        """Temporarily disable adapters, or return a no-op context."""
        ...


@runtime_checkable
class RuntimeModel(ReplayModel, Protocol):
    """Minimum model contract shared by trainer runtime and Ray sync."""

    def load_trainable_state(self, state_dict: Mapping[str, Any]) -> Any:
        """Load the flattened trainable-state payload pushed to rollout workers."""
        ...


class ReplayRequestContract:
    """Per-type replay request contract: which segments a model can replay and
    whether it indexes a timestep axis.

    Both were literals at each ``replay_forward`` call site with the model's own
    ``type(self).__name__`` threaded in as ``owner=`` — the ownership signal
    from the placement rules, since the type owns both facts. Declaring them
    makes "what can this family replay?" answerable from the class header, and
    the identity is read where it lives instead of passed around.

    The two checks stay separate methods because they are separate theorems:
    janus_pro's supported set is request-dependent (its R1 path validates a
    different set than its default one), so it needs the segment check on its
    own branch while the timestep check always runs.
    """

    # No default: a replay model that forgets to declare its segments raises
    # AttributeError at the first validation instead of silently accepting any
    # selection.
    replay_segments: ClassVar[tuple[str, ...]]
    # Denoise families index scheduler timesteps; token families replay one
    # whole sequence and have no timestep axis to select.
    replay_indexes_timesteps: ClassVar[bool] = True

    def reject_replay_timestep_selection(self, timestep_idx: int) -> None:
        """No-op for families that do index timesteps."""

        if self.replay_indexes_timesteps or timestep_idx == 0:
            return
        raise ValueError(
            f"{type(self).__name__} replay has no timestep axis; "
            f"timestep_idx must be 0, got {timestep_idx}",
        )

    def reject_unsupported_replay_segments(
        self,
        request: ReplayRequest | None,
        *,
        segments: tuple[str, ...] | None = None,
    ) -> None:
        """``segments`` overrides the declared set for a request-dependent contract."""

        if request is None or request.segment_names is None:
            return
        supported = self.replay_segments if segments is None else segments
        requested = request.segment_names
        unsupported = tuple(name for name in requested if name not in supported)
        if requested and not unsupported:
            return
        raise ValueError(
            f"{type(self).__name__} replay supports segments {supported!r}; got {requested!r}",
        )


def replay_context_image_size(
    batch: Any,
    *,
    token_count: int,
    expected_token_num: Callable[[int, int], int],
    owner: str,
) -> tuple[int, int]:
    """Read and validate the rollout image size from the trajectory context.

    The executor records ``image_height``/``image_width`` in family-defined
    units: Emu3 latent grid dimensions or GLM-Image pixel dimensions. These
    cannot be derived from token count alone for non-square ratios.
    Returns ``(height, width)`` after confirming the recorded size reproduces
    ``token_count`` under the family ``expected_token_num``.
    """

    context = getattr(batch, "context", None) or {}
    trajectory = getattr(batch, "trajectory", None)
    if not context and trajectory is not None:
        context = getattr(trajectory, "context", None) or {}
    height = context.get("image_height")
    width = context.get("image_width")
    if height is None or width is None:
        raise RuntimeError(
            f"{owner} replay requires image_height/image_width in the rollout "
            "context to rebuild the replay token schedule.",
        )
    height = require_int(height, path=f"{owner} replay image_height", minimum=1)
    width = require_int(width, path=f"{owner} replay image_width", minimum=1)
    expected = expected_token_num(height, width)
    if expected != token_count:
        raise RuntimeError(
            f"{owner} replay token count {token_count} does not match the "
            f"{height}x{width} image size (expected {expected}).",
        )
    return height, width


def _require_protocol(value: Any, proto: Any, *, owner: str) -> Any:
    """isinstance-or-raise shared by the boundary casts below; the method list
    derives from the Protocol (same pattern as the contract tests), so a method
    add/rename auto-widens the runtime checks and error messages."""

    methods = tuple(sorted(proto.__protocol_attrs__))
    missing = [name for name in methods if not callable(getattr(value, name, None))]
    if not missing and isinstance(value, proto):
        return value
    detail = f"; missing: {', '.join(missing)}" if missing else ""
    raise TypeError(f"{owner} must satisfy {proto.__name__}({', '.join(methods)}){detail}")


def require_replay_model(value: Any, *, owner: str = "model") -> ReplayModel:
    """Return ``value`` as a ReplayModel or fail at the replay boundary."""

    return cast("ReplayModel", _require_protocol(value, ReplayModel, owner=owner))


def require_runtime_model(value: Any, *, owner: str = "model") -> RuntimeModel:
    """Return ``value`` as a RuntimeModel or fail at the runtime boundary."""

    return cast("RuntimeModel", _require_protocol(value, RuntimeModel, owner=owner))


__all__ = [
    "ReplayModel",
    "ReplayRequest",
    "ReplayResult",
    "ReplaySegmentResult",
    "RuntimeModel",
    "replay_context_image_size",
    "require_replay_model",
    "require_runtime_model",
]
