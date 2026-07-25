"""Trainer replay and runtime model contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Protocol, cast, runtime_checkable


@dataclass(frozen=True, slots=True)
class ReplayRequest:
    """Options for replaying a recorded trajectory."""

    segment_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.segment_names is None:
            return
        for name in self.segment_names:
            if not isinstance(name, str) or not name:
                raise ValueError("ReplayRequest.segment_names must contain non-empty strings")


@dataclass(slots=True)
class ReplaySegmentResult:
    """Current replay result for one trajectory segment."""

    segment: str
    values: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.segment, str) or not self.segment:
            raise ValueError("ReplaySegmentResult.segment must be a non-empty string")
        if not isinstance(self.values, dict):
            raise TypeError("ReplaySegmentResult.values must be a dict")

    def require_value(self, key: str) -> Any:
        """Return a named replay payload or fail with the available payload keys."""

        if key not in self.values:
            raise KeyError(
                f"ReplaySegmentResult for segment {self.segment!r} "
                f"missing required key {key!r}; got {sorted(self.values)}",
            )
        return self.values[key]

    def logprobs(self, token_ids: Any | None = None, *, temperature: float = 1.0) -> Any:
        """Per-token log-probs for this segment.

        Families either store them directly (``log_probs``) or store logits
        under the ``logits`` key. That field-name knowledge lives here with the
        payload contract, so consumers (evaluators) do not switch on payload
        keys and a new modality only touches this method.

        ``temperature`` is the rollout sampling temperature (recorded in the
        rollout context). It only applies on the logits path: rollout scoring
        divides logits by temperature, so replay must renormalize with the
        same temperature to keep old/new log-prob parity. Directly stored
        ``log_probs`` were already scaled at rollout time.
        """
        direct = self.values.get("log_probs")
        if direct is not None:
            return direct.float()

        logits = self.require_value("logits")  # raises with available keys

        if token_ids is None:
            token_ids = self.require_value("token_ids")

        from vrl.math.token.logprob import gather_categorical_log_probs

        return gather_categorical_log_probs(logits, token_ids, temperature=temperature)


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


def require_replay_segments(
    request: ReplayRequest | None,
    supported_segments: tuple[str, ...],
    *,
    owner: str,
) -> None:
    """Reject an explicit segment selection the replay implementation cannot honor."""

    if request is None or request.segment_names is None:
        return
    requested = request.segment_names
    unsupported = tuple(name for name in requested if name not in supported_segments)
    if requested and not unsupported:
        return
    raise ValueError(
        f"{owner} replay supports segments {supported_segments!r}; got {requested!r}",
    )


def require_zero_replay_timestep(timestep_idx: int, *, owner: str) -> None:
    """Reject denoise-step selection for a replay path with no timestep axis."""

    if timestep_idx != 0:
        raise ValueError(
            f"{owner} replay has no timestep axis; timestep_idx must be 0, got {timestep_idx}",
        )


def single_segment_result(name: str, values: dict[str, Any]) -> ReplayResult:
    """Wrap one segment's payload as a ``ReplayResult`` (key == segment name)."""

    return ReplayResult(
        segments={name: ReplaySegmentResult(segment=name, values=values)},
    )


def replay_context_image_size(
    batch: Any,
    *,
    token_count: int,
    expected_token_num: Callable[[int, int], int],
    owner: str,
) -> tuple[int, int]:
    """Read and validate the rollout image size from the trajectory context.

    The executor stores pixel ``image_height``/``image_width`` in the trajectory
    context (NOT derivable from the token count alone for non-square ratios).
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
    height, width = int(height), int(width)
    expected = expected_token_num(height, width)
    if expected != token_count:
        raise RuntimeError(
            f"{owner} replay token count {token_count} does not match the "
            f"{height}x{width} image size (expected {expected}).",
        )
    return height, width


# Derived from the Protocol definitions (same pattern as the contract tests),
# so a method add/rename auto-widens the runtime checks and error messages.
_REPLAY_MODEL_METHODS = tuple(sorted(ReplayModel.__protocol_attrs__))
_RUNTIME_MODEL_METHODS = tuple(sorted(RuntimeModel.__protocol_attrs__))


def require_replay_model(value: Any, *, owner: str = "model") -> ReplayModel:
    """Return ``value`` as a ReplayModel or fail at the replay boundary."""

    if isinstance(value, ReplayModel):
        return cast("ReplayModel", value)
    missing = _missing_callables(value, _REPLAY_MODEL_METHODS)
    detail = f"; missing: {', '.join(missing)}" if missing else ""
    raise TypeError(
        f"{owner} must satisfy ReplayModel({', '.join(_REPLAY_MODEL_METHODS)}){detail}",
    )


def require_runtime_model(value: Any, *, owner: str = "model") -> RuntimeModel:
    """Return ``value`` as a RuntimeModel or fail at the runtime boundary."""

    if isinstance(value, RuntimeModel):
        return cast("RuntimeModel", value)
    missing = _missing_callables(value, _RUNTIME_MODEL_METHODS)
    detail = f"; missing: {', '.join(missing)}" if missing else ""
    raise TypeError(
        f"{owner} must satisfy RuntimeModel({', '.join(_RUNTIME_MODEL_METHODS)}){detail}",
    )


def _missing_callables(value: Any, names: tuple[str, ...]) -> list[str]:
    return [name for name in names if not callable(getattr(value, name, None))]


__all__ = [
    "ReplayModel",
    "ReplayRequest",
    "ReplayResult",
    "ReplaySegmentResult",
    "RuntimeModel",
    "replay_context_image_size",
    "require_replay_model",
    "require_replay_segments",
    "require_runtime_model",
    "require_zero_replay_timestep",
    "single_segment_result",
]
