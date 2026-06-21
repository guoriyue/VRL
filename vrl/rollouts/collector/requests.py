"""Rollout-to-generation request adapter for collectors."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any, NamedTuple

from vrl.generation import GenerationRequest


class CollectorRequest(NamedTuple):
    request: GenerationRequest
    metadata: dict[str, Any]


class GenerationRequestBuilder:
    """Build ``GenerationRequest`` payloads from resolved rollout config."""

    def __init__(
        self,
        *,
        family: str,
        task: str,
        request_prefix: str,
        config: Any,
        return_artifacts: tuple[str, ...],
        default_task_type: str | None = None,
        metadata_key: str | None = None,
    ) -> None:
        if not return_artifacts:
            raise ValueError(f"{family} request builder requires return_artifacts")
        self.family = family
        self.task = task
        self.request_prefix = request_prefix
        self.config = config
        self.return_artifacts = return_artifacts
        self.default_task_type = default_task_type
        self.metadata_key = metadata_key

    def build(
        self,
        prompts: list[str],
        group_size: int,
        kwargs: dict[str, Any],
    ) -> CollectorRequest:
        seed = kwargs.get("seed")
        policy_version = kwargs.get("policy_version")
        sampling = self._sampling()
        if seed is not None:
            sampling["seed"] = seed
        sampling.update(dict(kwargs.get("request_overrides", {})))

        metadata = self._metadata(kwargs)
        if "fps" in sampling:
            metadata.setdefault("video_fps", sampling["fps"])
        request_metadata = dict(metadata)
        if self.metadata_key is not None:
            request_metadata = {self.metadata_key: dict(metadata)}
        if kwargs.get("runtime_debug"):
            request_metadata["_runtime_debug"] = True

        request = GenerationRequest(
            request_id=f"{self.request_prefix}-{uuid.uuid4()}",
            family=self.family,
            task=self.task,
            prompts=prompts,
            # GenerationRequest names this `samples_per_prompt` (generation-domain
            # wording); the value is the collector's `group_size` — the GRPO group,
            # sourced from rollout.n_samples_per_prompt. Same number, three domain
            # names (config / GRPO collector / generation request), each accurate to
            # its layer; distinct from the eval-only trainer.eval.samples_per_prompt.
            samples_per_prompt=group_size,
            sampling=sampling,
            return_artifacts=set(self.return_artifacts),
            metadata=request_metadata,
            policy_version=policy_version,
        )
        return CollectorRequest(
            request=request,
            metadata=metadata,
        )

    def _sampling(self) -> dict[str, Any]:
        request_sampling = getattr(self.config, "request_sampling", None)
        if callable(request_sampling):
            raw_sampling = request_sampling()
        else:
            raw_sampling = getattr(self.config, "values", None)
        if not isinstance(raw_sampling, Mapping):
            raise TypeError(
                f"{type(self.config).__name__} must expose "
                "request_sampling() or values",
            )
        return {
            str(field_name): list(value) if isinstance(value, tuple) else value
            for field_name, value in raw_sampling.items()
        }

    def _metadata(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        sample_metadata = kwargs.get("sample_metadata")
        if sample_metadata:
            metadata.update(sample_metadata)
        target_text = kwargs.get("target_text")
        if target_text:
            metadata["target_text"] = target_text
        references = kwargs.get("references")
        if references:
            metadata["references"] = references
        if self.default_task_type is not None:
            metadata["task_type"] = kwargs.get("task_type", self.default_task_type)
            reference_image = kwargs.get("reference_image")
            if reference_image is not None:
                metadata["reference_image"] = reference_image
            reference_video = kwargs.get("reference_video")
            if reference_video is not None:
                metadata["reference_video"] = reference_video
        return metadata


__all__ = [
    "CollectorRequest",
    "GenerationRequestBuilder",
]
