"""Multi-reward registry — weighted combination of named reward functions.

Ported from the multi_score() pattern in flow_grpo/rewards.py.
"""

from __future__ import annotations

from typing import Any

from vrl.rewards.base import RewardCleanupError, RewardFunction
from vrl.rewards.inference import RewardMemoryReleaseProof
from vrl.rewards.types import RewardRollout

# Registry of reward function factories.
# Each factory takes (device,) and returns a RewardFunction instance.
_REWARD_REGISTRY: dict[str, type[RewardFunction]] = {}


def get_reward(name: str) -> type[RewardFunction]:
    """Look up a registered reward function class by name."""
    if name not in _REWARD_REGISTRY:
        raise KeyError(f"Unknown reward function: {name!r}. Available: {list(_REWARD_REGISTRY)}")
    return _REWARD_REGISTRY[name]


def _register_builtins() -> None:
    from vrl.rewards.functions.aesthetic import AestheticReward
    from vrl.rewards.functions.cosmos3_reasoner import Cosmos3ReasonerReward
    from vrl.rewards.functions.geneval import GenEvalReward
    from vrl.rewards.functions.kling_video_reward import KlingVideoReward
    from vrl.rewards.functions.motion_dynamics import MotionDynamicsReward
    from vrl.rewards.functions.nsfw_safety import NSFWSafetyReward
    from vrl.rewards.functions.ocr import OCRReward
    from vrl.rewards.functions.phymotion import PhyMotionReward
    from vrl.rewards.functions.pickscore import PickScoreReward
    from vrl.rewards.functions.target_dino_similarity import TargetDinoSimilarityReward
    from vrl.rewards.functions.unified_reward_video import UnifiedRewardVideoReward
    from vrl.rewards.functions.videocon_physics import VideoConPhysicsReward
    from vrl.rewards.functions.videoscore2 import VideoScore2Reward

    _REWARD_REGISTRY.update(
        {
            "aesthetic": AestheticReward,
            "geneval": GenEvalReward,
            "nsfw_safety": NSFWSafetyReward,
            "ocr": OCRReward,
            "pickscore": PickScoreReward,
            # Future Reward suite (SPRINT_future_reward): DINOv2 perceptual anchor + RAFT
            # motion guard. They replaced the deleted pixel-L1 target_video_similarity (it was
            # reward-hackable, see S1). The IDM action-following signal is designed in S3 but
            # not shipped.
            "target_dino_similarity": TargetDinoSimilarityReward,
            "motion_dynamics": MotionDynamicsReward,
            "kling_video_reward": KlingVideoReward,
            "cosmos3_reasoner": Cosmos3ReasonerReward,
            "videocon_physics": VideoConPhysicsReward,
            "videoscore2": VideoScore2Reward,
            "unified_reward_video": UnifiedRewardVideoReward,
            "phymotion": PhyMotionReward,
        }
    )


class MultiReward(RewardFunction):
    """Weighted combination of named reward functions.

    Component scores are returned with the totals for the same scoring call.
    They deliberately do not live on this shared object: continuous rollout
    can score future groups while the trainer consumes an older group, so a
    mutable last-value cache would attach metrics to whichever call finished
    last instead of to the batch being trained.

    Usage::

        reward_fn = MultiReward.from_dict(
            {"ocr": 1.0, "aesthetic": 0.3},
            device="cuda",
        )
        totals, components = await reward_fn.score_batch_with_components([rollout])
        # components -> {"ocr": [0.87], "aesthetic": [5.2]}
    """

    def __init__(
        self,
        rewards: list[tuple[str, float, RewardFunction]],
    ) -> None:
        super().__init__()
        self.rewards = rewards
        # Composite teardown is retryable: remember children whose shutdown
        # already succeeded so a retry reaches only the ones that actually
        # failed instead of double-shutting siblings.
        self._shutdown_completed_children: set[int] = set()

    async def shutdown(self) -> None:
        errors: list[BaseException] = []
        for _, _, fn in self.rewards:
            child_id = id(fn)
            if child_id in self._shutdown_completed_children:
                continue
            try:
                await fn.shutdown()
            except BaseException as error:
                errors.append(error)
            else:
                self._shutdown_completed_children.add(child_id)
        if errors:
            raise RewardCleanupError("reward shutdown failures", errors)

    @classmethod
    def from_dict(
        cls,
        score_dict: dict[str, float],
        device: str = "cuda",
        reward_kwargs: dict[str, dict[str, Any]] | None = None,
        memory_parking_required: bool | None = None,
    ) -> MultiReward:
        """Build from ``{"name": weight}`` dict, like flow_grpo config.reward_fn.

        ``reward_kwargs`` allows passing per-reward init kwargs, keyed by name,
        e.g. ``{"ocr": {"debug_dir": "out/ocr_debug"}}``.
        """
        _register_builtins()
        reward_kwargs = reward_kwargs or {}
        configured_weights = {name: float(weight) for name, weight in score_dict.items()}
        reward_classes = {name: get_reward(name) for name in configured_weights}
        if memory_parking_required:
            validate_reward_memory_parking_components(
                tuple(reward_classes),
                device=device,
                reward_kwargs=reward_kwargs,
            )

        triples: list[tuple[str, float, RewardFunction]] = []
        for name, weight in configured_weights.items():
            reward_cls = reward_classes[name]
            # `or {}`: a bare YAML key (kwargs: <name>:) parses as None.
            extra = dict(reward_kwargs.get(name) or {})
            if "execution" in extra:
                raise ValueError(
                    f"reward.kwargs.{name}.execution is no longer supported: the "
                    "Ray reward pool was removed and rewards score in-process. "
                    "Drop the key; shared-GPU parking is derived from distributed "
                    "resource topology.",
                )
            component_device = reward_cls.resolve_execution_device(
                device=device,
                kwargs=extra,
            )
            # The resolved value is passed once through the constructor's common
            # device argument; remove a component override after it has served as
            # the CPU-downgrade input.
            extra.pop("device", None)
            if memory_parking_required is True and component_device.startswith("cuda"):
                # GPU ownership comes from topology. A shared reward cannot rely
                # on every preset remembering an independent parking knob.
                parking = reward_cls.memory_parking
                if parking is None:
                    raise ValueError(
                        f"reward {name!r} has no complete memory-parking contract",
                    )
                extra["sleep_offload"] = True
                extra["memory_parking_residual_bytes_limit"] = int(
                    parking.residual_bytes_limit,
                )
            elif memory_parking_required is not None:
                # A dedicated reward owns its GPU and remains resident even if
                # an inherited reward preset carried the old shared-phase knob.
                # CPU-only components also never receive a GPU parking knob.
                extra.pop("sleep_offload", None)
            triples.append(
                (
                    name,
                    weight,
                    reward_cls(device=component_device, **extra),
                ),
            )
        return cls(triples)

    async def score(self, rollout: RewardRollout) -> float:
        totals, _ = await self.score_batch_with_components([rollout])
        return totals[0]

    async def score_batch(self, rollouts: list[RewardRollout]) -> list[float]:
        totals, _ = await self.score_batch_with_components(rollouts)
        return totals

    async def score_batch_with_components(
        self,
        rollouts: list[RewardRollout],
    ) -> tuple[list[float], dict[str, list[float]]]:
        """Return totals and batch-aligned raw components from one call."""

        self.last_results = []
        self.last_timing_ms = {}
        totals = [0.0] * len(rollouts)
        components: dict[str, list[float]] = {}
        operation_error: BaseException | None = None
        try:
            for name, weight, fn in self.rewards:
                sub_scores = await fn.score_batch(rollouts)
                self._append_inference_observations(fn)
                components[name] = list(sub_scores)
                for i, s in enumerate(sub_scores):
                    totals[i] += weight * s
        except BaseException as error:
            operation_error = error
        _, cleanup_error = await self._park_all_memory()
        _raise_operation_and_cleanup(operation_error, cleanup_error)
        return totals, components

    async def park_memory(self) -> tuple[RewardMemoryReleaseProof, ...]:
        """Actively park every component; never infer release from cached state."""

        proofs, error = await self._park_all_memory()
        if error is not None:
            raise error
        return proofs

    async def _park_all_memory(
        self,
    ) -> tuple[tuple[RewardMemoryReleaseProof, ...], BaseException | None]:
        proofs: list[RewardMemoryReleaseProof] = []
        errors: list[BaseException] = []
        for name, _, fn in self.rewards:
            try:
                proofs.extend(await fn.park_memory())
            except BaseException as error:
                errors.append(RuntimeError(f"reward component {name!r} failed to park"))
                errors[-1].__cause__ = error
        if errors:
            return tuple(proofs), RewardCleanupError("reward memory parking failures", errors)
        return tuple(proofs), None

    def _append_inference_observations(self, fn: RewardFunction) -> None:
        self.last_results.extend(list(getattr(fn, "last_results", []) or []))
        for key, value in (getattr(fn, "last_timing_ms", {}) or {}).items():
            self.last_timing_ms[str(key)] = self.last_timing_ms.get(str(key), 0.0) + float(value)


def validate_reward_memory_parking_components(
    names: tuple[str, ...],
    *,
    device: str = "cuda",
    reward_kwargs: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Fail before model construction when a shared reward cannot fully park."""

    _register_builtins()
    kwargs_by_name = reward_kwargs or {}
    gpu_components = [
        name
        for name in names
        if get_reward(name)
        .resolve_execution_device(
            device=device,
            kwargs=dict(kwargs_by_name.get(name) or {}),
        )
        .startswith("cuda")
    ]
    if not gpu_components:
        raise ValueError(
            "shared reward GPU topology has no configured GPU reward "
            "component. Declare the reward as CPU execution "
            "(num_gpus=0, gpus_per_worker=0, num_workers=1) instead.",
        )
    if len(gpu_components) > 1:
        raise ValueError(
            "shared reward parking supports at most one configured GPU "
            f"component per process, got {gpu_components}. vLLM CuMemAllocator.sleep "
            "is process-wide: tags select which pages are backed up, not which pages "
            "are unmapped. Keep CPU reward siblings or use a dedicated/remote reward.",
        )
    unsupported = [name for name in gpu_components if get_reward(name).memory_parking is None]
    if unsupported:
        raise ValueError(
            "shared reward GPU requires complete topology-driven memory parking, "
            f"but these reward components do not provide it: {unsupported}. "
            "Use a dedicated reward GPU or a reward with complete parking support.",
        )


def _raise_operation_and_cleanup(
    operation_error: BaseException | None,
    cleanup_error: BaseException | None,
) -> None:
    if operation_error is not None and cleanup_error is not None:
        raise RewardCleanupError(
            "reward operation and memory parking both failed",
            [operation_error, cleanup_error],
        )
    if operation_error is not None:
        raise operation_error
    if cleanup_error is not None:
        raise cleanup_error
