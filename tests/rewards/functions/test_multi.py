"""Tests for vrl.rewards.functions.registry."""

from __future__ import annotations

import pytest

from vrl.rewards.base import RewardBatchReport, RewardCleanupError, RewardFunction
from vrl.rewards.functions.registry import MultiReward
from vrl.rewards.runtime import InProcessRewardRuntime
from vrl.rewards.service.client import HttpRewardRuntime
from vrl.rewards.types import RewardRollout
from vrl.utils.cuda_memory import CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT


def _make_rollout(prompt: str) -> RewardRollout:
    return RewardRollout(
        prompt=prompt,
        output=None,
        source_request_id="request-0",
        sample_id=f"sample-{prompt}",
        group_id="group-0",
        trajectory_id=f"trajectory-{prompt}",
    )


class _QueuedBatchReward(RewardFunction):
    def __init__(self, batches: list[list[float]]) -> None:
        super().__init__()
        self.batches = list(batches)

    async def score(self, rollout: RewardRollout) -> float:
        return self.batches.pop(0)[0]

    async def score_batch(self, rollouts: list[RewardRollout]) -> list[float]:
        scores = self.batches.pop(0)
        assert len(scores) == len(rollouts)
        return scores


class _TimedBatchReward(RewardFunction):
    def __init__(self, scores: list[float], timing_ms: dict[str, float]) -> None:
        super().__init__()
        self.scores = list(scores)
        self.timing_ms = dict(timing_ms)

    async def score_batch_report(self, rollouts: list[RewardRollout]) -> RewardBatchReport:
        assert len(rollouts) == len(self.scores)
        return RewardBatchReport(
            scores=list(self.scores),
            timing_ms=dict(self.timing_ms),
        )


@pytest.mark.asyncio
async def test_multi_reward_preserves_typed_rollout_lineage_for_every_component() -> None:
    seen: dict[str, list[tuple[str, str, str, str]]] = {}

    class _CaptureReward(RewardFunction):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        async def score_batch_report(
            self,
            rollouts: list[RewardRollout],
        ) -> RewardBatchReport:
            seen[self.name] = [
                (
                    rollout.source_request_id,
                    rollout.sample_id,
                    rollout.group_id,
                    rollout.trajectory_id,
                )
                for rollout in rollouts
            ]
            return RewardBatchReport(scores=[1.0] * len(rollouts))

    reward = MultiReward(
        [
            ("first", 1.0, _CaptureReward("first")),
            ("second", 1.0, _CaptureReward("second")),
        ],
    )
    rollouts = [_make_rollout("a"), _make_rollout("b")]

    await reward.score_batch_report(rollouts)

    expected = [
        ("request-0", "sample-a", "group-0", "trajectory-a"),
        ("request-0", "sample-b", "group-0", "trajectory-b"),
    ]
    assert seen == {"first": expected, "second": expected}


@pytest.mark.asyncio
async def test_multi_reward_returns_components_for_each_scoring_call() -> None:
    """Component observations stay aligned to their own score batch."""
    reward = MultiReward(
        [
            ("ocr", 1.0, _QueuedBatchReward([[0.1, 0.2], [0.3]])),
            ("aesthetic", 0.5, _QueuedBatchReward([[1.0, 2.0], [3.0]])),
        ]
    )

    first = await reward.score_batch_report(
        [_make_rollout("a"), _make_rollout("b")],
    )
    second = await reward.score_batch_report(
        [_make_rollout("c")],
    )

    assert first.scores == pytest.approx([0.6, 1.2])
    assert second.scores == pytest.approx([1.8])
    assert first.components == pytest.approx(
        {"ocr": [0.1, 0.2], "aesthetic": [1.0, 2.0]},
    )
    assert second.components == pytest.approx(
        {"ocr": [0.3], "aesthetic": [3.0]},
    )


@pytest.mark.asyncio
async def test_zero_weight_component_is_scored_without_changing_total() -> None:
    """Checks observation-only scores are logged but excluded from the reward."""
    reward = MultiReward(
        [
            ("train", 1.0, _QueuedBatchReward([[2.0, 3.0]])),
            ("observe", 0.0, _QueuedBatchReward([[0.7, 0.8]])),
        ],
    )

    report = await reward.score_batch_report(
        [_make_rollout("a"), _make_rollout("b")],
    )

    assert report.scores == pytest.approx([2.0, 3.0])
    assert report.components == pytest.approx(
        {"train": [2.0, 3.0], "observe": [0.7, 0.8]},
    )


@pytest.mark.asyncio
async def test_multi_reward_aggregates_inference_observations() -> None:
    """Checks multi reward exposes child reward inference timings."""
    reward = MultiReward(
        [
            (
                "first",
                1.0,
                _TimedBatchReward(
                    [0.1, 0.2],
                    {
                        "latency_ms": 5.0,
                        "queue_wait_ms": 1.0,
                        "inference_ms": 4.0,
                    },
                ),
            ),
            (
                "second",
                2.0,
                _TimedBatchReward(
                    [1.0, 2.0],
                    {
                        "latency_ms": 7.0,
                        "queue_wait_ms": 3.0,
                        "inference_ms": 6.0,
                    },
                ),
            ),
        ],
    )

    report = await reward.score_batch_report([_make_rollout("a"), _make_rollout("b")])

    assert report.scores == pytest.approx([2.1, 4.2])
    assert report.timing_ms == pytest.approx(
        {
            "latency_ms": 12.0,
            "queue_wait_ms": 4.0,
            "inference_ms": 10.0,
        },
    )


@pytest.mark.asyncio
async def test_multi_reward_parks_every_child_after_score_failure() -> None:
    """A failed component cannot skip parking later component runtimes."""

    events: list[str] = []

    class _Child(RewardFunction):
        def __init__(self, name: str, *, fail_score: bool = False) -> None:
            super().__init__()
            self.name = name
            self.fail_score = fail_score

        async def score_batch(self, rollouts: list[RewardRollout]) -> list[float]:
            events.append(f"score:{self.name}")
            if self.fail_score:
                raise RuntimeError(f"score failed:{self.name}")
            return [1.0] * len(rollouts)

        async def park_memory(self):
            events.append(f"park:{self.name}")
            return ()

    reward = MultiReward(
        [
            ("first", 1.0, _Child("first", fail_score=True)),
            ("second", 1.0, _Child("second")),
        ],
    )

    with pytest.raises(RuntimeError, match="score failed:first"):
        await reward.score_batch([_make_rollout("a")])

    assert events == ["score:first", "park:first", "park:second"]


@pytest.mark.asyncio
async def test_multi_reward_aggregates_parking_and_shutdown_failures() -> None:
    """Parking and shutdown each attempt every child before reporting errors."""

    events: list[str] = []

    class _Child(RewardFunction):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        async def park_memory(self):
            events.append(f"park:{self.name}")
            raise RuntimeError(f"park failed:{self.name}")

        async def shutdown(self) -> None:
            events.append(f"shutdown:{self.name}")
            raise RuntimeError(f"shutdown failed:{self.name}")

    reward = MultiReward(
        [("first", 1.0, _Child("first")), ("second", 1.0, _Child("second"))],
    )

    with pytest.raises(RewardCleanupError) as park_group:
        await reward.park_memory()
    assert len(park_group.value.errors) == 2
    with pytest.raises(RewardCleanupError) as shutdown_group:
        await reward.shutdown()
    assert len(shutdown_group.value.errors) == 2
    assert events == [
        "park:first",
        "park:second",
        "shutdown:first",
        "shutdown:second",
    ]


@pytest.mark.asyncio
async def test_multi_reward_shutdown_retries_only_failed_children() -> None:
    """A partial composite cleanup never double-shuts a completed sibling."""

    events: list[str] = []

    class _Child(RewardFunction):
        def __init__(self, name: str, *, fail_once: bool = False) -> None:
            super().__init__()
            self.name = name
            self.fail_once = fail_once

        async def shutdown(self) -> None:
            events.append(f"shutdown:{self.name}")
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError(f"shutdown failed:{self.name}")

    reward = MultiReward(
        [
            ("completed", 1.0, _Child("completed")),
            ("retry", 1.0, _Child("retry", fail_once=True)),
        ],
    )

    with pytest.raises(RewardCleanupError) as first:
        await reward.shutdown()
    assert len(first.value.errors) == 1
    assert events == ["shutdown:completed", "shutdown:retry"]

    await reward.shutdown()
    await reward.shutdown()

    assert events == [
        "shutdown:completed",
        "shutdown:retry",
        "shutdown:retry",
    ]


def test_factory_parking_policy_distinguishes_cpu_and_dedicated_rewards() -> None:
    """CPU-only and dedicated rewards stay resident without a parking pool."""
    cpu_reward = MultiReward.from_dict(
        {"ocr": 1.0},
        device="cpu",
        memory_parking_required=False,
    )
    dedicated_reward = MultiReward.from_dict(
        {"aesthetic": 1.0},
        device="cuda:1",
        reward_kwargs={"aesthetic": {"sleep_offload": True}},
        memory_parking_required=False,
    )

    cpu_runtime = cpu_reward.rewards[0][2].runtime
    dedicated_runtime = dedicated_reward.rewards[0][2].runtime
    assert isinstance(cpu_runtime, InProcessRewardRuntime)
    assert isinstance(dedicated_runtime, InProcessRewardRuntime)
    assert cpu_runtime.requires_memory_parking is False
    assert dedicated_runtime.requires_memory_parking is False


def test_shared_parking_allows_one_gpu_reward_with_cpu_sibling() -> None:
    """CPU siblings do not create a second process-wide CuMem owner."""
    reward = MultiReward.from_dict(
        {"aesthetic": 1.0, "ocr": 1.0},
        device="cuda:0",
        memory_parking_required=True,
    )

    runtimes = {name: fn.runtime for name, _, fn in reward.rewards}
    functions = {name: fn for name, _, fn in reward.rewards}
    assert runtimes["aesthetic"].requires_memory_parking is True
    assert runtimes["aesthetic"]._parking_residual_bytes_limit == CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT
    assert runtimes["ocr"].requires_memory_parking is False
    assert functions["ocr"]._model._device == "cpu"


def test_shared_parking_rejects_multiple_gpu_reward_components() -> None:
    """CuMem tags cannot make multiple reward pools independently sleepable."""
    with pytest.raises(ValueError, match="at most one configured GPU"):
        MultiReward.from_dict(
            {"aesthetic": 1.0, "pickscore": 1.0},
            device="cuda:0",
            memory_parking_required=True,
        )


@pytest.mark.parametrize(
    ("resolved_device", "component_device"),
    [("cpu", "cuda:0"), ("cuda:0", "cuda:1")],
)
def test_component_device_cannot_override_resource_device(
    resolved_device: str,
    component_device: str,
) -> None:
    """CPU resources cannot escalate to CUDA or change a CUDA ordinal."""
    with pytest.raises(ValueError, match=r"distributed resources|CUDA device"):
        MultiReward.from_dict(
            {"kling_video_reward": 1.0},
            device=resolved_device,
            reward_kwargs={
                "kling_video_reward": {
                    "worker_config": {"device": component_device},
                },
            },
            memory_parking_required=False,
        )


def test_gpu_resource_allows_component_cpu_downgrade() -> None:
    """A CPU sibling consumes no GPU lease under a CUDA ownership ceiling."""
    reward = MultiReward.from_dict(
        {"aesthetic": 1.0, "kling_video_reward": 1.0},
        device="cuda:0",
        reward_kwargs={
            "kling_video_reward": {"worker_config": {"device": "cpu"}},
        },
        memory_parking_required=True,
    )

    runtimes = {name: fn.runtime for name, _, fn in reward.rewards}
    runtime = runtimes["kling_video_reward"]
    assert isinstance(runtime, InProcessRewardRuntime)
    assert runtime._worker_config["device"] == "cpu"
    assert runtime.requires_memory_parking is False
    assert runtimes["aesthetic"].requires_memory_parking is True


def test_from_dict_rejects_removed_pool_execution() -> None:
    """A stale reward.kwargs execution key fails loud with the migration hint."""
    with pytest.raises(ValueError, match="resource topology"):
        MultiReward.from_dict(
            {"kling_video_reward": 1.0},
            device="cpu",
            reward_kwargs={"kling_video_reward": {"execution": "pool"}},
        )


def test_from_dict_validates_zero_weight_observation_components() -> None:
    """Observation-only scorers are live, so a typo must fail validation."""

    with pytest.raises(KeyError, match="Unknown reward function"):
        MultiReward.from_dict(
            {"not_a_registered_reward": 0.0, "aesthetic": 1.0},
            device="cpu",
        )


def test_http_disk_reward_builds_transport_without_local_model_config(tmp_path) -> None:
    reward = MultiReward.from_dict(
        {"videoscore2": 1.0},
        device="cuda:0",
        reward_kwargs={
            "videoscore2": {
                "artifact_dir": str(tmp_path),
                "inference": {
                    "kind": "http",
                    "endpoint": "http://reward:8300",
                    "expected_model": "videoscore2-v1",
                },
            },
        },
        memory_parking_required=False,
    )

    component = reward.rewards[0][2]
    assert component.artifact_transport == "disk"
    assert isinstance(component.runtime, HttpRewardRuntime)
    assert component.scoring_is_nonblocking is True
    assert component.external_accelerator_isolation_verified is False
    assert reward.scoring_is_nonblocking is True
    assert reward.external_accelerator_isolation_verified is False


def test_http_reward_rejects_inmemory_artifact_component() -> None:
    with pytest.raises(ValueError, match="in-memory artifacts"):
        MultiReward.from_dict(
            {"aesthetic": 1.0},
            device="cpu",
            reward_kwargs={
                "aesthetic": {
                    "inference": {
                        "kind": "http",
                        "endpoint": "http://reward:8300",
                        "expected_model": "aesthetic-v1",
                    },
                },
            },
        )


@pytest.mark.asyncio
async def test_preflight_reaches_every_remote_runtime_and_skips_local_ones() -> None:
    checked: list[str] = []

    class _RemoteRuntime:
        def __init__(self, name: str) -> None:
            self._name = name

        async def ensure_ready(self) -> None:
            checked.append(self._name)

        async def shutdown(self) -> None:
            return None

    remote_a = RewardFunction(reward_name="a", runtime=_RemoteRuntime("a"))
    remote_b = RewardFunction(reward_name="b", runtime=_RemoteRuntime("b"))
    local = RewardFunction(reward_name="c", runtime=InProcessRewardRuntime({}))
    reward = MultiReward([("a", 1.0, remote_a), ("b", 1.0, remote_b), ("c", 1.0, local)])

    await reward.preflight()

    assert checked == ["a", "b"]


def test_mixed_runtime_components_fail_closed_for_generation_overlap(tmp_path) -> None:
    reward = MultiReward.from_dict(
        {"videoscore2": 1.0, "ocr": 0.5},
        device="cpu",
        reward_kwargs={
            "videoscore2": {
                "artifact_dir": str(tmp_path),
                "inference": {
                    "kind": "http",
                    "endpoint": "http://reward:8300",
                    "expected_model": "videoscore2-v1",
                },
            },
        },
    )

    assert reward.scoring_is_nonblocking is False
    assert reward.external_accelerator_isolation_verified is False


def test_http_reward_rejects_local_worker_config() -> None:
    with pytest.raises(ValueError, match="belongs to the standalone reward service"):
        MultiReward.from_dict(
            {"videoscore2": 1.0},
            device="cpu",
            reward_kwargs={
                "videoscore2": {
                    "inference": {
                        "kind": "http",
                        "endpoint": "http://reward:8300",
                        "expected_model": "videoscore2-v1",
                    },
                    "worker_config": {"device": "cuda:0"},
                },
            },
        )
