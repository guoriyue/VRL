"""Tests for vrl.rewards.functions.registry."""

from __future__ import annotations

import pytest

from vrl.config.reward_inference import RewardInferenceConfig
from vrl.rewards.base import (
    DiskArtifactRewardFunction,
    InferenceRewardFunction,
    RewardCleanupError,
    RewardFunction,
)
from vrl.rewards.functions.registry import (
    MultiReward,
    validate_reward_memory_parking_components,
)
from vrl.rewards.functions.videoscore2 import VideoScore2Reward
from vrl.rewards.runtime import InProcessRewardScorer, RewardFunctionRuntime
from vrl.rewards.service.client import HttpRewardScorer
from vrl.rewards.service.server import RewardService
from vrl.rewards.types import RewardOutput, RewardSample
from vrl.utils.cuda_memory import CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT


def _make_sample(prompt: str) -> RewardSample:
    return RewardSample(
        prompt=prompt,
        output=None,
        sample_id=f"sample-{prompt}",
        metadata={"prompt_key": prompt},
    )


class _QueuedBatchReward(RewardFunction):
    def __init__(self, batches: list[list[float]]) -> None:
        super().__init__()
        self.batches = list(batches)

    async def score(self, sample: RewardSample) -> float:
        return self.batches.pop(0)[0]

    async def score_batch(self, samples: list[RewardSample]) -> RewardOutput:
        scores = self.batches.pop(0)
        assert len(scores) == len(samples)
        return RewardOutput(scores=tuple(scores))


class _TimedBatchReward(RewardFunction):
    def __init__(self, scores: list[float], timing_ms: dict[str, float]) -> None:
        super().__init__()
        self.scores = list(scores)
        self.timing_ms = dict(timing_ms)

    async def score_batch(self, samples: list[RewardSample]) -> RewardOutput:
        assert len(samples) == len(self.scores)
        return RewardOutput(
            scores=tuple(self.scores),
            timing_ms=dict(self.timing_ms),
        )


@pytest.mark.asyncio
async def test_multi_reward_preserves_samples_for_every_component() -> None:
    seen: dict[str, list[tuple[str, str]]] = {}

    class _CaptureReward(RewardFunction):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        async def score_batch(
            self,
            samples: list[RewardSample],
        ) -> RewardOutput:
            seen[self.name] = [
                (sample.sample_id, str(sample.metadata["prompt_key"])) for sample in samples
            ]
            return RewardOutput(scores=(1.0,) * len(samples))

    reward = MultiReward(
        [
            ("first", 1.0, _CaptureReward("first")),
            ("second", 1.0, _CaptureReward("second")),
        ],
    )
    samples = [_make_sample("a"), _make_sample("b")]

    await reward.score_batch(samples)

    expected = [
        ("sample-a", "a"),
        ("sample-b", "b"),
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

    first = await reward.score_batch(
        [_make_sample("a"), _make_sample("b")],
    )
    second = await reward.score_batch(
        [_make_sample("c")],
    )

    assert first.scores == pytest.approx([0.6, 1.2])
    assert second.scores == pytest.approx([1.8])
    assert first.components["ocr"] == pytest.approx([0.1, 0.2])
    assert first.components["aesthetic"] == pytest.approx([1.0, 2.0])
    assert second.components["ocr"] == pytest.approx([0.3])
    assert second.components["aesthetic"] == pytest.approx([3.0])


@pytest.mark.asyncio
async def test_zero_weight_component_is_scored_without_changing_total() -> None:
    """Checks observation-only scores are logged but excluded from the reward."""
    reward = MultiReward(
        [
            ("train", 1.0, _QueuedBatchReward([[2.0, 3.0]])),
            ("observe", 0.0, _QueuedBatchReward([[0.7, 0.8]])),
        ],
    )

    report = await reward.score_batch(
        [_make_sample("a"), _make_sample("b")],
    )

    assert report.scores == pytest.approx([2.0, 3.0])
    assert report.components["train"] == pytest.approx([2.0, 3.0])
    assert report.components["observe"] == pytest.approx([0.7, 0.8])


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

    report = await reward.score_batch([_make_sample("a"), _make_sample("b")])

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

        async def score_batch(self, samples: list[RewardSample]) -> RewardOutput:
            events.append(f"score:{self.name}")
            if self.fail_score:
                raise RuntimeError(f"score failed:{self.name}")
            return RewardOutput(scores=(1.0,) * len(samples))

        async def park_memory(self):
            events.append(f"park:{self.name}")
            return True

    reward = MultiReward(
        [
            ("first", 1.0, _Child("first", fail_score=True)),
            ("second", 1.0, _Child("second")),
        ],
    )

    with pytest.raises(RuntimeError, match="score failed:first"):
        await RewardFunctionRuntime(reward).score(
            [_make_sample("a")],
            require_memory_release=True,
        )

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

    cpu_runtime = cpu_reward.rewards[0][2].scorer
    dedicated_runtime = dedicated_reward.rewards[0][2].scorer
    assert isinstance(cpu_runtime, InProcessRewardScorer)
    assert isinstance(dedicated_runtime, InProcessRewardScorer)
    assert cpu_runtime.requires_memory_parking is False
    assert dedicated_runtime.requires_memory_parking is False


@pytest.mark.parametrize(
    "inference_configs",
    [
        {},
        {
            "ocr": RewardInferenceConfig(),
            "stale": RewardInferenceConfig(),
        },
    ],
)
def test_from_dict_rejects_inconsistent_resolved_inference_keys(
    inference_configs: dict[str, RewardInferenceConfig],
) -> None:
    with pytest.raises(ValueError, match="inference config keys must match component keys"):
        MultiReward.from_dict(
            {"ocr": 1.0},
            device="cpu",
            inference_configs=inference_configs,
        )

    with pytest.raises(ValueError, match="inference config keys must match component keys"):
        validate_reward_memory_parking_components(
            ("ocr",),
            device="cpu",
            inference_configs=inference_configs,
        )


def test_shared_parking_allows_one_gpu_reward_with_cpu_sibling() -> None:
    """CPU siblings do not create a second process-wide CuMem owner."""
    reward = MultiReward.from_dict(
        {"aesthetic": 1.0, "ocr": 1.0},
        device="cuda:0",
        memory_parking_required=True,
    )

    runtimes = {name: fn.scorer for name, _, fn in reward.rewards}
    functions = {name: fn for name, _, fn in reward.rewards}
    assert runtimes["aesthetic"].requires_memory_parking is True
    assert runtimes["aesthetic"]._parking_residual_bytes_limit == CUDA_RUNTIME_RESIDUAL_BYTES_LIMIT
    assert runtimes["ocr"].requires_memory_parking is False
    assert functions["ocr"].resolve_execution_device(device="cuda:0", kwargs={}) == "cpu"


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

    runtimes = {name: fn.scorer for name, _, fn in reward.rewards}
    runtime = runtimes["kling_video_reward"]
    assert isinstance(runtime, InProcessRewardScorer)
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
        reward_kwargs={"videoscore2": {"artifact_dir": str(tmp_path)}},
        inference_configs={
            "videoscore2": RewardInferenceConfig(
                kind="http",
                endpoint="http://reward:8300",
                expected_model="videoscore2-v1",
            ),
        },
        memory_parking_required=False,
    )

    component = reward.rewards[0][2]
    assert isinstance(component, DiskArtifactRewardFunction)
    assert isinstance(component.scorer, HttpRewardScorer)
    assert component.scoring_is_nonblocking is True
    assert component.external_accelerator_isolation_verified is False
    assert reward.scoring_is_nonblocking is True
    assert reward.external_accelerator_isolation_verified is False


def test_http_reward_rejects_inmemory_artifact_component() -> None:
    with pytest.raises(ValueError, match="in-memory artifacts"):
        MultiReward.from_dict(
            {"aesthetic": 1.0},
            device="cpu",
            inference_configs={
                "aesthetic": RewardInferenceConfig(
                    kind="http",
                    endpoint="http://reward:8300",
                    expected_model="aesthetic-v1",
                ),
            },
        )


@pytest.mark.parametrize("key", ["scorer", "runtime"])
def test_reward_config_rejects_runtime_injection_keys(key: str) -> None:
    with pytest.raises(ValueError, match="runtime injection keys"):
        MultiReward.from_dict(
            {"aesthetic": 1.0},
            device="cpu",
            reward_kwargs={"aesthetic": {key: object()}},
        )


class _NeverScoredRuntime:
    """Service-side scoring runtime; preflight only ever reaches /ready and /info.

    The stub asserts that itself, so a preflight that started scoring would fail
    here rather than pass quietly.
    """

    async def score_batch(self, request):
        raise AssertionError("preflight must not score")

    async def shutdown(self) -> None:
        return None


class _NeverScoredModel:
    def __call__(self, artifact):
        raise AssertionError("preflight must not score")


@pytest.mark.real_cover(
    "tests/rewards/service/test_service.py::test_client_scores_through_async_server_and_validates_identity",
    why=(
        "preflight must not score, so this runtime raises if reached; real "
        "scoring over a live aiohttp server is that counterpart, which runs "
        "on every PR rather than behind an opt-in flag"
    ),
)
@pytest.mark.asyncio
async def test_preflight_reaches_every_remote_runtime_and_skips_local_ones(tmp_path) -> None:
    """Preflight must complete a round trip for *each* remote component.

    A component nobody contacted at launch fails after the first generation batch
    instead, wasting the whole warmup. The observable is per-component and the
    service produces it, not the test:
    ``external_accelerator_isolation_verified`` only flips once that client's own
    ``/ready`` + ``/info`` returned and ``/info`` advertised the overlap-safe
    capability, so a preflight that stopped after the first component leaves the
    second one False.

    Call *order* is deliberately not asserted: ``/info`` does not carry a reward
    name, so the real wire cannot attribute a request to a component, and
    preflight fails fast on any component — nothing downstream depends on which
    one is contacted first.
    """

    service = RewardService(
        _NeverScoredRuntime(),
        host="127.0.0.1",
        port=0,  # the OS picks a free port, so there is no bind race to lose
        artifact_roots=[tmp_path],
        model_name="unit-model",
        model_version="unit-v1",
        generation_overlap_safe=True,
    )
    await service.start()
    host, port = service.address

    def _client() -> HttpRewardScorer:
        return HttpRewardScorer(
            RewardInferenceConfig(
                kind="http",
                endpoint=f"http://{host}:{port}",
                timeout_s=5,
                expected_model="unit-model",
            ),
        )

    remote_a = VideoScore2Reward(
        reward_name="a",
        scorer=_client(),
        artifact_dir=str(tmp_path / "remote-a"),
    )
    remote_b = VideoScore2Reward(
        reward_name="b",
        scorer=_client(),
        artifact_dir=str(tmp_path / "remote-b"),
    )
    # A real in-process runtime is the "skips local ones" half: it has no
    # ensure_ready at all, so preflight's capability dispatch must step over it.
    local = InferenceRewardFunction(
        reward_name="c",
        score_key="c",
        scorer=InProcessRewardScorer(model=_NeverScoredModel()),
    )
    assert not hasattr(InProcessRewardScorer, "ensure_ready")
    reward = MultiReward([("a", 1.0, remote_a), ("b", 1.0, remote_b), ("c", 1.0, local)])
    try:
        assert remote_a.external_accelerator_isolation_verified is False
        assert remote_b.external_accelerator_isolation_verified is False

        await reward.preflight()

        assert remote_a.external_accelerator_isolation_verified is True
        assert remote_b.external_accelerator_isolation_verified is True
    finally:
        await remote_a.scorer.shutdown()
        await remote_b.scorer.shutdown()
        await service.shutdown_async()


def test_mixed_runtime_components_fail_closed_for_generation_overlap(tmp_path) -> None:
    reward = MultiReward.from_dict(
        {"videoscore2": 1.0, "ocr": 0.5},
        device="cpu",
        reward_kwargs={"videoscore2": {"artifact_dir": str(tmp_path)}},
        inference_configs={
            "videoscore2": RewardInferenceConfig(
                kind="http",
                endpoint="http://reward:8300",
                expected_model="videoscore2-v1",
            ),
            "ocr": RewardInferenceConfig(),
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
                    "worker_config": {"device": "cuda:0"},
                },
            },
            inference_configs={
                "videoscore2": RewardInferenceConfig(
                    kind="http",
                    endpoint="http://reward:8300",
                    expected_model="videoscore2-v1",
                ),
            },
        )
