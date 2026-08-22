# SPRINT: Reward service and generation/reward overlap

Status: **DONE (2026-07-18)**. The service, fail-closed isolation contract, and
CPU contract coverage are implemented. Multi-GPU throughput measurement is a
separate hardware-triggered task in
[`SPRINT_video_rollout_stage_overlap.md`](../info/SPRINT_video_rollout_stage_overlap.md).

## Decision

VRL supports two reward inference deployments:

- `InProcessRewardInferenceRuntime` is the default inference transport for local
  and single-GPU training.
- `HttpRewardInferenceRuntime` calls an operator-owned service over async HTTP.

Transport and accelerator placement are independent axes:

| Axis | Question | Evidence |
| --- | --- | --- |
| Execution/transport | Does scoring yield the collector event loop? | `scoring_is_nonblocking` |
| Physical placement | Can reward model work run beside generation without sharing an accelerator? | service capability learned during preflight |

HTTP answers only the first question. A localhost HTTP server can still use the
same physical GPU as rollout or training. It is therefore incorrect to treat
`kind: http` as proof that reward/generation overlap is safe.

Streaming is enabled only when both facts are true:

```text
collector.supports_reward_generation_overlap
  = scoring_is_nonblocking
  AND external_accelerator_isolation_verified
```

The service-side `generation_overlap_safe` attestation defaults to `false`.
Before `/info` preflight, the client also remains fail-closed. A CPU service is
inferred safe because it performs no accelerator work; a GPU service advertises
the capability only when its operator explicitly sets the attestation.

This attestation is not a convenience performance switch. Set it only when the
service accelerator is physically disjoint from every trainer or rollout
accelerator that can be active at the same time. The current wire protocol does
not carry GPU UUIDs, so the operator owns this deployment fact.

## Deployment configuration

Trainer-side inference config keeps transport separate from model deployment:

```yaml
reward:
  components:
    videoscore2: 1.0
  kwargs:
    videoscore2:
      artifact_dir: /shared/vrl/reward_artifacts
      inference:
        kind: http
        endpoint: http://reward.internal:8300
        timeout_s: 1800
        expected_model: videoscore2-v1
```

The operator-owned service config contains the model, device, and overlap
attestation:

```yaml
host: 0.0.0.0
port: 8300
model_name: videoscore2-v1
model_version: TIGER-Lab/VideoScore2@main
artifact_roots:
  - /shared/vrl/reward_artifacts
max_concurrency: 1
max_pending_requests: 8
max_cached_requests: 1024

# Keep false unless deployment proves the service accelerator is disjoint from
# every concurrently active trainer/rollout accelerator. CPU is inferred safe.
generation_overlap_safe: false

worker_config:
  model_factory: vrl.rewards.models.videoscore2:VideoScore2Model
  reward_model_name: TIGER-Lab/VideoScore2@main
  device: cuda:0
```

Launch the service with:

```bash
uv sync --extra reward --extra reward-service
vrl-reward-service --config /path/to/service.yaml
```

HTTP reward components reject local execution fields such as `device`,
`sleep_offload`, and `worker_config`. Those belong to the service config.
`worker_config.service_url` is removed; endpoint selection belongs to the typed
`inference` block.

## Resource and schedule semantics

### Resource resolver

When every reward component is HTTP, the VRL resource resolver creates no local
reward GPU or CPU bundle and injects no local reward parking. Mixed deployments
reserve resources only for their in-process components.

This is ownership accounting, not remote placement discovery. VRL cannot see
which GPU an external endpoint uses. Running a heavy HTTP reward service on the
sole trainer/rollout GPU is unsupported and does not bypass the phase lease.

### `strict_on_policy`

Strict collection keeps the existing batched serial baseline unless the complete
overlap capability is present:

```text
serial baseline:
  generate all groups
  score all groups in one reward call

safe streaming:
  score group N while generating group N+1
```

Local shared-GPU reward uses the topology-derived phase handoff and never
overlaps generation. In-process scoring is synchronous, so even a dedicated
local reward remains on the batched serial path. A preflighted async service may
stream only when its isolation attestation is present and no local topology
handoff is required.

At most one streaming score task exists. The next task starts only after the
previous task drains, which provides bounded backpressure and deterministic
result order. Generation failure, scoring failure, cancellation, and dual
failure all retain explicit cleanup ownership.

### `continuous`

Continuous rollout already requires disjoint trainer and rollout GPUs and rejects
local reward topologies that require either role to park mid-iteration.

For an external reward service:

- a verified isolation capability allows reward work to overlap generation and
  training; and
- without it, schedule construction fails before the continuous owner starts.
  The service remains usable through the strict batched-serial path.

HTTP placement is not statically visible, so `generation_overlap_safe=false`
rejects continuous training rather than pretending lower collect concurrency
solves physical contention. Even one collect task overlaps trainer backward.

## Request, sample, and artifact identity

Generation, collector, artifact, and reward result preserve:

- `source_request_id`
- `sample_id`
- `group_id`
- `trajectory_id`
- `policy_version`

Disk materialization uses unique IDs and filenames. The wire includes
`size_bytes` and `sha256`; the server validates absolute paths, configured roots,
file size, and digest before inference.

Artifact ownership remains attached to the reward call until success, failure,
or cancellation reaches a terminal state. `retain_artifacts=true` transfers
ownership to a debug/output directory. If a transport failure leaves remote work
ambiguous and cancellation cannot confirm a terminal state, the client preserves
the artifact and warns rather than deleting a file the service may still read.

The implemented transport is `shared_filesystem_paths`. HTTP carries the control
protocol, not video bytes; trainer and service must see the same filesystem path.

## Service protocol

The current protocol provides:

- a versioned JSON envelope;
- `/live`, `/ready`, and `/info`;
- model identity, artifact transport, and scheduling capabilities;
- typed error code, retryability, and request identity;
- bounded admission and backpressure;
- in-flight idempotent join, completed success replay, and retryable-failure retry;
- explicit request cancellation;
- graceful shutdown that waits for non-preemptible synchronous model work; and
- trainer startup preflight before rollout actors are launched or collection begins.

`/live` is operator-facing. Trainer preflight consumes readiness, identity, and
capabilities. The client becomes overlap-capable only after this preflight has
validated the service response.

## Telemetry implemented in this sprint

Collection now records actual wall intervals rather than inferring overlap from
the existence of an async task:

- `collect.wall`
- `collect.generation_wall`
- `collect.reward_wall`
- `collect.generation_reward_overlap`
- `collect.group_count`
- `collect.sample_count`

Reward timing aggregates every scoring call rather than retaining only the last
one:

- `reward.latency_s` — sum across calls
- `reward.queue_wait_s` — sum across calls
- `reward.inference_s` — sum across calls
- `reward.call_count`
- `reward.latency_p50_s`
- `reward.latency_p95_s`

Additional timing keys are surfaced as reward phases:

- `reward.artifact_materialization_s`
- `reward.artifact_validation_s`
- `reward.service_inference_wall_s`
- `reward.transport_roundtrip_s`

The server measures real semaphore wait and adds it to `queue_wait_ms`; it no
longer reports a default zero while requests are queued. The current percentile
implementation covers end-to-end reward latency only. Queue-wait and inference
percentiles would require retaining their own per-call sample series and are not
implemented.

## Single-GPU verification boundary

One GPU is sufficient for correctness and failure-path coverage:

| Scenario | One GPU? | What it proves |
| --- | ---: | --- |
| In-process shared reward phase lease | Yes | strict handoff, memory release proof, restore, cleanup |
| CPU/fake HTTP service end-to-end | Yes | wire, preflight, fail-closed capability, identity, queue, cancel, idempotency, artifact cleanup |
| CPU/fake reward with generation | Yes | event ordering and positive measured interval intersection |
| Heavy HTTP reward on the same GPU | Unsupported | neither a safe deployment nor valid performance evidence |
| Dedicated rollout/reward GPU throughput | No | requires at least two physical GPUs or a real remote isolated service |
| Continuous trainer/rollout overlap | No | requires disjoint trainer and rollout accelerators |

A fake or CPU service can prove orchestration, but it cannot prove that a heavy
GPU reward workload improves end-to-end throughput.

## Performance acceptance

For group generation cost `G` and reward cost `R`, the ideal steady-state lower
bound is `max(G, R)` rather than `G + R`. The theoretical speedup is therefore:

```text
(G + R) / max(G, R)
```

If that upper bound is below `1.10x`, or the collection contains only one group,
keep the batched serial path.

A valid benchmark must compare three identical workloads with the same model,
config, seed, sample count, and artifact format:

1. **A — production serial:** generate all groups, then one batched reward call.
2. **B — per-group serial control:** same call granularity as streaming, without
   overlap.
3. **C — per-group streaming:** reward N overlaps generation N+1.

C must beat A, not merely B. B is needed to expose the batching and transport tax
introduced by per-group scoring.

Accept streaming for a heavy isolated service only when repeated steady-state
runs show all of the following:

- at least `10%` end-to-end collection wall/throughput improvement over A;
- a positive lower confidence bound over at least five runs;
- generation p95 regression no greater than `5%`;
- measured overlap intervals are positive;
- realized saved wall time is at least half of theoretical `min(G, R)`;
- queue depth and queue wait remain bounded; and
- correctness, cancellation, artifact ownership, and error rate do not regress.

Kill the optimization when net gain is below `10%`, queue/hash/network overhead
consumes more than half the theoretical saving, generation slows by more than
`5%`, queue wait grows without bound, or reward shares an active trainer/rollout
accelerator.

## Architecture hygiene

Keep these thin boundaries:

- `RewardInferenceRuntime`: transport-neutral inference protocol.
- `InProcessRewardInferenceRuntime` and `HttpRewardInferenceRuntime`: inference
  adapters with deliberately different scheduling capabilities.
- `vrl/rewards/service/__init__.py`: empty package import boundary that avoids
  eagerly importing the server before `python -m vrl.rewards.service.server`.
- `wire.py`: versioned protocol adapter.
- `owner.py`: synchronous model thread/event-loop ownership.
- `DiskArtifactRewardFunction`: artifact capability visible before registry
  construction.

Keep protocol constants such as `WIRE_PROTOCOL`, `WIRE_VERSION`,
`SHARED_FILESYSTEM_ARTIFACT_TRANSPORT`, and
`GENERATION_OVERLAP_SAFE_CAPABILITY`. They are real wire/schema boundaries, not
duplicated business vocabulary. Allowed service-config keys remain derived from
the `RewardServiceConfig` dataclass fields rather than a hand-maintained constant.

Do not flatten these modules to reduce line count. Transport, wire, lazy import,
and thread ownership are useful failure and debugging boundaries.

## Non-goals

- Do not remove the single-GPU in-process phase lease.
- Do not support resident trainer, rollout, and heavy reward work on one GPU.
- Do not infer accelerator isolation from an HTTP URL or process boundary.
- Do not upload video bytes or introduce an object-store URI in this sprint.
- Do not add `HttpGenerationRuntime`; the existing generation runtime remains the
  transport boundary until an independent rollout fleet has a real use case.
- Do not claim dedicated-accelerator performance from a one-GPU fake-service test.

## References

- Runtime selection: `vrl/rewards/runtime.py`
- Inference/result protocol: `vrl/rewards/inference.py`
- Reward service client/server/protocol: `vrl/rewards/service/`
- Artifact lifecycle and timing: `vrl/rewards/base.py`
- Multi-component capability folding: `vrl/rewards/functions/registry.py`
- Resource ownership: `vrl/ray/resources.py`
- Collector capability: `vrl/rollouts/collector/core.py`
- Batched scoring: `vrl/rollouts/collector/rewards.py`
- Streaming and interval telemetry: `vrl/rollouts/orchestration/prompt_collection.py`
- Continuous fail-fast gate: `vrl/rollouts/orchestration/continuous/schedule.py`
- Stats aggregation: `vrl/utils/stats.py`
- Service tests: `tests/rewards/service/test_service.py`
- Collector tests: `tests/rollouts/collector/test_runtime.py`
- Streaming tests: `tests/rollouts/orchestration/test_prompt_collection.py`
