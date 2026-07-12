# Sprint：Reward service 与 generation/reward overlap

## 结论

本 sprint 保留两种 reward 执行方式，但只在能力真实成立时开启并发：

- `InProcessRewardRuntime` 仍是默认。单卡 shared GPU 使用完整 phase handoff；dedicated local reward 保持批量串行，避免把一个大 batch 拆成多个小调用。
- `HttpRewardRuntime` 面向 operator-owned reward service。服务独占自己的进程和设备，trainer 不为它预留本地 reward GPU。
- generation/reward streaming 只在以下两个条件同时成立时启用：
  1. topology 不要求 rollout→reward 或 trainer→reward GPU handoff；
  2. 所有 reward component 都声明 `supports_generation_overlap=true`。当前只有真正 async 的 HTTP transport 满足该能力。

这不是一个用户可强开的性能开关。调度器从 topology 和 runtime capability 派生答案，无法安全 overlap 时自动保留原来的“全部 generate，再一次性 batch score”路径。

## 已实现的边界

### Typed deployment config

Trainer config 将 transport 与外部模型部署分离：

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

HTTP component 禁止 `worker_config`、`device`、`sleep_offload` 等本地执行字段。模型和设备属于 service config：

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
worker_config:
  model_factory: vrl.rewards.models.videoscore2:VideoScore2Model
  reward_model_name: TIGER-Lab/VideoScore2@main
  device: cuda:0
```

启动方式：

```bash
uv sync --extra reward --extra reward-service
vrl-reward-service --config /path/to/service.yaml
```

### Resource ownership

- 全部 component 都是 HTTP 时，resource resolver 不创建本地 reward GPU/CPU bundle，也不注入 parking/handoff。
- HTTP 与 local component 混用时，只为 local component 解析资源。
- 单卡 heavy reward 不能通过“本机同一张卡上另起 HTTP server”绕过 phase lease；这种部署对 resolver 不可见，会造成显存竞争。单卡真实训练继续使用 `in_process`。

### Request、sample 与 artifact identity

generation → collector → reward artifact/result 全程保留：

- `source_request_id`
- `sample_id`
- `group_id`
- `trajectory_id`
- `policy_version`

Disk artifact 每次 materialization 使用唯一 ID 和文件名，不会因 `sample-{index}` 重复覆盖。HTTP wire 还携带 `size_bytes` 与 `sha256`；server 在进入模型前校验绝对路径、allowed root、文件大小和内容 digest。

Artifact 默认由当前 reward call 持有，并在确认 success、failure、cancel 已进入 terminal state 后清理。`retain_artifacts=true` 会把 ownership 显式转交给 debug/output 目录；POST timeout 或断连后如果 best-effort cancellation 也无法确认 terminal state，同样保留文件并告警，避免 remote model 仍在读取时删除 shared artifact。

### Service control protocol

当前 wire protocol 提供：

- versioned JSON envelope；
- `/live`、`/ready`、`/info`；
- model identity 与 artifact transport capability；
- typed error code、retryable、request ID；
- bounded admission/backpressure；
- 同 request ID 的 in-flight join、success replay 和 retryable failure retry；
- request cancellation；
- graceful shutdown，并等待不可抢占的同步 model call 结束后再释放 admission；
- trainer 启动 preflight：`run_online_recipe` 在构建 rollout backend 之前调用
  `MultiReward.preflight → HttpRewardRuntime.ensure_ready`，service 不可达、未
  ready 或 model identity 不匹配时在 launch 阶段直接失败，而不是等第一个
  generation batch 打到 scoring 才暴露。`/live` 仍是 operator/process-supervisor
  探针，trainer client 不消费它。

当前 artifact transport 是 `shared_filesystem_paths`。HTTP 负责控制面，不上传视频字节；trainer 与 service 必须看到同一 shared filesystem 路径。

## Streaming overlap

安全路径最多持有一个 scoring task：

```text
generate group 0
├─ score group 0 ─────────────┐
└─ generate group 1 ──────────┘
   ├─ score group 1 ─────────────┐
   └─ generate group 2 ──────────┘
```

启动下一次 score 前先 drain 前一次 score，因此 queue 有明确 backpressure，batch 返回顺序不变。generation、score 或 cancellation 任一失败时，owner 会取消并收尾 in-flight reward task；双失败通过聚合异常保留两个 root cause。

以下情况保持严格串行：

- reward 与 rollout 共用 GPU；
- reward 与 trainer 共用 GPU；
- 任一 component 是同步 `InProcessRewardRuntime`；
- collector fake/第三方实现没有显式提供 overlap capability。

## 单卡验证矩阵

一张 GPU 可以完成大部分 correctness 验证，但不能证明 dedicated-GPU overlap 的吞吐收益：

| 场景 | 单卡可验证 | 预期 |
|---|---:|---|
| In-process shared reward phase lease | 是 | 严格 handoff，无 overlap |
| CPU/fake HTTP service E2E | 是 | wire、identity、queue、cancel、cleanup 全链路 |
| HTTP reward + fake generation overlap | 是 | event 顺序证明 score N 与 generate N+1 重叠 |
| Heavy reward service 与 trainer 共用同一 GPU | 不支持 | 不能作为性能或安全部署 |
| Dedicated rollout/reward GPU 吞吐 benchmark | 否 | 需要至少两张物理 GPU 或远端 service |

## 性能判断

Overlap 的理论隐藏量接近 `min(reward_time, next_generation_time)`；reward 越重、generation group 越多，价值越大。它不会让 reward 本身更快，也不会改善只有一个大 generation batch 的调用。

因此 acceptance 不是“HTTP 一定更快”，而是：

1. fast reward 继续 in-process batch score，不支付网络、hash 和小 batch 开销；
2. heavy external reward 能在 generation 仍进行时工作；
3. trainer/reward 有独立资源后再测 wall-clock、GPU utilization、queue wait 和 inference latency；
4. 没有独立资源时，系统保持 phase lease，不宣称 overlap 性能收益。

## Architecture hygiene

应保留的 thin boundaries：

- `RewardInferenceRuntime`：in-process 与 HTTP transport protocol；
- `vrl/rewards/service/__init__.py`：optional dependency 的 lazy public facade；
- `wire.py`：versioned protocol adapter；
- `owner.py`：同步 model event-loop/thread ownership；
- `DiskArtifactRewardFunction`：registry construction 前可读取的 artifact capability。

应删除或禁止的内容：

- `worker_config.service_url` 和 HTTP component 上的 local device/model knobs；
- 手写、重复 typed dataclass field 的 allowed-key 常量；
- `sample-{index}` artifact fallback；
- topology-safe 但 runtime 实际同步时的伪 overlap。

保留的 ALL_CAPS 常量只有真实协议/schema 边界，例如 `WIRE_PROTOCOL`、`WIRE_VERSION`、derived media/format literals。不会为了减少文件或 LOC 合并上述 protocol、lazy import、thread owner boundaries；跨 transport 的一致性比扁平化更重要。

## Non-goals

- 不删除单卡 in-process phase lease。
- 不在同一张 GPU 上支持 trainer、rollout、heavy reward resident concurrency。
- 本 sprint 不实现视频上传或 object-store URI；跨主机使用 shared filesystem。
- 不增加 `HttpGenerationRuntime`。现有 `GenerationRuntime` 保持 transport boundary，等独立 rollout fleet 真实出现时复用同一 identity/control 约定。

## 主要代码位置

- `vrl/config/reward_inference.py`
- `vrl/rewards/runtime.py`
- `vrl/rewards/service/`
- `vrl/rewards/inference.py`
- `vrl/rewards/artifacts.py`
- `vrl/rewards/functions/registry.py`
- `vrl/ray/resources.py`
- `vrl/rollouts/collector/rewards.py`
- `vrl/rollouts/orchestration/prompt_collection.py`
