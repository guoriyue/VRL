# SPRINT：video_reward inference infrastructure

## 结论

这个 sprint 的主体不是 Cosmos-Predict2.5，也不是 DiffusionNFT。

真正要补的是一层可复用的 `video_reward` inference infrastructure：

```text
rollout output image/video
  -> stable reward artifact
  -> `video_reward` inference runtime
  -> normalized reward result
  -> RL reward score
  -> debug/audit records
```

核心设计决定：

- 重 video reward model 应该是独立 reward inference runtime，和 rollout runtime 一样是 data plane，不应该塞进 trainer policy module。
- repo-owned serving / inference execution 默认走 Ray：rollout 是 Ray rollout workers，reward 是 Ray reward workers。
- trainer-facing API 可以继续保持同步 batch 语义：`rollout -> reward -> pack -> train`。
- runtime execution 必须支持 async / Ray worker pool；不要让 `VideoReward.score_batch(...)` 在训练进程里串行跑重模型。
- 第一版保持 on-policy correctness，不做 stale batch training；后续再用 versioned queue 做 rollout / reward / train overlap。

Cosmos-Predict2.5 + DiffusionNFT 只作为第一条真实验收 consumer。不要继续把这件事写成 Cosmos model architecture sprint，否则会让人误解成 Cosmos runtime 或 DiffusionNFT objective 还没有接好。

## 当前阶段范围

当前阶段只补 reward inference data plane，不改 trainer 架构。

范围内：

- 保持 trainer 是现有 driver-side online trainer。
- 保持 rollout 已有 Ray path。
- 新增 reward artifact / request / result contract。
- 新增 Ray reward workers，让本地 heavy reward model 也走 Ray。
- 保持 trainer 只 await scored batch，不感知 reward worker 细节。

范围外：

- 不把 trainer 迁进 RayTrainGroup。
- 不在这个 sprint 接 FSDP trainer。
- 不做 multi-node training rank orchestration。
- 不做 rollout / reward / train full async pipeline。

后续 FSDP sprint 可以让 Ray 负责启动和放置 trainer ranks，但 rank 内训练策略仍然是 PyTorch FSDP/NCCL；这个 reward sprint 只保证 reward 侧资源边界不会和未来 trainer strategy 冲突。

## 当前事实

已经存在：

```text
vrl/rewards/video_reward.py
vrl/rewards/remote_video.py
configs/base/reward/video_reward.yaml
```

但当前 `VideoReward` 还是 reward entrypoint + runtime selection + media stack + remote client glue 混在一起：

```text
Rollout -> VideoReward -> RemoteVideoRewardClient -> float score
```

这对短期 plumbing 可以，但不适合作为长期 `video_reward` inference 层，原因是：

- 没有 first-class artifact contract。
- 没有统一 `VideoRewardRequest` / `VideoRewardResult`。
- Ray reward model inference 还没有 runtime 边界。
- external_http reward debug 只记录 raw service response，不够表达 artifact、model version、latency、score breakdown。
- Cosmos recipe 容易被误写成“真实 video reward 已验证”，但当前 recipe 可以退回 OCR/simple reward。
- 当前没有 dedicated `tests/rewards/test_video_reward.py`；这个 sprint 应新增它，而不是假设它已经存在。

## 命名收敛规则

不要在用户可见 config 里引入一套和当前 repo 不同的 reward taxonomy。

保持这些名字不变：

```text
reward.components.video_reward
reward.kwargs.video_reward
reward.kwargs.video_reward.inference_runtime: ray | external_http
vrl/rewards/video_reward.py
vrl/rewards/remote_video.py
RemoteVideoRewardClient
VideoReward
RewardScorer
```

新增内部模块也应该贴近现有名字：

```text
vrl/rewards/video_inference/
VideoRewardArtifact
VideoRewardArtifactStore
VideoRewardRequest
VideoRewardResult
VideoRewardRuntime
```

也就是说，repo-owned reward inference 用 `inference_runtime=ray` 表达，而不是新增 `local_ray` 或 `reward_model.*` 用户配置。`reward_model_version` 只作为 result metadata 字段保留，因为它描述的是被调用模型的版本，不是配置命名空间。

## Execution/resource split

不要用 `remote | local` 作为 GPU 资源语义。它把两个维度混在一起：

```text
execution runtime:
  who runs the inference request?

resource role:
  which GPUs can that runtime use?
```

目标 public config 应该拆成：

```yaml
reward:
  kwargs:
    video_reward:
      inference_runtime: ray          # ray | external_http
      reward_name: cosmos_reason1
      score_key: overall_reward

distributed:
  backend: ray
  resources:
    reward:
      num_gpus: 1
      gpus_per_worker: 1
      num_workers: 1
```

规则：

- `inference_runtime=ray`：repo 启动 Ray reward workers；GPU 分配只来自 `distributed.resources.reward`。
- `inference_runtime=external_http`：repo 调外部 reward service；本 repo 不声明这部分 GPU。
- 不允许 `reward.kwargs.video_reward` 直接写 `device=cuda:1` / `num_gpus` / `devices`。
- 不允许用 `backend=local` / `backend=remote` 表达资源；legacy `backend` 字段在 Phase 4 迁移到 `inference_runtime`。
- 这和 `SPRINT_distributed_resource_config.md` 一致：模型/组件配置描述 what to run，GPU ownership 属于 `distributed.resources`。

## Slime 对照

`/home/mingfeiguo/Desktop/slime` 里可以借鉴的是大方向，不是照搬它的 reward 配置形状。

Slime 的实际路径是：

```text
Ray placement group
  -> rollout manager / SGLang serving
  -> generate sample
  -> reward/verifier inside rollout data generation
  -> Sample.reward
  -> train_data["rewards"] / train_data["raw_reward"]
  -> training actor group
```

关键事实：

- Slime README 把 rollout 定义成“生成新数据，包括 rewards/verifier outputs”。
- `generate_and_rm(...)` 先生成 sample，再调用 `async_rm(...)` 或 `batched_async_rm(...)` 填 `sample.reward`。
- `remote_rm` 是 async HTTP reward call；`custom_rm_path` 可以替换成用户自定义 reward 逻辑。
- SGLang multi-model config 可以声明 frozen `reward` model，`update_weights: false`，然后自定义 rollout function 通过 router 调 reward model。
- Slime 的 placement group 只有 actor / rollout 两个主资源切分；reward model 如果作为 SGLang model 存在，GPU 预算算在 `rollout_num_gpus` / SGLang serving config 里。

所以对本 repo 的结论是：

- 应该复制 Slime 的 data-plane 思路：reward 不进 trainer policy module，trainer 只消费已经带分数的 rollout batch。
- 应该复制 Slime 的版本保护思路：async path 不能在 generation 中途更新 rollout weights，训练只消费完整 batch。
- 不应该照搬 Slime 的 reward 资源命名：text reward/verifier 放进 rollout serving pool 可以接受，但 heavy video reward 应该是 first-class `reward` role。
- 不应该把 video reward 隐藏在 rollout worker 里。video artifact materialization、reward model latency、GPU worker lifecycle 都需要独立 debug 和 release 语义。

对应到本 sprint：

```text
Slime "reward model in rollout serving"
  -> 本 repo 的 `inference_runtime=ray`
  -> GPU ownership 放进 `distributed.resources.reward`
```

也就是说，Slime 证明了 reward 应该是 Ray-managed inference/data-generation side，而不是 trainer 内部模块；但本 repo 应把 reward 从 rollout serving pool 里提升成单独 role，避免 video reward 和 sample generation 抢同一套隐式资源。

## 本 repo 落地设计

这一节是实现边界，不是额外抽象。

现有代码里已经有三条必须保留的主链路：

```text
RolloutCollector.collect(...)
  -> runtime.generate(...)
  -> RewardScorer.score(...)
  -> RewardFunction.score_batch(...)
  -> RolloutBatch
```

```text
resolve_distributed_resources(...)
  -> RolloutBackendConfig.from_cfg(...)
  -> RayRolloutLauncher
  -> RayDistributedRuntime
```

```text
MultiReward.from_dict(...)
  -> VideoReward(...)
  -> score_batch(...)
```

所以实现不应该绕开 collector / MultiReward / resource resolver。正确做法是在这些 seam 上增量扩展。

### 0. 保留的接口边界

保留：

- `RewardScorer.score(...)` 仍然只返回 trainer 需要的 `torch.Tensor` reward。
- `RewardFunction.score_batch(...)` 仍然是 reward function 的外部入口。
- `MultiReward` 不感知 video reward 的 artifact / Ray worker / external_http 细节。
- `RolloutCollector` 仍然是 `generate -> reward_score -> rollout_batch_from_trajectory`。
- trainer 不 import `vrl.rewards.video_inference.*`，只看到 `RolloutBatch.rewards`。

改变：

- `VideoReward` 从“backend selector + remote client glue”变成 thin adapter。
- `RemoteVideoRewardClient` 变成 `external_http` runtime 的内部 client，不再被 `VideoReward` 直接当主抽象。
- `distributed.resources` 从 trainer / rollout 扩展到 trainer / rollout / reward。
- Ray reward runtime 独立于 Ray rollout runtime；两者可以复用 placement/logging style，但不要共用 rollout worker class。

### 1. 新增 package

新增：

```text
vrl/rewards/video_inference/
  __init__.py
  types.py
  schema.py
  artifacts.py
  runtime.py
  runtimes/
    __init__.py
    base.py
    external_http.py
    ray.py
```

职责：

- `types.py`：`VideoRewardArtifact` 等纯数据类型。
- `schema.py`：`VideoRewardRequest` / `VideoRewardResult`、score key aggregation、validation。
- `artifacts.py`：把 rollout output materialize 到 image/video/npy path，并写 manifest。
- `runtime.py`：从 config 选择 `VideoRewardRuntime`，只认 `inference_runtime`。
- `runtimes/base.py`：`VideoRewardRuntime` protocol。
- `runtimes/external_http.py`：包装现有 `RemoteVideoRewardClient`。
- `runtimes/ray.py`：repo-owned Ray reward worker pool client。

不要新增：

```text
vrl/rewards/video_inference/backends/
reward.kwargs.video_reward.backend
reward.kwargs.video_reward.local
reward.kwargs.video_reward.devices
```

### 2. `VideoReward` 改造方式

目标文件：

```text
vrl/rewards/video_reward.py
```

目标形状：

```text
VideoReward.__init__(inference_runtime, reward_name, score_key, artifact_dir, debug_dir, ...)
  -> build VideoRewardArtifactStore
  -> build VideoRewardRuntime

VideoReward.score_batch(rollouts)
  -> materialize artifacts
  -> build VideoRewardRequest
  -> await runtime.score_batch(request)
  -> validate result length / score keys / finite scores
  -> store last_results
  -> return [result.selected_score]
```

关键规则：

- `backend` 参数从 public config 删除；如果用户传 `backend=stub/local/remote`，Phase 4 后 fail-fast，并提示使用 `inference_runtime=ray|external_http`。
- fake scorer 只在 tests 里通过 direct runtime/test double 或 Ray worker scorer spec 注入。
- `VideoReward` 不直接调用 HTTP，也不直接 import heavy reward model wrapper。
- `VideoReward` 不持有 GPU tensor；artifact materialization 后只把 path/schema 发给 runtime。
- `last_results` 只做 debug/metrics，不参与 trainer 主语义。

### 3. Artifact materialization

目标文件：

```text
vrl/rewards/video_inference/artifacts.py
```

输入来自当前 collector 构造的 `Rollout`：

```text
rollout.trajectory.prompt
rollout.trajectory.output
rollout.metadata
```

输出：

```text
VideoRewardArtifact(path=..., prompt=..., sample_id=..., policy_version=...)
outputs/<run>/reward_artifacts/manifest.jsonl
```

规则：

- tensor 必须先 move 到 CPU，再写成 stable artifact；不要把 CUDA tensor 放进 Ray object store。
- image/video 保存失败必须 fail-fast，不允许 reward  silently 变 0。
- `artifact_id` 必须能 join：request、result、manifest、debug JSONL、trainer metric。
- `policy_version` 从 rollout metadata 透传；没有就写 `None`，但 async mode 必须要求非空。
- unit tests 也用 temp path artifact，不引入 `tensor_ref`。

### 4. External HTTP runtime

目标文件：

```text
vrl/rewards/video_inference/runtimes/external_http.py
vrl/rewards/remote_video.py
```

第一阶段不要重写 remote client 的 HTTP 细节。做法是：

```text
ExternalHttpVideoRewardRuntime.score_batch(request)
  -> request.artifacts -> legacy media/prompt payload
  -> RemoteVideoRewardClient.score_batch(...)
  -> VideoRewardResult list
```

迁移规则：

- `RemoteVideoRewardClient.extract_scores(...)` 的 composite `score_key="a+b"` 语义保留，但入口放到 `schema.py` 统一校验。
- raw enqueue / fetch response 写入 `VideoRewardResult.raw_response` 和 debug JSONL。
- latency 必须写 `latency_ms`；能拆就写 `queue_wait_ms` / `inference_ms`。
- `external_http` 不读取 `distributed.resources.reward`，因为 GPU 不属于本 repo。

### 5. Ray reward runtime

新增：

```text
vrl/distributed/ray/reward/
  __init__.py
  worker.py
  launcher.py
  runtime.py
  types.py
```

职责：

- `worker.py`：Ray actor，加载 reward scorer wrapper，执行 shard scoring。
- `launcher.py`：读取 resolved reward resources，创建 placement group / actors。
- `runtime.py`：collector-facing async runtime，负责 shard、gather、shutdown、release。
- `types.py`：worker metadata、runtime spec、shard request/result。

Ray actor 形状：

```text
RayVideoRewardWorker(worker_id, spec)
  load_scorer()
  score_batch(request_shard) -> list[VideoRewardResult]
  worker_metadata() -> {worker_id, node_ip, gpu_ids, reward_model_version}
  shutdown()
```

Runtime 形状：

```text
RayVideoRewardRuntime.score_batch(request)
  -> shard artifacts by artifact count / frame count
  -> actor.score_batch.remote(...)
  -> gather results
  -> preserve original artifact order
  -> validate one result per artifact
```

关键规则：

- Ray reward worker 的 `num_gpus` 来自 `distributed.resources.reward.gpus_per_worker`。
- Ray reward worker 的数量来自 `distributed.resources.reward.num_workers`。
- Ray actual assigned `gpu_ids` 必须和 resolved reward devices 对齐；不对齐 fail-fast。
- worker 返回 CPU float / JSON metadata，不返回 GPU tensor。
- reward wrapper 不接收 trainer model，不接收 rollout model live object。
- 没有真实 wrapper 时 fail-fast；只有 tests 可以注入 fake scorer。

### 6. Resource resolver 扩展

目标文件：

```text
vrl/distributed/resources.py
tests/distributed/test_resources.py
tests/distributed/test_reward_resource_lifecycle.py
```

新增 config dataclass：

```text
RewardResourceConfig(RoleResourceConfig):
  gpus_per_worker: float = 1.0
  num_workers: int | str = "auto"
  share_with_rollout: bool = False
```

新增 resolved fields：

```text
reward_devices
reward_num_gpus
reward_num_workers
reward_gpus_per_worker
reward_shared_with_rollout
```

解析顺序：

```text
visible_devices
  -> trainer_devices
  -> rollout_devices
  -> reward_devices
```

默认行为：

- 没有配置 `distributed.resources.reward` 时，`reward_num_gpus=0`，不影响非 video_reward recipe。
- `inference_runtime=ray` 必须显式或通过 recipe base 得到 `reward_num_gpus > 0`，否则 fail-fast。
- `inference_runtime=external_http` 可以没有 reward GPU。
- `reward.devices` 显式设置时必须是 `visible_devices` 子集。
- reward 和 trainer overlap 默认 fail-fast。
- reward 和 rollout overlap 默认 fail-fast；只有 `share_with_rollout=true` 且 phase-exclusive release 打开时允许。

P1 共享 GPU 的条件：

```text
distributed.resources.reward.share_with_rollout=true
distributed.rollout.release_after_collect=true
distributed.reward.release_after_score=true
distributed.resources.allow_overlap=false
```

这表达的是 rollout/reward phase-exclusive shared inference pool，不是 trainer/reward 同时占一张卡。

### 7. Placement / lifecycle

Ray reward launcher 应该镜像 rollout launcher 的风格，但不能复用 rollout worker：

```text
RayRolloutLauncher
  -> RayRolloutWorker
  -> RayDistributedRuntime

RayVideoRewardLauncher
  -> RayVideoRewardWorker
  -> RayVideoRewardRuntime
```

新增 release wrapper：

```text
ReleasableRayVideoRewardRuntime
```

语义：

- `release_after_score=false`：reward actors 常驻，适合 P0 static role allocation。
- `release_after_score=true`：score 完一个 batch 后释放 actors，适合单 GPU debug 或 P1 shared inference pool。
- release 后下一次 `score_batch(...)` 重建 actors，并重新加载 reward wrapper。
- release/recreate 必须把 model reload latency 计入 debug，不要藏起来。

### 8. Collector / trainer 集成

当前 collector 已经有正确位置：

```text
RolloutCollector._output_batch_to_experience_batch(...)
  -> reward_outputs_from_trajectory(...)
  -> RewardScorer.score(...)
  -> rollout_batch_from_trajectory(...)
```

第一版不改 collector 主流程，只做两点：

- `RewardScorer` 在调用 `VideoReward.score_batch(...)` 后，如果 reward function 暴露 `last_results`，把轻量 debug summary 写入 phase/debug sink。
- `RolloutCollector.release_runtime_memory(...)` 后续可以配合 reward runtime 的 `release_memory()`，但不要让 trainer 直接控制 reward worker。

trainer 侧规则：

- trainer 只看到 `RolloutBatch.rewards`。
- trainer 不知道 `VideoRewardArtifact`。
- trainer 不知道 `VideoRewardRuntime`。
- async replay 未实现前，trainer 不消费 delayed reward batch。

### 9. Config migration

当前 legacy：

```yaml
reward:
  kwargs:
    video_reward:
      backend: stub
```

目标：

```yaml
distributed:
  backend: ray
  resources:
    reward:
      num_gpus: 1
      gpus_per_worker: 1
      num_workers: 1
  reward:
    release_after_score: false

reward:
  kwargs:
    video_reward:
      inference_runtime: ray
      reward_name: cosmos_reason1
      score_key: overall_reward
      artifact_dir: outputs/reward_artifacts
      debug_dir: outputs/reward_debug
      model_path: ${oc.env:VIDEO_REWARD_MODEL_PATH,""}
      dtype: bf16
      reward_model_version: ${oc.env:VIDEO_REWARD_MODEL_VERSION,""}
```

迁移规则：

- `backend: stub` 删除，不作为 public fallback。
- `backend: remote` 迁到 `inference_runtime: external_http`。
- `backend: local` 迁到 `inference_runtime: ray`。
- `stub_scale` 删除；需要假分数的 tests 使用 fake runtime。
- `device` 删除；GPU ownership 只在 `distributed.resources.reward`。

### 10. 最小实现顺序

最小可合并顺序：

1. 加 `VideoRewardArtifact` / `VideoRewardRequest` / `VideoRewardResult` schema 和 tests。
2. 加 `VideoRewardArtifactStore`，让 tensor/image/video 都能 materialize 到 temp/run dir。
3. 把 `RemoteVideoRewardClient` 包到 `ExternalHttpVideoRewardRuntime`，保持现有 service 兼容。
4. 把 `VideoReward` 改成 adapter，删除 public stub backend。
5. 扩展 `distributed.resources.reward`，但先只支持 P0 static reward role。
6. 加 Ray reward worker/launcher/runtime，先允许 fake scorer worker 做 integration test。
7. 加 `release_after_score`，再接 P1 shared rollout/reward inference pool。
8. 接真实 `cosmos_reason1` / `dance_grpo` wrapper，作为 Phase 8 consumer validation。

## 目标

建立一个独立 `video_reward` inference 层，让 image/video reward model 能被所有 visual RL recipe 复用。

目标接口：

```text
VideoRewardArtifact
VideoRewardArtifactStore
VideoRewardRequest
VideoRewardResult
VideoRewardRuntime
VideoReward adapter
```

目标能力：

- reward runtime 能看见稳定 artifact，而不是临时 tensor shape。
- external_http service、Ray reward model inference、test fake scorer 共享同一套 request/result schema。
- raw request、raw response、score breakdown、latency、artifact path 都能落盘。
- trainer 仍然只通过现有 `RewardFunction.score_batch(...)` 消费 float reward。
- reward infra 支持 image 和 video，不写死 Cosmos。
- 重 reward model 可以跑在独立 GPU / Ray actor / external service 上。
- reward scoring 有 batch-level latency、queue wait、model inference time、artifact materialization time 的可观测记录。
- 每个 result 记录 `policy_version`、`reward_model_version`，为后续 async overlap 留出 correctness guard。
- Cosmos-Predict2.5 + DiffusionNFT 可以作为真实 video reward 验收，但不是这个 infra 的唯一目标。

## 不做的事

- 不把 `dance_grpo` / `cosmos_reason1` 写死进 Cosmos trainer。
- 不把 reward model 加载进训练 policy module。
- 不默认把 reward model 加载进 rollout worker。rollout worker 只负责 sample generation；heavy reward inference 有自己的 runtime。
- 不把 external reward service 当成本地必需依赖。
- 不引入任何 public `backend` 字段，包括 `backend: stub`。测试或本地调试需要假分数时，也通过 `inference_runtime=ray` 的 Ray worker 注入 fake scorer。
- 不把 OCR/aesthetic-only 当成 future `video_reward` 验收。
- 不做 supervised V2W / SFT / reconstruction loss。
- 不把 generated video artifact 只放在 transient tensor 里。
- 第一版不做 stale RL，不让 trainer 消费 policy version 不清楚的 delayed reward batch。
- 不把 reward scoring 写成 per-artifact serial loop；runtime 必须有 batch / bounded-concurrency 入口。

## 设计

### 1. VideoRewardArtifact

新增 artifact contract，表达 reward model 实际看到的媒体：

```text
vrl/rewards/video_inference/types.py
```

建议字段：

```text
artifact_id: str
media_type: "image" | "video"
prompt: str
sample_id: str
seed: int | None
path: str | None
shape: tuple[int, ...] | None
fps: float | None
num_frames: int | None
model_family: str
task: str
metadata: dict[str, object]
policy_version: str | int | None
```

原则：

- `artifact_id` 是 reward request、debug、trainer metric 的 join key。
- `path` 优先用于 audit 和 external_http/Ray worker 解耦。
- 不提供 `tensor_ref` 字段。unit tests 也应该写 temp artifact path，避免 production code 走未定义的 tensor shortcut。
- video 必须带 `fps` / `num_frames` 或显式 unknown。

### 2. VideoRewardArtifactStore

新增 artifact store，把 rollout 输出稳定保存：

```text
vrl/rewards/video_inference/artifacts.py
```

建议输出：

```text
outputs/<run>/reward_artifacts/manifest.jsonl
outputs/<run>/reward_artifacts/videos/*.mp4
outputs/<run>/reward_artifacts/images/*.png
outputs/<run>/reward_artifacts/tensors/*.npy
```

manifest 每行至少包含：

```text
artifact_id
prompt
media_type
path
sample_id
seed
fps
num_frames
model_family
task
reward_scores
metadata
```

原则：

- reward model inference 前先 materialize artifact。
- debug 不依赖训练进程里的 live tensor。
- 失败时也要能从 artifact manifest 复现 reward call。

### 3. VideoRewardRequest / VideoRewardResult

新增统一 request/result schema：

```text
vrl/rewards/video_inference/schema.py
```

`VideoRewardRequest`：

```text
request_id: str
reward_name: str
score_key: str
score_aggregation: "sum"
artifacts: tuple[VideoRewardArtifact, ...]
inference_runtime: "ray" | "external_http"
timeout_s: float
metadata: dict[str, object]
```

`VideoRewardResult`：

```text
request_id: str
artifact_id: str
reward_name: str
scores: dict[str, float]
selected_score: float
inference_runtime: "ray" | "external_http"
reward_model_version: str | None
policy_version: str | int | None
latency_ms: float | None
queue_wait_ms: float | None
inference_ms: float | None
raw_response: dict[str, object] | None
error: str | None
```

原则：

- request 保留现有 config 命名：`reward.kwargs.video_reward.score_key`。
- composite score 沿用当前 `RemoteVideoRewardClient.extract_scores(...)` 语义：`score_key="a+b"` 拆成 `("a", "b")`，`selected_score = scores["a"] + scores["b"]`。
- `score_aggregation` 第一版只支持 `"sum"`；weighted / first / custom aggregation 以后单独加，不能让 adapter 各自决定。
- score schema 要 fail-fast：缺 key、NaN、长度不匹配都不能静默变 0。
- raw response 是 audit payload，不参与训练主语义。
- `policy_version` 绑定 rollout batch，`reward_model_version` 绑定 reward runtime；async path 只能在这两个字段可追踪时打开。

### 4. VideoRewardRuntime

新增 runtime protocol：

```text
vrl/rewards/video_inference/runtimes/base.py
```

接口：

```python
class VideoRewardRuntime(Protocol):
    async def score_batch(
        self,
        request: VideoRewardRequest,
    ) -> list[VideoRewardResult]: ...
```

实现两类 production runtime：

```text
vrl/rewards/video_inference/runtimes/external_http.py
vrl/rewards/video_inference/runtimes/ray.py
```

要求：

- `external_http` 封装当前 `RemoteVideoRewardClient` 语义，但输入输出改成 request/result。
- `ray` 定义 repo-owned reward worker runtime 边界，第一版可以 fail-fast，直到接入明确的 reward model wrapper。
- fake scorer 不是 public runtime；只能作为 tests 或 `inference_runtime=ray` Ray worker 的 scorer implementation 注入，不能出现在 `reward.kwargs.video_reward` 的生产配置里。

### 4.1 Sync semantics vs async execution

当前 collector 边界是：

```text
rollout runtime generate
  -> collector.reward_score
  -> collector trajectory batch build
  -> trainer step
```

这个外层语义应该保留为第一版默认路径：

```text
batch N rollout
  -> materialize reward artifacts
  -> reward runtime scores batch N
  -> pack scored batch N
  -> train on batch N
  -> sync trainable weights to rollout workers
```

这样不会引入 stale policy update，也不会让同一个 train batch 混入多个 policy version。

但这不等于 reward inference 要在 trainer process 里同步串行执行。正确边界是：

- `RewardScorer` / `VideoReward` 只 await 一个 runtime future。
- external_http runtime 用 async HTTP + bounded concurrency。
- Ray reward runtime 用 Ray actors 执行 GPU inference。
- trainer 进程只拿回 CPU float scores 和 debug records。
- artifact 写盘和 reward inference 不应该保留 trainer GPU tensor。

第一版不做 pipeline overlap，先保证 correctness。后续如果 reward model 太慢，再加 explicit async queue：

```text
rollout(N+1) can run while reward/train(N) is finishing
```

开启条件：

- 所有 artifacts、scores、batch metadata 都有 `policy_version`。
- reward model frozen 时 `reward_model_version` 稳定；reward model 可更新时，一个 train batch 内版本必须一致。
- queue depth 有上限，例如 `max_inflight_scored_batches=1` 或 `2`。
- trainer 只消费完整 `ScoredRolloutBatch`，不能消费 partial reward。
- weight sync 仍然只发生在 train step 后，不能 mid-generation 更新 rollout worker。

### 4.2 GPU/resource placement

heavy video reward 的推荐部署是独立 GPU role：

```text
trainer GPU(s): backward / optimizer / checkpoint
rollout GPU(s): sample generation
reward GPU(s): `video_reward` inference
```

不要默认把 reward model 放在 rollout worker 里。原因：

- rollout worker 的生命周期是 generation serving，reward worker 的生命周期是 scoring service；两者 release / reload 时机不同。
- video reward model 可能和 rollout model 同时占用大显存，放在同一个 worker 会让 OOM 和 latency 互相污染。
- rollout throughput 和 reward throughput 的 optimal worker count 不一定一样。
- 后续 async pipeline 需要独立 queue 和 backpressure，塞进 rollout worker 会把资源边界写死。

允许的模式：

```text
inference_runtime=external_http  external service boundary; this repo does not own that GPU
inference_runtime=ray            Ray reward workers own reward GPU
```

目标资源 schema 应扩展 `distributed.resources.reward`：

```yaml
distributed:
  resources:
    trainer:
      num_gpus: 1
    rollout:
      num_gpus: auto
      gpus_per_worker: 1
      num_workers: auto
    reward:
      num_gpus: 1
      gpus_per_worker: 1
      num_workers: 1
    allow_overlap: false
```

规则：

- `reward.devices` 默认从 `visible_devices - trainer.devices - rollout.devices` 里 resolve。
- 如果 reward 和 trainer/rollout overlap，必须显式 `allow_overlap=true`，并且只作为 local debug。
- heavy video reward 的 repo-owned throughput path 是 `inference_runtime=ray`，由 Ray reward workers 执行。
- 不提供 recipe-level `local.runtime=process`；in-process fake 只能作为 runtime unit test helper。
- Ray reward worker 返回 CPU scores / JSON debug，不返回 GPU tensor。

### 4.3 GPU reuse and async scheduling

这里要把两个目标分开：

```text
GPU reuse:
  rollout and reward are serialized, but whichever phase is active can use
  the free inference GPUs.

Pipeline async:
  rollout(N+1), reward(N), train(N) overlap with versioned queues.
```

第一版不要直接跳到 full pipeline async。原因是 full async 会引入 policy staleness、queue backpressure、weight sync 时序、debug 复现难度。当前更实用的目标是先做 phase-exclusive GPU reuse。

#### P0：static role allocation

这是最简单、最安全的默认：

```text
rollout devices: fixed Ray rollout workers
reward devices: fixed Ray reward workers
```

优点是实现简单，可以后续 pipeline overlap。缺点是如果同一个 step 内 rollout 和 reward 严格串行，那么 reward GPU 在 rollout phase 空闲，rollout GPU 在 reward phase 空闲。

#### P1：phase-exclusive shared inference GPUs

如果 rollout 和 reward 当前不会同时发生，应该支持一个 shared inference pool：

```text
phase 1:
  launch Ray rollout workers on any free inference GPU
  generate artifacts
  shutdown/release rollout workers

phase 2:
  launch Ray reward workers on any free inference GPU
  score artifacts
  shutdown/release reward workers if memory is needed

phase 3:
  trainer consumes scored batch
```

这个模式可以回答“能不能让 rollout 用任何空闲 GPU”：可以，但前提是 rollout/reward actors 是 phase-scoped，上一阶段必须 release GPU。Ray 只能调度到它认为 free 的 GPU；如果 reward actor 长驻并持有 GPU，Ray 不会把那张 GPU 临时借给 rollout。

P1 的语义：

- 不引入 stale RL，因为同一个 batch 仍然是 `rollout -> reward -> train`。
- `rollout.release_after_collect=true` 已经是当前单 GPU colocate 的类似机制。
- reward 侧需要新增对称能力：`distributed.reward.release_after_score=true`。
- rollout/reward 可以共享 inference GPU pool，但 trainer GPU 不默认进入这个 pool。
- 如果要把 trainer GPU 也放进 shared pool，必须显式 offload trainer model，并且只作为 local debug。
- 每次 actor 重建会有 model reload cost；如果 reload cost 大于 idle waste，应使用 static role allocation。

候选配置保持贴近现有命名：

```yaml
distributed:
  resources:
    trainer:
      num_gpus: 1
    rollout:
      num_gpus: auto
      gpus_per_worker: 1
      num_workers: auto
    reward:
      num_gpus: auto
      gpus_per_worker: 1
      num_workers: auto
      share_with_rollout: true
    allow_overlap: false

  rollout:
    release_after_collect: true

  reward:
    release_after_score: true
```

`share_with_rollout` 是 proposed field，不要先实现成 ad-hoc device hack。实现时应该进入同一个 resource resolver / placement layer，让 Ray 实际 assigned `gpu_ids` 进入 debug log。

#### P2：versioned pipeline async

只有在 P0/P1 都跑通后再做：

```text
rollout(N+1) runs while reward(N) or train(N) runs
```

打开条件：

- batch 有 `policy_version`。
- reward result 有 `reward_model_version`。
- queue depth 有硬上限。
- rollout weight update 不能打断正在生成的 request。
- trainer 只消费完整 scored batch。
- debug records 能把 prompt、artifact、reward result、train step join 起来。

如果 GPU 数少，P2 不一定比 P1 好；没有足够独立 GPU 时，所谓 async 只会变成排队和频繁 reload。

### 5. External HTTP runtime

external_http runtime 负责：

- 把 `VideoRewardArtifact` 转成 service payload。
- 支持 path-based upload 或 tensor/npy fallback。
- 记录 enqueue / fetch / poll latency。
- 保存 raw request / raw response。
- 校验 batch size 和 score keys。

输出：

```text
outputs/<run>/reward_debug/video_reward_requests.jsonl
outputs/<run>/reward_debug/video_reward_results.jsonl
outputs/<run>/reward_debug/remote_raw.jsonl
```

第一版继续兼容 cosmos-rl style endpoint：

```text
enqueue_url
fetch_url
reward_name
score_key
video_infos
```

但不要把 cosmos-rl response shape 写进上层 `VideoReward`。

### 6. Ray reward runtime

Ray reward runtime 是未来大块，必须先设计边界再接模型。这里的重点不是“能在本进程 import 一个 reward model”，而是把本地 reward model 作为 Ray-managed GPU worker runtime。

目标：

```text
VideoRewardRuntime
  load reward model
  score artifacts
  release resources
```

建议文件：

```text
vrl/rewards/video_inference/runtimes/ray.py
vrl/rewards/video_inference/runtime.py
vrl/distributed/ray/reward/worker.py
vrl/distributed/ray/reward/launcher.py
vrl/distributed/ray/reward/types.py
```

设计原则：

- reward model wrapper 不和 policy model 混在同一个 module。
- Ray reward runtime 默认就是 Ray actor worker。
- 独立 device / worker count 来自 `distributed.resources.reward`，不从 `reward.kwargs.video_reward` 私自解析 GPU。
- in-process local 不作为 recipe runtime；schema/unit tests 使用 direct fake runtime/test double，Ray integration tests 使用 `inference_runtime=ray` + fake scorer worker。
- Ray reward wrapper 必须声明 `reward_model_version`。
- Ray reward runtime 不存在时 fail-fast，不能自动退回 fake scorer。
- worker 启动时加载 reward model 一次，后续只接收 artifact request。
- Ray worker pool 按 artifact count / frame count shard，不能按 rollout loop 一个个串行调用。
- launcher 负责 resource allocation、worker healthcheck、shutdown、debug metadata。

后续 candidate：

```text
dance_grpo local wrapper
cosmos_reason1 local wrapper
video-caption / VLM judge wrapper
```

### 7. VideoReward adapter

`VideoReward` 长期只做薄 adapter：

```text
Rollout -> VideoRewardArtifactStore -> VideoRewardRequest -> VideoRewardRuntime -> float scores
```

建议文件：

```text
vrl/rewards/video_reward.py
```

要求：

- 保留 `RewardFunction.score_batch(...)` API。
- 不在 `VideoReward` 里写 runtime-specific protocol。
- `VideoReward` 从 config 选择 runtime，并把 rollout metadata 转成 artifact metadata。
- `VideoReward.last_results` 或 debug writer 可用于 trainer 记录 component score breakdown。

### 8. Config

建议在当前 `configs/base/reward/video_reward.yaml` 上增量扩展，不做一次性 config rename：

```yaml
distributed:
  backend: ray
  resources:
    reward:
      num_gpus: 1
      gpus_per_worker: 1
      num_workers: 1
  reward:
    release_after_score: false

reward:
  components:
    video_reward: 1.0
  kwargs:
    video_reward:
      inference_runtime: ray
      reward_name: cosmos_reason1
      score_key: overall_reward
      media_type: video
      artifact_dir: outputs/reward_artifacts
      debug_dir: outputs/reward_debug
      timeout_s: 60.0
      max_inflight_batches: 1
      scheduling: sync
      model_path: ${oc.env:VIDEO_REWARD_MODEL_PATH,""}
      dtype: bf16
      reward_model_version: ${oc.env:VIDEO_REWARD_MODEL_VERSION,""}
      enqueue_url: ${oc.env:REMOTE_REWARD_ENQUEUE_URL,""}
      fetch_url: ${oc.env:REMOTE_REWARD_FETCH_URL,""}
      token: ${oc.env:REMOTE_REWARD_TOKEN,""}
      poll_interval_s: 1.0
      max_wait_s: 600.0
```

原则：

- 当前 `configs/base/reward/video_reward.yaml` 仍然是 legacy `backend: stub`；这是必须删除的 code debt，不是目标 public config。Phase 4 必须把默认/local debug 路径迁到 `inference_runtime=ray` Ray fake scorer path。
- 保持当前 `video_reward` kwargs 命名，不把 legacy backend 字段迁到另一套 namespace。
- Ray reward runtime 的 GPU/worker sizing 只走 `distributed.resources.reward`。
- `model_path` / `dtype` / `reward_model_version` 是 Ray runtime 的 model config，不是 GPU placement。
- `enqueue_url` / `fetch_url` / `token` 只在 `inference_runtime=external_http` 时读取；该模式不消费本 repo 的 `distributed.resources.reward`。
- `scheduling: sync` 是第一版默认值；phase-shared 或 pipeline async 必须单独打开并记录 version。

## 实施阶段

当前阶段的实现优先级是：

```text
schema/artifact -> runtime protocol -> external_http migration -> VideoReward adapter
  -> reward resource resolver -> Ray reward workers
  -> latency/version guard
```

也就是说，先把 reward result 的 contract 和 debug/audit 固定下来，再接 Ray reward worker。不要先把某个具体 reward model 直接塞进 worker 里，否则 schema 会被单个模型反向污染。

### Phase 1：schema + artifact store

编辑：

```text
vrl/rewards/video_inference/types.py
vrl/rewards/video_inference/schema.py
vrl/rewards/video_inference/artifacts.py
tests/rewards/test_video_reward_artifacts.py
tests/rewards/test_video_reward_schema.py
```

完成标准：

- image/video tensor 能 materialize 成 artifact。
- manifest 包含 artifact id、prompt、sample id、fps、path。
- bad media shape fail-fast。

### Phase 2：runtime protocol + fake scorer test double

编辑：

```text
vrl/rewards/video_inference/runtimes/base.py
tests/rewards/test_video_reward_runtime.py
```

完成标准：

- runtime 输入输出都是 request/result。
- test fake scorer 直接实现 `VideoRewardRuntime` protocol 或作为 Ray worker scorer 注入；不经过 `backend=stub` config。
- score key 缺失、NaN、长度不匹配 fail-fast。

### Phase 3：external_http runtime migration

编辑：

```text
vrl/rewards/video_inference/runtimes/external_http.py
vrl/rewards/remote_video.py
tests/rewards/test_external_http_video_reward_runtime.py
```

完成标准：

- 兼容当前 cosmos-rl remote reward service。
- raw enqueue / fetch response 进入 debug JSONL。
- `VideoRewardResult` 包含 selected score、raw response、latency。
- external_http runtime 不再把 service response shape 泄漏到 `VideoReward`。

### Phase 4：VideoReward adapter

编辑：

```text
vrl/rewards/video_reward.py
configs/base/reward/video_reward.yaml
tests/rewards/test_video_reward.py
```

完成标准：

- `VideoReward.score_batch(...)` 通过 artifact/request/runtime/result 链路。
- `MultiReward` 不需要改调用方式。
- training recipe 仍然只看到 float reward。
- debug artifact 和 raw reward result 都能落盘。
- Phase 4 删除 public `backend=stub` 行为。所有 legacy stub configs/tests 必须迁到 direct fake runtime unit tests，或 `inference_runtime=ray` Ray fake scorer integration tests。
- adapter 把现有 `score_key: "a+b"` 映射成 `VideoRewardRequest.score_key="a+b"` 和 `score_aggregation="sum"`，不引入 user-facing `score_keys` plural config。

### Phase 5：reward resource resolver and lifecycle

编辑：

```text
vrl/distributed/resources.py
vrl/distributed/ray/placement/group.py
tests/distributed/test_resources.py
tests/distributed/test_reward_resource_lifecycle.py
```

完成标准：

- `distributed.resources.reward` 进入同一个 resolver，不在 reward launcher 里私自解析 GPU。
- P0 static role allocation 能解析 trainer / rollout / reward 三个 role，overlap 默认 fail-fast。
- P1 `share_with_rollout: true` 能表达 rollout/reward phase-exclusive shared inference GPU pool。
- `distributed.reward.release_after_score=true` 是 release lifecycle 的显式开关。
- Ray actual assigned `gpu_ids` 必须进入 debug metadata；和 resolved plan 不一致时 fail-fast。
- 不把 trainer GPU 放进 shared inference pool，除非显式 allow overlap/offload local debug。

### Phase 6：Ray reward runtime boundary

编辑：

```text
vrl/rewards/video_inference/runtimes/ray.py
vrl/rewards/video_inference/runtime.py
vrl/distributed/ray/reward/worker.py
vrl/distributed/ray/reward/launcher.py
vrl/distributed/ray/reward/types.py
tests/rewards/test_ray_video_reward_runtime.py
```

完成标准：

- Phase 6 是 protocol boundary gate，不要求真实 reward model GPU inference。
- Ray reward runtime 配置存在但没有 wrapper 时 fail-fast。
- Ray reward runtime 允许声明 dtype、model path、version。
- 不允许自动退回 external_http 或 fake scorer。
- `inference_runtime=ray` 可以用 fake/test wrapper 启动 N 个 Ray reward workers，并把 request shard 分发给 workers。
- worker count / GPU ownership 来自 `distributed.resources.reward`。
- worker 返回 `VideoRewardResult`，包含 latency、reward_model_version、policy_version。
- reward workers 不持有 trainer model，不返回 GPU tensor。
- 真实 `dance_grpo` / `cosmos_reason1` wrapper inference 由 Phase 8 或后续 consumer validation gate 覆盖。

### Phase 7：latency and version guard

编辑：

```text
vrl/rollouts/collector/rewards.py
vrl/rewards/video_reward.py
vrl/rewards/video_inference/schema.py
tests/rewards/test_video_reward_versioning.py
```

完成标准：

- `RewardScorer.score(...)` 仍然返回 trainer 需要的 `torch.Tensor` reward。
- debug records 记录 artifact materialization、queue wait、inference、total reward latency。
- 每个 reward result 带 `policy_version` 和 `reward_model_version`。
- Phase 3+ production runtime 的 `VideoRewardResult.latency_ms` 必须非空；如果 runtime 能拆分 queue/inference，也要填 `queue_wait_ms` / `inference_ms`。
- 默认 `scheduling: sync` 不允许 stale batch。
- async mode 未实现前必须 fail-fast，不能 silently behave like sync。

### Phase 8：Cosmos DiffusionNFT validation consumer

编辑：

```text
configs/experiment/cosmos_predict2_5_2b_diffusionnft.yaml
vrl/scripts/cosmos/train.py
tests/config/test_load_all_experiments.py
```

完成标准：

- Cosmos recipe 可以选择 `video_reward` runtime，但不把 runtime 逻辑写进 Cosmos trainer。
- 真实 reward run 至少完成一个 optimizer step。
- 记录：

```text
global_step > 0
grad_norm > 0
trainable_sha256_changed = true
reward_std > 0 or nonzero advantage batch
```

- 生成：

```text
outputs/<run>/optimization_check.json
outputs/<run>/reward_artifacts/manifest.jsonl
outputs/<run>/reward_debug/video_reward_requests.jsonl
outputs/<run>/reward_debug/video_reward_results.jsonl
```

## 验收命令

基础 infra：

```bash
pytest tests/rewards/test_video_reward_artifacts.py \
  tests/rewards/test_video_reward_schema.py \
  tests/rewards/test_video_reward_runtime.py \
  tests/rewards/test_external_http_video_reward_runtime.py \
  tests/rewards/test_ray_video_reward_runtime.py \
  tests/rewards/test_video_reward_versioning.py \
  tests/rewards/test_video_reward.py
```

资源 resolver：

```bash
pytest tests/distributed/test_resources.py \
  tests/distributed/test_reward_resource_lifecycle.py
```

配置：

```bash
pytest tests/config/test_load_all_experiments.py
```

Cosmos consumer：

```bash
python -m vrl.scripts.train --config experiment/cosmos_predict2_5_2b_diffusionnft
```

## 完成标准

- `video_reward` inference 有独立 schema、artifact store、runtime protocol。
- `VideoReward` 是 thin adapter，不再混 runtime protocol 细节。
- external_http/ray inference runtime 边界清楚；fake scorer 不作为 public runtime。
- heavy video reward 可以作为独立 worker/service 运行，不常驻 trainer policy module 或 rollout worker。
- 默认训练语义仍是 synchronous scored batch，不引入 silent stale RL。
- reward latency 和 version 信息可追踪，后续可以安全扩展 async overlap。
- public config 不再有 `backend` 字段，包括 `backend=stub`；fake scorer 不能作为真实验收。
- reward artifact 和 reward debug result 可复现一次 reward call。
- Cosmos-Predict2.5 + DiffusionNFT 可以作为 consumer 跑真实 video reward optimizer update。
- README 不能把 Cosmos-Predict2.5 DiffusionNFT 写成 validated route，除非真实 `video_reward` runtime run 通过。

## 参考路径

- `/home/mingfeiguo/Desktop/wm-infra/vrl/rewards/video_reward.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/rewards/remote_video.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/rewards/base.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/rewards/multi.py`
- `/home/mingfeiguo/Desktop/wm-infra/configs/base/reward/video_reward.yaml`
- `/home/mingfeiguo/Desktop/wm-infra/configs/experiment/cosmos_predict2_5_2b_diffusionnft.yaml`
