# SPRINT：reward inference runtime and video reward consumer

## 结论

这个 sprint 的主体不是 Cosmos-Predict2.5，也不是 DiffusionNFT。

真正要补的是一层可复用的 reward inference runtime；`video_reward` 只是第一条重模型 consumer：

```text
rollout output artifact
  -> stable reward inference artifact
  -> RewardInferenceRuntime
  -> normalized RewardInferenceResult
  -> RL reward score
  -> debug/audit records
```

核心设计决定：

- 重 reward model 应该是独立 reward inference runtime，和 rollout runtime 一样是 data plane，不应该塞进 trainer policy module。
- repo-owned heavy execution 第一版 backend 仍然用 Ray；collector、reward、generation adapter 只依赖 top-level `vrl/ray/` 的 domain-neutral scheduling contract。
- trainer-facing API 可以继续保持同步 batch 语义：`rollout -> reward -> pack -> train`。
- runtime execution 必须支持 async / Ray worker pool；不要让 `VideoReward.score_batch(...)` 在训练进程里串行跑重模型。
- 第一版保持 on-policy correctness，不做 stale batch training；后续再用 versioned queue 做 rollout / reward / train overlap。
- reward model forward 是正常的 rollout scoring data-plane 阶段。它不是 policy training forward，也不是 diffusion/AR generation forward；它应该发生在 generation output 已经稳定之后、trainer batch packing 之前。

Cosmos-Predict2.5 + DiffusionNFT 只作为第一条真实验收 consumer。不要继续把这件事写成 Cosmos model architecture sprint，否则会让人误解成 Cosmos runtime 或 DiffusionNFT objective 还没有接好。

## 当前阶段范围

这个 sprint 保留为完整架构路线，但当前可合并交付只做 Phase 1-4：先让 `VideoReward` 不再直接懂 `stub/local/remote`，而是通过统一的 Ray-backed `RewardInferenceRuntime` 拿结果。

Phase 5-8 是后续 gate，用来约束资源规划、GPU placement smoke、版本保护和真实模型接入的方向；不要在 Phase 1-4 还没稳定时同时开做。

当前阶段只补 reward inference data plane，不改 trainer 架构。

范围内：

- 保持 trainer 是现有 driver-side online trainer。
- 保持 rollout 已有 Ray path。
- 新增 generic reward inference artifact / request / result contract。
- 新增 `RewardInferenceRuntime` protocol 和 test fake runtime。
- 新增 repo-owned Ray-backed reward runtime boundary，Ray mechanics 复用 `vrl/ray/`。
- 把 generation 现有 Ray launcher 逐步改成 `vrl/ray/` 的 thin adapter consumer，reward Ray adapter 也只能复用 `vrl/ray/`，不能私有实现 generic Ray wrapper。
- 把 video reward 作为第一条 media-specific adapter 接到 generic reward inference runtime。
- 删除 public `backend=stub/local/remote` 语义；fake scorer 只能留在 tests。
- 保持 trainer 只 await scored batch，不感知 reward worker 细节。

范围外：

- 不把 trainer 迁进 RayTrainGroup。
- 不在这个 sprint 接 FSDP trainer。
- 不做 multi-node training rank orchestration。
- 不做 rollout / reward / train full async pipeline。
- Phase 1-4 可以抽 `vrl/ray/actor_group.py` / `dependencies.py` / actor lifecycle primitives，并让 generation launcher 复用；但不迁 `vrl/generation/resources.py` 到 `vrl/ray/resources.py`。
- Phase 1-4 不做真实 GPU reward model smoke；Ray fake/test scorer 只用于 runtime boundary。
- Phase 1-4 不接真实 `cosmos_reason1` / `dance_grpo` wrapper。

后续 FSDP sprint 可以让 Ray 负责启动和放置 trainer ranks，但 rank 内训练策略仍然是 PyTorch FSDP/NCCL；这个 reward sprint 只保证 reward 侧资源边界不会和未来 trainer strategy 冲突。

## Scope control

这个 sprint 的风险不是方向错，而是 scope 太大。执行时必须按下面顺序收紧：

- `backend=stub/local/remote` 必须从 public config 删除；fake scorer 只能作为 unit test double 或 Ray integration test scorer，不进入 recipe。
- public `inference_runtime` 第一版只支持 `ray`。GPU ownership 只在 `distributed.resources.reward`，不放进 `reward.kwargs.video_reward.device`。
- Reward runtime 不能复用 rollout worker；reward latency、OOM、release lifecycle 必须和 generation worker 分开，但 Ray actor group/placement/lifecycle mechanics 必须复用 `vrl/ray/`。
- Generation 也必须复用同一套 `vrl/ray/` mechanics。`vrl/generation/ray/launcher.py` 可以保留为 generation-specific adapter，但不能长期拥有 generic actor group、placement probing、kill/shutdown helper。
- 第一版只做外层同步、内部异步；trainer 等完整 scored batch，runtime 内部可以用 Ray actor pool / bounded shard gather。
- 当前实现 checkpoint 只到 Phase 4。Phase 5-8 保留为后续 gate，不作为本次小重构的完成条件。

## 当前推荐调度

第一版采用“外层同步、内部异步”的 schedule：

```text
RolloutCollector.collect(...)
  -> generation_runtime.generate(...)
  -> RewardScorer.score(...)
  -> VideoReward.score_batch(...)
  -> RewardInferenceRuntime.score_batch(...)
  -> TrajectoryRolloutBatchBuilder.build(...)
  -> trainer step
```

外层同步的意思是：同一个 train batch 必须先完成 rollout、reward、pack，然后才训练，不引入 stale reward batch。

内部异步的意思是：`RewardInferenceRuntime.score_batch(...)` 可以用 Ray actor pool、bounded concurrency 或 shard/gather，但 trainer 进程只 await 一个 future，并只拿回 CPU float scores 和 debug metadata。

为什么 rollout 有 schedule，而 reward 第一版没有单独 schedule：

- rollout schedule 是 RL phase schedule：它管理 `generate -> reward -> pack -> train` 的跨 step 顺序、policy weight sync、driver model offload、release-after-collect、以及 one-batch-overlap correctness。
- reward runtime 是这个 schedule 里的 scoring data-plane stage。第一版只负责把一个已完成的 rollout batch 评分完成，不拥有训练 step 推进权，也不独立决定 stale batch 是否可训练。
- 如果以后做 rollout/reward/train full async pipeline，新增的应该是 rollouts orchestration / pipeline schedule，而不是 `vrl/rewards/*` 私有 schedule。reward 只提供 versioned request/result、latency、queue wait 和 release hooks。

当前不要把 heavy video reward forward 放在这些位置：

```text
trainer policy module
generation family executor
rollout worker private helper
algorithm loss
```

正确归属是：

```text
vrl/rewards/inference.py
  owns generic request -> runtime -> result contract and reward-specific sharding.

vrl/rewards/artifacts.py
  owns image/video artifact materialization and video_reward adapter glue.

vrl/rollouts/collector/rewards.py
  owns reward scoring call site.

vrl/rollouts/orchestration/*
  owns phase schedule, policy_version guard, and future overlap.
```

## 当前事实

已经存在：

```text
vrl/rewards/video_reward.py
configs/reward/video_reward.yaml
vrl/rollouts/collector/rewards.py
vrl/rollouts/collector/core.py
vrl/generation/resources.py
vrl/generation/ray/*.py
vrl/rollouts/orchestration/*
```

旧版 `VideoReward` 曾把 reward entrypoint + runtime selection + media stack + legacy remote client glue 混在一起：

```text
Rollout -> VideoReward -> legacy remote client -> float score
```

这对短期 plumbing 可以，但不适合作为长期 `video_reward` inference 层，原因是：

- 没有 first-class reward inference artifact contract。
- 没有统一 `RewardInferenceRequest` / `RewardInferenceResult`。
- Ray reward model inference 还没有 runtime 边界。
- legacy network reward path 只记录 raw service response，不够表达 artifact、model version、latency、score breakdown；它不应成为新 infra 的 public backend。
- Cosmos recipe 容易被误写成“真实 video reward 已验证”，但当前 recipe 可以退回 OCR/simple reward。
- 当前没有 dedicated `tests/rewards/test_video_reward.py`；这个 sprint 应新增它，而不是假设它已经存在。
- 当前资源 resolver 在 `vrl/generation/resources.py`，只覆盖 trainer / rollout。加 reward role 时，这个名字会过窄；Phase 5 应迁到 domain-neutral `vrl/ray/resources.py`。
- 当前 Ray helper 在 `vrl/ray/dependencies.py`。reward runtime 不能反向 import generation Ray package；共享 Ray helper 应进入 `vrl/ray/`。

## 命名收敛规则

不要在用户可见 config 里引入一套和当前 repo 不同的 reward taxonomy。

保持这些名字不变：

```text
reward.components.video_reward
reward.kwargs.video_reward
reward.kwargs.video_reward.inference_runtime: ray
vrl/rewards/video_reward.py
VideoReward
RewardScorer
```

新增内部模块也应该贴近现有名字：

```text
vrl/rewards/inference.py
RewardInferenceArtifact
RewardInferenceRequest
RewardInferenceResult
RewardInferenceRuntime
shard_reward_request

vrl/ray/
RayResourcePlan
RayPlacement
RayActorPool
RayLifecycle

vrl/rewards/artifacts.py
VideoRewardArtifact
VideoRewardArtifactStore
```

边界规则：

- `vrl/rewards/inference.py` 是 generic reward model inference substrate，未来 OCR service、image QA judge、VLM verifier、video reward 都应该复用。
- `vrl/rewards/artifacts.py` 只放 video/image/tensor artifact materialization 和 media metadata；`VideoReward` glue 留在 `vrl/rewards/video_reward.py`。
- Ray actor pool、request sharding、latency/version guard、resource lifecycle 不属于 `vrl/rewards/artifacts.py`，不能被 video reward 私有化。
- `vrl/ray/` 是 domain-neutral Ray scheduling substrate。它只表达 worker pool、placement、resources、lifecycle，不知道 generation chunk 或 reward artifact。
- 可以新增顶层 `vrl/ray/`，但不恢复 `vrl/distributed/ray/` 作为主结构。
- 顶层 `vrl/ray/` 允许直接叫 Ray，因为它的职责就是 Ray scheduling substrate。generation 现有 `vrl/generation/ray/*.py` 是兼容层；reward 也可以有很薄的 `vrl/rewards/ray/*.py` adapter，但只能放 Ray runtime/launcher glue，不能放 request/result contract、artifact materialization、scoring worker 语义。
- `vrl/generation/execution/` 继续保留 generation-specific planner、request batching、stage plan、chunk id 逻辑；这个 sprint 不把它迁到共享层。
- `vrl/ray/` 只做 domain-neutral resource / placement / actor pool / lifecycle，不能 import generation 或 rewards。
- 如果未来有非 Ray backend，另开对应 substrate sprint；当前 sprint 不抽象一个 generic backend registry。

也就是说，repo-owned reward inference 用 `inference_runtime=ray` 表达，而不是新增 `local_ray` 或 `reward_model.*` 用户配置。`reward_model_version` 只作为 result metadata 字段保留，因为它描述的是被调用模型的版本，不是配置命名空间。

## Runtime/resource split

不要用 `remote | local` 作为 GPU 资源语义。这个 sprint 不提供 public remote runtime；repo-owned reward inference 只走 Ray：

```text
inference runtime:
  who runs the inference request?

resource role:
  which GPUs can that runtime use?
```

目标 public config 应该拆成：

```yaml
reward:
  kwargs:
    video_reward:
      inference_runtime: ray
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
- 不允许 `reward.kwargs.video_reward` 直接写 `device=cuda:1` / `num_gpus` / `devices`。
- 不允许用 `backend=local` / `backend=remote` 表达资源；legacy `backend` 字段在 Phase 4 fail-fast 删除，不迁移成另一个 public backend。
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
- Slime 也支持 custom reward hooks，但本 sprint 不把 out-of-process hook 边界纳入 public infra。
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
  -> GenerationRuntimeConfig.from_cfg(...)
  -> RayGenerationLauncher
  -> RayGenerationRuntime
```

```text
MultiReward.from_dict(...)
  -> VideoReward(...)
  -> score_batch(...)
```

所以实现不应该绕开 collector / MultiReward / resource resolver。正确做法是在这些边界上增量扩展。

### 0. 保留的接口边界

保留：

- `RewardScorer.score(...)` 仍然只返回 trainer 需要的 `torch.Tensor` reward。
- `RewardFunction.score_batch(...)` 仍然是 reward function 的外部入口。
- `MultiReward` 不感知 reward inference artifact / Ray worker 细节。
- `RolloutCollector` 仍然是 `generate -> reward_score -> rollout_batch_from_trajectory`。
- trainer 不 import `vrl.rewards.inference.*` 或 `vrl.rewards.artifacts`，只看到 `RolloutBatch.rewards`。

改变：

- `VideoReward` 从“backend selector”变成 thin adapter。
- 旧 remote client path 不升级成新 runtime；它是 legacy cleanup target，不进入目标 public path。
- `distributed.resources` 从 trainer / rollout 扩展到 trainer / rollout / reward。
- Reward runtime 独立于 rollout runtime；两者可以复用 `vrl/ray/` 的 placement/logging/actor group mechanics，但不要共用 rollout worker class。

### 0.1 Directory migration map

这里的“继承”不是 Python inheritance，而是职责继承和依赖方向：generation / reward 继续拥有自己的 request/result contract，只复用 `vrl/ray/` 的底层资源和 Ray primitive。

旧的 `vrl/generation/ray/` 偏重，是历史上把三类职责放在了一起：

- generation worker semantics：load policy、execute chunk、weight sync、policy_version。
- generation execution semantics：chunk planner、engine plan、gatherer、stage plan。
- Ray substrate mechanics：placement group、actor construction、submit/wait/gather、healthcheck、shutdown。

reward 不应该复制旧结构。这个 sprint 只抽第三类里的通用部分到 `vrl/ray/`，generation 自己保留一个很薄的 `vrl/generation/ray/` adapter，reward 自己保留 request/result/worker 语义，并只新增很薄的 `vrl/rewards/ray/` runtime/launcher adapter。

不搬的目录：

```text
vrl/generation/ar/
vrl/generation/diffusion/
vrl/generation/execution/
vrl/generation/runtime/
vrl/rollouts/
```

原因：

- `vrl/generation/ar/` 和 `vrl/generation/diffusion/` 是 family-specific executors，继续负责 AR decode、diffusion layout/gather/executor。
- `vrl/generation/execution/` 是 generation-specific planning：chunks、ids、request_batch、stage_plan、planner。它懂 `GenerationRequest` / `GenerationOutput`，不能放进 generic compute。
- `vrl/generation/runtime/` 是 generation runtime config/factory/launch input wiring，继续返回 collector-facing `GenerationRuntime`。
- `vrl/rollouts/` 是 RL data flow：collect、reward scoring、batch build、orchestration、evaluators。它不变成 compute layer。

会搬出的内容：

```text
vrl/generation/resources.py
  -> vrl/ray/resources.py
```

迁移后它不再只描述 generation rollout GPU，而是统一描述 trainer / rollout / reward 三种 role：

```text
RoleResourceConfig
RolloutResourceConfig
RewardResourceConfig
DistributedResourceConfig
ResolvedDistributedResources
resolve_distributed_resources(...)
trainer_torch_device(...)
format_distributed_resource_plan(...)
```

已抽成 shared Ray primitive 的内容：

```text
vrl/ray/dependencies.py
vrl/ray/placement.py
vrl/ray/lifecycle.py
vrl/ray/actor_group.py
vrl/ray/actor_pool.py
vrl/ray/runtime.py

vrl/generation/ray/launcher.py generation-specific placement builder
  -> vrl/generation/ray/placement.py
```

具体包括：

- lazy `require_ray(...)`
- `import_from_path(...)`
- actor-side `current_node_ip(...)`
- actor-side `current_gpu_ids(...)`
- generic placement group create/remove helpers
- generic actor kill/shutdown helpers
- generic placement probing helpers where they do not mention generation request/chunk semantics

继续留在 `vrl/generation/ray/*.py` 的内容：

```text
vrl/generation/ray/executor.py
vrl/generation/ray/launcher.py
vrl/generation/ray/placement.py
vrl/generation/ray/runtime.py
vrl/generation/ray/worker.py
vrl/generation/ray/weight_sync.py
```

原因：

- launcher 仍然负责把 generation launch contract 转成 `RayGenerationRuntime`，但它应该是 thin adapter：把 `RayGenerationWorker`、worker config、资源 role 交给 `vrl/ray/RayActorGroup`，不再私有实现 generic placement probing / actor kill / shutdown。
- generation-specific placement builder 放在 `vrl/generation/ray/placement.py`，不让 `vrl/generation/ray/launcher.py` 继续膨胀。
- executor 是 Ray actor-method gatherer，使用 `vrl/ray/actor_pool.py`；它不拥有 chunk plan 构造。
- runtime 仍然实现 `GenerationRuntime.generate(...)` 和 generation weight sync。
- worker 仍然加载 policy / generation model，只处理 generation chunks。
- planner / payload types / worker core 仍在 `vrl/generation/execution/distributed/`，不变成 reward actor pool，也不进入 shared `vrl/ray/`。
- weight sync 是 train -> generation worker 的 domain-specific 逻辑，不属于 generic compute。

reward 新增的内容：

```text
vrl/rewards/inference.py
  generic RewardInferenceRequest / RewardInferenceResult / RewardInferenceRuntime

vrl/rewards/scoring_worker.py
  reward scorer worker semantics

vrl/rewards/artifacts.py
  video/image artifact materialization and VideoReward glue
```

依赖方向：

```text
vrl/generation/ray/*.py
  imports vrl.ray.* for Ray primitives
  keeps Ray launcher/runtime/actor adapter only

vrl/generation/execution/*
  keeps generation chunk planner, payload types, and worker core

vrl/rewards/inference.py
  keeps reward request/result contract, sharding, validation, and runtime protocol

vrl/rewards/ray/*.py
  keeps Ray-backed RewardInferenceRuntime adapter and build_reward_ray_runtime(...)

vrl/rewards/scoring_worker.py
  keeps reward scorer/model worker semantics

vrl/ray/*
  imports neither vrl.generation nor vrl.rewards

vrl/rollouts/*
  calls GenerationRuntime and RewardFunction/RewardScorer
  does not import vrl.ray directly in Phase 1-4
```

### 1. 新增 package

新增：

```text
vrl/ray/
  __init__.py
  resources.py          # trainer / rollout / reward role resource plan
  placement.py          # domain-neutral Ray placement intent
  actor_group.py        # domain-neutral Ray actor construction / healthcheck / lifecycle
  actor_pool.py         # domain-neutral Ray submit / wait / gather protocol
  lifecycle.py          # release / shutdown contracts
  runtime.py            # domain-neutral actor-method runtime
  types.py
  dependencies.py

vrl/rewards/inference.py        # RewardInferenceArtifact / Request / Result / public runtime selection
vrl/rewards/scoring_worker.py   # load scorer/model, validate worker config, score shards
vrl/rewards/ray/
  __init__.py
  runtime.py                    # RewardInferenceActorRuntime over vrl.ray.runtime
  launcher.py                   # build_reward_ray_runtime(...)

vrl/rewards/artifacts.py        # video/image/tensor artifact materialization
```

职责：

- `vrl/ray/*`：domain-neutral Ray scheduling substrate，包含 Ray actor pool、placement、dependency checks、lifecycle helpers，不能 import generation / rewards。
- `vrl/generation/ray/launcher.py`：保留 generation adapter 入口，但要消费 `vrl/ray/actor_group.py` / `placement.py` / `lifecycle.py`，不能继续拥有通用 Ray substrate，也不能承载 chunk planner/executor。
- `vrl/generation/ray/executor.py`：承载 Ray actor-method submit / gather 和 generation chunk result validation；它不构造 EnginePlan。
- `vrl/generation/execution/distributed/`：承载 generation distributed execution semantics；`planner.py` / `types.py` / `worker.py` 不放在 Ray adapter package 里，也不散落在 `execution/` 根目录。
- `vrl/generation/ray/worker.py`：只允许是 Ray actor wrapper；模型加载、chunk execution、profiling 逻辑放在 `vrl/generation/execution/distributed/worker.py`。
- `vrl/ray/runtime.py`：通用 Ray actor-method runtime；不知道 reward request/result。
- `vrl/rewards/inference.py`：通用 `RewardInferenceRequest` / `RewardInferenceResult`、score key aggregation、validation、request sharding、runtime protocol；`build_reward_inference_runtime(...)` 只做 public runtime selection 并委托 Ray launcher。
- `vrl/rewards/ray/runtime.py`：reward-specific Ray runtime adapter，负责 shard -> generic Ray actor method runtime -> validate result。
- `vrl/rewards/ray/launcher.py`：reward-specific Ray factory，负责把 `worker_config` / num_workers / resource knobs 转成 `vrl.ray.runtime.RayActorMethodRuntime`。
- `vrl/rewards/scoring_worker.py`：reward-specific scorer worker semantics；它可以被 `vrl/ray/` 的 generic actor runtime 包起来，但不自己定义 actor lifecycle。
- `vrl/rewards/artifacts.py`：把 image/video rollout output materialize 到 image/video/npy path，并写 manifest。

不要新增：

```text
vrl/distributed/ray/
vrl/rollouts/orchestration/runtime/
vrl/rewards/inference/ray/
vrl/rewards/inference/runtimes/
vrl/rewards/ray.py
vrl/rewards/video_inference/
vrl/rewards/video_inference/backends/
vrl/rewards/video_inference/ray/
vrl/rewards/video_inference/runtimes/base.py
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
  -> build RewardInferenceRuntime

VideoReward.score_batch(rollouts)
  -> materialize artifacts
  -> build RewardInferenceRequest
  -> await runtime.score_batch(request)
  -> validate result length / score keys / finite scores
  -> store last_results
  -> return [result.selected_score]
```

关键规则：

- `backend` 参数从 public config 删除；如果用户传 `backend=stub/local/remote`，Phase 4 后 fail-fast，并提示使用 `inference_runtime=ray`。
- fake scorer 只在 tests 里通过 direct runtime/test double 或 Ray worker config 注入。
- `VideoReward` 不直接调用 legacy network client，也不直接 import heavy reward model wrapper。
- `VideoReward` 不持有 GPU tensor；artifact materialization 后只把 path/schema 发给 runtime。
- `last_results` 只做 debug/metrics，不参与 trainer 主语义。

### 3. Artifact materialization

目标文件：

```text
vrl/rewards/artifacts.py
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

### 4. Ray-backed reward runtime boundary

目标文件：

```text
vrl/ray/
  actor_group.py        # generic Ray actor construction, healthcheck, submit/wait/gather
  runtime.py            # generic actor-method runtime
  placement.py
  lifecycle.py
  resources.py

vrl/rewards/inference.py        # RewardInferenceRuntime protocol, request/result contract, factory
vrl/rewards/scoring_worker.py   # reward scorer/model worker semantics and config validation
vrl/rewards/ray/runtime.py      # RewardInferenceActorRuntime
vrl/rewards/ray/launcher.py     # build_reward_ray_runtime(...)
```

不要把 request/result contract、artifact materialization、scoring worker 塞到 `vrl/rewards/ray/`。reward 的 workload contract 放在 `vrl/rewards/inference.py`，worker scorer 语义放在 `vrl/rewards/scoring_worker.py`，Ray 的 actor / placement / lifecycle / actor-method runtime mechanics 放在 `vrl/ray/`，reward Ray package 只做薄 adapter：

```text
RewardInferenceRequest
  -> RewardInferenceRuntime
  -> vrl.ray.RayActorMethodRuntime
  -> RewardScoringWorker.score_batch(...)
  -> list[RewardInferenceResult]
```

拆分规则：

- `vrl/rewards/inference.py`：实现 reward request sharding、result validation、runtime protocol 和 public `build_reward_inference_runtime(...)` selection；它不能自己管理 actor lifecycle。
- `vrl/rewards/ray/runtime.py`：实现 `RewardInferenceActorRuntime`，把 request shards 交给 `vrl.ray.runtime.RayActorMethodRuntime`。
- `vrl/rewards/ray/launcher.py`：实现 `build_reward_ray_runtime(...)`，把 reward worker config 和 Ray worker knobs 转成 `RayActorMethodRuntime`。
- `vrl/rewards/scoring_worker.py`：加载 scorer/model，校验 serializable worker config，执行 request shard，返回 CPU result；不 import generation。
- 不单独拆 worker config 文件；禁止 live model/callable 直接穿过 Ray boundary，worker config 必须是普通可序列化数据。
- `vrl/ray/actor_group.py`：创建 Ray actors、healthcheck、submit/wait/gather。
- `vrl/ray/runtime.py`：通用 actor-method runtime，负责 actor group lifecycle、bounded method call、release-after-call。
- `vrl/ray/placement.py`：根据 resolved role resources 创建 placement/bundles，并验证 assigned GPU metadata。

### 5. Shared Ray actor substrate

Reward runtime 和 generation runtime 都使用同一个 actor substrate，但不共享 worker class。通用 submit / wait / gather / placement / lifecycle 必须先进 `vrl/ray/`。

新增：

```text
vrl/ray/
  dependencies.py
  actor_group.py
  actor_pool.py
  runtime.py
  placement.py
  resources.py
  lifecycle.py
  types.py

vrl/rewards/inference.py
vrl/rewards/scoring_worker.py
vrl/generation/ray/executor.py
vrl/generation/execution/distributed/planner.py
vrl/generation/execution/distributed/types.py
vrl/generation/execution/distributed/worker.py
vrl/generation/ray/launcher.py
```

职责：

- `vrl/ray/actor_group.py`：generic Ray actor construction / healthcheck / lifecycle wrapper。
- `vrl/ray/actor_pool.py`：generic Ray actor submit / wait / gather。
- `vrl/ray/runtime.py`：generic Ray actor-method runtime for workload adapters。
- `vrl/ray/placement.py`：generic placement group / bundle helpers。
- `vrl/ray/lifecycle.py`：generic shutdown / release helpers。
- `vrl/generation/ray/launcher.py`：改成 shared Ray actor substrate 的 generation adapter，仍然组装 `RayGenerationRuntime`。
- `vrl/generation/execution/distributed/*`：保留 chunk planner、payload types、worker core；这些不是 Ray package 的职责。
- `vrl/rewards/inference.py`：构造 reward shard request、验证 `RewardInferenceResult`。
- `vrl/rewards/scoring_worker.py`：加载 reward scorer wrapper、返回 `RewardInferenceResult`。

Ray actor 形状：

```text
RayActorMethodRuntime(RewardScoringWorker, worker_id, worker_config)
  load_scorer()
  score_batch(request_shard) -> list[RewardInferenceResult]
  worker_metadata() -> {worker_id, node_ip, gpu_ids, reward_model_version}
  shutdown()
```

Runtime 形状：

```text
RewardInferenceRuntime.score_batch(request)
  -> shard artifacts by artifact count / frame count
  -> vrl.ray actor group submit/wait/gather
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
vrl/ray/resources.py
vrl/generation/runtime/config.py
vrl/generation/ray/launcher.py
vrl/scripts/common/online.py
tests/ray/test_resources.py
tests/rewards/test_reward_resource_lifecycle.py
```

当前 resolver 在 `vrl/generation/resources.py`。加 reward role 时不要继续让 generation package 拥有 trainer / rollout / reward 全局资源规划；Phase 5 应把该文件迁到 `vrl/ray/resources.py`，更新调用方，并删除旧位置。不要新增 `vrl/distributed` 兼容目录。

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

Reward runtime 应该复用 `vrl/ray/` 的 actor group / placement / lifecycle；`vrl/rewards/ray/launcher.py` 只能是薄 factory，不能私有实现 actor group / placement / lifecycle：

```text
RayRolloutLauncher
  -> RayRolloutWorker
  -> RayDistributedRuntime

RewardInferenceRuntime
  -> vrl.ray.RayActorGroup
  -> RewardInferenceWorker
```

新增 release wrapper 不需要再建第二层 launcher；它可以是 `RewardInferenceRuntime` 的 lifecycle policy：

```text
RewardInferenceRuntime(release_after_score=true)
```

语义：

- `release_after_score=false`：reward actors 常驻，适合 P0 static role allocation。
- `release_after_score=true`：score 完一个 batch 后释放 actors，适合单 GPU debug 或 P1 shared inference pool。
- release 后下一次 `score_batch(...)` 重建 actors，并重新加载 reward wrapper。
- release/recreate 必须把 model reload latency 计入 debug，不要藏起来。

### 8. Collector / trainer 集成

当前 collector 已经有正确位置：

```text
RolloutCollector._output_batch_to_rollout_batch(...)
  -> TrajectoryRolloutBatchBuilder.reward_scoring_input(...)
  -> RewardScorer.score(...)
  -> TrajectoryRolloutBatchBuilder.build(...)
```

第一版不改 collector 主流程，只做两点：

- `RewardScorer` 在调用 `VideoReward.score_batch(...)` 后，如果 reward function 暴露 `last_results`，把轻量 debug summary 写入 phase/debug sink。
- `RolloutCollector.release_runtime_memory(...)` 后续可以配合 reward runtime 的 `release_memory()`，但不要让 trainer 直接控制 reward worker。

trainer 侧规则：

- trainer 只看到 `RolloutBatch.rewards`。
- trainer 不知道 `VideoRewardArtifact`。
- trainer 不知道 `RewardInferenceRuntime`。
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
- `backend: remote` 删除，不迁移成新的 public runtime。
- `backend: local` 删除；repo-owned heavy inference 统一走 `inference_runtime: ray`。
- `stub_scale` 删除；需要假分数的 tests 使用 fake runtime。
- `device` 删除；GPU ownership 只在 `distributed.resources.reward`。

### 10. 最小实现顺序

当前必须先合并的顺序：

1. 加 `RewardInferenceRequest` / `RewardInferenceResult` schema 和 tests。
2. 加 `VideoRewardArtifactStore`，让 tensor/image/video 都能 materialize 到 temp/run dir。
3. 加 shared Ray actor substrate；generation launcher 和 reward runtime 都必须通过它启动 actors，reward 先用 fake/test scorer 验证 actor group / worker / runtime contract。
4. 把 `VideoReward` 改成 adapter，删除 public stub/remote/local backend。

后续 gate：

5. 扩展 `distributed.resources.reward` 和 release lifecycle，但先只支持 P0 static reward role。
6. 加真实 GPU smoke reward worker，验证 resource resolver -> Ray placement -> GPU assignment -> score result 链路。
7. 加 reward latency/version guard，再接 P1 shared rollout/reward inference pool。
8. 接真实 `cosmos_reason1` / `dance_grpo` wrapper，作为 Phase 8 consumer validation。

## 目标

建立一个独立 generic reward inference 层，让 image/video reward、VLM judge、verifier、future OCR service 都能复用同一套调度、Ray worker、资源生命周期。

目标接口：

```text
RewardInferenceArtifact
RewardInferenceRequest
RewardInferenceResult
RewardInferenceRuntime
RewardInferenceScheduler
VideoRewardArtifact
VideoRewardArtifactStore
VideoReward adapter
```

目标能力：

- reward runtime 能看见稳定 artifact，而不是临时 tensor shape。
- Ray reward model inference 和 test fake scorer 共享同一套 generic request/result schema。
- raw request、raw response、score breakdown、latency、artifact path 都能落盘。
- trainer 仍然只通过现有 `RewardFunction.score_batch(...)` 消费 float reward。
- reward infra 支持 image、video、VLM judge、verifier，不写死 Cosmos。
- 重 reward model 可以跑在独立 GPU / Ray actor 上。
- reward scoring 有 batch-level latency、queue wait、model inference time、artifact materialization time 的可观测记录。
- 每个 result 记录 `policy_version`、`reward_model_version`，为后续 async overlap 留出 correctness guard。
- Cosmos-Predict2.5 + DiffusionNFT 可以作为真实 video reward 验收，但不是这个 infra 的唯一目标。

## 不做的事

- 不把 `dance_grpo` / `cosmos_reason1` 写死进 Cosmos trainer。
- 不把 reward model 加载进训练 policy module。
- 不默认把 reward model 加载进 rollout worker。rollout worker 只负责 sample generation；heavy reward inference 有自己的 runtime。
- 不把 reward serving endpoint 作为 public runtime 或本地必需依赖。
- 不引入任何 public `backend` 字段，包括 `backend: stub`。测试需要假分数时，Phase 1-4 用 direct fake runtime/test double 或 Ray fake scorer integration test。两者都不能进入 recipe。
- 不把 OCR/aesthetic-only 当成 future `video_reward` 验收；它们可以以后接 generic reward inference runtime，但不是本 sprint 的真实 video consumer。
- 不做 supervised V2W / SFT / reconstruction loss。
- 不把 generated video artifact 只放在 transient tensor 里。
- 第一版不做 stale RL，不让 trainer 消费 policy version 不清楚的 delayed reward batch。
- 不把 reward scoring 写成 per-artifact serial loop；runtime 必须有 batch / bounded-concurrency 入口。

## 设计

### 1. VideoRewardArtifact

新增 artifact contract，表达 reward model 实际看到的媒体：

```text
vrl/rewards/artifacts.py
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
- `path` 优先用于 audit 和 Ray worker 解耦。
- 不提供 `tensor_ref` 字段。unit tests 也应该写 temp artifact path，避免 production code 走未定义的 tensor shortcut。
- video 必须带 `fps` / `num_frames` 或显式 unknown。

### 2. VideoRewardArtifactStore

新增 artifact store，把 rollout 输出稳定保存：

```text
vrl/rewards/artifacts.py
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
- Phase 1-4 artifact materialization 必须在 driver side 完成；Ray reward workers 只读取 artifact path，不并发写主 manifest 或主 media 目录。
- debug 不依赖训练进程里的 live tensor。
- 失败时也要能从 artifact manifest 复现 reward call。
- 如果后续 Ray worker 必须写派生文件，只能写到 `worker_artifacts/<request_id>/<worker_id>/` 这种 worker-private 子目录；driver 负责汇总 manifest。
- 禁止多个 worker 同时 append 同一个 `manifest.jsonl` 或写同一个 `videos/` / `images/` filename namespace。

### 3. RewardInferenceRequest / RewardInferenceResult

新增统一 request/result schema：

```text
vrl/rewards/inference.py
```

`RewardInferenceRequest`：

```text
request_id: str
reward_name: str
score_key: str
score_aggregation: "sum"
artifacts: tuple[RewardInferenceArtifact, ...]
inference_runtime: "ray"
timeout_s: float
metadata: dict[str, object]
```

`RewardInferenceResult`：

```text
request_id: str
artifact_id: str
reward_name: str
scores: dict[str, float]
selected_score: float
inference_runtime: "ray"
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
- composite score 由 schema 统一定义：`score_key="a+b"` 拆成 `("a", "b")`，`selected_score = scores["a"] + scores["b"]`。
- `score_aggregation` 第一版只支持 `"sum"`；weighted / first / custom aggregation 以后单独加，不能让 adapter 各自决定。
- score schema 要 fail-fast：缺 key、NaN、长度不匹配都不能静默变 0。
- raw response 是 audit payload，不参与训练主语义。
- `policy_version` 绑定 rollout batch，`reward_model_version` 绑定 reward runtime；async path 只能在这两个字段可追踪时打开。

### 4. RewardInferenceRuntime

新增 runtime protocol：

```text
vrl/rewards/inference.py
```

接口：

```python
class RewardInferenceRuntime(Protocol):
    async def score_batch(
        self,
        request: RewardInferenceRequest,
    ) -> list[RewardInferenceResult]: ...
```

实现一个 production runtime：

```text
vrl/rewards/inference.py
vrl/ray/runtime.py
```

要求：

- `ray` 通过 `vrl/ray/runtime.py` 的 generic actor-method runtime 执行 repo-owned reward worker，第一版可以 fail-fast，直到接入明确的 reward model wrapper。
- fake scorer 不是 public runtime；Phase 1-4 可以作为 tests 的 direct runtime/test double，也可以作为 `inference_runtime=ray` Ray worker 的 scorer implementation 注入；不能出现在 `reward.kwargs.video_reward` 的生产配置里。

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
- Reward runtime 用 `vrl/ray/` actor group 执行 GPU inference。
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
resource resolver 本身不能 import Ray；它只产出 domain-neutral role plan。Ray placement 由 `vrl/ray/placement.py` 消费这个 plan。

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

### 5. Reward runtime over shared Ray substrate

Reward runtime 是未来大块，必须先设计边界再接模型。这里的重点不是“能在本进程 import 一个 reward model”，而是把本地 reward model 作为 shared `vrl/ray/` 管理的 GPU actor worker。

目标：

```text
RewardInferenceRuntime
  shard request
  submit to vrl.ray.RayActorMethodRuntime
  gather RewardInferenceResult
  apply release policy
```

建议文件：

```text
vrl/rewards/inference.py
vrl/rewards/scoring_worker.py
vrl/ray/dependencies.py
vrl/ray/actor_group.py
vrl/ray/actor_pool.py
vrl/ray/runtime.py
vrl/ray/placement.py
vrl/ray/resources.py
vrl/ray/lifecycle.py
vrl/ray/types.py
```

设计原则：

- reward model wrapper 不和 policy model 混在同一个 module。
- reward worker 默认通过 `vrl/ray/` actor group 运行。
- worker config 由 `runtime.py` 从 config 组装，由 `worker.py` 校验；不要单独拆文件。
- 独立 device / worker count 来自 `distributed.resources.reward`，不从 `reward.kwargs.video_reward` 私自解析 GPU。
- in-process local 不作为 recipe runtime；schema/unit tests 使用 direct fake runtime/test double，Ray integration tests 使用 `inference_runtime=ray` + fake scorer worker。
- reward wrapper 必须声明 `reward_model_version`。
- shared Ray actor substrate 不存在时 fail-fast，不能自动退回 fake scorer。
- worker 启动时加载 reward model 一次，后续只接收 artifact request。
- Ray worker pool 按 artifact count / frame count shard，不能按 rollout loop 一个个串行调用。
- `vrl/ray/actor_group.py` 负责 resource allocation、worker healthcheck、shutdown、debug metadata。
- reward runtime 不能 import `vrl.ray.dependencies`；共享 Ray import/healthcheck helper 先放到 `vrl/ray/dependencies.py`。

后续 candidate：

```text
dance_grpo local wrapper
cosmos_reason1 local wrapper
video-caption / VLM judge wrapper
```

### 6. VideoReward adapter

`VideoReward` 长期只做薄 adapter：

```text
Rollout -> VideoRewardArtifactStore -> RewardInferenceRequest -> RewardInferenceRuntime -> float scores
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

### 7. Config

建议在当前 `configs/reward/video_reward.yaml` 上增量扩展，不做一次性 config rename：

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
```

原则：

- 当前 `configs/reward/video_reward.yaml` 仍然是 legacy `backend: stub`；这是必须删除的 code debt，不是目标 public config。Phase 4 删除默认/local debug recipe path，相关测试迁到 direct fake runtime/test double 或 Phase 3 Ray fake scorer integration test。
- 保持当前 `video_reward` kwargs 命名，不把 legacy backend 字段迁到另一套 namespace。
- Ray-backed reward worker 的 GPU/worker sizing 只走 `distributed.resources.reward`。
- `model_path` / `dtype` / `reward_model_version` 是 Ray runtime 的 model config，不是 GPU placement。
- `scheduling: sync` 是第一版默认值；phase-shared 或 pipeline async 必须单独打开并记录 version。

## 实施阶段

当前实现 checkpoint 是 Phase 1-4：

```text
generic schema -> video artifact store -> runtime protocol -> shared Ray actor substrate -> VideoReward adapter
```

后续 gate 才进入：

```text
reward resource resolver -> GPU reward smoke worker
  -> latency/version guard -> real model consumer validation
```

也就是说，先把 reward result 的 contract 和 debug/audit 固定下来，再接真实 GPU reward wrapper。不要先把某个具体 reward model 直接塞进 worker 里，否则 schema 会被单个模型反向污染。

### Phase 1：generic schema + video artifact store

编辑：

```text
vrl/rewards/inference.py
vrl/rewards/artifacts.py
tests/rewards/test_video_reward_artifacts.py
tests/rewards/test_reward_inference_schema.py
```

完成标准：

- `RewardInferenceRequest` / `RewardInferenceResult` 不包含 video-specific field。
- image/video tensor 能 materialize 成 artifact。
- manifest 包含 artifact id、prompt、sample id、fps、path。
- artifact store 是 driver-side writer，tests 覆盖不会让多个 worker append 同一个 manifest。
- bad media shape fail-fast。

### Phase 2：runtime protocol + fake scorer test double

编辑：

```text
vrl/rewards/inference.py
tests/rewards/test_reward_inference_runtime.py
```

完成标准：

- runtime 输入输出都是 request/result。
- test fake scorer 直接实现 `RewardInferenceRuntime` protocol；Phase 2 不经过 `backend=stub` config，也不走 Ray worker。
- score key 缺失、NaN、长度不匹配 fail-fast。

### Phase 3：shared Ray actor substrate + generation/reward adapters

编辑：

```text
vrl/rewards/inference.py
vrl/rewards/scoring_worker.py
vrl/ray/dependencies.py
vrl/ray/actor_group.py
vrl/ray/actor_pool.py
vrl/ray/runtime.py
vrl/ray/placement.py
vrl/ray/lifecycle.py
vrl/ray/types.py
vrl/rewards/ray/runtime.py
vrl/rewards/ray/launcher.py
vrl/generation/ray/executor.py
vrl/generation/execution/distributed/planner.py
vrl/generation/execution/distributed/types.py
vrl/generation/execution/distributed/worker.py
vrl/generation/ray/launcher.py
tests/ray/test_ray_actor_pool.py
tests/generation/ray/test_rollout_launcher.py
tests/rewards/test_ray_reward_inference_runtime.py
```

完成标准：

- `vrl/rewards/ray/` 只能包含 `__init__.py` / `runtime.py` / `launcher.py`；不能放 contract、artifact store、worker semantics 或 `spec.py`。
- 不新增 `vrl/rewards/ray.py`。
- `vrl/ray/` 提供 generic actor group / actor pool / placement / lifecycle。
- `vrl/generation/ray/launcher.py` 变成 thin adapter，复用 `vrl/ray/actor_group.py` / `placement.py` / `lifecycle.py`，现有 rollout launcher tests 继续通过。
- `vrl/generation/execution/distributed/` 承载真实 chunk planner、payload types、worker core；不能重新拆成 `execution/distributed_*.py`。
- `vrl/generation/ray/executor.py` 承载 Ray actor-method submit / gather；不能把 EnginePlan 构造塞回 Ray adapter。
- `vrl/generation/ray/worker.py` 只能是 Ray actor wrapper；真实 worker core 必须在 `vrl/generation/execution/distributed/worker.py`。
- 不新增 `vrl/rewards/inference/` package；`vrl/rewards/inference.py` 是单个 domain contract/factory module。
- `vrl/ray/runtime.py` 提供 actor-method runtime；`vrl/rewards/ray/runtime.py` 只做 reward request/result adapter。
- `vrl/rewards/scoring_worker.py` 提供 reward-specific worker semantics，并校验 serializable worker config；不单独拆 worker config 文件。
- runtime 配置存在但没有 scorer wrapper 时 fail-fast。
- `inference_runtime=ray` 可以用 fake/test wrapper 启动 N 个 Ray reward workers，并把 request shard 分发给 workers。
- worker count / GPU ownership 来自 `distributed.resources.reward`。
- worker 返回 `RewardInferenceResult`，包含 latency、reward_model_version、policy_version。
- reward workers 不持有 trainer model，不返回 GPU tensor。

### Phase 4：VideoReward adapter

编辑：

```text
vrl/rewards/video_reward.py
configs/reward/video_reward.yaml
tests/rewards/test_video_reward.py
```

完成标准：

- `VideoReward.score_batch(...)` 通过 artifact/request/runtime/result 链路。
- `MultiReward` 不需要改调用方式。
- training recipe 仍然只看到 float reward。
- debug artifact 和 raw reward result 都能落盘。
- Phase 4 删除 public `backend=stub` 行为。所有 legacy stub configs/tests 必须迁到 direct fake runtime unit tests；`inference_runtime=ray` Ray fake scorer integration tests 在 Phase 3 覆盖。
- adapter 把现有 `score_key: "a+b"` 映射成 `RewardInferenceRequest.score_key="a+b"` 和 `score_aggregation="sum"`，不引入 user-facing `score_keys` plural config。

### Phase 5：shared resource resolver and lifecycle

编辑：

```text
vrl/ray/resources.py
vrl/generation/runtime/config.py
vrl/generation/ray/launcher.py
vrl/scripts/common/online.py
tests/ray/test_resources.py
tests/generation/ray/test_rollout_launcher.py
tests/rewards/test_reward_resource_lifecycle.py
```

完成标准：

- `distributed.resources.reward` 进入同一个 resolver，不在 reward runtime 里私自解析 GPU。
- 当前 `vrl/generation/resources.py` 迁到 `vrl/ray/resources.py`；更新所有 imports，不保留长期 alias。
- generation runtime config 和 launcher 必须消费 `vrl/ray/resources.py`，不能继续从 `vrl/generation/resources.py` 读全局 trainer/rollout resource plan。
- P0 static role allocation 能解析 trainer / rollout / reward 三个 role，overlap 默认 fail-fast。
- P1 `share_with_rollout: true` 能表达 rollout/reward phase-exclusive shared inference GPU pool。
- `distributed.reward.release_after_score=true` 是 release lifecycle 的显式开关。
- Ray actual assigned `gpu_ids` 必须进入 debug metadata；和 resolved plan 不一致时 fail-fast。
- 不把 trainer GPU 放进 shared inference pool，除非显式 allow overlap/offload local debug。

### Phase 6：GPU reward runtime smoke

这一阶段补 shared Ray actor substrate 和真实 Cosmos consumer 之间的缺口。它不接 `cosmos_reason1`，只验证 repo-owned reward GPU worker 链路真的可用。

编辑：

```text
vrl/rewards/inference.py
vrl/rewards/scoring_worker.py
vrl/ray/runtime.py
tests/rewards/test_ray_reward_gpu_smoke.py
```

完成标准：

- 使用一个真实申请 GPU 的 trivial reward scorer wrapper，例如加载一个很小的 `torch.nn.Module`，检查 artifact tensor/media shape，然后返回 float score。
- scorer 必须通过 `inference_runtime=ray` 启动 Ray reward actor，不允许 in-process fallback。
- 测试必须覆盖 `distributed.resources.reward -> vrl/ray/resources.py -> Ray placement -> assigned gpu_ids -> RewardInferenceResult`。
- worker metadata 记录 `worker_id`、`gpu_ids`、`reward_model_version`、`node_ip`。
- 如果环境没有 GPU，测试可以 skip；但代码路径不能降级成 CPU fake scorer 后伪装通过。
- artifact 写入仍然由 driver side 完成；worker 只读 artifact path。

### Phase 7：latency and version guard

编辑：

```text
vrl/rollouts/collector/rewards.py
vrl/rewards/video_reward.py
vrl/rewards/inference.py
tests/rewards/test_video_reward_versioning.py
```

完成标准：

- `RewardScorer.score(...)` 仍然返回 trainer 需要的 `torch.Tensor` reward。
- debug records 记录 artifact materialization、queue wait、inference、total reward latency。
- 每个 reward result 带 `policy_version` 和 `reward_model_version`。
- Phase 3+ production runtime 的 `RewardInferenceResult.latency_ms` 必须非空；如果 runtime 能拆分 queue/inference，也要填 `queue_wait_ms` / `inference_ms`。
- 默认 `scheduling: sync` 不允许 stale batch。
- async mode 未实现前必须 fail-fast，不能 silently behave like sync。

### Phase 8：Cosmos DiffusionNFT validation consumer

编辑：

```text
configs/experiment/online/ocr/video_diffusion_nft.yaml
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

当前 checkpoint，Phase 1-4：

```bash
pytest tests/rewards/test_video_reward_artifacts.py \
  tests/rewards/test_reward_inference_schema.py \
  tests/rewards/test_reward_inference_runtime.py \
  tests/ray/test_ray_actor_pool.py \
  tests/generation/ray/test_rollout_launcher.py \
  tests/rewards/test_ray_reward_inference_runtime.py \
  tests/rewards/test_video_reward.py
```

后续 gate，Phase 5-8：

```bash
pytest tests/ray/test_resources.py \
  tests/ray/test_ray_actor_pool.py \
  tests/generation/ray/test_rollout_launcher.py \
  tests/rewards/test_reward_resource_lifecycle.py \
  tests/rewards/test_ray_reward_inference_runtime.py \
  tests/rewards/test_ray_reward_gpu_smoke.py \
  tests/rewards/test_video_reward_versioning.py
```

配置：

```bash
pytest tests/config/test_load_all_experiments.py
```

Cosmos consumer：

```bash
python -m vrl.scripts.train --config experiment/online/ocr/video_diffusion_nft
```

## 当前 checkpoint 完成标准

- `video_reward` inference 有独立 schema、artifact store、runtime protocol。
- `VideoReward` 是 thin adapter，不再混 runtime protocol 细节。
- shared Ray actor substrate + reward worker boundary 清楚；fake scorer 不作为 public runtime。
- generation launcher 已复用 shared Ray actor substrate，不再独占 generic placement / actor lifecycle helper。
- 默认训练语义仍是 synchronous scored batch，不引入 silent stale RL。
- public config 不再有 `backend` 字段，包括 `backend=stub`；fake scorer 不能作为真实验收。
- reward artifact 和 reward debug result 可复现一次 reward call。

## Full sprint 完成标准

- heavy video reward 可以作为独立 Ray worker runtime 运行，不常驻 trainer policy module 或 rollout worker。
- Ray-backed reward runtime 边界清楚，且不复用 rollout worker。
- generation 和 reward 都通过同一个 `vrl/ray/` substrate 启动 / 放置 / 释放 actors。
- `distributed.resources.reward -> Ray placement -> assigned GPU -> RewardInferenceResult` 有 trivial GPU scorer smoke 覆盖。
- reward latency 和 version 信息可追踪，后续可以安全扩展 async overlap。
- Cosmos-Predict2.5 + DiffusionNFT 可以作为 consumer 跑真实 video reward optimizer update。
- README 不能把 Cosmos-Predict2.5 DiffusionNFT 写成 validated route，除非真实 `video_reward` runtime run 通过。

## 参考路径

- `/home/mingfeiguo/Desktop/wm-infra/vrl/rewards/video_reward.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/rewards/base.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/rewards/multi.py`
- `/home/mingfeiguo/Desktop/wm-infra/configs/reward/video_reward.yaml`
- `/home/mingfeiguo/Desktop/wm-infra/configs/experiment/online/ocr/video_diffusion_nft.yaml`
