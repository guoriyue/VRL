# SPRINT: Ray Distributed Resource Configuration

## 0. Core Decision

Ray 在本 repo 里的目标是做 rollout execution / serving data plane，不是第一阶段的 trainer parallelism。

目标闭环是：

```text
trainer GPU(s):
  own optimizer/backward/checkpoint

Ray rollout GPU(s):
  generate samples/chunks
  compute rollout-side metadata when needed
  return CPU training payload through Ray object store

trainer GPU(s):
  move returned payload to trainer device
  run replay loss/backward/optimizer
  sync versioned trainable weights back to rollout workers
```

也就是说，本 sprint 要产品化的是我们讨论的 split mode：rollout 可以放到另一组 GPU 上，训练数据通过 CPU/Ray object boundary 回到训练 GPU。单 GPU colocate 只保留为 local debug 路径。

本 sprint 只支持一套 canonical schema：role-level resource allocation。

不采用 `mode + num_gpus + trainer_gpus` 作为主配置，因为它会变成第二套 shorthand，后面和 `trainer/rollout/reward/ref` 这类 role 配置冲突。`split` / `colocate` 不作为用户必须手写的 primary knob，而是从 `trainer.devices` 和 `rollout.devices` 是否重叠推导出来。

主路径不要求用户手写物理 GPU 编号。用户应该表达“trainer 要几张 GPU、rollout 要几张 GPU、每个 rollout worker 几张 GPU”。物理 GPU pinning 只作为高级 override，用于单机调试、多任务混部、或明确避开某些卡。

目标配置：

```yaml
distributed:
  backend: ray

  resources:
    visible_devices: auto

    trainer:
      num_gpus: 1

    rollout:
      num_gpus: auto
      gpus_per_worker: 1
      num_workers: auto

    allow_overlap: false
```

上面是推荐写法。`devices` 字段在内部 schema 里存在，默认值是 `auto`，但普通 recipe 不应该显式写它。

高级手动 pinning：

```yaml
distributed:
  backend: ray

  resources:
    visible_devices: [0, 1, 2, 3]

    trainer:
      devices: [0]

    rollout:
      devices: [1, 2, 3]
      gpus_per_worker: 1
      num_workers: auto

    allow_overlap: false
```

单 GPU colocate local debug：

```yaml
distributed:
  backend: ray

  resources:
    visible_devices: auto

    trainer:
      num_gpus: 1

    rollout:
      num_gpus: 1
      gpus_per_worker: 1
      num_workers: 1

    allow_overlap: true

  rollout:
    release_after_collect: true
```

## 1. Why This Shape

这个设计跟现代 RL serving/training 框架的资源抽象一致：

- 用户默认声明 role 需要多少资源，不默认写物理 GPU 编号。
- Ray placement group 负责 actor placement。
- 物理 GPU pinning 是高级 override，用于单机调试、混部、或明确避开某些卡。
- 模型配置不包含 GPU placement。`model` 只描述 checkpoint、dtype、LoRA、backend；GPU 分配属于 `distributed.resources`。

`slime` / `MILES` 的主语义是 `actor_num_gpus_per_node`、`rollout_num_gpus`、`rollout_num_gpus_per_engine`、`colocate`。它们不是让普通用户手写 `trainer=[0]`、`rollout=[1,2,3]`；物理卡选择通常交给 Ray placement group 或外层 `CUDA_VISIBLE_DEVICES`。本 repo 不直接复制它们的 CLI 形状，但保留同一个资源边界：trainer 和 rollout 是两个 role，Ray placement 负责切资源。

`slime` 的 local implementation 验证了这个方向。非 colocate 时，它把 actor/trainer GPU slot 放在前面，把 rollout GPU slot 放在 offset 后面：

```python
elif args.colocate:
    num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
    rollout_offset = 0
else:
    num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node + args.rollout_num_gpus
    rollout_offset = args.actor_num_nodes * args.actor_num_gpus_per_node
```

训练循环也是 rollout 先生成，训练 actor 再消费返回的数据引用，然后把新权重同步回 rollout：

```python
rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
actor_model.update_weights()
```

`slime` 的 async path 还明确禁止 colocate：

```python
assert not args.colocate, "Colocation is not supported for async training."
```

这个对照给本 repo 两个结论：

- split rollout/training 是正确方向，不是临时 hack。
- 真正稳定的 split 不能只靠 `ray.remote(num_gpus=...)`。`slime` 的 trainer 是 Ray actor group，所以 Ray placement group 真正知道 actor/trainer 占用了哪些 GPU；本 repo 当前 trainer 是 driver process，Ray 默认不知道 driver 已经占用 `cuda:0`。因此本 sprint 必须增加 trainer GPU reservation bundle，或者后续把 trainer 移进 RayTrainGroup。第一版选择 reservation bundle，因为它不改变 trainer 架构。

`SGLang` 自身有 `base_gpu_id` / `gpu_id_step` 这类低层 serving 参数，用来在一个 serving instance 内启动 TP/PP scheduler 进程。那不是本 repo 的 RL recipe 主入口；这里最多在 rollout runtime 里把 resolved placement 转换成 SGLang 需要的低层参数。

## 1.1 Current Code Reality

当前代码已经有数据闭环，但资源 ownership 还不够清晰：

```python
RemoteRolloutWorker = ray.remote(
    num_cpus=rollout_config.cpus_per_worker,
    num_gpus=rollout_config.gpus_per_worker,
)(RayRolloutWorker)
```

`RolloutBackendConfig` 现在还同时持有 `num_workers`、`gpus_per_worker`、`allow_driver_gpu_overlap`。这把 rollout execution 参数和 trainer/rollout GPU ownership 混在了一起。

rollout 数据回流方向是对的：Ray worker 返回 CPU payload，trainer 再把 batch move 到自己的 device。这个 contract 应保留，并在新 resolver 里显式表达。

## 1.2 Boundary: Multi-Node and Multi-Card Trainer

这个 sprint 不把 multi-node training、FSDP、Megatron 也一起做掉。

本 sprint 覆盖的是：

- single-process trainer 选择自己的 driver device。
- Ray rollout workers 按 role-level GPU budget 启动。
- trainer / rollout 是否 overlap 的 fail-fast 校验。
- single-GPU colocate local debug 和 multi-GPU split rollout。

本 sprint 不覆盖的是：

- 一个模型跨多张 GPU 训练，例如 FSDP、Tensor Parallel、Pipeline Parallel、Megatron。
- trainer actor group 的 rank/world-size/env 初始化。
- multi-node trainer 的 node/rank/device mapping。
- multi-node rollout 的 hostname/rack-aware placement、per-node GPU quota、跨节点亲和性。

如果 Ray cluster 已经是 multi-node，Ray rollout placement group 可能能把 rollout workers 放到多个节点上；但本 sprint 只要求记录和校验 Ray 实际分配的 `node_ip` / `gpu_ids`，不承诺稳定的 per-node placement 语义。也就是说：multi-node rollout 可以作为 best-effort 基础路径，multi-node training 不在本 sprint 内。

后续 FSDP / Megatron sprint 应该扩展 role schema，而不是把多卡训练塞进 `devices` list：

```yaml
distributed:
  trainer:
    strategy: fsdp        # single_process | fsdp | megatron
    num_nodes: 1
    gpus_per_node: 8

  resources:
    trainer:
      num_gpus_per_node: 8
    rollout:
      num_gpus: auto
      gpus_per_worker: 1
```

这个 future schema 才对应 `slime` / `MILES` 的 `actor_num_nodes`、`actor_num_gpus_per_node` 和 Megatron/FSDP world-size 语义。

## 2. Required Semantics

### 2.1 `visible_devices`

`visible_devices` 定义本次训练允许使用的 GPU budget。

规则：

- `auto`：读取当前进程 / Ray cluster 可见 CUDA devices；如果用户想只暴露部分物理卡，推荐用外层 `CUDA_VISIBLE_DEVICES=...`。
- `[]`：CPU-only，Ray rollout worker 必须 `gpus_per_worker=0`。
- 显式 list：所有 role 的 `devices` 必须是它的子集。
- 不在这里做 free-memory 预测。显存动态变化，只能做 static resource ownership 和 OOM retry。

### 2.2 `trainer`

`trainer` 定义 driver/training side 的 GPU ownership。

规则：

- `trainer.devices: auto` 时，默认取 `visible_devices` 的前 `trainer.num_gpus` 张。
- `trainer.num_gpus` 只在 `devices: auto` 时生效。
- 如果 `trainer.devices` 显式设置，`trainer.num_gpus` 必须为空或等于 `len(devices)`。
- 当前阶段只支持 single-process trainer，因此 `len(trainer.devices)` 必须是 `0` 或 `1`。
- 后续 FSDP / RayTrainGroup 阶段再放开 `trainer.devices` 多卡。
- 普通 recipe 应该只写 `trainer.num_gpus`，不写 `trainer.devices`。

### 2.3 `rollout`

`rollout` 定义 Ray rollout workers 的 GPU budget。

规则：

- `rollout.devices: auto` 且 `allow_overlap=false`：默认使用 `visible_devices - trainer.devices`。
- `rollout.devices: auto` 且 `allow_overlap=true`：如果没有剩余 GPU，可以复用 `trainer.devices`。
- `rollout.num_gpus: auto`：等于 `len(rollout.devices)`。
- `rollout.num_workers: auto`：等于 `rollout.num_gpus / gpus_per_worker`，必须整除。
- `gpus_per_worker` 支持 `0`、`1`，后续多 GPU per worker 再支持 `>1`。
- `rollout.devices` 为空且 `gpus_per_worker > 0` 必须 fail-fast。
- 普通 recipe 应该只写 `rollout.num_gpus`、`rollout.gpus_per_worker`、`rollout.num_workers`，不写 `rollout.devices`。

### 2.4 overlap policy

默认不允许 trainer 和 rollout overlap。

规则：

- 如果 `trainer.devices ∩ rollout.devices != ∅` 且 `allow_overlap=false`，启动时报错。
- 如果 overlap 被允许，必须同时满足 `distributed.rollout.release_after_collect=true`，否则单 GPU/混部路径会保留两份模型，OOM 风险不可控。
- overlap 只用于 local debug，不作为 throughput path。

### 2.5 Ray Reservation Policy

当前 trainer 不是 Ray actor，而是 driver process。Ray scheduler 不会自动知道 driver 正在使用哪张 GPU，所以只给 rollout worker 设置 `num_gpus=1` 不足以保证 worker 不落到 trainer GPU 上。

本 sprint 的非重叠 Ray split path 必须满足：

- Ray placement group 覆盖 trainer reservation bundle 和 rollout worker bundle。
- placement creation 先 probe / sort all bundles，建立 actual GPU id -> bundle mapping。
- trainer reservation bundle 由轻量 no-op Ray actor 占住，只用于资源 reservation 和实际 GPU id discovery。
- driver trainer 使用与 `resolved.trainer_devices` 对应的 reservation bundle。
- rollout workers 只调度到与 `resolved.rollout_devices` 对应的 rollout bundle。
- 启动日志同时打印 resolved logical plan 和 Ray actual placement。
- Ray actual worker `gpu_ids` 与 resolved rollout devices 不一致时 fail-fast。

这相当于在不引入 RayTrainGroup 的前提下，复制 `slime` 的同一个 placement group / offset 语义：

```text
placement group:
  resolved trainer devices      -> trainer reservation bundles
  resolved rollout devices      -> rollout worker bundles
```

后续 RayTrainGroup sprint 可以删除 no-op reservation actor，让 trainer actor 自己占用这些 bundle。

colocate path 不启动 full-GPU trainer reservation actor。原因是单 GPU colocate 下，reservation actor 和 rollout worker 如果都向 Ray 申请 `num_gpus=1`，Ray scheduler 会认为同一张 GPU 被重复占用。colocate 的正确表达是 `allow_overlap=true` + `release_after_collect=true`：rollout worker 可以占用同一张 GPU，但必须在 trainer replay/backward 前释放 runtime。

### 2.6 Data Transport and Weight Sync Contract

rollout worker 到 trainer 的数据边界必须是 CPU/Ray object boundary。

规则：

- rollout worker 不返回 GPU tensor。
- rollout payload 只包含训练真正需要的 tensors / arrays / metadata。
- visual model 不应该默认传完整 decoded image/video，如果 loss 只需要 tokens、latents、logprobs、rewards、masks，就只传这些。
- trainer 侧负责把 batch move 到 `trainer_torch_device(resolved, actual_trainer_devices=...)` 返回的 device。
- weight sync 必须是 versioned trainable state，不把 FSDP/Megatron shard 或 wrapped module key 暴露给 rollout worker。
- `sync_trainable_state=lora_only` 是当前推荐路径；full model sync 以后必须单独评估吞吐和 object store 压力。
- async rollout 在 update weights 前必须等待当前 generation 完成，避免同一个 rollout batch 混用多个 policy version。

## 3. Target Internal Types

新增文件：

```text
vrl/distributed/resources.py
```

目标 dataclass：

```python
@dataclass(frozen=True, slots=True)
class RoleResourceConfig:
    num_gpus: int | str | None = "auto"
    devices: list[int] | str = "auto"


@dataclass(frozen=True, slots=True)
class RolloutResourceConfig(RoleResourceConfig):
    gpus_per_worker: float = 1.0
    num_workers: int | str = "auto"


@dataclass(frozen=True, slots=True)
class DistributedResourceConfig:
    visible_devices: list[int] | str = "auto"
    trainer: RoleResourceConfig = field(default_factory=RoleResourceConfig)
    rollout: RolloutResourceConfig = field(default_factory=RolloutResourceConfig)
    allow_overlap: bool = False


@dataclass(frozen=True, slots=True)
class ResolvedDistributedResources:
    visible_devices: tuple[int, ...]
    trainer_devices: tuple[int, ...]
    rollout_devices: tuple[int, ...]
    rollout_num_gpus: int
    rollout_num_workers: int
    rollout_gpus_per_worker: float
    total_gpu_slots: int
    ray_total_bundles: int
    requires_trainer_reservation: bool
    colocated: bool
```

核心函数：

```python
def resolve_distributed_resources(cfg: Any) -> ResolvedDistributedResources:
    ...
```

`resolve_distributed_resources()` 是唯一入口。不要让 train script、Ray launcher、runtime backend 各自解析 GPU config。

## 4. Code Changes

### Phase 1: Config Schema

修改：

```text
configs/base/distributed/ray_rollout.yaml
configs/base/distributed/ray_rollout_single_gpu.yaml
vrl/rollouts/runtime/config.py
```

目标：

- `ray_rollout.yaml` 使用 `distributed.resources`，默认多 GPU split。
- `ray_rollout_single_gpu.yaml` 使用 explicit overlap，并设置 `release_after_collect=true`。
- 两个 preset 都不应该显式写 `devices`。多 GPU split 由 `trainer.num_gpus` / `rollout.num_gpus` 自动 resolve；单 GPU local debug 由 `allow_overlap=true` + `release_after_collect=true` 表达。
- `RolloutBackendConfig` 继续保留 rollout execution 参数，但不再负责 trainer/rollout device allocation。

`RolloutBackendConfig` 应保留：

```yaml
distributed:
  rollout:
    cpus_per_worker: 4.0
    placement_strategy: SPREAD
    max_inflight_chunks_per_worker: 1
    sync_trainable_state: lora_only
    release_after_collect: false
```

`RolloutBackendConfig` 不应继续拥有：

```yaml
num_workers
gpus_per_worker
allow_driver_gpu_overlap
```

这些字段由 `ResolvedDistributedResources` 统一提供。

### Phase 2: Resource Resolver

新增：

```text
vrl/distributed/resources.py
tests/distributed/test_resources.py
```

测试覆盖：

- 4 visible GPUs, trainer auto 1, rollout auto -> trainer `(0,)`, rollout `(1,2,3)`, workers `3`。
- explicit trainer `[0]`, rollout `[1,2,3]` -> no overlap。
- explicit overlap `[0]` / `[0]` with `allow_overlap=false` -> fail。
- explicit overlap `[0]` / `[0]` with `allow_overlap=true` -> colocated true。
- explicit devices not subset of visible -> fail。
- `num_workers: auto` with non-divisible `num_gpus / gpus_per_worker` -> fail。
- single GPU auto split with `allow_overlap=false` -> fail with clear message。
- single GPU overlap with `release_after_collect=false` -> fail at backend validation。
- resolved plan exposes `total_gpu_slots = len(trainer_devices ∪ rollout_devices)`。
- split path exposes `ray_total_bundles = len(trainer_devices) + rollout_num_workers`。
- colocate path exposes `ray_total_bundles = rollout_num_workers` because trainer reservation is intentionally disabled。

### Phase 3: Ray Placement Integration

修改：

```text
vrl/distributed/ray/placement/group.py
vrl/distributed/ray/rollout/launcher.py
vrl/distributed/ray/rollout/types.py
```

目标：

- `RayRolloutLauncher.launch()` 接收 `ResolvedDistributedResources` 或从 cfg 内部 resolve。
- 非重叠 split 下，placement group bundle 数量来自 trainer reservation slots + `resolved.rollout_num_workers`。
- colocate 下，placement group bundle 数量只来自 `resolved.rollout_num_workers`。
- 非重叠 split 下，trainer reservation actor 占住 trainer bundle，避免 rollout worker 被 Ray 调度到 trainer GPU。
- rollout actor `num_gpus` 来自 `resolved.rollout_gpus_per_worker`。
- worker metadata 记录 assigned Ray GPU IDs，并和 resolved rollout devices 做一致性检查。
- 非重叠 split 下，trainer device helper 使用 reservation actor 发现的 actual trainer GPU，而不是提前假设 physical GPU 0。
- colocate 下，trainer device helper 使用 resolved trainer device，并依赖 `release_after_collect=true` 控制显存生命周期。

注意：

- Ray 不能直接保证“物理 GPU1-3”除非它的 cluster 可见资源、placement group 和 reservation actor 对齐。
- 当前 trainer 是 driver process；没有 trainer reservation actor 时，Ray 可能把 rollout worker 放到 driver 正在使用的 GPU 上。
- 第一版必须在日志里打印 resolved plan 和 Ray 实际 assigned GPU IDs。
- 如果 Ray 返回的 GPU IDs 不在 `rollout_devices` 内，启动失败，不 silent fallback。

### Phase 4: Driver CUDA Ownership Validation

修改：

```text
vrl/rollouts/runtime/backend.py
vrl/scripts/sd3_5/train.py
vrl/scripts/wan_2_1/train.py
vrl/scripts/cosmos/train.py
vrl/scripts/janus_pro/train.py
vrl/scripts/nextstep_1/train.py
vrl/scripts/wan_2_1/train_dpo.py
```

目标：

- train script 不再直接写 `torch.device("cuda" if torch.cuda.is_available() else "cpu")`。
- train script 调用 `resolve_distributed_resources(cfg)`。
- direct/no-Ray path 可以把 `trainer_devices[0]` 转成 driver device。
- Ray rollout path 必须在 placement group / trainer reservation 完成后使用 actual trainer GPU 作为 driver device。
- backend validation 从“driver 是否在 CUDA”升级成“driver CUDA device 是否和 rollout devices overlap”。
- overlap 时必须检查 `allow_overlap=true` 和 `release_after_collect=true`。

错误信息必须具体：

```text
Trainer device cuda:0 overlaps rollout devices [0], but resources.allow_overlap=false.
Use CUDA_VISIBLE_DEVICES=0,1,2,3 with auto split for throughput, or set allow_overlap=true with rollout.release_after_collect=true for single-GPU local debug.
```

### Phase 5: README and Examples

修改：

```text
README.md
configs/base/distributed/ray_rollout.yaml
configs/base/distributed/ray_rollout_single_gpu.yaml
```

README 需要新增两个例子：

自动 split：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vrl.scripts.train \
  --config experiment/sd3_5_ocr_grpo \
  distributed.resources.trainer.num_gpus=1 \
  distributed.resources.rollout.num_gpus=auto
```

单 GPU colocate local debug：

```bash
CUDA_VISIBLE_DEVICES=0 python -m vrl.scripts.train \
  --config experiment/sd3_5_ocr_grpo \
  /base/distributed=ray_rollout_single_gpu
```

显式 pinning，高级调试才用：

```bash
python -m vrl.scripts.train \
  --config experiment/sd3_5_ocr_grpo \
  distributed.resources.visible_devices='[0,1,2,3]' \
  distributed.resources.trainer.devices='[0]' \
  distributed.resources.rollout.devices='[1,2,3]'
```

文档必须明确：

- 默认推荐 auto split。
- manual pinning 是高级选项。
- 单 GPU 推荐用 `CUDA_VISIBLE_DEVICES=...` 限制外层可见卡，而不是在 recipe 里写物理 GPU 编号。
- 单 GPU colocate 只用于 local debug。
- throughput path 是 trainer GPU(s) 和 rollout GPU(s) 分离。

## 4.1 File-by-File Implementation Map

这个 sprint 的实现不要按“看见哪里有 GPU 就改哪里”的方式推进。必须先建立 resolver，然后把所有调用点改成只消费 resolver 的结果。

### 4.1.1 Core Resolver

新增：

```text
vrl/distributed/resources.py
```

预期内容：

- 定义 `RoleResourceConfig`、`RolloutResourceConfig`、`DistributedResourceConfig`、`ResolvedDistributedResources`。
- 实现 `resolve_distributed_resources(cfg)`。
- 实现 trainer device helper，例如 `trainer_torch_device(resolved, actual_trainer_devices=None)`。
- 实现 readable plan formatter，例如 `format_distributed_resource_plan(resolved, actual_placement=None)`。
- 解析 `visible_devices=auto` 时读取当前可见 CUDA devices；如果无 CUDA，则支持 CPU-only。
- 校验 explicit devices 必须是 `visible_devices` 子集。
- 校验当前阶段 trainer 只能是 `0` 或 `1` 张 GPU。
- 校验 `rollout.num_workers=auto` 时由 `rollout.num_gpus / gpus_per_worker` 推导，且必须整除。
- 校验 single-GPU no-overlap split 必须 fail-fast。
- 计算 `total_gpu_slots` 和 `ray_total_bundles`，供 Ray placement group 一次性 plan trainer reservation + rollout。
- 记录 resolved plan 的可读字符串，供 train script / Ray launcher logging 使用。

### 4.1.2 Base Configs

修改：

```text
configs/base/distributed/ray_rollout.yaml
configs/base/distributed/ray_rollout_single_gpu.yaml
```

`ray_rollout.yaml` 预期改成多 GPU split 的主路径：

```yaml
distributed:
  resources:
    visible_devices: auto
    trainer:
      num_gpus: 1
    rollout:
      num_gpus: auto
      gpus_per_worker: 1
      num_workers: auto
    allow_overlap: false

  rollout:
    cpus_per_worker: 4.0
    placement_strategy: SPREAD
    max_inflight_chunks_per_worker: 1
    sync_trainable_state: lora_only
    release_after_collect: false
```

`ray_rollout_single_gpu.yaml` 预期改成 colocate local debug：

```yaml
distributed:
  resources:
    visible_devices: auto
    trainer:
      num_gpus: 1
    rollout:
      num_gpus: 1
      gpus_per_worker: 1
      num_workers: 1
    allow_overlap: true

  rollout:
    cpus_per_worker: 4.0
    placement_strategy: STRICT_PACK
    max_inflight_chunks_per_worker: 1
    sync_trainable_state: lora_only
    release_after_collect: true
```

两个 base config 都不应该手写 `trainer.devices` / `rollout.devices`。manual pinning 只能作为用户 override。

### 4.1.3 Rollout Backend Config

修改：

```text
vrl/rollouts/runtime/config.py
```

预期改动：

- `RolloutBackendConfig` 删除 resource ownership 字段：
  - `num_workers`
  - `gpus_per_worker`
  - `allow_driver_gpu_overlap`
- `RolloutBackendConfig` 只保留 rollout execution 字段：
  - `backend`
  - `cpus_per_worker`
  - `placement_strategy`
  - `max_inflight_chunks_per_worker`
  - `sync_trainable_state`
  - `release_after_collect`
- `from_cfg()` 不再从 `distributed.rollout` 解析 worker/GPU 数。
- `to_dict()` 不再输出 worker/GPU ownership 字段。

### 4.1.4 Ray Placement and Launcher

修改：

```text
vrl/distributed/ray/placement/group.py
vrl/distributed/ray/rollout/launcher.py
vrl/distributed/ray/rollout/types.py
```

`placement/group.py` 预期改动：

- `create_rollout_placement_group()` 接收 `RolloutBackendConfig` 和 `ResolvedDistributedResources`。
- placement group bundle 在非重叠 split 下分为 trainer reservation bundles 和 rollout worker bundles。
- 非重叠 split 的 trainer reservation bundle 数来自 `len(resolved.trainer_devices)`。
- colocate 的 trainer reservation bundle 数是 `0`。
- rollout bundle 数来自 `resolved.rollout_num_workers`。
- rollout bundle GPU 数来自 `resolved.rollout_gpus_per_worker`。
- CPU 数继续来自 `RolloutBackendConfig.cpus_per_worker`。
- 新增轻量 `InfoActor`，先 probe 每个 placement bundle 的 actual `node_ip` / `gpu_ids`，按 node/GPU id 稳定排序，建立 actual GPU id 到 bundle 的映射。
- auto split 使用排序后的前 `trainer_slots` 个 bundle 做 trainer reservation，后面的 bundle 做 rollout；manual pinning 使用 `resolved.trainer_devices` / `resolved.rollout_devices` 精确选择 bundle。
- colocate 不创建 trainer reservation actor；只校验 `allow_overlap=true` 和 `release_after_collect=true`。
- 新增轻量 `TrainerReservationActor`，用于占住 trainer GPU 并返回 actual `node_ip` / `gpu_ids`。
- placement metadata 必须保留 logical bundle index、actual bundle index、node ip、gpu ids。

`rollout/launcher.py` 预期改动：

- `RayRolloutLauncher.launch()` 接收 `resolved_resources`，或接收完整 cfg 并内部调用 resolver。推荐显式传入 `ResolvedDistributedResources`，避免隐式重复解析。
- Ray actor `num_gpus` 来自 `resolved.rollout_gpus_per_worker`。
- worker 数来自 placement 的 rollout bundle 数，而不是 rollout config。
- 非重叠 split 下，driver trainer device 使用 trainer reservation metadata；不能在 Ray placement 完成前初始化 policy/model 到 CUDA。
- colocate 下没有 trainer reservation metadata，driver trainer device 使用 resolved trainer device。
- Ray placement 需要支持 two-phase usage：先创建 placement/reservation 以确定 trainer device，再初始化 trainer model，最后把已有 placement 传给 launcher 启动 rollout workers。
- 启动后日志必须打印：
  - resolved trainer devices
  - resolved rollout devices
  - Ray actual trainer reservation `node_ip` when reservation exists
  - Ray actual trainer reservation `gpu_ids` when reservation exists
  - Ray 实际 worker `node_ip`
  - Ray 实际 worker `gpu_ids`
- 如果 Ray 返回的 GPU id 不在 `resolved.rollout_devices` 内，第一版必须 fail-fast。
- 如果 trainer reservation `gpu_ids` 与 resolved trainer devices 不一致，第一版必须 fail-fast。

two-phase usage 目标形状：

```python
resolved = resolve_distributed_resources(cfg)
placement = create_distributed_placement(backend_cfg, resolved)
device = trainer_torch_device(resolved, actual_trainer_devices=placement.trainer_gpu_ids)

# Initialize trainer policy/model on device here.

runtime = RayRolloutLauncher().launch(
    backend_cfg,
    runtime_spec,
    gatherer,
    resolved_resources=resolved,
    placement=placement,
)
```

`rollout/types.py` 预期改动：

- `RayWorkerHandle` 保留 `node_id` / `gpu_ids`。
- 如有必要，增加 resolved/actual metadata 字段，方便诊断实际 placement。

### 4.1.5 Runtime Backend Validation

修改：

```text
vrl/rollouts/runtime/backend.py
```

预期改动：

- backend validation 不再判断 “driver policy 是否在 CUDA” 这个粗粒度条件。
- 改成判断 `resolved.trainer_devices ∩ resolved.rollout_devices`。
- overlap 且 `resources.allow_overlap=false` 时 fail-fast。
- overlap 且 `distributed.rollout.release_after_collect=false` 时 fail-fast。
- single-GPU colocate 的错误信息必须提示：
  - throughput path 用 auto split / 多可见 GPU。
  - debug path 用 `allow_overlap=true` + `release_after_collect=true`。

### 4.1.6 Runtime Inputs

修改：

```text
vrl/rollouts/runtime/launch_inputs.py
```

预期改动：

- 不再用旧 `RolloutBackendConfig.gpus_per_worker` 判断 rollout worker device。
- 改用 `resolved.rollout_gpus_per_worker`：

```python
rollout_device = "cuda" if resolved.rollout_gpus_per_worker > 0 else "cpu"
```

- `build_rollout_runtime_inputs()` 接收 `ResolvedDistributedResources`，或在上层传入 `rollout_device`，避免自己解析资源。

### 4.1.7 Training Scripts

修改：

```text
vrl/scripts/sd3_5/train.py
vrl/scripts/wan_2_1/train.py
vrl/scripts/cosmos/train.py
vrl/scripts/janus_pro/train.py
vrl/scripts/nextstep_1/train.py
vrl/scripts/wan_2_1/train_dpo.py
```

预期改动：

- 删除直接猜 device 的逻辑：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

- 改成：

```python
resolved_resources = resolve_distributed_resources(cfg)
device = trainer_torch_device(resolved_resources, actual_trainer_devices=placement.trainer_gpu_ids)
```

- 对非重叠 Ray split backend，必须先 resolve resources、创建 placement/reservation、确认 actual trainer GPU，再把 model/policy 初始化到 CUDA。
- 对 Ray colocate backend，没有 trainer reservation actor，但仍必须先 validate overlap policy，再把 model/policy 初始化到 CUDA。
- 对 direct/no-Ray backend，可以没有 `placement`，此时 `trainer_torch_device(resolved_resources)` 直接使用 resolved trainer device。
- GRPO train scripts 把 `resolved_resources` 传给：
  - `build_rollout_runtime_inputs()`
  - `build_rollout_backend_from_cfg()`
  - Ray launcher path
- DPO train script 虽然没有 rollout，也必须用同一个 `trainer_torch_device()` 选择 trainer device。
- 每个 train script 启动时记录 resolved plan。

### 4.1.8 Tests

新增：

```text
tests/distributed/test_resources.py
```

覆盖：

- 4 GPU auto split -> trainer `(0,)`, rollout `(1,2,3)`, workers `3`。
- explicit trainer `[0]`, rollout `[1,2,3]` -> no overlap。
- explicit overlap + `allow_overlap=false` -> fail。
- explicit overlap + `allow_overlap=true` -> colocated true。
- explicit devices not subset of visible -> fail。
- `num_workers=auto` but `num_gpus / gpus_per_worker` not divisible -> fail。
- single GPU auto split + `allow_overlap=false` -> fail。
- overlap + `release_after_collect=false` -> backend validation fail。

更新：

```text
tests/distributed/ray/test_placement_group.py
tests/distributed/ray/test_rollout_launcher.py
tests/distributed/ray/test_real_ray_rollout_validation.py
tests/rollouts/test_runtime_inputs.py
tests/engine/generation/test_runtime_factory.py
tests/config/test_load_all_experiments.py
```

预期改动：

- 测试不再通过 `distributed.rollout.num_workers` / `distributed.rollout.gpus_per_worker` 表达资源 ownership。
- Ray launcher 测试构造 `ResolvedDistributedResources`。
- placement group 测试覆盖 trainer reservation + rollout bundle offset。
- launcher 测试断言 rollout worker actual GPU 不和 trainer reservation actual GPU overlap。
- runtime input 测试检查 rollout device 来自 resolved resources。
- config tests 检查 active experiment 能加载新 `distributed.resources` schema。

### 4.1.9 README

修改：

```text
README.md
```

预期新增：

- auto split 例子：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vrl.scripts.train \
  --config experiment/sd3_5_ocr_grpo \
  distributed.resources.trainer.num_gpus=1 \
  distributed.resources.rollout.num_gpus=auto
```

- single-GPU colocate local debug 例子：

```bash
CUDA_VISIBLE_DEVICES=0 python -m vrl.scripts.train \
  --config experiment/sd3_5_ocr_grpo \
  /base/distributed=ray_rollout_single_gpu
```

- 文档说明：
  - 默认推荐 auto split。
  - manual `devices` pinning 是高级调试选项。
  - 单 GPU colocate 只用于 local debug。
  - throughput path 是 trainer GPU 和 rollout GPU 分离。

### 4.1.10 Implementation Order

推荐按这个顺序做：

1. 新增 `vrl/distributed/resources.py` 和 `tests/distributed/test_resources.py`。
2. 修改两个 distributed base config。
3. 收窄 `RolloutBackendConfig`。
4. 修改 Ray placement / launcher，先实现 trainer reservation + rollout bundle offset。
5. 修改 train scripts 的启动顺序：resolve resources -> validate overlap -> create Ray placement/reservation when split -> choose trainer device -> 初始化 model 到 CUDA。
6. 修改 backend validation / runtime inputs。
7. 更新 tests 和 README。

## 5. Non-Goals

本 sprint 不做：

- FSDP trainer 多卡。
- RayTrainGroup 接入主训练入口。
- full async pipeline training/rollout overlap。第一版只要求 sync split；async overlap 要等 policy version / stale rollout 策略稳定后再做。
- reward model serving。
- multi-node hostname / rack-aware placement。
- 基于实时 free memory 的动态 GPU 选择。
- SGLang server 复用。
- per-token AR scheduling。

这些都依赖更高层的 distributed training / serving 设计，不应该塞进本 sprint。

## 6. Acceptance Criteria

代码层：

- 所有训练脚本都通过同一个 `resolve_distributed_resources()` 选择 trainer device。
- `RolloutBackendConfig` 不再解析 `num_workers`、`gpus_per_worker`、`allow_driver_gpu_overlap`。
- Ray launcher 不再从 rollout config 猜 worker 数，而是使用 resolved resources。
- 非重叠 split 下，Ray placement group reserve trainer GPU，即使 trainer 仍是 driver process。
- 非重叠 split 下，Ray rollout worker actual GPU 不和 trainer reservation actual GPU overlap。
- colocate 下，actual overlap 只允许在 `allow_overlap=true` 且 `release_after_collect=true` 时发生。
- rollout worker 返回 CPU payload；trainer 侧负责 move 到 trainer device。
- rollout weight sync 只发送 versioned trainable state。
- driver / rollout device overlap 有明确 fail-fast。
- 单 GPU local debug 必须显式 `allow_overlap=true` 和 `release_after_collect=true`。

测试层：

```bash
python -m pytest -q tests/distributed/test_resources.py
python -m pytest -q tests/distributed/ray
python -m pytest -q tests/config/test_load_all_experiments.py
python -m pytest -q tests/scripts
```

文档层：

- README 有 auto split 和 manual pinning 示例。
- `configs/base/distributed/ray_rollout.yaml` 表达多 GPU split。
- `configs/base/distributed/ray_rollout_single_gpu.yaml` 表达 colocate local debug。

真实 checkpoint DoD：

- 在至少一个 diffusion family 上用真实 checkpoint 跑一次 single-GPU colocate validation run。
- 在至少一个 family 上用 4 visible GPUs 跑一次 resolved plan validation run，确认日志显示 trainer `cuda:0`，rollout workers 分配到 `1,2,3`。
- 4 GPU validation run 必须确认 Ray actual placement 里 trainer reservation actor 在 trainer GPU，rollout workers 不在 trainer GPU。
- 如果没有 4 GPU 环境，必须保留 skip 标记和清晰的手动命令，不把 fake Ray test 当成真实 DoD。

## 7. Expected Final Behavior

4 GPU 默认行为：

```text
visible_devices = [0, 1, 2, 3]
trainer.num_gpus = 1
rollout.num_gpus = auto

resolved:
  trainer_devices = [0]
  rollout_devices = [1, 2, 3]
  rollout_num_gpus = 3
  rollout_num_workers = 3
  total_gpu_slots = 4
  ray_total_bundles = 4
  colocated = false

ray placement:
  reserve trainer bundle on GPU 0
  launch rollout workers on GPUs 1, 2, 3
  return rollout payload through CPU/Ray object store
```

单 GPU local debug：

```text
visible_devices = [0]
trainer.num_gpus = 1
rollout.num_gpus = 1
allow_overlap = true
release_after_collect = true

resolved:
  trainer_devices = [0]
  rollout_devices = [0]
  rollout_num_gpus = 1
  rollout_num_workers = 1
  total_gpu_slots = 1
  ray_total_bundles = 1
  colocated = true

ray placement:
  no full-GPU trainer reservation actor
  rollout worker intentionally overlaps trainer device
  release rollout runtime after collect before replay/backward
```

错误配置：

```yaml
distributed:
  resources:
    visible_devices: [0]
    trainer:
      devices: [0]
    rollout:
      devices: [0]
    allow_overlap: false
```

必须失败：

```text
Trainer devices [0] overlap rollout devices [0], but resources.allow_overlap=false.
```

## 8. References

Local references:

- `/home/mingfeiguo/Desktop/wm-infra/vrl/rollouts/runtime/config.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/rollouts/runtime/backend.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/distributed/ray/placement/group.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/distributed/ray/rollout/launcher.py`
- `/home/mingfeiguo/Desktop/wm-infra/configs/base/distributed/ray_rollout.yaml`
- `/home/mingfeiguo/Desktop/wm-infra/configs/base/distributed/ray_rollout_single_gpu.yaml`

Architecture references:

- `/home/mingfeiguo/Desktop/slime/train.py`
- `/home/mingfeiguo/Desktop/slime/train_async.py`
- `/home/mingfeiguo/Desktop/slime/slime/utils/arguments.py`
- `/home/mingfeiguo/Desktop/slime/slime/ray/placement_group.py`
- `/home/mingfeiguo/Desktop/slime/slime/ray/rollout.py`
- `/home/mingfeiguo/Desktop/slime/slime/backends/megatron_utils/actor.py`
