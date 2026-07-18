# SPRINT: Ray 阶段生命周期整理

状态：**DONE — P0–P3 已提交 `e4864bf`「Implement Ray phase lifecycle plan (P0-P3)」**（wm-infra 与 `~/Desktop/VRL` 两个 checkout 同步在 `main`，blob 一致）。验收 §10 八条达成；P4（public YAML 兼容，刻意暂不动）/ weight-sync / sleeping 是 §3 与「後續」明确划出的**独立后续 sprint**，不属本 sprint 的未完工作。本 doc 移入 `done/`。

> **关账复核（2026-06-16）**：旧 status 行「未提交,待 review」已 **stale**——代码非但没烂，早已落库。实测：`should_release_memory_before_reward` / `_ReleaseAfterCollectState` 已从 `vrl/` 消失；`RayLifecyclePlan`(`vrl/ray/resources.py:97`) / `ActorLeasePolicy`(:70) / `PhaseHandoffPolicy`(:83) 在位，旧 `rollout_release_* / reward_release_*` flag 作为兼容视图同处派生（:103-105 docstring）。测试：`tests/ray/test_resources.py` 44 passed、`tests/rollouts/collector/test_runtime.py` + `tests/rollouts/test_runtime_protocol_contract.py` 19 passed。

> **实现状态(2026-06-16)**:P0–P3 全部落地,808 个非环境缺失测试全绿(`.venv` 缺 peft/transformers/vllm/ray
> 的用例本就 skip/fail,与本改动无关),ruff 全过。落点:
> - **P1**:`vrl/ray/resources.py` 新增 `RayLifecyclePlan` / `ActorLeasePolicy` / `PhaseHandoffPolicy`,由
>   `resolve_distributed_resources` 一次性从拓扑派生,挂在 `ResolvedDistributedResources.lifecycle`;旧三个 flag
>   作为兼容视图保留(同一处派生,不分叉)。`format_distributed_resource_plan` 增加 `lifecycle=.../handoff=...` 行(验收 #8)。
> - **P0**:`tests/ray/test_resources.py` 增 5 个计划派生锁测(disjoint→resident、shared→on_demand、colocated、
>   persistent、formatter)。
> - **P2**:`RayGenerationRuntime.release_memory → release`、`_ReleaseAfterCollectState → _RuntimeLease`;
>   `RayActorMethodRuntime` 增 `release()`(与 generation runtime 同词汇)。`shutdown()` 仍是 final teardown。
> - **P3**:从 `GenerationRuntime` protocol 和 runtime 删除 `should_release_memory_before_reward`;collector 改读
>   自己的 `lifecycle.handoff.release_rollout_before_reward`(经 `build_collector_from_cfg` 从 cfg 派生注入,不再问
>   runtime);launcher 按 `plan.rollout.mode` 选 resident/on_demand;reward kwargs 的 `release_after_score` 改由
>   `plan.reward.mode` 派生。
>
> **三处刻意偏离(都有理由,见 §5/§7 对照)**:
> 1. **不另起 `lifecycle.enter(phase)` 协调器对象**(§6 的"目标形状")。改成 collector/launcher/reward 各自读
>    plan——plan 已是唯一权威,达成 #4/#5,且避免一个单调用者协调器 + 整条编排链改写。
> 2. **不加投机性的 public `acquire()`**(会是 dead code;懒加载的 `_ensure_*` 本就是 acquire)。只加 `release()`
>    做跨家族词汇一致。
> 3. **`is_colocated()` 留在 runtime**:它驱动的是 driver model 的 CPU offload,与 actor lease 是两条轴,且不属于
>    `release_after_*`;§5 的 ActorLeasePolicy/PhaseHandoffPolicy 也不含 colocation。
>
> 命名:`_ReleaseAfterCollectState` 只重命名了**类**,没动属性 `_release_after_collect`——因为公有工厂
> `with_release_after_collect` 含子串 `_release_after_collect`,盲改会破坏 API。注意仓库里 `lifecycle` 已被占两次
> (`vrl/ray/lifecycle.py` 的 kill/remove 工具、`vrl/rollouts/orchestration/lifecycle.py` 的 `RolloutLifecycle`),
> 本 sprint 的 `RayLifecyclePlan` 是 `vrl/ray/resources.py` 里的独立类型,无标识符冲突。

## 原始设计

状态：提案 / 设计。

## 0. 结论

当前 `release_after_collect` / `release_before_reward_model` / `release_after_score`
不是三种业务能力，而是同一个资源问题散落在三个入口：

```text
某个 phase 要使用 GPU 时，哪些 Ray actor 必须先让位？
```

本 sprint 的目标是把这个问题收成一个内部 `RayLifecyclePlan`：

```text
topology + execution mode
  -> RayLifecyclePlan
  -> actor lease acquire/release
```

runtime 不再各自解释 release flag。runtime 只提供统一动作：

```text
acquire actor
release actor
shutdown actor
```

phase lifecycle plan 决定什么时候调用。

## 1. 正确的 no-release 规则

多 GPU 可以 no release，但条件不是“有多张 GPU”，而是 **role GPU ownership disjoint**。

```text
trainer=[0], rollout=[1], reward=[]
  -> trainer / rollout 不共卡
  -> rollout resident
  -> no release_after_collect

trainer=[0], rollout=[1], reward=[2]
  -> trainer / rollout / reward 都独占
  -> rollout resident，reward resident
  -> no release

trainer=[0], rollout=[1], reward=[1]
  -> trainer / rollout 不共卡
  -> reward / rollout 共卡
  -> reward_score 前 rollout 必须让位

trainer=[0], rollout=[0], reward=[]
  -> trainer / rollout 共卡
  -> 默认 release-after-collect
  -> tiny debug 才能显式 persistent_colocated_workers + gpu_memory_fraction
```

所以准确规则是：

```text
GPU disjoint -> resident, no release
GPU shared -> phase handoff
GPU shared but tiny debug explicitly opts into resident -> require gpu_memory_fraction
```

`persistent_colocated_workers` 不是 resident 的唯一来源。普通 split-GPU rollout 本来就是
resident；`persistent_colocated_workers` 只是“共卡仍然常驻”的 debug override。

## 2. 当前代码事实

资源层现在已经能从 topology 派生 release flags：

```python
rollout_release_after_collect = _derived_release_flag(
    config.rollout_release_after_collect,
    derived=(colocated or reward_shared_with_rollout)
    and not config.rollout_persistent_colocated_workers,
)
rollout_release_before_reward_model = _derived_release_flag(
    config.rollout_release_before_reward_model,
    derived=reward_shared_with_rollout,
)
reward_release_after_score = _derived_release_flag(
    config.reward_release_after_score,
    derived=reward_shared_with_rollout
    or (ray_reward_count > 1 and reward_gpus_per_worker > 0),
)
```

但执行分散在几个地方：

```text
vrl/ray/resources.py
  topology 派生 release_after_collect / release_before_reward_model / release_after_score

vrl/generation/ray/launcher.py
  根据 release_after_collect 选 resident vs release-after-collect runtime

vrl/rollouts/collector/core.py
  问 runtime.should_release_memory_before_reward()

vrl/ray/runtime.py
  reward 的 release_after_call 在 actor-method runtime 内部执行
```

这会制造两个问题：

1. 同一个概念有多个名字：`release_after_collect`、`release_before_reward_model`、
   `release_after_score`、`release_after_call`。
2. runtime 看起来在做 policy decision，但它本来应该只是 actor transport boundary。

## 3. 范围边界

本 sprint 只做 lifecycle cleanup，不做 weight-sync data plane 重写。

明确不并入本 sprint：

```text
CPU state-dict -> CUDA IPC / NCCL weight transport
sleep / wake / offload engine
trainer Ray actor 化
独立 reward HTTP service
```

原因：`WeightTransport` 是权重同步数据面，不是 phase lifecycle。它会牵涉 trainer
strategy、FSDP、LoRA state、Ray object store、NCCL communicator，blast radius 比本
sprint 大很多。应该拆成单独的 weight-sync topology sprint。

sleeping/offload 也不进入 MVP。slime/vLLM 的 sleep/wake 值得参考，但 cosmos-rl 在扩散
场景里选择关闭 vLLM sleep，走常驻 + 静态显存限额。这说明 sleeping 不是无条件目标。
本 sprint 可以为它保留扩展点，但不能把它当 acceptance。

## 4. 对齐 slime / cosmos-rl 的抽象

slime / cosmos-rl 值得学的不是某个 flag 名，而是角色和 phase 的分离：

```text
roles:
  trainer / policy
  rollout / generation
  reward

execution mode:
  colocated synchronous
  disaggregated asynchronous

handoff:
  只有角色共享 GPU 时，phase 之间才需要让位
```

VRL 应该用同一套语言表达：

```text
resident role pool
on-demand role lease
phase handoff
```

而不是让 generation runtime / reward runtime 各自解释 release flag。

## 5. 目标内部模型

新增内部结构，不急着改 public YAML：

```python
@dataclass(frozen=True, slots=True)
class RayLifecyclePlan:
    rollout: ActorLeasePolicy
    reward: ActorLeasePolicy
    handoff: PhaseHandoffPolicy


@dataclass(frozen=True, slots=True)
class ActorLeasePolicy:
    # resident  = actor stays alive across phases
    # on_demand = actor is released at handoff and reacquired next use
    mode: Literal["resident", "on_demand"]


@dataclass(frozen=True, slots=True)
class PhaseHandoffPolicy:
    release_rollout_before_train: bool
    release_rollout_before_reward: bool
    release_reward_after_score: bool
```

旧 flag 映射：

```text
rollout.release_after_collect=true
  -> rollout.mode = on_demand
  -> 若 trainer/rollout 共卡或显式要求 train 前释放,则 handoff.release_rollout_before_train=true

rollout.release_before_reward_model=true
  -> handoff.release_rollout_before_reward

reward.release_after_score=true
  -> handoff.release_reward_after_score
  -> reward.mode = on_demand

disjoint rollout GPU
  -> rollout.mode = resident

persistent_colocated_workers=true
  -> rollout.mode = resident
  -> only valid when trainer/rollout share GPU and gpu_memory_fraction is set
```

Runtime 只读 lease policy：

```text
RayGenerationRuntime
  resident lease: actor exists and stays up
  on_demand lease: acquire on generate, release on lifecycle request

RayActorMethodRuntime
  resident lease: actor group stays up across map calls
  on_demand lease: acquire on map, release on lifecycle request
```

### 可选后续扩展：sleeping

`sleeping` 可以作为后续扩展，不进本 sprint MVP：

```text
sleeping lease:
  actor process stays alive
  release calls actor.offload()
  next acquire calls actor.reload()
```

启用前必须先做 reload-cost probe：

```text
measure kill+relaunch cost:
  actor restart + from_pretrained + compile recapture + object-store weight push

measure offload/reload upper bound:
  model.to("cpu") + model.to("cuda") + empty_cache
```

如果差距不够大，保留 `resident | on_demand` 两态，不做 sleeping。

## 6. 阶段调度形状

目标形状：

```python
await lifecycle.enter("rollout_collect")
...

await lifecycle.enter("reward_score")
...

await lifecycle.enter("train_step")
...
```

`enter(phase)` 只做资源交接：

```text
1. 查 phase 需要哪些 roles
2. 查这些 roles 和当前 resident actors 是否 GPU 冲突
3. release 冲突 actor leases
4. acquire 当前 phase 需要的 actor leases
```

重要边界：continuous queue 仍然只放 trainer-ready batch。

当前 `ContinuousRolloutProducer` 调用的是 `collect_prompt_batches(...)`，该函数内部已经
`collect_unscored -> score_rollouts`，返回的是已打分、可训练的 batch。这个 sprint 不能把
continuous ready queue 改成 unscored queue，也不能破坏 `SPRINT_slime_overlap_strategy.md`
已经确定的 “ready queue 只放已打分 batch” contract。

所以 lifecycle hook 可以插入 `collect_prompt_batches` 内部 phases，但 continuous producer
对外仍然生产 scored batch。

## 7. 实施计划

### P0. 行为锁定测试

先锁住现有行为，不改 runtime：

```text
multi-GPU disjoint trainer/rollout/reward -> no release
trainer/rollout colocated -> release rollout before train by default
reward/rollout shared -> release rollout before reward
multiple GPU rewards sharing reward pool -> reward release after score
persistent colocated debug -> no rollout release, requires gpu_memory_fraction
continuous queue remains trainer-ready scored batches
```

覆盖文件：

```text
tests/ray/test_resources.py
tests/generation/ray/test_runtime_config.py
tests/generation/ray/test_rollout_launcher.py
tests/rewards/ray/test_resource_lifecycle.py
tests/rollouts/collector/test_runtime.py
tests/rollouts/orchestration/continuous/
```

### P1. 添加 `RayLifecyclePlan`

在资源解析层新增内部派生结果：

```text
vrl/ray/resources.py
  ResolvedDistributedResources.lifecycle: RayLifecyclePlan
```

旧字段暂时保留：

```text
rollout_release_after_collect
rollout_release_before_reward_model
reward_release_after_score
```

但它们改成和 `lifecycle` 同处派生的兼容字段，避免两套 source of truth。

### P2. 统一 runtime lease API 命名

把 runtime 的 release 语言统一成 lease 语言：

```text
RayGenerationRuntime.acquire()
RayGenerationRuntime.release()
RayActorMethodRuntime.acquire()
RayActorMethodRuntime.release()
```

现有方法保留兼容：

```text
release_memory() -> release()
shutdown() -> final teardown
```

目标是让 `release_after_call` 不再是 reward runtime 的独立习惯，而是 actor lease policy。

### P3. 把 handoff decision 移出 runtime

把这些调用点收口：

```text
collector/core.py:
  runtime.should_release_memory_before_reward()
  -> lifecycle.enter("reward_score")

launcher.py:
  if config.release_after_collect
  -> if lifecycle.rollout.mode == "on_demand"

ray/runtime.py:
  if release_after_call
  -> if lifecycle.reward.mode == "on_demand"
```

### P4. Public config 兼容

不先动 YAML，等内部模型稳定后再决定 public config 是否要保留旧字段。

兼容策略：

```text
旧字段继续可用
旧字段只作为 override
默认值只由 topology 派生
错误信息指向 phase lifecycle，而不是某个 runtime class
```

### 后续. Weight-sync topology 独立 sprint

单独开 sprint 处理：

```text
same-GPU colocated sync -> CUDA IPC handle path
split-GPU sync -> NCCL broadcast
cross-node / huge payload -> disk or delta path
```

当前证据：

```text
vrl/trainers/weight_sync.py:
  trainer state is converted to CPU first

vrl/generation/ray/weight_sync.py:
  state is put into Ray object store once, then sent to rollout workers
```

这是性能问题，但不是本 sprint 的 lifecycle cleanup。

### 后续. Sleeping/offload 探针

单独做一次性 probe：

```text
compare on_demand kill+relaunch vs model.to(cpu/cuda)
record numbers in this sprint or a follow-up note
only implement sleeping if the measured gap justifies the complexity
```

## 8. 非目标

不做：

```text
不把 trainer 变成 Ray actor
不把 reward 强行拆成 HTTP service
不把 GlobalRayPlacementOwner 变成 phase scheduler
不改变 continuous ready queue 的 trainer-ready batch contract
不取消 single-GPU release-after-collect 默认安全策略
不把 tiny colocated debug 伪装成通用吞吐配置
不实现 IPC / NCCL weight transport
不实现 sleeping / offload runtime
```

## 9. Thin functions / ALL_CAPS 清理

应保留：

```text
RayGenerationRuntime
  rollout transport boundary；统一 resident / on_demand lease 行为

RayActorMethodRuntime
  generic actor-method adapter；reward 只是它的一个使用者

RayRewardRuntime
  reward inference transport boundary；不要摊平成 collector 逻辑

GlobalRayPlacementOwner
  run-level placement owner；只管 placement group，不做 phase policy
```

应改变：

```text
release_after_collect / release_before_reward_model / release_after_score
  不再作为 runtime-scattered policy 使用
  统一派生成 RayLifecyclePlan

RayGenerationRuntime._ReleaseAfterCollectState
  改名为 _RuntimeLease 或并入统一 lease object
```

## 10. 验收标准

完成标准：

```text
1. multi-GPU disjoint config 明确解析成 resident no-release lifecycle。
2. shared-GPU config 明确解析成 phase handoff lifecycle。
3. colocated resident config 必须显式 gpu_memory_fraction。
4. runtime 不再自己解释 release_after_*；只执行 lease acquire/release。
5. collector 不再直接问 runtime "reward 前要不要 release"。
6. continuous ready queue 仍然只包含 scored trainer-ready batches。
7. 现有 single-GPU safety、reward shared-GPU handoff、owner-managed PG 测试全部保持通过。
8. resource plan log 能打印 lifecycle plan，方便读配置时立刻看出哪些 phase 会 release。
```

## 11. 参考

仓库代码：

- `vrl/ray/resources.py`
- `vrl/generation/ray/launcher.py`
- `vrl/generation/ray/runtime.py`
- `vrl/generation/execution/worker.py`
- `vrl/ray/runtime.py`
- `vrl/ray/lifecycle.py`
- `vrl/rewards/ray/runtime.py`
- `vrl/rollouts/collector/core.py`
- `vrl/rollouts/orchestration/prompt_collection.py`
- `vrl/rollouts/orchestration/continuous/producer.py`
- `vrl/trainers/weight_sync.py`
- `vrl/generation/ray/weight_sync.py`

仓库 reading / design：

- `docs/sprints/planned/SPRINT_slime_overlap_strategy.md`
- `docs/sprints/done/SPRINT_reward_execution.md`
- `docs/sprints/planned/SPRINT_memory_plan_full.md`
- `docs/sprints/reading/slime.md`
- `docs/sprints/reading/cosmos-rl.md`
- `docs/sprints/reading/SPRINT_cosmos_rl_scaling_learnings.md`
- `docs/sprints/reading/SPRINT_framework_lessons_vrl.md`

外部：

- vLLM sleep mode: https://docs.vllm.ai/en/latest/features/sleep_mode/
- veRL hybrid flow: https://verl.readthedocs.io/en/latest/hybrid_flow.html
- OpenRLHF hybrid engine: https://openrlhf.readthedocs.io/en/latest/hybrid_engine.html
- THUDM/slime: https://github.com/THUDM/slime
- NVIDIA cosmos-rl: https://github.com/NVIDIA/cosmos-rl
