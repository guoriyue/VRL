# SPRINT: Global Ray placement owner（closeout）

状态：implemented / closeout（2026-06-14）。核心 run-level owner、role bundle
mapping、rollout/reward owner-managed placement、online recipe cleanup 已落地；本文保留
原设计作为决策记录。当前 reward execution contract 归
`docs/sprints/done/SPRINT_reward_service.md`；多 GPU 吞吐测量归
`docs/sprints/parked/SPRINT_video_rollout_stage_overlap.md`。

> Runtime follow-up（2026-07-30）：当前只有一个 `RayGenerationRuntime` lifecycle
> owner。它 eager 持有或 deferred 创建一个无 lifecycle 的 `RayGenerationSession`；
> 下文 `release-after-collect` / `Releasable` 用词只记录本 sprint 落地时的旧 runtime
> 形状，不是当前 API。placement group 的 run-level ownership 决策保持不变。

## Closeout 状态（2026-06-14）

已完成：

- `vrl/ray/placement.py` 提供 `GlobalRayPlacementOwner` / `RolePlacement`，由一个
  run-level owner 创建、probe、持有并关闭唯一 placement group。
- `vrl/ray/resources.py` 的 `BundleLayout` 是 role bundle plan 的 source of truth；
  旧的 reward offset / resource plan 冗余字段已删除。
- `vrl/generation/ray/launcher.py` 的 `create_runtime` 接收 owner-managed
  `RolePlacement`；private session launch 只创建 rollout actors，runtime/session
  parking 或 close 都不移除 owner PG。
- `vrl/ray/runtime.py` / `vrl/rewards/ray/runtime.py` 接收 reward role placement；
  reward runtime release 只释放 reward actors，不移除 owner PG。
- `vrl/scripts/common/online.py` 在 run 开始创建 owner，并在 `finally` 中按
  collector -> reward -> placement owner 顺序 shutdown；训练异常、launch 异常、构建异常、
  checkpoint 保存异常都不会泄漏 owner PG，shutdown 失败也不会盖掉原始训练异常。
- `tests/scripts/test_online_lifecycle.py` 覆盖 run-level 生命周期：正常结束、owner create
  失败、rollout launch 失败、reward/collector 构建失败、final checkpoint 失败、cleanup
  失败不覆盖训练异常。
- `tests/generation/ray/test_rollout_launcher.py` 已覆盖 deferred runtime 的
  session launch/cleanup 不删除 owner-managed PG。

剩余项 / 转出：

- Reward async scoring 与 overlap capability contract 已由
  `docs/sprints/done/SPRINT_reward_service.md` 收口；真实多 GPU 吞吐测量留在
  `docs/sprints/parked/SPRINT_video_rollout_stage_overlap.md`。
- 历史 `reward_cost` 成本感知 auto-placement 提案已撤回，没有 active owner；当前 device
  placement 只认 `vrl/ray/resources.py` 的 typed resource contract。
- Memory budget、ReplayModel parity、把 trainer 变成 Ray actor 都不是本 sprint 目标。
- 不保留 `vrl/generation/ray/placement.py` 兼容 shim；调用方直接从 `vrl.ray.placement`
  导入全局 placement 边界。

## 0. Core Decision

把 rollout 和 reward 的 Ray placement ownership 上移到训练入口，由一个
`GlobalRayPlacementOwner` 在 run 开始时创建、probe、持有并在 run 结束时释放
placement group。

落地前结构是两套独立 PG：

```text
rollout:
  online.py
    -> legacy generation launcher
       -> create_generation_placement_group(...)

reward:
  RayRewardRuntime
    -> RayActorMethodRuntime._ensure_placement(...)
       -> create_placement_group(...)
```

这导致 release / reacquire 场景反复创建 rollout PG、probe bundle map；reward 侧还要维护
`gpu_reservation_count` offset math，避免单独的 reward PG 抢错 GPU 或挂在
`pg.ready()`。

已落地目标结构：

```text
online.py
  -> resolve distributed resources once
  -> create GlobalRayPlacementOwner once
  -> pass rollout RolePlacement to RayGenerationLauncher
  -> pass reward RolePlacement to RayRewardRuntime / RayActorMethodRuntime
  -> shutdown one placement owner at run end
```

## 1. Why Full Is A Separate Sprint

`RayGenerationRuntime` 的 release-after-collect scoped 优化只解决 rollout PG 每 epoch
重建。
full 版本要改的是 resource model：

```text
rollout actor placement
reward actor placement
trainer GPU reservation
shared rollout/reward phase handoff
cross-node placement validation
resource tests
```

它不是 `vrl/generation/ray` 内部改动，会触达：

```text
vrl/scripts/common/online.py
vrl/generation/ray/{launcher,runtime}.py
vrl/rewards/ray/runtime.py
vrl/ray/{runtime,resources,placement}.py
tests/ray/test_resources.py
tests/generation/ray/*
tests/rewards/ray/* 或新增覆盖
```

## 2. Required Design

### 2.1 Placement owner

新增一个长期对象，例如：

```text
GlobalRayPlacementOwner
```

职责：

```text
create one Ray placement group
build role -> bundle_indices map
probe actual GPU ids once
validate actual placement
launch / kill reservation actors only when truly needed
expose scheduling handles for rollout and reward runtimes
remove placement group exactly once at run shutdown
```

它应该是唯一 PG owner。`RayGenerationRuntime` 和 `RayActorMethodRuntime` 在
owner-managed placement 模式下只 owns workers，不 owns PG。

### 2.2 Bundle plan from resources

`vrl/ray/resources.py` 当前输出的是设备和 reservation count。full 版本应输出 role
bundle plan：

```text
trainer_reserved_bundles
rollout_bundles
reward_bundles
shared_inference_bundles
```

关键点：trainer 不是 Ray actor。trainer bundle 的作用不是运行 trainer，而是保护 driver
GPU 不被 Ray actors 抢走，并记录实际 GPU id 用于 validation。

### 2.3 Shared rollout/reward GPU

如果 reward 与 rollout 共享 GPU，global PG 不能给它们各建一份 `GPU:1` bundle。
正确模型是同一个 shared inference bundle 分时运行：

```text
collect phase:
  rollout workers live on shared bundle
  reward workers absent

reward phase:
  rollout workers released
  reward workers live on same shared bundle

train phase:
  shared inference workers released when trainer shares that GPU
```

因此 role placement API 必须支持：

```text
same placement_group
same bundle_indices
different actor classes at different phases
```

不能同时 resident 的角色要由 release flags 和 lifecycle barrier 保证。

### 2.4 Dedicated reward GPU

如果 reward 有独立 GPU，reward bundle 是 permanent role bundle：

```text
rollout bundle != reward bundle
reward actor may stay resident if reward.release_after_score=false
```

这种情况下 full owner 仍然有价值：reward 不再需要单独 `gpu_reservation_count` 来
把 Ray 调度推到正确 GPU。

### 2.5 Cross-node

cross-node 模式不能用 driver-local CUDA ordinal 简单比交集。owner 必须沿用当前
cross-node preflight 原则：

```text
driver/head node should not expose Ray GPUs for remote rollout
actual actor node_ip + gpu_ids must be validated per role
trainer reservation is skipped when it cannot protect remote nodes
```

## 3. Implementation Phases

### P0. Characterization tests

先写不改行为的测试，锁住现状：

```text
rollout creates its own PG
reward creates its own PG
RayGenerationRuntime release-after-collect release removes rollout PG
reward_gpu_reservation_count protects reward placement in split-GPU plans
shared rollout/reward derives release_after_collect / release_before_reward_model / release_after_score
```

这些测试后面要被替换或改写，但先防止重构时看不见行为变化。

### P1. Placement owner skeleton

新增 owner，不接入 runtime：

```text
vrl/ray/placement.py
```

它只根据 `ResolvedDistributedResources` 创建 bundle plan、PG、probe actual placement，
并能 shutdown。先用单元测试覆盖：

```text
dedicated trainer / rollout / reward
shared rollout+reward
colocated trainer+rollout debug
cross-node no trainer reservation
```

### P2. Owner-managed placement for rollout

让 generation launcher 接收 owner-managed placement；当前 API 是
`RayGenerationLauncher.create_runtime(..., placement=...)`，实际 actor construction
由 private session launch 消费同一个 placement：

```text
placement: RolePlacement
```

要求：

```text
eager runtime keeps the launched session
deferred runtime parks or closes its session but not the owner-managed PG
launcher error path only kills workers it created
launcher does not remove the owner-managed PG
```

### P3. Owner-managed placement for reward

让 `RayActorMethodRuntime` 接收 owner-managed placement：

```text
placement_group
bundle_indices
owns_placement: false
```

然后 `RayRewardRuntime` 透传它。

要求：

```text
release_after_call kills reward actors but leaves the owner-managed PG alive
no gpu_reservation_count needed when role bundle indices are supplied
validation still checks actual reward gpu_ids
```

### P4. Wire online.py

`online.py` 在 build collector / reward / rollout 之前创建 owner，并把 role placement
传给 rollout 与 reward。

生命周期：

```text
create owner after resources resolve
launch rollout runtime with owner.rollout_placement
construct reward runtime with owner.reward_placement
trainer loop runs
finally: shutdown trainer/collector/reward/placement owner
```

必须处理异常路径：bundle build、rollout launch、reward launch、trainer construction 任一失败，
owner 都要释放 PG。

### P5. Delete old reservation math

只有 P2-P4 稳定后再删：

```text
reward_gpu_reservation_count
generation-only trainer reservation actor creation inside create_generation_placement_group
reward runtime private PG offset math
```

`resources.py` 保留资源解析和 topology validation，但不再输出 reward offset count。

### P6. Test rewrite

重写 `tests/ray/test_resources.py` 的 reservation-count 断言，改成 role bundle plan 断言：

```text
which roles share bundles
which roles own dedicated bundles
which release flags are required
which topology is rejected
actual placement validation inputs
```

补 runtime tests：

```text
rollout owner-managed PG is not removed by release_memory
reward owner-managed PG is not removed by release_after_call
shared rollout/reward use same bundle index but are not resident together
shutdown removes PG once
error path removes PG once
```

## 4. Acceptance

```text
pytest -q tests/ray/test_resources.py
pytest -q tests/generation/ray/
pytest -q tests/rewards/ray/ tests/rewards/
pytest -q tests/rollouts/orchestration/
pytest -q tests/trainers/
```

Manual smoke for each topology:

```text
single GPU colocated debug
2-GPU trainer+rollout split
3-GPU trainer+rollout+reward dedicated
2-GPU trainer + shared rollout/reward
cross-node rollout
```

Runtime proof:

```text
PG created once per run
PG removed once per run
rollout actor release does not remove the owner-managed PG
reward actor release does not remove the owner-managed PG
shared rollout/reward never launch simultaneously on the same bundle
```

## 5. What Changes

```text
online.py becomes the owner of run-level Ray placement lifecycle.
RayGenerationLauncher accepts owner-managed placement.
RayActorMethodRuntime accepts owner-managed placement.
resources.py emits a role bundle plan instead of reward gpu_reservation_count.
tests assert bundle ownership and phase handoff, not reservation-count arithmetic.
```

## 6. What Stays

```text
RayActorGroup stays the actor launch facade.
RayGenerationRuntime stays the collector-facing generation runtime facade.
RayRewardRuntime stays the reward-facing runtime facade.
RayGenerationLauncher still knows how to launch rollout workers.
RayActorMethodRuntime still maps payloads over homogeneous actor methods.
release_after_collect / release_before_reward_model / release_after_score stay as phase lifecycle policy.
```

These thin facades should stay because they are protocol boundaries. Flattening them into `online.py`
would make placement simpler locally but would mix entrypoint orchestration with actor launch,
runtime facade, and reward transport concerns.

## 7. Architecture Hygiene

### Constants / hardcoded data

Keep:

```text
closed strategy vocabularies such as chunk placement strategies
metadata keys that are external contracts
resource role names if centralized in one bundle-plan module
```

Change:

```text
do not keep reward_gpu_reservation_count as a derived public field after role bundle plan exists
do not duplicate role names across launcher/reward/runtime tests; derive them from the bundle plan object
```

Why:

```text
reservation count is an implementation artifact of separate PGs.
role bundle mapping is the new source of truth.
```

### Thin functions / files

Keep:

```text
RayGenerationLauncher: create one runtime and launch its eager/deferred actor session
RayGenerationRuntime: sole collector-facing lifecycle and phase owner
RayGenerationSession: launched executor/weight-sync/actor resource boundary without lifecycle
RayRewardRuntime: reward transport boundary
RayActorMethodRuntime: generic actor-method adapter
RayActorGroup: actor launch facade
```

Do not flatten these into one owner. The new owner should own placement lifecycle only, not actor
startup semantics or reward/generation business contracts.

## 8. Non-Goals

```text
不把 trainer 变成 Ray actor
不重写 rollout scheduling / chunk dispatch
不重写 reward scoring semantics
不改变 weight sync protocol
不改变 continuous rollout queue / staleness behavior
不把 scoped Releasable PG reuse 和 full global owner 混在同一个 MR
不为了删除 LOC flatten runtime facades
```

## 9. 后续不要重做

这份 sprint 的 placement owner 主体已经落地。后续 agent 不要按旧 prompt 重建
`GlobalRayPlacementOwner`、`RolePlacement`、owner-managed rollout/reward runtime 或
run-level online cleanup。

只剩两类工作应该继续：

```text
reward execution:
  - P1 async scoring / reward backlog
  - P2 cost-aware auto placement
  - P2 remove reward-key hardcoding in factory.py

verification:
  - keep run-level lifecycle tests green
  - keep deferred runtime session parking/close from deleting owner PG
```

## 10. References

- `vrl/scripts/common/online.py`
- `vrl/generation/ray/launcher.py`
- `vrl/generation/ray/runtime.py`
- `vrl/generation/ray/session.py`
- `vrl/rewards/ray/runtime.py`
- `vrl/ray/runtime.py`
- `vrl/ray/placement.py`
- `vrl/ray/resources.py`
- `tests/scripts/test_online_lifecycle.py`
- `tests/ray/test_resources.py`
- `tests/generation/ray/`
