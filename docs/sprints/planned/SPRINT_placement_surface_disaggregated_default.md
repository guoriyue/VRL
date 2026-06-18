# SPRINT: Disaggregated-default placement surface

状态：**P0 + P1 + P2 已落地（2026-06-18）；P3 deferred。** 全程保持向后兼容，182 个相关测试通过。实现细节见各 Work item 的「已落地」注。

## 目标

把分布式 placement 的用户配置收敛成一个简单模型：

1. 用户日常只声明每个 role 要多少 GPU：`trainer.num_gpus`、`rollout.num_gpus`、`reward.num_gpus`。
2. 默认 disaggregated：有空闲 GPU 时，trainer / rollout / reward 自动分到不同卡。
3. 只有需要共享 GPU 时才显式写共享意图；用户不手写 release 时机。
4. `devices: [id]` 保留为 debug/advanced override，不作为最终主路径。

当前复杂度来自这些低级 knob 混在一起：

- `cross_node`
- `allow_overlap`
- `colocate_with_trainer`
- `devices`
- `share_with_rollout`

这几个不是同一类概念。目标是把 public surface 改成“资源需求 + 少量共享策略”，把 release/resident 生命周期留给 engine 从 topology 派生。

## 当前事实

`resolve_distributed_resources` 已经能做到单机多卡默认 disaggregated。只写每个 role 的 `num_gpus: 1` 时，3-GPU 机器会解析成：

```text
trainer=(0,) rollout=(1,) reward=(2,)
colocated=False
rollout lifecycle=resident
```

机制：

- `_resolve_rollout_devices` 默认排除 trainer devices，从剩余 GPU 里取 rollout。
- `_resolve_reward_devices` 默认先找 trainer/rollout 之外的 spare GPU。
- 如果 reward 没有 spare GPU，当前 `share_with_rollout is None` 会隐式 fallback 到 rollout pool，并由 lifecycle 派生 `release_rollout_before_reward` / `release_reward_after_score`。这个行为有效，但作为最终用户 surface 太绕：用户配置时应该只选择 reward 的 GPU pool 来源 —— `auto` 或 `rollout`。

所以缺的不是能力，而是 public naming 和参数关系。

## 参数设计

### 1. 主路径：只写 GPU 数量

```yaml
distributed:
  resources:
    trainer: {num_gpus: 1}
    rollout: {num_gpus: 1}
    reward:  {num_gpus: 1}
```

含义：

- trainer 需要 1 张 GPU。
- rollout 需要 1 张 GPU。
- 只有 active `execution: pool` reward 时，reward 才需要 GPU pool。
- engine 自动选择物理 GPU、Ray bundle、lifecycle。

### 2. Reward execution 决定 reward 是否进入资源规划

Reward 有两种执行形态，必须和 GPU resource surface 分清：

```yaml
reward:
  kwargs:
    small_local_reward:
      execution: inline
```

`execution: inline` 表示 in-process/local reward。它不走 Ray reward pool，不需要 `distributed.resources.reward`，也不应该占用 reward GPU。

```yaml
reward:
  kwargs:
    kling_video_reward:
      execution: pool
```

`execution: pool` 表示 Ray reward actor pool。只有这种 reward 才消费：

```yaml
distributed:
  resources:
    reward:
      num_gpus: 1
      gpus_per_worker: 1.0
      num_workers: 1
```

CPU-only reward pool 仍然允许，但必须显式表达为 CPU pool：

```yaml
distributed:
  resources:
    reward:
      num_gpus: 0
      gpus_per_worker: 0.0
      num_workers: 1
```

规则：

- 没有 active `execution: pool` reward 时，`distributed.resources.reward` 应该省略或解析成 0 GPU。
- 有 active GPU pool reward 时，`reward.num_gpus > 0` 且 `gpus_per_worker > 0`。
- 有 active CPU pool reward 时，`reward.num_gpus: 0` 且 `gpus_per_worker: 0.0`。

### 3. Trainer + rollout 共卡：`rollout.colocate`

```yaml
distributed:
  resources:
    trainer: {num_gpus: 1}
    rollout: {num_gpus: 1}

  rollout:
    colocate:
      memory_fraction: 0.4
```

含义：

- rollout worker 和 trainer process 常驻共享同一张 GPU。
- `memory_fraction` 是 rollout worker 的 CUDA allocator 上限，避免 trainer backward 被 rollout 挤 OOM。
- 这是 resident colocation，不是 rollout/train async，也不是 reward/rollout handoff。

命名选择：

- 当前代码名是 `distributed.rollout.colocate_with_trainer`。
- 目标名是 `distributed.rollout.colocate.memory_fraction`。
- 不放到 `distributed.resources.rollout` 下，避免把“runtime 共卡内存预算”和“GPU 数量需求”混在一个 block。

为什么不是只写 `colocate: true`：

- trainer/rollout 共卡不是 reward/rollout 那种 phase handoff。trainer process 通常仍持有模型/optimizer/LoRA CUDA memory，rollout worker 又是另一个进程；两个 PyTorch allocator 不会自动公平分配显存。
- 如果 resident rollout 没有 cap，它可以把共享 GPU 的剩余显存吃光，下一次 trainer backward 才 OOM。这个 OOM 看起来像“训练中途炸”，但根因是 colocated rollout 没有预算。
- 所以在当前 runtime 下，**resident trainer/rollout colocation 必须有 GPU memory budget**。可以把字段名做短、把文档写清楚，但不能静默无上限。

未来如果有完整 memory budget system，可以允许：

```yaml
distributed:
  rollout:
    colocate: true
```

但前提是 engine 能从全局 memory policy 推导 rollout 的 GPU cap。没有这个能力前，bare `colocate: true` 应该报错并提示补 `memory_fraction`，而不是默认 uncapped。

### 4. Reward GPU pool：`gpu_pool`

当前 `share_with_rollout: bool | None` 是一个混乱的三态：

- `None`：先找 dedicated spare GPU；找不到就 fallback 到 rollout pool。
- `True`：强制 reward 使用 rollout pool。
- `False`：要求 dedicated reward GPU；没有 spare 就报错。

目标名改成更接近用户心智的 pool 选择：

```yaml
distributed:
  resources:
    reward:
      num_gpus: 1
      gpu_pool: auto
```

语义：

- `gpu_pool: auto`：engine 自动给 reward 找 GPU pool；优先使用 dedicated spare GPU。没有 spare GPU 时是否允许自动退到 rollout pool 由当前 resolver 兼容期决定，长期应配合 preset 明确化。
- `gpu_pool: rollout`：reward 使用 rollout GPU pool；rollout 在 reward scoring 前释放，reward score 后释放。

可选值：

| 值 | 含义 | 对应旧行为 |
|---|---|---|
| `auto` | engine 自动选择 reward GPU pool；推荐主路径 | `share_with_rollout: null` |
| `rollout` | 明确使用 rollout GPU pool，并通过 lifecycle handoff 释放/重建 actor | `share_with_rollout: true` |

不再引入 `fallback_gpu_pool`。它把用户本来只想表达的 pool 来源说成了“失败后的处理”，名字太绕。
`share_with_rollout: false` 这种“必须 dedicated，否则报错”的严格模式先作为兼容输入保留；如果未来确有长期需求，再给 advanced 名字，不进入主路径。

Reward/rollout sharing 和 trainer/rollout colocate 的区别：

- `rollout.colocate.memory_fraction`：trainer 和 rollout 同时常驻一张 GPU，需要显存配额。
- `reward.gpu_pool: rollout`：reward 和 rollout 分阶段 handoff，同一时间只让一个 heavy actor 占用 GPU，不需要 memory fraction。

### 5. Debug/advanced override：`devices`

```yaml
distributed:
  resources:
    trainer: {devices: [0]}
    rollout: {devices: [1]}
    reward:  {devices: [2]}
```

`devices` 保留，但不是推荐主路径。它只用于：

- 本地 debug。
- 复现实验。
- 临时绕过 resolver 自动分配。
- cross-node 当前的预算 token 方案。

长期目标仍然是：用户声明 `num_gpus`，不是物理 GPU ID。

## 参数关系

### Resource block 只表达 GPU/CPU capacity

`distributed.resources.*` 只回答“这个 role 需要多少资源”：

- `num_gpus`
- `gpus_per_worker`
- `num_workers`
- `devices`（advanced）
- `gpu_pool`（reward only，选择 reward GPU pool 来源）

它不表达 release timing。

### Runtime block 表达 actor/runtime 行为

`distributed.rollout.*` 和 `distributed.reward.*` 只放 runtime 行为：

- `distributed.rollout.colocate.memory_fraction`
- `distributed.rollout.chunk_placement_strategy`
- `distributed.rollout.sync_trainable_state`
- `distributed.reward.cpus_per_worker`
- `distributed.reward.placement_strategy`
- `distributed.reward.max_inflight_batches`

这些不是 GPU 数量需求，不能和 `resources` 混成一个概念。

### Lifecycle 全部派生

用户不写：

- `release_after_collect`
- `release_before_reward_model`
- `persistent_colocated_workers`

Engine 从 resolved topology 派生：

- dedicated GPU -> resident
- trainer/rollout resident colocate -> resident rollout with memory cap
- reward/rollout shared pool -> rollout before reward 释放，reward after score 释放
- trainer/rollout non-resident overlap -> rollout collect 后释放

## Work items

### P0：只改 surface 设计和兼容解析

- 把目标配置文档改成 `num_gpus` 主路径。
- 新增 `distributed.rollout.colocate.memory_fraction`，兼容旧 `colocate_with_trainer.memory_fraction`，但新配置和 preset 只用 `colocate.memory_fraction`。
- 新增 `distributed.resources.reward.gpu_pool: auto|rollout`。
- 保留旧 `share_with_rollout` 作为兼容输入，并把它映射到新 policy：
  - omitted/null legacy -> `gpu_pool: auto`
  - `True` -> `gpu_pool: rollout`
  - `False` -> dedicated-only compatibility path；暂不进入推荐 surface
- 更新 schema、preset、错误信息和 tests。

> **✅ 已落地（P0）。** `vrl/ray/resources.py`：`RewardResourceConfig.gpu_pool`（替代
> `share_with_rollout` 字段）、`_parse_colocate`（新 `colocate` + 旧 `colocate_with_trainer`
> 二选一）、`_parse_reward_gpu_pool`（新 `gpu_pool` + 旧 `share_with_rollout` 映射）；reward
> device 解析 + overlap 校验改读 `gpu_pool`；所有用户面错误信息改推荐新名。`vrl/config/schema.py`
> 同时识别新旧键。`vrl/generation/ray/config.py` 错误信息改 `colocate`。preset
> `sd3_5/online_grpo_ocr_single_gpu_async_debug.yaml` 迁到 `colocate`。`tests/ray/test_resources.py`
> 加 8 个新键 + 新旧等价 + 冲突/非法值测试。resolver 核心 device-set 算法未动（守住非目标）。

### P1：移除 release mirror

- `RayGenerationConfig.release_after_collect` 只是 lifecycle 的扁平镜像。
- 把校验改成读 `resources.lifecycle.rollout.mode` / handoff policy。
- 删除 `RayGenerationConfig.release_after_collect` 和相关 fallback。

> **✅ 已落地（P1）。** 删除 `RayGenerationConfig.release_after_collect` 字段 + `__post_init__`
> 的 persistent↔release 互斥校验 + `from_cfg` 回灌。`_validate_driver_cuda_ownership`（共卡碰撞守卫）
> 改读 `resources.lifecycle.rollout.mode` + `resources.rollout_persistent_colocated_workers`；
> `launcher.py` on_demand 判定去掉 `else config.release_after_collect` fallback（无 resolved plan
> 时默认 resident）。tests（memory_guards / runtime_config / rollout_launcher）改掉对扁平字段的
> 构造/断言，改读 `resources.rollout_release_after_collect`。

### P2：multi-node auto detect

- 当前 `resolve_distributed_resources` 在 `ray.init()` 前运行，所以不能可靠读 `ray.nodes()`。
- 后续需要 init/attach Ray 后再做 node-aware resolve，或做 two-phase resolve。
- `cross_node` 最终应变成 override，不是 common path 必填项。

> **✅ 已落地（P2，gated 安全版）。** `vrl/scripts/common/online.py` 新增
> `_maybe_autodetect_cross_node(cfg, ray)` + `_cluster_has_non_driver_gpus(ray)`，在 resolve **之前**
> 调用。逻辑对标 slime/cosmos-rl（operator 先 `ray start`、driver 再 attach）：cross_node 未显式设置时，
> 用 `ray.init(address="auto")` **attach** 到已起集群；attach 成功且非 driver 节点有 GPU → 自动
> `cross_node=true`。**单机不变**：无外部集群 → `ConnectionError` → 跳过 → 后面正常 `ray.init()` 起本地实例。
> 显式 `cross_node`（true/false）永远是 override。集群不可探测时 best-effort no-op（不误开）。
> 测试 `tests/scripts/test_online_cross_node_autodetect.py`（stub ray，6 例）。
>
> **仍需真机验证**：多节点自动开启那条路只在 stub 下测过，需真 2-node 集群跑一次确认 attach 时机不影响
> trainer 模型放置 / CUDA context。另：自动开启 cross_node 后 resolver 仍要求各 role `num_gpus` 是显式整数
> （非 `auto`）——这是 cross_node 既有约束，不在本 sprint 范围。

### P3：可选 parallelism surface

等 FSDP multi-replica / rollout multi-replica 真正落地后，再考虑：

```yaml
distributed:
  resources:
    trainer:
      num_gpus: 8
      parallelism: {dp: 8}
    rollout:
      num_gpus: 4
      parallelism: {tp: 1, dp: 4}
```

现在不要提前加空 scaffolding。

## 非目标

- 不重写 resolver 的核心 device-set 算法；它已经能做 disaggregated default。
- 不把 `trainer` 全仓重命名成 `policy`。
- 不把 `devices` 做成主路径。
- 不暴露 release timing 给用户。
- 不把 reward inline/in-process reward 强行塞进 GPU resource planning。
- 不为了兼容 `share_with_rollout: false` 而污染最终推荐 surface；旧键先兼容，真实需求确认后再给 advanced 名字。

## 证据

- `vrl/ray/resources.py`：`RewardResourceConfig.share_with_rollout` 当前是三态 placement preference。
- `vrl/ray/resources.py`：`_resolve_reward_devices` 当前先找 spare GPU，再 fallback 到 rollout pool。
- `vrl/ray/resources.py`：`reward_runtime_resource_kwargs` 只为 Ray reward pool 提供 worker/runtime resource kwargs。
- `vrl/rewards/runtime.py`：`execution: inline` 使用 `LocalRewardRuntime`，`execution: pool` 使用 `RayRewardRuntime`。
- `vrl/scripts/common/factory.py`：只有 active `execution: pool` reward 才合入 resolved reward runtime kwargs。
- `vrl/config/schema.py`：当前 `distributed.resources.reward` 是 resource block，`distributed.reward` 是 reward runtime block。
