# SPRINT：按 GPU 细化角色重叠与生命周期

Status: **DONE / superseded (2026-07-11)** by
[`SPRINT_miles_phase_lease_and_one_continuous.md`](../SPRINT_miles_phase_lease_and_one_continuous.md).

> 本提案依赖 role `memory_fraction` 与同卡 resident lease；两者已经从产品面删除，因此 T2、
> T3 和以它们为前提的 T4 不再实施。未来若需要逐卡 engine budget，必须从具体 backend 的真实
> consumer 重新设计，不能恢复通用 role cap。以下正文只保留为被取代的设计记录。

本文描述尚未实现的增量；当时 main 只有角色级
重叠、标量显存上限和角色级 lease。本轮文档纠正此前误写的“已实现 T1–T4”。

## 0. 当前事实基线

当前 resolver 已支持给 trainer / rollout / reward 显式指定 GPU id，并由集合交集
推导角色是否共卡。公开 grammar 以角色的 `devices`、`gpu_pool` 和 rollout 的
`memory_fraction` 为主。

但派生结果仍是角色级：

```python
@dataclass(frozen=True, slots=True)
class DistributedResourceConfig:
    allow_overlap: bool = False
    rollout_gpu_memory_fraction: float | None = None

@dataclass(frozen=True, slots=True)
class ActorLeasePolicy:
    mode: Literal["resident", "on_demand"]
```

当前 main **没有**以下对象或行为：

- `OverlapPolicy` 或按角色对区分的 overlap 权限；
- `{gpu_id: fraction}` 形式的逐卡显存上限；
- `ActorLeasePolicy.by_gpu`；
- `_per_gpu_lease` 或运行时逐卡消费 lease；
- 已验证的 FSDP rank 与 rollout worker 多卡共卡执行路径。

因此现状能表达“这些角色是否有交集”，不能完整表达同一角色跨越“共享卡 + 专用卡”
时每张卡不同的常驻、让卡和显存策略。

## 1. 目标

让显式 GPU id 成为逐卡运行策略的输入，而不只是角色级 `colocated: bool` 的来源：

1. overlap 权限按角色对区分；
2. rollout 显存上限可以按物理 GPU 表达；
3. lease 和 handoff 可以区分同一角色中的共享卡与专用卡；
4. 只有运行时真正消费逐卡计划后，才允许 FSDP 多卡共卡配置通过。

单卡应自然成为逐卡模型的 N=1 情形，但不能为了形式统一提前放开没有执行语义的
多卡配置。

## 2. 实施顺序

### T1：按角色对解析 overlap 权限

将全局 bool 扩展为具名策略：

```python
@dataclass(frozen=True, slots=True)
class OverlapPolicy:
    trainer_rollout: bool = False
    trainer_reward: bool = False
```

兼容规则：bool 只作为两个字段的广播简写；mapping 拒绝未知 key。rollout 与 reward
是否共享仍由 reward 的 GPU-pool grammar 决定，不复制到第三个开关。

### T2：按 GPU 解析 rollout 显存上限

允许 `memory_fraction` 是标量或 `{gpu_id: fraction}`。标量表示所有共享 rollout 卡
使用同一上限；mapping 的 key 必须精确覆盖需要上限的共享卡，并对每个值执行 `(0, 1]`
校验。

派生结构保留物理 GPU id 与 fraction 的绑定，worker 必须用实际分配到的 GPU 查值。
只有 resolver 接受 mapping、运行时仍读取标量，不算完成。

### T3：逐卡 lease 并贯通运行时

为 lease 增加逐卡真相，同时保留 `mode` 作为向后兼容的保守汇总：

```python
@dataclass(frozen=True, slots=True)
class ActorLeasePolicy:
    mode: Literal["resident", "on_demand"]
    by_gpu: tuple[tuple[int, Literal["resident", "on_demand"]], ...] = ()
```

完成条件不是 dataclass 多一个字段，而是 launcher、handoff 和 actor lifecycle 按 worker
所在 GPU 消费它。否则 `by_gpu` 只会成为日志/测试字段，违反 resolved struct 的消费规则。

### T4：最后开放 FSDP 多卡共卡

只有 T1–T3 的运行时路径完成后，resolver 才能接受 FSDP trainer 与 rollout 的部分重叠。
必须证明每个共享 rank 的常驻显存、weight sync 和 phase handoff 都有真实执行语义；不能
先让配置 resolve 成功，再把未实现行为留给 GPU 运行时报错或 OOM。

trainer 与 reward 共卡保持禁止，直到存在同等完整的 resident/release 协调机制。

## 3. 验收

CPU-only 契约门：

- bool overlap 与旧配置得到相同派生结果；
- mapping 能允许 trainer↔rollout、同时拒绝 trainer↔reward；
- 未知 overlap key、缺失/多余 GPU fraction key、越界 fraction 全部 fail fast；
- 混合 GPU 池的 `by_gpu` 同时包含 resident 与 on-demand，且每项有非日志消费者；
- 单卡 `mode` 与唯一 `by_gpu` 项一致；
- FSDP overlap 在 T4 前继续被拒绝。

CPU 门通过后，另开有空闲 GPU 的硬件验收：

- 部分重叠拓扑不会错误释放专用卡上的 worker；
- 共享卡应用自己的 fraction，专用卡不套共享上限；
- FSDP rank 与 rollout worker 的交接无 OOM、无 actor 重建泄漏；
- 记录每阶段 wall time，证明共卡策略的收益或代价。

## 4. 应保持不变

- `GlobalRayPlacementOwner`、`BundleLayout` 和 `RayLifecyclePlan` 继续作为现有协议边界；
- 角色 GPU 集合交集仍是 topology 的事实来源，不新增第二套 `colocated` 配置；
- `gpu_pool=auto|trainer|dedicated` 保持公开 grammar；
- rollout↔reward 共享不重复塞进 `OverlapPolicy`；
- 已删除的 `rollout.colocate`、`release_after_collect` 等 key 继续硬拒绝；
- 没有必要拆新薄文件，解析与派生仍留在 `vrl/ray/resources.py`。

这些薄结构提供跨 launcher / collector / reward runtime 的协议边界，应保留；本 sprint
不是为了减少行数而拍平它们。

## 5. 非目标

- 不实现 MPS、MIG 或跨 stream 真并行；
- 不声称“显存放得下”等于吞吐更高；共卡可能只是在 phase 间交替；
- 不重建已完成的 global placement owner 和 phase lifecycle；
- 不在没有硬件证据时把多卡 colocated 标为 runnable；
- 不新增 module-level ALL_CAPS 角色表；角色关系应从 typed policy 和设备集合派生。

## 6. 代码证据

- `vrl/ray/resources.py`：当前 bool overlap、标量 fraction、角色级 lease 和派生结果；
- `vrl/generation/ray/config.py`：generation worker 的显存上限输入；
- `vrl/generation/execution/worker.py`：worker 应用显存上限的位置；
- `tests/ray/test_resources.py`：当前 gpu-pool grammar、旧 key 拒绝和 lifecycle 契约；
- `docs/sprints/done/SPRINT_global_ray_placement_owner.md`：保持不变的 placement 底座；
- `docs/sprints/done/SPRINT_ray_phase_lifecycle_plan.md`：保持不变的 lifecycle 底座。
