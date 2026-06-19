# SPRINT: design-smell loose ends（weight_sync_barrier + release-flag 影子字段）(planned)

状态：**planned（2026-06-18 从 [[SPRINT_design_smell_audit]] 拆出）**。design_smell 审计的确证安全项已全部落地、那个 sprint 已归档 `done/`；这里只收它剩下的 2 个**架构卫生尾巴**。两者都不是"去实现一个已知安全的改动"，而是各带一个取舍，所以单列、低优先。

## 0. 来历

[[SPRINT_design_smell_audit]]（现 `done/`）做完后，本 doc 自有的开放项只剩 2 个 derived/shadow 字段的处置。其余原"仍开放"项已收口：`advantage_low` 非对称数学、`pool`-reward 双写、reward `repo@rev` 解析均已落地（多数随 `00ec830`「Resolve reward cleanup decisions」），`media_type` 由 `artifact_format` 派生经复核**决策不放松**（保留 fail-loud gate，非目标）。这 2 个尾巴各需一个明确取舍，故拆出来、不阻塞 design_smell 归档。

## 1. `weight_sync_barrier` 改 derive-only（design_smell §4.1 P2）

- **surface**：`vrl/trainers/core/types.py:174` `weight_sync_barrier: str | None = field(default=None)`；`__post_init__`（:182-187）在 `None` 时按 mode 派生（continuous → `pause_admission_and_drain_inflight`，否则 `before_sync`）；`_validate_synchronous`（:200-203）/ `_validate_continuous`（:210-214）校验显式值。
- **smell**：运行时行为 100% 由 `mode` 决定，yaml 里已无人写它；它仍是可设 kwarg，只因测试用它构造 + "矛盾 override 时大声失败"的 guardrail。即"本可从 `mode` 派生的知识，仍暴露成一个可手填字段"。
- **proposal**：保留 `__post_init__` 的派生，但**停止把它当可设 kwarg**（`init=False` 或转校验后字段/property），测试改成构造后断言派生值。
- **risk / 为何 held**：[[SPRINT_resolved_struct_field_audit]] §9.1 已把它判为 KEEP/派生消费——即"派生 + 作为 guardrail 保留"本身是可接受状态。改 derive-only 是收紧 public 构造面、动多处测试的 kwarg 构造，**收益（少一个可手填字段）相对 churn 偏小**。先决策：真要收紧，还是按 §9.1"保留派生"定案（那样本项即 no-op done）。

## 2. release-flag 影子字段处置（design_smell §5.2）

- **surface**：`vrl/ray/resources.py:146` `rollout_release_before_reward_model: bool` + `:149` `reward_release_after_score: bool`，在 `ResolvedDistributedResources` 上作为 stored 字段（set 于 :368/:371）。
- **smell**：这两个扁平字段**只被日志格式化器读**（`format_distributed_resource_plan` :466-468）；真实行为走 `resolved.lifecycle.*`——`reward_runtime_resource_kwargs` 读 `resolved.lifecycle.reward.mode == "on_demand"`（:1026），handoff 走 `lifecycle.*`。即扁平字段在**影子化**拓扑派生的 `RayLifecyclePlan`。类 docstring（:117-121）现称它们为 "compatibility views derived beside this plan"。
- **取舍（两条路，需拍板）**：
  1. **删**：移除两个 stored 字段，把日志行 :466-468 改读 `resolved.lifecycle.*`（单一真相源）。
  2. **保留**：正式承认为 compatibility view，加注释钉死"仅供日志/兼容、行为不经它们"（现状即此）。
- **为何 held**：[[SPRINT_resolved_struct_field_audit]] §3 曾把它们标 NECESSARY；design_smell §5.2 疑为 LOGGING_ONLY。两份审计在 local-vs-stored 判据上有过冲突，需一次明确裁定。涉及 public resolved struct + 多处测试断言，动手前按 §3 做法逐项 grep 消费方 + 跑 `tests/ray/test_resources.py`。

## 3. 非目标

- 不重开 design_smell 已落地/已决策的项（`advantage_low` / `pool`-reward / `repo@rev` / `media_type`）。
- 不动 `resolved.lifecycle.*` 这条真实拓扑派生路径——本 sprint 只处置它的扁平影子字段。

## 相关

- [[SPRINT_design_smell_audit]]（`done/`，父 sprint）
- [[SPRINT_resolved_struct_field_audit]]（`done/`，§9.1 `weight_sync_barrier` KEEP、§3 release-flag NECESSARY/LOGGING_ONLY 冲突）
