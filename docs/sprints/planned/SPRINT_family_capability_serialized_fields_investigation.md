# SPRINT: `FamilyCapability` 序列化字段调查（planned，调查类非删除类）

状态：未开始（2026-06-20）。**这是调查 sprint，不是删除 sprint。**
范围：判定 `vrl/generation/capabilities.py` 的 `FamilyCapability` / `ExecutionStageCapability` 上「无直接类型读者、但经 `to_dict()` 进 Ray 序列化」的字段，到底是死字段还是跨进程 capability 传输契约。
来源：dead-dataclass-hunt 把 5 个 `FamilyCapability` 字段标为 display-only 死字段，但**手动验证证伪了「机械可删」**——见 §0。承接 [[SPRINT_segment_signal_dead_field_cleanup]]。

## 0. Core Decision —— 自动审计高估了置信，这些字段不可机械删

自动审计称 `FamilyCapability.{trainable_segments, reward_views, supports_batched_requests, supports_batched_forward, cache_kinds}` 只被 `__post_init__` 自校验 + `to_dict()` 读取，判 display-only 死字段。**但 `to_dict()` 的输出是被消费的**，流向三个序列化 sink：

```python
# vrl/generation/ray/launcher.py:250
runtime_extra["family_capability"] = entry.capability.to_dict()
# vrl/generation/execution/planner.py:154
"capability": self.capability.to_dict(),
# vrl/generation/execution/chunk_placement.py:121
capability_summary = engine_plan.capability.to_dict()
```

这些字段因此**跨 Ray 进程边界**进了 `runtime_extra["family_capability"]` 和 chunk placement summary。是否「死」取决于 worker 侧 / placement 逻辑是否按这些 key 重建 capability —— 这是序列化边界消费问题，**不是 grep 能直接判的死字段**。盲删可能破坏 worker 侧 capability 重建。

> ⚠️ 另有同名陷阱：`reward_views` 的 grep 命中 `vrl/trajectory/validation.py:113,203` 是 **`TrajectoryBatch.reward_views`**（活，不同 struct），与 `FamilyCapability.reward_views` 无关——不能据此判活、也不能混为一谈。

## 1. 现状实锤

### 1.1 字段与读取面
`vrl/generation/capabilities.py` `FamilyCapability`（def `:124-131`）五字段的**仅有**直接读取：
- `__post_init__` 自校验：`:144-146` `require_string_tuple("FamilyCapability.trainable_segments"/"reward_views"/"cache_kinds", ...)`（能 raise，但只校验自身类型）。
- `to_dict()`：`:217-224` 把它们 `list(...)` 进序列化 dict。
- `with_runtime_caps()`：`:178-203` 拷贝。

无任何 `capability.trainable_segments` 之类的控制流按值读取（对比兄弟字段——`supports_chunked_execution`(`chunks.py:121`)、`supports_torch_compile`(`launcher.py:424`)、`supports_reference_conditioning`(`registry.py:127`) 都有真控制流分支，**是活的，勿连坐**）。

### 1.2 `ExecutionStageCapability.metadata`
`capabilities.py:72` 附近。序列化进 `capability_summary`，metadata dict 透传给 chunk。同属序列化边界问题。

## 2. 调查方案（不是删除）

### A. 追 `runtime_extra["family_capability"]` 的 worker 侧消费
- grep `family_capability` 全仓，定位 worker / launcher 反序列化点，确认是否按 `trainable_segments` / `cache_kinds` 等 key 读取重建 capability。
- 若 **worker 侧确实读** → 这些字段是**跨进程 capability 传输契约，活字段，保留**，并在字段处加注释说明「经 to_dict 跨 Ray 边界，勿因无本进程读者误判死」。

### B. 追 chunk placement summary 消费
- 确认 `chunk_placement.py:121` 的 `capability_summary` 是否按这些 key 做 placement 决策。

### C. 仅对「序列化后下游也不读」的字段才删
- 若 A/B 证明某字段序列化后 worker/placement 均不读 → 它才是真死，删字段 + 同步 `to_dict`/`__post_init__` 的 `require_string_tuple`/`bool_fields` 列表 + `with_runtime_caps`。
- 否则保留并注释。

## 3. 验证（调查产出）
- 产出一张表：每个字段 → worker/placement 是否反序列化消费 → 活/死 判定 + 证据。
- 仅删 C 确证为真死的字段；`pytest tests/generation/ -q` 全绿；Ray 多 worker capability 传输冒烟无回归（重点：worker 侧 capability 重建不缺 key）。

## 4. 非目标 / Non-Goals
- **不按 dead-dataclass-hunt 的 display-only 判定机械删** —— 已证伪「仅 to_dict 读取 = 死」。
- 不碰 `supports_chunked_execution` / `supports_torch_compile` / `supports_reference_conditioning`（活，有控制流分支）。
- 不混淆 `FamilyCapability.reward_views` 与 `TrajectoryBatch.reward_views`。

## References
- `vrl/generation/capabilities.py:72,124-131,144-146,178-203,217-224`
- `vrl/generation/ray/launcher.py:250`、`vrl/generation/execution/planner.py:154`、`vrl/generation/execution/chunk_placement.py:121`
- 活字段对照：`vrl/generation/execution/chunks.py:121`、`launcher.py:424`、`vrl/rollouts/families/registry.py:127`
- 关联：[[SPRINT_segment_signal_dead_field_cleanup]]
