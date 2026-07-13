# SPRINT: `FamilyCapability` 序列化字段调查（done，调查 → 删除）

> Superseded on 2026-07-12: the remaining three-field `FamilyCapability`
> duplicated registry identity and collector classification, so the whole seam
> was removed. The Ray launch contract now carries only the behavior-consumed
> `generation_kind` (`diffusion` or `ar`), while executor `family`/`task` are
> checked directly against the contract.

状态：done（2026-06-21）。调查结论：6 个「经 `to_dict()` 跨 Ray 边界但无 in-process 读者」的字段，逐一追踪序列化下游消费后**全部证实为真死**（序列化了但 worker / placement 侧从不按 key 读回）——遂全部删除。验证：`ruff` 全绿，`pytest tests/generation/test_capabilities.py tests/generation/execution/ tests/generation/ray/test_oom_split.py` **24 passed**，`to_dict`/`from_value` 往返 + `with_runtime_caps` 冒烟通过。
范围：`vrl/generation/capabilities.py` 的 `FamilyCapability` 5 字段 + `ExecutionStageCapability.metadata`。
来源：dead-dataclass-hunt 把 5 个 `FamilyCapability` 字段标为 display-only 死字段，但「机械可删」被证伪——必须先查序列化边界消费，见 §0。承接 [[SPRINT_segment_signal_dead_field_cleanup]]。

## 0. Core Decision —— 序列化边界消费追踪（调查产出）

自动审计称这些字段只被 `__post_init__` 自校验 + `to_dict()` 读取。但 `to_dict()` 输出跨 Ray 进程进 `runtime_extra["family_capability"]`（`launcher.py:250`）、planner summary（`planner.py:154`）、chunk placement summary（`chunk_placement.py:121`）。**是否死，取决于序列化后下游是否按这些 key 读回**——这是 grep 单点判不了的，必须逐 sink 追消费。

追踪结论：三个 sink 全是 **telemetry / logs 旁路**，无一按这些 key 做决策或重建：
- `worker.py` 用 `family_capability_from_value(...)` 把整个 capability 反序列化回来，但 merge 后只读 `.family`/`.task`/`.trajectory_kind` 做交叉校验——5 字段一个都不碰。
- `chunk_placement` 的 `capability_summary` 在 `worker.py:299` 被整体塞进 `metrics["capability"]`，纯 telemetry dump，placement 决策只用 `estimate_chunk_cost`（采样步数/样本数）+ worker 数。
- `planner.summary()["capability"]` 文档自述「lightweight, serializable plan summary for logs/metrics」。

> ⚠️ 同名陷阱（已规避）：`reward_views` 的真消费者全是 `TrajectoryBatch.reward_views`（活，不同 struct，在 `vrl/trajectory/`、`vrl/rollouts/collector/`），与 `FamilyCapability.reward_views` 无关。

## 1. 判定表（每字段 → in-process 读者？→ 序列化后消费？→ 判定）

| 字段 | in-process 类型读者 | 序列化后 worker/placement 消费 | 判定 |
|---|---|---|---|
| `FamilyCapability.trainable_segments` | 无（仅 `capabilities.py` def/validate/copy/emit/rebuild） | 无 | **死** |
| `FamilyCapability.reward_views` | 无（同名命中全是 `TrajectoryBatch.reward_views`） | 无 | **死** |
| `FamilyCapability.supports_batched_requests` | 无 | 无（无 `.attr` 读、无 key 读、无 runtime_caps producer） | **死** |
| `FamilyCapability.supports_batched_forward` | 无 | 无 | **死** |
| `FamilyCapability.cache_kinds` | 无 | 无 | **死** |
| `ExecutionStageCapability.metadata` | 仅 `planner.py:312` `dict(stage.metadata)` 拷进运行期 `ExecutionStage.metadata`，后者只再序列化（`planner.py:187`） | 无控制流消费 | **死** |

**活字段对照（勿连坐，均有真控制流分支）**：`supports_chunked_execution`（`chunks.py:122`）、
`supports_torch_compile`（后续 compile config cleanup 后 gate `model.torch_compile.enable`）、
`supports_reference_conditioning`（`registry.py:127`）——全部保留。

## 2. 落地（删除 + 同步序列化点）

- `capabilities.py`：删 `FamilyCapability` 5 字段 + `__post_init__` 三处 `require_string_tuple`（连带删 unused import）+ `to_dict` 五 key + `from_value` 五读 + `with_runtime_caps`（bool_fields 去 `supports_batched_requests`/`supports_batched_forward`、删 `cache_kinds` 块、nested update 去 `trainable_segments`/`reward_views`）；删 `ExecutionStageCapability.metadata` + 其 `to_dict`/`from_value`。
- `planner.py:312`：去掉 `metadata=dict(stage.metadata)` kwarg（运行期 `ExecutionStage.metadata` 保留默认 `{}`，与其它 4 个构造点一致；该字段属 planner 运行期结构，非本 sprint 范围，不在此动）。
- 构造模板：`vrl/models/ar/capabilities.py`（两模板）、`vrl/models/diffusion/capabilities.py` 去掉 `trainable_segments=`/`reward_views=`/`cache_kinds=` 字段赋值（模板的 `trainable_segment(s)` **参数**仍用于构造 `execution_stages`，保留）。
- 测试：`tests/generation/test_capabilities.py` 两处构造去掉已删字段 kwarg（往返测试仍成立）。

## 3. 验证（实锤）
- `grep` 确认 `trainable_segments`/`reward_views`/`supports_batched_*`/`cache_kinds` 在 `vrl/`、`tests/` 已无该 struct 的残留（仅剩模板函数参数与无关 `TrajectoryBatch.reward_views`）。
- `ruff` 全绿；`pytest tests/generation/test_capabilities.py tests/generation/execution/ tests/generation/ray/test_oom_split.py` 24 passed。
- 往返冒烟：`to_dict()` 不再吐 5 key；`from_value(to_dict())` 等价；`with_runtime_caps({...含已删 key})` 静默忽略（死旋钮移除后 backend 即便仍 emit 也变无害 no-op）。
- 注：`tests/generation/ray/test_rollout_launcher.py` 4 个 Ray actor 启动用例在**净 baseline 上同样 fail**（fresh venv 环境问题），与本改动无关。

## 4. Non-Goals（遵守）
- 不碰 `supports_chunked_execution`/`supports_torch_compile`/`supports_reference_conditioning`（活）。
- 不混淆 `FamilyCapability.reward_views` 与 `TrajectoryBatch.reward_views`。
- 不扩张到 planner 运行期 `ExecutionStage.metadata`（不同 struct，留给 planner 专项）。

## References
- `vrl/generation/capabilities.py`、`vrl/generation/execution/planner.py:154,187,312`
- `vrl/generation/ray/launcher.py:250,424`、`execution/chunk_placement.py:107,121`、`execution/worker.py:299`
- `vrl/models/ar/capabilities.py`、`vrl/models/diffusion/capabilities.py`、`tests/generation/test_capabilities.py`
- 活字段对照：`execution/chunks.py:122`、`launcher.py:424`、`vrl/rollouts/families/registry.py:127`
- 关联：[[SPRINT_segment_signal_dead_field_cleanup]]
