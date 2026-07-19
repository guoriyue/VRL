# SPRINT: 简化 colocation / release 公有配置

> **Historical correction (2026-07-13).** The internal
> `RayGenerationConfig.allow_driver_gpu_overlap` compatibility mirror discussed
> below was later deleted. `RayGenerationConfig.resources` is now required and
> current launcher/runtime code reads `ResolvedDistributedResources.colocated`
> directly.

状态：**done（已落地 main，94dec03 "Simplify colocation/release public config surface"；2026-06-17 归档至 done/）**。落点：
- **P2 解析/校验**：`vrl/ray/resources.py` — `_parse_colocate_with_trainer`（块在=resident colocation，`memory_fraction` 必填且校验 (0,1]）、`_reject_removed_distributed_keys`（5 个旧 public key 硬报错并指向新形状）；`_resolve_rollout_devices` 新增 `colocate_with_trainer` 强制 placement（auto→trainer GPU，显式 disjoint→报错），colocate 隐含 overlap 许可。删 `release_after_collect/release_before_reward_model/persistent_colocated_workers/gpu_memory_fraction` 与 `distributed.reward.release_after_score` 的 public 解析；内部派生 / 兼容视图字段 / runtime lease 零改动。
- **public schema**：`vrl/config/schema.py` — `distributed.rollout` 不再以 `RayGenerationConfig` 作 public surface，改为显式 known-key 列表 + `colocate_with_trainer:{memory_fraction}` 子块；`distributed.reward` 去掉 `release_after_score`。
- **错误信息**：`vrl/generation/ray/config.py` + `resources.py` 中指向旧 key 的提示改指 `colocate_with_trainer` / “derived automatically”。
- **迁移**：`configs/experiment/diffusion/sd3_5/online_grpo_ocr_single_gpu_async_debug.yaml` 改用 `colocate_with_trainer`；`tests/{ray/test_resources,generation/ray/test_runtime_config,rewards/ray/test_resource_lifecycle,ray/test_global_placement,scripts/test_common_factory,config/test_load_all_experiments}.py` 迁移 + 新增 rejection/placement 锁测。
- **验收**：`python -m vrl.config.lint` 两扫全绿；上述测试 + `tests/ray tests/generation/ray tests/config tests/rewards tests/rollouts` 全通过；ruff 全过。（无关的既有失败 `tests/scripts/test_online_precision_bridge.py::...logprob_mismatch_metrics` 与本改动无关，stash 本 diff 仍复现。）
- **取舍**：删除两个语义已不可表达的测试（`persistent + release_after_collect=true`、`reward_release_after_score=false` 强制多 reward 拒绝），其覆盖已由派生路径锁测替代。`DistributedResourceConfig` 的 `rollout_release_*/reward_release_after_score` 输入字段保留为内部表示（始终 None→派生），按 §5 不动。

---

原始设计（状态：planned / 设计）：

本 sprint 兑现 `SPRINT_ray_phase_lifecycle_plan.md`（done）刻意推迟的 **P4「Public config 兼容」**——
那个 sprint 已经把「何时 release」从 runtime 收成内部 `RayLifecyclePlan`，但公有 YAML 字段一个没动。
现在动它。

## 0. 结论

当前 `distributed.rollout` 把**「单卡共享 + 常驻」这一个意图**拆成了 3-4 个互相牵制的开关，用户看不懂、也没必要懂。本 sprint 做两件事：

1. **删**：`distributed.rollout.release_after_collect` / `distributed.rollout.release_before_reward_model` / `distributed.reward.release_after_score` 三个 schedule 开关从公有 YAML 移除——引擎已经在 `RayLifecyclePlan` 里从 GPU 拓扑派生，用户写它们没有价值（且唯一"有意思"的写法会被 validation 拒绝）。
2. **收口**：`distributed.rollout.persistent_colocated_workers` + `distributed.rollout.gpu_memory_fraction` 收成**一个自解释的 `colocate_with_trainer` 块**；`distributed.resources.allow_overlap` 先按 P1 代码证据决定是否继续保留为资源安全闸。

划线原则：**「什么时候 release」是引擎的事（已经是了）；「要不要共卡 + 显存怎么切」是用户意图，但要让用户一眼看懂。** 这条线和 slime / cosmos-rl 划在同一处（§3）。

## 1. 当前公有配置事实（"太复杂"复现在哪）

用户在 YAML 里写的（真实例子，`configs/experiment/diffusion/sd3_5/online_grpo_ocr_single_gpu_async_debug.yaml:27-37`）：

```yaml
distributed:
  rollout:
    release_after_collect: false        # 开关 1
    persistent_colocated_workers: true  # 开关 2
    gpu_memory_fraction: 0.45           # 开关 3
```

这三行只表达**一个**意思——「rollout 跟 trainer 共一张卡，且不要每周期让位」。复杂度来自三个**互相牵制的隐藏约束**，全在校验里、不在配置面上：

```text
persistent_colocated_workers=true  要求  allow_driver_gpu_overlap=true   (config.py:60-63)
persistent_colocated_workers=true  要求  release_after_collect=false     (config.py:64-67)
persistent_colocated_workers=true  要求  显式 gpu_memory_fraction        (resources.py:301-305)
```

公有解析入口同样散（`vrl/ray/resources.py:530-544`）：

```python
rollout_release_after_collect=_parse_optional_bool(
    cfg_get(rollout_runtime, "release_after_collect", None)),
rollout_release_before_reward_model=_parse_optional_bool(
    cfg_get(rollout_runtime, "release_before_reward_model", None)),
rollout_persistent_colocated_workers=bool(
    cfg_get(rollout_runtime, "persistent_colocated_workers", False)),
rollout_gpu_memory_fraction=_parse_optional_float(
    cfg_get(rollout_runtime, "gpu_memory_fraction", None)),
reward_release_after_score=_parse_optional_bool(
    cfg_get(reward_runtime, "release_after_score", None)),
```

问题两条：

1. **一个意图、多个开关 + 隐藏互斥**：要表达"共卡常驻"得记住"设 persistent、还要把 release 设成 false、还要给 fraction、还要开 overlap"。任一漏设就报一条看起来不相关的错。
2. **schedule 开关根本是冗余的**：`release_after_*` 是 tri-state，默认 `None` = 按拓扑派生（`resources.py:51-54`）。普通用户本来就不该写；而显式值若与拓扑矛盾（共卡却 `release=false` 等）会被 validation 直接拒（`resources.py:284-288`、`config.py:181-185`）——也就是说这个 override 连唯一有意义的用法都没有。

## 2. 拆成两类：哪些删、哪些留

| 公有 YAML 字段 | 类别 | 处置 |
|---|---|---|
| `rollout.release_after_collect` | schedule（可派生） | **删**，引擎从拓扑派生 |
| `rollout.release_before_reward_model` | schedule（可派生） | **删**，引擎从拓扑派生 |
| `distributed.reward.release_after_score` | schedule（可派生） | **删**，引擎从拓扑派生 |
| `rollout.persistent_colocated_workers` | 意图（共卡常驻） | **收进 `colocate_with_trainer` 块** |
| `rollout.gpu_memory_fraction` | 预算（显存切分） | **收进 `colocate_with_trainer` 块** |
| `distributed.resources.allow_overlap` / `allow_driver_gpu_overlap` | 资源安全闸 / 内部派生 | **P1 先定边界；本 sprint 不默认并掉** |
| `distributed.resources.reward.share_with_rollout` | 意图（reward 放哪） | **保留不动**（已是干净单 tri-state，不属本次"复杂"家族） |

判据：能从 **GPU 拓扑唯一推出**的 → 引擎派生，不暴露给用户；属于**用户意图或显存预算、引擎推不出**的 → 保留，但收成一个自解释的块。

## 3. 对齐 slime / cosmos-rl

这两家也是同一条线（依据见 reading 笔记）：

| | schedule（引擎派生） | 意图 + 显存预算（用户给） |
|---|---|---|
| **slime** | `needs_offload` 从 GPU range 重叠**自动算**："only server groups whose GPU range overlaps Megatron's get `needs_offload=True`"（`slime.md` 引 `rollout.py:1026-1027`） | 用户传 `--colocate` / `--offload-*` 选模式 + 手填 fractional GPU（train `0.4` / engine `0.2`，`placement_group.py:111-119`、`rollout.py:99`） |
| **cosmos-rl** | rollout 引擎**直接不 sleep**（`enable_sleep_mode=False  # could corrupt the cuda allocator`，`vllm_rollout.py:267`），用常驻+静态显存帽把"何时 release"用设计删掉 | 用户在 config 里选 `mode: disaggregated/colocated/colocated_separated`（`launch_all.py:510-548`）+ 填 `gpu_memory_utilization`（`vllm_rollout.py:270`） |

结论印证：**三家都没让引擎决定"要不要共卡 / 显存怎么切"，但也都没让用户手写"何时 release"。** VRL 现在的毛病不是暴露了第二类，而是同时暴露了第一类、且把第二类拆碎了。

## 4. 目标公有配置

**只留一个块表达"rollout 常驻在 trainer 卡上"这个意图**，块在 = 强制 rollout 与 trainer 共卡且常驻；块不在 = 引擎按拓扑自己定（多卡 disjoint→resident；单卡共享→默认 release-after-collect）：

```yaml
distributed:
  resources:
    reward:
      share_with_rollout: null  # 不变：reward 放哪是真实意图，tri-state
  rollout:
    # 块存在 = 把 rollout worker 常驻在 trainer 的 GPU 上（tiny-debug 共卡）。
    # 块不存在 = 引擎按 GPU 拓扑自动决定 placement 与 release，无需任何开关。
    colocate_with_trainer:
      memory_fraction: 0.45   # 必填：常驻 rollout 在共享卡上的显存上限 (0,1]，trainer 拿剩下的
```

新旧映射：

```text
旧 3 行（release_after_collect=false + persistent_colocated_workers=true + gpu_memory_fraction=0.45）
  -> 新 1 块  colocate_with_trainer: { memory_fraction: 0.45 }

旧 release_after_collect / release_before_reward_model / distributed.reward.release_after_score（任意值）
  -> 删除；引擎派生（None 行为不变，矛盾值不再可写）

无 colocate_with_trainer 块
  -> 引擎按拓扑：disjoint=resident，shared=release-after-collect（= 今天 None 的行为）
```

设计取舍：

- **用"块的存在"而不是一个 `colocate: bool` + 散落的 fraction**——意图和它唯一需要的预算（`memory_fraction`）放在一起，消灭"设了 true 却忘了 fraction"这类隐藏约束。`memory_fraction` 成为块内**必填**（缺了直接报错，错误信息就在块里）。
- **块必须决定 placement，不只是决定 lifecycle flag**：今天 `allow_overlap=true` 只允许 `_slice_pool_with_overlap_fallback` 在非重叠 GPU 不够时回退到 trainer GPU；如果机器有 spare GPU，auto rollout 仍会拿 spare GPU（`resources.py:672-698`、`:719-729`）。因此 `colocate_with_trainer` 存在时必须强制 rollout devices 与 trainer devices 重叠：
  - `rollout.devices=auto` / `rollout.num_gpus=auto|1`：解析/resolve 后把 rollout 放到 trainer devices（tiny debug 单卡语义）。
  - `rollout.devices` 显式且与 trainer disjoint：直接报错，不能把"colocate"静默解释成 split-GPU resident。
  - `training.strategy=fsdp`：仍按现有规则拒绝 trainer/rollout overlap，本块只支持 single_process tiny debug。
- **`allow_overlap` 先不默认并入块**：`distributed.resources.allow_overlap` 现在还承担普通单卡 on-demand release 路径、reward/trainer overlap 校验、auto fallback 是否允许重叠等职责（`resources.py:196-220`、`:259-264`、`:701-729`）。本 sprint 可以让 `colocate_with_trainer` 内部隐含它所需的 overlap 许可，但除非 P1 用代码证据证明没有非 colocate 用途，否则不要删除 `distributed.resources.allow_overlap` 这个资源安全闸。
- **`allow_driver_gpu_overlap` 继续是内部 resolved 结果**：它由 `RayGenerationConfig.from_cfg` 从 `resources.colocated` 派生（`config.py:96`），不再作为 public YAML 入口讨论。
- **不要误删 per-reward runtime knob**：本 sprint 删除的是 `distributed.reward.release_after_score`。`reward.kwargs.<name>.release_after_score` 属于单个 reward runtime 的 kwargs（`rewards/ray/runtime.py:107`），不是这个 public resource schedule，不能在本 sprint 里扫掉。

## 5. 内部不变量（什么绝不动）

- `RayLifecyclePlan` / `ActorLeasePolicy` / `PhaseHandoffPolicy` 仍是唯一权威派生（`resources.py:69-110`），本 sprint 不改派生规则，只改**喂给它的公有入口**。
- `ResolvedDistributedResources` 上的 `rollout_release_after_collect` 等**兼容视图字段保留**（lifecycle sprint 的产物，launcher/collector 仍读它/读 plan）。本 sprint 删的是**公有 YAML → 这些字段的 user 入口**，不是字段本身。
- `RayGenerationConfig.release_after_collect`（`config.py:35`）等内部 knob 保留：launcher 在 `resources is None`（手搭测试配置）时仍回落到它（`launcher.py:190-194`）。它们由 resolved resources 喂入，不再由 public YAML 喂入。
- runtime 工厂 `with_release_after_collect` / 私有属性 `_release_after_collect`（`runtime.py:62,58`）不动——改名会破坏 API，lifecycle sprint 已论证过。

一句话：**本 sprint 只缩 public surface，不碰已经收好的内部派生与 runtime。**

## 6. 实施计划

### P0. 行为锁定测试

先锁住"派生结果"在新旧配置下一致，再改解析：

```text
单卡共享 + 旧三开关  -> rollout.mode=on_demand=false? 即 resident, fraction 生效
单卡共享 + 新 colocate 块  -> 解析出与上面完全相同的 ResolvedDistributedResources / RayLifecyclePlan
多卡 + colocate 块 + rollout auto -> rollout 被强制放到 trainer devices，而不是拿 spare GPU
多卡 + colocate 块 + rollout 显式 disjoint -> 报错指向 colocate_with_trainer 与 devices 矛盾
多卡 disjoint + 无块  -> resident no-release（不受影响）
单卡共享 + 无块  -> release-after-collect（= 今天 None 的派生）
删掉的 release_* key 出现  -> 报错指向"已自动派生，请删除"
缺 memory_fraction 的 colocate 块  -> 报错就在块里
```

覆盖文件：`tests/ray/test_resources.py`、`tests/generation/ray/test_runtime_config.py`、`tests/config/test_load_all_experiments.py`、`tests/config/test_unknown_keys.py`。

### P1. 先回答 overlap 耦合问题（§4 ⚠️）

读 `allow_overlap` / `allow_driver_gpu_overlap` 的全部 call site，确认是否有非 colocate 用途，定下"块隐含 overlap"能并到哪一步。**这一步是设计闸，先于写解析。**

当前代码证据已经提示 `allow_overlap` 不是纯 persistent-colocate 附属开关：

```text
resources.py:196-220   rollout auto 分配是否允许 fallback 到 trainer devices
resources.py:259-264   reward 与 trainer overlap 的安全闸
resources.py:701-729   _slice_pool_with_overlap_fallback 的通用 fallback 闸
config.py:181-188      allow_overlap=true + release_after_collect=true 是普通单卡 on-demand debug 路径
```

默认决策：**保留 `distributed.resources.allow_overlap`，但 `colocate_with_trainer` 内部可以设置 resolver 所需的 overlap 许可**。只有 P1 证明这些用途已被其他显式意图完整覆盖时，才把它进一步收进块。

### P2. 加新块解析 + 删旧 schedule key

`vrl/ray/resources.py` 的 from-cfg / resolve 路径（:500-544 + placement resolve）：

```text
新增   解析 distributed.rollout.colocate_with_trainer -> persistent + gpu_memory_fraction + force trainer/rollout overlap（内部仍用旧字段承接，§5）
新增   colocate_with_trainer 存在且 rollout explicit devices 与 trainer disjoint -> ValueError 指向 colocate_with_trainer 与 devices 矛盾
删除   解析 release_after_collect / release_before_reward_model / distributed.reward.release_after_score
新增   见到这三个旧 key -> ValueError："release schedule is now derived from GPU topology; remove this key"
新增   见到旧 persistent_colocated_workers / gpu_memory_fraction（旧位置）-> ValueError 指向 colocate_with_trainer
```

同时更新 public schema / unknown-key 注册边界：

```text
新增   public-facing distributed.rollout key registry，包含 launch/runtime 仍需 public 的字段 + colocate_with_trainer
避免   继续用 RayGenerationConfig 直接作为 distributed.rollout 的 public schema source（它保留内部 release_* fallback knobs）
新增   tests/config/test_unknown_keys.py 覆盖新 key 已知、旧 key 不再被当成有效 public key
```

旧 key 用**硬报错而非静默兼容**：保留两套写法正是"太复杂"的根因，且本仓库作者可控、配置量小，一次切干净优于长期 deprecation 窗口（符合 AGENTS.md "fix root causes, not band-aids"）。

### P3. 迁移在用的配置 + 测试

```text
configs/experiment/diffusion/sd3_5/online_grpo_ocr_single_gpu_async_debug.yaml:27-37  -> 改成 colocate_with_trainer 块
tests/config/test_load_all_experiments.py:308  -> 改成断言派生出的 lifecycle/resolved 字段，而非断言已删的 public key
扫 configs/ 下其余 distributed.rollout.{release_after_collect,persistent_colocated_workers,gpu_memory_fraction} 一并迁移
扫 configs/tests 下 distributed.reward.release_after_score；只迁移 distributed 层，不动 reward.kwargs.<name>.release_after_score
```

不动 `docs/runs/**/resolved_config.yaml`——那是历史 run 的归档输出（one-shot 产物），不是活配置。

## 7. 非目标

```text
不改 RayLifecyclePlan 的派生规则（只改它的公有入口）
不删/改 ResolvedDistributedResources 的兼容视图字段
不改 runtime 的 with_release_after_collect / release() / lease 行为
不动 distributed.resources.reward.share_with_rollout（已是干净意图开关）
不动 reward.kwargs.<name>.release_after_score（per-reward runtime kwargs，不是 distributed resource schedule）
不默认删除 distributed.resources.allow_overlap（除非 P1 证明所有非 colocate 用途已有替代）
不引入 sleeping/offload（仍是 lifecycle sprint 划走的独立后续）
不为旧 key 保留静默兼容层（硬报错引导迁移）
```

## 8. 验收标准

```text
1. 公有 YAML 的 distributed.rollout 下不再暴露 release_after_collect / release_before_reward_model / persistent_colocated_workers / gpu_memory_fraction；共卡常驻只通过 colocate_with_trainer 表达（含 distributed.resources.reward.share_with_rollout 不变）。
2. 旧 colocate 三连（release_after_collect=false + persistent + fraction）与新 colocate_with_trainer 块解析出完全相同的 ResolvedDistributedResources + RayLifecyclePlan（P0 锁测证明）。
3. colocate_with_trainer 块存在时，rollout placement 必须与 trainer devices 重叠；auto 走 trainer devices，显式 disjoint 报错。
4. 无块时，disjoint/shared 两种拓扑的派生行为与今天 None 完全一致。
5. 删掉的三个 distributed release_* key、或旧位置的 persistent/fraction 出现在 YAML 时，报错信息直接指向新形状或"已自动派生"。
6. colocate_with_trainer 块缺 memory_fraction 时，报错就在块里、不再借 persistent 的间接校验。
7. public schema / unknown-key sweep 与新形状一致：新 key 已知，旧 public key 不再作为有效入口注册。
8. 全部在用 configs/tests 迁移到新形状，套件全绿（环境缺失 skip 不计）。
9. 内部派生、runtime lease、兼容视图字段零改动（diff 只在 public 解析 + 校验 + configs/tests）。
```

## 9. 参考

仓库代码：

- `vrl/ray/resources.py:50-66`（`DistributedResourceConfig` 公有字段）、`:284-305`（互斥校验）、`:307-360`（派生 + `RayLifecyclePlan`）、`:500-544`（from-cfg 解析）
- `vrl/generation/ray/config.py:29,35-40,60-67,181-185`（`RayGenerationConfig` knob + 隐藏约束）
- `vrl/config/schema.py:376-399`（public config known-key 注册；不能继续把内部 `RayGenerationConfig` 当完整 public surface）
- `tests/config/test_unknown_keys.py`（whole-tree unknown-key sweep；新旧 key 迁移必须覆盖）
- `vrl/rewards/ray/runtime.py:95-108`（per-reward `release_after_score` kwargs，非本 sprint 删除对象）
- `vrl/generation/ray/launcher.py:185-201`（plan 优先、config flag 回落）
- `vrl/generation/ray/runtime.py:58,62,125`（工厂/属性/`release()`，不动）
- `configs/experiment/diffusion/sd3_5/online_grpo_ocr_single_gpu_async_debug.yaml:27-37`（待迁移的真实例子）
- `tests/config/test_load_all_experiments.py:308`、`tests/ray/test_resources.py`、`tests/generation/ray/test_runtime_config.py`

相关 sprint：

- `docs/sprints/done/SPRINT_ray_phase_lifecycle_plan.md`（本 sprint 是其刻意推迟的 P4）
- `docs/sprints/done/SPRINT_compile_rollout_lifecycle.md`（resident/on_demand 三态的下游消费者；本 sprint 不改三态语义，只改它们的公有命名入口）

reading：

- `docs/sprints/reading/slime.md`（`needs_offload` 派生、`--colocate` + fractional GPU）
- `docs/sprints/reading/cosmos-rl.md`（`mode` + `gpu_memory_utilization`、`enable_sleep_mode=False`）
