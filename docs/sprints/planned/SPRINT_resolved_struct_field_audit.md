# SPRINT: `ResolvedDistributedResources` 等"派生型胖结构体"字段必要性审计（planned）

状态：proposed / planned。这是一次"派生结构体里的字段是否真有人消费、如何防止它继续膨胀"的审计。
findings + 路径 + 整改逻辑都在本文。按 P0→P3 分批做，每批独立 PR，互不依赖。

> 方法：用 1 个编排 workflow（23 个字段各 1 个审计 agent + 对每个"可删候选"再派 1 个**对抗式反驳
> agent** 去反证它其实被用 + 2 个并行扫描 agent 找同类胖结构体与测试改动面，共 29 个 agent）逐字段
> 真实 grep/read 源码。我随后独立复核了全部 4 个删除候选与广义扫描结论。**先验证再下结论**：
> `ResolvedDistributedResources` 对象在全仓**没有任何动态访问**（无 `asdict`/`fields()`/`getattr`/
> `replace`/`vars`/`pickle`/`OmegaConf` 序列化），全部是显式属性读 —— 这条是后续所有"死字段"判断
> 的前提：一个只在 test 里被读的字段就是**真死**，不会有反射路径偷偷用到它。

---

## 1. 核心结论 (TL;DR)

**`ResolvedDistributedResources`（`vrl/ray/resources.py:64`，23 字段）里 19 个字段被真实行为消费、必须
保留；只有 4 个有问题：2 个纯死字段（只被测试断言读）、2 个只进日志。真正该删的就 3 个，测试改动面
极小（1 个测试文件、7 行断言）。**

更重要的是广义结论：**问题不是"字段多"。** 用户感觉到的"一堆超大 dataclass/config"里，绝大多数是
**input_config（用户/spec 面向的 YAML 配置结构）**，宽是**合理的**，不该碰。真正会"静默堆积没人读的
预计算字段"的，只有**派生型结构体（resolved/derived：算一次、到处读）**。全仓这类胖结构体只有 2 个
出问题：`ResolvedDistributedResources`（本文主体）和 `FamilyCapability`（第二个 offender，见 §5）。

按动作拆分（23 字段）：

| 动作 | 数量 | 字段 |
|---|---|---|
| **KEEP / NECESSARY**（真实行为消费） | 19 | 见 §3 |
| **DELETE**（DEAD，零生产读取，仅测试断言） | 2 | `reward_num_gpus`、`total_gpu_slots` |
| **DELETE**（LOGGING_ONLY + 重复预计算会漂移） | 1 | `ray_total_bundles` |
| **KEEP + 注释**（LOGGING_ONLY 但承载不可派生的溯源信息） | 1 | `visible_devices`（删为可选项，倾向保留） |

**真正的靶子只有 3 个删除项**，对抗式反驳已逐个确认 `removable=true`、查无隐藏消费方。其余 19 个字段
**都有真实消费链**（多数经 `reward_runtime_resource_kwargs` 流入 reward Ray runtime，或经
`RayGenerationConfig.from_cfg` 流入 placement / 内存释放 / 跨节点校验），**不要碰**。

---

## 2. 判据 (the rubric)

对每个字段问：**它的 stored 值（构造后存在对象上的那份）被谁读？读了之后做什么？**

三档分类：

- **NECESSARY** — 被生产行为读：控制流分支、传给 Ray/runtime/config 的值、会 raise 的校验。
- **LOGGING_ONLY** — 唯一读者是 `format_distributed_resource_plan`（一条日志/报错字符串）和/或测试。
- **DEAD** — 唯一读者是测试，或根本没人读。

两条关键判据（决定了多个字段的归类）：

1. **resolver 内部的局部变量 ≠ stored field 必要。** `resolve_distributed_resources` 函数体里
   `reward_num_gpus = len(reward_devices)` 这种是**局部计算**，喂给校验/构造器。删掉 dataclass 上
   同名 stored 字段，局部变量照样工作。只有"构造完之后、从对象上读 `resolved.X`"的外部读者才证明
   stored 字段必要。
2. **`format_distributed_resource_plan` 不算 behavior。** 它只拼一条日志/报错串。一个字段若只被它
   读，是 LOGGING_ONLY，不是 NECESSARY —— 但日志有调试价值，是否删要看该字段能否从别处派生（见
   `visible_devices` vs `ray_total_bundles` 的不同处理）。

---

## 3. `ResolvedDistributedResources` 逐字段结论

| 字段 | 结论 | 决定性消费方（非日志、非测试、非 resolver 内部） |
|---|---|---|
| `visible_devices` | **LOGGING_ONLY**（KEEP+注释） | 仅 `resources.py:292` 日志；承载不可派生的"完整可见 GPU 池"溯源 |
| `trainer_devices` | NECESSARY | `placement.py:46` 按它建 trainer 预留 bundle；`trainer_torch_device()`→`online.py:161`/`train_dpo.py:144` 定 trainer device |
| `rollout_devices` | NECESSARY | `config.py:165` driver/rollout 重叠校验 raise；`launcher.py:261`→`validate_actor_gpu_ids` |
| `reward_devices` | NECESSARY | `resources.py:852` `expected_gpu_ids`→reward Ray runtime（`ray/runtime.py:99/110/116` 门控+校验） |
| `rollout_num_gpus` | NECESSARY | `launcher.py:295` 跨节点 preflight `if non_driver_gpus < needed: raise` |
| `rollout_num_workers` | NECESSARY | `config.py:74`→`RayGenerationConfig.num_workers`→`placement.py:51` 每 worker 一个 bundle |
| `rollout_gpus_per_worker` | NECESSARY | `launcher.py:249` 门控校验；`config.py:75`→bundle GPU 大小/actor num_gpus/offload |
| `reward_num_gpus` | **DEAD（删）** | 无。仅 `test_resources.py:359` 断言 |
| `reward_num_workers` | NECESSARY | `resources.py:846`→reward runtime `num_workers`（`ray/runtime.py:91` 建池、`:59` 分片） |
| `reward_gpus_per_worker` | NECESSARY | `resources.py:847`→reward runtime `gpus_per_worker`（`rewards/ray/runtime.py:93`） |
| `reward_shared_with_rollout` | NECESSARY | `runtime.py:132` `should_release_memory_before_reward()`→`collector/core.py:147` 实际释放显存 |
| `rollout_release_after_collect` | NECESSARY | `config.py:98`→`launcher.py:181` 选 Releasable vs 常驻 runtime |
| `rollout_release_before_reward_model` | NECESSARY | `config.py:99`→`runtime.py:129`→`collector/core.py:150` reward 前拆 rollout actor |
| `reward_release_after_score` | NECESSARY | `resources.py:850`→`release_after_call`→`ray/runtime.py:62` 每次打分后 shutdown actor 组 |
| `reward_placement_strategy` | NECESSARY | `resources.py:851`→`create_placement_group(strategy=)`（`ray/runtime.py:138` 真实调度） |
| `reward_cpus_per_worker` | NECESSARY | `resources.py:848`→Ray `num_cpus` 预留 + PG CPU bundle |
| `reward_max_inflight_batches` | NECESSARY | `resources.py:849`→reward actor 池并发上限（`ray/runtime.py:59`） |
| `reward_gpu_reservation_count` | NECESSARY | `resources.py:853`→pin GPU 槽位的 reservation actor 数（`ray/runtime.py:119/126/140/150`） |
| `total_gpu_slots` | **DEAD（删）** | 无，连日志都没有。仅 `test_resources.py:75` 断言 |
| `ray_total_bundles` | **LOGGING_ONLY（删）** | 仅 `resources.py:308` 日志；且重复了 `placement.py` 独立算的 bundle 数（漂移风险） |
| `requires_trainer_reservation` | NECESSARY | `placement.py:45` 门控是否给 PG 加 trainer 预留 bundle |
| `colocated` | NECESSARY | `config.py:82`→`allow_driver_gpu_overlap`；`runtime.py:140` `is_colocated()`→`schedule.py:205` 禁止共享 GPU 上连续 rollout |
| `cross_node` | NECESSARY | `launcher.py:68` preflight 门控；`:253/263`→`placement.py:79` 切换 node-aware 校验；`config.py:151` 跳过 driver 重叠校验 |

---

## 4. 必须处理项 (action items)

### P0 — 删 2 个纯死字段（DEAD，零风险）

这两个字段在 `resolve_distributed_resources` 里算出来、存进对象，但**全仓没有任何生产代码读它**，
连日志都没有，唯一读者是各 1 行测试断言。对抗式反驳确认 `removable=true`、查无隐藏/反射消费方。

**1. `reward_num_gpus`**（`vrl/ray/resources.py:74` 声明，`:252` 构造赋值）
- ⚠️ **只删 stored 字段，别动局部变量**：`resources.py:174` 的局部 `reward_num_gpus = len(reward_devices)`
  仍被 `:175`、`:186` 的校验和 `:183` 的 `_resolve_role_num_workers` 调用使用 —— **保留**。
  只删：dataclass 字段声明（`:74`）+ 构造器实参（`:252`）+ 测试断言（`test_resources.py:359`）。
- `reward_runtime_resource_kwargs` 用的是 `reward_num_workers`，从不读 `reward_num_gpus`。

**2. `total_gpu_slots`**（`vrl/ray/resources.py:85` 声明，`:263` 构造赋值，`:239` 局部计算）
- 局部 `:239`、构造 `:263`、声明 `:85` 全是为这个字段服务，没有外部读者，**整条都删**。
- 删测试断言（`test_resources.py:75`）。

### P0 — 删 `ray_total_bundles`（LOGGING_ONLY + 重复预计算会漂移）

`ray_total_bundles`（`:86` 声明，`:240-242` 局部计算 = `rollout_num_workers + reward_num_workers +
trainer 预留数`，`:264` 构造，`:308` 日志）唯一读者是 `format_distributed_resource_plan` 的日志串。

它**该删**而非"保留+注释"，原因有二，正是 AGENTS.md 警告的"手写副本随源漂移而静默腐烂"：

1. **真实的 PG bundle 数是 `placement.py` 独立算的**（`create_generation_placement_group`：按
   `requires_trainer_reservation`+`trainer_devices` 建预留 bundle、按 `range(config.num_workers)`
   建 rollout bundle；reward 走完全独立的 reward-runtime PG）。`ray_total_bundles` 从不参与建任何
   真实 PG —— 它只是把这套数学在 resolver 里又抄了一遍。`placement.py` 的 bundle 逻辑一改，这个被
   打进日志的数就和真实 PG 静默对不上。
2. **日志里已有更细的分项**：`format_distributed_resource_plan` 已经分别打了 `rollout_workers`、
   `reward_workers`、`trainer_reservation`，三者之和就是 bundle 数。`ray_bundles=` 是冗余汇总。

删：字段声明（`:86`）+ 局部 `:240-242` + 构造 `:264` + 日志行 `:308` + 5 行测试断言
（`test_resources.py:76/129/202/225/269`）。

> 注意：删 `ray_total_bundles` 的局部计算会移除一处对 `reward_num_workers` 局部值的引用，但
> `reward_num_workers` stored 字段经 `reward_runtime_resource_kwargs` 独立必要，**不受影响**。

**P0 总测试影响**（扫描 agent 实测）：**只 1 个文件** `tests/ray/test_resources.py`，删 7 行断言
（75、76、129、202、225、269、359），**零测试函数删除、零重写**（每个宿主测试都还保留其它有效断言）；
另改 1 行生产日志（`resources.py:308`）。改完跑 `pytest tests/ray/test_resources.py` 验证零回归。

### P1 — `visible_devices`：判断（默认 KEEP+注释，删为可选）

`visible_devices`（`:67` 声明，`:245` 构造，`:292` 日志）也是 LOGGING_ONLY（测试里那 6 处
`test_runtime_inputs.py` 和 `test_resources.py:134` 都是**输入配置覆盖串 / 报错 match 正则**，不是读
stored 字段），但它和 `ray_total_bundles` **本质不同，倾向保留**：

- 它承载的是**整个解析的输入快照 —— 完整可见 GPU 池**，这份信息**无法从其它字段派生**（可见池可能
  比 trainer+rollout+reward 的并集大，多出来的是 spare GPU）。删了就丢掉日志里最有用的那行调试信息：
  "机器一共可见哪些 GPU，各角色分别拿了哪些"。
- 它**不是重复预计算**，没有漂移风险（不像 `ray_total_bundles`）。

**建议**：保留，并在字段上加一行注释标明它是 **display/provenance-only**（仅供
`format_distributed_resource_plan` 打印输入池，无行为消费方），让后人不会误以为有人按它分配。
这与 §6 的防腐约定一致：日志-only 字段允许存在，但必须显式标注。

若坚持极致精简，可删字段 + 日志行 `:292`（无测试断言挡路）—— 但这是**可选项**，不是 P0。

---

## 5. 广义模式：其它胖结构体（用户"一堆超大 dataclass"的真实版图）

并行扫描 agent 扫了 `vrl/` 下所有 ≥14 字段的 dataclass/config，结论：**"字段多"绝大多数是合理的，
真正同类风险的只有 1 个。**

**第二个 offender（需逐 flag 复核后再动）：**

- **`FamilyCapability`**（`vrl/generation/capabilities.py:116`，23 字段，**resolved_derived**）——
  和 `ResolvedDistributedResources` 同种风险，且死字段密度更高。扫描 agent 报告多个 `supports_*` 旗标
  被设置/序列化但**没有任何决策消费方**：`supports_prefill_decode_split`（全仓 0 引用）、
  `supports_cuda_graph` / `supports_resident_rollout_state`（0 读）、`supports_kv_decode`（仅定义）、
  `supports_batched_requests/forward`（仅在 `batch_signature` 内部）、`supports_token_logprobs`
  （AR 上设 true 但从不分支）。
  ⚠️ **这是 scan 级结论，置信度低于 §3**：未经本文对 `ResolvedDistributedResources` 那种逐字段
  对抗式反驳。**列为 follow-up**：用同一套 workflow 方法（审计→反驳→测试影响）逐 flag 确认后再删，
  不要凭这张表直接动手。

**其余 14 个宽结构体 = 合理，明确不碰**（按 kind 分桶）：

- **input_config（用户/spec 面向，宽是本分）**：`TrainerConfig`(26)、`NextStep1Config`(19)、
  `JanusProConfig`(18)、`OfflineDPOTrainerConfig`(14)、`VideoGenerationRequest`(14)。每字段都有
  builder/trainer 消费，且多数已按项目约定显式 `field(default=...)` 逐行拼写（见
  [[feedback_explicit_field_spelling]]）。**宽 ≠ 病**。
- **working-state（per-rollout 私有工作状态，刻意做胖以免泄漏进引擎契约）**：
  `CosmosPredict25SamplingState`(17)、`CosmosPredict2SamplingState`(16)、`SegmentSignal`(15)、
  `NextStep1ARChunkResult`(14)、`DiffusionChunkResult`(14)。docstring 明确禁止 collector 内省。
- **DTO / live-handle / plugin-table**：`OnlineRecipeStack`(14，活对象句柄)、
  `OnlineRecipeDefinition`(15，每家族稀疏的 Optional 回调表)、`RayActorMethodRuntime`(19，含 3 个
  `init=False` 内部态)、`RewardInferenceResult`(15)、`TrainStepMetrics`(21，每字段都被 CSV
  logger / precision_guard 消费)。
- 这些里有零星**观测-only 字段**（`peak_memory_mb`、`stage_durations`、`engine_counters`、reward 的
  timing 诊断）——和 `visible_devices` 同类，**保留+可选加注释**即可，不值得单独发 PR。

---

## 6. 系统性防腐（P3，根因）

死字段能堆到 23 字段里有 2 个全死、2 个只进日志，根因是**没有任何机制对"加了字段却没人消费"形成
反压**。两个层次的修法：

1. **约定（首选，零风险）**：在 `AGENTS.md` 的"架构卫生（Architecture Hygiene）"补一条，与已有的
   ALL_CAPS 派生条款对称：
   > 派生/解析型结构体（"算一次、到处读"，如 `Resolved*` / `*Capability`）的每个字段都必须有
   > **非日志的消费方**；只供 `format_*`/日志打印的字段必须在定义处显式注释为 `display/provenance-only`。
   > 既不被行为消费、又不是显式 display-only 的字段视为死字段，应删除。
2. **守卫测试（可选）**：给 `ResolvedDistributedResources`（和确认后的 `FamilyCapability`）加一个
   轻量单测：断言每个 `fields(...)` 的字段名在 `vrl/`（排除 `resources.py` 自身 + 一个显式
   `_DISPLAY_ONLY` 白名单）里至少被引用一次。能挡住未来无声新增死字段。
   ⚠️ 但 grep 式守卫偏脆（重命名/间接访问会误报），且本仓无动态访问、字段不多，**约定 (1) 已基本够用**；
   守卫测试仅在 review 中反复出现死字段时再上。

---

## 7. 非目标（明确不做）

- **不把 `ResolvedDistributedResources` 拆成嵌套子结构体**（rollout/reward/topology 分组）。当前
  全部消费方都是直读 `resources.<field>`、零动态访问，flat shape 的 **grep 一致性**正是它的价值；
  分组要改所有 call site，是为口味而搅动正确架构 —— 违反 [[feedback_no_big_refactors]] /
  [[feedback_consistency_over_cleanup]]。本 sprint 只删确证的死字段，不重组。
- **不碰 19 个 NECESSARY 字段**，也不碰它们流经的 `reward_runtime_resource_kwargs` /
  `format_distributed_resource_plan` 形状（除删 `ray_total_bundles` 那一行日志）。
- **不碰 14 个健康的宽结构体**（input_config / working-state / DTO）。宽不是病。
- **不凭 §5 的 scan 表直接删 `FamilyCapability` 字段** —— 先逐 flag 对抗式复核。
- **不为清理而清理**：只动"零消费方"或"重复预计算会漂移"的字段，其余视为正确。

---

## 8. 校验记录（对自动审计的独立复核）

- **方法可信度**：每个删除候选都过了"审计 agent 判 + 反驳 agent 反证"两道；4 个候选
  （`reward_num_gpus`/`total_gpu_slots`/`ray_total_bundles`/`visible_devices`）反驳均
  `removable=true`、`hiddenConsumerFound=null`。前提是全仓对该对象**无动态访问**，已由多个 agent
  独立确认（`asdict`/`fields()`/`getattr`/`replace`/`vars`/`pickle`/`OmegaConf` 命中的都是别的
  dataclass，如 `runtime_build`、reward artifacts、`DPOStepMetrics`）。
- **修正一处**：初判曾记 `total_gpu_slots` 在 `test_resources.py:75,127` 两处被断言；复核确认 **:127
  实际是 `assert resolved.colocated is True`**，`total_gpu_slots` 只有 :75 一处。不影响"DEAD、可删"结论。
- **`reward_num_gpus` 的局部 vs stored 区分已复核**：局部变量参与真实校验（`:175/:186`），故 P0 明确
  "只删 stored 字段 + 构造实参 + 测试，保留局部计算"。
- **`FamilyCapability` 标注为低置信度 follow-up**：扫描 agent 的"0 引用"未经逐 flag 反驳，不直接执行。

---

## 关键文件引用

- `vrl/ray/resources.py:64-90`（`ResolvedDistributedResources` 定义），`:95-268`（resolver），
  `:284-312`（`format_distributed_resource_plan`），`:840-854`（`reward_runtime_resource_kwargs`）
- 删除目标行：`reward_num_gpus`（`:74`,`:252`）、`total_gpu_slots`（`:85`,`:239`,`:263`）、
  `ray_total_bundles`（`:86`,`:240-242`,`:264`,`:308`）；`visible_devices` 倾向保留（`:67`,`:245`,`:292`）
- 生产消费方：`vrl/generation/ray/config.py:74-99,151,165-185`、`vrl/generation/ray/placement.py:45-51`、
  `vrl/generation/ray/launcher.py:68-69,181,249-263,295-296`、`vrl/generation/ray/runtime.py:129-140`、
  `vrl/rollouts/collector/core.py:147-206`、`vrl/rollouts/orchestration/continuous/schedule.py:205`、
  `vrl/rewards/ray/runtime.py:91-101`、`vrl/ray/runtime.py:59-150`、`vrl/scripts/common/factory.py:134-140`、
  `vrl/scripts/common/online.py:159-161`、`vrl/scripts/diffusion/wan_2_1/train_dpo.py:142-144`
- 测试改动面：`tests/ray/test_resources.py:75,76,129,202,225,269,359`（删 7 行断言）
- 第二个 offender（follow-up）：`vrl/generation/capabilities.py:116`（`FamilyCapability`）
- 防腐约定落点：`AGENTS.md` → "Architecture Hygiene"
