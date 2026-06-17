# SPRINT: `ResolvedDistributedResources` 等"派生型胖结构体"字段必要性审计（planned）

状态：**部分完成 —— 关账复核 2026-06-16**。这是一次"派生结构体里的字段是否真有人消费、如何防止它继续膨胀"的审计。

> **已落地**：**P0** 三个死字段（`reward_num_gpus` stored 字段 / `total_gpu_slots` / `ray_total_bundles`）+ 7 行测试断言 + `ray_bundles=` 日志行已由 `eb5d421`「Remove redundant resource plan fields」删除——`reward_num_gpus` **局部变量**按 §4 设计保留（`vrl/ray/resources.py:238/240/248/251`），`tests/ray/test_resources.py` 44 passed。**P1** `visible_devices` 已加 display/provenance-only 注释（`resources.py:117-119`）。**§9** 8 个 config 死键已合入 main（`freeze_vq`/`freeze_vision_encoder`/`freeze_aligner`/`freeze_image_head`/`uncentralized_training` grep 归零）。
>
> **仍开放（本 doc 留在 `planned/` 的原因）**：**P3** AGENTS.md 防腐约定（"派生/解析型结构体每字段须有非日志消费方"）尚未写入；**§5 `FamilyCapability`** 死 flag follow-up 未做（`vrl/generation/capabilities.py:134/135/138` 的 `supports_kv_decode`/`supports_prefill_decode_split`/`supports_cuda_graph` 仍在，需逐 flag 对抗式复核后再删）。
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

更重要的是广义判据：**判据从来不是"字段数"，而是"每个字段有没有真实消费方"——这条对 config 和
派生结构体都成立，且对 config 更严。** 一个声明了却没人读的**配置键**是面向用户的"空操作陷阱"
（用户填了以为生效、实则静默无效），比内部死字段更糟。所以**"精确匹配"（每个字段都对应真实效果）
是对的目标**——之前写"input_config 宽是合理的、不该碰"是错的：字段多只要每个都被消费才健康
（如 `TrainerConfig` 26 个 knob 全被 builder 读），病的是"声明了没人读"，config 同样要查。

注意区分两种胖结构体的风险，但**都要审**：**派生型（resolved/derived：算一次、到处读）**最容易静默
堆积没人读的预计算字段——已逐字段证过的有 `ResolvedDistributedResources`（本文主体）和
`FamilyCapability`（第二个 offender，见 §5）；**input_config / DTO（14 个宽结构体）**我目前只做了
scan 级、**没逐字段证**，属"待同等审计"而非"已证清白"（见 §5、§7 修正）。

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

**其余 14 个宽结构体 = 待同等审计（目前只到 scan 级，不是"免审/明确不碰"）**（按 kind 分桶）：

- **input_config（用户/spec 面向）**：`TrainerConfig`(26)、`NextStep1Config`(19)、`JanusProConfig`(18)、
  `OfflineDPOTrainerConfig`(14)、`VideoGenerationRequest`(14)。scan agent 称"每字段都有 builder/trainer
  消费"，但**未逐字段证**（不像 §3 过了审计+反驳两道）。**config 的判据更严**：一个声明了没人读的配置
  键是面向用户的"空操作陷阱"，比内部死字段更糟。follow-up 应对它们跑同一套审计找死键。删之前给三类
  **合法 passthrough 显式白名单+注释、勿误删**：(a) 分支门控字段（仅某功能开启时被读）、(b) 整体传给
  外部库的 kwargs（diffusers/PEFT/transformers 消费，vrl 内 grep 不到）、(c) dump 进 checkpoint 供
  复现/溯源的 schema 字段。字段多只要每个都被消费就健康——病的只是"声明了没人读"。
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
  分组要改所有 call site，是为口味而搅动正确架构 —— 违反 AGENTS.md "no big refactors / consistency
  over cleanup"。本 sprint 只删确证的死字段，不重组。
- **不碰 19 个 NECESSARY 字段**，也不碰它们流经的 `reward_runtime_resource_kwargs` /
  `format_distributed_resource_plan` 形状（除删 `ray_total_bundles` 那一行日志）。
- **不靠裸 grep 删字段**："精确匹配"的目标对，但要靠证据审计（审计+反驳两道）才能"删对"。对分支门控
  字段 / 库 passthrough kwargs / checkpoint provenance 字段给显式白名单+注释，别把 load-bearing 的
  误判成死字段。
- **不预先豁免 config 宽结构体**：14 个宽 config/DTO 是"待审"不是"免审"（已纠正 §1/§5 早前"宽是本分"
  的错误措辞）。**config 死键审计已执行，见 §9**（`FamilyCapability` 仍为 follow-up）。
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

## 9. 配置死键审计（已执行 —— 兑现"精确匹配"）

§5 把 config 宽结构体列为"待审"，本节是审计结果。**这才是"死配置键比内部死字段更值得抓"的实活**：
死配置键是面向用户的 YAML 旋钮，用户填了以为生效、实则静默无效，比内部死字段危害更大。

> 方法：同一套 workflow（发现 → 逐结构体逐字段审计 → 对每个死键候选对抗式反驳），35 个 agent，
> 跑在干净克隆 `~/Desktop/VRL`。发现并审计 **18 个 YAML/spec-backed 配置结构体**（TrainerConfig 整棵
> 嵌套树 + 模型/DTO config + 算法/Ray config）。我随后**逐个独立读源码复核**了每个删除候选（见下方
> 复核记录），未盲信 agent。

### 9.1 结论（18 结构体 / 共 ~180 字段）

| 处置 | 数量 | 字段 |
|---|---|---|
| **DELETE（已执行）** | 8 | `JanusProConfig.{freeze_vq,freeze_vision_encoder,freeze_aligner}`、`NextStep1Config.{freeze_image_head,image_size}`、`DiffusionNFTConfig.uncentralized_training`、`PrecisionDriftGuardConfig.max_batches`、`OfflineDPOTrainerConfig.num_train_timesteps` |
| **DEFER（本节记录，未删）** | 7 | `TrainerConfig.log_freq` + `VideoGenerationRequest.{prompt,references,task_type,model_name,shift,extra}` |
| **KEEP（escape_hatch / 验证用，非死键）** | — | `TrainerConfig.{gradient_checkpointing,resume_from}`（provenance/checkpoint 消费）、`RolloutOrchestrationConfig.weight_sync_barrier`（`__post_init__` 派生消费 types.py:158） |
| **CLEAN（全字段必要）** | 10 结构体 | `OptimConfig`、`EMAConfig`、`DebugConfig`、`ContinuousRolloutConfig`、`TorchProfilerConfig`、`GRPOConfig`、`RayGenerationConfig`、`DistributedResourceConfig`、`RolloutResourceConfig`、`RewardResourceConfig` |

### 9.2 已删的 8 个死键（含 footgun 说明 + 完整 surface）

每个都过了"审计 + 对抗式反驳"两道，`removable=true`、`hiddenConsumer=null`，并经我读源码确认。

**1–3. Janus `freeze_vq` / `freeze_vision_encoder` / `freeze_aligner`**（最典型 footgun）
- 真相：`JanusProModel.__init__` 无条件 `for p in mmgpt.parameters(): p.requires_grad_(False)`
  （model.py:182-185，注释"Freeze everything by default — LoRA wrap re-enables only attention
  projections"）。这 3 个 flag **从不被读**。用户在 `1b.yaml` 写 `freeze_vq: false` 想训 VQ 模块——
  静默无效，照样全冻。这是 LoRA-only 设计下的纯残留旋钮。
- surface：model.py:102-104（字段）、schema.py:266/269/270、runtime.py:179-181（spec copy-list）、
  `configs/model/ar/janus_pro/1b.yaml:23-25`。

**4. NextStep `freeze_image_head`**（干净对照：`freeze_vae` 真被读、它没有）
- `NextStep1Model.__init__` 只有 `if config.freeze_vae:` 这一条 freeze 分支（model.py:146-148）；
  `freeze_image_head` 从不被读。注释写"train the 157M flow head"却无实现。
- surface：model.py:93、schema.py:267、runtime.py:178（copy-list）、`1_1.yaml:26`。

**5. NextStep `image_size`**（重复 sampling 参数的死副本）
- `config.image_size` 从 `sampling.image_size` 拷进来（runtime.py:173），但 `__init__` 从不读它；
  真正解码用的是 per-request 的 `params.image_size`（AR layout，generation/ar/layout.py，
  runtime.py:324 `decode_image_tokens(image_size=params.image_size)`）。decode 方法签名里的 `image_size`
  当场 `del`。**只删 config 字段 + copy-list 条目**；`sampling.image_size` YAML 键 + schema:249 保留（活）。
- surface：model.py:89（字段）、runtime.py:173（sampling copy-list 条目）；连带删孤儿常量
  `NEXTSTEP_DEFAULT_PIXEL_SIZE`（model.py:54 + `__all__`，唯一消费者就是被删的字段）。

**6. DiffusionNFT `uncentralized_training`**
- loss 读 `cfg.eps/adv_clip_max/global_std/nft_beta/kl_beta/advantage_*/weight_copy_decay`，**从不读
  `uncentralized_training`**（loss 路径无条件 uncentralized）。用户写 `false` 想要 centralized——无效。
- surface：diffusion_nft.py:26、schema.py:110、`configs/base/algorithm/diffusion_nft.yaml:11`。

**7. PrecisionDriftGuard `max_batches`**（仅自我校验的旋钮）
- 唯一读者是自己的 `__post_init__`：`if int(self.max_batches) != 1: raise "only supports 1"`。除了
  "只准等于 1"它什么都不做，没有任何地方拿它去 bound 批数。无 YAML、无 schema、无测试引用。
- surface：types.py:73（字段）+ types.py:82-83（`__post_init__` 校验）。

**8. OfflineDPO `num_train_timesteps`**（set 了但被刻意不读）
- `train_dpo.py:213` 从 scheduler 把它塞进 config，但 trainer 的 `_sample_timesteps` **故意不回退**到它
  （dpo.py:222-235 直接 raise，红线测试 `test_offline_dpo_timesteps.py::test_empty_timesteps_raises`
  守的就是"不许 silently fall back"）。红线测试用的是 **scheduler 的** `num_train_timesteps`
  （`SimpleNamespace`），不是这个 config 字段——删字段不动红线行为。
- surface：dpo.py:54（字段）、train_dpo.py:213（构造实参）。

### 9.3 缓删的 7 个（DEFER，本 sprint 未动）

- **`TrainerConfig.log_freq`** —— 确属死键（schema.py:306 + 9 个 experiment YAML 都写 `log_freq: 1`，但
  在线 loop 每 epoch 无条件记录、无人读它）。**缓删原因**：(1) surface 最宽（schema + 9 个实验配置 +
  base 注释 + 1 个 e2e override），改动面大；(2) 它是**唯一"删 vs 接线"意图真歧义**的一个——可能团队
  本就想让日志按 `log_freq` 节流。需你拍板：删旋钮，还是把日志节流接上（两者都对，现状是旋钮空转）。
- **`VideoGenerationRequest.{prompt,references,task_type,model_name,shift,extra}`** —— 它是**代码构造的
  请求 DTO，不是 YAML 配置键**，不属本次"死配置键"范畴，**单独立项、独立 MR 复核，本 MR 不碰**。
  但我已用只读追踪**复核确认这 6 个确属死字段**（之前怀疑 `prompt` 判死是 agent 误判 —— 是我错了，
  agent 判对了）：
  - **从不写、从不读（4 个，最干净）**：`references` / `task_type` / `model_name` / `shift` —— RL 路径
    `build_video_request`（executor.py:342-368）只塞 8 个 base 采样字段 + `extra`，从不设它们；脚本
    `anima/generate.py:321` 也不设；无任何 model 读它们。
  - **只写不读（write-only，2 个）**：`prompt`（仅 `anima/generate.py:322` 写，但脚本用单独
    `model.encode_prompt(...)` 的 `encoded`，RL 用 `chunk.prompt`，executor.py:260 —— 请求上的 prompt
    无人读）；`extra`（`build_video_request` + 脚本都写，但 `DiffusionBackboneInput.extra` 是
    `prepare_sampling` 里**新建**的，`predict2/model.py:367 extra={...}`，**不从 `request.extra` 拷**；
    runner 里的 `request.extra` 读的是 `DiffusionBackboneInput`（带 `.prompt_embeds`），另一种类型）。
  - **结论/为何独立 MR**：6 个删掉收益真实（DTO 表面收窄、消除"写了没人读"），但 `prompt`/`extra` 是
    write-only，删字段须连带删写入点（executor.py + anima/generate.py），属 DTO 重构而非配置键清理，
    与本 MR 主题正交 —— 放独立 MR 更干净。

### 9.4 复核记录 + 验证

- **独立复核**：8 个删除项我都读了实现确认（Janus/NextStep 的 `__init__` freeze 逻辑、NFT loss 的
  `cfg.*` 读取面、DPO `_sample_timesteps`、`image_size` 的 sampling-param 真实流向）。`prompt` 判死可疑
  → 主动剔出删除集。
- **测试**（`PYTHONPATH=~/Desktop/VRL ~/Desktop/wm-infra/.venv/bin/python -m pytest`）：
  `tests/config/test_load_all_experiments.py`（全实验配置加载）、`test_schema.py`、`test_unknown_keys.py`、
  `tests/trainers/test_offline_dpo_timesteps.py`、`tests/trainers/online/test_precision_drift_guard.py`、
  `tests/models/ar/{janus_pro,nextstep_1}/*` **全过**（共 95 passed）。`tests/algorithms/test_diffusion_nft.py`
  7 例失败**仅因本地 venv 缺 `peft`**（fixture import 处即崩，与改动无关，属环境缺口同
  [[test_env_torchvision_gap]]）；改动本身已验证 inert（`DiffusionNFTConfig` 去掉字段后 import 干净）。
- **改动落点**：`~/Desktop/VRL` 分支 `audit/config-dead-keys`，12 文件 `-32/+1`，**未 commit**（等你过目）。

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

配置死键审计（§9，已执行于 `~/Desktop/VRL` 分支 `audit/config-dead-keys`）：
- Janus freeze flags：`vrl/models/ar/janus_pro/model.py:182-185`（无条件冻结，footgun 根因）、
  `runtime.py:177-183`（spec copy-list）、`vrl/config/schema.py:263-270`、`configs/model/ar/janus_pro/1b.yaml`
- NextStep：`vrl/models/ar/nextstep_1/model.py:89,93,146-148`、`runtime.py:166-178`、`configs/model/ar/nextstep_1/1_1.yaml`
- DiffusionNFT：`vrl/algorithms/diffusion_nft.py:26`、`configs/base/algorithm/diffusion_nft.yaml:11`
- PrecisionDriftGuard / DPO：`vrl/trainers/core/types.py:73,82-83`、`vrl/trainers/offline/dpo.py:54` + `vrl/scripts/diffusion/wan_2_1/train_dpo.py:213`
- 缓删：`vrl/trainers/core/types.py:251`（`log_freq`，+ schema.py:306 + 9 个实验 YAML）、`vrl/generation/diffusion/layout.py:19`（`VideoGenerationRequest` DTO）
