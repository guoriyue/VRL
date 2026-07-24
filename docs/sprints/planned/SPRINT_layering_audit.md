# SPRINT: 分层审计——dataclass/arg 归属层与各层必要性（planned·已复核）

状态：**RECONCILED（2026-07-24）**，基线从审计时的旧树 `88ed756e` 迁到当前 main
`@ 7c748532`（= origin/main tip，审计后又落地 ~63 个 cleanup/refactor commit）。原 14 条确认
改动经**逐条独立复核**后：**12 条仍需做**（8 STILL_VALID + 4 RELOCATED，见 §1）、**1 条已由
origin 落地**（见 §2）、**1 条情况已变需重评**（见 §3）。RELOCATED 指发现依然成立、但其代码位置
在这 63 个 commit 中移动了行号/文件，§1 已把「位置」行更新到当前树。§4「已否决」照旧保留防重议。
来源：layering-audit workflow（原判决 JSON：`scratchpad/layering_confirmed.json`）。关联：
[[SPRINT_deadcode_00_overview]]、[[SPRINT_single_caller_inlines]]、
[[SPRINT_generation_regime_decision_layering]]、[[SPRINT_native_generation_engine_program]]。

> **执行状态（2026-07-24）**：9 条已落地 `b54d4205`（类型下沉/去重/门禁，无 import 环，models→trainers 边归零）。原延后 2 条已落地 `7a4f2b5f`：§1.C.3 probe Protocol（`DiffusionStagedChunkExecutor`）+ §1.A.3 signal——**采用 lazy-import 方案 (b) 而非搬文件方案 (a)**，消除 config-parse 期 eager torch 且**不引入** contested 的 rollouts→algorithms 反向边。

## 0. 一句话（复核后仍成立的两个结论）

**担忧二「这些层是否都必要」——大部分 YES 且有据。** 没有一个包级层被判「多余可删」：
`generation/composition/` 单模块层、`vrl/ray` vs `vrl/generation/ray`、`vrl/families` vs
`vrl/models/families`、generation 五层栈都是 substrate-vs-adapter / 中立 registry vs 实现 /
plugin-implements-framework 的真实边界。真正缺的不是「删层」，而是给两条只靠约定维持的边界补
架构门禁（families import-lightness §1.C.2、generation→models interface-floor——后者**情况已变**，
见 §3）。

**担忧一「dataclass/arg 是否在正确层」——一小撮真错层 + 若干重复形状，绝大多数仍在。** 复核确认
signal 契约仍困在 rollouts、`RolloutStats` 仍困在中立 utils、`unwrap_compile_and_ddp` 仍定义在
trainers 却被 models 向上 import、byte 级相同的 `IMAGE_SUFFIXES` 仍抄两份、family task 词表仍散成
多处手维护投影、gradient-checkpointing resolver 仍在 trainers 却被中立 families 向上 import。这些都是
低风险机械迁移。**唯一已被 origin 修掉的是 prompt loader 派生规则重复**（§2）。

**没有一条建议要动 protocol/lazy-import/跨家族一致性边界。** 所有迁移都只搬「放错家的纯数据/纯函数」。

---

## 1. 待删清单（仍有效）

复核结论：以下 12 条在 `7c748532` 上依旧成立。RELOCATED 项已把「位置」行更新到当前行号/文件，并标注
「已移位」。原始证据/动作照旧。

### 1.A 错层的类型与参数

#### 1.A.1 `unwrap_compile_and_ddp` — inverted-dependency（risk=low）· **STILL_VALID**
- **位置**：定义在 `vrl/trainers/weight_sync.py:150`；被 `vrl/models/utils.py:183` 与 `:257`
  **向上懒 import**——这是全仓 `vrl/models -> vrl/trainers` 仅有的两条包级边。
- **证据**：`grep "def unwrap_compile_and_ddp" vrl/` 命中 weight_sync.py:150；`grep "from vrl.trainers"
  vrl/models/` 恰好命中 utils.py:183 与 :257（都是 `import unwrap_compile_and_ddp`，懒），别无他处；
  函数体是纯 `getattr` 循环剥 `_orig_mod`/`.module` 两种 wrapper，torch-free、无 trainer 域状态。其
  docstring 自己就引用 `models/utils.py` 的 `load_weights_into` 为对偶方——双侧消费、单侧却定义在上层。
- **动作**：把函数**下沉到 `vrl/models/utils.py`**。models/utils.py:38 与 :64 已在手写单趟
  `getattr(module, "_orig_mod", module)` 剥壳（compile-only 剥壳，接收侧从不 DDP 包裹，属刻意），迁到
  models/utils.py 时可用统一循环替换做 form-4 去重（行为等价，不是「修 nested compile(DDP(m)) 顺序 bug」
  ——该顺序在接收侧并不触发）。改 weight_sync.py:109 与 strategy.py 多处懒 import 从 `vrl.models.utils`
  import。效果：`models -> trainers` 上行边整体归零。
- **注意**：另一判决曾提议放 `vrl/utils`（也 torch-free 合法）。选 models/utils.py 因它能顺带收敛那两处
  手写剥壳；若评审倾向纯粹性，放 utils 亦可，但会漏掉消重收益。

#### 1.A.2 `RolloutStats` + `StatsSink` — misplaced-type（risk=medium）· **STILL_VALID**
- **位置**：`vrl/utils/stats.py:38,206`。
- **证据**：`grep -E "class RolloutStats|class StatsSink" vrl/` 恰好命中 utils/stats.py:38 与 :206；在
  `vrl/generation`、`vrl/rewards` **零命中**；真实消费者只有 `rollouts/orchestration`（约 9 文件）、
  `trainers/online`、`scripts/common`。字段本身是 rollout/reward 域词汇（`reward_queue_wait_ms`、
  `reward_inference_ms`、`collect.*` 相位）——域形状放在「零域导入」的 utils 叶层不当。模块零 vrl 依赖。
- **动作**：整模块迁至 `vrl/rollouts/stats.py`，更新 rollouts 内导入 + `trainers/online`
  （trainers→rollouts 边已存在）+ `scripts/common`（scripts→rollouts 已存在）。
- **注意**：docstring 宣称数据流经 generation→reward→trainer，但那是**数据流**而非导入/类型流——generation
  只吐 primitive phase dict，从不 import 或构造该类型；累加器 travels ON rollout 层类型。故「utils 中立性」
  理由不成立，按消费者事实应下沉。

#### 1.A.3 signal 契约 `SegmentSignal` / `TrajectorySignalBatch` / `SignalRequest` — inverted-dependency（risk=low）· **STILL_VALID**
- **位置**：`vrl/rollouts/evaluators/types.py:10,32,84`（纯 dataclass，仅依赖 dataclasses+typing）。
- **证据**：`grep -nE "^class (SegmentSignal|TrajectorySignalBatch|SignalRequest)"` 命中 :10/:32/:84（精确
  对上原发现）。`vrl/algorithms` 对 `vrl.rollouts` 仅有的两条 import 就是 `algorithms/trajectory.py:9` 与
  `algorithms/grpo/multisegment.py:17`（复核后行号：multisegment 现为 **:17**，原审计记 :13）导入
  `TrajectorySignalBatch`；`AlgorithmInput.__post_init__` 对其做 isinstance 运行时硬依赖。module 级 import
  触发 `rollouts/evaluators/__init__.py`（`import torch` + `models.interfaces`），使 config parse 期落入
  torch，抵消 `continuous.py` 的函数级延迟导入。
- **动作**（收窄版）：新建 `vrl/algorithms/signals.py`，**仅**移 `SegmentSignal` + `TrajectorySignalBatch`
  + 私有 helper `_require_same_shape`（TrajectorySignalBatch.__post_init__ 唯一用户）。`SignalRequest`
  **留在** `rollouts/evaluators/types.py`——它被 evaluators 消费、由 trainer/evaluators 构造，零 algorithms
  引用，搬它反增 rollouts→algorithms 边且违背 consumer 规则。改 `algorithms/trajectory.py:9`、
  `grpo/multisegment.py:17` 的 import——此后 **algorithms→rollouts 边归零**。
- **注意**：此举引入代码库首个 `rollouts→algorithms` 方向（evaluators import algorithms.signals），与
  `schedule.py` 记录的「rollout 层不 import vrl.algorithms」意图相反；辩护点是导入的是纯 stdlib 数据契约而非
  algorithm 逻辑，且 TrajectorySignalBatch 的数据消费者确为 algorithms。落地时须在提案里显式记录此方向反转，
  或改采「TYPE_CHECKING 注解 + isinstance 点内 lazy import」的更外科替代（不移文件、不引反向边）。原「三个类
  整体搬去 `vrl/trajectory/signals.py`」的初版判决**否决**（会拖走 SignalRequest 并新增 rollouts→trajectory 边）。

### 1.B 重复形状（一份数据/规则抄了两遍）

#### 1.B.1 `IMAGE_SUFFIXES` byte 级重复 — duplicate-shape（risk=low）· **STILL_VALID**
- **位置**：`vrl/rewards/base.py:617`（frozenset，用于 :648 decode 帧）与
  `vrl/trainers/data/artifacts.py:21`（set，用于 :344 分类）——两处 7 元素
  `{.bmp,.gif,.jpeg,.jpg,.png,.ppm,.webp}` 字节级相同。
- **证据**：`grep IMAGE_SUFFIXES vrl/` 仅这两处定义 + 两处消费，并排读取确认字节相同，均未搬走。
  `rewards/base.py` docstring 辩称「依赖 artifact 契约故置于 rewards」——但后缀**集合本身**与契约无关，
  只有 `decode_artifact_frames` 才依赖契约。
- **动作**：把常量上收到 **`vrl/utils/artifacts.py`**（torch-free 叶子，trainers 已从中 import
  `ArtifactManifestError` 等），统一为 `frozenset`，加一行 WHY 注释说明是媒体扩展名分类、与 artifact 契约
  无关。trainers 侧改 `from vrl.utils.artifacts import IMAGE_SUFFIXES`；rewards 侧在 `decode_artifact_frames`
  函数内既有惰性 import 块追加同一 import（保持 rewards/base.py 原惰性边界，不引模块级新边）。
- **注意**：**不要**搬到 `vrl/utils/media.py`——其顶层 `import torch/numpy` 会把 torch 拖进当前 torch-free
  的 manifest 校验模块 `trainers/data/artifacts.py`（该文件并未 import media，「两个调用点都已 import media」
  对 trainers 侧不成立）。**非目标**：`DEFAULT_ARTIFACT_FIELDS`（由 `fields(PromptExample)` 派生，SoT 在
  trainers）与 `SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS`（tuple 顺序承载报错语义，NOT auto-derived）不动；
  `rewards→trainers` 单向 provenance import 是正当耦合。

#### 1.B.2 family task 词表多处手维护投影 — duplicate-shape（risk=medium）· **RELOCATED**
- **位置（已移位）**：事实源 `ModelFamilyEntry.task`（`vrl/families/registry.py:157`，**裸 str，无 owner
  Literal**；原审计记 :106/:110）；投影① `_DEFAULT_TASK_TYPE_BY_FAMILY_TASK`
  （`rollouts/collector/requests.py:13-19`，**位置未变**，仍漏 `ar_t2i`/`ar_t2i_r1`，靠 `.get()→None` 兜底）；
  投影② `_reward_modality_for_task`（`vrl/trajectory/builders.py:800-804`，**已从 :922-926 移到 :800**，body
  仍硬编码 `if self.request.task in {"t2v","i2v","v2w","t2w"}` else image）。复核确认无 `semantics.py` owner
  被创建（`grep task_type_for|VIDEO_TASKS|task_modality vrl/` 为空）。
- **证据**：registry 只存 task 字面量子集，投影②新增一个 video task token 时会**静默落到 else 被判 "image"**
  ——这是唯一有真实静默误配风险的表。task token 字面量散布 14 个文件。活跃 token 实为 **7 个**（含
  `ar_t2i`、`ar_t2i_r1`，janus_pro/janus_pro_r1 用），非 5 个。
- **动作**：在 `vrl/families/semantics.py`（import-light）新增 task 词表 owner：
  (1) `Task = Literal["t2i","t2v","t2w","i2v","v2w","ar_t2i","ar_t2i_r1"]`——**必须枚举全部 7 个活跃 token**，
  否则会拒掉 janus_pro/janus_pro_r1 合法 entry；(2) 派生 `VIDEO_TASKS = frozenset({"t2v","i2v","v2w","t2w"})`
  与 `task_modality(task)`，供 `builders.py` import 替换硬编码集合（消除静默误配）；(3) task→task_type 表迁入
  `semantics.py` 作 `task_type_for(task)`（`requests.py` 已 import `vrl.families.registry`，零新增耦合）。对
  reward_modality 有更彻底的根因替代：在 `GenerationRequest` 加显式 `reward_modality` 字段由 request builder
  从 family task 填充，trajectory 直接读 `self.request.reward_modality` **不引 trajectory→families 新边**——
  该边虽被 gate 允许但会打破 trajectory 当前完整的 family-neutrality，取舍时优先此 3b 方案。
- **注意**：**否决「form-4 registry re-derivation」定性**——registry 只存 task 字面量，并不编码 task_type 或
  modality 映射，这是「两张手维护查找表集中化」而非「重推导 registry 已有信息」。task→task_type 表对 token
  家族**故意返回 None**，文档须写明是「故意兜底」不是「漏配」。

### 1.C 缺失的边界（约定成立但无门禁 / 门面漏项）

#### 1.C.1 `ReferenceConditionedChunks` 绕过 bindings 包门面 — missing-boundary（risk=low）· **STILL_VALID**
- **位置**：定义在 `vrl/generation/bindings/full_sequence_denoise/executor.py:112`，在模块 `__all__`
  （**:678**，原审计记 :689）却**未被包 `__init__.py` re-export**。
- **证据**：`full_sequence_denoise/__init__.py` re-export 了 `DiffusionChunkExecutorBase` 等 7 类却独缺
  `ReferenceConditionedChunks`；唯二例外是 `cosmos/predict2/runtime.py:15` 与 `wan_2_1/runtime.py:16` 走
  `.executor` 深路径 import 这个类——门面清单漏项，非刻意内部类型（真私有不会出现在模块 `__all__` 中）。
- **动作**：把 `ReferenceConditionedChunks` 加进 `full_sequence_denoise/__init__.py` 的 import 与 `__all__`，
  两处深路径 import 改走门面。一行级修复，使「家族插件只经三个 binding 门面消费类型」契约无例外。

#### 1.C.2 `vrl/families` import-lightness 无架构门禁 — missing-boundary（risk=low）· **STILL_VALID**
- **位置**：`tests/architecture/test_generation_rollout_boundaries.py` 有 generation/trajectory/`vrl/ray`
  的门禁，但**没有 `vrl/families` 的**（`grep families` 该文件仅命中一处无关的 rollouts/families 移除断言）。
- **证据**：`vrl/families/*.py` 顶层 import 只有 stdlib（`__future__`/`collections.abc`/`dataclasses`/
  `enum`/`typing`）或 `vrl.families.*`；所有 `vrl.models`/`vrl.config`/`vrl.trainers`/`vrl.utils` import
  都是函数级懒加载（复核：现约 12 处，含 registry.py:346-350 的 gradient-checkpointing 懒 import）。真实
  families→generation 依赖走 dotted string（`GENERIC_FULL_SEQUENCE_DENOISE_EXECUTOR`、per-entry
  `executor_cls`/`gatherer_cls`），对 AST 门禁不可见。
- **动作**：加 `test_families_registry_stays_import_light`：对每个 `vrl/families/*.py` **只走 module 级语句**
  （`ast tree.body`，**不是** `ast.walk`——否则现有 `_forbidden_imports` 会误伤那些刻意的函数级懒 import），
  断言每条顶层 import 是 stdlib 或 `vrl.families.*`。把 §4 的 lazy-import 纪律从「约定」变成「机械强制」。

#### 1.C.3 `probe_chunk_size` duck-type binding 的五个 stage 方法 — missing-boundary（risk=low）· **STILL_VALID**
- **位置（已移位）**：`vrl/generation/execution/worker.py`——`probe_chunk_size` 现于 **:298**，方法名五元组
  `build_prompt_stage_input`/`run_prompt_encode_stage`/`run_prepare_stage`/`run_denoise_stage`/
  `run_decode_stage` 硬编码在 getattr 循环 **:347**，直接调用在 **:382**（原审计记 :346-357/:382-400）；
  这些方法只定义在 `bindings/full_sequence_denoise/executor.py`。
- **证据**：这五个名字在 bindings 与本探针之外零出现；`GenerationChunkExecutor` 协议只声明
  `forward_chunk_plan`/`gather_chunks`，探针真实契约比声明宽。execution 禁止 import bindings，所以这条跨层
  reach 对任何 import 门禁不可见。复核确认无 Protocol、无 N2 的 `ResolvedGenerationExecution`。
- **动作**：在 `vrl/generation/protocols.py` 紧邻 `GenerationChunkExecutor` 声明一个 `runtime_checkable`
  Protocol（如 `DiffusionStagedChunkExecutor`，含五个 stage 签名），`probe_chunk_size` 改用 `isinstance`
  判它而非 getattr 循环，并加测试断言 `DiffusionChunkExecutorBase` 满足它。
- **注意**：**与在途 sprint 协调**——`SPRINT_native_generation_engine_program.md` 的 N2 阶段若先落地
  `ResolvedGenerationExecution`，则「supports staged probing」应进那个 capability struct 而非 isinstance；
  排在该 sprint 之后或与其合并，避免 Protocol 与它新增的 stage 契约打架。

#### 1.C.4 kill-actors-retain-failed-handles 在两个 Ray 层重复 — duplicate-shape（risk=low）· **RELOCATED**
- **位置（已移位）**：`vrl/ray/actor_group.py:112-117`（`RayActorGroup.shutdown`，原记 :113-124）与
  `vrl/generation/ray/runtime.py:536-542`。**关键变化：原第二站点 `generation/ray/worker_fleet.py:249-258`
  的 `_kill_owned_workers_locked` 所在文件已被删除**，同一 retain-failed 模式在 `runtime.py:536-542`
  重新出现（`worker_failures = kill_actors(...); failed_worker_actor_ids = {id(actor) for actor,_ in ...};
  self._owned_workers[:] = [... id(worker.actor) in failed_worker_actor_ids]`），与 actor_group.py:112-117
  byte-parallel。复核确认无 `kill_and_retain` helper（`resource_cleanup.py` 仍只导出 `kill_actors`/
  `remove_placement_group`）。
- **证据**：两段函数体近乎逐字相同——`failures = kill_actors(ray, actors)` → 取 failed id 集 →
  `list[:] = [x for x in list if x.actor is not None and id(x.actor) in failed_ids]`，仅 handle 字段名与结尾
  （raise vs return）不同。属死代码审计 form-4。`resource_cleanup.py` 的 `kill_actors` docstring 本就预告
  「callers that own lifecycle truth can retain failed handles」——这两处正是那两个 caller。
- **动作**：抽一个 helper 进 `vrl/ray/resource_cleanup.py`（已 own `kill_actors`），如
  `kill_and_retain(ray, items, get_actor) -> (surviving, failures)`，两处都调它。
- **顺带回答「`vrl/ray` 与 `generation/ray` 是否重复」**：**不合并**——真正的 generic-substrate vs
  domain-adapter 拆分，`vrl/ray` 被 gate 强制域中立，`generation/ray` 单向消费它。只是这段 retain-handle
  逻辑该住在中立 substrate 供两边共用。
- **注意**：原「worker_fleet.py 在在途 sprint 改动面内、须排其后」的碰撞提示**已作废**（该文件已删）；第二
  站点现落在 `generation/ray/runtime.py`，落地前确认与 native-generation-engine sprint 的 runtime 改动不冲突。

### 1.D config 流向

#### 1.D.1 `kl_reward_coef` / `trajectory_storage` 逐 rollout 重解析 — config-resolved-too-deep（risk=low）· **RELOCATED**
- **位置（已移位）**：`vrl/rollouts/collector/core.py:219-222`（原记 :239-242）——仍在 `score_rollouts` 的
  `for rollout in unscored`（**:212**）循环内构造 `RolloutBatchBuildContext` 时逐个 rollout 跑
  `float(cfg_get(self.config, "kl_reward_coef", 0.0))` 与
  `trajectory_storage_policy_from_cfg(cfg_get(self.config, "trajectory_storage", None))`。
- **证据**：因是 dict-of-values 投影 + 字符串 key，**typo 的 key 会在 runtime 静默取 0.0/None 默认，而不是
  config build 期报错**；且 `trajectory_storage` 每次都重跑 `trajectory_storage_policy_from_cfg` 生成同一个
  policy 对象。行块从 :239-242 移到 :219-222，问题不变。
- **动作**（最小正确修法）：把两处解析从循环内**上提到循环外解析一次**（`self.config` frozen，本 batch 内
  恒定），或在 `RolloutCollector.__init__` 缓存为 `self._kl_reward_coef` / `self._trajectory_storage_policy`。
- **注意**：`trajectory_storage` **绝不移出 `values`**——它必须继续经 `request_sampling()` 过 wire 供 worker
  侧 `executor.py` 的 `request.sampling.get("trajectory_storage")` 消费（wire 传输节省的关键点），且
  `test_family_registry.py` pin 了它在 `values` 中。`kl_reward_coef` 若要提升 typed 字段，须同步改
  `config.py` 的 `return_kl` 派生（return_kl 仍需过 wire），不是纯机械 rename——不想承担这层改写就只做
  「循环外解析一次」，风险最低。此条是**正确性 + 微性能**修复，不是 values 投影重构。

#### 1.D.2 `RolloutWorkerSection` 逐字段镜像 `RayGenerationConfig` — duplicate-shape（contested，risk=low）· **STILL_VALID**
- **位置（行号已移位）**：`vrl/config/schema.py:731`（`RolloutWorkerSection`，原记 :726-769）vs
  `vrl/generation/ray/config.py:73`（`RayGenerationConfig`，原记 :24-110）。
- **证据**：7 个 schema 字段恰好是 `RayGenerationConfig` 字段减 `resources`；`30.0/1/1.0/False/True` 默认值
  出现在**三处**（schema 字段默认、dataclass 默认、from_cfg 兜底）；health-check 校验两处各一份。复核确认无
  parity 测试（`tests/config/` 只有无关的 LoraSection/ModelMemorySection 断言）。
- **动作**（采纳 revise：**加 parity 测试，不合并类型**）：在 `tests/config/test_schema.py` 加断言
  `set(RolloutWorkerSection.model_fields) == {f.name for f in dataclasses.fields(RayGenerationConfig)}
  - {"resources"}` + 逐字段默认相等（schema / dataclass / from_cfg 三处）。
- **注意（复核修正）**：原 finding 依赖的「`health_check_first_wait_s` 正被对称删除、无 schema-less drift」
  前提**已被证伪**——该字段的删除**从未落地**，`first_wait` 至今仍是**三处 triplicated 的活字段**
  （`schema.py:754` + validator `:775`、`config.py:31` + validator `:58`、`runtime.py:75`），且
  `test_removed_health_check_first_wait_is_unknown` 不存在。这不改变结论（parity 测试仍有效、仍未做），但
  落地时应把动机表述改为「三处默认字面量偏斜 + 反向的 schema 字段被 from_cfg 忽略成 no-op 旋钮」这两个残余
  缺口，并把 first_wait 一并纳入 parity 覆盖。为何不合并：pydantic schema 是 lint-only 边界、runtime
  dataclass 才是真消费者，二者结构上无法直接合一——contested，落地前值得拍板「加测试」vs「维持现状」。

---

## 2. 已由 origin 落地（本次复核确认，无需再做）

- **prompt loader 派生规则**（duplicate-shape，原 §4.1）—— `image_caption_jsonl → prompt_image_manifest
  else prompt_manifest` 曾在 `config/schema.py` 与 `trainers/data/prompts.py` 各抄一份并带「Keep this rule
  in sync」注释；现已抽出单一 owner `vrl/config/data.py::resolve_data_loader`（data.py:22 持唯一派生），
  两处旧拷贝改为调用它（schema.py:32 import + :193 调用；prompts.py:12 import + :122 调用），手抄表达式与
  sync 注释均已消失。**落地 commit：`456f7069` refactor(data): resolve prompt loader once。**

---

## 3. 情况已变（需重新评估）

- **generation→models「interface floor」架构门禁**（missing-boundary，原 §3.1）—— **CHANGED，原动作不能
  照搬。** 门禁本身仍缺（测试文件只有 rollouts/training 的 denylist gate，无 floor allowlist）。但发现的
  事实前提变了：原发现称「generation 恰好在 6 个站点 import models，全在 floor（interfaces/loader/dtypes）
  上」。当前树新增了两处**off-floor** import——`vrl.models.checkpoint_identity` 被
  `vrl/generation/ray/launcher.py:242` 与 `vrl/generation/execution/worker.py:664` 导入，由审计后的
  `95952909`（feat(models): derive immutable checkpoint identity）+ `eaa681c6` 引入（在 `88ed756e` 上
  `git grep checkpoint_identity` over `vrl/generation/` 为空）。因此原提案的 3-module allowlist
  （interfaces/loader/dtypes）**今天会 false-fail**。**重评动作**：先把 floor 定义**扩到含
  `vrl.models.checkpoint_identity`**（确认它确是中立、只读的 identity 派生模块、无反向 generation 依赖），
  再让门禁 allowlist 覆盖 4 个前缀后才能落地——否则新加的 checkpoint-identity 边会被误判越界。

---

## 4. 已考虑但否决（防重议，照旧保留）

对抗验证**推翻**了以下条目，记录在此避免下次重开：

- **`generation/composition/` 单模块层不删**：token-AR 组合层的真实概念位，与 denoise 的 bindings/composition
  对称。**保留。**
- **`LossUnit` + `TrainingView.loss_units` 不删**：验证发现有真消费者，非死。**保留。**
- **`GenerationRuntimeCapabilities.runs_in_isolated_subprocess` 的「重复 magi_1 binding」定性被否**：作为
  **分层重复**不成立；作为**死字段**成立——见 [[SPRINT_deadcode_model_families]]，本 sprint 不重复处理。
- **`distributed.training.strategy` 在 `families/registry.py` 读 raw cfg 不算错层**：registry 组装
  ModelBuild 的合法一步，验证判 KEEP。
- **health-check 校验 schema/runtime 双实现不算可合并重复**：见 §1.D.2——两侧边界性质不同，「合并」被否，
  仅「加 parity 测试」。
- **不合并 `RolloutWorkerSection` 与 `RayGenerationConfig`**、**不下沉 `GenerationRequest`/`GenerationOutput`
  到新共享包**：models↔generation 双向是 plugin-implements-framework 的正确形态，只加门禁/parity 测试。

---

## 5. 验证协议

每条落地后：
- `ruff check <touched>` + `ruff format --check <touched>` 全绿。
- 迁移类（§1.A、§1.B）：`grep` 确认旧路径零残留 import；跑受影响层的 `pytest`（signal→
  `tests/algorithms tests/rollouts`；stats→`tests/rollouts tests/trainers`；unwrap→
  `tests/trainers tests/models`；task 词表→`tests/rollouts tests/models tests/trajectory`）。
- 加门禁类（§1.C.1/§1.C.2）：新测试先在**当前树**跑绿（证明约定当前成立），再验证故意加一条越界 import
  会让它红后回退。§3 的 floor 门禁须先扩 floor 定义再验证。
- 全部完成：`python -m vrl.config.lint` + `pytest -m "not e2e and not slow_test"` 子集**不新增失败**。
- **基线（2026-07-24，main @ 7c748532）**：落地前须先在当前 checkout 上重跑 fast subset + `vrl.config.lint`
  + `ruff check <touched>` 取得绿色基线（原 `88ed756e` 的 2620-passed 快照已过期，不再作为门槛），落地后
  这三项须保持不新增失败。

## 6. Non-Goals

- **不删任何包级层**——composition/、双 ray 层、双 families 层、generation 五层栈都必要（§0）。
- **不下沉 `GenerationRequest`/`GenerationOutput` 到新共享包**——models↔generation 双向是
  plugin-implements-framework 的正确形态（§3、§4）。
- **不合并 `RolloutWorkerSection` 与 `RayGenerationConfig`**——只加 parity 测试（§1.D.2）。
- **不动** `DEFAULT_ARTIFACT_FIELDS`、`SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS`、
  `distributed.training.strategy` 的 registry 读取——均为合法保留（§1.B.1、§4）。
- **不碰在途 sprint 改动面**——§1.C.3 probe Protocol 排在 `SPRINT_native_generation_engine_program` 之后
  或与其合并；§1.C.4 第二站点现落在 `generation/ray/runtime.py`（worker_fleet.py 已删），落地前确认不冲突。
- 不为省行数扁平化 protocol / lazy-import / 跨家族一致性边界。

## 7. References

- 仍需做（§1）：`vrl/trainers/weight_sync.py:150`、`vrl/models/utils.py:38,64,183,257`、
  `vrl/utils/stats.py:38,206`、`vrl/rollouts/evaluators/types.py:10,32,84`、
  `vrl/algorithms/trajectory.py:9`、`vrl/algorithms/grpo/multisegment.py:17`、
  `vrl/trainers/activation_checkpointing.py:87`、`vrl/families/registry.py:157,346-350`、
  `vrl/rewards/base.py:617,648`、`vrl/trainers/data/artifacts.py:21,344`、`vrl/utils/artifacts.py`、
  `vrl/rollouts/collector/requests.py:13-19`、`vrl/trajectory/builders.py:800-804`、
  `vrl/families/semantics.py`、`vrl/generation/protocols.py`、
  `vrl/generation/bindings/full_sequence_denoise/{__init__,executor}.py:112,678`、
  `vrl/generation/execution/worker.py:298,347,382`、`vrl/ray/actor_group.py:112-117`、
  `vrl/generation/ray/runtime.py:536-542`、`vrl/ray/resource_cleanup.py`、
  `vrl/rollouts/collector/core.py:212,219-222`、`vrl/config/schema.py:731,754,775`、
  `vrl/generation/ray/config.py:31,58,73`、`vrl/generation/ray/runtime.py:75`、
  `tests/architecture/test_generation_rollout_boundaries.py`、`tests/config/test_schema.py`
- 已落地（§2）：`vrl/config/data.py:22`、`vrl/config/schema.py:32,193`、`vrl/trainers/data/prompts.py:12,122`
  （commit `456f7069`）
- 情况已变（§3）：`vrl/generation/ray/launcher.py:242`、`vrl/generation/execution/worker.py:664`、
  `vrl/models/checkpoint_identity`（commit `95952909`、`eaa681c6`）
- 层卡与原始判决：`scratchpad/layer_cards.md`、`scratchpad/layering_confirmed.json`；复核基线 main @ 7c748532
- 关联：[[SPRINT_deadcode_00_overview]]、[[SPRINT_single_caller_inlines]]、
  [[SPRINT_generation_regime_decision_layering]]、[[SPRINT_native_generation_engine_program]]
