# SPRINT: 分层审计——dataclass/arg 归属层与各层必要性（planned）

状态：**planned（2026-07-23）**。10 张层卡（逐层通读真实代码 + AST 导入图 + 306 个 typed
struct 清单）→ 4 个跨层判决（类型归属 / generation 栈深度 / 包级层必要性 / config 流向）→
每条发现过对抗验证。共 **14 条确认改动**（12 low + 2 medium）+ **6 条否决**（记于 §4，防重议）。
来源：layering-audit workflow。关联：[[SPRINT_deadcode_00_overview]]、
[[SPRINT_single_caller_inlines]]（§4 中立 registry / generation 不依赖 rollouts）、
[[SPRINT_generation_regime_decision_layering]]（PolicySemantics 为 dispatch 事实源）。

## 0. 一句话（先回答你的两个担忧）

**担忧二"这些层是否都必要"——大部分答案是 YES，且有据。** 审计逐层核对后，没有一个包级层被判
"多余可删"：`generation/composition/` 单模块层、`vrl/ray` vs `vrl/generation/ray`、
`vrl/families` vs `vrl/models/families`、generation 的五层栈（steps/bindings/composition/
execution/ray）都被**对抗验证判为"保留有理由"**——它们分别是 substrate-vs-adapter、中立 registry
vs 实现、plugin-implements-framework 的真实边界，不是仪式性文件夹。真正缺的不是"删层"，而是**给
两条只靠约定维持的边界补上架构门禁测试**（families import-lightness、generation→models
interface-floor）。

**担忧一"dataclass/arg 是否在正确层"——找到一小撮真错层 + 若干重复形状。** 6 类类型/常量住错了
层或被手抄两遍：算法层的 signal 契约困在 rollouts、rollout 域的 `RolloutStats` 困在中立 utils、
模型无关的 `unwrap_compile_and_ddp` 定义在 trainers 却被 models 向上 import、byte 级相同的
`IMAGE_SUFFIXES` 抄了两份、family task 词表散成三处手维护投影、gradient-checkpointing resolver
放在 trainers 却被中立 families 向上 import。这些都是**低风险机械迁移**，但每条都消除一条"错误方向
的依赖边"或一处"registry 加 task 就静默漏配"的腐烂点。

**没有一条建议要动 protocol/lazy-import/跨家族一致性边界。** 你历史上 revert 过两次过度扁平化，本
sprint 严格站在保护那些边界的一侧——所有迁移都只搬"放错家的纯数据/纯函数"，不碰抽象。

---

## 1. 错层的类型与参数（担忧一）

### 1.1 `unwrap_compile_and_ddp` — inverted-dependency（risk=low）
- **位置**：定义在 `vrl/trainers/weight_sync.py:150-170`；被 `vrl/models/utils.py:183` 与 `:257`
  **向上懒 import**——这是全仓 `vrl/models -> vrl/trainers` 仅有的两条包级边。
- **证据**：`grep -rn "vrl.trainers" vrl/models/` 恰好命中这两处，别无他处；函数体是纯 `getattr`
  循环剥 `_orig_mod`/`.module` 两种 wrapper，torch-free、无 trainer 域状态。其 docstring 自己就
  引用 `models/utils.py` 的 `load_weights_into` 为对偶方——双侧消费、单侧却定义在上层。
- **动作**：把函数**下沉到 `vrl/models/utils.py`**（不是 utils——见下）。models/utils.py:38 与
  :64 已经在手写单趟 `getattr(module, "_orig_mod", module)` 剥壳，是这个循环的**近似重复且漏掉
  嵌套 `compile(DDP(m))` 的顺序**——迁到 models/utils.py 同时消重并修掉这个潜伏 bug。改
  `trainers/weight_sync.py:109` 与 `trainers/strategy.py`（661,676,690,709,901,951 共 7 处懒
  import）改从 `vrl.models.utils` import。效果：`models -> trainers` 上行边整体归零。
- **注意**：另一判决曾提议放 `vrl/utils`（也 torch-free 合法）。选 models/utils.py 是因为它能顺带
  消掉那两处漏 bug 的手写剥壳；若评审倾向纯粹性，放 utils 亦可，但会漏掉消重收益。

### 1.2 `RolloutStats` + `StatsSink` — misplaced-type（risk=medium）
- **位置**：`vrl/utils/stats.py:38,206`。
- **证据**：`grep RolloutStats|StatsSink` 在 `vrl/generation`、`vrl/rewards` **零命中**；真实消费者
  只有 `rollouts/orchestration`（约 9 文件）、`trainers/online`、`scripts/common`。字段本身是
  rollout/reward 域词汇（`reward_queue_wait_ms`、`reward_inference_ms`、`collect.*` 相位）——
  域形状放在"零域导入"的 utils 叶层不当。模块零 vrl 依赖，迁移纯机械。
- **动作**：整模块迁至 `vrl/rollouts/stats.py`，更新 rollouts 内导入 + `trainers/online`
  （trainers→rollouts 边已存在）+ `scripts/common`（scripts→rollouts 已存在）。
- **注意**：**保留条件**——`stats.py` docstring 宣称数据流经 generation→reward→trainer。若确有
  generation 层记录点的近期规划，则维持 utils 现状并在模块头标注"为跨 generation 中立性置于
  utils"，因为 generation 禁止 import rollouts（gate `test_generation_rollout_boundaries.py:13-24`），
  搬走即关闭该扩展路径。**按当前消费者事实应下沉；迁移前确认无 generation 记录点规划。**

### 1.3 signal 契约 `SegmentSignal` / `TrajectorySignalBatch` / `SignalRequest` — inverted-dependency（risk=low→算法层 eager torch）
- **位置**：`vrl/rollouts/evaluators/types.py:10,32,84`（纯 dataclass，仅依赖 dataclasses+typing）。
- **证据**：`grep vrl/algorithms` 里对 `vrl.rollouts` 仅有的两条 import 就是
  `algorithms/trajectory.py:9` 与 `algorithms/grpo/multisegment.py:13` 导入这两个 signal 类型；
  `AlgorithmInput.__post_init__`（trajectory.py:23-26）对 `TrajectorySignalBatch` 做 isinstance
  运行时检查——硬依赖。**实测**：`algorithms/trajectory.py:9` 的 module 级 import 触发
  `rollouts/evaluators/__init__.py`，后者 `import torch` + `models.interfaces`，使 config parse 期
  调用 `algorithm_config_class()` 沿 config→algorithms→rollouts→models **落入 torch**，抵消了
  `continuous.py` 的函数级 torch 延迟导入。
- **动作**（采纳收窄版）：新建 `vrl/algorithms/signals.py`，**仅**移 `SegmentSignal` +
  `TrajectorySignalBatch` + 私有 helper `_require_same_shape`（TrajectorySignalBatch.__post_init__
  的唯一用户）。`SignalRequest` **留在** `rollouts/evaluators/types.py`——它被 evaluators 消费、由
  trainer/evaluators 构造，零 algorithms 引用，搬它反增 rollouts→algorithms 边。改
  `algorithms/trajectory.py:9`、`grpo/multisegment.py:13` 改 import——此后 **algorithms→rollouts
  边归零**，config-parse eager-torch 消除。
- **注意**：初版判决曾提议把三个类整体搬去 `vrl/trajectory/signals.py`。**否决**：那会把 SignalRequest
  也拖走、并新增 rollouts→trajectory 依赖，还不消除 eager-torch 根因。收窄到"只搬构成 algorithms→
  rollouts 边的两个类、放进 algorithms 自己"才是根因修复。

### 1.4 `resolve_gradient_checkpointing_mode` — inverted-dependency（risk=low）
- **位置**：定义在 `vrl/trainers/activation_checkpointing.py:86-98`；被中立 registry
  `vrl/families/registry.py:228-232` 在 `ModelFamilyEntry.resolve_model_build` 里**向上懒 import**
  （nextstep_1 的 `gradient_checkpointing_at_load` 分支）。这是全仓 `vrl/families -> vrl/trainers`
  **唯一一条边**。
- **证据**：`grep -rn "trainers" vrl/families/` 仅命中 `registry.py:228`。函数体（已读）是
  `OmegaConf.select("actor.gradient_checkpointing")` + `TrainerConfig` dataclass 默认派生，**无
  trainer 运行时状态**——是 config-read，不是 trainer 逻辑。
- **动作**：把 `resolve_gradient_checkpointing_mode` + `_normalize_gradient_checkpointing` 移入
  `vrl/config`（纯 OmegaConf 读 + TrainerConfig 默认派生；其 TrainerConfig import 已是函数级懒
  加载，且 config→trainers 是既有方向）。`trainers/activation_checkpointing.py` 保留 runtime apply +
  compile 冲突守卫，从 `vrl.config` import resolver。`families/registry.py:228` 改走 `vrl.config`
  import——**families→trainers 边归零**，与 §4 中立 registry 决策一致。

---

## 2. 重复形状（一份数据/规则抄了两遍）

### 2.1 `IMAGE_SUFFIXES` byte 级重复 — duplicate-shape（risk=low）
- **位置**：`vrl/rewards/base.py:617`（frozenset）与 `vrl/trainers/data/artifacts.py:21`（set）——
  两处 7 元素 `{.bmp,.gif,.jpeg,.jpg,.png,.ppm,.webp}` 完全一致。
- **证据**：并排读取确认字节级相同；消费点仅两处（rewards `:648` decode 帧、trainers `:344` 分类）。
  `rewards/base.py:625-635` docstring 辩称"依赖 artifact 契约故置于 rewards 而非 utils"——但后缀
  **集合本身**与 artifact 契约无关，只有 `decode_artifact_frames` 才依赖契约。
- **动作**：把常量上收到 `vrl/utils/artifacts.py`（torch-free 叶子，trainers 已从中 import
  `ArtifactManifestError` 等），加一行 WHY 注释说明是媒体扩展名分类、与 artifact 契约无关。两处删本地
  定义改为 import，用法不变。
- **注意**：**非目标**——`DEFAULT_ARTIFACT_FIELDS`（由 `PromptExample` 字段派生，事实源在 trainers）
  与 `SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS`（tuple 顺序承载报错语义，注释明言 NOT
  auto-derived）**不动**；`rewards→trainers` 那条单向 provenance import 是正当耦合。目的地选
  `utils/artifacts.py` 而非 `utils/media.py`：trainers 已 import 前者，零新增依赖。

### 2.2 family task 词表三处手维护投影 — duplicate-shape（risk=medium）
- **位置**：事实源 `ModelFamilyEntry.task`（`vrl/families/registry.py:110`，**裸 str，无 owner
  Literal**，8×t2i / 6×t2v / t2w·v2w·i2v·ar_t2i_r1 各 1）；投影① `_DEFAULT_TASK_TYPE_BY_FAMILY_TASK`
  （`rollouts/collector/requests.py:13-19`，手映射 task→task_type 长名，漏 ar_t2i_r1，靠 `.get()→None`
  兜底）；投影② `_reward_modality_for_task`（`trajectory/builders.py:922-926`，硬编码
  `{t2v,i2v,v2w,t2w}→video else image`）。
- **证据**：task token 字面量散布 14 个文件；registry 新增一个 video task token 时，投影② 会**静默
  落到 else 被判成 "image"**——这是唯一有真实静默误配风险的表。`GenerationChunkExecutor` 协议也带
  `task: str`（`protocols.py:110`），同样吃不到类型约束。
- **动作**：在 `vrl/families/semantics.py`（import-light）新增 task 词表 owner：
  (1) `Task = Literal["t2i","t2v","t2w","i2v","v2w","ar_t2i","ar_t2i_r1"]`——**必须枚举全部 7 个
  活跃 token**（含 `ar_t2i`、`ar_t2i_r1`），否则会拒掉 janus_pro/janus_pro_r1 合法 entry；
  (2) 派生 `VIDEO_TASKS = frozenset({"t2v","i2v","v2w","t2w"})` 与 `task_modality(task)`，供
  `builders.py` import 替换硬编码集合（消除静默误配）；(3) task→task_type 那张表迁入
  `semantics.py` 作 `task_type_for(task)`——`requests.py` 已 import `vrl.families.registry`，零新增
  耦合。
- **注意**：**否决"form-4 registry re-derivation"定性**——registry 只存 task 字面量，并不编码
  task_type 或 modality 映射，所以这不是"重新推导 registry 已有信息"，而是"两张手维护查找表集中化"。
  task→task_type 表对 token 家族**故意返回 None**，文档须写明这是"故意兜底"不是"漏配"，别伪装成 bug。

---

## 3. 缺失的边界（约定成立但无门禁 / 门面漏项）

### 3.1 generation→models "interface floor" 无架构门禁 — missing-boundary（risk=low）
- **位置**：`tests/architecture/test_generation_rollout_boundaries.py:13-24` 禁止 generation import
  rollouts/trainers/rewards/algorithms，但**不约束 generation 可以 import 哪些 models 子模块**。
- **证据**：AST 扫全 `vrl/generation`：import `vrl.models` 恰好 6 处，**全在 floor 上**
  （`vrl.models.interfaces` / `.loader` / `.dtypes`）——off-floor 计数为 0。约定成立但纯靠自觉。
- **动作**：给该测试加一条门禁：`vrl/generation` 下每个文件，凡匹配 `vrl.models.*` 的 import 必须前缀
  为 `vrl.models.interfaces` / `vrl.models.loader` / `vrl.models.dtypes`（复用 `_imports`/
  `_python_files`，做一个 module-prefix allowlist 变体）。
- **顺带回答"要不要把 GenerationRequest/Output 下沉到新共享包破 models↔generation 双向"**：**不要**。
  62 条 `models→generation` 边由 executor-base 子类化主导（bindings 14 + execution.chunks 10 +
  steps.token 7 + composition 2），是 plugin-implements-framework 的健康形态；反向仅 6 条窄面 floor
  import。双向是**分层正确**的表现，不是耦合病。

### 3.2 `vrl/families` import-lightness 无架构门禁 — missing-boundary（risk=low）
- **位置**：`test_generation_rollout_boundaries.py` 有 generation/trajectory/`vrl/ray` 的门禁，但**没有
  `vrl/families` 的**；唯一探针 `tests/rollouts/runtime/test_family_registry.py:57-63` 只测 names-only
  import 不加载 registry。
- **证据**：AST walk `tree.body`（仅 module 级）确认 `vrl/families/*.py` 顶层 import 只有 stdlib
  （`__future__`/`collections.abc`/`dataclasses`/`enum`/`typing`）或 `vrl.families.*`；所有
  `vrl.models`/`vrl.config`/`vrl.trainers`/`vrl.utils` import 都是函数级懒加载（8 处）。
- **动作**：加 `test_families_registry_stays_import_light`：对每个 `vrl/families/*.py` **只走 module 级
  语句**（`ast tree.body`，**不是** `ast.walk`——否则现有 `_forbidden_imports` 会误伤 8 处刻意的
  函数级懒 import），断言每条顶层 import 是 stdlib 或 `vrl.families.*`。这把 §4 的 lazy-import 纪律从
  "约定"变成"机械强制"。

### 3.3 `ReferenceConditionedChunks` 绕过 bindings 包门面 — missing-boundary（risk=low）
- **位置**：定义在 `vrl/generation/bindings/full_sequence_denoise/executor.py:112`，在模块 `__all__`
  （`:689`）却**未被包 `__init__.py` re-export**。
- **证据**：`grep "from vrl.generation.bindings" vrl/models`：家族对 `ARChunkInputs` 等所有 import 均走
  包门面（三个 `__init__.py` re-export 齐全）；唯二例外是 `wan_2_1/runtime.py:16` 与
  `cosmos/predict2/runtime.py:15` 走 `.executor` 深路径 import 这个类——是门面清单漏项，非刻意内部
  类型。
- **动作**：把 `ReferenceConditionedChunks` 加进 `full_sequence_denoise/__init__.py` 的 import 与
  `__all__`，两处深路径 import 改走门面。一行级修复，使"家族插件只经三个 binding 门面消费类型"契约无
  例外。

### 3.4 `probe_chunk_size` duck-type binding 的五个 stage 方法 — missing-boundary（risk=low）
- **位置**：`vrl/generation/execution/worker.py:346-357` 硬编码方法名五元组
  `build_prompt_stage_input`/`run_prompt_encode_stage`/`run_prepare_stage`/`run_denoise_stage`/
  `run_decode_stage`（getattr 循环 + 报错文案 "samples_per_chunk: auto is diffusion-only"），在
  `:382-400` 调用；这些方法只定义在 `bindings/full_sequence_denoise/executor.py:368-481`。
- **证据**：读全 `probe_chunk_size`，注释写着 "The four stage methods ARE forward_chunk_plan's body"；
  这五个名字在 bindings 与本探针**之外零出现**。execution 禁止 import bindings，所以这条"regime 中立
  执行核心伸进某个 binding 的 stage 词表"的跨层 reach **对任何 import 门禁都不可见**。
- **动作**：在 `vrl/generation/protocols.py` 紧邻 `GenerationChunkExecutor` 声明一个
  `runtime_checkable` Protocol（如 `DiffusionStagedChunkExecutor`，含五个 stage 签名），
  `probe_chunk_size` 改用 `isinstance` 判它而非 getattr 循环，并加测试断言
  `DiffusionChunkExecutorBase` 满足它。
- **注意**：**与在途 sprint 冲突**——`docs/sprints/SPRINT_native_generation_engine_program.md` 的 N2
  阶段正在动 execution/stage 词表。**本条须排在该 sprint 之后**，或与其合并，避免 Protocol 与它新增的
  stage 契约打架。

### 3.5 kill-actors-retain-failed-handles 在两个 Ray 层重复 — duplicate-shape（risk=low）
- **位置**：`vrl/ray/actor_group.py:113-124`（`RayActorGroup.shutdown`）与
  `vrl/generation/ray/worker_fleet.py:249-258`（`_kill_owned_workers_locked`）。
- **证据**：两段函数体近乎逐字相同——`failures = kill_actors(ray, actors)` → 取 failed id 集 →
  `list[:] = [x for x in list if x.actor is not None and id(x.actor) in failed_ids]`，仅 handle 字段名
  （`handles` vs `_owned_workers`）与结尾（raise vs return）不同。属死代码审计 form-4（函数体等同某
  共享实现的自然归属）。
- **动作**：抽一个 helper 进 `vrl/ray/resource_cleanup.py`（已 own `kill_actors`），如
  `kill_and_retain(ray, items, get_actor) -> (surviving, failures)`，两处都调它。
- **顺带回答"`vrl/ray` 与 `generation/ray` 是否重复、要不要合并"**：**不合并**。这是真正的
  generic-substrate vs domain-adapter 拆分——`vrl/ray` 被 gate 强制域中立
  （`test_generation_rollout_boundaries.py:164-177` 禁其 import generation/rollouts/trainers/rewards），
  `generation/ray` 单向消费它（13 条 import 边）。两层都必要；只是这段 retain-handle 逻辑该住在中立
  substrate 里供两边共用。
- **注意**：`worker_fleet.py` 属 `generation/ray/`，在在途 native-generation-engine sprint 的改动面
  内——落地前确认与该 sprint 的 worker-fleet 改动不冲突，必要时排其后。

---

## 4. config 流向

### 4.1 prompt loader 派生规则在 schema 与 runtime 各一份 — duplicate-shape（risk=low）
- **位置**：`vrl/config/schema.py:195-205`（`DataConfig._validate_data`，`:204` 写回 `self.loader`）与
  `vrl/trainers/data/prompts.py:116-150`（`load_prompt_examples_from_config`，`:126-128` 重新派生）。
- **证据**：`prompts.py:124-125` 注释直书 "Keep this rule in sync with DataConfig._validate_data in
  vrl/config/schema.py"——手工同步复制体，正是 AGENTS.md "derive, don't duplicate" 禁止形态。根因：
  pydantic schema 是 **lint-only 边界**——`_validate_data` 把派生结果写回 `self.loader`，但该 pydantic
  对象**不传给 runtime**（runtime 拿 raw OmegaConf cfg），所以 `prompts.py` 必须从 raw cfg 重派生。
  两份表达式逐字相同：`"prompt_image_manifest" if fmt == "image_caption_jsonl" else "prompt_manifest"`。
- **动作**：抽单一纯函数 `resolve_prompt_loader(loader, preprocessing_format) -> str` 放 runtime owner
  `vrl/trainers/data/prompts.py`，`schema.py` `_validate_data` 反向调它做校验。规则只留一处，新增
  loader 类型只改一处。低风险纯函数抽取，`test_load_all_experiments` + prompts 测试即可回归。

### 4.2 `kl_reward_coef` / `trajectory_storage` 逐 rollout 重解析 — config-resolved-too-deep（risk=low）
- **位置**：`vrl/rollouts/collector/core.py:239-242`——`float(cfg_get(self.config, "kl_reward_coef",
  0.0))` 与 `trajectory_storage_policy_from_cfg(cfg_get(self.config, "trajectory_storage", None))` 在
  `score_rollouts` 的 `for rollout in unscored` **循环内**逐个 rollout 从 `RolloutCollectorConfig.values`
  重解析。
- **证据**：读全 `core.py:230-243`——因为是 dict-of-values 投影 + 字符串 key，**typo 的 key 会在
  runtime 静默取 0.0/None 默认，而不是在 config build 期报错**；且 `trajectory_storage` 每次都重跑
  `trajectory_storage_policy_from_cfg` 生成同一个 policy 对象。
- **动作**（最小正确修法）：把两处解析从循环内**上提到循环外解析一次**（`self.config` frozen，本 batch
  内恒定），或在 `RolloutCollector.__init__` 缓存为 `self._kl_reward_coef` /
  `self._trajectory_storage_policy`。
- **注意**：`trajectory_storage` **绝不移出 `values`**——它必须继续经 `request_sampling()` 过 wire 供
  `executor.py:350` 消费，且 `test_family_registry.py:187` pin 了它在 `values` 中。若要给 typed 字段
  只作 driver 侧便利，也只能作 `values["trajectory_storage"]` 的派生缓存，原始值仍留 `values`。此条是
  **正确性 + 微性能**修复，不是 values 投影重构。

### 4.3 `RolloutWorkerSection` 逐字段镜像 `RayGenerationConfig` — duplicate-shape（**contested**，risk=low）
- **位置**：`vrl/config/schema.py:726-769`（7 字段 + `_validate_health_check` @ `:757`）vs
  `vrl/generation/ray/config.py:24-110`（dataclass 默认 `:28-45` + `__post_init__` 复校 `:58-63` +
  `from_cfg` cfg_get 默认 `:85-109`）。
- **证据**：7 个 schema 字段（`cpus_per_worker`/`max_inflight_chunks_per_worker`/
  `health_check_interval_s`/`health_check_timeout_s`/`chunk_placement_strategy`/
  `sync_trainable_state`/`pipelined`）恰好是 `RayGenerationConfig` 字段减 `resources`；
  `30.0/1/1.0/False/True` 默认值出现在**三处**（schema 字段默认、dataclass 默认、from_cfg 兜底）；
  health-check 校验两处各实现一份。
- **动作**（采纳 revise：**加 parity 测试，不合并类型**）：在 `tests/config/test_schema.py` 加断言
  `set(RolloutWorkerSection.model_fields) == {f.name for f in dataclasses.fields(RayGenerationConfig)}
  - {"resources"}` + 逐字段默认相等（schema / dataclass / from_cfg 三处）。
- **注意（为何不合并）**：一个对抗验证者**否决了"合并"**——理由是 pydantic schema 是 lint-only 边界、
  runtime dataclass 才是真消费者，"schema 校验的每个值都流向 runtime"这个前提不成立，二者**结构上无法
  直接合一**。所以调和结论是"**承认三处重复但用 parity 测试钉死同步，不强行合并**"。同时**否决原
  finding 里"drift 已发生"的说法**：`health_check_first_wait_s` 两处都声明过、正被对称删除，无历史
  schema-less 漂移。这条列为 contested，落地前值得你拍板"加测试" vs "维持现状"。

---

## 5. 已考虑但否决（防重议）

对抗验证**推翻**了以下 6 条，记录在此避免下次重开：

- **`generation/composition/` 单模块层不删**：只含 `token_autoregressive/token_loop.py`（385L）看似
  仪式性，但它是 token-AR 组合层的真实概念位，与 denoise 的 bindings/composition 对称；删掉破坏跨
  regime 一致形状。**保留。**
- **`LossUnit` + `TrainingView.loss_units` 不删**：判决曾疑"每 batch 构建校验却无人读"，验证发现有真
  消费者，非死。**保留。**
- **`GenerationRuntimeCapabilities.runs_in_isolated_subprocess` 的"重复 magi_1 binding"定性被否**：它
  作为**分层重复**不成立（binding 并未编码这个 bool）。但它作为**死字段**成立——见
  [[SPRINT_deadcode_model_families]]（那里按 test-only reader 删字段并改写测试），本 sprint 不重复处理。
- **`distributed.training.strategy` 在 `families/registry.py:211-217` 读 raw cfg 不算错层**：设
  `defer_trainable_device_move` 是 registry 组装 ModelBuild 的合法一步，验证判 KEEP。
- **health-check 校验 schema/runtime 双实现不算可合并重复**：见 §4.3——两侧边界性质不同，"合并"被否，
  仅"加 parity 测试"。

---

## 6. 验证协议

每条落地后：
- `ruff check <touched>` + `ruff format --check <touched>` 全绿。
- 迁移类（§1、§2）：`grep` 确认旧路径零残留 import；跑受影响层的 `pytest`（signal→
  `tests/algorithms tests/rollouts`；stats→`tests/rollouts tests/trainers`；unwrap→
  `tests/trainers tests/models`；task 词表→`tests/rollouts tests/models tests/trajectory`）。
- 加门禁类（§3.1/3.2）：新测试先在**当前树**跑绿（证明约定当前成立），再验证故意加一条越界 import 会
  让它红（证明门禁有效）后回退。
- 全部完成：`python -m vrl.config.lint` + `pytest -m "not e2e and not slow_test"` 子集**不新增失败**。
- **基线（2026-07-23）**：fast subset **2620 passed / 7 项预先失败**（架构边界 + causvid/magi_1 打包
  摘要，与本 sprint 无关）；`vrl.config.lint` 与 `ruff check .` 全绿。落地后这三项须保持。

## 7. Non-Goals

- **不删任何包级层**——§0 已论证 composition/、双 ray 层、双 families 层、generation 五层栈都必要。
- **不下沉 `GenerationRequest`/`GenerationOutput` 到新共享包**——models↔generation 双向是
  plugin-implements-framework 的正确形态（§3.1）。
- **不合并 `RolloutWorkerSection` 与 `RayGenerationConfig`**——边界性质不同，只加 parity 测试（§4.3）。
- **不动** `DEFAULT_ARTIFACT_FIELDS`、`SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS`、
  `distributed.training.strategy` 的 registry 读取——均为合法保留（§2.1、§5）。
- **不碰在途 sprint 改动面**（§3.4 probe Protocol、§3.5 worker_fleet）——排在
  `SPRINT_native_generation_engine_program` 之后或与其合并。
- 不为省行数扁平化 protocol / lazy-import / 跨家族一致性边界。

## References

- 迁移类：`vrl/trainers/weight_sync.py:150-170`、`vrl/models/utils.py:38,64,183,257`、
  `vrl/utils/stats.py:38,206`、`vrl/rollouts/evaluators/types.py:10,32,84`、
  `vrl/algorithms/trajectory.py:9`、`vrl/algorithms/grpo/multisegment.py:13`、
  `vrl/trainers/activation_checkpointing.py:86-98`、`vrl/families/registry.py:110,211-217,228-232`
- 重复形状：`vrl/rewards/base.py:617,648`、`vrl/trainers/data/artifacts.py:21,344`、
  `vrl/utils/artifacts.py`、`vrl/rollouts/collector/requests.py:13-19`、
  `vrl/trajectory/builders.py:922-926`、`vrl/families/semantics.py`、`vrl/generation/protocols.py:110`
- 缺失边界：`tests/architecture/test_generation_rollout_boundaries.py:13-24,164-177`、
  `vrl/generation/bindings/full_sequence_denoise/{__init__,executor}.py:112,689`、
  `vrl/generation/execution/worker.py:346-357,382-400`、`vrl/ray/actor_group.py:113-124`、
  `vrl/generation/ray/worker_fleet.py:249-258`、`vrl/ray/resource_cleanup.py`
- config 流向：`vrl/config/schema.py:195-205,726-769`、`vrl/trainers/data/prompts.py:116-150`、
  `vrl/rollouts/collector/core.py:230-243`、`vrl/generation/ray/config.py:24-110`
- 层卡与原始判决：`scratchpad/layer_cards.md`、`scratchpad/layering_confirmed.json`；逐 agent journal
  在 workflow 转录目录
- 关联：[[SPRINT_deadcode_00_overview]]、[[SPRINT_single_caller_inlines]]、
  [[SPRINT_generation_regime_decision_layering]]、[[SPRINT_native_generation_engine_program]]
