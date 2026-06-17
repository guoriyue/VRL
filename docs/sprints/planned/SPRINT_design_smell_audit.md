# SPRINT: 全仓设计异味审计与清理（design-smell audit）(planned)

状态：**部分完成 —— 2026-06-16**。这是一次"全 `vrl/` 范围找丑设计 + 清掉确证安全项"的审计，
起因是用户点的两个例子：(a) `data.loader` 要手填一个魔法字符串名（`prompt_manifest` /
`prompt_image_manifest`），(b) release 时机过去要手动指定、现已改为从 GPU 拓扑派生——用户想把"这类
冗余手填 / 应从单一真相源派生"的味道在全仓系统找出来清掉。

> **已落地（§3，8 个安全清理，全部保行为，测试绿）**：删 4 个死字段
> （`RecipeDeviceContext.distributed_resources` / `RolloutScheduleState.initialized` /
> `ARAttentionConfig.dtype/device` / `RayGenerationConfig.to_dict()`）、合并 1 对逐字节相同的
> `/proc` 读取器、改 1 个装饰性命名（`load_anima_transformer_component`→`load_anima_transformer`）、
> 删 1 个假旋钮（`TokenGRPOConfig.mask_key`）、修 1 个 fixed-eval **丢条件 bug**
> （`reference_video` 没转发）。`pytest` 相关全套 318 + 85 passed，唯一失败是 `transformers` 未装进
> `.venv` 的预存环境缺口（与改动无关，同 [[test_env_torchvision_gap]]）。未 commit（按规矩等指令）。
>
> **仍开放（本 doc 留在 `planned/` 的原因）**：约 30 条 findings 待你拍板（§4）——多数触及契约面 /
> public 导出 / "删 vs 接线"意图歧义 / 跨模块依赖边，按 north-star 排序、每条给 surface + 修法 +
> 风险 + 为何 held。其中 release flag 残留（`reward_release_after_score` /
> `rollout_release_before_reward_model` 影子化 `RayLifecyclePlan`）与
> [[SPRINT_resolved_struct_field_audit]] §3 存在结论冲突，**归并到那个 sprint 复核**，不在此重复立项
> （§5.2）。

> 方法：1 个编排 workflow，10 个子系统各 1 个 survey agent（逐文件真实 read/grep 找候选），每个子系统
> 的候选再派 1 个**对抗式验证 agent**（专门去反证它其实是故意的/必要的，并把候选与用户已知偏好的护栏
> 逐条对撞），最后 1 个汇总 agent 去重归并。共 21 个 agent、53 候选 → 44 survived → 8 refuted。
> **先验证再下结论**：用户的 loader 例子被对抗式 agent **当场证伪**（"直接推导掉"会破坏线上配置，
> §5.1）；我随后对全部 8 个落地项**逐个独立读源码 + grep 消费方 + 跑测试**复核，未盲信 agent 输出；
> 落地前还把 R9/R10 与已完成的 resolved-struct 审计交叉核对，发现冲突并据 local-vs-stored 判据归并
> （§5.2、§8）。

---

## 1. 核心结论 (TL;DR)

**north-star 判据：丑 = "调用方/配置必须手填一份本可从内容、拓扑、类型、或另一处已有字段派生的知识"，
以及它的近亲（同一 switch 散落多处、声明了没人读的旋钮、逐字节复制、装饰性命名）。**

三条最该记住的结论：

1. **用户的 loader 例子确实丑，但"一把推导掉"会炸**（evidence-first 的胜利，§5.1）。对抗式 agent 实证：
   `loader: prompt_manifest` 在线上同时配 `format: text` 和 `format: jsonl`（不是双射）；
   `video_world_v2w.yaml` 用 `prompt_manifest` 却带 `conditioning: reference_image`（按 conditioning
   推导会把它错路由进 `ImageCaptionPromptDataset` 直接坏掉）；还有第三个 loader `pickapic_preference`
   在别处分发。真正的冗余更窄（`videophy_i2v.yaml` 同时写 `task_type: image_to_video` 和
   `loader: prompt_image_manifest`）——属"需决策"，未自动改。

2. **release-flag 残留印证了用户的原始观点**（§5.2）：拓扑派生 `RayLifecyclePlan` 落地后，
   `ResolvedDistributedResources` 上留下了 `reward_release_after_score` /
   `rollout_release_before_reward_model` 两个**扁平字段在影子化那个 plan**——它们作为 stored 字段只被
   一行日志读，真实行为走 `resolved.lifecycle.*`。**但这条与 [[SPRINT_resolved_struct_field_audit]]
   §3 冲突，归并到那个 sprint 复核。**

3. **53 候选 → 44 真异味 → 8 误报**；其中 8 个是确证安全的（死代码 / 逐字节复制 / 假旋钮 / 丢条件
   bug / 装饰命名），已落地（§3）。其余按动作分桶：

| 动作 | 数量 | 去向 |
|---|---|---|
| **已清理**（保行为、测试绿） | 8 | §3（DONE） |
| **待决策**（真异味，修法触及契约/意图歧义/依赖边） | ~22 | §4 |
| **归并到 resolved-struct 审计**（与 §3 of 那个 sprint 冲突，本 doc 不重复立项） | 4 | §5.2 |
| **误报**（对抗式验证推翻，记录以免后续重报） | 8 | §6 |

---

## 2. 判据 (the rubric)

### 2.1 算异味的七类（每条 finding 打一类标签）

- **redundant-spec**（最高优先，贴 north-star）— 调用方/配置必须手填、却可从内容/拓扑/类型/已有字段派生。
- **duplicated-dispatch** — 同一 string/enum switch 在多个文件实现、必须人工保持同步。
- **hardcoded-constant** — 手维护的 ALL_CAPS/集合复制了一个 typed 结构，本应派生（`frozenset(f.name for f in fields(X))`）。
- **stringly-dispatch** — 该用 enum/registry 的边界用了魔法字符串，或字符串在远离定义处被检查。
- **dead-field-or-code** — resolved/derived 结构体上设了却没人读的字段、不可达分支、未用参数。
- **naming** — 装饰性后缀（`_component`/`_manager`/`_handler`/`_helper`/`_util`）或借词碰撞（`runtime`/`backend`/`context`）。
- **thin-helper** — 抽出来、实际只有一个真实调用方、且没提供真实边界的函数/文件。

### 2.2 护栏（用户回退过的清理，命中即判误报，**不报、不动**）

- 配置 dataclass 的**显式 `field(...)` 拼写是故意的**（逐行可审计），永不报。[[feedback_explicit_field_spelling]]
- **跨 family 一致的注册表 / 约定抽象是故意的**，哪怕看着像 ceremony；保持 wan/cosmos/sd3_5/janus_pro/nextstep_1
  并行可 grep 的 thin 函数**必须留**，不许 flatten registry 或把并行函数 data 化。[[feedback_consistency_over_cleanup]]
- **不因"只有一个调用方"就报 single-caller helper**——只有在它**还**没提供任何边界价值（无 protocol/接口/
  lazy-import/test-fake/public-API 理由）时才报。[[feedback_no_single_caller_helpers]]
- **不提大重构 / 不重写能跑的架构**，优先小而外科手术式的保行为改动；正确性 > 清理。[[feedback_no_big_refactors]]
- 不报"不写死配置值的测试"、perf 测量 sprint、`*_smoke`/`*_spike`/`*_probe`/`_scratch_*` 一次性脚本。

---

## 3. 已落地的 8 个安全清理（DONE，逐个独立复核 + 测试绿）

每个都：读源码确认无生产消费方 / 逐字节相同 / 行为等价 → 改 → 跑相关测试套。

| # | 类别 | 改动 | surface（已改动行） | 为何安全 |
|---|---|---|---|---|
| 1 | dead-field | 删 `RecipeDeviceContext.distributed_resources` | `vrl/scripts/common/types.py:39`（字段）+ `online.py:392`（构造 kwarg） | 全仓仅 `online.py:389` 构造该对象，`.distributed_resources` 零读取；下游用 `context.device`/`weight_dtype` 与局部 `resources` |
| 2 | dead-field | 删 `RolloutScheduleState.initialized` | `vrl/rollouts/orchestration/types.py:26`（字段）；`strict_on_policy.py:81`（→文档化 no-op `reset()`）；`continuous/schedule.py:133`（删该写；teardown 其余保留） | 仅 2 处 `=False` 写、零读；`reset()` 旧实现只设这个从不被读的 flag = 旧行为本就 no-op，保 no-op 等价 |
| 3 | dead-field | 删 `ARAttentionConfig.dtype/device` + 随之孤儿的 `_maybe_string` | `vrl/nn/layers/attention/paged.py:29-30`；`vrl/nn/modules/ar_attention_backends.py:131-132`（构造）+`:142-144`（孤儿 helper） | backend 消费 config 时只读 `family/model_key/block_size/cache_layout_version/extra`；`config.dtype/device` 零读（仓内 `config.dtype` 命中的是 nextstep/janus 的别类 config） |
| 4 | dead-code | 删 `RayGenerationConfig.to_dict()` + 修测试假注释 | `vrl/generation/ray/config.py:143-158`；`tests/generation/ray/test_runtime_config.py:235-236` | 零生产调用者（仓内所有 `.to_dict()` 都是 `FamilyCapability`/axis/stage）；Releasable relaunch 直接持有 typed `RayGenerationConfig`（`ray/runtime.py:25`），不经 dict round-trip——测试注释"rebuild from to_dict()"事实错误 |
| 5 | dup | 合并逐字节相同的 `_read_proc_status_mb`/`_read_meminfo_mb` → `_read_proc_field_mb(path, field)` | `vrl/utils/memory.py:107-136`（合并）+`:36-38`（3 个调用点传 path） | 两函数除 `path` 常量与字段名外完全相同 |
| 6 | naming | `load_anima_transformer_component` → `load_anima_transformer`（`_component` 是 AGENTS.md 点名的装饰填充） | `vrl/models/diffusion/cosmos/anima/runtime.py:150/251/279` + `tests/models/interfaces/test_minimal_replay_runtime_wiring.py:287`（monkeypatch 串） | 纯重命名；与私有 `_load_anima_transformer`（model.py，函数内 import）以前导下划线区分，无碰撞 |
| 7 | 假旋钮 | 删 `TokenGRPOConfig.mask_key`（no-op，却被 yaml 宣传成功能） | `vrl/algorithms/grpo/token.py:19` + `vrl/config/schema.py:106` + `configs/base/algorithm/token_grpo.yaml:2,8` | `compute_loss` 用 `signals.mask`（上游产出，固定键），从不读 `cfg.mask_key`；evaluator 各有自己的 `mask_key` 参数；无任何实验配置 override 它；`MultiSegmentTokenGRPOConfig` 继承自它、同样不读 |
| 8 | **bug** | fixed-eval `_fixed_eval_collect_kwargs` 补 `reference_video` 分支 | `vrl/scripts/common/online.py:831-834` ← 真相源 `rollouts/orchestration/prompt_collection.py:142-144` | docstring 自称"Mirrors"训练路径却漏转发 `reference_video`，视频条件 eval prompt 会**静默丢条件**、用与训练不同的条件打分；`reference_video` 确被下游消费（`requests.py`、`rewards/artifacts.py`） |

**测试记录**：`tests/{utils/test_stats,generation/ray/test_runtime_config,rollouts,models/interfaces/test_minimal_replay_runtime_wiring,scripts/test_online_lifecycle,scripts/test_online_precision_bridge,algorithms/test_grpo_token,algorithms/test_multisegment_token_grpo,nn/modules/test_ar_attention_backends,nn/layers/test_paged_attention_contract,config,trainers/test_data}` → **318 passed**；`tests/{nn,scripts}`（除 `transformers`-gap 2 文件）+ `generation/ar/test_janus_paged_attention_one_step` + `trainers/test_memory_guards` + `trainers/online/test_reward_update_flow` → **85 passed / 1 skipped**；`ruff` + `py_compile` 全过。失败只有 `transformers` 未装的环境缺口（`kling_video_reward` 导入崩，与改动无关）。

---

## 4. 待决策 backlog (action items —— 按 north-star 排序)

> 这些是**真异味**，但修法触及契约面 / public 导出 / "删 vs 接线"意图歧义 / 跨模块依赖边，鉴于用户回退过
> 过度清理，**默认不动**。优先级 P0=贴 north-star 的清晰赢、P1=清晰异味、P2=真但次要、P3=可选。
> 路径来自 audit agent 的 grep，**未逐条独立复核**（不同于 §3 已复核项），动手前应像 §3 那样逐项验证。

### 4.1 redundant-spec / 应从单一真相源派生（north-star）

- **P1 · `data.loader` 魔法字符串与 `task_type` 冗余**（`vrl/trainers/data/prompts.py:74-124`，9 个
  `configs/dataset/*.yaml`）。详见 §5.1 ——**不是干净 1:1**，安全修法很窄（loader 缺失时按 prompt-* 家族
  fallback，`loader` 仅保留区分 `pickapic_preference`）。需你拍板。
- **P1 · `_reward_modality_for_task` 硬编码 video-task 集、漏了 `i2v`**（`vrl/trajectory/builders.py:594-597`）。
  Wan i2v 任务串 `"i2v"` 不在集合里、被标成 "image" reward view。**当前无行为影响**（`RewardView.modality`
  在打分路径无读取者，i2v GRPO 尚未跑），但是个埋雷。**加一行 `"i2v"` 是安全底线修复，强烈建议做**；
  "从 registry `default_task_type` 派生"那版才彻底，但需逐家族验证。
- **P2 · `weight_sync_barrier` 由 `mode` 1:1 派生、无运行时读取者、只自校验**（`vrl/trainers/core/types.py:163-206`）。
  运行时全由 `mode` 决定（`strict_on_policy.py`/`continuous/schedule.py`），yaml 里已无人写。它是"矛盾
  override 时大声失败"的 guardrail，且测试用 `weight_sync_barrier=` 作 kwarg 构造。**建议**：`__post_init__`
  保留派生但停作可设 kwarg。（注：[[SPRINT_resolved_struct_field_audit]] §9.1 已把它列 KEEP/派生消费，**与本条一致**，可在那边一并处理。）
- **P2 · `runtime_role` 与 `loads_full_generation_modules` 1:1 耦合可派生**（`vrl/models/replay_loading.py:15-18,31-64`）。
  贴 north-star，但 `runtime_role` 是序列化 metadata 契约的 named key（被 `test_memory_guards.py` 等断言），
  属未完成的 minimal-replay 在制基建。collapse 是契约变更。**倾向不动**（见护栏 2.2）。
- **P2 · `DiffusionNFTConfig.advantage_low` 恒为 `-advantage_high`**（`vrl/algorithms/diffusion_nft.py:24-25`，`schema.py:96`）。
  loss 的 `reward_mix` 数学只在 `low==-high` 时良态，否则静默破坏 [0,1] 中心化 = footgun 冗余 spec。两个修法
  （派生 `low=-high` / 修数学支持非对称）都动 public config 面。
- **P2 · `media_type` 可由 `artifact_format` 派生（mp4→video）**（`vrl/rewards/artifacts.py:28-33`，`config/validation.py:134-140`）。
  仅 mp4 情形冗余（tensor 格式仍需 image/video 独立轴）。修复会**放松一个生产 gate**（今天 `mp4+media_type=image`
  会 raise，改后静默 coerce）——所以是决策项。
- **P3 · `require_separate_gpus` 是手动 flag、护着拓扑可派生的 colocation 检查**（`vrl/rollouts/orchestration/continuous/schedule.py:199-209`）。
  贴 north-star，但它是 single-GPU async-debug recipe 的正当逃生舱（`async_debug.yaml` + 测试走 false 路径）。
  最多加一条澄清注释。最低优先。
- **P2 · `backend_label`/`cache_dtype` 经 untyped `extra` dict 路由、派生默认在每个消费者重抄**
  （`vrl/nn/modules/ar_attention_backends.py:84-85,104-108` + `ar_decoder.py` + `torch_attention.py`）。
  canonical 模板 `f"{family}_..._attention"` 出现在 4 处，`.get` fallback 正常流不可达。安全小修：消费者直接读
  `config.extra["backend_label"]`、删重抄的 f-string fallback、保留 test override seam。升 typed 字段是更宽的口味选择。

### 4.2 duplicated-dispatch（含用户点名的 kind switch）

- **P2 · `algorithm.kind` 在三文件分发、必须同步**（`vrl/config/builders.py:195-222` / `scripts/common/factory.py:171-267` /
  `config/schema.py:427-462`）。三处职责真不同（建 config / 建 runtime+各异 evaluator / 校验 per-kind 必填），
  **不是三份同表，不建议塞 `ALGORITHM_REGISTRY`、不统一 factory 的 evaluator wiring**。唯一安全窄赢：让
  `schema._cross_field_validate` 的 allowed kinds 从 `get_args(...)` 派生（`resolve_algorithm_kind` 已示范）。
- **P3 · `(family, kind)` 兼容性硬编码在 schema + factory 两层**（`schema.py:438-456` / `factory.py:62-65,227-230`）。
  family 名以裸字面 `"janus_pro"`/`"nextstep_1"` 远离 `FAMILY_REGISTRY` 校验。安全窄步：family 名改经 registry
  常量引用（rename 才能被捕获）。**别**给 registry 加 `supported_kinds` 矩阵（改 guarded 结构）。
- **P2 · "pool reward" 谓词在 factory 与 resources 双写**（`vrl/scripts/common/factory.py:132-137` 与
  `vrl/ray/resources.py:1211-1226`）。同一规则 `weight>0 AND execution=="pool"` 两份，但**数据形态不同**
  （factory 走 built dict 用 `float(weight)>0.0`；resources 走 raw cfg 用 try/except）——抽取须逐字节对齐否则
  引入微妙行为变化。`"pool"` 字面散在 9 处，但 `rewards/base.py`/`runtime.py` 是正当 runtime 边界，不要并入。
- **P3 · sampler 类型合法集在 `config/schema.py` 手抄两遍**（`schema.py:153-162, 176-185`，真正 dispatch 在
  `trainers/checkpointing.py:382-388`）。**安全的小赢**：把两个 prompt-loader 分支里逐字节相同的 sampler 校验块
  合并成一处（**纯文件内**）。⚠️ **不要**从 `checkpointing.py` import frozenset——它引入 `torch`，会把 torch
  拖进当前 torch-free 的 config 导入路径 = 真实回归。
- **P2 · `reward_model_name@revision` HF-repo 解析在 kling/videocon 各自复制**（`rewards/models/kling_video_reward.py:629-634`
  与 `videocon_physics.py:257-262`，各带自己的 `_DEFAULT_REVISION="main"`）。`repo@rev` grammar 非家族特异，可抽
  `parse_reward_repo_spec` 进已存在的 `rewards/models/base.py`（守 "no new lean files"）。但用户刻意保持两家族
  parallel，需确认 shared-vs-parallel。

### 4.3 dead-field / dead-code（触及前瞻基建或 public 面）

- **P1 · `hardcoded-constant`：Kling checkpoint 子目录硬编码 `checkpoint-11352`**（`rewards/models/kling_video_reward.py:31,671-675`）。
  resolver 别处已 `glob("checkpoint-*")`，这里却硬钉一个具体目录 = 脆。改 `any(root.glob("checkpoint-*"))` 对齐。
  **未自动改**：动的是 reward 模型加载路径，且本地 `.venv` 缺 `transformers` 跑不了该模型测试，无法验证。
- **P2 · `RayGenerationConfig.workload`/`EnginePlan.workload`（`WorkloadSignature`）每 plan 构造却从不读**
  （`generation/execution/planner.py:229-234` + `capabilities.py` 的 `batch_signature`）。整条 `workload→capability_key→batch_signature→supports_batched_*`
  死链。但 `WorkloadSignature` 是 public 导出、是 planned `ContinuousBatchingScheduler` 的前瞻 scaffolding
  （见 [[SPRINT_continuous_scheduler_redesign]]）。低风险替代：留类型、只删未读的 plumbing。**倾向不动**。
- **P2 · `score_aggregation` 是 typed 字段但只接受 `"sum"`、三处 `__post_init__` 各 guard 一遍**（`rewards/inference.py:91-92,119-120,153`）。
  多值假象。**安全子修复**：折叠重复 guard（零行为变化）。整字段删除触及被 `asdict()` 序列化进 manifest 的 frozen
  public dataclass（wire 变更）。
- **P2 · `MEDIA_TYPES` 把存储格式 `"tensor"` 混进真实媒体种类 `"image"/"video"`**（`rewards/inference.py:14,39-42`）。
  `"tensor"` 全仓无人设置、`_validate_media_shape` 对它 no-op。但 `MEDIA_TYPES` 是 exported protocol 常量，移除
  已接受的值是 API 收窄。
- **P2 · `DiffusionNFT` prepare-input hook 有两个名 + 永不触发的 getattr fallback**（`algorithms/diffusion_nft.py:179-186` +
  `cosmos predict2_5/model.py` 的 alias）。删 alias、collapse 成单 getattr 在当前仓安全，但收窄 protocol surface
  （外部模型作者的不成文承诺）。
- **P2 · `online.py` 重算 `rollout_batch_size // gas` 而非读 reconciled `microbatch_size`**（`scripts/common/online.py:92,142`
  ← `trainers/core/types.py` 的 `microbatch_size`）。仅日志/规划路径，低价值；两处改读 reconciled 字段更一致。
- **P3 · `OnlineRecipeStack.component_names` 写在 stack 上但从不从它读**（`scripts/common/types.py:80` + 写点 `online.py:560`）；
  所有消费者用闭包 local。stack 是 hook API，可能想作 reserved surface。
- **P3 · `reward_artifact_bytes` 只被自己测试用的一行 wrapper**（`rollouts/collector/artifacts.py:46-49`）；删除安全但移除一个
  public `__all__` domain-named alias。
- **P3 · `worker.py` 复刻 `import_from_path`**（`generation/execution/worker.py:467-475` vs `vrl/ray/dependencies.py:19-29`，逐字节同）。
  worker **刻意 Ray-agnostic**（不 import `vrl.ray`），直接 import 会破边界；保边界的修法是把 helper 迁中立家
  （`vrl/utils`）两边 import。
- **P3 · `RolloutStats.from_phase_dict` 是"migration boundary"但已无生产调用者**（`utils/stats.py:113-120`）；
  `continuous/test_contracts.py` 仍走它。**倾向保留**，仅删 docstring 里"migration boundary"措辞。
- **P3 · `requires_minimal_replay_loader` / `replay_modules` / `generation_only_modules` 每家族手列但生产不读**
  （`models/replay_loading.py` + 各 family runtime，含 anima `runtime.py:105-108` 传别家不传的 module 列表）。
  机械上删除保行为，但这是**未完成 minimal-replay 在制契约** + 跨家族声明式 manifest（`test_minimal_replay_runtime_wiring.py`
  断言每家族 replay/generation 拆分作 wiring 契约）。**故意不动**（护栏 2.2 + 前瞻基建）。

---

## 5. 用户两个原始例子的归宿

### 5.1 `data.loader` 两名字 dispatch —— 真丑、但 NOT 干净可派生（已实证）

对抗式 agent 专门去证伪"loader 可自动推导"，**推翻了**这个直觉，理由经我复核为真：

- **loader ⊥ format**：`grep -rn "loader:" configs/dataset` 显示 `prompt_manifest` 同时配 `format: text`
  （drawbench/ocr/pickscore/videophy）和 `format: jsonl`（anime_anatomy/geneval）。`format` 当前**根本不被任何
  dispatch 消费**——`load_prompt_manifest` 是按文件后缀 `.jsonl`/`.txt` 自动分流的（`prompts.py:39-52`）。
- **loader ⊥ conditioning**：`video_world_v2w.yaml` 用 `loader: prompt_manifest` 却带 `conditioning: reference_image`。
  "有 image 条件就走 image loader"会把它错路由进 `ImageCaptionPromptDataset`（要求每行有 `image`/`caption` 字段）
  直接坏掉。
- **第三个 loader**：`pickapic_preference`（离线 DPO，`scripts/data/bootstrap.py:67` 另处分发），无 `format`。

**真正的冗余更窄**：`videophy_i2v.yaml` 同时写 `task_type: image_to_video` 和 `loader: prompt_image_manifest`，
两者表达同一件事；且 `prompt_manifest` 已按后缀自动分流，唯独 image-vs-text 这层要手填魔法字符串——不一致。
**安全修法**：`_load_prompt_examples` 在 `loader` 缺失时按 prompt-* 家族 fallback（从 `task_type`/conditioning 推），
`loader` 仅保留区分 `pickapic_preference`；不强制全推导。**属 §4.1 P1 决策项，未自动改。**

> 教训：最"显然"的清理（直接推导掉 loader）会破坏线上配置。这正是 AGENTS.md "Evidence-First Work" 的实活——
> 先读 9 个 dataset yaml + dispatch + 三个 loader，再下结论。

### 5.2 release flags —— 拓扑派生后的扁平残留（归并到 resolved-struct 审计）

用户原始观点（"release 时机不该手填、应从拓扑派生"）已落地为 `RayLifecyclePlan`。审计发现残留：
`ResolvedDistributedResources` 上的 `reward_release_after_score`（`resources.py:134`）和
`rollout_release_before_reward_model`（`:131`）作为 **stored 字段**只被 `format_distributed_resource_plan`
日志行读（`:473-475`）；真实行为走 `resolved.lifecycle.*`——`reward_runtime_resource_kwargs` 读的是
`resolved.lifecycle.reward.mode == "on_demand"`（`:1036`），handoff 走 `lifecycle.handoff.*`。即这两个扁平字段
**影子化**拓扑派生的 lifecycle plan。

> ⚠️ **结论冲突**：[[SPRINT_resolved_struct_field_audit]] §3 把这两个字段标为 **NECESSARY**（traced
> "resources.py:850→release_after_call→ray/runtime.py:62"）。复核发现这是把 resolver **内部局部变量**
> （`:317`/`:321` 算出、喂给 lifecycle 构造 `:355-360` 与校验）的消费方，**误记到了 stored 字段头上**——
> 而那个 sprint 自己的 §4 判据正是"resolver 内部局部变量 ≠ stored field 必要"。按它自己的 rubric，这两个
> **stored 字段**应是 LOGGING_ONLY。
>
> **处置**：`ResolvedDistributedResources` 由 [[SPRINT_resolved_struct_field_audit]] 拥有，**本 doc 不重复
> 立项**。把"`reward_release_after_score` / `rollout_release_before_reward_model` 两个 stored 字段疑为
> LOGGING_ONLY，需按 local-vs-stored 判据复核、删则把日志行 `:473-475` 改读 `resolved.lifecycle.*`"作为
> 一条 follow-up 记到那个 sprint。涉及 public resolved struct + 多处测试断言，归一处处理避免双改冲突。

---

## 6. 误报记录（对抗式验证推翻的 8 个，记下以免后续重报）

| finding | 为何是误报（最强反驳） |
|---|---|
| `data.loader` 可由 `format`/conditioning 派生 | 实证 loader ⊥ format ⊥ conditioning，"直接推导"会破坏线上配置（§5.1） |
| `_dataclass_payload` 的 cross-section ignore set 是 rot-prone 常量 | `kind`/`kl_reward` **不在**任何算法 dataclass 字段里——它是故意的 cross-section pass-through（`kind` 是判别器、`kl_reward` 给 collector），非复制 typed 结构；两 helper 契约也不同（一个算 `missing` 必填、一个只过滤） |
| `request_prefix` 复制了 family 名 | `test_family_registry.py` 断言每条 entry 的 `request_prefix` truthy；且这是护栏要保的跨 family 并行注册形状；anima override 证明它是真 first-class knob |
| `drop_policy` 合法集在 queue 与 trainer config 双写 | 去重要让 `trainers/core/types.py` import `vrl.rollouts`，跨**故意的架构边界**（config 层不依赖 rollout 运行时；`schedule.py:81` 有明文注释）；2 值集低 rot 风险 |
| mode 合法集未从 `RolloutScheduleMode` enum 派生 | 同上——派生要让 config 层 import rollout enum，跨同一边界；mode 字符串还驱动 3 个 mode-specific 分支，enum 只能去重 2 元素成员检查 |
| `FamilyCapability` 14 个 `supports_*` 中 11 个无读者 | 候选具体断言有错（`supports_chunked_execution` 在 `chunks.py:122` 是真 gate、`supports_torch_compile` 在 `launcher.py:399` 被读）；其余是护栏要保的跨 family `runtime_caps` 表（与 [[SPRINT_resolved_struct_field_audit]] §5 同结论：低置信、需逐 flag 复核） |
| `ResolvedDistributedResources.visible_devices` 是 logging-only | 候选自己也承认它承载不可派生的"完整可见 GPU 池"溯源、`:460` 日志真消费、proposed_fix 就是"留着"（与 [[SPRINT_resolved_struct_field_audit]] §4 P1 一致：KEEP+注释） |
| GRPO/NFT 的 `compute_advantages_from_tensors` wrapper 重复 | 实际算法是共享的 `group_relative_advantages`；只重复 3 行 cfg-unpack，属两个**不同** config 类（GRPOConfig vs DiffusionNFTConfig）——抽取会建单一用途双调用 helper + 耦合两个独立 config，命中护栏 |

---

## 7. 非目标（明确不做）

- **不塞 `ALGORITHM_REGISTRY` / 不统一 factory 的 evaluator wiring**——三处 kind dispatch 职责真不同（§4.2）。
- **不 flatten 任何注册表 / 不 data 化跨 family 并行函数**（护栏 2.2）；不碰 `FAMILY_REGISTRY` 的 guarded 结构。
- **不跨架构边界去重 2-值集**（drop_policy / mode）——config 层故意不依赖 rollout 运行时（§6）。
- **不动前瞻基建**：`WorkloadSignature`/`ContinuousBatchingScheduler` scaffolding、minimal-replay 的
  `replay_modules`/`generation_only_modules`/`runtime_role` 契约、`weight_sync_barrier` 派生——除非确认对应在制
  工作不再用到。
- **不在本 doc 重复处理 `ResolvedDistributedResources` 字段**——归 [[SPRINT_resolved_struct_field_audit]]（§5.2）。
- **不为清理而清理**：只动"零消费方 / 逐字节复制 / 假旋钮 / 真 bug / 装饰命名"，其余视为正确。
- **不靠裸 grep 删**：§4 路径是 agent grep 级，动手前须像 §3 那样逐项读源码 + 跑测试复核。

---

## 8. 校验记录（对自动审计的独立复核）

- **方法可信度**：每个候选过了"survey + 对抗式验证"两道；用户的 loader 例子被当场证伪（§5.1）。8 个落地项
  我**逐个独立读源码 + grep 消费方 + 跑测试**复核（见 §3 表的"为何安全"列），未盲信 agent。
- **R1（`_apply_precision_policy` 重复）刻意未落地**：确认 `build_trainer_config`（`builders.py:178-183`）已设全
  4 个 precision 字段、`_apply_precision_policy`（`online.py:360`）重设同样 4 个=冗余；但它跑在 `configure_trainer`
  hook（`online.py:358-359`，当前无 family 提供）**之后**——这可能是"hook 不能覆盖 precision"的故意锁，属设计
  意图问题，留用户定。
- **R9/R10 与 resolved-struct 审计的冲突已查实并归并**（§5.2）：grep 全部消费方后确认两个 stored 字段只被日志行
  读、真实行为走 `resolved.lifecycle.*`；按 local-vs-stored 判据归 [[SPRINT_resolved_struct_field_audit]] 复核，
  避免双改 public struct 冲突。
- **环境缺口**：`.venv` 缺 `transformers`，`kling_video_reward` / `wan_train` preflight 相关测试在导入处即崩，
  **与本次改动无关**（同 [[test_env_torchvision_gap]]）；R13（kling checkpoint glob）因此无法本地验证、留 §4.3。

---

## 关键文件引用

落地项（§3）：
- `vrl/scripts/common/types.py:33-39`、`vrl/scripts/common/online.py:389-393,831-834`
- `vrl/rollouts/orchestration/types.py:21-26`、`strict_on_policy.py:80-81`、`continuous/schedule.py:122-133`
- `vrl/nn/layers/attention/paged.py:22-31`、`vrl/nn/modules/ar_attention_backends.py:117-134`
- `vrl/utils/memory.py:32-39,107-120`、`vrl/generation/ray/config.py:141`（已删 to_dict）
- `vrl/models/diffusion/cosmos/anima/runtime.py:150,251,279`、`vrl/algorithms/grpo/token.py:15-20`
- `vrl/config/schema.py:87-110`、`configs/base/algorithm/token_grpo.yaml`
- 测试：`tests/generation/ray/test_runtime_config.py:227-234`、`tests/models/interfaces/test_minimal_replay_runtime_wiring.py:287`

backlog 锚点（§4，agent grep 级，待复核）：
- `vrl/trainers/data/prompts.py:74-124`、`vrl/trajectory/builders.py:594-597`、`vrl/trainers/core/types.py:163-206`
- `vrl/config/builders.py:195-222`、`vrl/scripts/common/factory.py:62-65,132-137,171-267`、`vrl/config/schema.py:153-185,427-462`
- `vrl/rewards/models/kling_video_reward.py:31,629-675`、`vrl/rewards/inference.py:14,91-153`、`vrl/rewards/artifacts.py:28-33`
- `vrl/generation/execution/{planner.py:229-234,worker.py:467-475}`、`vrl/models/replay_loading.py`

相关 sprint：
- [[SPRINT_resolved_struct_field_audit]]（拥有 `ResolvedDistributedResources`；§5.2 的 release-flag follow-up 归它）
- [[SPRINT_continuous_scheduler_redesign]]（`WorkloadSignature` 前瞻消费方，§4.3）
- `docs/sprints/done/SPRINT_allcaps_constants_audit.md`、`done/SPRINT_small_function_consolidation.md`（同类架构卫生审计先例）
