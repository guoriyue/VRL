# SPRINT: model families 死代码与 `LORA_DEFAULTS` 去重（planned）

状态：**planned（2026-07-23）**。本簇 32 条已对抗验证的死代码/去重发现（10 条 medium + 22 条 low），覆盖 `vrl/models/families/*` 各家族与 `vrl/models/steps`、`vrl/models/precision`、`vrl/models/utils`、`vrl/families/` 共享层。
来源：dead-code-audit workflow（五种死代码形态 + 对抗验证 + 删除类二次字符串引用检查）。
关联：[[SPRINT_deadcode_00_overview]]、[[SPRINT_trajectory_views_types_dead_fields_cleanup]]（死字段规则同源）、[[SPRINT_generation_regime_decision_layering]]（`PolicySemantics` 为分发 source of truth，约束 §1.2 的落地方向）、[[SPRINT_sampling_config_knob_unification]]（`final_image_policy`/`guidance_scale_2` 的收敛记录，约束 §1.13/§1.23）、[[SPRINT_dead_code_wrapper_sweep]]（`count_trainable_params` 的历史合并意图，约束 §1.32）、[[SPRINT_native_generation_engine_program]]（**in-flight**，§1.27 与其未提交改动同文件，须排在其后）。

## 0. 一句话

本簇是 model families 层的死代码清理：主形态是**死字段/死参数/死配置旋钮**（form 1/2）与 **form-4 重复实现**。最锋利的一条是五个 token 家族（`emu3`/`janus_pro`/`glm_image`/`nextstep_1`/`llamagen`）各自在 `runtime.py` 手维护一份 `_*_LORA_DEFAULTS` dict，同时又在 `model.py` 的 Config dataclass 上维护同一份 `lora_*` 默认值——两份已经漂移（4/5 家族的 `target_modules` 不一致），正是 AGENTS.md 「手维护常量复制类型结构会静默腐烂」的教科书案例。误删风险集中在两点：**（a）** `runtime_capabilities`/`PolicySemantics` 上的死字段与其**同结构里的活字段**混杂，删字段而非删结构；**（b）** `LORA_DEFAULTS` 去重时必须保留 dataclass 默认值这一侧（它被 `if config.use_lora` 分支真实消费、被直接构造与测试 PEFT-wrap 复用），删掉的是 `runtime.py` 那份从不生效的 fallback dict。

## 1. 待删清单（逐条，带证据与动作）

> 按家族分组，便于逐家族 review；每条标注 risk。medium（共 10 条）：§1.1、§1.2、§1.9、§1.11、§1.12、§1.13、§1.14、§1.18、§1.23、§1.26。

### families/ 共享层

#### 1.1 `GenerationRuntimeCapabilities.runs_in_isolated_subprocess` — dead-field（risk=medium）

- 位置：`vrl/families/registry.py:41`（定义）、`:463`（`magi_1` 处 `runs_in_isolated_subprocess=True`）
- 判死证据：
  ```
  $ grep -rn 'runs_in_isolated_subprocess' vrl/ tests/
  vrl/families/registry.py:41:    runs_in_isolated_subprocess: bool = False
  vrl/families/registry.py:463:            runs_in_isolated_subprocess=True,
  tests/models/steps/denoise/common/test_vae_decode_memory.py:329: if not entry.runtime_capabilities.runs_in_isolated_subprocess:
  tests/rollouts/runtime/test_family_registry.py:141: assert causvid.runs_in_isolated_subprocess is False
  tests/rollouts/runtime/test_family_registry.py:143: assert magi.runs_in_isolated_subprocess is True
  ```
  非测试的 `runtime_capabilities` 生产消费者只读 `memory_parking`（`vrl/generation/execution/memory_parking.py:92`）、`supports_policy_replay`（`vrl/scripts/common/factory.py:177`、`registry.py:258`）、`supports_torch_compile`（`vrl/generation/ray/launcher.py:236`）、`accepts_samples_per_chunk`（`launcher.py:297`）——`runs_in_isolated_subprocess` 只有测试 reader。无 YAML/序列化/动态派发路径；`docs/sprints/` 无任何计划中的 consumer。按 AGENTS.md 死字段规则，test-only reader = 死。
- 动作：删 `registry.py:41` 字段与 `:463` 的 setter 行（保留同一 `magi_1` entry 上的 `supports_policy_replay=False`）。改 `tests/models/steps/denoise/common/test_vae_decode_memory.py:327-337`：把「按 capability 选择」替换为**显式硬编码的 subprocess 家族集合 `{"magi_1"}`**，并加注释说明这些 runtime 在上游进程内自持 decode 显存；**不要**改用 `supports_policy_replay` 作 key。删 `tests/rollouts/runtime/test_family_registry.py:141,143` 两条断言，并把 `test_chunk_family_trainability_and_process_boundaries_are_explicit` 重命名为 `test_chunk_family_trainability_is_explicit`（保留 `:140,142` 的 `supports_policy_replay` 断言）。
- 注意（medium）：`supports_policy_replay=False` 与 `runs_in_isolated_subprocess` 只在 `magi_1` 上**恰好同真**，是两个不同概念（可 replay 性 vs 进程隔离）。`test_vae_decode_memory.py` 用该字段**强制一个真实不变量**（隔离 runtime 不得调用 `apply_generation_memory_policy`、且必须声明 `rollout_runtime_builder`）；若改用 `supports_policy_replay` 会给未来任何「不可 replay 但在进程内」的家族错误定界。故必须硬编码家族集合而非借用另一字段。

#### 1.2 `PolicySemantics.trajectory_layout` — dead-field（risk=medium）

- 位置：`vrl/families/semantics.py:38`（定义）、`:49,54,56,62`（自身 `__post_init__`）；构造点 `vrl/families/registry.py:341,357,370,396,720`
- 判死证据：
  ```
  $ grep -rn 'trajectory_layout' vrl/ tests/
  vrl/families/registry.py:341,357,370,396,720   （构造/builder 参数默认）
  vrl/families/semantics.py:38                    （定义）
  vrl/families/semantics.py:49,54,56,62           （仅自身 __post_init__ 自校验）
  tests/scripts/test_online_lifecycle.py:256      （测试构造）
  tests/rollouts/runtime/test_family_registry.py:43（测试构造）
  ```
  唯一 reader 是它自己的 `__post_init__` 内部一致性校验（对硬编码 registry literal 的自校验，非 dead-field 规则下的 consumer）。真正的 multisegment 路由在 `vrl/rollouts/collector/batch_builder.py:308-316` 上，是按 `trajectory.family == "janus_pro_r1" or trajectory.task == "ar_t2i_r1"` 硬编码的字符串——不读此字段。`grep 'ar_t2i_r1'` 确认该 task 只由 `janus_pro_r1` 家族产出（`registry.py:716`、`janus_pro/runtime.py:245`），即 layout 检查可同时替代该条件的两半（form-4 data twin）。同结构里 `generation_regime`/`step_kind`/`action_distribution` 均有真实消费者（`factory.py:220,289`、`online.py:908,982,1061`、`registry.py:122`），故结构活、字段死。
- 动作：**取「使字段成为行为 source of truth」这一支（放弃删除支）**，因为删除支违背 [[SPRINT_generation_regime_decision_layering]]（`semantics.py:119`「PolicySemantics 是 source of truth；生产分发读取 semantics」）。具体：(1) 给 `RolloutBatchBuildContext`（`vrl/rollouts/collector/batch_builder.py:26-35`）新增 `trajectory_layout` 字段，默认 `None`；(2) 在 `vrl/rollouts/collector/core.py:227` 从 collector 构造时已持有的 resolved `ModelFamilyEntry`（`build_rollout_collector`，`core.py:351-363` 接收 `entry`）取 `entry.policy_semantics.trajectory_layout` 穿线进 `RolloutCollector`（与已存储的 `request_builder`/`entry` 并列）；(3) 把 `_is_multisegment_categorical`（`batch_builder.py:308-316`）里的 family/task 硬编码替换为 `self.context.trajectory_layout == "multisegment_token"`，保留既有 `len(trainable) > 1 and all categorical` 通用 fallback；(4) 更新直接构造 `RolloutBatchBuildContext` 的 multisegment 测试点 `tests/rollouts/runtime/test_janus_pro_r1_wiring.py:150-152`（传入 layout，或确认走通用 fallback）；`tests/rollouts/collector/test_runtime.py`、`tests/generation/bindings/*` 用的是非 multisegment trajectory，无需改。(5) `semantics.py` 与 `registry.py` 保持不动——字段、`Literal`、`__post_init__` 交叉规则在第 (3) 步落地后即成活。
- 注意（medium）：`batch_builder.py:312` 处 `TrajectoryRolloutBatchBuilder` **作用域内没有 registry entry**（只收 `GenerationOutput` + `RolloutBatchBuildContext`），故 layout 必须从 collector 侧已 resolve 的 entry 穿线下来，不能就地 `entry.policy_semantics...`。这是本条唯一实现难点。
- ⚠ 复核偏差：finding 记录的测试构造点为 `tests/scripts/test_online_lifecycle.py:237`，本次复核 grep 实测在 `:256`（同一 `trajectory_layout="denoise"` 构造，仅行号漂移 19 行；`registry.py`/`semantics.py` 所有行号与 finding 完全一致，判死结论不受影响）。该测试文件不在本条动作范围内，仅作构造点存在性佐证。

### causvid（families-video）

#### 1.3 `CausVidResolvedArtifacts.{source_root, source_revision, base_model_revision, checkpoint_revision}` — dead-field（risk=low）

- 位置：`vrl/models/families/causvid/model.py:54-63`（定义）、`:852-869`（`_resolve_artifacts` 构造）
- 判死证据：
  ```
  $ grep -rn 'CausVidResolvedArtifacts' vrl tests
  model.py:55 (class def), :852 (返回注解), :862 (构造), :1152 (__all__)
  $ grep -n 'artifacts' model.py  →  仅读 :794/:808 artifacts.base_model_dir、:796 artifacts.checkpoint_file
  ```
  四个字段零 reader（无属性读、无 log、无 `asdict/fields` 迭代）。`CausVidResolvedArtifacts` 无外部 importer（registry 只按字符串派发 `CausVidModel`/`CausVidReplayModel`）。`_resolve_artifacts` 唯一 caller 是 `_load_official_backend:788`。causvid 测试只 import 私有 resolver 函数（`test_replay_and_loading.py:27-29`），从不 import 该 struct。注：`model.py:916` 的字符串 `"...requires immutable model.base_model_revision"` 是 YAML 旋钮名的报错文案，非属性读——`_resolve_source_root`/`_resolve_checkpoint` 因其校验副作用保留。
- 动作：删四个死字段，把 `CausVidResolvedArtifacts` 收缩为 `base_model_dir + checkpoint_file`（或化为 `_resolve_artifacts` 返回的二元组），并从 `__all__`（`model.py:1152`）删除该名。`_resolve_source_root`/`_resolve_checkpoint` 继续因校验副作用运行；无测试清理。

#### 1.4 `CausVidCausalRunner.__init__(math_dtype=...)` — dead-arg（risk=low）

- 位置：`vrl/models/families/causvid/runner.py:214-225`（构造参数 + `self.math_dtype`）、`:280`、`:331`（两处使用）
- 判死证据：
  ```
  $ grep -rn 'CausVidCausalRunner(' vrl tests
  model.py:656; tests/.../test_runner.py:124; tests/.../test_replay_and_loading.py:194   （三处构造，均只传 backend+geometry=+schedule=，无 math_dtype）
  $ grep -rn 'math_dtype' vrl/models/families/causvid/
  runner.py:220,225,280,331; model.py:548: math_dtype=torch.float32   （replay scorer 硬编码 fp32）
  ```
  无 YAML/config 生产者（`precision.diffusion_math` 只喂 `DiffusionSDELogProbEvaluator`，causvid 分支在 `factory.py:253-260` 提前返回 `ChunkAutoregressiveDenoiseLogProbEvaluator()`，无 `math_dtype` 属性）。
- 动作：删构造参数与 `self.math_dtype`，在两处使用点内联 `torch.float32`（`:280` 亦可直接省略该 kwarg——`vrl/math/denoise/renoise.py:57` 对 `None` 默认 fp32）。无测试清理。
- 注意（parity）：replay 侧 `model.py:548` 硬编码 `math_dtype=torch.float32`，任何非 fp32 的 rollout 取值都会静默破坏 rollout/replay log-prob parity——此旋钮只会致害，fp32 是正确性常量而非偏好。

#### 1.5 `CausVidCausalBackend.predict_x0_cached(finalize_cache=...)` — dead-arg（risk=low）

- 位置：`vrl/models/families/causvid/runner.py:143`（protocol）、`:269,292,304`（三处 call site）；实现 `model.py:182-185`
- 判死证据：
  ```
  $ grep -rn 'finalize_cache' vrl tests
  runner.py:143 (protocol), :269 False, :292 False, :304 True
  model.py:182 (param), :185 del finalize_cache   （唯一生产实现直接丢弃）
  tests/.../test_replay_and_loading.py:126,137 (fake 分支); test_runner.py:34,72,83,151 (fake 记录并断言)
  ```
  唯一生产实现 `model.py:185` 以 `del finalize_cache` 开头，KV cache 通过 `current_start/current_end` 每次 forward 无条件提交，flag 无行为效果。`finalize_cache=True` 唯一 call site（`runner.py:298-306`）恰是 `timestep == schedule.cache_timestep` 的那次；`CausVidSchedule.__post_init__` 硬校验 `cache_timestep == 0`（`runner.py:88-89`），仓内唯一构造的 schedule 是 `OFFICIAL_CAUSVID_SCHEDULE`（`prediction_timesteps=(1000,757,522)`），故 `timestep==0` 唯一标识 finalize 调用——flag 与已在线上的 timestep 信息冗余。
- 动作：从 protocol、`_OfficialCausVidBackend.predict_x0_cached`（其 body 以 `del finalize_cache` 起）、三处 runner call site 删除 `finalize_cache`；把 `tests/models/families/causvid/test_runner.py`、`test_replay_and_loading.py` 的 fake backend 改为按 `timestep == schedule.cache_timestep`（0）判定。保留 protocol 边界本身。

### cosmos/anima（families-video）

#### 1.6 `TransformerBlock.__init__(use_self_attn=...)` — dead-arg / form 2（risk=low）

- 位置：`vrl/models/families/cosmos/anima/adapter.py:53-93`
- 判死证据：
  ```
  $ grep -rn 'use_self_attn' vrl tests
  adapter.py:24 (唯一 True 生产者), :60 (定义), :63/:64/:84 (self 赋值 + 两处 guard)
  ```
  唯一构造点 `AnimaLLMAdapter.__init__`（`adapter.py:20-27`）对全部 6 个 block 传 `use_self_attn=True`；`False` 分支会构造 cross-attn-only block，其 checkpoint key（无 `norm_self_attn.`/`self_attn.`）匹配不到任何已发布 Anima checkpoint。anima 测试用 `torch.nn.Identity()` stub adapter，不触 `TransformerBlock`。
- 动作：删 `use_self_attn` 参数、`self.use_self_attn` 属性、`__init__` 与 `forward` 里两处 `if self.use_self_attn:` guard，无条件构造 `norm_self_attn`/`self_attn`。module 名与 state-dict key 不变，`_load_anima_llm_adapter`（`model.py:550`）的 `strict=True` 加载不受影响。无测试清理。

### echo（families-video）

#### 1.7 `EchoModel._lora_transformer` — duplicate-impl / form 4（risk=low）

- 位置：`vrl/models/families/echo/model.py:145-146`
- 判死证据：
  ```
  $ grep -rn '_lora_transformer' vrl tests
  vrl/models/steps/denoise/common/lora.py:32 (LoraModelMixin 默认, body: return self.transformer)
  vrl/models/steps/denoise/common/lora.py:57 (唯一 call site)
  vrl/models/families/echo/model.py:145 (override, body 字节等同: return self.transformer)
  ```
  override body 与 mixin 默认逐字节相同。Echo 满足默认契约：`model.py:132` `self.transformer = echo.model`，`model.py:141-143` `_set_transformer` 同步。约 15 个 `LoraModelMixin` 家族中只有 Echo override 此方法；另一个 pipeline-less 家族 `AnimaModel` 已直接用默认。MRO 安全：`EchoModel(LoraModelMixin, DiffusionModelBase)`，`DiffusionModelBase`（含其 §1.27 的未提交改动）既不定义 `_lora_transformer` 也不定义 `transformer` property。
- 动作：删两行 override。无测试清理。
- 注意（sprint 关联）：MRO fall-through 依赖 `vrl/models/steps/denoise/base.py`（in-flight sprint 修改文件）不引入 `_lora_transformer`/`transformer` property——复核已确认其未提交 diff 仅动 `_set_transformer`/`export_batch_context`，安全；建议落地时再 grep 一次 `base.py` 确认。

### emu3（families-token）

#### 1.8 `Emu3ReplayModel.text_config` — duplicate-impl / form 4（risk=low）

- 位置：`vrl/models/families/emu3/model.py:696-698`
- 判死证据：
  ```
  $ grep -rn 'text_config' vrl/models/families/emu3/model.py
  :265-266  Emu3Model.text_config → return self.emu3.config.text_config
  :377      eos = getattr(self.text_config, "eos_token_id", None)   （唯一行为消费者）
  :697-698  Emu3ReplayModel.text_config → 字节相同
  ```
  MRO `Emu3ReplayModel(ARReplayRolloutStubs, Emu3Model)`（`:658`），`ARReplayRolloutStubs`（`vrl/models/steps/token/base.py:90-101`）只定义 `decode_image_tokens`，不 shadow `text_config`。replay 的 `self.emu3` 是 `Emu3ReplayCore`，其 `__init__` 有 `self.config = config`（`:646`），`.config.text_config` 解析一致。姊妹 replay 模型 `GlmImageReplayModel`（`glm_image/model.py:789`）继承基类 `text_config` 无 override，`JanusProReplayModel`（`janus_pro/model.py:1073`）根本无此 property。无 `"text_config"` 字符串派发。
- 动作：删 `:696-698` 三行 override。无测试清理。

#### 1.9 五个 token 家族的 `_*_LORA_DEFAULTS` dict vs Config dataclass `lora_*` 默认值 — duplicate-impl / form 4-data（risk=medium）

- 位置：`emu3/runtime.py:22-28` vs `emu3/model.py:71-81`；`janus_pro/runtime.py:38-44` vs `model.py:84-96`；`glm_image/runtime.py:21-27` vs `model.py:87-98`；`nextstep_1/runtime.py:30-36` vs `model.py:69-81`；`llamagen/runtime.py:23-29` vs `model.py:74-79`
- 判死证据：
  ```
  $ grep -rn '_LORA_DEFAULTS' vrl tests
  nextstep_1/runtime.py:30 (def), :42 (token_model_config_base(build, _NEXTSTEP_LORA_DEFAULTS))
  emu3/runtime.py:22, :33 ; llamagen/runtime.py:23, :35 ; glm_image/runtime.py:21, :32 ; janus_pro/runtime.py:38, :50
  ```
  每个家族把同一份 LoRA 默认值维护两遍：`runtime.py` 的 module-level dict 与 `model.py` 的 Config dataclass `lora_*` 字段。两份已漂移——`rank/alpha/dropout/init` 一致，但 `target_modules` 在 4/5 家族不一致（`runtime.py` 的 `('q_proj','v_proj')` vs dataclass 的 4-projection 集合；`llamagen` 两侧都是 `('wqkv','wo')` 却仍重复）。`token_model_config_base`（`vrl/models/steps/token/build.py:46-57`）做 `lora = dict(lora_defaults); lora.update((build.model_config or {}).get("lora") or {})`——YAML block 覆盖在 dict 之上。所有五个 model preset（`vrl/config/presets/model/{emu3/gen_9b,glm_image/9b,janus_pro/1b,llamagen/xl_256,nextstep_1/1_1}.yaml`）都显式给全 `lora` block（含 `target_modules`），故**生产训练用的是 4-projection adapter，与 dataclass 默认一致**；是 `runtime.py` 那份 dict 从不生效。dataclass 默认被真实消费：`config or Emu3Config()` 直接构造（`emu3/model.py:203,668` 等）、`use_lora=True` 测试 PEFT-wrap（`tests/models/families/emu3/test_replay.py:164`、`glm_image/test_replay.py:186`）在 `if config.use_lora` 分支读 `lora_rank/alpha/target_modules/dropout`。
- 动作：**保留 Config dataclass 的 `lora_*` 默认值为唯一 source of truth，删除五个手维护的 `_*_LORA_DEFAULTS` dict。** 改 `token_model_config_base`（`vrl/models/steps/token/build.py`）接收家族 Config 类（各 `runtime.py` 已 import 其 model 模块），从 dataclass 字段默认（如 `dataclasses.fields` 查默认）构建 merge-base `{"rank": cls.lora_rank, "alpha": cls.lora_alpha, "target_modules": cls.lora_target_modules, "dropout": cls.lora_dropout, "init": cls.lora_init}`，YAML `model.lora` block merge 逻辑不变。**不改任何 YAML preset，不删/改 dataclass 默认值。** 行为 delta 仅限从不触发的 fallback：`emu3/janus_pro/glm_image/nextstep_1` 的 fallback `target_modules` 从 `('q_proj','v_proj')` 变为 4-projection 集合——与每个生产 preset、每个 dataclass 默认、preset 自身注释已经声明的一致（`llamagen` 本已一致）。无测试引用 `_*_LORA_DEFAULTS`，无测试清理。
- 注意（medium）：finding 的原始「生产 YAML 路径总是应用 `('q_proj','v_proj')`」前提是**反的**——生产实际用 4-projection（来自 YAML preset 与 dataclass 默认）；死的是 `runtime.py` dict 那份从不生效的 fallback。因此必须**删 dict、保留 dataclass 默认值**，切勿反向删 dataclass 默认（会破坏直接构造与 `use_lora=True` 的 PEFT-wrap 测试）。落地后跑 `tests/models/families/{emu3,janus_pro,glm_image,nextstep_1,llamagen}/` 与 `tests/models/steps/token/`。

### glm_image（families-token）

#### 1.10 `GlmImageChunkExecutor._runner_cls / _runner_attention_family` — dead-field / form 1（risk=low）

- 位置：`vrl/models/families/glm_image/runtime.py:80-81`
- 判死证据：
  ```
  $ grep -rn '_runner_cls\|_runner_attention_family' vrl tests
  executor.py:44-45 (基类默认), :95-103 (唯一生产 reader, 在基类 _ar_runner body 内)
  glm_image/runtime.py:80-81 (声明)
  tests/models/families/glm_image/test_model_build.py:86-87 (仅有的 reader: assert executor._runner_cls is GlmImageTokenRunner / == "glm_image")
  ```
  `GlmImageChunkExecutor` 完全 override `_ar_runner`（`runtime.py:97-115`，返回 `GlmImageTokenRunner(self.model)`，从不 `super()`、从不查这两个属性，对显式 `attention_backend` 请求 raise）。`GlmImageTokenRunner.__init__(self, model)`（`runner.py:73`）不接受 `attention_backend` kwarg，基类路径即便被走到也会 TypeError。无子类、无字符串派发、`"glm_image"` 未在 `vrl/nn/modules/ar_attention_backends.py` 注册。test-only reader = 死。
- 动作：删两个类属性 + 删 `tests/models/families/glm_image/test_model_build.py:86-87` 两条断言（保留 `:84-85` family/task 断言与 `:90-101` 的 `test_executor_rejects_explicit_attention_backend`）。

### janus_pro（families-token）

#### 1.11 `JanusProModel.decode_image_tokens(image_size=...)` — dead-arg / form 2 + 无效用户旋钮（risk=medium）

- 位置：`vrl/models/families/janus_pro/model.py:975-1000`
- 判死证据：
  ```
  $ grep -rn 'image_size' vrl/models/families/janus_pro/
  runtime.py:164  image_decode_kwargs={"image_size": params.image_size}
  runtime.py:286  image_size=params.image_size   （→ generate_with_refine）
  model.py:510    generate_with_refine(image_size: int = JANUS_IMAGE_PIXEL_SIZE)
  model.py:573/693 decode_image_tokens(..., image_size=image_size)
  model.py:783    "image_size": int(image_size)   （context 写入, 零 reader）
  model.py:980    decode_image_tokens(image_size: int = JANUS_IMAGE_PIXEL_SIZE)   （body 从不引用它）
  ```
  `decode_image_tokens` body 只用 `B, L, side, latent_channels`；输出尺寸完全由 VQ decoder 决定。旋钮却是 required-and-set：preset `discrete_image_384_576tok.yaml:4`/`r1_image_384_576tok.yaml:4` 设 `image_size: 384`，`layout.parse_sampling_params` 因 `default_image_size=None` 使其必填。`context["image_size"]` 无 reader（context 读的是 temperature/guidance_scale/primary_segment/segment_order/image_height/image_width）。对照：`LlamaGenModel.decode_image_tokens`（`llamagen/model.py:436-440`）对同参数做校验而非忽略——这是修法。
- 动作：**取校验支（放弃删除支）**。在 `decode_image_tokens`（`model.py:975-1000`）算出 `side` 后，当 `int(image_size) != side * JANUS_IMAGE_PATCH_SIZE` 时 raise `ValueError`，镜像 `llamagen/model.py:431-440`。这把无效旋钮变成每条路径（executor `image_decode_kwargs` 与 `generate_with_refine` 两处 `:573/:693`）上的一致性检查，YAML preset、`ARSamplingParams`、`JanusProChunkExecutor`、`generate_with_refine` 签名、现有测试全不改（fixture 现值均一致：384=√576·16、32=√4·16、64=√16·16）。保留 `context["image_size"]`（`model.py:783`），校验后即准确。可选加一条断言 mismatch 会 raise 的小测试（对照 `tests/models/families/llamagen/test_model.py:121-132`）。**不删**：`ARSamplingParams.image_size`（共享、被 nextstep_1 `runtime.py:184` 消费、被 llamagen 校验）、`image_decode_kwargs`（共享 protocol 字段）、YAML sampling key。
- 注意（medium）：删除支不可行——`ARSamplingParams.image_size` 是共享 required 字段，janus 单方面去掉必填只能靠注入 dummy `default_image_size` 或破坏共享 layout，blast radius 远超本条。校验支是零改动且与仓内 prior art 一致的正解。

#### 1.12 janus sampling `'task_stages'` + `_parse_task_stages` — dead-config-knob / form 2（risk=medium）

- 位置：`vrl/models/families/janus_pro/runtime.py:283,420-425`；`model.py:507,523-528,786,~611-626,681,700`
- 判死证据：
  ```
  $ grep -rn 'task_stages' vrl tests
  model.py:507 (param 默认 JANUS_R1_SEGMENTS), :523 (stages=tuple), :786 (context 写入, 零 reader)
  runtime.py:283 (sampling.get 读), :420 (_parse_task_stages def)
  test_r1_model.py:322 (fake param), :332 (del)   ← fake 仅接收 runtime 转发的 kwarg
  ```
  `'task_stages'` sampling key **零生产者**：不在 `SamplingConfig`（`schema.py:313-350`，声明的 key registry；姊妹旋钮 `max_reflect_len` 在册），不在任何 YAML preset，无 collector/test request 设置。生产值恒为 `None → JANUS_R1_SEGMENTS`，故 `_parse_task_stages` 的 str-split/tuple 分支零生产者；`context["task_stages"]` 零 reader。
- 动作：`runtime.py` 删 `_parse_task_stages`（`420-425`）与 `:283` 的 `task_stages=_parse_task_stages(sampling.get("task_stages"))` kwarg。`model.py` 删 `task_stages` 参数（`:507`）、`stages = tuple(task_stages)`（`:523`）与「unknown-stage / requires-initial_image」校验块（`:523-528`，pinned 后 vacuous）、`"task_stages": stages` context 项（`:786`）、`stages` else 分支里失去唯一 selector 的 `selfcheck_text`（`~611-626`）。`final_image` else 臂**不删**：`:681` 由 `if "final_image" in stages and mode != "never":` 简化为 `if mode != "never":`，`:700` 由 `if "final_image" not in stages or mode == "never":` 简化为 `if mode == "never":`（`mode == "never"` 经 `sampling.refine_mode`/`config.r1_refine_mode` 仍可产出，见 §1.13）。保留 `JANUS_R1_SEGMENTS`（活的 segment taxonomy：`runtime.py:446`、trajectory 断言）。`tests/models/families/janus_pro/test_r1_model.py`：从 `_ExecutorModel.generate_with_refine` fake 签名删 `task_stages` kwarg 及其 `del`（`:321,332`）。
- 注意（medium）：`model.py:681/700` 的 `final_image` 臂是 `stages` 与 `mode` 的复合条件，只可删 `stages` 半、保留 `mode == "never"`（独立可产出的 selector）。本条须与 §1.13 一并落地以保持 `mode` 分支语义一致。

#### 1.13 janus sampling `'refine_mode'` fallback + `JanusProConfig.r1_refine_mode`（+ `mode=='never'` 分支）— dead-config-knob / form 2（risk=medium）

- 位置：`vrl/models/families/janus_pro/runtime.py:428-439`；`model.py:102,533-535,681-707`
- 判死证据：
  ```
  $ grep -rn 'refine_mode\|r1_refine_mode' vrl tests
  model.py:102  r1_refine_mode: str = "selfcheck"  # "selfcheck"|"always"|"never"
  model.py:533  mode = (refine_mode or self.config.r1_refine_mode).lower()
  runtime.py:436 sampling.get("refine_mode", ...); :437 getattr(..., "r1_refine_mode", "selfcheck")
  tests only: test_r1_model.py:105,310 ; fixtures.py:233 (docstring)
  ```
  对 `janus_pro_r1`（唯一到达此码的家族，`schema.py:917-920` 强制 `algorithm.kind=token_grpo_multisegment`），`rollout.final_image_policy` 为 REQUIRED 且须 `'always_generate'|'use_selfcheck'`（`schema.py:924-935`），collector 把它 flat-merge 进 `request.sampling`，故 `_resolve_refine_mode` 恒走前两分支，fallback 在任何校验过的 run 里不可达。`sampling['refine_mode']` 无生产者；`r1_refine_mode` 从不被 `janus_config_from_build` 转发（`runtime.py:47-67` 只转 `guidance_scale/temperature/image_token_num` 等），只有测试 setter = TEST-ONLY = 死。于是 `mode` 只能是 `'always'|'selfcheck'`，`model.py:681,700-701` 的 `'never'` 分支零生产者。
- 动作：`_resolve_refine_mode`（`runtime.py:428-439`）删 `sampling.get('refine_mode', getattr(..., 'r1_refine_mode', ...))` fallback，改为当 `sampling['final_image_policy']` 不属 `{'always_generate','use_selfcheck'}` 时 raise（镜像 schema 合法性检查），删无用的 `model` 参数与 call site 实参（`runtime.py:287`）。`model.py`：删 `JanusProConfig.r1_refine_mode`（`:102`）；**改写 `:533` 去掉 config fallback**——把 `refine_mode` 设为必填 str kwarg，或默认 `'selfcheck'`（`mode = (refine_mode or 'selfcheck').lower()`）以保直接调用 API 可跑；把 `:534-535` 校验集收缩为 `{'selfcheck','always'}`；`:681` 改 `if "final_image" in stages:`，`:700` 改 `if "final_image" not in stages:`（保留 stages 条件——见 §1.12）。测试：`test_r1_model.py` 删 `build_stub_janus_model(..., r1_refine_mode="selfcheck")`（`:105`）与 `_ExecutorModel` 的 `config = SimpleNamespace(r1_refine_mode="selfcheck")`（`:310`）；`tests/models/steps/token/fixtures.py:233` docstring 去掉 `r1_refine_mode` 提及（及 stub-config kwarg，若有）。跑 `tests/models/families/janus_pro/`、`tests/rollouts/runtime/test_janus_pro_r1_wiring.py`、`tests/config/test_schema.py`。
- 注意（medium）：`model.py:533` 读的正是要删的字段，动作必须显式给出 `:533` fallback 的改写（否则直接删字段会给用公开签名默认 `refine_mode=None` 的直接调用者留下 `AttributeError`）。本条与 §1.12 强耦合，须同批落地。关联 [[SPRINT_sampling_config_knob_unification]]（`final_image_policy` 已收敛为 rollout 单一 source、仅两枚举值）。

### llamagen（families-token）

#### 1.14 `LlamaGenModel._lm_trunk` — dead-function / form 1（risk=medium）

- 位置：`vrl/models/families/llamagen/model.py:183-193`
- 判死证据：
  ```
  $ grep -rn '_lm_trunk' vrl tests
  vrl/nn/modules/ar_attention_backends.py:88,136,176-179 (唯一通用 consumer, 经 resolve_attention_backend, 仅由基类 _ar_runner 调)
  llamagen/model.py:175 (docstring 提 Janus 的 _lm_trunk), :183 (定义)
  其余 _lm_trunk hit 均为 janus_pro/emu3/nextstep_1/glm_image 各家族自实现自消费
  llamagen 家族与测试只用 _gpt_trunk (runner.py:83,215; model.py:150,193,...; test_model.py:100,109)
  ```
  `LlamaGenChunkExecutor` override `_ar_runner`（`runtime.py:81-97`）直接构造 `LlamaGenARModelRunner(self.model)` 并对任何 `attention_backend` 请求 raise，故没有生产者能把 llamagen model 交给 backends。body 是一行 `return self._gpt_trunk()`，docstring 自述「shared attention backends ... cannot drive it」——协议已不存在。parked `SPRINT_attention_kernel_medium.md` 只为 Janus/NextStep 规划 `_lm_trunk` driver，零 llamagen 提及。
- 动作：删该方法（保留 `_gpt_trunk`——真实 accessor，被 model 与测试用）。删除后 `build_torch_native_backend(llamagen_model)` 会在 `ar_attention_backends.py:179` fail-fast 而非深入 vendored GPT，失败模式反而更好。无测试清理。
- 注意（medium）：这是「误导性 capability」而非跨家族一致性——保留会宣告一个家族无法参与的协议。`_gpt_trunk` docstring 里对「Janus' `_lm_trunk`」的提及指的是 janus_pro 的 hook，保持准确。

#### 1.15 `LlamaGenChunkExecutor._runner_cls / _runner_attention_family` — dead-field / form 1（risk=low）

- 位置：`vrl/models/families/llamagen/runtime.py:69-70`
- 判死证据：见 §1.10 同一 grep。唯一 reader 是基类 `ARChunkExecutorBase._ar_runner`（`executor.py:95-103`），`LlamaGenChunkExecutor` 完全 override（`runtime.py:81-97`，`return LlamaGenARModelRunner(self.model)`，从不查属性，对 `sampling.attention_backend` raise）。`LlamaGenARModelRunner.__init__(self, model)`（`runner.py:54`）无 `attention_backend` kwarg，基类路径即便走到也 TypeError。无 llamagen 测试读这两个属性（测试 reader 只有 emu3/glm_image）。
- 动作：删两个类属性。无测试清理。

### lumina2（families-image）

#### 1.16 `Lumina2Model.encode_prompt(system_prompt kwarg)` — dead-arg / form 2（risk=low）

- 位置：`vrl/models/families/lumina2/model.py:185`
- 判死证据：
  ```
  $ grep -rn 'system_prompt' vrl tests
  lumina2/model.py:11,166 (docstring), :185 (system_prompt=kwargs.get("system_prompt"))
  vrl/config/presets/experiment/lumina2/online_grpo_pickscore_validation.yaml:3 (# 注释行, 非 config key)
  ```
  唯一读点即 `:185` 本身。所有 `encode_prompt` call site（`executor.py:135/624`、`scripts/...`）只转发 `max_sequence_length/guidance_scale/request/reference_image`，无自由 dict 转发，故 `kwargs.get("system_prompt")` 恒为 `None`。installed diffusers `Lumina2Pipeline.encode_prompt` 签名 `system_prompt: str | None = None`，None 时套默认模板。
- 动作：把 `system_prompt=kwargs.get("system_prompt")` 改为显式 `system_prompt=None`（或删该实参），保留 docstring 中「None 选择 pipeline 默认模板」的说明；修 module docstring `:10-12`（勿再暗示该 kwarg 可由用户设置）。无测试清理。

#### 1.17 `Lumina2Model.prepare_sampling(cfg_trunc_ratio guard)` — dead-branch / form 2（risk=low）

- 位置：`vrl/models/families/lumina2/model.py:214-220`
- 判死证据：
  ```
  $ grep -rn 'cfg_trunc_ratio' vrl tests
  lumina2/model.py:24 (docstring), :214 (kwargs.get), :215 (if != 1.0), :217 (raise 文案)
  ```
  仓内仅出现于 lumina2/model.py 自身。`prepare_sampling` 的 kwargs 只来自 `build_prepare_kwargs`：基类实现（`executor.py:664-676`）`del ...; return None`，唯一 override（`ReferenceConditionedChunks`，`executor.py:143`）返回 `{"reference_image": ...}`（仅 cosmos/wan）。脚本直接 caller（`full_sequence_denoise.py:185` 等）均传零 kwargs。故 `kwargs` 永不含 `cfg_trunc_ratio`，raise 永不触发。
- 动作：删该不可达 guard；module docstring bullet（`:24-27`）收缩为一句「not supported」note（保留 replay-contract 依据）。无测试清理。

### nextstep_1（families-token）

#### 1.18 `NextStep1Model.decode_image_tokens(image_size=...)` — dead-arg / form 2 + 无效用户旋钮（risk=medium）

- 位置：`vrl/models/families/nextstep_1/model.py:331-347`
- 判死证据：
  ```
  $ grep -rn 'image_size' vrl/models/families/nextstep_1/
  model.py:334 (param), :337 del image_size   （body 丢弃, 尺寸来自 side=√(tokens.shape[1]) 与 f8 VAE）
  runtime.py:184 (call), :204 ("image_size": params.image_size, context 写入零 reader), :109 (default_image_size=None → 必填)
  vrl/config/presets/sampling/token/continuous_image_256_1024tok.yaml:4 (image_size: 256)
  ```
  唯一 caller 是 `NextStep1ChunkExecutor.forward_chunk_plan`（`runtime.py:184`）；nextstep_1 是 continuous-token 家族，`ARDiscreteChunkExecutorBase` 的共享 `decode_image_tokens` 调用（`executor.py:283`）不适用。旋钮 required-but-ignored；用户把 256→512 静默仍得 256px。`context["image_size"]`（`runtime.py:204`）零 reader（replay 只读 guidance_scale/num_steps/noise_level）。
- 动作：**取校验支**。`decode_image_tokens`（`model.py:331-347`）把 `del image_size` 换成镜像 llamagen（`llamagen/model.py:435-440`）的 mismatch 检查：从 token 网格与 VAE 空间下采样因子算期望像素尺寸（f8 → `side * 8`，尽量从 VAE config 取而非硬编码），当 `image_size` 非 None 且不符时 raise。保留 preset key（`continuous_image_256_1024tok.yaml:4`）、executor call（`runtime.py:184`）、required-field 测试（`test_request_parsing.py:54`）——校验就位后旋钮成活。**另删** `runtime.py:199-205` context dict 里零 reader 的 `"image_size": params.image_size`。加一条 mismatch raise 小测试（对照 `tests/models/families/llamagen/test_model.py:132`）。
- 注意（medium）：删除支不可行——需把共享 `ARRequestLayout/ARSamplingParams`（被 llamagen/janus_pro 消费）的 `image_size` 改为可选，且删 preset key 会让每条 nextstep rollout 在 `runtime.py:129` raise `"request.sampling.image_size"`。校验支是与姊妹家族一致的正解。

#### 1.19 nextstep `metadata` dict 的 `'family'` key（+ module-level `logger`）— dead-field / form-1 data twin（risk=low）

- 位置：`vrl/models/families/nextstep_1/runner.py:226,271`；`vrl/models/steps/token/paged_attention_helpers.py:253,317`；`nextstep_1/model.py:42,44`
- 判死证据：
  ```
  $ grep -rn '"family"' nextstep_1/runner.py paged_attention_helpers.py
  runner.py:226 metadata={"family": "nextstep_1", "image_token_num": ...}
  runner.py:271 metadata={"family": "nextstep_1"}
  paged_attention_helpers.py:253 metadata={"family": self.family, "image_token_num": ...}
  paged_attention_helpers.py:317 metadata={"family": self.family}
  $ grep -rnF 'metadata["family"]' vrl tests   → 无匹配（零 reader）
  $ grep -n 'logger\|init_logger' nextstep_1/model.py
  :42 from vrl.utils.logging import init_logger ; :44 logger = init_logger(__name__)   （零 logger.* 调用）
  ```
  `ARAttentionPrefillInput/ARAttentionStepInput.metadata` 的唯一 consumer 是 `_max_new_tokens_from_metadata`（`vrl/nn/modules/ar_decoder.py:542-544`），只读 `'image_token_num'`。backend 家族身份走 `ARAttentionConfig.family`（`paged.py:24,124`）——另一条活路径。`self.family`（`paged_attention_helpers.py:176` `owner = self.lane_owner_prefix or self.family`）是活的非 metadata 消费者，**不删**。
- 动作：从四处 metadata dict 删 `'family'`（保留 `'image_token_num'`）；dict 变空处（`runner.py:271`、`paged_attention_helpers.py:317`）整体省略 `metadata=` 实参（默认空 dict）。保留 dataclass 上的 `Mapping metadata` 字段（有活 reader 的 protocol escape hatch）与 `self.family` 属性。删 `nextstep_1/model.py:44` 的 `logger = init_logger(__name__)` 及 `:42` 其唯一用途的 import。无测试清理。

#### 1.20 nextstep `'token_dim'` 转发项 — dead-config-knob / form 2（risk=low）

- 位置：`vrl/models/families/nextstep_1/runtime.py:44-50`（`nextstep_config_from_build` 转发 key 列表内的 `"token_dim"`）
- 判死证据：
  ```
  $ grep -rn 'token_dim' vrl/models/families/nextstep_1/ vrl/config/presets/
  runtime.py:49 "token_dim"   （转发项, 唯一 quoted-key hit）
  model.py:90 token_dim: int = ... ; :136-137/:431-432 self.config.token_dim = int(getattr(image_head,"input_dim",...))
  runner.py:78 token_dim = cfg.token_dim   （活 reader, 但在 clobber 之后）
  presets/ → 无 token_dim
  ```
  `"token_dim"` 全仓仅此一处 quoted-key。不在 `SamplingConfig`（`schema.py:313-350`），无 YAML/test 生产者。且 `NextStep1Model.__init__`（`model.py:136-138`）与 replay `__init__`（`:431-433`）在唯一行为 reader（`runner.py:78`）前，用 `image_head.input_dim` 覆盖 `config.token_dim`——即便有假想 YAML 生产者也会被静默覆盖。
- 动作：从转发 sampling key 列表删 `"token_dim"`。`NextStep1Config.token_dim` 字段保留（活：默认 + getattr fallback + `runner.py:78` reader）。无测试清理。

### sana（families-image）

#### 1.21 `SanaModel.encode_prompt(complex_human_instruction kwarg)` — dead-arg / form 2（risk=low）

- 位置：`vrl/models/families/sana/model.py:218`
- 判死证据：
  ```
  $ grep -rn 'complex_human_instruction' vrl tests
  sana/model.py:20 (docstring), :218 (=kwargs.get("complex_human_instruction"))
  scripts/eval/sana_inference.py:27, sana_checkpoint_compare.py:217,339 ("official_pipeline_default" 是 OFFICIAL SanaPipeline 的 provenance 元数据串, 非 SanaModel.encode_prompt 输入)
  presets/.../sana/online_grpo_pickscore_validation.yaml:4 (# 注释)
  tests/scripts/eval/test_sana_checkpoint_compare.py:52 assert "complex_human_instruction" not in kwargs
  ```
  所有能到达 `SanaModel.encode_prompt` 的 call site 只传固定 kwarg，无 `**` 转发 config dict，故 `kwargs.get` 恒 None。diffusers `pipeline_sana.py` 的 `encode_prompt` 参数默认 `None`（`if not complex_human_instruction:` 关 CHI）。
- 动作：把 `complex_human_instruction=kwargs.get("complex_human_instruction")` 改为显式 `complex_human_instruction=None` 加一行注释（显式 None 记录 VRL 有意关 CHI，防上游默认漂移）；修 module docstring `:20-23`（勿称其「exposed as a sampling kwarg」）。无测试更新（唯一相关测试断言该 key **不在** official-pipeline kwargs）。
- 注意：与 DO-NOT-FLAG 的 sana `prepare_latents=` 修复不同；本条 None-in/None-in 行为不变。

### sd3_5（families-image）

#### 1.22 `SD3_5Model.uses_vrl_attention_processor` — dead-field / test-only reader（risk=low）

- 位置：`vrl/models/families/sd3_5/model.py:119,125,420,426`
- 判死证据：
  ```
  $ grep -rn 'uses_vrl_attention_processor' vrl tests
  sd3_5/model.py:119,125,420,426  self.uses_vrl_attention_processor = install_sd3_joint_attention_processor(...)
  tests/models/families/sd3_5/test_attention_processor_install.py:32,45,63  assert model.uses_vrl_attention_processor is True
  ```
  写四处、读只三处测试断言，全仓无其他 reader、无 getattr-by-string。非 dataclass 字段。`install_sd3_joint_attention_processor` 函数本身保留（其 bool 返回被 `vrl/scripts/perf/quantized_sd3_forward_profile.py:112` 控制流消费）。
- 动作：把四处 `self.uses_vrl_attention_processor = install_sd3_joint_attention_processor(...)` 换成裸调用 `install_sd3_joint_attention_processor(...)`（side effect 才是要点）。删 `test_attention_processor_install.py:32,45,63` 三条 flag 断言——每个测试已用更强的 `isinstance(transformer.processor, SD3JointAttentionProcessor)` 断真实行为，保留之。

### wan_2_1（families-video）

#### 1.23 `_resolve_guidance_scale_2`（`request.extra['guidance_scale_2']` override 分支）— dead-branch / form 2（risk=medium）

- 位置：`vrl/models/families/wan_2_1/model.py:1370-1380`
- 判死证据：
  ```
  $ grep -rn 'guidance_scale_2' vrl tests vrl/config
  model.py:1377 raw = request.extra.get("guidance_scale_2") if request.extra else None
  model.py:494 state.guidance_scale_2 or state.guidance_scale (expert 路由, 活 replay wire)
  wan_2_2/a14b.yaml:6 boundary_ratio 0.875 / i2v_a14b.yaml:6 0.9  （dual-expert 路径活）
  vrl/config/ 无 guidance_scale_2 key ; presets 无生产者
  ```
  `request.extra['guidance_scale_2']` 零生产者：rollout 路径唯一 extra 构造 `full_sequence_denoise/executor.py:239-242` 只设 `max_sequence_length`，无处向 `request.extra` 写入该 key。非默认值只出现于测试（`test_backbone_parity.py:225` 直构 state）与 `wan_i2v_base_sample.py`（调 diffusers pipeline，非此路径）。[[SPRINT_sampling_config_knob_unification]] 记录其「wan guidance_scale_2 未动」。dual-expert 路径活，故 `guidance_scale_2` 在每条生产 rollout 里静默恒等于 `guidance_scale`。
- 动作：二选一——**优选接一个真实生产者**（在 executor 把 `sampling.guidance_scale_2` config key 映射进 `VideoGenerationRequest.extra`；Wan2.2 dual-expert preset 在用，上游 `WanPipeline` 确有该 per-expert 旋钮），或删 extra-override 分支使 dual-stage 恒显式镜像 `guidance_scale`。无论哪支，保留 `SamplingState` 字段/batch-context plumbing（它是 rollout 所用值的 replay wire，`model.py:494` 读它做 expert 路由）。删除支无测试清理。
- 注意（medium）：这是「旋钮从未接线」而非纯死代码——优选接线以恢复用户可配的 per-expert guidance。

#### 1.24 `WanT2VReplayModel.__init__(expert_lifecycle_profiling=...)` — dead-arg / form 1（risk=low）

- 位置：`vrl/models/families/wan_2_1/model.py:813-835`（replay `__init__` 参数）
- 判死证据：
  ```
  $ grep -rn 'expert_lifecycle_profiling' vrl tests vrl/config
  schema.py:411 (bool=False) ; presets/.../online_grpo_dual_expert_proof.yaml:63 (true)
  model.py:145/158/202-203 (rollout __init__ / from_build) ; :543 (attr read)
  model.py:822 (replay __init__ param — 本条), :834 (attr set), :862-863 (prepare_replay 重导)
  model.py:977-978 (rollout I2V from_build)
  tests/trainers/test_wan_fsdp_distributed.py:526 policy._expert_lifecycle_profiling = True  （直设属性, 非 ctor kwarg）
  ```
  只有两处传该 kwarg（`model.py:202,977`）都在 **rollout** `from_build` 内（走 rollout `__init__`，`pipeline=...`），非 replay ctor。replay 唯一生产构造是通用 replay loader（`vrl/models/steps/denoise/build.py:241-249`，只传 transformer/scheduler/device），随后 `prepare_replay`（`model.py:862-864`）从 `build.model_config` 重导真实值。`WanI2VReplayModel.__init__`（`:1181`）转发到 `WanT2VReplayModel.__init__` 时**不带**它。
- 动作：删 replay `__init__` 的 `expert_lifecycle_profiling` 参数，保留 `self._expert_lifecycle_profiling = False` 初始化（属性活：`:543` 读、`prepare_replay` 覆盖）。YAML 旋钮/schema 字段经 `build.model_config` → `prepare_replay`/`from_build` 仍全功能。无测试清理。

### models 核心层（models-core）

#### 1.25 `ReplaySegmentResult.logprobs`（`image_logits` / `text_logits` fallback）— dead-branch / form 2（risk=low）

- 位置：`vrl/models/interfaces/replay.py:68-71`（+ docstring `:52-53`）
- 判死证据：
  ```
  $ grep -rnF '"image_logits"' vrl tests ; grep -rnF '"text_logits"' vrl tests
  replay.py:69 self.values.get("image_logits") ; :71 self.values.get("text_logits")  （定义自身）
  tests/rollouts/replay/test_replay_result_signals.py:143 values={"image_logits": ...}  （仅用作 require_value 报错测试的任意错 key）
  ```
  所有 `ReplaySegmentResult(` 构造点 values 的 key 是 `log_probs`/`tokens`/`logits`/`image_token_ids`/noise-pred——从不是 `image_logits`/`text_logits`。janus R1 的 `replay_r1_segment` 视觉与文本两支都存到 `"logits"`（`janus_pro/model.py:495`）。`forward_image_logits`/`forward_text_logits` 是方法名，输出存 `"logits"`，与这两个 fallback 无关。
- 动作：删 `:69,71` 两个 fallback lookup，从 docstring `:52-53` 去掉这两个 key 名，保留 `log_probs → logits → require_value("logits")` 序列（删后自然收敛为 `require_value("logits")`，保留测试断言的报错信息）。`test_replay_result_signals.py:143` 用 `"image_logits"` 只作任意错 key、经 `require_value("logits")` 校验，保持不变。无测试更新。

#### 1.26 `forward_autocast` — single-caller-merge / form 3（risk=medium）

- 位置：`vrl/models/precision.py:27-45`（并入 `model_autocast` `:48-59`）
- 判死证据：
  ```
  $ grep -rn 'forward_autocast' vrl tests
  precision.py:27 (def), :55 (model_autocast 内唯一生产 caller), :106 (__all__, 零 importer)
  vrl/models/steps/denoise/base.py:79 (陈旧 docstring 提及)
  tests/models/test_precision.py:81,86 (test-only caller, enabled= 的另一唯一用户)
  $ grep -rn 'model_autocast' vrl tests
  真实外部 caller: denoise/base.py:57, causvid/model.py:559, algorithms/diffusion_nft.py:259-267, trainers/offline/dpo.py:300,307
  ```
  一个决策（「哪个 autocast context 生效」）被拆成两函数，第二个（`forward_autocast`）仅一个生产 caller（`model_autocast`）；`enabled=` kwarg 也只为 `model_autocast` 传 `precision.outer_autocast` 而存在。`__all__` 里导出却全仓零 importer，无 `from vrl.models.precision import *`。`registry.py:185` 的 `getattr(precision, ...)` 读的是 PrecisionPolicy role 属性，非本模块。
- 动作：把 `forward_autocast` body 内联进 `model_autocast`（其唯一生产 caller），删 `forward_autocast` 及其 `enabled=` kwarg。改 `tests/models/test_precision.py:81-86` 用带 `RolePrecision` 的 stub model 走合并后的 `model_autocast`（`RolePrecision` 暴露 `dtype`/`outer_autocast`，可覆盖同一 `(dtype, enabled)` 矩阵）。从 `__all__` 删 `forward_autocast`，修 `vrl/models/steps/denoise/base.py:79` 陈旧 docstring 提及。
- 注意（medium）：`__all__` 导出使其看似 public facade——但零 importer + 仅一 caller 用的 `enabled` kwarg，该拆分已不再命名任何被消费的概念，故按 form-3 合并。内联后 `dtype not in ("fp16","bf16")` ValueError 分支成不可达防御码（`RolePrecision.__post_init__` 已把 dtype 限于 fp32/bf16/fp16），可删可留（照抄无害）。`precision.py` 不在 in-flight sprint 修改集；`base.py:79` 只是一句 docstring 编辑，不与 sprint 冲突。

#### 1.27 `DiffusionModelBase.load` — dead-function / form 1（risk=low）

- 位置：`vrl/models/steps/denoise/base.py:67-69`
- 判死证据：
  ```
  $ grep -rn 'async def load' vrl tests   →  只 base.py:67（无家族 override）
  $ git status --porcelain vrl/models/steps/denoise/base.py  →  ' M vrl/models/steps/denoise/base.py'
  ```
  `grep '\.load(' vrl tests`（排除 json/yaml/torch/load_state_dict）零 `model.load()` call site；无 `await ...load`（所有命中是 `offload()`）；无 `load.remote` Ray 调用；无 `"load"` 字符串派发（`magi_1` runtime dict 的 `"load"` 是 Megatron 风格 checkpoint 路径 config key，非方法派发）。`vrl/models/interfaces/`（`replay.py`/`runtime.py`）无 `load` protocol。body 是裸 `return None`，docstring「Default no-op for adapters constructed eagerly」——协议已不存在（git log 显示仅在 families reorg `d75400dc` 里被带过来）。
- 动作：删 `async def load` 方法。无测试更新。
- 注意（**sprint 重叠**）：本条所在 `vrl/models/steps/denoise/base.py` 在 in-flight [[SPRINT_native_generation_engine_program]] 的未提交改动集内。复核确认其未提交 diff 仅动 `export_batch_context` docstring、`load` 未被触碰；且 active/planned/parked sprint 文档只提 `offload()`、无 `load()` hook 计划。仍应**排在该 sprint 之后**落地，落地前重跑 `grep 'async def load' base.py` 确认无新 override，避免与并发编辑冲突。

#### 1.28 `DiffusionBackboneOutput.metrics` — dead-field / test-only reader（risk=low）

- 位置：`vrl/models/steps/denoise/common/backbone.py:42`（定义）、`:169-172`（唯一 population `metrics={"transformer_calls": raw_calls}`）
- 判死证据：
  ```
  $ grep -rn 'transformer_calls' vrl tests
  backbone.py:169 (注释), :172 metrics={"transformer_calls": raw_calls}
  tests/models/steps/denoise/common/test_backbone_contract.py:93,116 assert output.metrics["transformer_calls"] == 1/2
  ```
  `as_dict()`（`backbone.py:44-49`）刻意省略 `metrics`，13 个家族 `forward_step` 都 `return output.as_dict()`，值在任何生产 consumer 前被丢弃。「no production reader」注释在 population 处而非字段定义处。两个测试在 `:87/:115` 已用 `len(transformer.calls) == 1/2` 对 test-owned fake 断同一契约，`:93/:116` 的 metrics 断言严格冗余。
- 动作：删 `metrics` 字段与 `metrics={"transformer_calls": raw_calls}` 构造；`test_backbone_contract.py:93,116` 的两条冗余断言直接删（fake-count 断言已覆盖 CFG-batching 契约）。

#### 1.29 `ARDiscreteTokenRunner.step_token(generator kwarg)` — dead-arg / form 2（risk=low）

- 位置：`vrl/models/steps/token/base.py:124-133`
- 判死证据：
  ```
  $ grep -rn 'step_kwargs' vrl
  token_loop.py:260 (field), :287 (dict()), :298 (**call)
  nextstep_1/runtime.py:180 step_kwargs=sample_kwargs   （唯一生产者）
  $ grep -rn 'def step_token\|del generator' vrl
  base.py:125 (定义), :132 del generator ; nextstep_1/runner.py:138 (独立 override, 消费 generator)
  ```
  token loop 经 `call_with_supported_kwargs(self.runner.step_token, state, step_batch, **call_step_kwargs)`（`token_loop.py:294-298`）调用，唯一 `step_kwargs` 生产者是 nextstep_1——而 nextstep_1 完全 override `step_token`（独立类，不继承 `ARDiscreteTokenRunner`）。所有用基类实现的家族（glm_image/llamagen 直接，janus_pro/emu3 经 `PagedCFGTokenRunner`，均不 override `step_token`）恒收默认 None；base 的 `del generator` 使即便转发也被丢。`call_with_supported_kwargs` 会过滤签名外 kwarg，删参数与 `del` 行为等价。
- 动作：删 base `step_token` 的 `generator: torch.Generator | None = None` kwarg 与 `del generator` 行。nextstep_1 自有 generator-消费签名，不受影响。无测试清理。

#### 1.30 `LatentOutputLayout`（`'video_bcthw'` 值）— dead-branch / form-2 data twin（risk=low）

- 位置：`vrl/models/steps/denoise/common/latent_decode.py:11`
- 判死证据：
  ```
  $ grep -rn 'video_bcthw' vrl tests
  latent_decode.py:11 LatentOutputLayout = Literal["image_bchw", "video_btchw", "video_bcthw"]   （仅 Literal 定义自身, 全仓唯一命中）
  ```
  14 个家族 `output_layout=` 生产者全部硬编码 `'image_bchw'` 或 `'video_btchw'`（纯 Python，无 YAML）；两个测试文件只用这两值。`ChunkedLatentDecoder.__call__`（`:45-47`）只 special-case `'video_btchw'`，`'video_bcthw'` 与 `'image_bchw'` passthrough 无差别。`test_wan_decode_latents_preserves_bcthw_layout` 是烟雾弹——它测 Wan 的 `'video_btchw'` plan，从不用 `'video_bcthw'` 串。
- 动作：从 `LatentOutputLayout` Literal 删 `'video_bcthw'`。无 decoder 逻辑改动、无测试改动（纯类型标注编辑）。

#### 1.31 `TrainableStateSlots.versions` — dead-function / test-only caller（risk=low）

- 位置：`vrl/models/utils.py:147-148`
- 判死证据：
  ```
  $ grep -rn '\.versions()' vrl tests
  tests/models/test_utils.py:23 == [1,2] ; :32 == [2,3] ; :51 == []   （无 vrl/ 命中）
  ```
  生产 consumer（`vrl/models/steps/denoise/base.py:330-366` 的 install/has/get/_evict，由 `vrl/generation/execution/worker.py` 驱动）从不调 `versions()`。body 是 `return sorted(self._slots)`——纯测试可观测。in-flight sprint 只规划 slot 激活/驱逐（has/install/get），非 version 枚举。
- 动作：删 `versions()` 方法；把 `tests/models/test_utils.py:23,32,51` 三条断言按 `has()`（对期望/驱逐版本号）改写（安装顺序、驱逐窗口、空安装 no-op 这些真实断言用 `has()` 全可表达；`:32` 已与相邻 `has()` 断言冗余，`:51` 改写为 `not slots.has(1)`）。

#### 1.32 `count_trainable_params` — dead-function / test-only caller（risk=low）

- 位置：`vrl/models/utils.py:158-161`
- 判死证据：
  ```
  $ grep -rn 'count_trainable_params' vrl tests
  vrl/models/utils.py:158 (def)
  tests/models/families/emu3/test_replay.py:21 (import), :166 (assert ... > 0)
  tests/models/families/glm_image/test_replay.py:16 (import), :188 (assert ... > 0)
  ```
  零生产 caller，无字符串/registry/YAML 引用，无 `__all__`/`__init__` 再导出。in-flight `base.py` 从 `vrl.models.utils` 只 import 另四个符号、不含此。body 是一行 sum comprehension。
- 动作：删 `count_trainable_params`（`:158-161`）。两个测试 call site 把 `assert count_trainable_params(model) > 0` 内联为 `assert sum(p.numel() for p in model.parameters() if p.requires_grad) > 0`，**并删两处现已无用的 import 行**（`emu3/test_replay.py:21`、`glm_image/test_replay.py:16`）。跑 ruff check 确认无 F401 残留。
- 注意（关联）：[[SPRINT_dead_code_wrapper_sweep]] 曾在删九个 per-family `trainable_param_count`/`has_lora_adapter` 时把这两测试有意接到此共享 util（done doc 不使其存活，但若想保留合并意图，可把函数移入 `tests/` 共享 helper 并重指两处 import，等价有效）。

## 2. 验证协议

- **每条删除后**：`ruff check <touched files>` + `ruff format --check <touched files>`（仅对本任务触及的 Python 文件；先 `ruff check --fix`，再 `ruff format`，最后 `ruff check` + `ruff format --check` 复核）。禁止全仓 `ruff format .` / `ruff check --fix .`。
- **全簇完成后**：`pytest tests/models/ tests/rollouts/ tests/families/ tests/generation/ tests/config/` + `pytest -m "not e2e and not slow_test"` 子集不新增失败。
- **基线（清理前，2026-07-23）**：fast subset **2620 passed / 7 pre-existing failures**（架构边界 + causvid/magi_1 打包摘要，与本清理无关）；`vrl.config.lint` 与 `ruff check .` 全绿。删除后这三项须保持。
- **逐条动作触及的测试文件**（从 action 提取）：
  - §1.1：`tests/models/steps/denoise/common/test_vae_decode_memory.py`、`tests/rollouts/runtime/test_family_registry.py`
  - §1.2：`tests/rollouts/runtime/test_janus_pro_r1_wiring.py`（源码侧动 `vrl/rollouts/collector/batch_builder.py`、`core.py`；`semantics.py`/`registry.py` 不动）
  - §1.3：无（causvid 测试只 import 私有 resolver）
  - §1.4：无
  - §1.5：`tests/models/families/causvid/test_runner.py`、`test_replay_and_loading.py`
  - §1.6：无（anima 测试 stub adapter）
  - §1.7：无
  - §1.8：无
  - §1.9：`tests/models/families/{emu3,janus_pro,glm_image,nextstep_1,llamagen}/`、`tests/models/steps/token/`（验证用，无 cleanup；源码侧动 `vrl/models/steps/token/build.py`）
  - §1.10：`tests/models/families/glm_image/test_model_build.py`
  - §1.11：可选新增 raise 断言测试；现有测试/preset 不改
  - §1.12：`tests/models/families/janus_pro/test_r1_model.py`
  - §1.13：`tests/models/families/janus_pro/test_r1_model.py`、`tests/models/steps/token/fixtures.py`（+ 验证 `tests/rollouts/runtime/test_janus_pro_r1_wiring.py`、`tests/config/test_schema.py`）
  - §1.14：无
  - §1.15：无
  - §1.16：无
  - §1.17：无
  - §1.18：可选新增 raise 断言测试；`tests/models/families/nextstep_1/test_request_parsing.py` 保持
  - §1.19：无
  - §1.20：无
  - §1.21：无（`tests/scripts/eval/test_sana_checkpoint_compare.py` 保持）
  - §1.22：`tests/models/families/sd3_5/test_attention_processor_install.py`
  - §1.23：无（接线或删分支均无 cleanup）
  - §1.24：无
  - §1.25：无（`tests/rollouts/replay/test_replay_result_signals.py` 保持）
  - §1.26：`tests/models/test_precision.py`
  - §1.27：无
  - §1.28：`tests/models/steps/denoise/common/test_backbone_contract.py`
  - §1.29：无
  - §1.30：无
  - §1.31：`tests/models/test_utils.py`
  - §1.32：`tests/models/families/emu3/test_replay.py`、`tests/models/families/glm_image/test_replay.py`

## 3. Non-Goals

- 不删被「能 raise 的校验」、控制流分支、runtime/config/Ray 调用消费的字段（AGENTS 死字段规则）。§1.19 的 `self.family`、§1.24 的 `self._expert_lifecycle_profiling` 属性、§1.20 的 `NextStep1Config.token_dim`、§1.3 保留的 `_resolve_source_root`/`_resolve_checkpoint` 校验副作用均属此列，保留。
- 不动 DO-NOT-FLAG 豁免项：`_GENERATION_CUDA_RUNTIME_RESIDUAL_BYTES`（`generation/execution/worker.py`）、`_REWARD_CUDA_RUNTIME_RESIDUAL_BYTES`（`rewards/base.py`）、`ensure_loaded`（`rewards/runtime.py`）、`process_gpu_used_bytes` NVML（`utils/cuda_memory.py`）、sana/hunyuan 的 `prepare_latents=` 修复。
- 不为省行数扁平化 protocol/lazy-import/跨家族一致性 thin function：§1.5 的 `predict_x0_cached` protocol 边界、§1.19 的 dataclass `Mapping metadata` escape hatch、§1.29 的 `step_token` protocol、§1.10/§1.15 中 emu3/janus_pro/nextstep_1 走基类的 `_runner_cls`/`_runner_attention_family`（它们是活的，只删 glm_image/llamagen 那两个 override 家族的）均保留。
- **cluster-specific non-goals**：
  - `LORA_DEFAULTS` 去重（§1.9）**只删 `runtime.py` 的五个 fallback dict**，绝不反向删 Config dataclass `lora_*` 默认值（被 `if config.use_lora` 分支、直接构造、`use_lora=True` PEFT-wrap 测试消费），也不改任何 YAML preset。
  - §1.11/§1.18 的 `decode_image_tokens(image_size=...)` **取校验支不取删除支**（删除支要改共享 `ARSamplingParams`/`ARRequestLayout`，blast radius 过大）。
  - §1.23 的 `guidance_scale_2` **优选接线而非纯删**（用户可配的 per-expert guidance），保留 `SamplingState`/batch-context replay wire。
  - §1.12/§1.13 强耦合，须同批落地，且只删 `final_image` 臂的 `stages` 半、保留 `mode == "never"`。
  - §1.2/§1.1 只删 `PolicySemantics`/`GenerationRuntimeCapabilities` 上的死字段，不删整个结构（同结构其余字段活）；§1.2 走「使字段成 source of truth」而非删除，遵从 [[SPRINT_generation_regime_decision_layering]]。

## References

- `vrl/families/registry.py:41,463`、`semantics.py:38,49,54,56,62`；`vrl/rollouts/collector/batch_builder.py:26-35,308-316`、`core.py:227,351-363`
- `vrl/models/families/causvid/model.py:54-63,182-185,548,852-869,1152`、`runner.py:143,214-225,269,292,304,331`
- `vrl/models/families/cosmos/anima/adapter.py:53-93`、`model.py:550`
- `vrl/models/families/echo/model.py:145-146`；`vrl/models/steps/denoise/common/lora.py:32-40,57`
- `vrl/models/families/emu3/model.py:264-266,377,646-653,696-698`、`runtime.py:22-33`
- `vrl/models/families/glm_image/runtime.py:21-32,80-81,97-115`、`runner.py:73`；`tests/models/families/glm_image/test_model_build.py:86-87`
- `vrl/models/families/janus_pro/model.py:102,507,523-528,533-535,573,681-707,783-786,975-1000`、`runtime.py:38-50,283,287,420-439`；`vrl/config/schema.py:313-350,917-935`
- `vrl/models/families/llamagen/model.py:175,183-193,435-440`、`runtime.py:23-35,69-70,81-97`、`runner.py:54`；`vrl/nn/modules/ar_attention_backends.py:88,136,176-179`
- `vrl/models/families/lumina2/model.py:11,166,185,214-220`
- `vrl/models/families/nextstep_1/model.py:42,44,90,136-137,331-347,431-432`、`runtime.py:44-50,109,180,184,199-205`、`runner.py:78,176`；`vrl/models/steps/token/paged_attention_helpers.py:253,317`
- `vrl/models/families/sana/model.py:20,218`
- `vrl/models/families/sd3_5/model.py:119,125,420,426`；`vrl/scripts/perf/quantized_sd3_forward_profile.py:112`；`tests/models/families/sd3_5/test_attention_processor_install.py:32,45,63`
- `vrl/models/families/wan_2_1/model.py:494,543,813-835,862-863,1370-1380`；`vrl/config/schema.py:411`、`presets/experiment/wan_2_2/online_grpo_dual_expert_proof.yaml:63`
- `vrl/models/interfaces/replay.py:52-53,68-71`
- `vrl/models/precision.py:27-45,55,106`；`vrl/models/steps/denoise/base.py:57,67-69,79`
- `vrl/models/steps/denoise/common/backbone.py:42,44-49,169-172`；`tests/models/steps/denoise/common/test_backbone_contract.py:87,93,115,116`
- `vrl/models/steps/token/base.py:124-133`；`vrl/generation/composition/token_autoregressive/token_loop.py:260,287,294-298`
- `vrl/models/steps/denoise/common/latent_decode.py:11,45-47`
- `vrl/models/utils.py:147-148,158-161`；`tests/models/test_utils.py:23,32,51`、`tests/models/families/{emu3,glm_image}/test_replay.py`
- 关联 sprint：[[SPRINT_deadcode_00_overview]]、[[SPRINT_trajectory_views_types_dead_fields_cleanup]]、[[SPRINT_generation_regime_decision_layering]]、[[SPRINT_sampling_config_knob_unification]]、[[SPRINT_dead_code_wrapper_sweep]]、[[SPRINT_native_generation_engine_program]]（in-flight）、[[SPRINT_attention_kernel_medium]]（parked）
