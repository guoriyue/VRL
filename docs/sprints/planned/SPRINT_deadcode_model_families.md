# SPRINT: model families 死代码与 `LORA_DEFAULTS` 去重（planned → reconciled）

状态：**RECONCILED（2026-07-24）**，对齐 main @ `7c748532`（= origin/main tip）。原 audit 跑在旧树 `88ed756e`，其后 origin 已落地约 63 个 cleanup/refactor commit，故原 32 条发现里很多已被独立完成。本版对当前 checked-out 树逐条复核后：**22 条仍需做**（16 STILL_VALID + 6 RELOCATED，见 §1）、**7 条已由 origin 落地**（见 §2，无需再做）、**3 条现场已变需重新评估**（见 §3）。
来源：dead-code-audit workflow（五种死代码形态 + 对抗验证 + 删除类二次字符串引用检查）；本版叠加 2026-07-24 对 main @ `7c748532` 的逐条复核（grep 排除 `.venv/ third_party/ outputs/ datasets/ docs/runs/ __pycache__/ egg-info`）。
关联：[[SPRINT_deadcode_00_overview]]、[[SPRINT_trajectory_views_types_dead_fields_cleanup]]（死字段规则同源）、[[SPRINT_generation_regime_decision_layering]]（`PolicySemantics` 为分发 source of truth，约束 §1.2 的落地方向）、[[SPRINT_sampling_config_knob_unification]]（`final_image_policy`/`guidance_scale_2` 的收敛记录，约束 §1.13/§1.23）、[[SPRINT_dead_code_wrapper_sweep]]（`count_trainable_params` 的历史合并意图，约束 §1.32）、[[SPRINT_native_generation_engine_program]]（**in-flight**，§1.27 与其未提交改动同文件，须排在其后）。

> **执行状态（2026-07-24）**：§1 全部 22 条已落地 `6e0cfd8e`（含 `trajectory_layout` 接为事实源）。§3 的 3 条 CHANGED 未做（origin 已处理）。

## 0. 一句话

本簇原是 model families 层的死代码清理（死字段/死参数/死配置旋钮 form 1/2 + form-4 重复实现）。经对齐 main @ `7c748532` 复核：最锋利的一条——五个 token 家族 `_*_LORA_DEFAULTS` 双维护（form-4-data）——已由 origin 独立落地（现从家族 Config 派生 fallback，见 §2）；另有 6 条死字段/死描述符/死转发项也已删除（§2）。3 条 `decode_image_tokens(image_size=)` 校验支 + `runs_in_isolated_subprocess` 的现场已变（§3）。**剩余 22 条待删项集中在死参数/死分支/死配置旋钮与 form-4 重复**，误删风险仍集中在：只删同结构里的死字段而非删整个结构（`PolicySemantics`/`GenerationRuntimeCapabilities`/`ReplaySegmentResult`——同结构其余字段活），以及 `guidance_scale_2`/`trajectory_layout` 优选接线/成 source of truth 而非纯删。

## 1. 待删清单（仍有效）

> 保留 STILL_VALID + RELOCATED 两类，逐条带原始证据/动作；RELOCATED 条目的「位置」已更新到 main @ `7c748532` 的当前 file:line 并标注行号迁移。每条附 `✓ 复核（2026-07-24）` 一行给出当前落点。medium（剩 6 条）：§1.2、§1.12、§1.13、§1.14、§1.23、§1.26。

### families/ 共享层

#### 1.2 `PolicySemantics.trajectory_layout` — dead-field（risk=medium）

- 位置：`vrl/families/semantics.py:38`（定义）、`:49-62`（自身 `__post_init__`）；构造点 `vrl/families/registry.py:476,496,509,539,916`
- 判死证据：
  ```
  $ grep -rn 'trajectory_layout' vrl/ tests/
  vrl/families/registry.py（构造/builder 参数默认）
  vrl/families/semantics.py:38                    （定义）
  vrl/families/semantics.py:49,54,56,62           （仅自身 __post_init__ 自校验）
  tests/scripts/test_online_lifecycle.py          （测试构造）
  tests/rollouts/runtime/test_family_registry.py:43（测试构造）
  ```
  唯一 reader 是它自己的 `__post_init__` 内部一致性校验（对硬编码 registry literal 的自校验，非 dead-field 规则下的 consumer）。真正的 multisegment 路由在 `vrl/rollouts/collector/batch_builder.py:229` 上，是按 `trajectory.family == "janus_pro_r1" or trajectory.task == "ar_t2i_r1"` 硬编码的字符串——不读此字段。`grep 'ar_t2i_r1'` 确认该 task 只由 `janus_pro_r1` 家族产出（`registry.py:716`、`janus_pro/runtime.py`），即 layout 检查可同时替代该条件的两半（form-4 data twin）。同结构里 `generation_regime`/`step_kind`/`action_distribution` 均有真实消费者（`factory.py:220,289`、`online.py:908,982,1061`、`registry.py:122`），故结构活、字段死。
- 动作：**取「使字段成为行为 source of truth」这一支（放弃删除支）**，因为删除支违背 [[SPRINT_generation_regime_decision_layering]]（`semantics.py`「PolicySemantics 是 source of truth；生产分发读取 semantics」）。具体：(1) 给 `RolloutBatchBuildContext`（`vrl/rollouts/collector/batch_builder.py`）新增 `trajectory_layout` 字段，默认 `None`；(2) 在 `vrl/rollouts/collector/core.py` 从 collector 构造时已持有的 resolved `ModelFamilyEntry`（`build_rollout_collector` 接收 `entry`）取 `entry.policy_semantics.trajectory_layout` 穿线进 `RolloutCollector`（与已存储的 `request_builder`/`entry` 并列）；(3) 把 `_is_multisegment_categorical`（`batch_builder.py:229` 附近）里的 family/task 硬编码替换为 `self.context.trajectory_layout == "multisegment_token"`，保留既有 `len(trainable) > 1 and all categorical` 通用 fallback；(4) 更新直接构造 `RolloutBatchBuildContext` 的 multisegment 测试点 `tests/rollouts/runtime/test_janus_pro_r1_wiring.py`（传入 layout，或确认走通用 fallback）；`tests/rollouts/collector/test_runtime.py`、`tests/generation/bindings/*` 用的是非 multisegment trajectory，无需改。(5) `semantics.py` 与 `registry.py` 保持不动——字段、`Literal`、`__post_init__` 交叉规则在第 (3) 步落地后即成活。
- 注意（medium）：`TrajectoryRolloutBatchBuilder` **作用域内没有 registry entry**（只收 `GenerationOutput` + `RolloutBatchBuildContext`），故 layout 必须从 collector 侧已 resolve 的 entry 穿线下来，不能就地 `entry.policy_semantics...`。这是本条唯一实现难点。
- ✓ 复核（2026-07-24）：STILL_VALID。字段仍在 `semantics.py:38`（def）/`49-62`（`__post_init__`），零行为 reader；batch_builder 家族/task 硬编码现漂到 `batch_builder.py:229`（原审计记的 308-316），registry 构造点现 476/496/509/539/916（原 341/357/370/396/720）。判死结论与动作思路不变，落地按当前行号定位。

### causvid（families-video）

#### 1.4 `CausVidCausalRunner.__init__(math_dtype=...)` — dead-arg（risk=low）

- 位置：`vrl/models/families/causvid/runner.py:220`（构造参数）、`:225`（`self.math_dtype`）、`:280`、`:331`（两处使用）
- 判死证据：
  ```
  $ grep -rn 'CausVidCausalRunner(' vrl tests
  model.py:660; tests/.../test_runner.py:124; tests/.../test_replay_and_loading.py:198   （三处构造，均只传 backend+geometry=+schedule=，无 math_dtype）
  $ grep -rn 'math_dtype' vrl/models/families/causvid/
  runner.py:220,225,280,331; model.py:552: math_dtype=torch.float32   （replay scorer 硬编码 fp32）
  ```
  无 YAML/config 生产者（`precision.diffusion_math` 只喂 `DiffusionSDELogProbEvaluator`，causvid 分支在 `factory.py` 提前返回 `ChunkAutoregressiveDenoiseLogProbEvaluator()`，无 `math_dtype` 属性）。
- 动作：删构造参数与 `self.math_dtype`，在两处使用点内联 `torch.float32`（`:280` 亦可直接省略该 kwarg——`vrl/math/denoise/renoise.py:57` 对 `None` 默认 fp32）。无测试清理。
- 注意（parity）：replay 侧 `model.py:552` 硬编码 `math_dtype=torch.float32`，任何非 fp32 的 rollout 取值都会静默破坏 rollout/replay log-prob parity——此旋钮只会致害，fp32 是正确性常量而非偏好。
- ✓ 复核（2026-07-24）：STILL_VALID。参数仍在 `runner.py:220`（def）/`225`/`280`/`331`；三处构造点均不传 `math_dtype`，replay 侧 `model.py:552` 仍硬编码 fp32。死参数 + parity hazard 不变。

#### 1.5 `CausVidCausalBackend.predict_x0_cached(finalize_cache=...)` — dead-arg（risk=low）

- 位置：`vrl/models/families/causvid/runner.py:143`（protocol）、`:269,292,304`（三处 call site）；实现 `model.py:187,190`
- 判死证据：
  ```
  $ grep -rn 'finalize_cache' vrl tests
  runner.py:143 (protocol), :269 False, :292 False, :304 True
  model.py:187 (param), :190 del finalize_cache   （唯一生产实现直接丢弃）
  tests/.../test_replay_and_loading.py (fake 分支); test_runner.py (fake 记录并断言)
  ```
  唯一生产实现 `model.py:190` 以 `del finalize_cache` 开头，KV cache 通过 `current_start/current_end` 每次 forward 无条件提交，flag 无行为效果。`finalize_cache=True` 唯一 call site（`runner.py:298-306`）恰是 `timestep == schedule.cache_timestep` 的那次；`CausVidSchedule.__post_init__` 硬校验 `cache_timestep == 0`（`runner.py:88-89`），仓内唯一构造的 schedule 是 `OFFICIAL_CAUSVID_SCHEDULE`（`prediction_timesteps=(1000,757,522)`），故 `timestep==0` 唯一标识 finalize 调用——flag 与已在线上的 timestep 信息冗余。
- 动作：从 protocol、`_OfficialCausVidBackend.predict_x0_cached`（其 body 以 `del finalize_cache` 起）、三处 runner call site 删除 `finalize_cache`；把 `tests/models/families/causvid/test_runner.py`、`test_replay_and_loading.py` 的 fake backend 改为按 `timestep == schedule.cache_timestep`（0）判定。保留 protocol 边界本身。
- ✓ 复核（2026-07-24）：STILL_VALID。全部 call site 仍在 `runner.py:143/269/292/304`；唯一生产实现仍以 `del finalize_cache` 开头（`model.py:190`，原 :185，整体 +5 行）。剩余 consumer 为两个 test fake。判死不变。

### cosmos/anima（families-video）

#### 1.6 `TransformerBlock.__init__(use_self_attn=...)` — dead-arg / form 2（risk=low）

- 位置：`vrl/models/families/cosmos/anima/adapter.py:60`（定义）、`:63,64,84`（`self` 赋值 + 两处 guard）
- 判死证据：
  ```
  $ grep -rn 'use_self_attn' vrl tests
  adapter.py:24 (唯一 True 生产者), :60 (定义), :63/:64/:84 (self 赋值 + 两处 guard)
  ```
  唯一构造点 `AnimaLLMAdapter.__init__`（`adapter.py:20-27`）对全部 6 个 block 传 `use_self_attn=True`；`False` 分支会构造 cross-attn-only block，其 checkpoint key（无 `norm_self_attn.`/`self_attn.`）匹配不到任何已发布 Anima checkpoint。anima 测试用 `torch.nn.Identity()` stub adapter，不触 `TransformerBlock`。
- 动作：删 `use_self_attn` 参数、`self.use_self_attn` 属性、`__init__` 与 `forward` 里两处 `if self.use_self_attn:` guard，无条件构造 `norm_self_attn`/`self_attn`。module 名与 state-dict key 不变，`_load_anima_llm_adapter`（`model.py:550`）的 `strict=True` 加载不受影响。无测试清理。
- ✓ 复核（2026-07-24）：STILL_VALID。`use_self_attn` 仍在 `adapter.py:24`（唯一 True 生产者）/`60`/`63`/`64`/`84`；False 分支零生产者。不变。

### echo（families-video）

#### 1.7 `EchoModel._lora_transformer` — duplicate-impl / form 4（risk=low）

- 位置：`vrl/models/families/echo/model.py:146`
- 判死证据：
  ```
  $ grep -rn '_lora_transformer' vrl tests
  vrl/models/steps/denoise/common/lora.py:34 (LoraModelMixin 默认, body: return self.transformer)
  vrl/models/steps/denoise/common/lora.py:59 (唯一 call site)
  vrl/models/families/echo/model.py:146 (override, body 字节等同: return self.transformer)
  ```
  override body 与 mixin 默认逐字节相同。Echo 满足默认契约：`model.py` `self.transformer = echo.model`，`_set_transformer` 同步。约 15 个 `LoraModelMixin` 家族中只有 Echo override 此方法；另一个 pipeline-less 家族 `AnimaModel` 已直接用默认。MRO 安全：`EchoModel(LoraModelMixin, DiffusionModelBase)`，`DiffusionModelBase`（含其 §1.27 的未提交改动）既不定义 `_lora_transformer` 也不定义 `transformer` property。
- 动作：删两行 override。无测试清理。
- 注意（sprint 关联）：MRO fall-through 依赖 `vrl/models/steps/denoise/base.py`（in-flight sprint 修改文件）不引入 `_lora_transformer`/`transformer` property；建议落地时再 grep 一次 `base.py` 确认。
- ✓ 复核（2026-07-24）：STILL_VALID。override 仍在 `echo/model.py:146`（原 145），body 与 `LoraModelMixin` 默认（`lora.py:34`，原 :32；call site `lora.py:59`）字节相同。无 test/registry 引用。不变。

### emu3（families-token）

#### 1.8 `Emu3ReplayModel.text_config` — duplicate-impl / form 4（risk=low）

- 位置：`vrl/models/families/emu3/model.py:663-664`（**已迁移**，原 696-698）
- 判死证据：
  ```
  $ grep -rn 'text_config' vrl/models/families/emu3/model.py
  :239-240  Emu3Model.text_config → return self.emu3.config.text_config
  :338      eos = getattr(self.text_config, "eos_token_id", None)   （唯一行为消费者）
  :663-664  Emu3ReplayModel.text_config → 字节相同
  ```
  MRO `Emu3ReplayModel(ARReplayRolloutStubs, Emu3Model)`，`ARReplayRolloutStubs`（`vrl/models/steps/token/base.py`）只定义 `decode_image_tokens`，不 shadow `text_config`。replay 的 `self.emu3` 是 `Emu3ReplayCore`，其 `__init__` 有 `self.config = config`，`.config.text_config` 解析一致。姊妹 replay 模型 `GlmImageReplayModel` 继承基类 `text_config` 无 override，`JanusProReplayModel` 根本无此 property。无 `"text_config"` 字符串派发。
- 动作：删该重复 override 三行。无测试清理。
- ✓ 复核（2026-07-24）：RELOCATED。override 从 696-698 迁到 `emu3/model.py:663-664`（`return self.emu3.config.text_config`），与继承的 `Emu3Model.text_config`（现 239-240，原 264-266）字节相同；唯一行为 consumer `model.py:338`。form-4 重复不变，仅行号迁移。

### janus_pro（families-token）

#### 1.12 janus sampling `'task_stages'` + `_parse_task_stages` — dead-config-knob / form 2（risk=medium）

- 位置：`vrl/models/families/janus_pro/runtime.py:283`（reader）、`:400`（`_parse_task_stages`）；`model.py:473,489,750-751`
- 判死证据：
  ```
  $ grep -rn 'task_stages' vrl tests
  model.py:473 (param 默认 JANUS_R1_SEGMENTS), :489 (stages=tuple), :750-751 (context 写入, 零 reader)
  runtime.py:283 (sampling.get 读), :400 (_parse_task_stages def)
  test_r1_model.py:270 (fake param, 传默认 JANUS_R1_SEGMENTS)
  ```
  `'task_stages'` sampling key **零生产者**：不在 `SamplingConfig`（`schema.py`，声明的 key registry；姊妹旋钮 `max_reflect_len` 在册），不在任何 YAML preset，无 collector/test request 设置。生产值恒为 `None → JANUS_R1_SEGMENTS`，故 `_parse_task_stages` 的 str-split/tuple 分支零生产者；`context["task_stages"]` 零 reader。
- 动作：`runtime.py` 删 `_parse_task_stages`（`:400`）与 `:283` 的 `task_stages=_parse_task_stages(sampling.get("task_stages"))` kwarg。`model.py` 删 `task_stages` 参数（`:473`）、`stages = tuple(task_stages)`（`:489`）与「unknown-stage / requires-initial_image」校验块（pinned 后 vacuous）、`"task_stages": stages` context 项（`:750-751`）、`stages` else 分支里失去唯一 selector 的 `selfcheck_text`。`final_image` else 臂**不删**：由 `if "final_image" in stages and mode != "never":` 简化为 `if mode != "never":`，`if "final_image" not in stages or mode == "never":` 简化为 `if mode == "never":`（`mode == "never"` 经 `sampling.refine_mode`/`config.r1_refine_mode` 仍可产出，见 §1.13）。保留 `JANUS_R1_SEGMENTS`（活的 segment taxonomy）。`tests/models/families/janus_pro/test_r1_model.py`：从 `_ExecutorModel.generate_with_refine` fake 签名删 `task_stages` kwarg 及其 `del`。
- 注意（medium）：`final_image` 臂是 `stages` 与 `mode` 的复合条件，只可删 `stages` 半、保留 `mode == "never"`（独立可产出的 selector）。本条须与 §1.13 一并落地以保持 `mode` 分支语义一致。
- ✓ 复核（2026-07-24）：STILL_VALID。仍在 `runtime.py:283`（reader）/`400`（`_parse_task_stages`，原 420-425）；`model.py` param `:473`（原 507），context write `:750-751`（原 786）。`vrl/config/` 无 `task_stages`，唯一 test producer 传默认 `JANUS_R1_SEGMENTS`——零真实生产者 = dead semantics 不变。

#### 1.13 janus sampling `'refine_mode'` fallback + `JanusProConfig.r1_refine_mode`（+ `mode=='never'` 分支）— dead-config-knob / form 2（risk=medium）

- 位置：`vrl/models/families/janus_pro/config.py:60`（`r1_refine_mode`，**已迁入新文件**，原 model.py:102）；`runtime.py:408,416-417`（`_resolve_refine_mode`，原 428-439）；`model.py:499,501`（mode 读，原 533-535）
- 判死证据：
  ```
  $ grep -rn 'refine_mode\|r1_refine_mode' vrl tests
  config.py:60  r1_refine_mode: str = "selfcheck"  # "selfcheck"|"always"|"never"
  model.py:499  mode = (refine_mode or self.config.r1_refine_mode).lower()
  runtime.py:408/416-417  sampling.get("refine_mode", ...); getattr(..., "r1_refine_mode", "selfcheck")
  tests only: test_r1_model.py ; fixtures.py (docstring)
  ```
  对 `janus_pro_r1`（唯一到达此码的家族，`schema.py` 强制 `algorithm.kind=token_grpo_multisegment`），`rollout.final_image_policy` 为 REQUIRED 且须 `'always_generate'|'use_selfcheck'`（`schema.py:278,994-997`），collector 把它 flat-merge 进 `request.sampling`，故 `_resolve_refine_mode` 恒走前两分支，fallback 在任何校验过的 run 里不可达。`sampling['refine_mode']` 无生产者；`r1_refine_mode` 从不被 `janus_config_from_build` 转发，只有测试 setter = TEST-ONLY = 死。于是 `mode` 只能是 `'always'|'selfcheck'`，`model.py` 的 `'never'` 分支零生产者。
- 动作：`_resolve_refine_mode`（`runtime.py:408,416-417`）删 `sampling.get('refine_mode', getattr(..., 'r1_refine_mode', ...))` fallback，改为当 `sampling['final_image_policy']` 不属 `{'always_generate','use_selfcheck'}` 时 raise（镜像 schema 合法性检查），删无用的 `model` 参数与 call site 实参。`config.py`/`model.py`：删 `JanusProConfig.r1_refine_mode`（`config.py:60`）；**改写 `model.py:499` 去掉 config fallback**——把 `refine_mode` 设为必填 str kwarg，或默认 `'selfcheck'`（`mode = (refine_mode or 'selfcheck').lower()`）以保直接调用 API 可跑；把 `:501` 校验集收缩为 `{'selfcheck','always'}`；`final_image` 臂改 `if "final_image" in stages:` / `if "final_image" not in stages:`（保留 stages 条件——见 §1.12）。测试：`test_r1_model.py` 删 `build_stub_janus_model(..., r1_refine_mode="selfcheck")` 与 `_ExecutorModel` 的 `config = SimpleNamespace(r1_refine_mode="selfcheck")`；`tests/models/steps/token/fixtures.py` docstring 去掉 `r1_refine_mode` 提及。跑 `tests/models/families/janus_pro/`、`tests/rollouts/runtime/test_janus_pro_r1_wiring.py`、`tests/config/test_schema.py`。
- 注意（medium）：`model.py:499` 读的正是要删的字段，动作必须显式给出该 fallback 的改写（否则直接删字段会给用公开签名默认 `refine_mode=None` 的直接调用者留下 `AttributeError`）。本条与 §1.12 强耦合，须同批落地。关联 [[SPRINT_sampling_config_knob_unification]]（`final_image_policy` 已收敛为 rollout 单一 source、仅两枚举值）。
- ✓ 复核（2026-07-24）：RELOCATED。`r1_refine_mode` 迁到新文件 `janus_pro/config.py:60`（原 model.py:102）；`_resolve_refine_mode` fallback 在 `runtime.py:408/416-417`，mode 读点在 `model.py:499/501`。`final_image_policy` 仍 schema-required + `Literal`-restricted（`schema.py:278,994-997`）强制前分支，无 `sampling['refine_mode']` 生产者，`r1_refine_mode` 仅测试 setter。dead-semantics 不变，动作按当前行号定位。

### llamagen（families-token）

#### 1.14 `LlamaGenModel._lm_trunk` — dead-function / form 1（risk=medium）

- 位置：`vrl/models/families/llamagen/model.py:153`（**已迁移**，原 183-193；docstring ref `:145`）
- 判死证据：
  ```
  $ grep -rn '_lm_trunk' vrl tests
  vrl/nn/modules/ar_attention_backends.py (唯一通用 consumer, 经 resolve_attention_backend, 仅由基类 _ar_runner 调)
  llamagen/model.py:145 (docstring 提 Janus 的 _lm_trunk), :153 (定义)
  其余 _lm_trunk hit 均为 janus_pro/emu3/nextstep_1/glm_image 各家族自实现自消费
  llamagen 家族与测试只用 _gpt_trunk
  ```
  `LlamaGenChunkExecutor` override `_ar_runner`（`runtime.py:146`）直接构造 `LlamaGenARModelRunner(self.model)` 并对任何 `attention_backend` 请求 raise，故没有生产者能把 llamagen model 交给 backends。body 是一行 `return self._gpt_trunk()`，docstring 自述「shared attention backends ... cannot drive it」——协议已不存在。parked `SPRINT_attention_kernel_medium.md` 只为 Janus/NextStep 规划 `_lm_trunk` driver，零 llamagen 提及。
- 动作：删该方法（保留 `_gpt_trunk`——真实 accessor，被 model 与测试用）。删除后 `build_torch_native_backend(llamagen_model)` 会在 `ar_attention_backends.py` fail-fast 而非深入 vendored GPT，失败模式反而更好。无测试清理。
- 注意（medium）：这是「误导性 capability」而非跨家族一致性——保留会宣告一个家族无法参与的协议。`_gpt_trunk` docstring 里对「Janus' `_lm_trunk`」的提及指的是 janus_pro 的 hook，保持准确。
- ✓ 复核（2026-07-24）：RELOCATED。方法从 183-193 迁到 `llamagen/model.py:153`（docstring ref `:145`），零 llamagen caller；`LlamaGenChunkExecutor` 仍 override `_ar_runner`（`runtime.py:146`），唯一通用 consumer 不可达。form-1 死不变，仅行号迁移。

### lumina2（families-image）

#### 1.16 `Lumina2Model.encode_prompt(system_prompt kwarg)` — dead-arg / form 2（risk=low）

- 位置：`vrl/models/families/lumina2/model.py:185`
- 判死证据：
  ```
  $ grep -rn 'system_prompt' vrl tests
  lumina2/model.py:11,166 (docstring), :185 (system_prompt=kwargs.get("system_prompt"))
  vrl/config/presets/experiment/lumina2/online_grpo_pickscore_validation.yaml (# 注释行, 非 config key)
  ```
  唯一读点即 `:185` 本身。所有 `encode_prompt` call site 只转发 `max_sequence_length/guidance_scale/request/reference_image`，无自由 dict 转发，故 `kwargs.get("system_prompt")` 恒为 `None`。installed diffusers `Lumina2Pipeline.encode_prompt` 签名 `system_prompt: str | None = None`，None 时套默认模板。
- 动作：把 `system_prompt=kwargs.get("system_prompt")` 改为显式 `system_prompt=None`（或删该实参），保留 docstring 中「None 选择 pipeline 默认模板」的说明；修 module docstring（勿再暗示该 kwarg 可由用户设置）。无测试清理。
- ✓ 复核（2026-07-24）：STILL_VALID。仍 `system_prompt=kwargs.get("system_prompt")` at `lumina2/model.py:185`（精确），docstring `:11,:166`。无生产者转发 → 恒 None。不变。

#### 1.17 `Lumina2Model.prepare_sampling(cfg_trunc_ratio guard)` — dead-branch / form 2（risk=low）

- 位置：`vrl/models/families/lumina2/model.py:214-217`
- 判死证据：
  ```
  $ grep -rn 'cfg_trunc_ratio' vrl tests
  lumina2/model.py:24 (docstring), :214 (kwargs.get), :215 (if != 1.0), :217 (raise 文案)
  ```
  仓内仅出现于 lumina2/model.py 自身。`prepare_sampling` 的 kwargs 只来自 `build_prepare_kwargs`：基类实现 `del ...; return None`，唯一 override（`ReferenceConditionedChunks`）返回 `{"reference_image": ...}`（仅 cosmos/wan）。脚本直接 caller 均传零 kwargs。故 `kwargs` 永不含 `cfg_trunc_ratio`，raise 永不触发。
- 动作：删该不可达 guard；module docstring bullet 收缩为一句「not supported」note（保留 replay-contract 依据）。无测试清理。
- ✓ 复核（2026-07-24）：STILL_VALID。guard 仍在 `lumina2/model.py:214-217`（`:214` kwargs.get，`:215` `if != 1.0` raise）；`cfg_trunc_ratio` 仅出现于本文件（docstring `:24` + guard）。零生产者 → 不可达 guard 不变。

### nextstep_1（families-token）

#### 1.19 nextstep `metadata` dict 的 `'family'` key（+ module-level `logger`）— dead-field / form-1 data twin（risk=low）

- 位置：`vrl/models/families/nextstep_1/runner.py:210,255`；`vrl/models/steps/token/paged_attention_helpers.py:244,300`；`nextstep_1/model.py:53,55`
- 判死证据：
  ```
  $ grep -rn '"family"' nextstep_1/runner.py paged_attention_helpers.py
  runner.py:210 metadata={"family": "nextstep_1", "image_token_num": ...}
  runner.py:255 metadata={"family": "nextstep_1"}
  paged_attention_helpers.py:244 metadata={"family": self.family, "image_token_num": ...}
  paged_attention_helpers.py:300 metadata={"family": self.family}
  $ grep -rnF 'metadata["family"]' vrl tests   → 无匹配（零 reader）
  $ grep -n 'logger\|init_logger' nextstep_1/model.py
  :53 from vrl.utils.logging import init_logger ; :55 logger = init_logger(__name__)   （零 logger.* 调用）
  ```
  `ARAttentionPrefillInput/ARAttentionStepInput.metadata` 的唯一 consumer 是 `_max_new_tokens_from_metadata`（`vrl/nn/modules/ar_decoder.py:544`），只读 `'image_token_num'`。backend 家族身份走 `ARAttentionConfig.family`——另一条活路径。`self.family`（`paged_attention_helpers.py:176` `owner = self.lane_owner_prefix or self.family`）是活的非 metadata 消费者，**不删**。
- 动作：从四处 metadata dict 删 `'family'`（保留 `'image_token_num'`）；dict 变空处（`runner.py:255`、`paged_attention_helpers.py:300`）整体省略 `metadata=` 实参（默认空 dict）。保留 dataclass 上的 `Mapping metadata` 字段（有活 reader 的 protocol escape hatch）与 `self.family` 属性。删 `nextstep_1/model.py:55` 的 `logger = init_logger(__name__)` 及 `:53` 其唯一用途的 import。无测试清理。
- ✓ 复核（2026-07-24）：STILL_VALID。metadata `'family'` key 仍在 `runner.py:210,255`（原 226,271）、`paged_attention_helpers.py:244,300`（原 253,317）；唯一 metadata reader `ar_decoder.py:544` 只读 `image_token_num`。logger 仍死（`model.py:53` import + `:55` binding，零 `logger.*` 调用）。`self.family`（helpers:176）活，保留。不变。

### sana（families-image）

#### 1.21 `SanaModel.encode_prompt(complex_human_instruction kwarg)` — dead-arg / form 2（risk=low）

- 位置：`vrl/models/families/sana/model.py:218`
- 判死证据：
  ```
  $ grep -rn 'complex_human_instruction' vrl tests
  sana/model.py:20 (docstring), :218 (=kwargs.get("complex_human_instruction"))
  scripts/eval/sana_inference.py, sana_checkpoint_compare.py ("official_pipeline_default" 是 OFFICIAL SanaPipeline 的 provenance 元数据串, 非 SanaModel.encode_prompt 输入)
  presets/.../sana/online_grpo_pickscore_validation.yaml (# 注释)
  tests/scripts/eval/test_sana_checkpoint_compare.py assert "complex_human_instruction" not in kwargs
  ```
  所有能到达 `SanaModel.encode_prompt` 的 call site 只传固定 kwarg，无 `**` 转发 config dict，故 `kwargs.get` 恒 None。diffusers `pipeline_sana.py` 的 `encode_prompt` 参数默认 `None`（`if not complex_human_instruction:` 关 CHI）。
- 动作：把 `complex_human_instruction=kwargs.get("complex_human_instruction")` 改为显式 `complex_human_instruction=None` 加一行注释（显式 None 记录 VRL 有意关 CHI，防上游默认漂移）；修 module docstring（勿称其「exposed as a sampling kwarg」）。无测试更新（唯一相关测试断言该 key **不在** official-pipeline kwargs）。
- 注意：与 DO-NOT-FLAG 的 sana `prepare_latents=` 修复不同；本条 None-in/None-in 行为不变。
- ✓ 复核（2026-07-24）：STILL_VALID。仍 `complex_human_instruction=kwargs.get(...)` at `sana/model.py:218`（精确），docstring `:20`。无生产者转发 → 恒 None；behavior-preserving 显式-None 修复未落地。不变。

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
- ✓ 复核（2026-07-24）：STILL_VALID。四处赋值仍在 `sd3_5/model.py:119/125/420/426`（精确）；唯一 reader 是 3 条 test assert（`test_attention_processor_install.py:32,45,63`）。test-only 字段不变。

### wan_2_1（families-video）

#### 1.23 `_resolve_guidance_scale_2`（`request.extra['guidance_scale_2']` override 分支）— dead-branch / form 2（risk=medium）

- 位置：`vrl/models/families/wan_2_1/model.py:1357`（**已迁移**，原 1370-1380；callers `model.py:648,1054`）
- 判死证据：
  ```
  $ grep -rn 'guidance_scale_2' vrl tests vrl/config
  model.py:1357 raw = request.extra.get("guidance_scale_2") if request.extra else None
  model.py (state.guidance_scale_2 or state.guidance_scale, expert 路由, 活 replay wire)
  wan_2_2/a14b.yaml boundary_ratio 0.875 / i2v_a14b.yaml 0.9  （dual-expert 路径活）
  vrl/config/ 无 guidance_scale_2 key ; presets 无生产者
  ```
  `request.extra['guidance_scale_2']` 零生产者：rollout 路径唯一 extra 构造 `full_sequence_denoise/executor.py` 只设 `max_sequence_length`，无处向 `request.extra` 写入该 key。非默认值只出现于测试（`test_backbone_parity.py` 直构 state）与 `wan_i2v_base_sample.py`（调 diffusers pipeline，非此路径）。[[SPRINT_sampling_config_knob_unification]] 记录其「wan guidance_scale_2 未动」。dual-expert 路径活，故 `guidance_scale_2` 在每条生产 rollout 里静默恒等于 `guidance_scale`。
- 动作：二选一——**优选接一个真实生产者**（在 executor 把 `sampling.guidance_scale_2` config key 映射进 `VideoGenerationRequest.extra`；Wan2.2 dual-expert preset 在用，上游 `WanPipeline` 确有该 per-expert 旋钮），或删 extra-override 分支使 dual-stage 恒显式镜像 `guidance_scale`。无论哪支，保留 `SamplingState` 字段/batch-context plumbing（它是 rollout 所用值的 replay wire，做 expert 路由）。删除支无测试清理。
- 注意（medium）：这是「旋钮从未接线」而非纯死代码——优选接线以恢复用户可配的 per-expert guidance。
- ✓ 复核（2026-07-24）：RELOCATED。函数从 1370-1380 迁到 `wan_2_1/model.py:1357`，body 仍 `request.extra.get('guidance_scale_2')` override 分支，callers `model.py:648,1054`。`vrl/config/`、`vrl/generation/` 无 `guidance_scale_2` key，零 `request.extra['guidance_scale_2']` 生产者 → g_s_2 恒等 guidance_scale。分支仍死，仅行号迁移。

#### 1.24 `WanT2VReplayModel.__init__(expert_lifecycle_profiling=...)` — dead-arg / form 1（risk=low）

- 位置：`vrl/models/families/wan_2_1/model.py:836`（**已迁移**，replay `__init__` 参数，原 813-835；`prepare_replay` 重导 `:872-873`）
- 判死证据：
  ```
  $ grep -rn 'expert_lifecycle_profiling' vrl tests vrl/config
  schema.py (bool=False) ; presets/.../online_grpo_dual_expert_proof.yaml (true)
  model.py:213/992 (rollout from_build 构造) ; :543 附近 (attr read)
  model.py:836 (replay __init__ param — 本条), :848 (attr set), :872-873 (prepare_replay 重导)
  tests/trainers/test_wan_fsdp_distributed.py policy._expert_lifecycle_profiling = True  （直设属性, 非 ctor kwarg）
  ```
  只有 rollout `from_build`（`model.py:213,992`，走 rollout `__init__`，`pipeline=...`）与 test helper `_build()`（`test_checkpoint_identity.py`，传给 build-config helper，**非** replay ctor）传/涉及该 kwarg；replay 唯一生产构造是通用 replay loader（`vrl/models/steps/denoise/build.py`，只传 transformer/scheduler/device），随后 `prepare_replay`（`model.py:872-873`）从 `build.model_config` 重导真实值。`WanI2VReplayModel.__init__` 转发到 `WanT2VReplayModel.__init__` 时**不带**它。
- 动作：删 replay `__init__` 的 `expert_lifecycle_profiling` 参数，保留 `self._expert_lifecycle_profiling = False` 初始化（属性活：`:543` 附近读、`prepare_replay` 覆盖）。YAML 旋钮/schema 字段经 `build.model_config` → `prepare_replay`/`from_build` 仍全功能。无测试清理。
- ✓ 复核（2026-07-24）：RELOCATED。replay ctor param 从 813-835 迁到 `model.py:836`（`prepare_replay` 重导 `:872-873`）；construction 只在 rollout `from_build`（`model.py:213,992`）与 test helper（传 build-config helper，非 replay ctor）。replay ctor param 零 caller = 死，仅行号迁移。

### models 核心层（models-core）

#### 1.25 `ReplaySegmentResult.logprobs`（`image_logits` / `text_logits` fallback）— dead-branch / form 2（risk=low）

- 位置：`vrl/models/interfaces/replay.py:69,71`
- 判死证据：
  ```
  $ grep -rnF '"image_logits"' vrl tests ; grep -rnF '"text_logits"' vrl tests
  replay.py:69 self.values.get("image_logits") ; :71 self.values.get("text_logits")  （定义自身）
  tests/rollouts/replay/test_replay_result_signals.py:138 values={"image_logits": ...}  （仅用作 require_value 报错测试的任意错 key）
  ```
  所有 `ReplaySegmentResult(` 构造点 values 的 key 是 `log_probs`/`tokens`/`logits`/`image_token_ids`/noise-pred——从不是 `image_logits`/`text_logits`。janus R1 的 `replay_r1_segment` 视觉与文本两支都存到 `"logits"`。`forward_image_logits`/`forward_text_logits` 是方法名，输出存 `"logits"`，与这两个 fallback 无关。
- 动作：删 `:69,71` 两个 fallback lookup，从 docstring 去掉这两个 key 名，保留 `log_probs → logits → require_value("logits")` 序列（删后自然收敛为 `require_value("logits")`，保留测试断言的报错信息）。`test_replay_result_signals.py:138` 用 `"image_logits"` 只作任意错 key、经 `require_value("logits")` 校验，保持不变。无测试更新。
- ✓ 复核（2026-07-24）：STILL_VALID。fallback 仍在 `replay.py:69`（`image_logits`）/`71`（`text_logits`）；8 个构造点无一存这两键，唯一 `image_logits` key 生产者是 deliberately-wrong-key 测试（`test_replay_result_signals.py:138`）。零真实生产者不变。

#### 1.26 `forward_autocast` — single-caller-merge / form 3（risk=medium）

- 位置：`vrl/models/precision.py:27`（def；并入 `model_autocast` `:55`）、`:106`（`__all__`）
- 判死证据：
  ```
  $ grep -rn 'forward_autocast' vrl tests
  precision.py:27 (def), :55 (model_autocast 内唯一生产 caller), :106 (__all__, 零 importer)
  vrl/models/steps/denoise/base.py:87 (陈旧 docstring 提及)
  tests/models/test_precision.py:81,86 (test-only caller, enabled= 的另一唯一用户)
  $ grep -rn 'model_autocast' vrl tests
  真实外部 caller: denoise/base.py, causvid/model.py, algorithms/diffusion_nft.py, trainers/offline/dpo.py
  ```
  一个决策（「哪个 autocast context 生效」）被拆成两函数，第二个（`forward_autocast`）仅一个生产 caller（`model_autocast`）；`enabled=` kwarg 也只为 `model_autocast` 传 `precision.outer_autocast` 而存在。`__all__` 里导出却全仓零 importer，无 `from vrl.models.precision import *`。`registry.py` 的 `getattr(precision, ...)` 读的是 PrecisionPolicy role 属性，非本模块。
- 动作：把 `forward_autocast` body 内联进 `model_autocast`（其唯一生产 caller），删 `forward_autocast` 及其 `enabled=` kwarg。改 `tests/models/test_precision.py:81-86` 用带 `RolePrecision` 的 stub model 走合并后的 `model_autocast`（`RolePrecision` 暴露 `dtype`/`outer_autocast`，可覆盖同一 `(dtype, enabled)` 矩阵）。从 `__all__` 删 `forward_autocast`，修 `vrl/models/steps/denoise/base.py:87` 陈旧 docstring 提及。
- 注意（medium）：`__all__` 导出使其看似 public facade——但零 importer + 仅一 caller 用的 `enabled` kwarg，该拆分已不再命名任何被消费的概念，故按 form-3 合并。内联后 `dtype not in ("fp16","bf16")` ValueError 分支成不可达防御码（`RolePrecision.__post_init__` 已把 dtype 限于 fp32/bf16/fp16），可删可留（照抄无害）。`precision.py` 不在 in-flight sprint 修改集；`base.py:87` 只是一句 docstring 编辑，不与 sprint 冲突。
- ✓ 复核（2026-07-24）：STILL_VALID。仍在 `precision.py:27`（def）/`55`（唯一生产 caller `model_autocast`）/`106`（`__all__`）；陈旧 docstring 提及现在 `denoise/base.py:87`（原 :79）；test-only caller `test_precision.py:81,86`。form-3 合并候选不变。

#### 1.27 `DiffusionModelBase.load` — dead-function / form 1（risk=low）

- 位置：`vrl/models/steps/denoise/base.py:75`（**已迁移**，原 67-69）
- 判死证据：
  ```
  $ grep -rn 'async def load' vrl tests   →  只 base.py:75（无家族 override）
  $ git status --porcelain vrl/models/steps/denoise/base.py  →  ' M vrl/models/steps/denoise/base.py'
  ```
  `grep '\.load(' vrl tests`（排除 json/yaml/torch/load_state_dict）零 `model.load()` call site；无 `await ...load`（所有命中是 `offload()`）；无 `load.remote` Ray 调用；无 `"load"` 字符串派发（`magi_1` runtime dict 的 `"load"` 是 Megatron 风格 checkpoint 路径 config key，非方法派发）。`vrl/models/interfaces/`（`replay.py`/`runtime.py`）无 `load` protocol。body 是裸 `return None`，docstring「Default no-op for adapters constructed eagerly」——协议已不存在。
- 动作：删 `async def load` 方法。无测试更新。
- 注意（**sprint 重叠**）：本条所在 `vrl/models/steps/denoise/base.py` 在 in-flight [[SPRINT_native_generation_engine_program]] 的未提交改动集内。复核确认其未提交 diff 仅动 `export_batch_context` 等，`load` 未被触碰；且 active/planned/parked sprint 文档只提 `offload()`、无 `load()` hook 计划。仍应**排在该 sprint 之后**落地，落地前重跑 `grep 'async def load' base.py` 确认无新 override，避免与并发编辑冲突。
- ✓ 复核（2026-07-24）：RELOCATED。`async def load` 从 67-69 迁到 `base.py:75`；零 override、零 caller。`base.py` 仍在 in-flight 未提交集内但方法仍死，仅行号迁移。

#### 1.30 `LatentOutputLayout`（`'video_bcthw'` 值）— dead-branch / form-2 data twin（risk=low）

- 位置：`vrl/models/steps/denoise/common/latent_decode.py:11`
- 判死证据：
  ```
  $ grep -rn 'video_bcthw' vrl tests
  latent_decode.py:11 LatentOutputLayout = Literal["image_bchw", "video_btchw", "video_bcthw"]   （仅 Literal 定义自身, 全仓唯一命中）
  ```
  14 个家族 `output_layout=` 生产者全部硬编码 `'image_bchw'` 或 `'video_btchw'`（纯 Python，无 YAML）；两个测试文件只用这两值。`ChunkedLatentDecoder.__call__` 只 special-case `'video_btchw'`，`'video_bcthw'` 与 `'image_bchw'` passthrough 无差别。`test_wan_decode_latents_preserves_bcthw_layout` 是烟雾弹——它测 Wan 的 `'video_btchw'` plan，从不用 `'video_bcthw'` 串。
- 动作：从 `LatentOutputLayout` Literal 删 `'video_bcthw'`。无 decoder 逻辑改动、无测试改动（纯类型标注编辑）。
- ✓ 复核（2026-07-24）：STILL_VALID。`video_bcthw` 仍恰一处命中：`latent_decode.py:11` Literal；无生产者设它，decoder 只 special-case `video_btchw`。死词汇值不变。

#### 1.31 `TrainableStateSlots.versions` — dead-function / test-only caller（risk=low）

- 位置：`vrl/models/utils.py:147`
- 判死证据：
  ```
  $ grep -rn '\.versions()' vrl tests
  tests/models/test_utils.py:23 == [1,2] ; :32 == [2,3] ; :51 == []   （无 vrl/ 命中）
  ```
  生产 consumer（`vrl/models/steps/denoise/base.py` 的 install/has/get/_evict，由 `vrl/generation/execution/worker.py` 驱动）从不调 `versions()`。body 是 `return sorted(self._slots)`——纯测试可观测。in-flight sprint 只规划 slot 激活/驱逐（has/install/get），非 version 枚举。
- 动作：删 `versions()` 方法；把 `tests/models/test_utils.py:23,32,51` 三条断言按 `has()`（对期望/驱逐版本号）改写（安装顺序、驱逐窗口、空安装 no-op 这些真实断言用 `has()` 全可表达；`:32` 已与相邻 `has()` 断言冗余，`:51` 改写为 `not slots.has(1)`）。
- ✓ 复核（2026-07-24）：STILL_VALID。`versions()` 仍在 `utils.py:147`；唯一 caller 是 3 条 test assert（`test_utils.py:23,32,51`）。零生产 caller = TEST-ONLY = 死，不变。

#### 1.32 `count_trainable_params` — dead-function / test-only caller（risk=low）

- 位置：`vrl/models/utils.py:158`
- 判死证据：
  ```
  $ grep -rn 'count_trainable_params' vrl tests
  vrl/models/utils.py:158 (def)
  tests/models/families/emu3/test_replay.py:21 (import), :163 (assert ... > 0)
  tests/models/families/glm_image/test_replay.py:16 (import), :185 (assert ... > 0)
  ```
  零生产 caller，无字符串/registry/YAML 引用，无 `__all__`/`__init__` 再导出。in-flight `base.py` 从 `vrl.models.utils` 只 import 另四个符号、不含此。body 是一行 sum comprehension。
- 动作：删 `count_trainable_params`（`:158`）。两个测试 call site 把 `assert count_trainable_params(model) > 0` 内联为 `assert sum(p.numel() for p in model.parameters() if p.requires_grad) > 0`，**并删两处现已无用的 import 行**（`emu3/test_replay.py:21`、`glm_image/test_replay.py:16`）。跑 ruff check 确认无 F401 残留。
- 注意（关联）：[[SPRINT_dead_code_wrapper_sweep]] 曾在删九个 per-family `trainable_param_count`/`has_lora_adapter` 时把这两测试有意接到此共享 util（done doc 不使其存活，但若想保留合并意图，可把函数移入 `tests/` 共享 helper 并重指两处 import，等价有效）。
- ✓ 复核（2026-07-24）：STILL_VALID。仍在 `utils.py:158`；唯一 caller 是测试（`emu3/test_replay.py:21` import/`:163` call；`glm_image/test_replay.py:16` import/`:185` call）。零生产 caller = TEST-ONLY = 死；测试 call-site 行号略移，判死不变。

## 2. 已由 origin 落地（本次复核确认，无需再做）

> 以下 7 条原发现已被 main @ `7c748532`（audit 旧树 `88ed756e` 之后的约 63 个 cleanup commit）独立完成，grep 全仓已无残留。仅作记录，无动作。

- **`CausVidResolvedArtifacts.{source_root, source_revision, base_model_revision, checkpoint_revision}}`**（原 §1.3，causvid/model.py）— `Resolved*` struct 上四个零 reader 死字段，现 struct 收缩为仅 `base_model_dir + checkpoint_file`，并新增回归测试 `test_resolved_artifacts_retain_only_runtime_paths` 锁定字段集 — landed by `a8848c12` refactor(causvid): retain only resolved runtime paths。
- **`_EMU3_LORA_DEFAULTS` + `_JANUS_/_GLM_IMAGE_/_NEXTSTEP_/_LLAMAGEN_LORA_DEFAULTS`**（原 §1.9，五个 token 家族 runtime.py，**本簇最锋利的一条**）— 手维护、与 Config dataclass 双维护且已漂移的 LoRA fallback dict，现全部删除，fallback 改从家族 Config 派生（正是原提案）；新增 `test_family_config_supplies_lora_defaults_when_block_is_absent` 佐证 — landed by `f4730774` refactor(token): derive LoRA fallbacks from family configs。
- **`GlmImageChunkExecutor._runner_cls / _runner_attention_family`**（原 §1.10，glm_image/runtime.py）— 被完全 override 的 `_ar_runner` 使其成 test-only 死描述符，连同两条 pin 断言一并删除 — landed by `d8235a10` refactor(ar): remove overridden runner descriptors。
- **`LlamaGenChunkExecutor._runner_cls / _runner_attention_family`**（原 §1.15，llamagen/runtime.py）— 同上，同一 commit 删除 — landed by `d8235a10` refactor(ar): remove overridden runner descriptors。
- **nextstep `'token_dim'` 转发项**（原 §1.20，nextstep_1/runtime.py）— `nextstep_config_from_build` sampling key 列表里零生产者的死转发项，已从列表移除（`NextStep1Config.token_dim` 字段仍活，保留）— landed by `3e6367c9` refactor(config): colocate NextStep model config。
- **`DiffusionBackboneOutput.metrics`**（原 §1.28，denoise/common/backbone.py）— `as_dict()` 刻意省略的 test-only 结果状态字段及其 `metrics={"transformer_calls": ...}` population，已删除 — landed by `6ed86b60` refactor(denoise): remove test-only result state。
- **`ARDiscreteTokenRunner.step_token(generator kwarg)`**（原 §1.29，token/base.py）— 唯一生产者（nextstep_1）走独立 override，基类此 kwarg 恒被 `del generator` 丢弃；kwarg 与 `del` 行已删 — landed by `6a75406a` refactor(token): thin token step protocol。

## 3. 情况已变（需重新评估）

> 以下 3 条的现场在 main @ `7c748532` 已变，原动作不再原样适用。

- **`GenerationRuntimeCapabilities.runs_in_isolated_subprocess`**（原 §1.1，`vrl/families/registry.py:61`）— **verdict=CHANGED**。原发现的两个前提已被 `e9bb20d0`（refactor(registry): derive policy replay support）改动：`magi_1` 处的 `runs_in_isolated_subprocess=True` setter **与两个 test reader**（`test_vae_decode_memory.py`、`test_family_registry.py`）**均已删除**。但**字段本身没被删**，现完全孤立在 `registry.py:61`（default `False`，零 reader，永不置 `True`，原 :41）。原动作（删字段 + 改 `test_vae_decode_memory` 硬编码家族集 + 删/改 `test_family_registry` 断言 + 重命名测试）里除「删字段一行」外**全部已不适用**——不再有 setter 行、不再有测试断言要改。重新评估结论：只剩一条裸死字段一行可删，删前无需再动任何测试；风险从 medium 降为 low。
- **`JanusProModel.decode_image_tokens(image_size=...)`**（原 §1.11，`janus_pro/model.py:930`）— **verdict=CHANGED**。原推荐的**校验支已由 `6a650983`（fix(models): reject unsupported inference inputs）落地**：`decode_image_tokens` 现校验 `image_size`——非 int raise `TypeError`（`:950-951`），`image_size != side * JANUS_IMAGE_PATCH_SIZE` raise `ValueError`（`:956-957`）。该 arg 已由 no-op 旋钮变为**行为消费**，正是原动作想要的正解。重新评估结论：**无需再做**，原「无效用户旋钮」判定已失效。
- **`NextStep1Model.decode_image_tokens(image_size=...)`**（原 §1.18，`nextstep_1/model.py:292-316`）— **verdict=CHANGED**。同 `6a650983`：旧 `del image_size` 已替换为对 VAE 输出的校验——`image_size is not None` 时若 `actual_size != (image_size, image_size)` 则 raise（`:307-316`）。该 arg 已行为消费，不再是 required-but-ignored 旋钮。重新评估结论：**无需再做**。

## 4. 验证协议

- **每条删除后**：`ruff check <touched files>` + `ruff format --check <touched files>`（仅对本任务触及的 Python 文件；先 `ruff check --fix`，再 `ruff format`，最后 `ruff check` + `ruff format --check` 复核）。禁止全仓 `ruff format .` / `ruff check --fix .`。
- **全簇完成后**：`pytest tests/models/ tests/rollouts/ tests/families/ tests/generation/ tests/config/` + `pytest -m "not e2e and not slow_test"` 子集不新增失败。
- **基线（本次复核，2026-07-24，main @ `7c748532`）**：清理前的 fast subset 须在 origin 已落地 §2 各 commit 之上重取一次（旧树 `88ed756e` 的 2620 passed / 7 pre-existing failures 已过期，不可作基线）；删除后须与该新基线持平，且 `vrl.config.lint` 与 `ruff check .` 保持全绿。
- **剩余各条动作触及的测试文件**（仅列 §1 保留条目）：
  - §1.2：`tests/rollouts/runtime/test_janus_pro_r1_wiring.py`（源码侧动 `vrl/rollouts/collector/batch_builder.py`、`core.py`；`semantics.py`/`registry.py` 不动）
  - §1.4：无
  - §1.5：`tests/models/families/causvid/test_runner.py`、`test_replay_and_loading.py`
  - §1.6：无（anima 测试 stub adapter）
  - §1.7：无
  - §1.8：无
  - §1.12：`tests/models/families/janus_pro/test_r1_model.py`
  - §1.13：`tests/models/families/janus_pro/test_r1_model.py`、`tests/models/steps/token/fixtures.py`（+ 验证 `tests/rollouts/runtime/test_janus_pro_r1_wiring.py`、`tests/config/test_schema.py`）
  - §1.14：无
  - §1.16：无
  - §1.17：无
  - §1.19：无
  - §1.21：无（`tests/scripts/eval/test_sana_checkpoint_compare.py` 保持）
  - §1.22：`tests/models/families/sd3_5/test_attention_processor_install.py`
  - §1.23：无（接线或删分支均无 cleanup）
  - §1.24：无
  - §1.25：无（`tests/rollouts/replay/test_replay_result_signals.py` 保持）
  - §1.26：`tests/models/test_precision.py`
  - §1.27：无
  - §1.30：无
  - §1.31：`tests/models/test_utils.py`
  - §1.32：`tests/models/families/emu3/test_replay.py`、`tests/models/families/glm_image/test_replay.py`

## 5. Non-Goals

- 不删被「能 raise 的校验」、控制流分支、runtime/config/Ray 调用消费的字段（AGENTS 死字段规则）。§1.19 的 `self.family`、§1.24 的 `self._expert_lifecycle_profiling` 属性、§1.3 保留的 `_resolve_source_root`/`_resolve_checkpoint` 校验副作用（该条主体已落地，见 §2）均属此列，保留。
- 不动 DO-NOT-FLAG 豁免项：`_GENERATION_CUDA_RUNTIME_RESIDUAL_BYTES`（`generation/execution/worker.py`）、`_REWARD_CUDA_RUNTIME_RESIDUAL_BYTES`（`rewards/base.py`）、`ensure_loaded`（`rewards/runtime.py`）、`process_gpu_used_bytes` NVML（`utils/cuda_memory.py`）、sana/hunyuan 的 `prepare_latents=` 修复。
- 不为省行数扁平化 protocol/lazy-import/跨家族一致性 thin function：§1.5 的 `predict_x0_cached` protocol 边界、§1.19 的 dataclass `Mapping metadata` escape hatch 均保留。
- **cluster-specific non-goals**：
  - `LORA_DEFAULTS` 去重（原 §1.9）已由 origin 落地（§2）——落地方向正是「删 `runtime.py` 五个 fallback dict、保留 Config dataclass `lora_*` 默认值、不改 YAML preset」；不再作为 planned 项。
  - `decode_image_tokens(image_size=...)` 的校验支（原 §1.11/§1.18）已由 `6a650983` 落地（§3）——删除支（改共享 `ARSamplingParams`/`ARRequestLayout`）依旧是 non-goal，且已无必要。
  - §1.23 的 `guidance_scale_2` **优选接线而非纯删**（用户可配的 per-expert guidance），保留 `SamplingState`/batch-context replay wire。
  - §1.12/§1.13 强耦合，须同批落地，且只删 `final_image` 臂的 `stages` 半、保留 `mode == "never"`。
  - §1.2 只删/激活 `PolicySemantics` 上的 `trajectory_layout` 死字段，不删整个结构（同结构其余字段活）；走「使字段成 source of truth」而非删除，遵从 [[SPRINT_generation_regime_decision_layering]]。

## References

- `vrl/families/semantics.py:38,49-62`；`vrl/rollouts/collector/batch_builder.py:229`、`core.py`（`build_rollout_collector` entry 穿线）；构造点 `registry.py:476,496,509,539,916`
- `vrl/models/families/causvid/model.py:187,190,552`、`runner.py:143,220,225,269,280,292,304,331`
- `vrl/models/families/cosmos/anima/adapter.py:24,60,63,64,84`、`model.py:550`
- `vrl/models/families/echo/model.py:146`；`vrl/models/steps/denoise/common/lora.py:34,59`
- `vrl/models/families/emu3/model.py:239-240,338,663-664`
- `vrl/models/families/janus_pro/config.py:60`、`model.py:473,489,499,501,750-751`、`runtime.py:283,400,408,416-417`；`vrl/config/schema.py:278,994-997`
- `vrl/models/families/llamagen/model.py:145,153`、`runtime.py:146`；`vrl/nn/modules/ar_attention_backends.py`
- `vrl/models/families/lumina2/model.py:11,166,185,214-217`
- `vrl/models/families/nextstep_1/model.py:53,55`、`runner.py:210,255`；`vrl/models/steps/token/paged_attention_helpers.py:176,244,300`；`vrl/nn/modules/ar_decoder.py:544`
- `vrl/models/families/sana/model.py:20,218`
- `vrl/models/families/sd3_5/model.py:119,125,420,426`；`vrl/scripts/perf/quantized_sd3_forward_profile.py:112`；`tests/models/families/sd3_5/test_attention_processor_install.py:32,45,63`
- `vrl/models/families/wan_2_1/model.py:648,836,848,872-873,1054,1357`；`presets/experiment/wan_2_2/online_grpo_dual_expert_proof.yaml`
- `vrl/models/interfaces/replay.py:69,71`
- `vrl/models/precision.py:27,55,106`；`vrl/models/steps/denoise/base.py:75,87`
- `vrl/models/steps/denoise/common/latent_decode.py:11`
- `vrl/models/utils.py:147,158`；`tests/models/test_utils.py:23,32,51`、`tests/models/families/{emu3,glm_image}/test_replay.py`
- 已落地条目 commit：`a8848c12`（causvid resolved artifacts）、`f4730774`（LoRA fallback 派生）、`d8235a10`（runner descriptors）、`3e6367c9`（NextStep config colocate）、`6ed86b60`（backbone metrics）、`6a75406a`（token step protocol）；情况已变条目：`e9bb20d0`（policy replay derive）、`6a650983`（decode_image_tokens 校验）
- 关联 sprint：[[SPRINT_deadcode_00_overview]]、[[SPRINT_trajectory_views_types_dead_fields_cleanup]]、[[SPRINT_generation_regime_decision_layering]]、[[SPRINT_sampling_config_knob_unification]]、[[SPRINT_dead_code_wrapper_sweep]]、[[SPRINT_native_generation_engine_program]]（in-flight）、[[SPRINT_attention_kernel_medium]]（parked）
