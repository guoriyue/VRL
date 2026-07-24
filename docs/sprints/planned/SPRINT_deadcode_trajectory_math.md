# SPRINT: `trajectory` 与 `math` 死字段/死参数/死分支清理（planned · RECONCILED）

状态：**RECONCILED（2026-07-24）against main @ `7c748532`**（= `origin/main` tip）。原审计跑在旧树 `88ed756e`，此后约 63 个 cleanup/refactor commit 落地，原 15 条中**多数已被 origin 自行清掉**。本次复核逐条对当前 checked-out 树重新验证，得出：
- **7 条仍需做**（4 条 STILL_VALID + 3 条 RELOCATED，位置行号已随周边重构漂移）— 见 §1；
- **6 条已由 origin 落地**（复核确认，无需再做）— 见 §2；
- **2 条情况已变**（需重新评估，原动作已不能原样执行）— 见 §3。

> **执行状态（2026-07-24）**：§1 全部 7 条已落地 `cd6beea5`（逐 receiver 消歧）。§3 的 2 条 CHANGED 未做（origin 已加回归测试）。

来源：dead-code-audit workflow（五种死代码形态 + 对抗验证 + 删除类二次字符串引用检查），原始逐条证据/动作文本保留在 `scratchpad/cluster_trajectory_math.json`。
关联：[[SPRINT_deadcode_00_overview]]、[[SPRINT_trajectory_views_types_dead_fields_cleanup]]（本簇沿用其「逐 receiver 消歧」纪律，并复用其 §1.3/§1.4 关于 `_reject_runtime_state` 是合法 consumer 的裁定）、[[SPRINT_grab_bag_file_audit]]（`_flow_terminal_mean` / `_apply_value_policy` / `_primary_segment_name` 的历史抽取记录）。

## 0. 一句话

本簇清理 `vrl/math/`（renoise、flow_matching）与 `vrl/trajectory/`（builders、ops、resolver、storage、types）两处的死字段、死参数、死分支与一个死函数。**复核后**：origin 已把大部分 dead data key / dead-arg（`trajectory_mode` / `segment_names` / `num_*` 结构计数键 / `reward_segments=` / `primary_trainable_segment_name(fallback=)` / `RenoiseStepResult.std_dev`）清掉。剩下仍需做的是 `vrl/math/token/flow_matching.py` 的两条数值路径（`saved_noise=None` fallback 与 `velocity_fn=`）、`renoise.py` 的 `generator=` 死参数、`builders.py` 的 `_segment_trainable` 死分支与两个 chunk builder 的 `*_axis` 死键、以及 `storage.py` 的两条 duplicate/dead-branch。**首要误删风险不变**：trajectory 各 struct 大量复用同名字段（`metadata` / `advantage_scope` / `segment_names` / `std_dev`），每条动作都必须先做 **逐 receiver 消歧**，确认命中的是目标类型而非同名兄弟字段，再落删。

## 1. 待删清单（仍有效）

保留 STILL_VALID + RELOCATED 两类，共 7 条。**子编号沿用原审计编号**（§1.1/§1.5/§1.7/§1.9/§1.10/§1.11/§1.12/§1.13 因已落地或情况变化，分别移入 §2 / §3，故此处编号有跳号，便于回溯 JSON 与复核结论）。RELOCATED 条目的「位置」已更新为当前 `file:line`，并注明行号漂移来源。

---

### 1.2 `renoise_step_with_logprob(generator=)` 与 fresh-noise randn fallback — dead-arg（risk=medium）｜**RELOCATED**

- 位置（复核 2026-07-24，已漂移）：`vrl/math/denoise/renoise.py:30`（`generator` 参数）、`:48-51`（校验块）、`:71-78`（`torch.randn` fallback）；测试 `tests/math/denoise/test_renoise.py`。
  - 行号漂移原因：commit `6ed86b60` 从 `RenoiseStepResult` 删了两个 test-only 结果字段（见 §2.1），把原 `32 / 50-53 / 74-80` 整体上移到 `30 / 48-51 / 71-78`。参数与 fallback 本体**仍在、仍死**。
- 判死证据（仍成立）：两个生产 caller 都不传 `generator`——`runner.py:276` 传 `noise=transition_noise`，`model.py:548` 传 `next_sample=`；无 `**kwargs` 转发。`runner.py` 的 `_transition_noise` 为每个 sample 各建一个 `torch.Generator` 并 `torch.randn(generator=...)`——单个 `generator=` 参数无法表达，这正是 `noise=` 路径存在的原因。与 `sde_step_with_logprob`（其 `generator=` 确被 `vrl/generation/steps/denoise/loop.py:232` 传入）不同，签名对齐不构成保留理由。fallback：唯一裸调用 `test_renoise.py:52`（原 51，随测试文件微移）用 `sigma<=0`，在 `renoise.py:66-67` 的 sigma 校验处即 raise，永不抵达 randn 分支——生产与测试都零抵达。
  - 消歧注意：`causvid` 侧 `runner.py:380` / `model.py:653` 出现的 `generator=` 是另一个独立 `torch.Generator`，**不转发给 renoise**，勿误判为 passer。
- 动作：(a) 删 `generator` 参数（:30）；(b) 将校验块（:48-51）改为两个显式检查：`noise` 与 `next_sample` 同时给出时 `raise ValueError("next_sample cannot be combined with noise")`，两者皆 None 时 `raise ValueError("exactly one of noise or next_sample is required")`；(c) 删 `torch.randn` fallback（:71-78），只保留 noise-supplied 分支（device/dtype 迁移 + shape 校验）；(d) 更新 docstring，去掉 `generator` 提及，说明采样恒消费 caller 提供的 noise。测试：更新 `test_renoise.py` 的 `test_renoise_rejects_deterministic_or_reverse_sigma`，传 `noise=torch.zeros(1, 2)`，使 required-arg 检查不抢先于它要断言的 sigma 校验；新增一条断言「有效 sigma 但缺 noise/next_sample 时抛新的 required-arg ValueError」。
- 注意（medium）：这是数值路径改动。新增的 required-arg 校验会改变该测试的错误抛出顺序——原本裸调用（既无 noise 也无 next_sample）期望的是「sigma must be > 0」，若 required-arg 检查置前会抢先抛错。故测试改写与参数删除必须同一条动作内完成，不可分离。

---

### 1.3 `flow_logprob_at` 的 `saved_noise=None` fallback 分支 — dead-branch（risk=medium）｜**STILL_VALID**

- 位置：`vrl/math/token/flow_matching.py:207`（默认 `None`）、`:238-245`（fallback 分支，含 `:242` 的「Fallback mode: fresh x_0 ~ N(0, I)」自述），docstring 陈旧策略 `220-221` 一并清。
  - 复核：`flow_matching.py` 自 `88ed756e` 起**零 commit**（`git log 88ed756e..HEAD -- vrl/math/token/flow_matching.py` 为空），行号与原审计逐一对齐。
- 判死证据：唯一生产 caller `nextstep_1/model.py:200` 无条件传 `saved_noise=saved_noise[:, j]`；该参数在 `model.py:167` 是必填 `torch.Tensor`，经 `replay["saved_noise"]` 方括号取值（`model.py:261`，缺失即 KeyError，不会是 None）。`build_ar_continuous_trajectory` 把 `saved_noise` 作必填 replay 张量记录，没有任何 producer 能供 None——form 2 dead-semantics 确认。分支自述为偏置 Monte-Carlo，无人选中。
- 动作：把 `saved_noise` 改为必填 `torch.Tensor`，删 `238-245` 的 fallback 分支，并裁掉 docstring 中已陈旧的策略「(b) accept a Monte-Carlo approximation by re-running with a fresh prior」（`220-221`）。唯一 test caller `tests/math/test_token_flow_matching.py` 传 `saved_noise=x0`，不触及 fallback，故无测试清理需求；`saved_noise` 位于 `*` 之前，关键字 caller 不受影响。
- 注意（medium）：数值语义分支。`info/SPRINT_rollout_performance.md:1059` 明确记录 `saved_noise` 重放（collection 点 importance ratio = 1）为设计意图，与 fresh-noise fallback 相反——删分支是与既定设计对齐，而非改变数值契约。

---

### 1.4 `velocity_fn`（`flow_sample_with_logprob` / `flow_logprob_at` / `_flow_terminal_mean`）— dead-arg（risk=medium）｜**STILL_VALID**

- 位置：`vrl/math/token/flow_matching.py:37`（`_flow_terminal_mean`）、`:112`（`flow_sample_with_logprob`）、`:213`（`flow_logprob_at`）三处签名；override 分支 `:58-59`（连带 docstring 与内部转发 kwarg）。
  - 复核：文件自 base 未变，行号对齐。
- 判死证据：生产 caller `nextstep_1/runner.py` 与 `model.py` 显式传每个其他 kwarg 却从不传 `velocity_fn`，且无 `**kwargs` 转发。全仓 passer 仅 `tests/math/test_token_flow_matching.py:63,90,109`，三处全传给 private `_flow_terminal_mean`——两个 public 函数的 `velocity_fn` 参数零 passer（含测试）。唯一行为用途是 `:58-59` 的 override 分支，其提供的 seam 已被既有 duck-typed `image_head.net(x,t,c)` 路径完全复制（同测试文件的 `_FakeHead` 已驱动两个 public 函数而不用 velocity_fn）。镜像的 `sde_logprob.py` 无同类 callable-override，故无跨家族一致性保留理由。
- 动作：从三处签名移除 `velocity_fn`，删 `_flow_terminal_mean._velocity` 里的 `if velocity_fn is not None` 分支；改写三处 test call site（`test_token_flow_matching.py:63,90,109`）为传一个 duck-typed fake head，暴露 `.net(x,t,c)`（`_flow_terminal_mean` 的 fake 不必带 `.input_dim`，只有 `flow_sample_with_logprob` 读它）。连带清理 docstring 与内部转发 kwarg。
- 注意（medium）：这三处是 test-only passer（form 1 的 test-only 变体）——public 函数零 passer，private `_flow_terminal_mean` 仅测试传入。改写测试须保持 CFG override 覆盖等价：extend 既有 `_FakeHead` 暴露 `.net` 即可复现同一 seam。

---

### 1.6 `_TrajectoryBatchBuilder._segment_trainable` 的死分支 — dead-branch（risk=medium）｜**RELOCATED**

- 位置（复核 2026-07-24，已漂移）：`vrl/trajectory/builders.py:807`（def）、`808-820`（六分支 body）。
  - 行号漂移原因：`builders.py` 因其他重构收缩，原 `928-942` 上移到 `807-820`；六分支（train/enabled/None→visual/dict/str/sequence/bool）与原审计逐字一致，**仍死**。
- 判死证据（仍成立，读体确认）：唯一生产 payload builder `_cat_segment_extra`（`janus_pro/runtime.py`）的键为 `name/token_ids/token_log_probs/token_mask/prompt_embeds/attention_mask/prompt_attention_mask/visual/cfg`——无 `train`、无 `enabled`。
  - `'enabled'` 分支零 producer：唯一另一个 `'enabled'` reader 在 `multi_segment_token_logprob.py:123`（原 126），读的是 replay-side payload（由 `_trajectory_segment_payload` 从 `segment.metadata` 构造），与 `_segment_trainable` 的 input payload 是两条不同数据链；且 `builders.py:683`（原 752-756）把 `metadata["train"] = trainable`（计算输出）写回，不构成反馈到 `_segment_trainable` 输入的回路。
  - `str/list-set/trailing-bool` 分支零 producer：`train_segments` 唯一 producer 链是 `algorithm.train_segments`（`dict[str,bool]` dataclass 字段 `multisegment.py:31`（原 27）；YAML mapping `token_grpo_multisegment.yaml:13`）经 `_copy_value`/`to_builtin_deep`，非 dict 值会在 `factory.py` 启动时崩溃。
  - `'train'` 分支是 TEST-ONLY producer（`test_r1_model.py`、`test_multisegment_token_logprob.py`）。
- 动作：缩到有 producer 的分支——dict `train_segments` → `value.get(name, False)`；`value` None → `payload.get('visual', True)`。删：`'enabled'` payload-key 分支、`isinstance str` / `list-tuple-set-frozenset` / 尾部 `bool(value)` 分支。`'train'` payload-key 分支删掉，并把 `test_r1_model.py`、`test_multisegment_token_logprob.py` 改为经 `request.sampling['train_segments']` 驱动 trainability（如 `{'initial_image': True, 'selfcheck_text': True, 'final_image': True}`，经保留的 dict 分支复现同一 trainable 值，含 `selfcheck_text=True`——若退到 visual-default 会被翻成 False）。第三个 test caller `tests/rollouts/runtime/test_janus_pro_r1_wiring.py` 的 payload 只含 `visual/cfg`、无 sampling `train_segments`，走保留的 None→visual 分支，无需改写。
- 注意（medium）：**逐 receiver 消歧关键**——判死前须读体确认两个 `'enabled'` 是不同 payload（input payload vs replay-side metadata payload），且 `metadata["train"] = trainable` 是计算输出而非反馈输入。

---

### 1.8 `segment.metadata['temporal_chunk_axis']` / `['transition_axis']` — dead-field（risk=low）｜**RELOCATED**

- 位置（复核 2026-07-24，已漂移）：`vrl/trajectory/builders.py:294-295`（两键在 `build_chunk_autoregressive_denoise_trajectory`）、`:363`（`temporal_chunk_axis` 在 `build_chunk_autoregressive_generation_trajectory`）。
  - 行号漂移原因：周边 context-mirror / metrics 重构上移，原审计 `302-305 / 388` → 现 `294-295 / 363`。两键仍是 axes dict 的 write-only 重复，**仍死**（`grep temporal_chunk_axis|transition_axis` 全仓仅命中这三处构造点，零 reader）。
- 判死证据（仍成立）：`TrajectorySegment.metadata` 的泛型 consumer（`validation.py` `_reject_runtime_state`、`ops.py` 盲 dict copy）content-agnostic；唯一按键读 segment metadata 的是 token evaluator 的 `'visual'/'cfg'/'train'`（不同 segment 不同键）。chunk replay 路径（`chunk_autoregressive_logprob.py`、`causvid/model.py`）零 `.metadata` 读。键重复编码了 `trajectory.axes` 已首类声明的轴名（`temporal_chunk`/`denoise_transition`）——duplicate-construction 数据形态。
- 动作：删两个 chunk builder 里的 `metadata={'temporal_chunk_axis': ..., 'transition_axis': ...}` 项（denoise builder 两键、generation builder 的 `temporal_chunk_axis`）。无测试断言这些键，无测试清理。
- 注意（low）：逐 receiver 消歧——`.metadata` 在 trajectory 层被多个 struct 复用，须确认命中的是这两个 chunk builder 的 `TrajectorySegment.metadata` 写入，而非同名的 `TrajectoryTensor.metadata`/`ReplayInput.metadata`。

---

### 1.14 `trajectory_storage_policy_from_cfg` 的 isinstance pass-through 与 cfg_get fallback — dead-branch（risk=low）｜**STILL_VALID**

- 位置：`vrl/trajectory/storage.py:48-49`（isinstance pass-through）、`:55-57`（cfg_get fallback）。
  - 复核：两分支在原行号原样存在（`git log -S 'isinstance(value, TrajectoryStoragePolicy)' 88ed756e..HEAD -- storage.py` 为空；commit `0b9d490d` 只动了 storage.py 别处）。函数体结构不变：`None→default`(46) / `to_builtin`(47) / isinstance pass-through(48-49) / `Mapping→parse`(51-54) / cfg_get fallback(55-57)。
- 判死证据：两个生产 caller（`full_sequence_denoise/executor.py`、`rollouts/collector/core.py`）只交付 plain builtins；`to_builtin` 把任何 `DictConfig` 展开为 plain dict，Mapping 分支覆盖所有 OmegaConf 形态；`schema.py` 的 `ConfigBlock(TrajectoryStoragePolicy)` 只派生已知键名 frozenset 供 unknown-key lint，从不实例化该类型。故无路径能送入 policy 实例或非 mapping 属性对象——form 2 确认。
- 动作：删 `isinstance(value, TrajectoryStoragePolicy)` pass-through（48-49）；将尾部 cfg_get fallback（55-57）**替换为显式** `raise TypeError(f"rollout.trajectory_storage must be a mapping with 'device'/'dtype' keys, got {value!r}")`，使非 mapping YAML 值（如 `trajectory_storage: cpu`）响亮失败而非静默退化为 default no-op policy。保留 `None → default` 与 `Mapping → parse`。可选在 `tests/trajectory/test_storage_policy.py` 加一条断言标量输入抛 TypeError；无既有测试触及被删分支。
- 注意（low，动作非纯删）：**不可裸删两分支**——裸删会让函数落到末尾隐式 return None，违反 `-> TrajectoryStoragePolicy` 契约，并在下游以不透明 AttributeError 崩溃。尾部必须改为显式 error path。一个可达的 YAML 误配（`trajectory_storage: cpu` 裸字符串经 `_copy_value`/`to_builtin_deep` 存活为非 None 非 Mapping 标量）今天被 fallback 静默吞成 default——改 raise 是把静默降级变为响亮失败，属根因修复。

---

### 1.15 `_apply_value_policy` — duplicate-impl（risk=low）｜**STILL_VALID**

- 位置（复核 2026-07-24，微移）：`vrl/trajectory/storage.py:97`（def）、`:108-118`（dict/list/tuple 手滚递归）。
  - 复核：原审计 `98-115` → 现 `97-118`（约 1 行漂移）。`map_tensor_tree` 仍存在于 `vrl/trajectory/device.py:17` 且 `storage.py` 未 import 它；容器 walk 仍逐行等于 `map_tensor_tree` 的 walk 减去 dataclass 分支，duplicate-impl **仍成立**。
- 判死证据：body diff：`_apply_value_policy` 的 dict/list/tuple 递归逐行等于 `map_tensor_tree` 的容器 walk 减去 dataclass 分支；leaf 分支是纯 `leaf_fn`。`device.py` 自身 docstring 陈述该 walker 存在是因为「codebase 曾带四份手滚递归、已在容器覆盖上分叉」——这是同 package 内的又一份幸存手滚副本。`_tensor_bytes`（`storage.py` 尾部）**不在此列**——它是带 seen-set 与 `TrajectoryBatch` 处理的 reduction，rebuilder 型的 `map_tensor_tree` 无法表达。
- 动作：用共享 walker 重写：`map_tensor_tree(value, leaf_fn, is_leaf=_is_torch_tensor)`（`from vrl/trajectory/device.py`），只保留 tensor leaf op（device/dtype kwargs）本地化。`device.py` 只依赖 stdlib，`storage.py` 的 `torch` import 仍 lazy 在 `_is_torch_tensor` 内，无 import-cycle 障碍。
- 注意（low，form-4 caller-level 陷阱）：commit `327446da` 曾以「trajectory/storage walkers 有意不折进 walker」为由把它排除——但该理由对 `_tensor_bytes`（seen-set reduction）成立、对 `_apply_value_policy`（纯 map）**事实错误**，正是 AGENTS.md 审计教义警告的「按 caller 判 can't-fold」错误裁定。折入带来的唯一行为变化（递归进 dataclass）正是 walker 存在要修的覆盖缺口；无测试钉 dataclass pass-through，故安全。

## 2. 已由 origin 落地（本次复核确认，无需再做）

以下 6 条在 `88ed756e..7c748532` 间已被 origin 自行清掉，复核（grep 全仓 + `git log -S`）确认生产 reader/producer 均已消失，无需再做：

- **§2.1 `RenoiseStepResult.std_dev`（+ test-only `next_sample_mean`）** — 死字段，零 reader；`RenoiseStepResult` 现只剩 `next_sample + log_prob`（`renoise.py:20-21`），`next_sample_mean` 全仓无命中 — landed `6ed86b60 refactor(denoise): remove test-only result state`。
- **§2.2 `context['trajectory_mode']`**（原 §1.5）— 死数据键，仅测试断言读；两个 producer 与测试断言均已移除 — landed `7f3b8d61 refactor(trajectory): remove derivable context mirrors`。
- **§2.3 `TrajectoryMetrics.values` 结构计数键 `num_steps/num_tokens/num_temporal_chunks/num_denoise_transitions/num_segments`**（原 §1.7）— 与 `axis_lengths` 重复的死数据键；`values={...}` 结构载荷已移除（`values=` 在 builders.py 无命中，`TrajectoryMetrics.values` 字段本身按预期保留）— landed `c6ef0027 refactor(trajectory): derive structural metrics`。
- **§2.4 `build_ar_multisegment_trajectory(reward_segments=)`**（原 §1.9）— 零 passer 死参数；参数、module-wrapper 参数、转发行与 `reward_segments or (primary_segment,)` fallback 全移除 — landed `2c3a5c76 fix(rollouts): score canonical generation outputs`。
- **§2.5 `context['segment_names']`**（原 §1.10）— 死数据键，唯一「消费者」pop-并-丢弃；`builders.py` producer 与 `batch_builder.py` 的 pop 均移除（残留 `segment_names` 命中只剩无关的 `ReplayRequest.segment_names` 字段测试）— landed `ca601d45 refactor(trajectory): make primary segment explicit`。
- **§2.6 `TrajectoryResolver.primary_trainable_segment_name(fallback=)`**（原 §1.13）— 零 passer 死参数；现签名为 `def primary_trainable_segment_name(self) -> str:`，`fallback` 参数与 `if fallback is not None` 分支移除，方法本身保留 — landed `6cebd279 refactor(trajectory): remove derived training views`。

## 3. 情况已变（需重新评估）

以下 2 条的死代码本体仍在，但原审计的删除前提/动作已被 origin 的其他改动打破，**不能原样执行**，需重新评估：

- **§3.1 `stack_trajectory_batches`（+ 私有助手 `_stack_values`/`_validate_stack_compatible`/`_segment_name_for_tensor`、`__all__` 条目）**（原 §1.11）｜verdict=CHANGED
  - 现状：`vrl/trajectory/ops.py:51`（def）、`:184/:223/:246`（三个私有助手）、`:256`（`__all__`），**生产仍零 caller**。
  - 变化：origin 在审计后**新增了测试 caller**——`c6ef0027` 创建 `tests/trajectory/test_ops.py`，在 `:44`（`test_stack_derives_combined_sample_structure`）、`:57`（`test_stack_rejects_different_non_sample_axis_lengths`）调用它，`tests/trajectory/test_builders.py:214` 也调它（`git grep stack_trajectory_batches 88ed756e -- tests/` 在 base 为空，确认这些测试是新增）。
  - 为何原动作失效：原审计判据是「zero test callers / no tests to delete」，据此「删函数且无测试清理」。现在删除需连带删掉这些新回归测试——origin 明确选择了**保留并回归测试**该函数，而非删除。需重新评估：是尊重 origin 的既定选择保留，还是连测试一起删（并说明为何推翻 origin 的保留决定）。

- **§3.2 `TrajectoryAxis.metadata`**（原 §1.12）｜verdict=CHANGED
  - 现状：字段仍在 `vrl/trajectory/types.py:56`，**仍无 producer**（13 处 `TrajectoryAxis(...)` 构造全为位置参数 `name/kind/length`，无一传 `metadata=`；tests 零构造）。
  - 变化：本条删除原与 §1.11 捆绑，其 keep/kill 前提是「唯一行为 reader `axis != other`（`_validate_stack_compatible` 的 frozen `__eq__`）仅从零-caller `stack_trajectory_batches` 可达」。该前提已失效：`stack_trajectory_batches` 现有测试 caller（见 §3.1），frozen `__eq__` 全字段比较在 `ops.py:238` 被这些新测试驱动。
  - 为何原动作失效：捆绑动作（删函数 + 删字段）无法干净执行——需先删新测试才能删函数。即便 `TrajectoryAxis.metadata` 字段本身仍无 producer，其唯一行为 reader 已被新测试激活，情况实质改变。须与 §3.1 一并重新评估（若决定保留 `stack_trajectory_batches`，则 `metadata` 字段的 `__eq__` reader 就是活的，字段应保留）。

## 4. 验证协议

- 每条删除后：`ruff check <touched files>` + `ruff format --check <touched files>`（仅本条触及的文件，先 `ruff check --fix` 再 `ruff format`，末尾复验）。
- 全簇（§1 的 7 条）完成后：`pytest tests/math/ tests/trajectory/ tests/rollouts/ tests/models/ tests/generation/bindings/`；再 `pytest -m "not e2e and not slow_test"` 子集不新增失败。
- 基线：测试现对 **main @ `7c748532`** 跑（原 `88ed756e` 基线已过时）。删除前先在该 tip 上跑一遍取当前 fast-subset pass/fail 快照，删除后须保持不新增失败；`vrl.config.lint` 与 `ruff check .` 全绿须保持。
- 逐条触及的测试文件（供定向复跑）：
  - §1.2：`tests/math/denoise/test_renoise.py`（改 `test_renoise_rejects_deterministic_or_reverse_sigma` 传 `noise=` + 新增 required-arg 断言）
  - §1.3：`tests/math/test_token_flow_matching.py`（无改，仅确认 `saved_noise=x0` 仍通过）
  - §1.4：`tests/math/test_token_flow_matching.py`（改写 line 63/90/109 为 fake head）
  - §1.6：`tests/models/families/janus_pro/test_r1_model.py`、`tests/rollouts/replay/test_multisegment_token_logprob.py`（改经 `train_segments` 驱动）；`tests/rollouts/runtime/test_janus_pro_r1_wiring.py`（仅复跑）
  - §1.14：`tests/trajectory/test_storage_policy.py`（可选新增 TypeError 断言）
  - §1.8 / §1.15：无测试改动（零测试引用）

## 5. Non-Goals

- 不删被「能 raise 的校验」、控制流分支、runtime/config/Ray 调用消费的字段（AGENTS 死字段规则）。明确保留：`trajectory_storage_policy_from_cfg` 的 `None→default` 与 `Mapping→parse` 分支（§1.14）、`_tensor_bytes`（seen-set reduction，非 map，§1.15）、`SDEStepResult.prev_sample_mean/std_dev_t`（被 `compute_kl_divergence` 消费，与已落地的 renoise `std_dev` 不对称）。
- 不动 DO-NOT-FLAG 豁免项（`_GENERATION_CUDA_RUNTIME_RESIDUAL_BYTES`、`_REWARD_CUDA_RUNTIME_RESIDUAL_BYTES`、`ensure_loaded`、`process_gpu_used_bytes` NVML、SANA/HunYuan `prepare_latents` 修复）——本簇任何 finding 均未涉及它们。
- 不为省行数扁平化 protocol/lazy-import/跨家族一致性 thin function。
- 不擅自推翻 origin 对 §3.1 `stack_trajectory_batches` 的**保留决定**——若要删，须给出高于「零生产 caller」的理由并连带处理新回归测试；§3.2 `TrajectoryAxis.metadata` 的去留跟随 §3.1 结论。
- cluster-specific：
  - **不用宽松 grep 判 trajectory 同名字段死活**——`metadata`/`std_dev` 等在多个 struct 复用；每条删除前必须逐 receiver 消歧（用带引号 `grep -rnF`、词边界 `\b`、读体确认命中类型），沿 [[SPRINT_trajectory_views_types_dead_fields_cleanup]] 的教训。
  - **不裸删 §1.14 的 fallback 分支**——须替换为显式 `raise TypeError`，保持函数返回契约。
  - **与 in-flight sprint 无冲突**：§1 全部文件位于 `vrl/math/` 与 `vrl/trajectory/`，均不在 `SPRINT_native_generation_engine_program.md` 的 uncommitted worktree 集合（`generation/ray/`、`vrl/ray/`、`models/steps/denoise/base.py`）内。

## References

- 原始逐条证据/动作 JSON：`scratchpad/cluster_trajectory_math.json`
- §1.2：`vrl/math/denoise/renoise.py:30,48-51,71-78`、`tests/math/denoise/test_renoise.py`
- §1.3 / §1.4：`vrl/math/token/flow_matching.py:37,58-59,112,207,213,238-245`、`tests/math/test_token_flow_matching.py:63,90,109,132`
- §1.6：`vrl/trajectory/builders.py:807-820`、`vrl/models/families/janus_pro/runtime.py`、`vrl/algorithms/grpo/multisegment.py:31`、`vrl/rollouts/evaluators/token/multi_segment_token_logprob.py:123`、`tests/models/families/janus_pro/test_r1_model.py`、`tests/rollouts/replay/test_multisegment_token_logprob.py`、`tests/rollouts/runtime/test_janus_pro_r1_wiring.py`
- §1.8：`vrl/trajectory/builders.py:294-295,363`
- §1.14 / §1.15：`vrl/trajectory/storage.py:48-49,55-57,97,108-118`、`vrl/trajectory/device.py:17`、`vrl/generation/bindings/full_sequence_denoise/executor.py`、`vrl/rollouts/collector/core.py`、`vrl/config/schema.py`、`tests/trajectory/test_storage_policy.py`
- §3.1 / §3.2：`vrl/trajectory/ops.py:51,184,223,246,256`、`vrl/trajectory/types.py:56`、`tests/trajectory/test_ops.py:44,57`、`tests/trajectory/test_builders.py:214`（origin 新增回归测试，`c6ef0027`）
- 已落地 commit：`6ed86b60`（§2.1）、`7f3b8d61`（§2.2）、`c6ef0027`（§2.3 + §3 新测试）、`2c3a5c76`（§2.4）、`ca601d45`（§2.5）、`6cebd279`（§2.6）
- 关联 sprint：[[SPRINT_deadcode_00_overview]]、[[SPRINT_trajectory_views_types_dead_fields_cleanup]]、[[SPRINT_grab_bag_file_audit]]、`docs/sprints/info/SPRINT_rollout_performance.md:1059`
