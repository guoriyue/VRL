# SPRINT: `trajectory` 与 `math` 死字段/死参数/死分支清理（planned）

状态：**planned（2026-07-23）**。15 条对抗验证确认的死代码（6 条 medium-risk 数值/共享结构区，9 条 low-risk 数据键/死函数），来自 dead-code-audit workflow。
来源：dead-code-audit workflow（五种死代码形态 + 对抗验证 + 删除类二次字符串引用检查）。所有 15 条 KEY grep 已在 2026-07-23 复核，全部仍成立，无复核偏差。
关联：[[SPRINT_deadcode_00_overview]]、[[SPRINT_trajectory_views_types_dead_fields_cleanup]]（本簇沿用其「逐 receiver 消歧」纪律，并复用其 §1.3/§1.4 关于 `_reject_runtime_state` 是合法 consumer 的裁定）、[[SPRINT_grab_bag_file_audit]]（`_flow_terminal_mean` / `_apply_value_policy` / `_primary_segment_name` 的历史抽取记录）。

## 0. 一句话

本簇清理 `vrl/math/`（renoise、flow_matching）与 `vrl/trajectory/`（builders、ops、resolver、storage、types）两处的死字段、死参数、死分支与一个死函数。主形态是 **dead data key / dead-arg**：多个 builder 往 `TrajectoryMetrics.values` 与 `segment.metadata` / `context` 里写入从无 reader 的键，以及三个从无 passer 的采样参数（`generator=` / `velocity_fn=` / `reward_segments=`）。最锋利的一条是 `flow_logprob_at` 的 `saved_noise=None` fallback 分支（form 2）——它自述为「Biased Monte-Carlo」模式，而所有 producer 都强制供给 replay 张量，没有任何调用能选中它。**首要误删风险**：trajectory 各 struct 大量复用同名字段（`metadata` / `advantage_scope` / `segment_names` / `std_dev`），[[SPRINT_trajectory_views_types_dead_fields_cleanup]] 已在此栽过跟头——本簇每条动作都必须先做 **逐 receiver 消歧**，确认命中的是目标类型而非同名兄弟字段，再落删。

## 1. 待删清单（逐条，带证据与动作）

顺序：medium-risk（1.1–1.6，reviewer 重点）在前，low-risk（1.7–1.15）在后；同文件相邻条目相邻排列。

---

### 1.1 `RenoiseStepResult.std_dev`（+ test-only `next_sample_mean`）— dead-field（risk=medium）

- 位置：`vrl/math/denoise/renoise.py:16-23, 105-110`；测试 `tests/math/denoise/test_renoise.py:31`
- 判死证据：
  ```
  $ grep -rnE '\.std_dev\b' vrl tests | grep -v std_dev_t
  (无匹配 — std_dev 零属性读取，全仓 ~60 处 std_dev 命中都是 SDE/DDIM/flow 家族的另一字段 std_dev_t)
  $ grep -rn 'next_sample_mean' vrl tests
  vrl/math/denoise/renoise.py:22 (def) / :108 (构造)
  tests/math/denoise/test_renoise.py:31: assert torch.equal(rollout.next_sample_mean, expected_mean)
  ```
  两个生产 caller 只消费 `.next_sample` / `.log_prob`（`causvid/runner.py:276-284`、`causvid/model.py:544-549`）。renoise 无 KL 路径（`ChunkAutoregressiveDenoiseLogProbEvaluator.evaluate` 只传 `log_prob/old_log_prob/mask/ref_log_prob`，从不传 mean/std），故与 `SDEStepResult.prev_sample_mean/std_dev_t`（被 `compute_kl_divergence` 消费）**不对称**。
- 动作：删 `std_dev`（全仓零 reader）。删 `next_sample_mean` 及其唯一 reader `tests/math/denoise/test_renoise.py:31` 的断言——该断言是 test-only reader，按死字段规则即死；同测试的 log_prob 密度断言（`test_renoise.py:25-32`）已通过密度同时钉住 mean+std，删除后仍完整覆盖。
- 注意（medium）：**逐 receiver 消歧**是本条前提。`std_dev` 与 SDE 家族的 `std_dev_t` 只差一个后缀，`next_sample_mean` 与 `prev_sample_mean` 也仅一词之差——绝不能用宽松 `grep std_dev` 判死。必须用 `\.std_dev\b`（词边界，排除 `std_dev_t`）确认 `RenoiseStepResult` 的 `std_dev` 零读取。SDE 家族的同名兄弟字段是活的，不在本簇删除范围。

---

### 1.2 `renoise_step_with_logprob(generator=)` 与 fresh-noise randn fallback — dead-arg（risk=medium）

- 位置：`vrl/math/denoise/renoise.py:32, 50-53, 74-80`；测试 `tests/math/denoise/test_renoise.py`
- 判死证据：
  ```
  $ grep -rn 'renoise_step_with_logprob' vrl tests
  causvid/runner.py:276 (noise=transition_noise, math_dtype=)
  causvid/model.py:544  (next_sample=actions[...], math_dtype=torch.float32)
  tests/math/denoise/test_renoise.py:18/19/40/51
  -> 零 generator= passer；无 **kwargs 转发
  ```
  runner.py:276-285 只传 `noise=` / `math_dtype=`（已读体确认，见下）。rollout 需 per-sample 播种，`_transition_noise`（`runner.py:370-386`）为每个 sample 各建一个 `torch.Generator` 并 `torch.randn(generator=...)`——单个 `generator=` 参数无法表达，这正是 `noise=` 路径存在的原因。与 `sde_step_with_logprob`（其 `generator=` 确被 `vrl/generation/steps/denoise/loop.py:232` 传入）不同，签名对齐不构成保留理由。fallback：唯一裸调用 `test_renoise.py:51` 用 `sigma<=0`，在 `renoise.py:68-69` 的 sigma 校验处即 raise，永不抵达 line 73 的 randn 分支——生产与测试都零抵达。
- 动作：(a) 删 `generator` 参数（line 32）；(b) 将 line 50-53 校验块改为两个显式检查：`noise` 与 `next_sample` 同时给出时 `raise ValueError("next_sample cannot be combined with noise")`，两者皆 None 时 `raise ValueError("exactly one of noise or next_sample is required")`；(c) 删 line 74-80 的 `torch.randn` fallback，只保留 noise-supplied 分支（device/dtype 迁移 + shape 校验）；(d) 更新 docstring，去掉 `generator` 提及，说明采样恒消费 caller 提供的 noise。测试：更新 `test_renoise.py:51` 的 `test_renoise_rejects_deterministic_or_reverse_sigma`，传 `noise=torch.zeros(1, 2)`，使 required-arg 检查不抢先于它要断言的 sigma 校验；新增一条断言「有效 sigma 但缺 noise/next_sample 时抛新的 required-arg ValueError」。
- 注意（medium）：这是数值路径改动。新增的 required-arg 校验会改变 `test_renoise_rejects_deterministic_or_reverse_sigma` 的错误抛出顺序——该测试原本裸调用（既无 noise 也无 next_sample）期望的是「sigma must be > 0」，若 required-arg 检查置前会抢先抛错。故测试改写与参数删除必须同一条动作内完成，不可分离。

---

### 1.3 `flow_logprob_at` 的 `saved_noise=None` fallback 分支 — dead-branch（risk=medium）

- 位置：`vrl/math/token/flow_matching.py:207, 238-245`（docstring 陈旧策略 220-221 一并清）
- 判死证据：
  ```
  $ grep -rn 'saved_noise' vrl tests --include='*.py' | grep -v flow_matching.py
  nextstep_1/model.py:215  saved_noise: torch.Tensor   (必填参数, 无默认)
  nextstep_1/model.py:248  saved_noise=saved_noise[:, j]  (无条件切片)
  nextstep_1/model.py:300  saved_noise = replay["saved_noise"]  (方括号取值, 缺失即 KeyError, 不会是 None)
  builders.py:525/565-567/602  (必填 kwarg -> ReplayInput tensor_ref)
  runtime.py:71/287  saved_noise: torch.Tensor  (非 Optional dataclass 字段)
  runner.py:84/203  torch.zeros(...) 逐位置填充
  ```
  唯一生产 caller `nextstep_1/model.py:248` 无条件传 `saved_noise=saved_noise[:, j]`；`build_ar_continuous_trajectory` 把 `saved_noise` 作必填 replay 张量记录。没有任何 producer 能供 None——form 2 dead-semantics 确认。分支自述为「Fallback mode: fresh x_0 ~ N(0, I)」偏置 Monte-Carlo，无人选中。
- 动作：把 `saved_noise` 改为必填 `torch.Tensor`，删 238-245 的 fallback 分支，并裁掉 docstring 中已陈旧的策略「(b) accept a Monte-Carlo approximation by re-running with a fresh prior」（lines 220-221）。唯一 test caller `tests/math/test_token_flow_matching.py:132` 传 `saved_noise=x0`，不触及 fallback，故无测试清理需求；`saved_noise` 位于 `*` 之前，关键字 caller 不受影响。
- 注意（medium）：数值语义分支。`info/SPRINT_rollout_performance.md:1059` 明确记录 `saved_noise` 重放（collection 点 importance ratio = 1）为设计意图，与 fresh-noise fallback 相反——删分支是与既定设计对齐，而非改变数值契约。

---

### 1.4 `velocity_fn`（`flow_sample_with_logprob` / `flow_logprob_at` / `_flow_terminal_mean`）— dead-arg（risk=medium）

- 位置：`vrl/math/token/flow_matching.py:37, 112, 213, 57-60`（连带 docstring 50/145 与内部转发 186/254）
- 判死证据：
  ```
  $ grep -rn 'velocity_fn' vrl tests
  vrl/math/token/flow_matching.py:37/50/58/59/112/145/186/213/254 (定义文件)
  tests/math/test_token_flow_matching.py:63/90/109 (三处全传给 private _flow_terminal_mean, image_head=None)
  -> 两个 public 函数的 velocity_fn 参数零 passer（含测试）
  ```
  生产 caller `nextstep_1/runner.py:191-200` 与 `model.py:244-253` 显式传每个其他 kwarg 却从不传 `velocity_fn`，且无 **kwargs 转发。唯一行为用途是 `flow_matching.py:58-59` 的 override 分支，其提供的 seam 已被既有 duck-typed `image_head.net(x,t,c)` 路径完全复制（同测试文件的 `_FakeHead` 已驱动两个 public 函数而不用 velocity_fn）。镜像的 `sde_logprob.py` 无同类 callable-override，故无跨家族一致性保留理由。
- 动作：从三处签名移除 `velocity_fn`，删 `_flow_terminal_mean._velocity` 里的 `if velocity_fn is not None` 分支；改写三处 test call site（`tests/math/test_token_flow_matching.py:63,90,109`）为传一个 duck-typed fake head，暴露 `.net(x,t,c)`（`_flow_terminal_mean` 的 fake 不必带 `.input_dim`，只有 `flow_sample_with_logprob` 读它）。连带清理 docstring 50/145 与内部转发 kwarg 186/254。
- 注意（medium）：这三处是 test-only passer（form 1 的 test-only 变体）——public 函数零 passer，private `_flow_terminal_mean` 仅测试传入。改写测试须保持 CFG override 覆盖等价：extend 既有 `_FakeHead` 暴露 `.net` 即可复现同一 seam。

---

### 1.5 `context['trajectory_mode']` — dead-field（risk=medium）

- 位置：`vrl/trajectory/builders.py:329, 408`；测试 `tests/generation/bindings/chunk_autoregressive_denoise/test_binding.py:101`
- 判死证据：
  ```
  $ grep -rn 'trajectory_mode' vrl tests
  vrl/trajectory/builders.py:329: "trajectory_mode": "trainable_chunk_denoise"
  vrl/trajectory/builders.py:408: "trajectory_mode": "generation_only"
  tests/generation/bindings/chunk_autoregressive_denoise/test_binding.py:101: assert trajectory.context["trajectory_mode"] == "generation_only"
  ```
  两个 producer，唯一 reader 是测试断言——test-only reader，按死字段规则即死。两种 record 形态的下游判别用 `segment.trainable`（`batch_builder.py:87-95` 的 `_trainable_segments()` 扫描），而非此键。context 只经泛型 copy（`ops.py` select/concat/move、gather 的 `dict(ordered[0].context)`）与 `validation.py._reject_runtime_state`（只做 value 类型校验，不读键）流动；trajectory context 从不被 `torch.save/json.dump/pickle` 持久化。
- 动作：从两个 chunk builder 删该键，并删 `test_binding.py:101` 的断言（该测试真正的判别断言 `segment.trainable is False`（line 98）保留，测试仍自洽）——或在两个写入点注解为 display/provenance-only。
- 注意（medium）：其余四个 builder（`147/517/654/810`）都不设 `trajectory_mode`，故这是一次性写入而非跨家族统一 shape，不构成一致性保留理由。逐 receiver 消歧：`generation_only` / `trainable_chunk_denoise` 字面值在 `magi_1/model.py` 另有一个无关的 `runtime-caps` 键 `"generation_only": True`，勿误判为同一消费链。

---

### 1.6 `_TrajectoryBatchBuilder._segment_trainable` 的死分支 — dead-branch（risk=medium）

- 位置：`vrl/trajectory/builders.py:928-942`（body 6 分支）
- 判死证据：
  ```
  $ grep -rn '_segment_trainable' vrl tests
  builders.py:694 (caller) / builders.py:929 (def)  — 无 registry/string 引用
  $ grep -rn '"enabled"' vrl/models/families vrl/trajectory vrl/rollouts tests/models tests/rollouts
  builders.py:932-933 (死分支本身)
  multi_segment_token_logprob.py:126 (读的是另一个 replay-side payload, 由 _trajectory_segment_payload 构造, 只发 'train' 从不发 'enabled')
  -> 'enabled' payload 键零 producer
  $ grep -rnF '"train":' tests/models/families/janus_pro/test_r1_model.py tests/rollouts/replay/test_multisegment_token_logprob.py
  test_r1_model.py:136 / test_multisegment_token_logprob.py:65  ('train' payload 是 TEST-ONLY producer)
  ```
  唯一生产 payload builder `_cat_segment_extra`（`janus_pro/runtime.py:449-483`）的键为 `name/token_ids/token_log_probs/token_mask/prompt_embeds/attention_mask/prompt_attention_mask/visual/cfg`——无 `train`、无 `enabled`。`train_segments` 唯一 producer 链是 `algorithm.train_segments`（`dict[str,bool]` dataclass 字段 `multisegment.py:27`；YAML mapping `token_grpo_multisegment.yaml:13`）经 `_copy_value`/`to_builtin_deep`（`collector/config.py:48-53`），非 dict 值会在 `factory.py:315` `dict(algorithm_config.train_segments or {})` 启动时崩溃，故 str/list-set/trailing-bool 分支无活 producer。这正是历史 form-2 模式（task-variant normalizer 带 producer-less alias）。
- 动作：缩到有 producer 的分支——dict `train_segments` -> `value.get(name, False)`；`value` None -> `payload.get('visual', True)`。删：`'enabled'` payload-key 分支（零 producer）、`isinstance str` / `list-tuple-set-frozenset` / 尾部 `bool(value)` 分支。`'train'` payload-key 分支是 TEST-ONLY（`test_r1_model.py:136`、`test_multisegment_token_logprob.py:65`）——删它，并把这两个测试改为经 `request.sampling['train_segments']` 驱动 trainability（如 `{'initial_image': True, 'selfcheck_text': True, 'final_image': True}`，经保留的 dict 分支复现同一 trainable 值，含 `selfcheck_text=True`——若退到 visual-default 会被翻成 False）。第三个 test caller `tests/rollouts/runtime/test_janus_pro_r1_wiring.py:127` 的 payload 只含 `visual/cfg`、无 sampling `train_segments`，走保留的 None->visual 分支，无需改写。
- 注意（medium）：**逐 receiver 消歧关键**——`'enabled'` 在 `multi_segment_token_logprob.py:126` 有一个 reader，但那是 replay-side payload（由 `_trajectory_segment_payload` 从 `segment.metadata` 构造），与 `_segment_trainable` 的 input payload 是两条不同的数据链；且 `builders.py:752-756` 把 `metadata["train"] = trainable`（计算输出）写回，不构成反馈到 `_segment_trainable` 输入的回路。判死前须读体确认两个 `'enabled'` 是不同 payload。

---

### 1.7 `TrajectoryMetrics.values` 键 `num_steps/num_tokens/num_temporal_chunks/num_denoise_transitions/num_segments` — dead-field（risk=low）

- 位置：`vrl/trajectory/builders.py:145, 322-325, 404, 515, 651, 807`
- 判死证据：
  ```
  $ for k in num_tokens num_temporal_chunks num_denoise_transitions num_segments; do grep -rnF "\"$k\"" vrl tests | grep -v builders.py; done
  (四键全部只在 builders.py 命中, 外部零命中)
  $ grep -rnF '"num_steps"' vrl tests | grep -v builders.py
  vrl/scripts/eval/*, vrl/scripts/generation/*, vrl/scripts/families/cosmos/* — 全是无关的 sampling.num_steps config 键
  ```
  `metrics.values` 的所有 reader 都是泛型：`validation.py:102` `FORBIDDEN_TRAJECTORY_METRICS` 交集（denylist = `{queue_wait_s, execution_s, peak_memory_mb, chunks}`，五键都不在内）、`validation.py:121` `_reject_runtime_state`、`ops.py` select/stack/move 转发、`storage.py:73/131`。读体确认字面重复：`num_steps==axis_lengths["denoise"]`、`num_tokens==axis_lengths["token"]`、`num_segments==len(segments)` 等。
- 动作：从全部六个 builder 删 `values={...}` 载荷。**保留 `TrajectoryMetrics.values` 字段本身**——其 `FORBIDDEN_TRAJECTORY_METRICS` denylist 校验（`validation.py:102-107`）是能 raise 的活不变量，按 keep-list 保留。无测试断言这些键（`tests/trajectory/test_builders.py` 零 `metrics`/`values` 命中），无测试清理。
- 注意（low，与既有裁定和解）：保留 `values` 字段而只删载荷，沿用 [[SPRINT_trajectory_views_types_dead_fields_cleanup]] §1.3 的规则——「能 raise 的校验是合法 consumer」。`num_steps` 与全仓大量 `sampling.num_steps` config 键同名，须用带引号 `grep -rnF '"num_steps"'` 消歧，确认外部命中都是 `RolloutCollectorConfig`（`collector/config.py:60` 从 `cfg.sampling` 构造）而非 `TrajectoryMetrics`。

---

### 1.8 `segment.metadata['temporal_chunk_axis']` / `['transition_axis']` — dead-field（risk=low）

- 位置：`vrl/trajectory/builders.py:302-305, 388`
- 判死证据：
  ```
  $ grep -rn 'temporal_chunk_axis\|transition_axis' vrl tests
  vrl/trajectory/builders.py:303/304/388  (仅构造点)
  ```
  全仓外部零命中。`TrajectorySegment.metadata` 的泛型 consumer（`validation.py:264` `_reject_runtime_state`、`ops.py:154` 盲 dict copy）content-agnostic；唯一按键读 segment metadata 的是 token evaluator 的 `'visual'/'cfg'/'train'`（`multi_segment_token_logprob.py:168-170`，不同 segment 不同键）。chunk replay 路径（`chunk_autoregressive_logprob.py`、`causvid/model.py`）零 `.metadata` 读。键重复编码了 `trajectory.axes` 已首类声明的轴名（`temporal_chunk`/`denoise_transition`）——duplicate-construction 数据形态。
- 动作：删 `build_chunk_autoregressive_denoise_trajectory`（302-305）与 `build_chunk_autoregressive_generation_trajectory`（388）里的 `metadata={'temporal_chunk_axis': ..., 'transition_axis': ...}` 项。无测试断言这些键，无测试清理。
- 注意（low）：逐 receiver 消歧——`.metadata` 在 trajectory 层被多个 struct 复用，须确认命中的是这两个 chunk builder 的 `TrajectorySegment.metadata` 写入，而非同名的 `TrajectoryTensor.metadata`/`ReplayInput.metadata`。

---

### 1.9 `build_ar_multisegment_trajectory(reward_segments=)` — dead-arg（risk=low）

- 位置：`vrl/trajectory/builders.py:664, 769, 1110, 1122`
- 判死证据：
  ```
  $ grep -rn 'reward_segments' vrl tests --include='*.py'
  builders.py:664  (方法签名参数)
  builders.py:769  reward_segment_names = reward_segments or (primary_segment,)
  builders.py:1110 (模块级 wrapper 签名参数)
  builders.py:1122 reward_segments=reward_segments  (wrapper 转发)
  -> 零外部 passer
  ```
  生产 caller `janus_pro/runtime.py:396` 与三个 test caller 都不传，只有 default 分支 `(primary_segment,)` 可达。
- 动作：(1) 删 line 664 方法签名与 line 1110 wrapper 签名的 `reward_segments: tuple[str, ...] | None = None,`；(2) 删 line 1122 wrapper 转发 `reward_segments=reward_segments,`（**不可漏**——只删签名不删转发会 NameError）；(3) 将 line 769 `reward_segment_names = reward_segments or (primary_segment,)` 改为 `reward_segment_names = (primary_segment,)`。无测试传该参数，无测试清理；跑 `tests/models/families/janus_pro/test_r1_model.py`、`tests/rollouts/replay/test_multisegment_token_logprob.py`、`tests/rollouts/runtime/test_janus_pro_r1_wiring.py` 确认。
- 注意（low，与规划中 sprint 和解）：`planned/SPRINT_janus_r1_agentic_credit_assignment.md:39` 点名此函数，但它规划一个**全新**的显式类型化 initial/refined/selected reward-view 契约（Gate 2 缺键 fail-fast），并明言「不能假设旧 reward_view config 仍存在」——与当前 `reward_segments` 的静默过滤（`if name in decoded_tensors`）相悖。该 sprint 替换而非复用此边界，故删 `reward_segments` 不与之冲突。

---

### 1.10 `context['segment_names']` — dead-field（risk=low）

- 位置：`vrl/trajectory/builders.py:812`；消费点 `vrl/rollouts/collector/batch_builder.py:205`
- 判死证据：
  ```
  $ grep -rnF '"segment_names"' vrl tests
  vrl/trajectory/builders.py:812:  "segment_names": tuple(segments)   (producer)
  vrl/rollouts/collector/batch_builder.py:205: rollout_context.pop("segment_names", None)   (pop-并-丢弃)
  tests/models/interfaces/test_replay_model_contract.py:109: pytest.raises(ValueError, match="segment_names")  (匹配的是 ReplayRequest.segment_names, 另一符号)
  ```
  唯一「消费者」是 `batch_builder.py:205` 的 pop-并-丢弃；下游 context 改用重算的 `"r1_segment_names"`（从 `trajectory.segments` 派生，`batch_builder.py:219-223`）。测试命中的 `segment_names` 是 `ReplayRequest.segment_names` dataclass 字段（`vrl/models/interfaces/replay.py:15`），已读体确认是不同符号。「唯一消费者把它删掉」的键即死。
- 动作：删 producer（`builders.py:812`）与 `batch_builder.py:205` 那句已无意义的 `rollout_context.pop("segment_names", None)`。
- 注意（low）：兄弟键控制——`primary_segment`（`batch_builder.py:305`、`multi_segment_token_logprob.py:221`）与 `segment_order`（`multisegment.py:63`）有真 reader，故 `builders.py:811` 与 `batch_builder.py:204` 的 pop 必须**保留**，只删 `segment_names` 那一对。逐 receiver 消歧：`ReplayRequest.segment_names` 是完全无关的字段，勿误删。

---

### 1.11 `stack_trajectory_batches` — dead-function（risk=low）

- 位置：`vrl/trajectory/ops.py:51-91`（连带私有助手 `_stack_values` 196-221、`_validate_stack_compatible` 235-253、`_segment_name_for_tensor` 256-260、`__all__` line 266）
- 判死证据：
  ```
  $ grep -rn 'stack_trajectory_batches' vrl tests docs pyproject.toml
  vrl/trajectory/ops.py:51  (def)
  vrl/trajectory/ops.py:266 (__all__)
  docs/sprints/parked/SPRINT_agentic_image_episode_runtime.md:155  ("不调用 stack_trajectory_batches()")
  ```
  vrl/ 与 tests/ 零 caller；唯一非定义引用是一句 parked 文档的「不要调用」注记。`vrl/trajectory/__init__.py` 不再导出 ops，唯一 importer `vrl/rollouts/batch/ops.py:11` 只导 `select_trajectory_batch`/`move_trajectory_batch`。三个私有助手仅被 `stack_trajectory_batches` 调用（`_stack_values` 含自递归）。
- 动作：删 `stack_trajectory_batches` 及其私有助手 `_stack_values`、`_validate_stack_compatible`、`_segment_name_for_tensor`，从 `__all__`（line 266）移除该条目，可选地从模块 docstring（3-5）去掉「stacking」一词。**保留** `_rebuild_trajectory`/`_select_value`/`_selector_positions`——它们服务活的 `select_trajectory_batch`/`move_trajectory_batch`（被 `vrl/rollouts/batch/ops.py:11` 导入）。无测试 caller，无测试清理。更新 `docs/sprints/parked/SPRINT_agentic_image_episode_runtime.md:155` 的陈旧引用以反映函数已不存在。
- 注意（low）：与 §1.12 强耦合——`stack_trajectory_batches` 是 `TrajectoryAxis.metadata` 唯一行为 reader（`_validate_stack_compatible` 的 `axis != other` frozen `__eq__`）的所在。**§1.11 必须先于或同步于 §1.12 执行**，否则 §1.12 的判死前提（metadata 唯一 reader 已删）不成立。

---

### 1.12 `TrajectoryAxis.metadata` — dead-field（随 §1.11 form-1 删除而落）（risk=low）

- 位置：`vrl/trajectory/types.py:57`
- 判死证据：
  ```
  $ grep -rn 'TrajectoryAxis(' vrl tests
  13 处构造, 全在 vrl/trajectory/builders.py, 全为位置参数 name/kind/length, 无一传 metadata=; tests 零构造
  $ grep -n 'metadata' vrl/trajectory/types.py
  57 (TrajectoryAxis) / 74 / 89 / 109  (四个不同 struct 各有 metadata 字段)
  ```
  零 producer（无构造传 `metadata=`）+ 唯一行为 reader 是 `ops.py:246` 的 `axis != other` frozen `__eq__`（仅从零-caller `stack_trajectory_batches` 可达，随 §1.11 消亡）。[[SPRINT_trajectory_views_types_dead_fields_cleanup]] §1.3（该 done 文档 line 40）曾以「`ops.py:246` frozen `__eq__` 全字段比较驱动控制流」为唯一保留理由——该理由随其函数一并失效。
- 动作：删 `vrl/trajectory/types.py:57` 的 `TrajectoryAxis.metadata` 字段。唯一 `fields()` 迭代是 `device.py:38` 的字段无关泛型重建，删字段不改其行为；无 `pickle`/`torch.save` 持久化，无 schema 版本风险。无测试构造 `TrajectoryAxis`，无测试清理。
- 注意（low）：**这是本簇最需 receiver 消歧的一条**。`types.py` 有四个 struct（57/74/89/109）都叫 `metadata`——只删 line 57 的 `TrajectoryAxis.metadata`，绝不能触及 `TrajectoryTensor`/`ReplayInput`/`TrajectorySegment` 的同名字段（后三者有真 producer 与 `_reject_runtime_state` 消费，是活的）。`builders.py` 内的 `metadata=` kwarg（139/302/388/509/646/752/801）都属这三个兄弟类型，与 `TrajectoryAxis` 无关。执行依赖 §1.11 先落。

---

### 1.13 `TrajectoryResolver.primary_trainable_segment_name(fallback=)` — dead-arg（risk=low）

- 位置：`vrl/trajectory/resolver.py:49-59`
- 判死证据：
  ```
  $ grep -rn 'primary_trainable_segment_name' vrl tests
  resolver.py:49  def primary_trainable_segment_name(self, fallback: str | None = None)
  resolver.py:101 name = segment_name or self.primary_trainable_segment_name()   (唯一 caller, 不传 fallback)
  ```
  全仓仅两处引用；`fallback` 参数与其 `if fallback is not None` 分支不可达。所有外部 `TrajectoryResolver` consumer 经 `replay_tensor_dict(segment_name)`/`tensor_value()`，`replay_tensor_dict` 在 `segment_name or ...` 处短路，永不抵达 fallback。
- 动作：删 `fallback` 参数及其 `if fallback is not None` 分支。**保留方法本身**——它是 caller 省略 `segment_name` 时的默认 resolver API（如 `vrl/algorithms/trajectory.py:76` 调 `replay_tensor_dict()` 不带 segment）。无测试 caller，无测试清理。
- 注意（low）：`done/SPRINT_grab_bag_file_audit.md:126` 明确裁定 `evaluators/trajectory._primary_segment_name(fallback)` 语义各异、不得折进 resolver——故本条删除不引入跨家族一致性损失（两者本就不统一）。

---

### 1.14 `trajectory_storage_policy_from_cfg` 的 isinstance pass-through 与 cfg_get fallback — dead-branch（risk=low）

- 位置：`vrl/trajectory/storage.py:48-49, 55-57`
- 判死证据：
  ```
  $ grep -rn 'trajectory_storage_policy_from_cfg' vrl tests
  两个生产 caller:
    vrl/generation/bindings/full_sequence_denoise/executor.py:349 (request.sampling.get('trajectory_storage') — Ray 序列化后的 plain dict)
    vrl/rollouts/collector/core.py:240 (cfg_get(...,'trajectory_storage',None) — YAML 经 _copy_value 的 values-dict 项)
  tests/trajectory/test_storage_policy.py:67/68/71 (仅 None/dict/error)
  ```
  `to_builtin`（`utils/config.py:73-88`）把任何 `DictConfig` 展开为 plain dict，Mapping 分支覆盖所有 OmegaConf 形态；`schema.py:310` 的 `ConfigBlock(TrajectoryStoragePolicy)` 只派生已知键名 frozenset 供 unknown-key lint，**从不实例化**该类型。故无路径能送入 policy 实例或非 mapping 属性对象——form 2 确认。
- 动作：删 `isinstance(value, TrajectoryStoragePolicy)` pass-through（48-49）；将尾部 cfg_get fallback（55-57）**替换为显式** `raise TypeError(f"rollout.trajectory_storage must be a mapping with 'device'/'dtype' keys, got {value!r}")`，使非 mapping YAML 值（如 `trajectory_storage: cpu`）响亮失败而非静默退化为 default no-op policy。保留 `None -> default` 与 `Mapping -> parse`。可选在 `tests/trajectory/test_storage_policy.py` 加一条断言标量输入抛 TypeError；无既有测试触及被删分支。
- 注意（low，动作非纯删）：**不可裸删两分支**——裸删会让函数落到末尾隐式 return None，违反 `-> TrajectoryStoragePolicy` 契约，并在 `executor.py:352` 下游以不透明 AttributeError 崩溃。尾部必须改为显式 error path。一个可达的 YAML 误配（`trajectory_storage: cpu` 裸字符串经 `_copy_value`/`to_builtin_deep` 存活为非 None 非 Mapping 标量）今天被 fallback 静默吞成 default——改 raise 是把静默降级变为响亮失败，属根因修复。

---

### 1.15 `_apply_value_policy` — duplicate-impl（risk=low）

- 位置：`vrl/trajectory/storage.py:98-115`
- 判死证据：
  ```
  $ grep -rn '_apply_value_policy' vrl tests
  仅 vrl/trajectory/storage.py (def 98; caller 69/72/73/89 全在文件内; 递归 110/112/114)
  $ grep -rn 'def map_tensor_tree' vrl
  vrl/trajectory/device.py:17
  ```
  body diff：`_apply_value_policy` 的 dict/list/tuple 递归（`storage.py:109-115`）逐行等于 `map_tensor_tree` 的容器 walk（`device.py:41-50`）减去 dataclass 分支；leaf 分支（99-108）是纯 `leaf_fn`。`device.py` 自身 docstring 陈述该 walker 存在是因为「codebase 曾带四份手滚递归、已在容器覆盖上分叉」——这是同 package 内的又一份幸存手滚副本。`_tensor_bytes`（118-145）**不在此列**——它是带 seen-set 与 `TrajectoryBatch` 处理的 reduction，rebuilder 型的 `map_tensor_tree` 无法表达。
- 动作：用共享 walker 重写：`map_tensor_tree(value, leaf_fn, is_leaf=_is_torch_tensor)`（`from vrl/trajectory/device.py`），只保留 tensor leaf op（device/dtype kwargs）本地化。`device.py` 只依赖 stdlib，`storage.py` 的 `torch` import 仍 lazy 在 `_is_torch_tensor` 内，无 import-cycle 障碍。
- 注意（low，form-4 caller-level 陷阱）：commit `327446da` 曾以「trajectory/storage walkers 有意不折进 walker」为由把它排除——但该理由对 `_tensor_bytes`（seen-set reduction）成立、对 `_apply_value_policy`（纯 map）**事实错误**，正是 AGENTS.md 审计教义警告的「按 caller 判 can't-fold」错误裁定。折入带来的唯一行为变化（递归进 dataclass）正是 walker 存在要修的覆盖缺口；无测试钉 dataclass pass-through（`tests/trajectory/test_storage_policy.py` 只走 plain tensor leaf），故安全。

## 2. 验证协议

- 每条删除后：`ruff check <touched files>` + `ruff format --check <touched files>`（仅本条触及的文件，先 `ruff check --fix` 再 `ruff format`，末尾复验）。
- 全簇完成后：`pytest tests/math/ tests/trajectory/ tests/rollouts/ tests/models/ tests/generation/bindings/`；再 `pytest -m "not e2e and not slow_test"` 子集不新增失败。
- 基线（清理前，2026-07-23）：fast subset 2620 passed / 7 pre-existing failures（架构边界 + causvid/magi_1 打包摘要，与本清理无关）；`vrl.config.lint` 与 `ruff check .` 全绿。删除后这三项须保持。
- 逐条触及的测试文件（从各条 action 提取，供定向复跑）：
  - §1.1 / §1.2：`tests/math/denoise/test_renoise.py`（§1.1 删 line 31 断言；§1.2 改 line 51 传 noise + 新增 required-arg 断言）
  - §1.3：`tests/math/test_token_flow_matching.py`（无改，仅确认 `saved_noise=x0` 仍通过）
  - §1.4：`tests/math/test_token_flow_matching.py`（改写 line 63/90/109 为 fake head）
  - §1.5：`tests/generation/bindings/chunk_autoregressive_denoise/test_binding.py`（删 line 101 断言）
  - §1.6：`tests/models/families/janus_pro/test_r1_model.py`、`tests/rollouts/replay/test_multisegment_token_logprob.py`（改经 `train_segments` 驱动）；`tests/rollouts/runtime/test_janus_pro_r1_wiring.py`（仅复跑）
  - §1.9：`tests/models/families/janus_pro/test_r1_model.py`、`tests/rollouts/replay/test_multisegment_token_logprob.py`、`tests/rollouts/runtime/test_janus_pro_r1_wiring.py`（仅复跑，不改）
  - §1.14：`tests/trajectory/test_storage_policy.py`（可选新增 TypeError 断言）
  - §1.7 / §1.8 / §1.10 / §1.11 / §1.12 / §1.13 / §1.15：无测试改动（零测试引用）

## 3. Non-Goals

- 不删被「能 raise 的校验」、控制流分支、runtime/config/Ray 调用消费的字段（AGENTS 死字段规则）。明确保留：`TrajectoryMetrics.values` 字段本身（`FORBIDDEN_TRAJECTORY_METRICS` 校验，§1.7）、`primary_trainable_segment_name` 方法本身（§1.13）、`trajectory_storage_policy_from_cfg` 的 `None->default` 与 `Mapping->parse` 分支（§1.14）、`_tensor_bytes`（seen-set reduction，非 map，§1.15）、`SDEStepResult.prev_sample_mean/std_dev_t`（被 `compute_kl_divergence` 消费，§1.1）、context 兄弟键 `primary_segment`/`segment_order` 及其 pop（§1.10）、`_rebuild_trajectory`/`_select_value`/`_selector_positions`（服务 select/move，§1.11）。
- 不动 DO-NOT-FLAG 豁免项（`_GENERATION_CUDA_RUNTIME_RESIDUAL_BYTES`、`_REWARD_CUDA_RUNTIME_RESIDUAL_BYTES`、`ensure_loaded`、`process_gpu_used_bytes` NVML、SANA/HunYuan `prepare_latents` 修复）——本簇任何 finding 均未涉及它们。
- 不为省行数扁平化 protocol/lazy-import/跨家族一致性 thin function。
- cluster-specific：
  - **不用宽松 grep 判 trajectory 同名字段死活**——`metadata`/`advantage_scope`/`segment_names`/`std_dev` 在多个 struct 复用；每条删除前必须逐 receiver 消歧（用带引号 `grep -rnF`、词边界 `\b`、读体确认命中类型），沿 [[SPRINT_trajectory_views_types_dead_fields_cleanup]] 的教训。
  - **不裸删 §1.14 的 fallback 分支**——须替换为显式 `raise TypeError`，保持函数返回契约。
  - **§1.12 不得先于 §1.11 执行**——`TrajectoryAxis.metadata` 的唯一 reader 在 `stack_trajectory_batches` 内，前提依赖后者已删。
  - **与 in-flight sprint 无冲突**：本簇全部文件位于 `vrl/math/` 与 `vrl/trajectory/`，均不在 `SPRINT_native_generation_engine_program.md` 的 uncommitted worktree 集合（`generation/ray/`、`vrl/ray/`、`models/steps/denoise/base.py`）内，无需 sequence-after 排期。

## References

- `vrl/math/denoise/renoise.py:16-23,32,50-53,74-80,105-110`（§1.1、§1.2）
- `tests/math/denoise/test_renoise.py:18-19,31,40,51`
- `vrl/math/token/flow_matching.py:37,50,57-60,112,145,186,207,213,238-245,254`（§1.3、§1.4）
- `tests/math/test_token_flow_matching.py:63,90,109,132`
- `vrl/trajectory/builders.py:145,302-305,322-325,329,388,404,408,515,651,664,741,769,807,812,928-942,1110,1122`（§1.5–§1.10、§1.12 producer）
- `vrl/trajectory/ops.py:51-91,152,196-221,235-260,266`（§1.11、§1.12 reader、§1.15 diff 对照）
- `vrl/trajectory/resolver.py:49-59,101`（§1.13）
- `vrl/trajectory/storage.py:48-49,55-57,98-115`（§1.14、§1.15）
- `vrl/trajectory/types.py:57`（§1.12）
- `vrl/trajectory/device.py:17-50`（§1.15 共享 walker）
- `vrl/trajectory/validation.py:102-107,121,264`、`vrl/rollouts/collector/batch_builder.py:87-95,204-223,305`、`vrl/models/families/janus_pro/runtime.py:396,449-483`、`vrl/models/families/nextstep_1/model.py:215,248,300`、`vrl/generation/bindings/full_sequence_denoise/executor.py:349,352`、`vrl/rollouts/collector/core.py:240`、`vrl/config/schema.py:310`
- `tests/generation/bindings/chunk_autoregressive_denoise/test_binding.py:98,101`、`tests/models/families/janus_pro/test_r1_model.py:136,158`、`tests/rollouts/replay/test_multisegment_token_logprob.py:65,87`、`tests/rollouts/runtime/test_janus_pro_r1_wiring.py:127`、`tests/trajectory/test_storage_policy.py:67-71`
- 关联 sprint：[[SPRINT_deadcode_00_overview]]、[[SPRINT_trajectory_views_types_dead_fields_cleanup]]、[[SPRINT_grab_bag_file_audit]]、`docs/sprints/planned/SPRINT_janus_r1_agentic_credit_assignment.md`、`docs/sprints/parked/SPRINT_agentic_image_episode_runtime.md:155`、`docs/sprints/info/SPRINT_rollout_performance.md:1059`
