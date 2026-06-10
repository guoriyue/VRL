# SPRINT: 字符串分支 → 多态 / 策略（OCP / LSP）

状态：D1 implemented（2026-06-09）；D2-D5 评估后不实施（见下）。父：`SPRINT_solid_architecture_audit.md`（子 sprint D）。

落地记录：
- D1 ✅ **方案修正**：`ReplaySegmentResult` 没有子类层级（统一 dataclass + dict payload），所以不是「子类多态」而是把字段名知识收进 result 类本身——新增 `ReplaySegmentResult.logprobs(token_ids)`（`models/interfaces/replay.py`），log_probs/logits/image_logits/text_logits 的 payload-key 解析归数据契约 owner；evaluator 的 `_extract_logprobs` 退化为「补 token_ids 兜底 + 调 result.logprobs()」。加新模态字段名只改 replay.py 一处。
- D2 ⏭️ 不实施：230 行 rollout 热路径重构，doc 要求 golden 逐位验证（需 GPU + Janus 权重跑真实 R1 rollout）；R1 executor 当前工作正常，按「不重写工作中的架构」原则保留继承形态，待真实需要第二个执行模式时再 Strategy 化。
- D3 ⏭️ 不实施：doc 要求与子 sprint B 的 T4 合做，B 被明确跳过。
- D4/D5 ⏭️ 按 doc 自身判断（YAGNI / 排后）记录不实施。
- 验证：rollouts+models 197 passed，ruff 干净。

## 0. Core Decision

把「在字符串 / 字段名 / 枚举上硬编码 if-else」的几处改成多态或策略组合，让「加一个新策略 / 新模态 / 新算法」**零改老函数体**（OCP）。

- **判据**：只改「新增同类必须改老分支」的真 OCP 违例；不为「分支天然聚集」（如 Pydantic 跨字段校验）强行拆。
- D2 同时修一个 LSP 违例（并行类层级 → 策略组合）。

## 1. 目标清单（实测位置）

| # | 符号 | 位置 | 现状 | 加新东西的代价 |
|---|---|---|---|---|
| D1 | `_extract_logprobs` | `rollouts/evaluators/ar/multi_segment_token_logprob.py:186-195` | if-else 查 `log_probs`/`image_logits`/`text_logits`/`logits` 字段名 | 加模态改它 |
| D2 | `JanusProR1PipelineExecutor` | `models/ar/janus_pro/runtime.py:761-997` | 整体 override `forward_plan` + `forward_chunk_plan` | 并行类层级 |
| D3 | `generate_with_refine` | `models/ar/janus_pro/model.py:531-837`(308行) | 硬分支 `'selfcheck_text' in stages` / `refine_mode` | 加 refine 策略改 308 行 |
| D4 | `_apply_rollout_compile_override` | `generation/ray/launcher.py:350-381` | 硬编码路径，只处理 `torch_compile` | 加 quant/sparsity 改它 |
| D5 | `RootConfig._cross_field_validate` | `config/schema.py` | 对 4 算法族硬分支 | 加算法改 validator |

## 2. 分步实施

### D1 [收益最大、风险最低] `_extract_logprobs` → 多态
- 现在 evaluator 知道每种 segment result 内部把 logprob 存在哪个字段。应由结果对象自己回答。
- 在 `ReplaySegmentResult`（grep 确认类型）加 `get_logprobs(token_ids) -> Tensor` 多态方法，各模态子类自己知道从哪取。
- `_extract_logprobs` 退化为 `return segment.get_logprobs(token_ids)`，删字段名 switch。
- 加新模态 = 新 result 子类实现 `get_logprobs`，evaluator 零改。

### D2 [LSP+OCP 双修] `JanusProR1PipelineExecutor` → Strategy
- 现状：R1 executor 同时整体 override `forward_plan` 和 `forward_chunk_plan`，把父类用 `ARDecodeLoop` 的两条路径整体替换成 `generate_with_refine()`。discrete-AR 与 R1-refinement 是**正交关注点**，用继承表达成了并行类层级。
- 改成 `ARExecutionStrategy` 接口，两个实现 `DiscreteARStrategy`（走 `ARDecodeLoop`）/ `R1RefinementStrategy`（走 `generate_with_refine`）。单个 executor 持有一个 strategy，按配置组合而非继承。
- 消除「新执行模式 = 新 executor 子类」。
- **风险**：executor 在 rollout 热路径，先抓 golden（Janus R1 rollout 输出），重构后逐位一致。

### D3 `generate_with_refine` → `RefinementPolicy`（与子 sprint B 的 T4 合做）
- 308 行同时管 tensor 拼接 + stage 分支（`'selfcheck_text' in stages`）+ refine 逻辑（`refine_mode == never/always/selfcheck`）+ masking。
- 提取 `RefinementPolicy{Never, Always, SelfCheck}`，每个 policy 决定是否 refine、如何 refine；`generate_with_refine` 留 tensor 编排骨架，调 `policy.should_refine()` / `policy.refine()`。
- **与 B-T4（JanusProModel 拆分）一起做**——拆 model 时这段顺势抽走，避免两次动同一文件。

### D4 `_apply_rollout_compile_override` → `PayloadOverrideRegistry`
- 现在硬编码 `_ROLLOUT_COMPILE_CFG_PATH` 且只认 `torch_compile`。
- 改成小注册表：override 类型 → handler。当前只注册 `torch_compile` 一项，但加 quant/sparsity override = 注册一行，不改本函数。
- **低优**：只有真要加第二种 override 时收益才兑现。可推迟到那时再做（YAGNI 边界，记录于此即可）。

### D5 `RootConfig._cross_field_validate` → `AlgorithmValidator` 注册表
- 在 Pydantic validator 里对 4 算法族 + kling 生产逻辑做跨字段校验。
- 抽 `AlgorithmValidator` 协议 + 按 `algo.kind` 注册的实现；`_cross_field_validate` 遍历注册表调对应 validator。加新算法 = 注册一个 validator，不改 `RootConfig`。
- **缓和因素**：Pydantic 校验天然聚集跨字段逻辑，影响中等。先做 D1–D3，D5 视收益排后。

## 3. 测试策略

- D1：给 `get_logprobs` 加多态契约测试；现有 `tests/rollouts/evaluators/` 用例回归。
- D2/D3：Janus R1 rollout 抓 golden，重构后逐位一致（热路径，最关键）。
- D5：`tests/config/` 下每种算法的校验用例必须仍按原样通过/报错。
- 所有改动是行为保持的结构重构——输出/报错 diff 都是 bug。

## 4. Non-Goals

- **不动** `select_tensor_tree`（`rollouts/batch/ops.py:53-65`）的 isinstance 分支——low，`singledispatch` 可解但不值得专项。
- D4/D5 在没有「第二种 override / 新算法」的真实需求前可只记录不实施（避免过度设计）。
- 不把 Pydantic 跨字段校验全拆散——只抽算法族那部分，kling 生产校验若与算法无关则留原处。

## 5. 关键参考文件

- `vrl/rollouts/evaluators/ar/multi_segment_token_logprob.py:186-195`、`ReplaySegmentResult`（grep）
- `vrl/models/ar/janus_pro/runtime.py:761-997`、`vrl/models/ar/janus_pro/model.py:531-837`
- `vrl/generation/ray/launcher.py:350-381`、`vrl/config/schema.py`
- `ARDecodeLoop`（grep，D2 的两条路径来源）
- 相关：子 sprint B 的 T4（D3 合做）
