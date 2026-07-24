# SPRINT: 死代码 + 分层审计程序——总览与索引（planned）

状态：**planned（2026-07-23）**。全仓 `vrl/`（~92k LOC）一次性深度审计的落地计划总入口。两条独立
审计线：**死代码 131 条确认** + **分层 14 条确认**，均经对抗验证。本篇是索引 + 严重度排序 + 执行顺序 +
验证协议 + 已否决清单；逐条动作在下列子 sprint 里。来源：dead-code-audit 与 layering-audit 两个
workflow。

> **方法**：死代码线——18 个按模块的审计 agent 逐符号按 AGENTS.md「五种死代码形态 + thin-function
> 保留清单」定罪，每条判决过一个对抗性 verify agent（专门反驳），删除类再过第二道字符串引用检查
> （`FAMILY_REGISTRY` 点号 import 路径 / YAML 预设 / 动态 dispatch / pyproject 入口点）。分层线——
> 10 张层卡（逐层读真代码 + AST 导入图 + 306 个 typed struct 清单）→ 4 个跨层判决 → 逐条对抗验证。
> 共约 400 个 agent、~2000 万 token。**偏置控制**：用户历史上 revert 过两次过度扁平化，所以「错误的
> 该改」比「漏报」更糟——凡有跨家族一致性 / 协议边界 / lazy-import / 可独立测试的概念抽取等理由的，
> 一律默认 KEEP。死代码线否决 7 条、分层线否决 6 条，均记录防重议。

## 0. 一句话

**131 条死代码（100 low / 31 medium）+ 14 条分层改动，全部机械可执行、逐条带 grep 证据与测试同步
清单。** 死代码里最该优先的是 **11 个「用户可设但无效」的 config 旋钮**（§1）——你在 YAML/CLI 里设了
以为生效、实则被静默忽略的旋钮，这是所有死代码里对用户危害最大的一类。分层线直接回答了你的两个担忧：
**层大体都必要**（composition/、双 ray 层、双 families 层、generation 五层栈全部判「保留有理由」），
只有一小撮**类型/常量住错了层**（signal 契约、`RolloutStats`、`unwrap_compile_and_ddp`、
`IMAGE_SUFFIXES`、task 词表、gradient-checkpointing resolver）——详见 [[SPRINT_layering_audit]]。

## 1. 最高优先级：用户可设但无效的 config 旋钮（散落 11 处）

这些是「no-op 旋钮」——用户在配置里设了值，代码里**要么无 reader、要么被兜底默认覆盖**，静默无效。按
AGENTS.md 死字段规则，用户可见的 no-op 旋钮比普通死字段**更糟**（用户以为在调参，实际没有）。落地时
它们各自归属所在模块的子 sprint，但集中列于此供你一眼看全：

| 旋钮 | 位置 | 症状 | 子 sprint |
|---|---|---|---|
| `data.source` | `config/schema.py:191` | 字段无 runtime reader | [[SPRINT_deadcode_config_knobs]] |
| `data.preprocessing.metadata_schema` | `config/schema.py:170` | YAML 可设，零 reader | [[SPRINT_deadcode_config_knobs]] |
| `data.preprocessing.target_text` | `config/schema.py:171` | YAML 可设，零 reader | [[SPRINT_deadcode_config_knobs]] |
| `data.preprocessing.media_type` **(med)** | `config/schema.py:167,221` | 必填却无 reader | [[SPRINT_deadcode_config_knobs]] |
| `sampling.same_latent` **(med)** | `generation/steps/denoise/config.py:18` | 解析后无行为读 | [[SPRINT_deadcode_generation]] |
| janus `sampling.refine_mode` **(med)** | `janus_pro/runtime.py` | fallback 恒为默认 | [[SPRINT_deadcode_model_families]] |
| janus `sampling.task_stages` **(med)** | `janus_pro/runtime.py` | `_parse_task_stages` 无生产者 | [[SPRINT_deadcode_model_families]] |
| nextstep `sampling.token_dim` | `nextstep_1/runtime.py:44-50` | key 列表死项 | [[SPRINT_deadcode_model_families]] |
| dpo `timestep_subset` **(med)** | `trainers/offline/dpo.py:60-62` | config 有、逻辑不读 | [[SPRINT_deadcode_rollouts_trainers_ray]] |
| `GlobalRayPlacementOwner.placement_strategy` | `ray/placement.py:223` | 构造传入无消费 | [[SPRINT_deadcode_rollouts_trainers_ray]] |
| `--keep-model-between-checkpoints` | `scripts/eval/cosmos_predict25_kling_eval.py` | CLI flag 无效 | [[SPRINT_deadcode_scripts]] |

**medium 的处理更谨慎**：`media_type` 带必填校验支、janus 两个旋钮牵连 `_parse_task_stages` /
`_resolve_refine_mode` 函数与 `JanusProConfig` 字段——按各 finding 指定的删除顺序与测试改写落地，别只删
旋钮留下悬空校验。

## 2. 死代码子 sprint 索引（8 篇，131 条）

| 子 sprint | 条数 | medium | 主体 | 与在飞 sprint 冲突 |
|---|---|---|---|---|
| [[SPRINT_deadcode_config_knobs]] | 5 | 1 | 4 个 no-op YAML 旋钮 + `config_reads_in_code(root=)` 死参 | 无 |
| [[SPRINT_deadcode_nn_paged_attention]] | 10 | 2 | paged-attention 死字段级联（debug_info-only / write-only / test-only）+ `Fp4Linear(recipe=)` | 无 |
| [[SPRINT_deadcode_generation]] | 16 | 5 | `GenerationRequest` 死字段、执行/绑定死参 + 1 no-op 旋钮 | **3 条**（generation/ray/）→ 排在 native-engine sprint 后 |
| [[SPRINT_deadcode_model_families]] | 32 | 10 | 五家族 `_*_LORA_DEFAULTS` form-4 去重 + registry/semantics/各家族死字段·死参·旋钮 | 无 |
| [[SPRINT_deadcode_trajectory_math]] | 16 | 6 | renoise/flow_matching 数值死参·死分支 + builders 六处死数据键 + `stack_trajectory_batches` 死函数 | 无 |
| [[SPRINT_deadcode_rewards]] | 13 | 1 | 死参 + lazy `__getattr__` facade + `resolve_model_root` snapshot fallback | 无 |
| [[SPRINT_deadcode_rollouts_trainers_ray]] | 23 | 5 | 死形参 + 死旋钮主体（含 dpo/placement 旋钮、`gather_full_state_dict`） | 无（`gc_collect` 在 cuda_memory.py 但不属豁免） |
| [[SPRINT_deadcode_scripts]] | 16 | 1 | data/eval/perf/generation 死参·死字段·facade | 无 |

## 3. 分层子 sprint（14 条）

[[SPRINT_layering_audit]]——直接回答「dataclass/arg 是否在正确层」「层是否都必要」：
- **6 类错层/重复**（低风险机械迁移）：signal 契约困在 rollouts（且致算法层 eager-torch）、`RolloutStats`
  困在中立 utils、`unwrap_compile_and_ddp` 与 `resolve_gradient_checkpointing_mode` 反向被上层定义、
  `IMAGE_SUFFIXES` byte 级抄两份、family task 词表三处手维护。
- **2 处缺失架构门禁**（约定成立但无测试）：generation→models interface-floor、families import-lightness。
- **必要性判决**：无一层被判多余；`composition/`、`vrl/ray` vs `generation/ray`、双 families 层、
  generation 五层栈全部「保留有理由」。
- **1 条 contested**：`RolloutWorkerSection` vs `RayGenerationConfig`——建议加 parity 测试而非合并，待
  你拍板。

## 4. 执行顺序

1. **可立即并行**（互不碰、不碰在飞 sprint）：config_knobs、nn_paged_attention、model_families、
   trajectory_math、rewards、scripts、rollouts_trainers_ray。每篇是一个独立 PR。
2. **排在 `SPRINT_native_generation_engine_program` 之后**：generation 篇的 3 条 generation/ray/ 条目
   （§1.2 `current_policy_version`、§1.12/§1.13 `validate_colocated_replay_memory` 参数）——同文件
   `config.py`/`worker.py` 正被在飞 sprint 改，行号已漂；届时以最新文件重新定位。generation 篇**其余
   13 条**（types/bindings/execution/steps/composition）**不碰在飞面，可立即做**。
3. **分层迁移**：低风险迁移可随时做；§3.4（probe Protocol）、§3.5（worker_fleet 去重）排在 native-engine
   sprint 后。
4. 建议顺序：先 config_knobs（用户可见收益最高、最小）→ 各模块死字段/死参 → 分层迁移 → 最后补两条架构
   门禁测试（固化本轮结论，防再腐烂）。

## 5. 验证协议（所有子 sprint 通用）

- **基线（清理前，2026-07-23，务必先复现）**：fast subset **2620 passed / 7 项预先失败**（架构边界 3 +
  causvid/magi_1 打包摘要 4，与本清理无关，属在飞 sprint 未提交改动所致）；`python -m vrl.config.lint`
  与 `ruff check .` 全绿。**清理后这三项须保持：7 项失败不增、不新增红。**
- **逐条**：删除后 `ruff check <touched>` + `ruff format --check <touched>` 全绿；删除类先跑该 finding
  列出的 `tests/` 子集。
- **复核偏差**：审计与写文档间隔约 1h，部分文件（尤其被在飞 sprint 改的 `generation/ray/config.py`、
  `test_schema.py`）行号已漂——子 sprint 已用 **⚠ 复核偏差** 标注，执行时**以文件当前上下文归属为准，
  不照抄行号**。所有 substantive 判死证据经 2026-07-23 复核仍成立。
- **全簇完成**：`pytest -m "not e2e and not slow_test"` 不新增失败；改动打包契约相关（registry import
  路径）时另跑 CI `package` job 的 wheel 校验（`.github/workflows/ci.yml:197-260`）。

## 6. 已考虑但否决（13 条，防重议）

**死代码线否决 7 条**（对抗验证或字符串引用检查救回）：`TrajectoryAxis.kind`（`ops.py:242` 有 raise
消费）、`ARRequestLayout.validate_chunk`（有活 runtime caller）、`Cosmos3ChunkExecutor(samples_per_chunk=)`
（非 no-op，另一路径消费）、`RewardScorer._score_many_locked` 的 `score_batch` 回退（e2e fake 用）、
`MultiSegmentTokenGRPO` dict-advantages 分支（今日无生产者但是活契约）、`CumemPool.require`（子进程脚本
字符串里用）、`scripts/train.py:TrainTarget`（YAML entrypoint 活派发路径）。

**分层线否决 6 条**：`composition/` 单模块层不删、`LossUnit`+`TrainingView.loss_units` 不删、
`runs_in_isolated_subprocess` 的「重复 binding」定性否（但它作**死字段**成立，归 model_families 篇处理）、
`distributed.training.strategy` 在 registry 读 raw cfg 合法、health-check 双校验不合并。详见
[[SPRINT_layering_audit]] §5。

## 7. Non-Goals（全程）

- 不删被「能 raise 的校验」/控制流分支/runtime·config·Ray 调用消费的字段（AGENTS 死字段规则）。
- 不动 DO-NOT-FLAG 豁免项：`_GENERATION_CUDA_RUNTIME_RESIDUAL_BYTES`、`_REWARD_CUDA_RUNTIME_RESIDUAL_BYTES`
  （显存残留配额）、`ensure_loaded`（`rewards/runtime.py`）、`process_gpu_used_bytes`（NVML，
  `utils/cuda_memory.py`）、sana/hunyuan `prepare_latents=` 修复——这些是活实验依赖的未提交 worktree 改动。
- 不删任何包级层、不下沉 `GenerationRequest`/`Output` 破 models↔generation 双向、不合并
  `RolloutWorkerSection`——见 [[SPRINT_layering_audit]] Non-Goals。
- 不为省行数扁平化 protocol / lazy-import / 跨家族一致性 thin function。
- 不做整文件级 one-shot artifact 清理（本轮 symbol 级审计不含此范围）。

## References

- 索引的 9 篇子 sprint（本目录 `docs/sprints/planned/SPRINT_deadcode_*.md` + `SPRINT_layering_audit.md`）
- 审计原始数据：`scratchpad/deadcode_confirmed.json`（131）、`scratchpad/layering_confirmed.json`（14）、
  `scratchpad/layer_cards.md`、`scratchpad/layer_import_graph.md`、`scratchpad/dataclass_inventory.md`；
  逐 agent journal 在各 workflow 转录目录
- 基线来源：`.github/workflows/ci.yml`（fast subset marker `not e2e and not slow_test`）
- 关联既往：[[SPRINT_single_caller_inlines]]、[[SPRINT_trajectory_views_types_dead_fields_cleanup]]、
  [[SPRINT_fbag_00_overview]]、[[SPRINT_generation_regime_decision_layering]]、
  [[SPRINT_native_generation_engine_program]]
