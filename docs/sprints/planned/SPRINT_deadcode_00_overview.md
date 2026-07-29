# SPRINT: 死代码 + 分层审计程序——总览与索引（reconciled）

状态：**RECONCILED（2026-07-24）**，对齐 main @ `7c748532`（= `origin/main` tip）。原审计跑在旧树
`88ed756e`，其后 origin 已落地约 63 个 cleanup/refactor commit，故原始发现里有相当一部分**已被独立
完成**。本篇是逐簇复核后的索引 + config 旋钮高亮 + 执行顺序 + 验证协议 + 已否决清单；逐条动作在下列
子 sprint 里，每篇均已叠加 2026-07-24 对当前 checked-out 树的逐条复核（`已由 origin 落地` / `仍有效` /
`情况已变` 三分类）。来源：dead-code-audit 与 layering-audit 两个 workflow（审计树 `88ed756e`）。

> **方法**：死代码线——18 个按模块的审计 agent 逐符号按 AGENTS.md「五种死代码形态 + thin-function
> 保留清单」定罪，每条判决过一个对抗性 verify agent，删除类再过第二道字符串引用检查。分层线——10 张
> 层卡 + AST 导入图 + 306 个 typed struct 清单 → 4 个跨层判决 → 逐条对抗验证。**偏置控制**：用户历史上
> revert 过两次过度扁平化，凡有跨家族一致性 / 协议边界 / lazy-import / 可独立测试的概念抽取等理由的一律
> KEEP。**本轮复核方法**：逐簇对 main @ `7c748532` 重跑 grep（排除 `.venv/ third_party/ outputs/
> datasets/ docs/runs/ __pycache__/ egg-info`），消歧同名符号碰撞，test-only 生产符号视作 DONE，行号漂移
> 更新到当前 `file:line`。

## 0. 一句话

**原 131 条死代码 + 14 条分层改动里，约 31 条已被 origin 在旧树之后独立落地——其中包括最锋利的几条**
（五家族 `_*_LORA_DEFAULTS` 双维护去重、4 个 `data.preprocessing.*` no-op YAML 旋钮、nextstep `token_dim`
死转发、`RewardScorer` 零消费者 facade、`RolloutBatch.dones` 等）。逐簇复核后的**在册总量收敛为：死代码
94 条仍需做 / 30 条已落地 / 8 条现场已变，分层 12 条仍需做 / 1 条已落地 / 1 条现场已变**（合计 145 条 →
**106 仍需做 / 31 已落地 / 9 现场已变**）。死代码里最该优先的仍是**「用户可设但无效」的 config 旋钮**
（§1）——但原 11 个里 **5 个已被删除**，只剩 6 个待做。分层线结论不变：**层大体都必要**，只有一小撮类型/
常量住错了层——详见 [[SPRINT_layering_audit]]。

## 1. 最高优先级：用户可设但无效的 config 旋钮（原 11 处 → 剩 6 处）

这些是「no-op 旋钮」——用户在配置里设了值，代码里**要么无 reader、要么被兜底默认覆盖**，静默无效。按
AGENTS.md 死字段规则，用户可见的 no-op 旋钮比普通死字段**更糟**。原 11 个中 **5 个已由 origin 删除**
（下表标 **DONE**），剩 6 个仍待做（**STILL-TODO**），各归所在模块的子 sprint：

| 旋钮 | 位置 | 状态 | 子 sprint |
|---|---|---|---|
| `data.source` | `config/schema.py`（`DataConfig.source`） | **DONE**（`1aef2ea8`） | [[SPRINT_deadcode_config_knobs]] §2 |
| `data.preprocessing.metadata_schema` | `config/schema.py` | **DONE**（`1aef2ea8`） | [[SPRINT_deadcode_config_knobs]] §2 |
| `data.preprocessing.target_text` | `config/schema.py` | **DONE**（`1aef2ea8`） | [[SPRINT_deadcode_config_knobs]] §2 |
| `data.preprocessing.media_type` **(med)** | `config/schema.py` | **DONE**（`1aef2ea8`） | [[SPRINT_deadcode_config_knobs]] §2 |
| nextstep `sampling.token_dim` | `nextstep_1/runtime.py` | **DONE**（`3e6367c9`） | [[SPRINT_deadcode_model_families]] §2 |
| `sampling.same_latent` **(med)** | `generation/steps/denoise/config.py:16`、`schema.py:304` | **STILL-TODO** | [[SPRINT_deadcode_generation]] §1.3 |
| janus `sampling.refine_mode` **(med)** | `janus_pro/config.py:60`、`runtime.py:408` | **STILL-TODO** | [[SPRINT_deadcode_model_families]] §1.13 |
| janus `sampling.task_stages` **(med)** | `janus_pro/runtime.py:283,400` | **STILL-TODO** | [[SPRINT_deadcode_model_families]] §1.12 |
| dpo `timestep_subset` **(med)** | `trainers/offline/dpo.py:60-62` | **STILL-TODO** | [[SPRINT_deadcode_rollouts_trainers_ray]] §1.5 |
| `GlobalRayPlacementOwner.placement_strategy` | `ray/placement.py`（原 :223，+3 行） | **STILL-TODO** | [[SPRINT_deadcode_rollouts_trainers_ray]] §1.8 |
| `--keep-model-between-checkpoints` | `scripts/eval/cosmos_predict25_kling_eval.py:116-120` | **STILL-TODO** | [[SPRINT_deadcode_scripts]] §1.6 |

**剩余 medium 的处理更谨慎**：janus `refine_mode`/`task_stages` 强耦合，须同批落地且只删 `final_image` 臂的
`stages` 半、保留 `mode == "never"`（[[SPRINT_deadcode_model_families]] §1.12/§1.13）；`same_latent` 端到端
删除、不补实现其承诺语义（[[SPRINT_deadcode_generation]] §1.3）。已删的 4 个 `data.preprocessing.*` 旋钮由
`1aef2ea8` 一次性落地，并新增 `tests/config/test_unknown_keys.py` 负向断言其为 UNKNOWN。

## 2. 死代码子 sprint 索引（8 篇 · 复核后 94 待做 / 30 已落地）

| 子 sprint | 仍需做 | 已由 origin 落地 | 现场已变 |
|---|---|---|---|
| [[SPRINT_deadcode_config_knobs]] | 1 | 4 | 0 |
| [[SPRINT_deadcode_nn_paged_attention]] | 10 | 0 | 0 |
| [[SPRINT_deadcode_generation]] | 10 | 6 | 0 |
| [[SPRINT_deadcode_model_families]] | 22 | 7 | 3 |
| [[SPRINT_deadcode_trajectory_math]] | 7 | 6 | 2 |
| [[SPRINT_deadcode_rewards]] | 10 | 3 | 0 |
| [[SPRINT_deadcode_rollouts_trainers_ray]] | 20 | 3 | 2 |
| [[SPRINT_deadcode_scripts]] | 14 | 1 | 1 |
| **小计** | **94** | **30** | **8** |

「仍需做」= STILL_VALID + RELOCATED（判死不变、部分行号漂移，已更新到当前 `file:line`）。「已由 origin
落地」= 生产符号已删（test-only 残留即算 DONE），逐条附 `landed_by` commit。「现场已变」= 判死前提被 origin
改动推翻或改写：model_families 的 3 条含 `decode_image_tokens(image_size=)` 两条已由 `6a650983` 转为
**行为消费**（不再是 no-op 旋钮，无需再做）+ `runs_in_isolated_subprocess` setter/测试已删（仅剩裸死字段一
行，风险降为 low）；trajectory_math 的 2 条（`stack_trajectory_batches`、`TrajectoryAxis.metadata`）新增了
origin 的回归测试，**捆绑删除不再原样适用**；scripts 的 `_video_to_cthw` 已被内联为**活生产实现**（不删）；
rollouts 的 2 条仍有效但支撑事实迁移（构造点迁至 `builders.py`、caller 数变化）。

## 3. 分层子 sprint（原 14 条 → 12 待做 / 1 已落地 / 1 已变）

[[SPRINT_layering_audit]]——直接回答「dataclass/arg 是否在正确层」「层是否都必要」，复核后：
- **12 条仍需做**（8 STILL_VALID + 4 RELOCATED，行号/文件已更新）：signal 契约困在 rollouts、`RolloutStats`
  困在中立 utils、`unwrap_compile_and_ddp` 与 `resolve_gradient_checkpointing_mode` 反向被上层定义、
  `IMAGE_SUFFIXES` byte 级抄两份、family task 词表三处手维护、families import-lightness 门禁缺失等。
- **1 条已由 origin 落地**：prompt loader 去重（`456f7069`）。
- **1 条现场已变**：generation→models interface-floor 门禁——因 origin 新增 `checkpoint_identity` 的
  off-floor 边，需先扩 floor 定义再加门禁测试，原动作不再原样适用。
- **必要性判决不变**：无一层被判多余；`composition/`、`vrl/ray` vs `generation/ray`、双 families 层、
  generation 五层栈全部「保留有理由」。`RolloutWorkerSection` vs `RayGenerationConfig` 仍建议加 parity 测试
  而非合并。

## 4. 执行顺序

1. **可立即并行**（互不碰、不碰在飞 sprint）：config_knobs（仅剩 1 条）、nn_paged_attention、model_families、
   trajectory_math、rewards、scripts、rollouts_trainers_ray。每篇一个独立 PR。落地前须先在当前 main 上重取
   一次干净基线（旧树数值已作废，见 §5）。
2. **排在 `SPRINT_native_generation_engine_program` 之后**：generation 篇的 3 条 generation/ray/ 条目
   （§1.2 `current_policy_version`、§1.7/§1.8 `validate_colocated_replay_memory` 参数）——同文件正被在飞
   sprint 改；generation 篇**其余 7 条仍需做项**（types/bindings/execution/steps）**不碰在飞面，可立即做**。
   model_families §1.27（`DiffusionModelBase.load`，`denoise/base.py`）同样排在该 sprint 之后。
3. **分层迁移**：低风险迁移可随时做；probe Protocol / worker_fleet 去重排在 native-engine sprint 后。
4. 建议顺序：先 config_knobs（剩 1 条最小）→ 各模块死字段/死参 → 分层迁移 → 最后补 families import-lightness
   与（扩 floor 后的）generation→models 两条架构门禁测试，固化本轮结论、防再腐烂。

## 5. 验证协议（所有子 sprint 通用）

- **基线须在当前 main @ `7c748532` 上重取**：旧树 `88ed756e` 的 `2620 passed / 7 pre-existing failures`
  数值**已作废**——§2/§3 的 origin 落地项已改变 `tests/` 内容（新增 `test_unknown_keys.py` 负向断言、
  `test_removed_fields_are_not_constructor_inputs` 等反向校验）。清理前先在当前 main 重跑
  `pytest -m "not e2e and not slow_test"` + `python -m vrl.config.lint` + `ruff check .` 建立干净基线，
  清理后须与该**新基线**持平。
- **逐条**：删除后 `ruff check <touched>` + `ruff format --check <touched>` 全绿（仅触及文件，禁止全仓
  `ruff format .`）；删除类先跑该 finding 列出的 `tests/` 子集。
- **行号以当前上下文为准**：RELOCATED 条目的行号已更新到 `7c748532`，但落地时仍**以文件当前上下文归属为准、
  不照抄行号**（尤其被在飞 sprint 改的 `generation/ray/`、`denoise/base.py`）。
- **全簇完成**：`pytest -m "not e2e and not slow_test"` 不新增失败；改动打包契约相关（registry import 路径）
  时另跑 CI `package` job 的 wheel 校验。

## 6. 已考虑但否决（防重议）

**死代码线否决**（对抗验证或字符串引用检查救回）：`TrajectoryAxis.kind`（`ops.py` 有 raise 消费）、
`ARRequestLayout.validate_chunk`（有活 runtime caller）、`Cosmos3ChunkExecutor(samples_per_chunk=)`
（另一路径消费）、`RewardScorer._score_many_locked` 的 `score_batch` 回退（e2e fake 用）、
`MultiSegmentTokenGRPO` dict-advantages 分支（活契约）、`CumemPool.require`（reward/generation
parking 的唯一生产构造器）、
`scripts/train.py:TrainTarget`（YAML entrypoint 活派发）。

**分层线否决**：`composition/` 单模块层不删、`LossUnit`+`TrainingView.loss_units` 不删、
`runs_in_isolated_subprocess` 的「重复 binding」定性否（但它作**死字段**成立，归 model_families 篇——现已因
origin 删 setter/测试降为一行裸删，见 §2 现场已变）、`distributed.training.strategy` 在 registry 读 raw cfg
合法、health-check 双校验不合并。详见 [[SPRINT_layering_audit]] §5。

## 7. Non-Goals（全程）

- 不删被「能 raise 的校验」/控制流分支/runtime·config·Ray 调用消费的字段（AGENTS 死字段规则）——尤其
  `decode_image_tokens(image_size=)` 已由 `6a650983` 转为行为消费，**不再是清理目标**。
- 不动 DO-NOT-FLAG 豁免项：`_GENERATION_CUDA_RUNTIME_RESIDUAL_BYTES`、`_REWARD_CUDA_RUNTIME_RESIDUAL_BYTES`、
  `ensure_loaded`（`rewards/runtime.py`）、`process_gpu_used_bytes`（NVML）、sana/hunyuan `prepare_latents=`
  修复——活实验依赖的未提交 worktree 改动。
- 不重复 origin 已落地项（§2/§3 的 30+1 条 DONE / 部分 CHANGED）——落地前先 grep 确认生产符号已不存在。
- 不删任何包级层、不下沉 `GenerationRequest`/`Output`、不合并 `RolloutWorkerSection`——见
  [[SPRINT_layering_audit]] Non-Goals。
- 不为省行数扁平化 protocol / lazy-import / 跨家族一致性 thin function。

## References

- 索引的 9 篇子 sprint（本目录 `docs/sprints/planned/SPRINT_deadcode_*.md` + `SPRINT_layering_audit.md`），
  每篇均含 §1 仍需做 / §2 已由 origin 落地 / §3 情况已变 三节
- 审计原始数据：`scratchpad/deadcode_confirmed.json`（131）、`scratchpad/layering_confirmed.json`（14）、
  `scratchpad/layer_cards.md`、`scratchpad/layer_import_graph.md`、`scratchpad/dataclass_inventory.md`
- 复核基线来源：`.github/workflows/ci.yml`（fast subset marker `not e2e and not slow_test`）；对齐 tip
  `7c748532`（= `origin/main`），审计旧树 `88ed756e`，区间约 63 个 cleanup/refactor commit
- 关联既往：[[SPRINT_single_caller_inlines]]、[[SPRINT_trajectory_views_types_dead_fields_cleanup]]、
  [[SPRINT_fbag_00_overview]]、[[SPRINT_generation_regime_decision_layering]]、
  [[SPRINT_native_generation_engine_program]]
