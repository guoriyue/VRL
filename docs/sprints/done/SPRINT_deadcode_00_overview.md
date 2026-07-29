# SPRINT: 死代码与分层审计——完成索引（done）

状态：**DONE（2026-07-24；2026-07-28 归档复核）**。

> 原 overview 仍把已经落地的工作写成 “106 条仍需做”，会诱导重复执行。
> 本文件只保留完成索引、撤销项和长期边界；逐条执行前证据保存在同目录的历史子 sprint。

## 1. 完成索引

| 簇 | 结果 | 落地提交 |
|---|---|---|
| config no-op / dead inputs | 完成 | `1aef2ea8`, `7c0ee4c1` |
| generation dead fields / args | 完成 | `78212af3`, `f73d2751`, `7056ea69`, `801a5b77` |
| model-family dead code | 完成 | `a8848c12`, `f4730774`, `d8235a10`, `3e6367c9`, `6ed86b60`, `6a75406a`, `6e0cfd8e` |
| paged-attention / fp4 dead fields | 完成 | `c5046266` |
| rewards dead code | 完成 | `94761143`, `b7714bdc` |
| rollouts / trainers / Ray dead code | 完成；1 条撤销 | `5b2236f1`, `7cfe90ef`, `a9bc6072`, `c951ceba`, `58c4e0b3` |
| scripts 内部死符号 | 完成；1 条现场活化 | `ae3a3e96` |
| trajectory / math dead code | 完成；2 条由行为测试救回 | `6ed86b60`, `7f3b8d61`, `c6ef0027`, `2c3a5c76`, `ca601d45`, `6cebd279`, `cd6beea5` |
| layering audit | 11 条完成；1 条门禁拆出 | `456f7069`, `b54d4205`, `7a4f2b5f` |

唯一仍需执行的分层项是
[[SPRINT_generation_models_interface_floor]]；它已从历史大文档中拆成独立、可验证的 planned sprint。

## 2. 撤销与现场变化

- `collect_prompt_batches` 不删：`remap_group_ids_` 承载 per-prompt 优势归一化。
- scripts 的 `_video_to_cthw` 不删：复核时已有活生产实现。
- `stack_trajectory_batches` 与 `TrajectoryAxis.metadata` 不按原计划删：后续行为测试证明其契约仍被使用。
- `decode_image_tokens(image_size=...)` 不删：Janus-Pro 与 NextStep 已加入可抛错的行为校验。
- `GenerationRuntimeCapabilities.runs_in_isolated_subprocess` 的旧 setter/test 已变化；最终残余随 model-family 簇处理。

## 3. 必须保持的边界

- 保留 protocol、public facade、lazy-import、framework adapter 与跨 family 一致形状；不以减少行数为由扁平化。
- 保留 `DiffusionStagedChunkExecutor`、`ReferenceConditionedChunks` package facade 与 `StatsSink`。
- 保留集中后的 `IMAGE_SUFFIXES` taxonomy；不要重新复制到 rewards/trainers。
- 保留能够影响控制流、传给 runtime/config/Ray、或能 raise 的 resolved/capability 字段。
- 历史子 sprint 中的 KEEP / Non-Goals 判决继续有效；归档不表示可以重新 sweep。

## 4. 历史文档使用规则

同目录以下文件是执行前审计记录，不是当前任务清单：

- `SPRINT_deadcode_config_knobs.md`
- `SPRINT_deadcode_generation.md`
- `SPRINT_deadcode_model_families.md`
- `SPRINT_deadcode_nn_paged_attention.md`
- `SPRINT_deadcode_rewards.md`
- `SPRINT_deadcode_rollouts_trainers_ray.md`
- `SPRINT_deadcode_scripts.md`
- `SPRINT_deadcode_trajectory_math.md`
- `SPRINT_layering_audit.md`

其中的路径、行号和 “STILL_VALID” 标签只描述 `7c748532` 附近的执行前现场。判断当前代码时必须重新从定义、生产者、消费者和测试取证，不能照旧行号执行。

## References

- 审计基线：`88ed756e`
- 首次 reconciled 基线：`7c748532`
- 原始材料：`scratchpad/deadcode_confirmed.json`, `scratchpad/layering_confirmed.json`,
  `scratchpad/layer_cards.md`, `scratchpad/layer_import_graph.md`,
  `scratchpad/dataclass_inventory.md`
- 生命周期规则：`docs/sprints/README.md`
