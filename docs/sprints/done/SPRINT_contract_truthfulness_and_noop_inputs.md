# SPRINT：Contract truthfulness and no-op inputs

状态：**done（2026-07-22）**。

父 program：[Argument and state ownership](../planned/SPRINT_argument_and_state_ownership_program.md)

前置：无。这是 program 的正确性 Sprint。

## 0. 结论先行

先修“参数看似被接受，实际含义错误或完全无效”的路径，再做 dataclass 瘦身。当前至少有两项会
直接改变训练/评估行为：

```python
# vrl/scripts/families/wan_2_1/train_dpo.py
if bool(gradient_checkpointing):
    transformer.enable_gradient_checkpointing()

# vrl/rollouts/orchestration/prompt_collection.py
if isinstance(remap, list):
    remap_group_ids_(batch, remap)
else:
    batch.group_ids[:] = remap
```

第一段把字符串 `"off"` 当成 true；第二段的 scalar 分支没有同步 evaluator 优先读取的
trajectory group IDs。两者都必须先用 true/false 测试钉住。

本 Sprint 同时清理 public no-op config 和 family silent ignore。共同定义是：**调用链活着，但
输入不能改变任何支持的行为，也没有明确拒绝 unsupported value。**

## 1. T0 — DPO activation checkpointing 与目标默认

### 当前问题

- public 值支持 `off | full | selective | bool`。
- DPO 自己做 `bool(value)`，所以 `"off"` 和 `"selective"` 都进入 full checkpointing。
- DPO 构造 `OfflineDPOTrainerConfig`，却从 online `TrainerConfig.__dataclass_fields__`
  借 `max_norm` / `resume_strict` 默认。

### 修复

1. DPO 调用共享 `enable_transformer_gradient_checkpointing(bundle, cfg)`；
2. 删除 DPO 的 bool 转换分支；
3. `max_grad_norm` 缺省来自 `OfflineDPOTrainerConfig`，显式保留
   `actor.max_norm → max_grad_norm` 的 public mapping；
4. `resume_strict` 暂时从 shared checkpoint policy 派生；不能继续把 online trainer 当默认注册表。

### 测试

- `"off"` / `False` 不调用 enable；
- `"full"` / `True` 调用 full；
- `"selective"` 调用 selective，unsupported family走明确 fallback/error；
- 缺 `actor.max_norm` 使用 offline 默认；
- 正值执行 clip，`0` 跳过 clip。

## 2. T1 — Prompt group remap 单一行为

### 当前问题

plain string list分支调用 `remap_group_ids_()`，同时更新 batch/trajectory；`PromptExample` scalar
分支只原地改 `batch.group_ids`。`TrajectorySignalBuilder.group_ids` 当前优先返回
`trajectory.group_ids`，设备或 dtype 转换打破 alias 后会读到 stale group。

### 修复

- scalar 与 list 均调用同一 `remap_group_ids_()`；
- 该 helper 在本 Sprint 仍同步两份，保证行为立即正确；
- 删除 trajectory 数字副本属于后续 single-source Sprint，不能混在本修复里。

### 测试

在测试中刻意构造不 alias 的 batch/trajectory tensors：

- scalar `PromptExample` remap 后两边相同，signal builder观察到新值；
- plain string list 的 0/1 两组仍同步；
- 空 remap保持原值；
- group count不匹配继续失败。

## 3. T2 — NFT algorithm input 显式化

### 当前问题

`AlgorithmInput.metadata` 的 production reader 只有：

```python
model = inputs.metadata.get("model")
batch = inputs.metadata.get("rollout_batch")
timestep_index = int(inputs.metadata.get("timestep_index", 0))
```

这是 closed execution contract，却被隐藏在 `dict[str, Any]`；缺 timestep 静默取 0，`-1` 会索引
最后一列。

### 修复

- 在现有 `AlgorithmInput` 增加 optional `model`、`rollout_batch`、`timestep_index`；
- 删除 `metadata`，不新增只有三个字段的 `NFTContext` 层；
- NFT 明确要求三个值存在，并验证 `0 <= timestep_index < width`；
- adapter 从 `inputs.rollout_batch` 校验 required replay tensors；
- `MultiSegmentTokenGRPO → TokenGRPO` 内层 input只传 `signals` 和 `advantages`。

### 测试

- NFT 的 index `0` 和最后一个合法 index通过；
- 分别缺 model/batch/index失败；
- `-1`、`width` 失败；
- GRPO 不需要 NFT-only 字段；
- multisegment aggregation数值不变。

## 4. T3 — Image-size 与 checkpoint 边界

### 当前问题

Janus 从 token count推 square grid，NextStep 直接 `del image_size`；两者都接受 public
`sampling.image_size` 却不验证。Janus 的 square-grid 和 checkpoint type check 使用 runtime
`assert`，在 `python -O` 下消失。

### 修复

- Janus 用 token grid × `JANUS_IMAGE_PATCH_SIZE` 推 expected size，与 request值比较；
- 非平方 token count显式 `ValueError`；
- checkpoint model type显式 `TypeError`；
- NextStep 至少在 VAE decode 后验证实际 `H/W` 等于 request size；若 upstream 暴露稳定 geometry，
  则提前验证；
- public `image_size` 参数保留，它是跨 family request protocol，不增加 capability/context DTO。

### 测试

- 匹配尺寸正常 decode；
- request 尺寸不匹配失败；
- 非平方 tokens失败；
- 在 `python -O` 子进程中错误仍被拒绝；
- LlamaGen 既有正确 size invariant保持。

## 5. T4 — Unsupported prompt/replay 参数 fail fast

### Negative prompt

Flux、HunyuanVideo、Echo 目前直接丢弃 `negative_prompt`。family 知道自身 distilled/DMD 架构是否
支持 unconditional branch，因此：

- `None`、空字符串、等价空 batch通过；
- 非空 string/list明确报 unsupported；
- 保留统一 `encode_prompt(..., negative_prompt=...)` 形状；
- CausVid/Magi 既有 fail-fast形态保持。

### Replay request

统一 `ReplayModel.replay_forward(batch, timestep_idx=0, *, request=None)` 是真实协议，**KEEP**：

- denoise evaluator真实使用 timestep；
- Janus R1/CausVid grouped replay真实使用 segment request。

各 family 实现需要拒绝其不支持的显式值：

- ordinary denoise：absent 或 `("denoise",)`；
- single-segment token：absent 或 `("image_tokens",)`；
- 无 timestep语义的 AR/grouped family：显式非零 timestep失败。

不拆 AR/Diffusion 两套 interface，不新增 registry capability。

## 6. T5 — 删除 public config no-op

### Data keys

删除 public schema与 presets 中：

- `data.source`；
- `data.preprocessing.metadata_schema`；
- `data.preprocessing.target_text`；
- `data.preprocessing.media_type`。

最后一个当前只要求 key存在但不读取值，`media_type: banana` 也能通过。真实媒体语义由
`data.task_type`、manifest格式和 loader表达。

必须保留：

- `PromptExample.target_text`：这是活的 manifest data field；
- `data.source_report`：production provenance/gate；
- artifact root、absolute-path policy、conditioning/reference fields。

### Production keys

把：

```python
production: Annotated[dict[str, Any] | None, OPEN] = None
```

换成 closed `ProductionSection` / `KlingVideoRewardProductionConfig(enabled=False)`。删除 8 个无人读取的
`report_path` assignments。保留 contract、file、backend 三个 validator，它们对应不同故障边界。

### 测试

- image manifest不再要求 `media_type`；
- 已删除的 legacy key由 unknown-key gate拒绝；
- `source_report` 正反 production validation不变；
- `enabled=false` 跳过 backend gate，`true` 执行；
- `enabld`、`report_path` 失败。

不增加兼容 alias warning；这些 key 当前没有可保留的行为，继续接受只会延长 no-op。

## 7. T6 — SDE request/loop owner

### 当前问题

- full-sequence layout唯一 parser总会创建 `DenoiseSDEParams`，所以
  `DiffusionSamplingParams.sde: ... | None` 和 executor 的 `None` branch不可达；
- `sde_window_size/range` 是 request级窗口策略，却存放在 loop-owned SDE params；
- 真正 denoise loop只消费已经选好的 `sde_window`。

### 修复

- `DiffusionSamplingParams.sde` 改 non-optional；
- 删除 executor 的 impossible branch；
- size/range放到 request/layout sampling params；
- layout为每个 chunk计算 `sde_window`，`DenoiseLoopConfig` 只接收 resolved window；
- 保留 loop SDE数学参数和 stage facade。

### 测试

- native/SDE mode都产生 non-null SDE config；
- window size 0 → `None`；
- 正数 window落在合法 range；
- 非法 range、size、mode失败；
- native/SDE true/false sampling path数值形状不变。

## 8. What changes / what stays

### 改变

- silent ignore变成真实支持或明确拒绝；
- DPO/default derivation使用目标类型和共享 resolver；
- group remap两条分支统一；
- closed input从 metadata dict提升为现有 DTO的显式字段；
- request policy与 loop math state分层。

### 保持

- shared protocol参数；
- family-uniform method shape；
- DPO public YAML naming；
- `PromptExample.target_text` 与 provenance/gate字段；
- production validator薄边界；
- trajectory双写暂时保留到后续迁移；
- diffusion stage methods与 chunk-size probe。

## 9. Non-goals

- 不在本 Sprint 删除所有 dead DTO fields。
- 不合并 config层。
- 不删除 ReplayModel 参数来消除 lint。
- 不顺手改 trajectory source of truth。
- 不运行模型 inference、Ray 或 GPU。

## 10. Acceptance gates

- 所有上列 true/false单测通过；
- 对删除 config key运行全部 bundled experiment load；需要同时更新所有 preset writers；
- `ruff` touched files；
- `git diff --check`；
- CPU-only，无 Ray cluster/GPU。

## 11. 实施结果与审计判定

| Suspect | 最小生产证据 | 判定 | 已落地结果 |
|---|---|---|---|
| DPO `bool(gradient_checkpointing)` | `"off"` 经 Python truthiness 进入 enable 分支 | **FIX** | DPO 复用 shared activation-checkpoint policy；bool/string/selective均有正反测试 |
| DPO 从 online `TrainerConfig` 借默认 | 构造目标实际是 `OfflineDPOTrainerConfig`；resume strict属于 checkpoint restore | **DERIVE** | max norm从目标 dataclass派生；strictness由 checkpoint protocol resolver拥有 |
| scalar prompt group remap | scalar只写 batch，evaluator优先读 trajectory | **FIX** | scalar/list统一经过 `remap_group_ids_()`，非 alias tensor回归通过 |
| `AlgorithmInput.metadata` | 三个 closed NFT reader隐藏在任意字典，index缺省为0 | **FIX** | model、rollout batch、timestep index成为显式 optional字段；NFT逐项 require并做上下界校验 |
| MultiSegment 内层重复字段 | inner TokenGRPO只消费 signals/advantages | **REMOVE** | 删除 rewards/group/model/batch/timestep等冗余 assignment，数值与最小输入测试通过 |
| Janus/NextStep `image_size` | Janus从 token数另算且用 `assert`；NextStep直接 `del` | **FIX** | token grid、patch size、VAE输出与请求尺寸均显式校验；`python -O` 仍失败 |
| Flux/HunyuanVideo/Echo negative prompt | family无 unconditional branch但曾静默丢弃输入 | **FIX** | empty值保留统一协议；非空值在 family边界 fail fast；Echo custom executor真实转发 |
| replay request/timestep | 多个 family保留统一签名却忽略不支持值 | **FIX/KEEP** | 保留一个 `ReplayModel` 协议；ordinary denoise、single-token AR、grouped/multisegment各自拒绝无效值 |
| 四个 public data key | preset有 writer，生产无 behavior reader；`media_type`只检查存在 | **REMOVE** | schema与全部 preset writer删除；unknown-key gate拒绝 legacy输入 |
| OPEN production dict / `report_path` | 只有 `enabled` 有 validator/preflight reader，8个 path无 reader | **FIX/REMOVE** | production改为 closed typed section；删除 path writer；保留三类 production validator边界 |
| optional SDE params与 window归属 | parser总构造 SDE；loop只消费已选择 window | **DERIVE/REMOVE** | SDE non-optional；window policy留在 request/layout；loop只接 resolved window；不可达分支删除 |

### ALL_CAPS 与薄边界

- **KEEP** `DEFAULT_CHECKPOINT_STRICT`：它是 checkpoint restore protocol默认，不是业务词表。
- **KEEP** closed production section：两个薄 dataclass承担 public schema/unknown-key边界。
- **KEEP** `require_replay_segments()`、`require_zero_replay_timestep()`：它们统一跨 family
  protocol backstop，避免每个实现复制且漂移。
- **KEEP** family-uniform `encode_prompt()` / `replay_forward()` 形状：一致性、可 grep性和 evaluator
  协议价值高于减少参数行数。
- 本 Sprint没有新增或保留混在 workflow中的大型 ALL_CAPS vocabulary/table。

### 提交

- `a0684864` — prompt/trajectory group remap同步。
- `c8dac327` — SDE request owner与不可达分支清理。
- `e6dd23e3` — DPO共享 checkpoint policy。
- `1aef2ea8` — public no-op config删除与 closed production schema。
- `73be6992` / `94033362` — NFT显式输入与 MultiSegment最小内层输入。
- `6a650983` — image geometry、negative prompt与 replay fail-fast。
- `3b6fa021` — DPO gradient clipping开/关回归。

### CPU-only acceptance

组合后的 correctness suite：

```text
766 passed, 2 deselected, 16 warnings
```

两个 deselect只依赖当前工作区缺失的
`third_party/CausVid/causvid` vendored source tree；同目录其余 CausVid contract均执行通过。
所有 touched Python文件通过 Ruff与 format check，diff whitespace check通过。未启动 Ray、未使用 GPU。

## 12. Definition of Done

- [x] `"off"` 不再开启 DPO checkpointing。
- [x] scalar/list remap都不会留下 stale evaluator grouping。
- [x] NFT 无隐式 timestep，负 index被拒绝。
- [x] image size、negative prompt、replay request无 silent no-op。
- [x] public data/production config只保留有行为的 key。
- [x] SDE request strategy与 loop resolved state各有单一 owner。

## 13. References

- `vrl/scripts/families/wan_2_1/train_dpo.py`
- `vrl/trainers/offline/dpo.py`
- `vrl/trainers/activation_checkpointing.py`
- `vrl/rollouts/orchestration/prompt_collection.py`
- `vrl/rollouts/batch/ops.py`
- `vrl/rollouts/evaluators/trajectory.py`
- `vrl/algorithms/trajectory.py`
- `vrl/algorithms/diffusion_nft.py`
- `vrl/algorithms/grpo/multisegment.py`
- `vrl/models/families/janus_pro/model.py`
- `vrl/models/families/nextstep_1/model.py`
- `vrl/models/families/flux/model.py`
- `vrl/models/families/hunyuan_video/model.py`
- `vrl/models/families/echo/model.py`
- `vrl/models/interfaces/replay.py`
- `vrl/config/schema.py`
- `vrl/config/validation.py`
- `vrl/generation/bindings/full_sequence_denoise/layout.py`
- `vrl/generation/bindings/full_sequence_denoise/executor.py`
- `vrl/generation/steps/denoise/config.py`
