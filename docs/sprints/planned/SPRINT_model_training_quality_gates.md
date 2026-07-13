# SPRINT：全模型训练质量门禁

状态：**PLANNED**

## 0. 结论与完成标准

这项工作不是给 SANA 再加一个 smoke test，而是把“可信推理是训练前提”变成所有模型都不能绕过的协议。
当前 registry 有 23 个 canonical family：17 个 diffusion family 和 6 个 AR family。每个 family、模型版本、
任务变体和训练 recipe 必须在启动 Ray 或创建训练 optimizer 之前，通过独立的原生推理基线；训练到首个有意义
checkpoint 后必须退出训练进程，在无 Ray 的独立进程中比较 base 与 checkpoint，通过后才允许恢复。

本 sprint 完成时必须同时满足：

1. `RolloutFamilyEntry` 的每个 canonical entry 都有非可选 quality binding；registry coverage test 从 typed registry
   派生 family 集合，不维护第二份硬编码清单。
2. `vrl-train` 在导入 family trainer 前完成 base gate；YAML、dotlist、直接恢复和 supervisor 均不能关闭或跳过。
3. fresh run、已有 checkpoint、checkpoint 缺 report、report 指纹过期、FAIL report 等恢复状态均有明确转移。
4. 23 个 family 都有精确到模型/变体的原生采样 profile、生产采样 profile、正例和反例；reference-conditioned
   family 使用真实 reference asset，不能用零张量代替。
5. base gate 比较独立的 native/reference path、真实 production rollout path、独立 replay path；不能让同一实现
   自证正确。
6. 首个训练 gate checkpoint 发布后，训练和 Ray 完全退出；独立 evaluator 验证无坍缩、语义/条件保持、
   reward corruption 防线和 rollout/replay 一致性，通过后再恢复。
7. CPU-only unit/config/contract tests 全绿；逐 family GPU audit 全绿后，才允许该 family 启动真实训练。
8. 报告绑定代码、dirty diff、resolved config、模型 revision、协议、输入资产、sampling、环境和 checkpoint hash；
   不匹配的历史 PASS 不能复用。

非目标：本 sprint 不证明训练会改善质量，也不以一套阈值强迫所有模态得分相同。它证明训练开始前输出正确，
并在最早可判断的 checkpoint 阻止颜色块、静帧、错条件、错 segment、token collapse 或 reward hacking 继续消耗算力。

## 1. 仓库事实与现有缺口

### 1.1 唯一 family source of truth

`vrl/rollouts/families/registry.py` 当前注册 23 个 canonical entry：

- T2I：`sd3_5`、`flux`、`qwen_image`、`sana`、`lumina2`、`hunyuan_image`、`pixart_sigma`、
  `cosmos-predict2-anima`。
- T2V：`hunyuan_video`、`mochi`、`cogvideox`、`wan_2_1`、`cosmos3`、`echo`。
- I2V：`wan_2_1_i2v`。
- V2W：`cosmos-predict2`。
- T2W：`cosmos-predict2.5`。
- AR：`janus_pro`、`janus_pro_r1`、`nextstep_1`、`emu3`、`glm_image`、`llamagen`。

alias 不是新的 coverage 单位；checkpoint/model identity 才是。CogVideoX 2B/5B、Wan 2.1/2.2、Wan T2V/I2V
等不能因为共享 family 名而共用一份模糊 PASS。

### 1.2 现有 probe 不能成为质量门

`vrl/scripts/diffusion/generate.py` 是有用的开发诊断，但它当前只能检查 finite/std 和首步 replay 数值：

- CLI 构造强制 `model.use_lora=False`，不能复现 checkpoint adapter/full-model 状态。
- rollout 和 replay 都从同一个 production model 对象导出，不能发现 production path 与原生 pipeline 同时写错。
- reference-conditioned family 没有完整的真实 image/video/world 条件协议。
- 自定义 executor family 不一定经过与生产一致的执行路径。
- 视频只保存中间帧，无法判断 freeze、repeat、flicker、reverse 或时序错乱。
- `std > 1e-3` 会让彩色块、噪声和 confetti 通过。

决定：**KEEP** 该脚本作为快速开发 probe；**REMOVE** 它作为训练放行依据的职责；质量门使用新的独立 runner，
避免把它逐渐扩成第二套训练系统。

### 1.3 当前入口可直接进入训练

`vrl/scripts/train.py` 的当前顺序是 load config → resolve `trainer.entrypoint` → import trainer → call trainer。
因此门禁必须位于 trainer import 之前。只在 `vrl/scripts/common/online.py` 增加检查会漏掉离线 DPO、未来 trainer、
直接 resume 和 supervisor restart。

`vrl/trainers/checkpointing.py:is_complete_checkpoint` 只证明 checkpoint 原子发布且大小可信，不证明图像/视频/token
质量通过。决定：**KEEP** checkpoint completeness 语义；新增独立的 quality report，不向 complete 塞入第二种含义。

## 2. 架构决策

### 2.1 Registry 强制绑定，不加可选开关

在 `RolloutFamilyEntry` 增加非默认、非可选的 `quality: QualityGateBinding`。最小 binding 只记录真实边界：

```python
@dataclass(frozen=True, slots=True)
class QualityGateBinding:
    protocol_resource: str
    reference_adapter: str
```

production adapter、collector kind、task、executor、runtime builder 和 replay builder 全部从现有 entry 派生。
不要复制到 quality binding。注册时缺 binding、protocol 不存在、profile 不能覆盖 resolved model identity 时立即失败。

不新增 `quality.enabled`、`skip_preflight`、环境变量 escape hatch 或默认 no-op adapter。训练质量不是实验偏好，而是
运行前置条件。release/nightly 可运行全 registry audit；普通启动只加载当前 resolved family/profile，并复用指纹完全
相同的 PASS，避免每次把 23 个模型全部装入 GPU。

### 2.2 `vrl-train` 成为 phase orchestrator

公共 `vrl-train` 负责状态转移，现有 trainer dispatch 移到内部 worker subprocess：

```text
resolve config and family
        |
        v
BASE PREFLIGHT (no Ray)
        |
        v
TRAIN_TO_GATE (worker owns Ray/trainer)
        |
        v
atomic checkpoint + worker/Ray exit
        |
        v
CHECKPOINT EVAL (no Ray)
        |
   PASS + matching fingerprint
        |
        v
resume worker to next gate/completion
```

新增 `vrl/scripts/train_worker.py` 作为内部 process boundary。它保留目前 `run_config()` 的 trainer import/call 逻辑，
但不安装公共 console script，也拒绝缺少 orchestrator phase token 的直接调用。这个 thin file 必须保留：它隔离 Ray/
CUDA ownership，确保 evaluator 不与训练进程共享 allocator、actors 或已修改的全局状态。

### 2.3 Trainer 只发布 gate checkpoint

online GRPO 和 Wan DPO loop 只增加统一的 `must_publish_gate_checkpoint(progress)` 判定：命中 gate 时原子保存并返回
typed phase result。trainer 不生成样本、不打 reward、不写 PASS。独立 orchestrator 接管后续评估。

首个 gate 分两层：

- first-update canary：验证 finite、梯度活性、policy version、rollout/replay 数值边界，尽早发现执行错误。
- first meaningful checkpoint：样本量足以检测坍缩和条件回退；不要求已经学到显著提升。

之后按 protocol 注册的 checkpoint milestones 重复质量门。门禁 schedule 是已解析协议的一部分，不能由训练 loop
自行猜测。

## 3. 不可绕过的状态转移

| 输入状态 | 行为 | 允许启动 Ray |
| --- | --- | --- |
| fresh run，无 base PASS | 运行 base preflight；FAIL 终止 | 否 |
| fresh run，base PASS 指纹匹配 | 启动 worker，训练到首 gate | 是 |
| checkpoint 低于下一 gate | 从完整 checkpoint 恢复到 gate | 是 |
| checkpoint 已到 gate、无 report | 直接独立 eval，不先恢复 trainer | 否 |
| checkpoint report 为 PASS 且指纹匹配 | 恢复到下一 gate 或完成 | 是 |
| report 为 FAIL | 写 non-retryable verdict 并终止 | 否 |
| report 指纹/hash 不匹配 | 作废旧 report，重新 eval | 否 |
| checkpoint 不完整 | 沿用现有完整 checkpoint 搜索规则 | 取决于最近完整点 |
| distributed checkpoint evaluator 不可见 | fail closed，报告拓扑错误 | 否 |

`run_verdict.json` 增加 phase 和 `retryable`。质量 FAIL、协议缺失、指纹不匹配后仍无法评估等属于
`retryable: false`；普通 transient crash 保留 supervisor 现有 bounded restart/circuit-breaker 行为。supervisor 必须读
verdict，而不是从 exit code 或日志文字猜测。

## 4. Typed contract 与报告

新增长期资产：

- `vrl/quality/contracts.py`：`QualityGateBinding`、resolved profile、assertion、report、phase result。
- `vrl/quality/run.py`：profile 解析、指纹、phase coordinator、report 校验。
- `vrl/quality/protocols/*.yaml`：按模态/真实采样协议组织的版本化 taxonomy/config asset。
- `vrl/quality/references/*.py`：真正独立的 upstream/native adapter。
- `vrl/scripts/quality_gate.py`：可复跑单个 family/profile 的公共 CLI。
- `datasets/preflight/`：小而固定、带 license/source/hash 的 prompt 与条件资产。
- `tests/quality/`：纯 CPU contract、transition、fingerprint、corruption、registry coverage tests。

protocol key 至少包含：

```text
canonical family
task and model identity
model revision / variant
recipe identity
native sampling profile
production sampling profile
prompt/reference manifest
seed grid
base assertions
checkpoint assertions
gate milestones
```

报告写入：

```text
<output_dir>/quality_gates/base/<profile_fingerprint>/report.json
<output_dir>/quality_gates/checkpoint-<global_step>/<profile_fingerprint>/report.json
```

每份报告绑定：source-tree digest（含 dirty diff 与 protocol assets）、resolved config/inference digest、模型和 reward
revision、输入资产 hash、package/CUDA/GPU 版本、sampling 参数、产物 hash、checkpoint hash、逐 assertion 结果。
所有 summary 必须能从逐样本记录重算；日志不是证据源。

## 5. Base preflight：所有 family 的共同要求

每个 profile 用固定 prompt/condition/seed 生成至少三组产物：

1. native/reference：使用上游公开 pipeline 或 family 原生实现，不导入 production executor/replay model。
2. production rollout：使用 registry 的真实 executor/runtime 和 resolved training sampling 参数。
3. independent replay：从落盘 trajectory/artifacts 重建，不复用 rollout 内存对象。

共同 hard checks：

- 模型 path、revision、task、dtype、scheduler、resolution/frame/token schedule 与 profile 一致。
- 输出、latents/logprobs、必要的 embeddings/tokens 全部 finite。
- 输出 shape、range、token/segment/schedule 合法。
- rollout/replay 在 family 允许的数值误差内；误差阈值来自 protocol，不按观察结果临时放宽。
- deterministic native/reference 与 production 达到模态适用的结构/语义相似度。
- stochastic 采样在同 seed 可复现；不同 seed 不是完全重复。
- 真实条件变化导致可检测的输出变化；交换 prompt/reference 的负例必须失败。
- 固色、纯噪声、patch block、confetti、全静帧、重复 token 等伪输出必须失败。

原生 sampler 与 RL sampler 合法不同的 family 使用双 baseline：先证明 native path 正常，再证明 production path 在
其注册的合法协议内正常。不能把 APG/CFG、DPM/DDIM、DMD/flow-SDE 的差异误判为 bug，也不能因“本来不同”而跳过
共同语义与防坍缩检查。

## 6. 模态门禁

### 6.1 图像

- 固定小 prompt set 覆盖主体、颜色、计数、空间关系、文字/细节与负面 prompt。
- 检查 resolution、dynamic range、edge/patch energy、色彩占用、近重复率、prompt-image 语义。
- reference image family 额外检查条件保持和 reference swap 反例。
- contact sheet 只用于人工审核；机器判定读取逐样本 metrics 和 artifact hash。

### 6.2 视频/世界模型

- 评估完整 clip，不抽中间帧代替视频。
- 检查帧数、fps/时间协议、temporal difference、motion range、freeze/repeat/flicker/reverse、prompt-video 语义。
- I2V/V2W/T2W 分别检查首帧/输入视频/世界条件保持；condition swap 必须触发失败。
- 对合法低运动 prompt 使用 profile-specific 下限，避免强迫静态场景产生虚假运动。

### 6.3 AR 图像生成

- 检查每个 family 的真实 token/latent schedule、special token、mask、segment 顺序和 decoder/VQ/flow path。
- 不以 4-token、64px、1-step flow 的 e2e smoke 代表原生质量。
- teacher-forced replay 与自回归 rollout 都要验证；invalid token、premature EOS、重复 token、segment swap 必须失败。

## 7. Diffusion family 覆盖矩阵

| Family | 必须固定的协议/变体 | 关键正例 | 必须失败的反例 |
| --- | --- | --- | --- |
| `sd3_5` | SD3.5 identity、Flow/CFG、真实精度 | native 与 production 结构/语义一致 | 错 autocast、错 scheduler |
| `flux` | FLUX variant、guidance、LoRA/full state | adapter on/off 均按 profile 生效 | 强制 `use_lora=False` 导致 checkpoint 无效 |
| `qwen_image` | model revision、resolution、prompt encoding | 文本/主体语义保持 | revision 漂移、prompt embed 错配 |
| `sana` | 1024px DPM native；RL Flow-Euler 独立 profile | 正常纹理、语义和 replay | bf16 base weights、错误 shift 产生色块 |
| `lumina2` | scheduler、resolution、text encoder | native/production 双基线正常 | 错 scheduler 或条件静默丢失 |
| `hunyuan_image` | APG native 与 CFG production 双基线 | 两条合法路径各自正常 | 把 sampler 差异当 PASS、prompt swap |
| `pixart_sigma` | DPM native 与 DDIM production 双基线 | 两条路径满足共同质量防线 | 只比像素或只查 finite |
| `hunyuan_video` | frame/fps/scheduler | 完整 clip 有合理时序 | middle-frame-only、冻结/重复视频 |
| `mochi` | sigma mapping 与 production SDE | 注册 mapping 下 replay 一致 | 错 sigma 仍因 finite 通过 |
| `cogvideox` | 2B/5B 分 profile | 每个 checkpoint identity 独立 PASS | 2B PASS 误放行 5B |
| `wan_2_1` | 2.1/2.2、T2V 分 profile | registry variant 与模型一致 | 从 cfg 再推导错误 variant |
| `wan_2_1_i2v` | 2.1/2.2、真实 reference image | 首帧/条件保持 | 零 reference、image swap 不失败 |
| `cosmos-predict2` | V2W 真实 input video | 条件与时序保持 | 假 video、错误 world condition |
| `cosmos-predict2.5` | T2W 原生 world protocol | 长度/条件/时序合法 | 错 task variant 或 schedule |
| `cosmos3` | 自定义 executor/replay | custom boundary 真实执行 | 被 generic probe 绕过 |
| `cosmos-predict2-anima` | artifact resolver、自定义 executor | artifact/revision 绑定 | 临时/缺失 artifact 静默 fallback |
| `echo` | DMD native 与 flow-SDE production 双基线 | 两条路径质量与时序均正常 | sampler 差异掩盖 freeze/collapse |

Wan-I2V、Predict2、Predict2.5、Cosmos3、Echo、Anima 的 custom executor/runtime thin boundary **KEEP**。
它们承载真实 family protocol，不为减少行数强行塞进 generic executor。

## 8. AR family 覆盖矩阵

| Family | 原生 profile | 关键正例 | 必须失败的反例 |
| --- | --- | --- | --- |
| `janus_pro` | 576 image tokens → 384px | 使用 `gen_head` 并可解码 | 错用 `lm_head`、token repetition |
| `janus_pro_r1` | initial image + selfcheck + final image | segment 顺序与 primary output 正确 | 缺 segment、segment swap、取错图 |
| `nextstep_1` | `[B, 1024, 64]` + 20-step flow → 256px | continuous tokens 与 flow decode 完整 | 4-token/1-step smoke 冒充质量门 |
| `emu3` | 4163-token schedule → 512px | schedule、mask、logprob 一致 | 强制错误 schedule/mask |
| `glm_image` | 1280 prior tokens + 20-step frozen DiT → 1024px | mRoPE 与 DiT 都执行 | 跳过 DiT、错误 mRoPE |
| `llamagen` | 256 tokens、T5 prefix 120 → 256px | prefix/null CFG/KV/VQ 正确 | prefix 长度错、KV/VQ path 错 |

不要新增 `_AR_FAMILIES` ALL_CAPS 集合。测试和 CLI choice 均从 `FAMILY_REGISTRY` 的
`entry.task` / `entry.collector.kind` 派生。

## 9. 首 checkpoint：质量与 reward corruption 防线

base PASS 只证明训练前正确。checkpoint gate 使用同一固定输入、相同 seed pairing 和独立 evaluator 比较 base 与
checkpoint。早期 checkpoint 不要求 reward 显著上升，但必须满足：

- 非有限、纯色块、噪声、重复产物、严重多样性坍缩为 hard FAIL。
- prompt/reference/temporal/segment 守卫不低于 protocol 下限。
- rollout/replay parity 与 policy/version/checkpoint hash 一致。
- reward 上升不能覆盖独立语义、条件或时序回退。

每个启用的训练 reward 必须用真实 base outputs 加人工构造 corruption 做方向测试：

- 图像：solid/color blocks、patch shuffle、极端 saturation/contrast、blur、confetti、prompt permutation。
- 视频：以上 corruption 加 temporal shuffle、freeze、repeat、flicker、reverse、condition swap。
- AR：token repetition、premature EOS、invalid token、forced schedule mutation、segment swap。

若 aesthetic 对某种色块给高分，这不是调 aesthetic threshold 的理由；独立结构与语义 guard 必须拒绝。训练 reward
负责优化目标，quality gate 负责证明输出仍是目标模态，二者不能共享单点失败模式。

## 10. Distributed ownership 与 fail-closed

当前 distributed training 的完整 checkpoint 由 rank 0 原子发布，跨节点场景可能依赖外部同步。独立 evaluator 必须
能读取同一个完整 checkpoint 和产物目录。phase handoff/shared checkpoint store 尚未实现前：

- 单机多卡可在所有 rank 退出后由 orchestrator eval。
- 多节点只有在 shared store 或显式完成同步协议可验证时放行。
- evaluator 不可见 checkpoint、hash 不一致或其他 rank/Ray 未退出时 fail closed。
- 不允许用 trainer 内联 eval 绕过 ownership 问题；那会重新引入 allocator、Ray 和分布式状态污染。

## 11. 文件级实施图

### 新增

- `vrl/quality/contracts.py`
- `vrl/quality/run.py`
- `vrl/quality/protocols/*.yaml`
- `vrl/quality/references/*.py`
- `vrl/scripts/quality_gate.py`
- `vrl/scripts/train_worker.py`
- `datasets/preflight/`
- `tests/quality/`

### 修改

- `vrl/rollouts/families/registry.py`：非可选 quality binding；从 registry 校验全量 coverage。
- `vrl/scripts/train.py`：phase orchestrator，在 trainer import 前 gate。
- `vrl/scripts/supervise.py`：识别 phase、non-retryable quality verdict 和 gate resume。
- `vrl/scripts/common/online.py`：只发布 gate checkpoint/typed phase result。
- `vrl/scripts/diffusion/wan_2_1/train_dpo.py`：与 online loop 相同的 phase handoff。
- `vrl/trainers/checkpointing.py`：暴露 checkpoint hash/provenance API；不改变 complete 定义。
- `vrl/models/diffusion/base.py`：统一 full-pipeline 与 replay component 的 revision 传播。
- SANA model/executor：修复已确认的 base parameter dtype 与 Flow-Euler shift 根因。
- config validation：拒绝未知的 gate/skip/no-op keys，不添加可选开关。
- `pyproject.toml`：package protocol assets 和 public `quality_gate` CLI；不暴露 worker CLI。
- `docs/ADDING_A_MODEL_FAMILY.md`：新 family 必须同时落 quality binding/profile/正反例。

### 保持不变

- registry family/task/collector、executor、runtime/replay builder 继续表达生产协议边界；
  worker 侧 generation kind 只从 `GenerationRuntimeLaunchContract.generation_kind` 读取。
- custom family executor 和 AR/diffusion 的统一 cross-family shape 保持不变。
- checkpoint complete、trainer resume state、Ray lifecycle ownership 各自保持单一含义。
- 现有 `vrl/scripts/diffusion/generate.py` 保留为开发诊断，不承载正式 verdict。
- 历史 one-shot smoke 输出不迁入长期协议；结论记录后按同 source + lifecycle 单独清理。

## 12. 测试计划

### CPU-only（每次改动都运行）

- registry 23/23 quality binding coverage；新增 entry 缺 binding 立即失败。
- profile resolution：family/model/revision/recipe true/false cases。
- fingerprint：代码、dirty diff、config、protocol、asset、revision、checkpoint 任一改变都会使旧 PASS 失效。
- transition table 全分支：fresh、resume、missing report、PASS、FAIL、stale report、incomplete checkpoint。
- report schema 与逐样本 summary 重算；日志-only/test-only 字段不算 behavior consumer。
- corruption detector true/false；正常低运动视频不能被误杀，冻结/重复视频必须失败。
- AR schedule/segment/token validator true/false。
- config unknown key/skip key rejection。
- supervisor non-retryable quality FAIL 不重启，普通 transient failure 保留 bounded retry。
- trainer import spy：base PASS 前不得 import trainer/Ray module。
- worker phase token、checkpoint atomic publish 与 evaluator visibility tests。

建议命令：

```bash
pytest -q tests/quality tests/rollouts/runtime/test_family_registry.py \
  tests/scripts/test_supervise.py tests/trainers/test_checkpointing.py
ruff check vrl/quality vrl/scripts/train.py vrl/scripts/train_worker.py tests/quality
pyright vrl/quality vrl/scripts/train.py vrl/scripts/train_worker.py
```

仅运行仓库实际配置的 CPU checker；若项目未配置 `pyright` 或命令名不同，先从 `pyproject.toml`/CI 读取真实命令，
不为文档命令临时新增工具。

### GPU（实施完成后的 registry audit，不在本 sprint 文档修改中执行）

对 23 个 canonical family 的每个注册 model identity/variant：

1. base native/reference、production rollout、independent replay；保存完整 artifact 和 report。
2. 执行每个正例与至少一个 family-specific 反例，证明 detector 本身会失败。
3. 用最小真实训练到 first-update canary 和 first meaningful checkpoint，退出 Ray，再独立 eval。
4. PASS 后才能进入常规长训练；FAIL 立即停止，不通过增加训练天数观察是否“自己变好”。

## 13. 实施顺序

### P0：Contract 与强制 coverage

- 建立 typed contracts、protocol loader、fingerprint 和 report schema。
- 给 23 个 registry entry 补非可选 binding。
- 先写 missing/stale/unknown profile 的失败测试，确保没有 optional bypass。

### P1：共享根因修复

- 修 full pipeline 与 component loader 的 revision 传播差异。
- 修 SANA base dtype 与 Flow-Euler shift，并用正确/错误两条 regression case 锁定。
- adapter/full-model checkpoint 加载必须 strict，不能因 missing keys 静默跑 base。

### P2：Base gate core

- 实现独立 native、production、replay 三路径 runner。
- 实现 artifact/report/provenance 输出和 PASS cache。
- 先完成一个图像、一个视频、一个 AR family 的纵向端到端，用共同 contract 验证三种模态。

### P3：全部 diffusion profile

- 落地 17 个 family 及其 model/variant profile。
- 对合法 sampler divergence 使用双 baseline。
- 保留 custom executors，补真实 reference condition 与完整视频评估。

### P4：全部 AR profile

- 落地 6 个原生 token/latent/schedule profile。
- 删除质量门对缩小 e2e smoke 的依赖；保留 smoke 作为结构测试。

### P5：Phase orchestrator

- 拆 public orchestrator 与 internal worker。
- online/DPO loop 返回 gate phase result。
- 串起 fresh/resume/supervisor 全状态转移与 non-retryable verdict。

### P6：Checkpoint 与 corruption gate

- 实现 first-update canary、first meaningful checkpoint 和后续 milestones。
- 接入模态 guard、reward corruption direction tests 和 paired base/checkpoint report。

### P7：Distributed handoff

- 验证所有 rank/Ray 退出、checkpoint 可见性和 hash。
- 未支持的跨节点拓扑 fail closed；实现后才开放相应 topology。

### P8：全 registry GPU audit

- 逐 profile 跑正/反例并归档报告。
- coverage 报告只从 registry 和 protocol resolution 派生，不手写“已测 family”表。

### P9：替代训练 canary

- 只有 base audit 通过后才启动新的 SANA 及其他目标训练。
- 首 gate 独立评估通过后再恢复；旧的 invalid SANA curve 不与新 run 拼接。

## 14. Architecture hygiene 决策

| Suspect | 证据 | 决定 |
| --- | --- | --- |
| 手写 `_ALL_FAMILIES` / `_AR_FAMILIES` | `FAMILY_REGISTRY` 已是 typed source of truth | **DERIVE** |
| 每个 YAML 的 `quality.enabled` | 会让直接配置或 override 绕过公共安全条件 | **REMOVE / 不新增** |
| `generate.py` finite/std probe | 有生产 forward/replay 诊断价值，但彩色块也可通过 | **KEEP** 为开发工具 |
| checkpoint complete = quality PASS | complete 只验证原子发布和文件可信 | **KEEP** 两种状态分离 |
| trainer 内联 quality eval | 与 Ray/CUDA ownership 混合，resume path 易绕过 | **REMOVE / 不新增** |
| custom executor thin files | 承载 reference/variant/artifact/segment 等真实协议 | **KEEP** |
| `train_worker.py` thin file | 提供 subprocess、Ray ownership 与非公开入口边界 | **KEEP** |
| protocol 中的大型 prompt/阈值表 | 属于版本化 domain taxonomy，不应混在 workflow code | **DERIVE/移动到 config asset** |
| quality report 的重复 family/task/generation_kind 字段 | registry、launch contract 与 resolved config 已提供，可由 provenance 绑定 | **DERIVE** |
| 只被日志和测试读取的 resolved 字段 | 没有生产行为 consumer，会静默腐烂 | **REMOVE** 或明确标注 provenance-only |

ALL_CAPS 只保留 schema key、环境变量名、checkpoint 文件名、模型维度、协议名和测试 fixture 等真实边界。
大型 prompt、backend 表、family taxonomy 放进命名清楚的 protocol asset。validation key set 从 dataclass/schema fields
派生，不手工维护副本。

## 15. 明确非目标

- 不用统一实现消灭 family-specific sampler、custom executor、AR segment 或 reference adapter。
- 不把 quality gate 做成 Ray actor；一致的 process ownership 比少一个进程更重要。
- 不在 gate 中优化训练超参数，也不因短 checkpoint 没有提升而判 learning 失败。
- 不用 aesthetic/reward 单分数代替语义、结构、条件和时序检查。
- 不用人工 contact sheet 代替机器报告；人工审核只能作为额外证据。
- 不把一次性 GPU probe 输出留在 production import graph。
- 不为减少 LOC 合并有意义的 public API、framework adapter、lazy import 或 cross-family consistency boundary。

## 16. 解锁条件

某个 model identity/variant 只有在以下全部满足后才允许长训练：

1. CPU contract/transition/corruption tests 通过。
2. base native、production rollout、independent replay PASS，且 report fingerprint 与当前工作树完全匹配。
3. family-specific 反例被同一 evaluator 拒绝，证明门不是恒 PASS。
4. first-update canary PASS。
5. first meaningful checkpoint 独立评估 PASS。
6. 当前 topology 的 checkpoint handoff 已验证；否则 fail closed。

这样，SANA 的颜色块事故会在训练前被 native-vs-production 与结构防线抓住；即使 base 正常但训练更新损坏输出，
也会在首 checkpoint 停下。相同合同同时覆盖 diffusion、video/world 与 AR，而不是依靠操作者记得为每个新模型手动
运行一个临时脚本。
