# SPRINT: Small-function consolidation (proposed)

状态：proposed（2026-06-10 审计完成，未实施）。

## 0. Core Decision

全仓 AST 扫描 + 逐条人工校验后的结论：**问题不是"薄函数太多"本身，而是三类具体的赘肉**——
(a) 无人调用的死 API 面，(b) 纯转发壳，(c) 少数热点文件里的碎片化 helper 堆积。
不做 279 个单调用方 helper 的全量 inline（那是 churn 不是清理）；只做下面有证据的子集。

审计方法（可复现）：AST 扫描 `vrl/` 264 文件——单调用方私有 helper 279 个、转发壳 25 个、
<60 行薄文件 28 个、零文本引用符号 26 个；每个候选再对 `configs/`（YAML entrypoint /
registry 名字字符串）、`tests/`、`docs/` 做字符串 grep 复核。扫描器已知假阳性已剔除：
返回 closure 的工厂（如 `weight_sync.py:_getter`）、装饰器函数、鸭子类型框架桩。

注：本想用 multi-agent workflow（14 审计 + 14 对抗校验），agent 跑到一半撞 API 月度
spend limit 全军覆没；改为机械扫描 + 主会话人工校验，对本任务（机械模式识别）等效且更可复现。

## 1. Phase A — 死 API 面删除（最高确定性，先做）

每条已验证 `vrl/ + configs/ + tests/ + docs/` 无真实调用方（tests 中仅有测它自己的用例）：

### A1. `workload_signature` 整个面（~5 文件）

```text
vrl/generation/protocols.py:76        PipelineExecutor.workload_signature 协议方法
vrl/generation/diffusion/executor.py:213   实现（仅 return WorkloadSignature.from_...）
vrl/models/ar/janus_pro/runtime.py:292     实现
vrl/models/ar/nextstep_1/runtime.py:245    实现
vrl/generation/types.py:81                 WorkloadSignature dataclass + from_request/from_request_and_capability
vrl/generation/__init__.py:25,40           re-export
```

全仓没有任何 `.workload_signature(` 调用；`WorkloadSignature` 类型除了这些定义/实现外无消费方。
这是一个从未接线的协议承诺。删除时同步删 2 个只测它的测试。

### A2. `forward_batch_plan`（成对死亡，62 行）

```text
vrl/generation/diffusion/executor.py:620   28 行，零调用方
vrl/generation/ar/executor.py:119          34 行，零调用方
```

引擎路径走的是 chunk/plan 接口；这个 batch 变体两个家族都没人调。跨家族**成对删除**，
不破坏 sibling 一致性。

### A3. `describe()` 跨家族死方法（6 处，统一删）

```text
vrl/models/diffusion/base.py:37（基类）+ wan_2_1/sd3_5/predict2/predict2_5/anima 各 1 个 override
```

零调用方（docs 里 7 处 "describe" 是普通英语词，非 API 引用，已逐条看过）。
跨家族统一删除 = 保持一致形状。

### A4. 单点死代码（逐条已过 configs/tests 复核）

```text
vrl/trajectory/resolver.py:188,199   resolve_training_view / resolve_loss_unit — 零引用
vrl/models/ar/nextstep_1/model.py:193  trainable_param_count — 零引用
vrl/generation/ar/decode_loop.py:81    mark_finished — 零引用
vrl/rewards/models/nsfw_safety.py:73   probability_batch — 仅自测引用
vrl/models/diffusion/cosmos/predict2_5/model.py:235  sync_previous_policy_adapter — 仅自测引用
```

### A5. 需再核一层才能删（标 verify，不直接进 A）

```text
vrl/trainers/data/samplers.py:63  set_epoch — PyTorch Sampler 鸭子类型惯例，DataLoader/DDP
                                  循环可能隐式调用；确认无 DDP 路径后才可删
vrl/nn/layers/attention/paged.py:177,202  can_reuse / free — paged attention 是上轮审计
                                  明确保留的 protocol 边界；"free" grep 噪声大，需人工核
```

### 明确不删（扫描命中但为框架桩）

```text
vrl/models/diffusion/cosmos/predict2/model.py:125,128 + predict2_5/model.py:37,40
check_text_safety / check_video_safety — cosmos pipeline 鸭子类型 guardrail 接口，
调用方在 diffusers/cosmos pipeline 内部，不在本仓。保留。
```

## 2. Phase B — 转发壳 inline（25 个候选中已剔除假阳性）

纯转发（函数体只有一个调用），单调用方，无装饰器。代表性已核条目：

```text
vrl/generation/ray/launcher.py:401   _device_to_string（2 行）
vrl/rollouts/collector/core.py:247   _device_from_model（2 行）
vrl/rollouts/orchestration/continuous/producer.py:246  _store_batch（2 行）
vrl/rollouts/evaluators/ar/multi_segment_token_logprob.py:220  _segment_enabled（2 行）
vrl/models/diffusion/cosmos/predict2_5/runtime.py:55   _skip_text_encoder_from_spec（2 行）
vrl/scripts/data/videophy_i2v.py:384  _normalize_caption / video_world.py:126 _dl（2 行）
```

**已剔除、严禁 inline 的扫描命中**：

```text
weight_sync.py:102 _getter                  — 返回的 closure，工厂模式本体
scripts/*/train.py _after_bundle_built      — 共享 factory 按名回调的跨家族 hook（2 家族成对）
wan_2_1/runtime.py:371 + predict2/runtime.py:288 _reference_image_for_request
                                            — 跨家族 sibling 对，保形状
trainers/online/trainer.py:109 _resolve_mixed_precision 等 own_occurrences>2 的
                                            — 实际多调用方，不是单调用壳
```

实施口径：逐个打开确认"无 seam 价值"后 inline；任何一个有第二调用方/测试直调/protocol
身份的，当场放弃该条，不强求清零。

## 3. Phase C — 热点文件碎片整理（按文件做，不按符号做）

单调用方 helper 密度最高的文件（个数 = 扫描命中）：

```text
22  vrl/scripts/data/danbooru.py            — god file，用户已否决拆分；只 inline 2-5 行琐碎层
11  vrl/trainers/online/trainer.py
 9  vrl/generation/ray/launcher.py
 9  vrl/rollouts/collector/batch_builder.py
 9  vrl/rollouts/evaluators/ar/multi_segment_token_logprob.py
 9  vrl/scripts/diffusion/cosmos/train.py   — bundle builder 是 entrypoint 接线，多数应保留
 8  vrl/rollouts/orchestration/continuous/producer.py
```

做法：每文件一个独立小 commit，"读全文 → 标注每个 helper 的去留理由 → inline 无理由者"。
预期不是清零，是把"读一个函数要跳 5 次"降为"跳 1-2 次"。trainer.py / launcher.py /
batch_builder.py 优先（核心路径读得最多）。

## 4. Phase D — 薄文件合并（28 个里只动 2 个）

```text
vrl/ray/types.py（19 行，唯一 importer = actor_group.py）→ 合入 actor_group.py
vrl/models/diffusion/common/timestep.py（35 行，仅被自家 __init__ re-export）
    → 先 grep __init__ 出口的真实消费方；若也为零则属 Phase A 死代码，否则合入消费方
```

**其余 26 个全部保留**，理由按 taxonomy：

```text
rewards/functions/*.py（单 importer = registry.py）   — registry 约定，每 reward 一文件，
                                                       上次 flatten 被回滚，严禁动
*/types.py, */base.py（algorithms, trainers/core, rewards, evaluators）— 共享类型/ABC 边界
ray/dependencies.py, ray/lifecycle.py, trajectory/device.py — 多消费方共享层（4-11 importers）
orchestration/continuous/staleness.py — 4 个 importer 的真实共享抽象
scripts/ar/nextstep_1/train.py — YAML entrypoint（configs 字符串引用，扫描器看不见）
```

## 5. Non-Goals（明确不做）

```text
1. 不做 279 个单调用方 helper 的全量 inline——单文件私有 helper 是正常组织方式
   （上轮 SOLID 审计已裁定），只处理 §2 转发壳和 §3 热点文件。
2. 不动 registry/约定式抽象、跨家族 sibling 形状（runner/state/runtime 并行文件）。
3. 不拆 god file（danbooru.py 等——用户已在上轮否决 sub-sprint B）。
4. 不重提上轮审计 Non-Goals：三套 sampling-state dataclass、anima 单模块 bundling、
   trajectory/validation.py、JANUS 架构维度常量。
5. 不做 rename / 文档化 / 新抽象——本 sprint 只有 inline / merge / delete 三种动作。
```

## 6. 验收

```text
每个 Phase 独立 commit；A 先行（零行为风险），B/D 次之，C 按文件分批。
每批后：pytest -q tests/ 全绿；grep 确认被删符号无残留引用。
LOC 预期：A ≈ -200 行死代码；B ≈ -60 行壳；C/D 视逐文件判断，不设指标
（指标驱动的 LOC 清理正是上次被回滚的模式）。
```
