# SPRINT: Small-function consolidation（执行归档）

状态：**done（2026-07-18 收口）**。2026-06-10 已落地 Phase A/B/D；A5 的两个候选现已
分别裁定为 REMOVE/KEEP，宽泛 Phase C 已退休为 non-goal，不再是本文件的未完成工作。
实施当时 `WorkloadSignature` 因尚有 planner 消费而保留，之后随消费链消失由独立 dead-code
cleanup 删除；`vrl/models/steps/denoise/common/timestep.py` 继续保留，当前由多个 model family
直接消费，属于真实共享边界。

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
vrl/generation/protocols.py:76        GenerationChunkExecutor.workload_signature 协议方法
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
vrl/trajectory/resolver.py:188,199   resolve_training_view / resolve_loss_unit — 只删这两个
                                     公共壳；私有 _resolve_loss_unit(:207) 有活调用方
                                     (:192,:205)，保留（2026-06-10 复核修正）
vrl/generation/ar/decode_loop.py:81    mark_finished — 零引用
vrl/rewards/models/nsfw_safety.py:73   probability_batch — 仅自测引用
历史的 predict2_5 `sync_previous_policy_adapter` 当时仅有自测引用
```

**当前边界修正**：同名能力后来作为 DiffusionNFT 的真实协议重新落地，当前生产消费者在
`vrl/algorithms/diffusion_nft.py`，family 实现在
`vrl/models/families/cosmos/predict2_5/model.py` 与 `vrl/models/families/flux/model.py`。
因此当前结论是 **KEEP**；上面的历史 A4 记录不授权再次删除它。

降级（2026-06-10 复核）：`nextstep_1/model.py:193 trainable_param_count` 不做单边删除——
janus_pro/model.py:309 日志字符串承诺 "will be reported by trainable_param_count()"，
且 janus/nextstep 为跨家族对；要么成对删 + 改日志，要么保留。

### A5. 二次核验结论（已关闭）

```text
REMOVE  DistributedKRepeatSampler.set_epoch
        旧 vrl/trainers/data/samplers.py 后续连同未使用 sampler 整体删除；当前生产、测试和
        config 均无该符号，不再保留一个假想 PyTorch hook。

REMOVE  ARPrefixCachePolicy.can_reuse
        未接线的 prefix-cache policy 后续作为死面删除，当前树无该符号。

KEEP    ARAttentionBackend.free
        vrl/nn/layers/attention/paged.py 的 backend lifecycle protocol；具体 backend 可覆盖，
        默认实现允许无外部资源的 backend 做 no-op。协议边界比少几行 LOC 更重要。
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

这是 2026-06 的候选快照，不是当前删除清单。例如 `_normalize_caption` 当前有多个调用方，
不再满足 single-caller 条件；任何再次清理都必须重读当前函数体与调用图。

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

## 3. Phase C — RETIRED / NON-GOAL

原计划按“单调用 helper 密度”逐文件整理 `danbooru.py`、`trainer.py`、`launcher.py`、
`batch_builder.py` 等热点。该指标本身不能证明 dead semantics，也不能证明 helper 缺少 protocol、
lazy-import、framework-adapter 或跨 family consistency 价值，因此不作为可执行 backlog 保留。

后续工作已经用更窄、更可证的 owner 取代这项宽泛清理：

- `done/SPRINT_helper_passthrough_hygiene.md` 收口了 `online.py` 的具体 DI-by-arg 与 execution
  controller 问题，并明确把其余热点排除在范围外；
- `done/SPRINT_function_organization_audit.md` 对 12 个 package 复核后只找到两个真实组织异味，
  一个修复、一个因 torch-free 边界而 KEEP；
- `done/SPRINT_single_caller_inlines.md` 只执行有调用图与函数体证据的 single-caller inline，
  同时保留 protocol、framework adapter 和跨 family 一致性边界。

因此 Phase C 不移交新的 sprint，也不继续以“热点文件”名义制造 churn。未来若出现具体 smell，
必须以定义、调用方、字符串 registry/getattr 引用和测试证据单独裁定。

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

## 6. 关闭验收

```text
Phase A/B/D：已落地并由后续回归持续覆盖。
A5：set_epoch/can_reuse 已 REMOVE；ARAttentionBackend.free 按 protocol boundary KEEP。
Phase C：明确 RETIRED/NON-GOAL；具体高置信问题由三个后续 done sprint 收口。
不以 LOC、helper 数量或文件长度作为重新打开本 sprint 的条件。
```
