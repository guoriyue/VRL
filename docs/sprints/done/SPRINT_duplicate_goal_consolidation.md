# SPRINT: Duplicate-goal consolidation (proposed)

状态：implemented（2026-06-10，本仓 5 个 commit：S1-S5/S7 + Phase A/B/D；S6 撤回、S8 未做——taste 项留给 maintainer）。与 `SPRINT_small_function_consolidation.md`
（小函数/死代码镜头）互补，本篇是**重复目标函数**镜头：同一件事在多处有各自的实现。

## 0. 方法与覆盖（诚实声明）

- 两次 multi-agent workflow（14 全覆盖审计 + 5 重复判官 + 2 对抗校验）均因 API 月度
  spend limit 全员失败（共烧 ~93 万 subagent tokens 无产出）。
- 实际方法：归一化 AST 结构哈希扫描全部 264 个文件（机械全覆盖，>=4 行函数、跨文件同构）
  + 同名函数 3+ 文件分布表 + **每个重复簇由主会话逐一 diff 当前 HEAD 函数体、查 git
  分化史、核落点**。"逐文件人工读完 264 个"没有发生；人工深读覆盖了全部 20 个结构重复簇
  + 全部死代码候选 + 抽查文件，机械层覆盖 100%。
- 判决基线：同名 ≠ 重复（ABC 表面跨家族实现是有意形状）；**同体才是重复**；同体但属
  家族接线（constructor wiring）仍判保留。上轮 SOLID 审计 sub-sprint C 已下沉
  apply_lora/_require_tensor，本篇是它的续集，遵守同一套 Non-Goals。

## 1. 判决表 — sink（建议下沉，按收益排序）

### S1. Replay 模型 raise 桩 ×20（~170 行，最大单项）

每个家族的 ReplayModel 为 ABC 要求的 rollout-only 方法手写同构 raise 桩：

```text
encode_prompt / prepare_sampling / decode_latents（diffusion 7 个 replay 类）
decode_image_tokens（AR 2 个 replay 类）
例：vrl/models/diffusion/cosmos/predict2/model.py:601,610 + 5 个家族同构
    raise RuntimeError("CosmosPredict2ReplayModel cannot encode prompts")
```

落点：`vrl/models/diffusion/base.py` 增加 replay 桩基类（或 base 上的默认实现），
报错消息用 `type(self).__name__` 保留各家族类名。跨家族**统一**下沉 = 保形状。

### S2. `_require_trainable_modules` ×2 — 已开始语义漂移（最强理由）

```text
vrl/trainers/weight_sync.py:158     Mapping 版："must be a non-empty mapping"
vrl/trainers/checkpointing.py:469   dict 版（更严）："must be a non-empty dict"
```

复制品已经分叉——这正是复制的腐烂模式。统一为 weight_sync 的 Mapping 版（更宽容、
与 flatten_trainable_module_state 的签名一致），checkpointing 改 import。

### S3. predict2 ↔ wan I2V 参考图条件链 ×2（~35 行）

```text
encode_prompt_for_chunk  predict2/runtime.py:220 ↔ wan_2_1/runtime.py:298（18 行，仅 docstring 异）
build_prepare_kwargs     predict2/runtime.py:270 ↔ wan_2_1/runtime.py:353（17 行，全同）
```

落点：`vrl/models/diffusion/common/`（reference-conditioning chunk helpers）。
风险注记：V2W 与 I2V 上游语义不同，未来可能分化；下沉后家族仍可 override。

### S4. `replay_forward` predict2_5 ↔ anima ×2（27 行全同）

```text
vrl/models/diffusion/cosmos/predict2_5/model.py:480 ↔ cosmos/anima/model.py:388
```

anima 本就是 predict2 家族变体；落点 cosmos 层共享（不违反 anima 单模块 Non-Goal——
helper 放 cosmos 公共处，anima 文件内只剩委托）。

### S5. `_trajectory_role_value` ×2（7 行全同，跨层复制）

```text
vrl/models/ar/janus_pro/model.py:69 ↔ vrl/rollouts/evaluators/ar/multi_segment_token_logprob.py:171
```

落点：`vrl/trajectory/`（role 枚举的所在层）。

### S6. ocr / nsfw_safety 绕过已有 reward 基类 —— 实施时再判为不做（2026-06-10）

```text
vrl/rewards/models/base.py:20 TorchRewardModel（_load/_ensure_loaded/score_media 契约）
aesthetic/pickscore 正确继承；ocr.py:34 与 nsfw_safety.py:28 独立类，自写 _ensure_loaded
```

细读后撤回：OCR 的 __call__ 依赖 artifact.metadata["target_text"]、NSFW 有批量
score_request + 可注入 scorer——两者是 artifact 级契约，与 score_media(media, prompt)
的逐媒体契约不同。强行继承只共享 3 行 _ensure_loaded，却要扭曲构造器与调用面，
属"为对齐而对齐"。保留现状。

### S7. `torch_compile_transformer` ×4（4 行同构）

```text
predict2/model.py:163 ↔ wan_2_1/model.py:148（diff 确认全同）+ sd3_5:172 / anima:170 同名
```

落点：base.py 默认实现（transformer 属性名统一时）。实施前先 diff sd3_5/anima 两份。

### S8.（可选）backbone runner 恒等 hook ×4（~40 行）

```text
postprocess_branch / finalize_noise_pred 恒等桩：wan runner:47,56,129,138 /
sd3_5 runner:74,83 / predict2_5 runner:63 —— DiffusionBackboneRunner 是 Protocol
（common/backbone.py:53），不需要 hook 的家族手写 identity 实现
```

落点：带默认实现的 RunnerBase，家族只 override 真 hook（predict2 有真实现）。
属 taste 项：上轮审计判"薄 protocol hook 可接受"，做不做由你定。

## 2. 判决表 — keep（同名/同构但保留，已逐条核）

```text
scripts _build_bundle/_build_replay_bundle ×4 — lazy-import 回调边界（OnlineRecipeDefinition
    按名注入），跨家族成对形状，保留
runtime/runner __init__ 同构 ×3/×2 — constructor 接线，家族属性各异同形，保留
from_build ×5 / export_batch_context ×5（其中 wan↔sd3_5 两份同体）— 5 家族 ABC 形状，
    只动其中 2 个会破对称，保留
encode_prompt/prepare_sampling/forward_step/restore_eval_state/decode_latents 同名 ×6
    — ABC 表面（rollout 正体实现各异），保留；只有 replay 桩（S1）下沉
disable_adapter AR ×2（4 行）/ _dtype_label ×2（4 行）/ max_peak_memory_mb ×2 —
    过小，为琐事耦合层级不值得，保留
compute_loss/evaluate/score_batch/shutdown 同名族 — protocol/ABC 表面，保留
rewards/functions/*.py registry 单文件 — 约定，严禁动（上次 flatten 被回滚）
```

## 3. 对上一份 sprint 的修正（自我质询结果）

```text
A1 workload_signature / A2 forward_batch_plan / A3 describe() — 复核通过，维持删除主张
A4 修正 1：resolve_training_view/resolve_loss_unit 只能删 trajectory/resolver.py:188,199
    的公共壳；私有 _resolve_loss_unit(:207) 有 2 个活调用方（:192,:205），保留
A4 修正 2：trainable_param_count 降级——janus_pro/model.py:309 的日志字符串承诺
    "will be reported by trainable_param_count()"，且 janus/nextstep 是跨家族对；
    要么成对删 + 改日志文案，要么保留，不做单边删除
```

（两条已同步进 `SPRINT_small_function_consolidation.md`。）

## 4. Non-Goals

```text
不为 <6 行的琐碎重复建共享层（_dtype_label 类）
不动 ABC/protocol 同名表面、registry 约定、constructor 同形接线
不重提上轮 SOLID 审计 Non-Goals（god-file、三套 sampling-state、anima 单模块等）
不新建抽象层——S1-S8 的落点全部是已存在的 base/common 文件
```

## 5. 实施顺序与验收

```text
顺序：S2（已漂移，最急）→ S1（最大量）→ S5/S6 → S3/S4 → S7 → S8（可选）
每项独立 commit；diff 函数体逐字节核对后替换为委托/继承；pytest -q tests/ 全绿；
S1 验收：grep "ReplayModel cannot" 只剩 base 一处定义。
预期净删 ~300 行复制体。
```
