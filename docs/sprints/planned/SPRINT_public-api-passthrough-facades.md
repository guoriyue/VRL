# SPRINT: 语义级公开 API 透传门面（`__all__` 导出）—— 标注而非内联（planned）

状态：未开始（2026-06-20）。

范围：给两个**机械上是纯透传、但经核实是承重的公开 API 门面**的函数补一行 WHY 注释，把「这是有意设计的 API 边界」这一意图显式写在定义处，让未来的审计/读者不再把它们重新 flag 成「透传 wrapper 异味」。**不内联、不删除、不缩 `__all__`、不动有真实逻辑的兄弟函数。** 交付物是注释 + 一条「经审计、决定保留」的记录，不是代码删除。

> 本 sprint 承接一次设计异味审计：两个候选 claim 都被打成 `thin_wrapper_fn`（异味），但对抗复核（verifier）都翻案成「公开 API 边界 / 字符串引用契约」。它们正好卡在「透传 wrapper 异味」与「有文档的公开 API 门面」的边界上 —— 按 AGENTS.md 的卫生准则，**公开 API 门面是 thin function 合法存在的理由之一**，所以正确的修法不是内联，而是把意图写清楚。

## 0. Core Decision（先看这一段）

裁决一个 thin function「该内联还是该保留」的，是 AGENTS.md 的这条准则：

> Keep thin functions/files only when they provide a protocol/interface boundary, **public API facade**, lazy import boundary, framework adapter, test fake/fixture, cross-family consistency, or a shared abstraction that removes real complexity.

两个候选都满足「public API facade」这一档，但满足的**证据形态不同**，所以注释措辞也不同：

1. `extract_anima_replay_runtime_spec`（`vrl/models/diffusion/cosmos/anima/runtime.py:48-60`）—— 它是 **replay 路径 vs full-generation 路径的命名契约**，被 `train.py` 按符号引用、被 e2e 测试**按 dotted-string 路径**引用，且 git 历史证明它曾经做过字段裁剪、未来可能再次分叉。它是一条**有意保持稳定的「分叉缝」（divergence seam）**：现在恰好和 full-generation 抽取一致，但存在的意义是「允许将来独立变化而不动调用方」。
2. `diffusion_sft_loss`（`vrl/algorithms/dpo.py:108-116`）—— 它是**为了对齐参考 DPO 实现的 loss surface 而刻意导出的辅助 loss API**，模块 docstring 和 `__all__` 都把它和 `diffusion_dpo_loss` 成对列出。它不是「顺手起了名的 `mse_loss` 别名」，而是 offline DPO trainer 的辅助 SFT loss 契约。

判定：**两个都留**，各补一行 WHY 注释，把上面的意图固化到定义处。下文逐条给出现状实锤（含 git 史与字符串引用证据）、注释落点、以及「若未来契约引用真的消失 AND 不再可能分叉，才考虑内联」的退出条件。

## 1. 现状实锤

### 1.1 `extract_anima_replay_runtime_spec` —— replay 命名契约 / 字符串引用门面（留）

函数体确实是纯透传，零新增逻辑（`vrl/models/diffusion/cosmos/anima/runtime.py:48-60`）：

```python
def extract_anima_replay_runtime_spec(
    cfg: Any,
    device: Any,
    weight_dtype: Any,
) -> RuntimeBuildSpec:
    """Trainer replay-only Anima runtime spec.

    With the whole ``cfg.model`` block carried wholesale, the replay model only
    reads the artifact paths / scheduler_shift / torch_compile it needs; the
    remaining fields ride along inertly, so no trimming is required.
    """

    return extract_anima_runtime_spec(cfg, device, weight_dtype)
```

它和兄弟函数 `extract_anima_runtime_spec`（`runtime.py:33-45`）当前都最终落到 `extract_runtime_spec(..., task_variant="text_to_image")`，所以**当下行为完全一致**。审计因此把它 flag 成「零增值透传」。但三条证据证明它是承重边界，不能内联：

**证据 A —— 被 `__all__` 导出且经 package `__init__` 再导出。** `runtime.py:260-268`：

```python
__all__ = [
    ...
    "extract_anima_replay_runtime_spec",
    "extract_anima_runtime_spec",
    ...
]
```

`vrl/models/diffusion/cosmos/anima/__init__.py:8,17` 再次导入并列入 `__all__`，把它抬成 family 的公开符号。

**证据 B —— 被 e2e 测试按 dotted-string 路径硬引用（不是普通调用，内联会直接打断 import 字符串）。** `tests/e2e/test_real_checkpoint_rl.py:356-359` 与 `:401-404` 两个 `RealCheckpointCase` 都写死：

```python
replay_runtime_spec_extractor=(
    "vrl.models.diffusion.cosmos.anima.runtime:"
    "extract_anima_replay_runtime_spec"
),
```

这是 replay 路径的**字符串契约**：测试通过 `module:function` 字符串动态解析它，与 `replay_runtime_builder=...:build_anima_replay_runtime_bundle` 成对，明确区分 replay 与 full-generation 两条装配链。内联进 `build_*` 调用方会让这个字符串无处可指。

**证据 C —— git 历史证明它曾经分叉、未来可能再分叉。** commit `571277787`（"Refactor Anima runtime bundle construction"，2026-05-23）把 `runtime.py` 从 182 行精简到约 77 行（`-132/+77`），正是这一轮把 replay 抽取里原本的字段裁剪折叠掉、统一到 `extract_anima_runtime_spec` 上。docstring 里「no trimming is required」记录的就是这次折叠后的现状 —— 它描述的是「曾经裁剪、现在不裁」的演化结果，而非「从来就是别名」。这正是一条 divergence seam：留着这个命名入口，下次 replay 模型需要不同字段集时改这一个函数即可，调用方与测试字符串都不动。

调用方（除测试外）：`vrl/scripts/diffusion/cosmos/train.py:122-127`，在 replay bundle builder 里成对调用：

```python
return build_anima_replay_runtime_bundle(extract_anima_replay_runtime_spec(cfg, device, weight_dtype))
```

另有 `tests/models/interfaces/test_minimal_replay_runtime_wiring.py:331,356` 直接调用它做 replay 装配 wiring 测试。

### 1.2 `diffusion_sft_loss` —— 对齐参考 DPO loss surface 的刻意导出（留）

函数体是 `F.mse_loss` 的薄封装（`vrl/algorithms/dpo.py:108-116`）：

```python
def diffusion_sft_loss(
    model_pred_winner: torch.Tensor,
    target_winner: torch.Tensor,
) -> torch.Tensor:
    """Plain MSE on the winner only — useful as auxiliary regulariser.

    Pass ``model_pred[:B]`` and ``target[:B]`` (the winner halves).
    """
    return F.mse_loss(model_pred_winner.float(), target_winner.float(), reduction="mean")
```

审计 flag 成「`mse_loss` 命名 wrapper」。但模块 docstring 与 `__all__` 都把它和真正含算法复杂度的 `diffusion_dpo_loss` 成对列为公开 loss surface：

模块 docstring（`dpo.py:7-11`）：

```python
DPO is offline preference learning, fundamentally different from GRPO/PPO
(no rollouts, no advantages). This module exposes the pure functional loss
(``diffusion_dpo_loss`` / ``diffusion_sft_loss``) and its config; the offline
DPO trainer calls the loss directly rather than through the online ``Algorithm``
protocol (which is reward/advantage-based and explicitly rejects DPO).
```

`__all__`（`dpo.py:119-123`）成对导出：

```python
__all__ = [
    "DiffusionDPOConfig",
    "diffusion_dpo_loss",
    "diffusion_sft_loss",
]
```

消费方：`vrl/trainers/offline/dpo.py:25` 与 `diffusion_dpo_loss` 一起 import，`:337-339` 在 `cfg.sft_weight > 0`（`DiffusionDPOConfig.sft_weight`，`dpo.py:33`）门控下调用作为辅助 SFT 正则项：

```python
sft_loss_val = diffusion_sft_loss(
    model_pred[:bsz].float(), target[:bsz].float(),
)
loss = loss + cfg.sft_weight * sft_loss_val
```

另有 `tests/algorithms/test_dpo.py:18,169` 对它单独建测。它是「DPO loss 这一对函数」的 public surface 的一员 —— 与 `diffusion_dpo_loss` 同进同出、对齐参考实现（Wallace et al., `dpo.py:3-5`）的 loss API 形态，而非 trainer 内部顺手写的一行 `F.mse_loss`。内联会让 DPO loss 的公开面少掉一半，并打断 `__all__` / docstring 已声明的「成对」契约。

## 2. 落地方案

每个 Phase 独立可提交，互不依赖。

### Phase A. 给 `extract_anima_replay_runtime_spec` 补 WHY 注释（divergence seam）

在 `vrl/models/diffusion/cosmos/anima/runtime.py:48` 定义处补一行注释，点明：这是 replay 路径的命名契约（被 `train.py` 与 e2e replay 测试按符号/字符串引用），与 full-generation 抽取**有意分叉**、曾做过字段裁剪、保持稳定入口以便将来再分叉。建议落在 docstring 末尾或紧邻 `def` 上方，例如：

```python
    # WHY keep this thin pass-through: it is the *named* replay-path contract,
    # referenced by symbol in train.py and by "module:function" string in the
    # e2e replay test (test_real_checkpoint_rl.py). It intentionally diverges
    # from full-generation extraction — it previously trimmed fields
    # (commit 571277787) and may diverge again; the stable entry point lets the
    # replay spec change without touching callers/tests. Do not inline.
    return extract_anima_runtime_spec(cfg, device, weight_dtype)
```

不动函数签名、不动现有 docstring 的现状描述、不动 `__all__`。

### Phase B. 给 `diffusion_sft_loss` 补 WHY 注释（刻意导出的辅助 loss API）

在 `vrl/algorithms/dpo.py:108` 定义处补一行注释，点明：这是为对齐参考 DPO 实现的 loss surface 而**刻意导出**的辅助 loss，与 `diffusion_dpo_loss` 成对（见模块 docstring 与 `__all__`），由 offline DPO trainer 在 `sft_weight > 0` 时调用，不是 incidental 的 `mse_loss` 别名。例如：

```python
    # WHY keep this thin wrapper exported: it is a deliberately exposed
    # auxiliary-loss API paired with diffusion_dpo_loss (see module docstring
    # and __all__) to mirror the reference DPO loss surface for the offline DPO
    # trainer (trainers/offline/dpo.py, gated on sft_weight>0). Not an
    # incidental F.mse_loss alias — do not inline.
    return F.mse_loss(model_pred_winner.float(), target_winner.float(), reduction="mean")
```

不动签名、不动 `__all__`、不动模块 docstring。

### Phase C. 记录 keep-decision（防止下一轮 sweep 重复 flag）

本 sprint 文件即为该记录。两条都登记为「经审计、判为公开 API 门面、决定保留 + 已就地标注」。退出条件（仅满足时才考虑内联，且需 owner 单独签字）：

- `extract_anima_replay_runtime_spec`：未来审计发现 `train.py` 调用 **AND** e2e 字符串引用 **AND** `__all__` 导出全部消失，且 replay 与 full-generation 不再可能分叉时，才内联回 `extract_anima_runtime_spec`。
- `diffusion_sft_loss`：未来 offline DPO trainer 不再调用、`__all__` 不再导出、参考实现对齐不再是目标时，才内联回 `F.mse_loss`。

## 3. 验证（finishing criteria）

- `grep -n "WHY" vrl/models/diffusion/cosmos/anima/runtime.py` 与 `grep -n "WHY" vrl/algorithms/dpo.py` 各命中新增的注释；diff 仅为注释新增，无逻辑/签名变更。
- 引用仍然存活（注释前后各跑一遍确认未误删）：
  - `grep -rn "extract_anima_replay_runtime_spec" vrl/ tests/` 仍命中 `train.py:124,127`、`anima/__init__.py:8,17`、`runtime.py:48,265`、`test_real_checkpoint_rl.py:358,403`、`test_minimal_replay_runtime_wiring.py:312,331,347,356`。
  - `grep -rn "diffusion_sft_loss" vrl/ tests/` 仍命中 `dpo.py:9,108,122`、`trainers/offline/dpo.py:25,337`、`test_dpo.py:18,169`。
- `ruff check vrl/models/diffusion/cosmos/anima/runtime.py vrl/algorithms/dpo.py`（含 format 检查）零报错 —— 纯注释改动不应触发 lint。
- `pytest tests/algorithms/test_dpo.py -q` 全绿（`diffusion_sft_loss` 行为未变）。
- `pytest tests/models/interfaces/test_minimal_replay_runtime_wiring.py -q` 全绿（replay 抽取 wiring 未变）。
- e2e（`tests/e2e/test_real_checkpoint_rl.py`）受真实 checkpoint/CUDA 门控，CI 不必跑；仅靠上面的 `grep` 确认两处 dotted-string 引用未被改动即可。

## 4. 非目标 / Non-Goals

- **不内联任何一个函数。** 把 `extract_anima_replay_runtime_spec` 内联回 `extract_anima_runtime_spec`、或把 `diffusion_sft_loss` 内联回 `F.mse_loss`，会抹掉一个有命名、被外部引用的公开 API 契约 —— 正是本 sprint 反对的修法。
- **不动有真实逻辑的兄弟函数：** `diffusion_dpo_loss`（`dpo.py:36-105`，含 chunk(2)/logsigmoid/implicit_acc 真算法）、`extract_anima_runtime_spec`（`runtime.py:33-45`）、`extract_runtime_spec`（`vrl/models/runtime_config.py`）。
- **不动其他 family 的 `extract_*_runtime_spec`**（如设 `task_variant="t2i"` 的 `extract_sd3_5_runtime_spec` 等）—— 它们设不同的 `task_variant`，是合法的 cross-family uniform shape，不在本主题内。
- **不删、不缩 `__all__` 导出**（`runtime.py:260-268`、`anima/__init__.py`、`dpo.py:119-123`）。
- **不 flag 单调用方私有 `_helper`**（如 `vrl/models/diffusion/cosmos/anima/runtime.py:202` 的 `_resolve_artifact`）—— 单调用方私有 helper 不属本主题。

## References

- `vrl/models/diffusion/cosmos/anima/runtime.py:33-45,48-60,260-268`
- `vrl/models/diffusion/cosmos/anima/__init__.py:8,17`
- `vrl/scripts/diffusion/cosmos/train.py:118-127`
- `tests/e2e/test_real_checkpoint_rl.py:352-405`（replay 字符串契约 `:356-359`、`:401-404`）
- `tests/models/interfaces/test_minimal_replay_runtime_wiring.py:312,331,347,356`
- git: commit `571277787`（"Refactor Anima runtime bundle construction", 2026-05-23, `runtime.py -132/+77`）
- `vrl/algorithms/dpo.py:1-12,33,36-105,108-116,119-123`
- `vrl/trainers/offline/dpo.py:25,330-340`
- `tests/algorithms/test_dpo.py:18,169`
- AGENTS.md「Architecture Hygiene」: thin functions 保留理由含 "public API facade"
