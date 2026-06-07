# SPRINT: Training Script Helper Consolidation

状态：implemented（基于通读 `vrl/scripts/` 下 9 个训练脚本，2026-06-06）。

## 0. 一句话

`vrl/scripts/` 里「一堆只用一次的 helper」大多不是随手拆的——它们是塞进 `OnlineRecipeDefinition` 的策略回调，由共享的 `run_online_recipe` 调用，属于有意架构。真正的问题相反：**同一个回调函数体在多个家族里逐字节复制**，违反「不要手工维护重复」的架构规则；并且 `train_dpo.py` 是唯一没走共享 recipe 的异类。本 sprint 把重复的回调收口到 `vrl/scripts/common/`，让各家族直接引用单一源。

## 1. 背景：为什么看起来「helper 满天飞」

GRPO 路径（wan / cosmos / sd3 / janus_pro / nextstep_1）全部走同一个壳：

```python
await run_online_recipe(
    cfg,
    OnlineRecipeDefinition(
        family="wan_2_1",
        build_bundle=_build_bundle,
        build_replay_bundle=_build_replay_bundle,
        after_bundle_built=_after_bundle_built,
        export_modules_getter=_export_modules,
        ...
    ),
)
```

这些 `_build_bundle` / `_export_modules` 在文件里「只出现一次」，是因为它们是**被传引用**给共享 runner，不是被内联调用。这是 strategy + lazy-import 模式，不是坏味道。

唯一的异类是 `vrl/scripts/diffusion/wan_2_1/train_dpo.py`：DPO 是离线、无 rollout 采集，走不了 `run_online_recipe`，于是把 240 行（runtime 构建、encoder、data、trainer、checkpoint 循环）全摊在 `train_wan_2_1_dpo()` 一个函数里。这是它和别人不一致的**正当**原因，本 sprint 不动它。

## 2. 盘点结论（9 个文件，三个桶）

### 2.1 🟢 该留（justified）——不动

| 类别 | 例子 | 留的理由 |
|---|---|---|
| Lazy-import bundle builder | `_build_bundle` / `_build_replay_bundle`（全家族）| 延迟加载重型 `*.runtime` 模块；签名跨家族统一、可 grep |
| Closure factory | `_build_encoders`（`train_dpo.py:43`）、`_offload_driver_frozen_modules`（sd3）、`_save_checkpoint`（online.py）| 一次绑定状态再返回闭包/参数元组 |
| Public API facade | `factory.py` 的 `build_*_from_cfg` 系列 | 都在 `__all__`、被多处调用 |
| Error-isolation | `_preflight_production_video_reward`、`_write_metric_row`（online.py）；`_i2v_collector_kwargs`（wan，含 manifest 校验）| 独立的 try/except 与诊断语义 |

`_build_encoders` 是个标准例子：它一次性绑定 `latents_mean / latents_std / vae / device / dtype`，返回 `encode_pixels` / `encode_text` 两个闭包传给 `OfflineDPOTrainer`。内联会迫使状态 setup 重复，**保留正确**。

### 2.2 🔴 该去重（shared-extract）——本 sprint 的活儿

| 重复逻辑 | 涉及文件 | 现状 |
|---|---|---|
| `_export_modules`（transformer 版）| wan_2_1 / cosmos / sd3_5 | 逐字节相同 ×3 |
| `_export_modules`（language_model 版）| janus_pro / nextstep_1 | 逐字节相同 ×2 |
| `_configure_trainer` | janus_pro / nextstep_1 | 逐字节相同 ×2 |
| grad-ckpt 段 `_after_bundle_built` | wan / cosmos（sd3 在其上再加 offload）| 核心动作重复，但必须保留 strict/optional 方法策略 |

证据（逐字节相同）：

```python
# wan_2_1/train.py:69-73 == cosmos/train.py:135-139 == sd3_5/train.py:60-64
def _export_modules(bundle, cfg):
    transformer = bundle.model.transformer
    if bool(cfg.model.use_lora) and hasattr(transformer, "save_pretrained"):
        return {LORA_WEIGHTS_NAME: transformer}
    return None

# janus_pro/train.py:84-87 == nextstep_1/train.py:56-59
def _export_modules(bundle, cfg):
    if bool(cfg.model.use_lora):
        return {LORA_WEIGHTS_NAME: bundle.model.language_model}
    return None

# janus_pro/train.py:79-81 == nextstep_1/train.py:51-53
def _configure_trainer(cfg, trainer_config):
    trainer_config.n = int(cfg.rollout.n_samples_per_prompt)
    trainer_config.rollout_batch_size = int(cfg.rollout.rollout_batch_size)
```

### 2.3 🟡 纯「给段落起名」——可内联（次要，收益小，本 sprint 不强制）

- `train.py`：`resolve_train_target`、`_import_callable`
- `online.py`：`_apply_precision_policy`、`_prepare_metrics_csv`
- `factory.py`：`_cfg_select`、`_with_resolved_reward_runtime_kwargs`
- `sd3_5`：`_extract_frozen_offload_config`

这些是单次调用、直线代码、无闭包/无独立错误处理，只为「命名一个段落」，加了一次 jump-to-definition 跳转却没买到复用。可内联为「注释 + 内联块」，但优先级低于 2.2。

## 3. 架构张力与裁决

AGENTS.md 同时有两条规则，在这里正面相撞：

- 「保留跨家族一致的薄函数，利于 grep / 调试」（cross-family consistency）
- 「不要手工维护重复，应从单一源派生」（no hand-maintained duplication）

**裁决**：一致性保护的是**统一的签名/形状**，不是**复制的函数体**。把逐字节相同的 body 抽成一个共享函数、各家族用同名引用，能**同时**拿到一致性（同名、可 grep）和去重（单一源）。因此走 extract，不走「内联回各家族」也不走「保留多份副本」。

风险：byte-identical 的 body 在哪天 LoRA 导出多一个模块时，需要改 5 个文件且容易漏改——这正是 AGENTS.md「duplicated constant 会随源类型新增字段而 rot」要防的失效模式。

## 4. 计划

### Phase 1 — 抽共享回调到 `vrl/scripts/common/online.py`

```python
# vrl/scripts/common/online.py
from __future__ import annotations
from typing import Any
from omegaconf import DictConfig
from vrl.trainers.checkpointing import LORA_WEIGHTS_NAME


def export_transformer_lora(bundle: Any, cfg: DictConfig) -> dict[str, Any] | None:
    """Diffusion families: export transformer LoRA weights when use_lora is on."""
    transformer = bundle.model.transformer
    if bool(cfg.model.use_lora) and hasattr(transformer, "save_pretrained"):
        return {LORA_WEIGHTS_NAME: transformer}
    return None


def export_language_model_lora(bundle: Any, cfg: DictConfig) -> dict[str, Any] | None:
    """AR families: export language-model LoRA weights when use_lora is on."""
    if bool(cfg.model.use_lora):
        return {LORA_WEIGHTS_NAME: bundle.model.language_model}
    return None


def configure_ar_rollout(cfg: DictConfig, trainer_config: Any) -> None:
    """AR families: bind n / rollout_batch_size from rollout config."""
    trainer_config.n = int(cfg.rollout.n_samples_per_prompt)
    trainer_config.rollout_batch_size = int(cfg.rollout.rollout_batch_size)


def enable_transformer_gradient_checkpointing(bundle: Any, cfg: DictConfig) -> None:
    """Diffusion families: enable grad-ckpt on the transformer when configured."""
    transformer = bundle.model.transformer
    ...
```

### Phase 2 — 各家族改引用

- wan_2_1 / cosmos / sd3_5：`export_modules_getter=export_transformer_lora`，删掉本地 `_export_modules`。
- janus_pro / nextstep_1：`export_modules_getter=export_language_model_lora`、`configure_trainer=configure_ar_rollout`，删掉本地两份。
- wan / cosmos / sd3 的 `_after_bundle_built` grad-ckpt 段改为复用 `enable_transformer_gradient_checkpointing`；wan/sd3 保持缺方法时报错，cosmos 保持缺方法时跳过；sd3 仍追加自己的 frozen-module offload。

### Phase 3 — 验证

- `ruff check` 覆盖 5 个训练入口和 `vrl/scripts/common/online.py`。
- 相关 scripts/trainers 用例通过。
- grep 确认无残留本地 `_export_modules` / `_configure_trainer` 重复定义。

### 非目标（Non-goals）

- 不动 🟢 桶（lazy-import builder、closure factory、public API facade、error-isolation）。
- 不抽 wan/cosmos 的 per-sample 参考图归一化。本段只是近似重复，wan 有 `data_root / allow_absolute / global_reference`，cosmos 有 `reference_mode` 和 `rollout_batch_size == 1` 约束，合并会把两个家族语义糊在一起。
- 不重写 `train_dpo.py` 的单体结构——离线 DPO 不适配 `run_online_recipe`，是正当的不一致。
- 2.3 的纯起名 helper 内联留作可选清理，不阻塞本 sprint。

## 5. 参考

- `vrl/scripts/diffusion/{wan_2_1,cosmos,sd3_5}/train.py`
- `vrl/scripts/ar/{janus_pro,nextstep_1}/train.py`
- `vrl/scripts/common/{online.py,factory.py,types.py}`
- `vrl/scripts/diffusion/wan_2_1/train_dpo.py:43`（`_build_encoders`，保留示例）
