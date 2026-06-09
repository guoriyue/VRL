# SPRINT: FP16 训练梯度缩放（GradScaler）

状态：implemented（2026-06-08）。GradScaler 已接入 OnlineTrainer 的裸 backward/step 路径，FP16 现在是一条安全、可验收的训练路径（scaler 创建条件、loss 放大、unscale-before-clip、跳坏步不传导 EMA/adapter、checkpoint 存取全部落地）。验收 gate G1–G4 有单测覆盖（`tests/trainers/online/test_grad_scaler.py` + `test_state_restore.py`），G5 smoke 为一次性产物按需跑。前置动作（所有 experiment config 统一默认 `precision: bf16`）见「背景」。

本 sprint 的目标：**当用户确实需要 FP16 训练时，给 OnlineTrainer 的裸 backward/step 路径补上 GradScaler，让 FP16 成为一条安全可验收的训练路径**，而不是原先"FP16 前向 + 裸反向、无任何下溢保护"的状态。

本 sprint 只覆盖 **训练侧（trainer replay backward）** 的 FP16 安全性。它与 `SPRINT_low_precision_rollout_production.md` 是两件事：那个 sprint 解决 **rollout/replay forward precision 对齐**（importance ratio parity），本 sprint 解决 **FP16 反向传播的梯度下溢**。两者正交。

## 0. 一句话

FP16 指数位只有 5 bit，最小可表示正数约 `6e-8`，反向传播中小于此值的梯度直接下溢成 0；GradScaler 把 loss 放大 `S` 倍把梯度顶进可表示区间、step 前 unscale 还原、并按 inf/nan 动态调 `S` 跳过坏步。bf16 指数位 8 bit（范围同 fp32）不下溢，所以不需要 scaler——这就是我们默认 bf16 的原因。FP16 只剩一个合理场景：**Volta/Turing（V100、T4、20 系）等不支持 bf16 的硬件**。

## 1. 已知事实（代码证据）

当前 FP16 前向已通过 autocast 实现，但反向完全没有 scaler：

- `vrl/trainers/online/trainer.py:122` — `_get_autocast` 用 `torch.amp.autocast(dtype=fp16/bf16)`，只 cast 前向算子，不改参数 storage dtype。
- `vrl/trainers/online/trainer.py:295-299` — `_backward`：无 accelerator 时直接 `loss.backward()`，**无 scale**。
- `vrl/trainers/online/trainer.py:301-320` — `_clip_and_step`：`nn.utils.clip_grad_norm_` + `optimizer.step()`，**无 unscale**。`max_norm == 0` 分支还手动对 `p.grad` 求平方和算诊断范数。
- `vrl/scripts/common/online.py:222` — online GRPO 构造 `OnlineTrainer` **不传 accelerator** → `self.accelerator is None` → 走上面这条裸路径。
- 对照：`vrl/scripts/diffusion/wan_2_1/train_dpo.py:199` 走 Accelerate（`mixed_precision=...`），Accelerate 内部自管 scaler；但 wan 用的是 bf16，本来也不需要。

结论：**online GRPO + FP16 = 当前唯一的不安全路径**，且现已无 config 使用（全部默认 bf16），所以可以安全地把它作为独立 sprint 推进，不阻塞主线。

## 2. 背景：已完成的前置动作

为了让"默认安全"立即生效，已把所有 experiment config 统一为 bf16：

| config | 改动前 | 改动后 |
|---|---|---|
| `sd3_5/online_grpo_ocr.yaml` | fp16 | bf16 |
| `sd3_5/online_grpo_ocr_crossnode_debug.yaml` | fp16 | bf16 |
| `sd3_5/online_grpo_pickscore.yaml` | fp32 | bf16 |
| `sd3_5/online_grpo_geneval.yaml` | fp32 | bf16 |

base 默认 `configs/base/actor.yaml:2` 本就是 `precision: bf16`。验证：`grep -rn "precision:" configs/` 全部为 bf16；`tests/config/test_load_all_experiments.py` + `tests/config/test_precision.py` 共 66 passed。

## 3. 目标 / 非目标

**目标**

- FP16 训练（`precision: fp16`、cuda、无 accelerator）时，反向链路自动启用 GradScaler，训练不出 NaN、梯度有限、收敛行为接近 bf16/fp32。
- scaler 状态随 checkpoint 存取，resume 后 scale 连续。
- EMA / `after_optimizer_step` 只在 optimizer **真正 step 了**的步上触发（scaler 可能跳过坏步）。

**非目标**

- 不动 accelerator 路径——Accelerate 已自管 scaler，重复包装会双重缩放。
- 不碰 bf16 / fp32 路径——scaler 对它们必须是 disabled 的 no-op。
- 不引入 fp8 / Transformer Engine。

## 4. 设计

### 4.1 scaler 的创建与启用条件

在 `OnlineTrainer.__init__` 创建一个 scaler，**仅** 在 `precision==fp16 且 device.type=="cuda" 且 accelerator is None` 时 enabled：

```python
self._grad_scaler = torch.amp.GradScaler(
    device.type,
    enabled=(
        self.accelerator is None
        and _resolve_mixed_precision(config) == "fp16"
        and device.type == "cuda"
    ),
)
```

`enabled=False` 时 GradScaler 的所有方法都是直通 no-op，所以 bf16/fp32 路径行为不变，无需分支判断。

### 4.2 `_backward`

无 accelerator 分支改为：

```python
self._grad_scaler.scale(loss).backward()
```

### 4.3 `_clip_and_step`（关键：unscale 必须在 clip 之前）

```python
self._grad_scaler.unscale_(optimizer)        # 先还原梯度真实尺度
grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_norm)  # 再按真实范数裁剪
stepped = self._grad_scaler.step(optimizer)  # 内部检测 inf/nan，坏步返回 None 并跳过
self._grad_scaler.update()
optimizer.zero_grad()
```

- **顺序硬约束**：若不先 `unscale_`，`clip_grad_norm_` 裁的是被放大 `S` 倍的范数，`max_norm` 完全失效。这是 scaler 接入最容易写错的点。
- `max_norm == 0`（不裁剪）分支：当前手动对 `p.grad` 求范数做诊断——这些 grad 仍是 scaled 的，必须先 `unscale_` 再算，否则诊断范数被放大 `S` 倍。
- `step()` 返回值（或 scale 是否下降）用于判断这步是否真的更新了权重，传给上层控制 EMA。

### 4.4 跳过坏步的传导

`_clip_and_step` 返回 `stepped: bool`（或把它并入现有返回）。调用处（`trainer.py:703` 附近的 `after_optimizer_step` / EMA 更新）只在 `stepped` 为真时执行——否则 EMA 会把一次未发生的更新算进去。

### 4.5 checkpoint

- save：在 trainer 的 state dict 里加 `"grad_scaler": self._grad_scaler.state_dict()`。
- load：`self._grad_scaler.load_state_dict(state["grad_scaler"])`，与现有 optimizer/ema 加载（`trainer.py:838-868`）放在一起；非 strict 缺失时 warning 跳过（bf16→fp16 切换或老 checkpoint 没有该字段）。

## 5. 验收 Gate

- **G1 单测 - 启用矩阵**：✅ `test_needs_grad_scaler_matrix`——`_needs_grad_scaler` 仅 fp16+cuda+无accelerator 为 True；fp16+cpu / bf16 / fp32 / 有 accelerator 为 False。
- **G2 单测 - 顺序**：✅ `test_unscale_runs_before_clip`——fake scaler + patch `clip_grad_norm_`，断言 `unscale_` 在 clip 之前。
- **G3 单测 - resume**：✅ `test_state_restore.py`——scaler `state_dict` round-trip。
- **G4 单测 - 跳步**：✅ 两层——`test_clip_and_step_reports_skipped`（scaler 跳步时 `_clip_and_step` 返回 `stepped=False`）+ `test_skipped_step_does_not_update_ema_or_adapter`（集成：skip 时 EMA.step / after_optimizer_step 未触发、global_step 仍 +1；对照 `test_applied_step_updates_ema_and_adapter`）。
- **G5 smoke**：（pending，一次性）在强制 `precision: fp16` 的 SD3.5 OCR recipe 上跑数十步，断言 loss/grad_norm 全程 finite、有权重更新。需 cuda+真实 GradScaler；记录结论后删除，命名 `*_smoke`。

## 6. 风险

- **双重缩放**：若误在 accelerator 路径也启用本 scaler → 梯度被缩放两次。靠 4.1 的 `accelerator is None` 条件杜绝，G1 覆盖。
- **诊断范数失真**：忘记在 `max_norm==0` 分支先 unscale → 报告的 grad_norm 被放大 `S` 倍，误导调参。G2/相关断言覆盖。
- **FP16 仍可能溢出**：scaler 解决下溢，但 attention 分数等前向激活仍可能 >65504 溢出成 inf。这属于 autocast/模型层面，超出本 sprint；若出现，记录为后续 follow-up，不在本 sprint 强行修。

## 7. 参考

- `vrl/trainers/online/trainer.py:113-137,295-320,703,838-868`（autocast / backward / clip-step / EMA hook / checkpoint）
- `vrl/scripts/common/online.py:222`（online GRPO 不传 accelerator）
- `vrl/scripts/diffusion/wan_2_1/train_dpo.py:199`（Accelerate 自管 scaler 的对照）
- `docs/sprints/SPRINT_low_precision_rollout_production.md`（正交：FP16 rollout/replay parity）
- PyTorch AMP: https://pytorch.org/docs/stable/amp.html#gradient-scaling
