# SPRINT: Training Correctness & Hardening

## 0. Core Decision

本 sprint 不加新功能。目标是把现有训练栈里几处**会悄悄训歪、且没有测试兜底**的正确性隐患修掉，补上守门的 CI 和缺失的核心数学测试，再顺手清掉两处有测试兜底的低风险重复。

判断依据：核心 CEA 闭环、checkpoint、weight sync、continuous 编排都已有高质量数值断言测试，大文件大多是真实领域复杂度。真正的风险集中在三类——

```text
P0  训练正确性隐患（无测试，错了不报警）
P1  缺 CI + 四块训练数学零覆盖
P2  局部重复（有测试兜底，不紧急）
```

明确边界：本 sprint **不重构** Cosmos 三变体主体、家族注册表、配置分层、scripts 入口模板——经审查这些是刻意的跨家族一致性，按 AGENTS.md "consistency over cleanup" 原则保留。

## 1. 为什么做（每条都已核实）

### P0-1：GRPO 优势归一化两条路径不一致 + `eps` 过小 🔴

两条优势计算路径对 std 下限处理不一致：

```python
# 张量路径 vrl/algorithms/grpo/continuous.py:74
denom = torch.clamp(std, min=cfg.eps)          # cfg.eps 默认 1e-8

# tracker 路径 vrl/trainers/online/stat_tracking.py:85
std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4
```

`GRPOConfig.eps` 默认 `1e-8`（`continuous.py:19`），而 tracker 路径用 `+1e-4`。当一个 prompt group 内 reward 几乎相等（std≈0）时，张量路径会把微小噪声放大约 `1e8` 倍，再被 `adv_clip_max=5` 截断 → 大量饱和的 ±5 优势 = 对噪声做满幅梯度。同一份 reward 走不同路径得到不同优势，行为不可复现。

### P0-2：DiffusionNFT 整条 loss 数学零测试 🔴

`vrl/algorithms/diffusion_nft.py`（305 行）的 `compute_loss` 走 `uses_evaluator=False` 分支（`vrl/trainers/online/trainer.py:549`），与所有已测的 GRPO 是**完全不同的代码路径**。loss 涉及 previous-policy adapter forward 和 reference forward，符号错或归一化错会**反向训练且无人发现**。三个生产配置在用它：

```text
configs/experiment/diffusion/cosmos_predict2_5/online_nft_video_reward.yaml
outputs/wan_1_3b_nft_4step/...
configs/.../recipe 中的 diffusion_nft.yaml
```

### P0-3：DPO 梯度累积 off-by-one（仅 `gradient_accumulation_steps>1`）🟠

```python
# vrl/trainers/offline/dpo.py:349-358
loss_scaled = loss / max(1, cfg.gradient_accumulation_steps)
loss_scaled.backward()
if (self.global_step + 1) % cfg.gradient_accumulation_steps == 0:
    optimizer.step(); zero_grad()
self.global_step += 1   # 全局单调递增，从不归零
```

`global_step` 跨整个训练单调递增。resume checkpoint 后若它不是 `gradient_accumulation_steps` 的整数倍，累积边界会错位，`zero_grad` 不在预期点触发，梯度跨多个累积窗口叠加。默认 `N=1` 时无害。

### P0-4：weight_sync policy version 无锁 🟠

```python
# vrl/trainers/weight_sync.py:58-63
await self.runtime.update_weights(state, policy_version)
self._next_policy_version = policy_version + 1   # 自增在 await 之后
```

当前 OnlineTrainer 的 `after_train_step` 是单协程顺序 await，不会触发。但读取+自增没有锁保护：一旦未来出现并发 push（两个训练步重叠），两次读到同一 version 再各自 +1 → 版本号重复/丢失。属于脆弱模式，应在引入并发前加固。

### P1-1：没有 CI 🔴（流程层最大风险）

无 `.github/workflows`、无 `conftest.py`、无 Makefile/tox/nox。`pytest --collect-only` 健康（526 tests，0 错误，2.26s），测试质量高（真 `backward()`、断言精确数值），但**完全不在提交时强制执行**。回归可以随便落地。

### P1-2：四块训练数学零覆盖

| 模块 | 未测内容 | 风险 |
| --- | --- | --- |
| `diffusion_nft.py` | 整条 loss（见 P0-2） | 反向训练不报警 |
| `grpo/continuous.py:102-141` | clipped surrogate（所有扩散 GRPO 的核心 loss） | `torch.maximum(unclipped, clipped)` 负优势侧的经典符号坑 |
| `online/ema.py` | decay 更新（只测了存取，没测衰减数学） | EMA 权重错误，eval 失真 |
| `orchestration/strict_on_policy.py` | `collect` 抛异常时 driver model 是否恢复 | 异常后跑在半初始化模型上 |

### P2：局部重复（有测试兜底）

- `restore_eval_state` 在两个 Cosmos `ReplayModel` 中逐行复制父类，仅 `self.pipeline.scheduler` vs `self.scheduler` 一处不同（`predict2/model.py:472` vs `:630`，`predict2_5/model.py:429` vs `:640`）；而父类已有 `scheduler` property。
- 三个 replay-tensor helper（`_align_replay_tensor` / `_replay_tensor` / `_shared_replay_tensor`）在 predict2/predict2_5 字节级相同，anima 已开始 drift。
- `ray/resources.py` rollout/reward 设备仲裁近重复 ~60-80 行（`:420-474` vs `:608-650`；`:477-497` vs `:693+`）。

## 2. 分阶段方案

### Phase 1 — CI 守门（先做，零代码风险，守住后续所有改动）

新增 `.github/workflows/ci.yml`，CPU runner，跑：

```yaml
- ruff check . && ruff format --check .
- pytest -m "not e2e"     # e2e 已被 WM_RUN_REAL_MODEL_TESTS 门控，CI 不跑
```

完成标准：PR 触发，红线能挡住 lint/test 回归。这一步先落地，后面每个 Phase 的测试都自动进 CI。

### Phase 2 — P0 正确性修复

1. **GRPO eps（P0-1）**：`GRPOConfig.eps` 默认 `1e-8 → 1e-4`；统一两条路径的 std 下限语义（建议都用 `clamp(std, min=eps)`，tracker 路径的 `+1e-4` 改为同一 clamp helper，消除 additive vs clamp 的差异）；clamp 到 `adv_clip_max` 的位置统一。
2. **DPO 累积（P0-3）**：引入独立 `micro_step` 计数器，`optimizer.step()` 后归零；不再用全局 `global_step` 判断累积边界。
3. **weight_sync 版本（P0-4）**：`push` 内加 `asyncio.Lock`，或先自增 `_next_policy_version` 再 `await update_weights`，保证读取+自增原子。

### Phase 3 — P1 测试补齐（每条都断言真实数值，不是 "不崩"）

1. **DiffusionNFT loss（P0-2 的兜底）**：小 fake transformer 暴露 adapter/reference forward，断言 hand-computed 输入的 loss 标量；`nft_beta<=0` 抛错；`after_optimizer_step` 按文档更新 previous-policy adapter。
2. **continuous-GRPO surrogate**：`ratio==1 → loss == -advantage.mean()`；正/负优势选对 clip 分支；`clip_fraction==1.0` 当所有 ratio 超 `eps_clip`（对齐 `test_grpo_token.py:103`）；`approx_kl == 0.5*mean((lp-old_lp)^2)`。
3. **EMA decay**：N 次 `step()` 后 `ema_parameters` 等于解析 EMA 值；`update_step_interval` 边界外跳过更新；`copy_ema_to(store_temp=True)` + `copy_temp_to` 精确还原原参数。
4. **strict 编排失败路径**：collector 抛异常时 `restore_driver_model_after_rollout` 和 `release_rollout_runtime_memory` 仍执行；weight sync 发生在 `after_train_step` 而非 collect 前。

### Phase 4 — P2 去重（有测试兜底，最后做）

1. 把父类 `restore_eval_state` 改用 `self.scheduler`（已是 property），删掉两个 Cosmos `ReplayModel` 的重复 override。
2. 三个 replay-tensor helper 提到 `vrl/models/diffusion/common/`，reconcile anima 的 drift 版本。
3. `ray/resources.py` 抽共享 "pool → requested → 切片 → overlap fallback" helper（参数化 role 名 + 排除集），rollout 与 reward 非共享分支都调它；删掉 rollout 专用的 `_requested_rollout_gpu_count`，统一用 `_requested_role_gpu_count`。reward 的 `share_with_rollout` 分支保留。

## 3. 关键文件

| 文件 | 角色 |
| --- | --- |
| `vrl/algorithms/grpo/continuous.py:19,74` | GRPOConfig.eps + 张量路径 clamp（P0-1） |
| `vrl/trainers/online/stat_tracking.py:85` | tracker 路径 std 下限（P0-1） |
| `vrl/algorithms/diffusion_nft.py` | NFT loss（P0-2 / P1 测试） |
| `vrl/trainers/online/trainer.py:549` | `uses_evaluator=False` 即 NFT 入口 |
| `vrl/trainers/offline/dpo.py:349-358` | DPO 累积（P0-3） |
| `vrl/trainers/weight_sync.py:58-63` | policy version（P0-4） |
| `vrl/trainers/online/ema.py` | EMA decay（P1 测试） |
| `vrl/rollouts/orchestration/strict_on_policy.py` | 失败路径（P1 测试） |
| `vrl/models/diffusion/cosmos/{predict2,predict2_5}/model.py` | restore_eval_state + replay helper（P2） |
| `vrl/models/diffusion/common/` | replay helper 落点（P2） |
| `vrl/ray/resources.py:420-474,608-650` | 设备仲裁去重（P2） |
| `.github/workflows/ci.yml` | 新增（Phase 1） |

## 4. 验证矩阵

| 项 | 验证方式 |
| --- | --- |
| Phase 1 | PR 上 CI 跑绿；故意引入 lint 错误能被挡 |
| P0-1 | 新单测：全相等 reward 的 group 优势为 0（不是 ±5 饱和）；两条路径同输入同输出 |
| P0-2 | 新单测断言 hand-computed loss 标量；现有 `pytest -m "not e2e"` 全绿 |
| P0-3 | 新单测：`gradient_accumulation_steps=3` 下 step/zero_grad 在正确边界；resume 后边界不错位 |
| P0-4 | 新单测：并发两次 `push` 得到不同递增 version |
| P1 测试 | 四块各自的数值断言通过 |
| P2 | 删除/合并后 `pytest tests/models tests/ray -q` 全绿；`ruff check` 通过 |
| 全局回归 | `WM_RUN_REAL_MODEL_TESTS=1` 在 GPU box 上跑 `tests/e2e/test_real_checkpoint_rl.py` 至少 sd3_5 + 一个 NFT 配置，确认 trainable 权重仍变化、loss 有限、grad_norm 正 |

## 5. 非目标

- 不重构 Cosmos 三变体主体（V2W/T2W/anime backbone 真实不同，强抽基类会变成 `if family==` god-base）。
- 不动家族注册表 `RolloutFamilyEntry`（import 路径存字符串是跨进程懒加载边界）。
- 不动 84 个 YAML 的分层组合、scripts 各家族 train.py 模板（刻意一致性）。
- 不拆 `danbooru.py`（340 行是隔离词表，有测试，非热路径，缓做）。
- 不拆 `executor.py:run_denoise_steps` / `planner.py`（本质复杂的热路径状态机）。
- 不做 FSDP 多 GPU 测试（需多卡环境，超出本 sprint）。

## 6. References

- 审查来源：本仓库 `vrl/` 四维度只读审查（abstraction/correctness/testing/maintainability），2026-05-30。
- 相关 sprint：`SPRINT_multi_gpu_training.md`（weight sync 契约会受 P0-4 影响）、`SPRINT_continuous_rollout_from_slime.md`（policy version 语义）。
- `AGENTS.md` — Architecture Hygiene（consistency over cleanup 原则，本 sprint 的 P2 边界依据）。
