# SPRINT: Denoise budget — fewer steps, no CFG（rollout 性能的最后一个大杠杆）

状态：proposed。接续 `SPRINT_rollout_performance.md`（其 P3 占位的展开）。

## 0. 一句话

rollout 时间 ≈ **步数 × 每步计算量 × 每次 forward 速度**。第三个因子已经吃完
（compile + bf16 + b8 ≈ 2x，见 SPRINT_rollout_performance）；前两个因子从没动过：
训练 rollout 走 10 步、每步开 CFG（条件+无条件 = 2 份计算）。两个都砍 = 理论 4x。
这是单卡 SD3.5 剩下唯一的大杠杆。

**关键区别**：这不是纯工程优化——减步数/关 CFG 会改变生成质量，从而改变 reward 和
RL 训练效果。所以验收标准是 **reward 曲线**，不是 wall clock。

## 1. 已知事实（代码证据）

- 训练 rollout 10 步 + CFG 4.5：`configs/experiment/diffusion/sd3_5/online_grpo_ocr.yaml`
  引用 `/sampling/denoise/10_step_cfg_4_5`（`num_steps: 10, cfg: true, guidance_scale: 4.5`）。
  eval 才是 40 步，与训练无关。
- denoise 占 rollout 约 98%：D0 实测 encode+decode ≈ 1.07s vs denoise ≈ 78.66s
  （SPRINT_rollout_performance §D0）——所以只有打 denoise 才有意义。
- CFG 的代价是真实 2x：`vrl/models/diffusion/sd3_5/runner.py:48` `cfg_mode = "batched_cfg"`，
  条件+无条件拼成双倍 batch 一次 forward。denoise 是 GPU-bound（profiler 证实），
  双倍 batch ≈ 双倍时间。
- no-CFG 预设已存在：`configs/sampling/denoise/10_step_no_cfg.yaml`
  （`cfg: false, guidance_scale: 1.0`），零代码即可实验。
- GRPO 的轨迹长度 = 去噪步数：`sde_step_with_logprob` 每步写一个 log_prob
  （`vrl/generation/diffusion/executor.py` → `buffers.log_probs`）。减步数 = RL 信号密度
  也变，这是比图像质量更深的一层影响，必须用训练曲线验证。
- 性能对照基线：compiled b8（`SPRINT_rollout_performance` 的 batching decision）。

## 2. 两个旋钮（不预设具体数值，拐点是实验的输出）

| 旋钮 | 含义 | 理论提速 | 成本 |
|---|---|---|---|
| E1: CFG on/off | 每步算 2 份（条件+无条件）还是 1 份 | off ≈ 2x | 零代码（`10_step_no_cfg` 预设已存在） |
| E2: 步数 sweep | `num_steps` 从 10（baseline）往下扫，找 reward 开始劣化的拐点 | 与步数成正比 | 每档一个配置文件 |

E2 的正确问法不是「能不能用 N 步」，而是「**reward-vs-steps 曲线的拐点在哪**」。
具体扫哪几档由前一档结果决定（例如先试一个明显低档看劣化程度，再二分逼近拐点），
不在 sprint 里写死。两个旋钮先各自单独扫，最后才试最优组合。

每组跑同样长度的短训练（与 baseline 同 epoch 数、同 seed、compiled b8、bf16），对比：

```text
reward 曲线（主验收——和 10 步 CFG baseline 比）
rollout wall clock / epoch（应接近理论倍数）
生成图像抽查（OCR 任务：文字是否还可读）
训练稳定性（loss/grad_norm 无发散）
```

## 3. 验收

```text
某组合 reward 终值接近 baseline（差距 < 约 5-10%，按 OCR reward 的 run-to-run 方差定）
且 rollout wall clock 下降接近理论倍数
  → 把该组合设为 SD3.5 OCR 默认，写回本 sprint + 更新 experiment config

所有组合 reward 都明显劣化
  → 记录数字结论，维持 10 步 CFG 现状，sprint 关闭（负结果也是结论）
```

## 4. 风险与注意

- **OCR reward 对图像质量敏感**：文字渲染是 SD3.5 的弱项，少步/无引导可能直接让文字
  不可读——所以先单因子各自扫，确认各自的安全范围后才试组合，不要一上来就两个旋钮全拧。
- **不要用单次 run 下结论**：reward 曲线有 run-to-run 方差，关键对比至少跑 2 个 seed。
- **CFG off 同时改变 RL 的探索分布**：no-CFG 样本更多样（无引导收缩），对 GRPO 可能
  是好事（多样性）也可能是坏事（低质量样本浪费 reward 调用）——让曲线说话。

## 5. Non-goals

- 不引入 LCM / Lightning / consistency 蒸馏 checkpoint——那是换模型权重的大工程，
  只有当"直接减步"失败、且仍需要少步数时才考虑（单独 sprint）。
- 不动 eval 的 40 步（eval 质量不是本 sprint 的问题）。
- 不动 trainer / GRPO 算法本身（`sde.window_size` 部分步训练是相关机制，但属于
  算法实验，超出本 sprint）。
- 不做多卡（P2，硬件阻塞）、不做 stage overlap（P4，已证明收益 < 2%）。

## 6. 参考

- `SPRINT_rollout_performance.md` §P3（本 sprint 的占位来源）、§D0（阶段时间）、
  batching decision（compiled b8 基线）
- `configs/sampling/denoise/10_step_cfg_4_5.yaml` / `10_step_no_cfg.yaml`
- `vrl/models/diffusion/sd3_5/runner.py:48`（batched_cfg）
- `vrl/generation/diffusion/executor.py`（`sde_step_with_logprob` per-step log_prob）
- Flow-GRPO 上游（`~/Desktop/flow_grpo`）的 step-count 设置可作交叉参照
