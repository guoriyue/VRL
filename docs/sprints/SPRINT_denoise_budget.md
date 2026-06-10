# SPRINT: Denoise budget — 步数 sweep（rollout 性能的最后一个大杠杆）

状态：proposed。接续 `SPRINT_rollout_performance.md`（其 P3 占位的展开）。

## 0. 一句话

rollout 时间 ≈ **步数 × 每步计算量 × 每次 forward 速度**。第三个因子已经吃完
（compile + bf16 + b8 ≈ 2x，见 SPRINT_rollout_performance）；本 sprint 打第一个因子：
训练 rollout 当前走 10 步，往下扫找拐点，上限 ~2x。这是单卡 SD3.5 剩下唯一的大杠杆。

第二个因子（CFG 每步算 2 份）**评估后判定不可砍**，原因见 §4a——记录在此防止以后重提。

**关键区别**：这不是纯工程优化——减步数会改变生成质量，从而改变 reward 和
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
- GRPO 的轨迹长度 = 去噪步数：`sde_step_with_logprob` 每步写一个 log_prob
  （`vrl/generation/diffusion/executor.py` → `buffers.log_probs`）。减步数 = RL 信号密度
  也变，这是比图像质量更深的一层影响，必须用训练曲线验证。
- 性能对照基线：compiled b8（`SPRINT_rollout_performance` 的 batching decision）。

## 2. 主线只有一个旋钮：步数 sweep

| 旋钮 | 含义 | 理论提速 | 成本 |
|---|---|---|---|
| 步数 sweep | `num_steps` 从 10（baseline）往下扫，找 reward 开始劣化的拐点 | 与步数成正比，上限 ~2x | 每档一个配置文件 |

正确问法不是「能不能用 N 步」，而是「**reward-vs-steps 曲线的拐点在哪**」。
具体扫哪几档由前一档结果决定（先试一个明显低档看劣化程度，再逼近拐点），不写死。

每组跑同样长度的短训练（与 baseline 同 epoch 数、同 seed、compiled b8、bf16），对比：

```text
reward 曲线（主验收——和 10 步 CFG baseline 比）
rollout wall clock / epoch（应接近理论倍数）
生成图像抽查（OCR 任务：文字是否还可读）
训练稳定性（loss/grad_norm 无发散）
```

## 3. 验收

```text
某档步数 reward 终值接近 baseline（差距 < 约 5-10%，按 OCR reward 的 run-to-run 方差定）
且 rollout wall clock 下降接近理论倍数
  → 把该档设为 SD3.5 OCR 默认，写回本 sprint + 更新 experiment config

所有低档 reward 都明显劣化
  → 记录数字结论，维持 10 步 CFG 现状，sprint 关闭（负结果也是结论）
```

## 4. 风险与注意

- **OCR reward 对图像质量敏感**：文字渲染是 SD3.5 的弱项，步数砍狠了文字可能直接
  不可读——先试一档看劣化程度，再逼近，不要跳档。
- **不要用单次 run 下结论**：reward 曲线有 run-to-run 方差，关键对比至少跑 2 个 seed。

### 4a. 为什么不关 CFG（已评估，决定不做——记录防止重提）

CFG 每步算 2 份（条件+无条件），关掉理论省一半。但对 SD3.5 + GRPO + OCR 这个组合
**大概率冷启动失败**，2026-06-09 评估后决定不做：

1. **没有 CFG，SD3.5 几乎渲染不出文字**，而 OCR reward 只看文字。
2. **GRPO 靠组内 reward 差异学习**：一组 8 张图全都没字 → 全员低分 → 组内无差异 →
   advantage ≈ 0 → 没有梯度方向。不是学得慢，是没有起跑信号。
3. **CFG 是被优化的策略本身**：flow_grpo 训练侧重算 log-prob 时做同样的 CFG 混合
   （`~/Desktop/flow_grpo/scripts/train_sd3.py:185-195`），rollout 与 replay 必须一致。
   关 CFG = 换一个起点差得多的策略类。

注意「降低 guidance_scale」**不省任何时间**——成本是双份 forward，与强度数值无关。
未来若真要砍这一半，仅有的两条原则性路径（都超出本 sprint）：
- **CFG interval**：只在中段步开 CFG、两端步跳过 uncond forward（部分省）；需要
  executor 支持按步开关且 replay 镜像同样的开关。
- **先做 CFG 蒸馏再 RL**：把引导行为蒸进权重，再无 CFG 训练（一次性 SFT 大工程）。

对照：cosmos predict2.5 + NFT 走 no-CFG 是**算法设计**（NFT 不需要 log-prob/CFG，
checkpoint 也经过后训练），与 SD3.5 GRPO 不可类比。

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
