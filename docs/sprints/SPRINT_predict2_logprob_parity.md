# SPRINT: predict2 GRPO logprob parity 修复（2026-06-10）

状态：root cause located（2026-06-10，静态分析 + smoke 日志离线取证，见 T2）；
修复未开始 —— 等工作区 P1.4 改动先提交，避免 diff 混杂。

## 0. 问题与口径

cross-model smoke（`SPRINT_cross_model_smoke.md` §2）发现：cosmos-predict2 GRPO 在
lr=0、权重一动未动的条件下，replay 重算的 logprob 与采集时偏差 **mean=13.9 / max=138.5**，
approx_kl=539，clip_fraction=0.53。ratio≈1 不变式被打破 ⇒ **predict2 上任何 GRPO
训练结果当前都是噪声**。这是仓库目前唯一已知的正确性问题，优先于一切性能工作
（smoke 文档自评 P0）。

对照组证明不变式本身和管线是好的：

| family | logprob abs diff (mean / max) | verdict |
| --- | --- | --- |
| sd3_5 | ~1e-8 | pass |
| anima / janus_pro | exactly 0 | pass |
| wan_2_1 | 0.0026 / 0.045 | warn（超 1e-3 guard，T4 顺带看） |
| cosmos-predict2 | **13.9 / 138.5** | **broken** |

完成标准（acceptance gates）：

```text
G1  lr=0 的 predict2 GRPO smoke：replay vs rollout logprob abs diff
    mean < 1e-3（目标对齐 sd3_5 的 ~1e-8 量级），approx_kl ≈ 0，clip_fraction ≈ 0
G2  根因有单元测试钉死（修什么就 pin 什么，防回归）
G3  T0 的 P1.4 live gate 三项全过（见下）
```

## T0. P1.4 live gate（半天，先跑掉）

P1.4（deferred per-epoch reward scoring，`SPRINT_cosmos_performance.md` §P1.4）代码
和单测已落地，缺一个 live GPU 验证。50-epoch 旧流程 run 已结束、GPU 空闲，正好补上。
parity 调试本身也要起小 GPU run，两件事同机不冲突。

起一个短的 predict2.5 NFT run（`online_nft_kling_video_reward`，n 小、epoch 数 3~5），验证：

```text
1. 日志里 `reward worker built model` 每 epoch 恰好 1 次（旧流程是 6 次）
2. epoch 时间从 ~13.4 min 降到 ~11.8 min 量级
3. reward 曲线与 per-group baseline（outputs/ 里 50-epoch run）一致：
   per-sample 分数 diff 应为零或浮点噪声 —— P1.4 只改调度不改语义
```

前置：工作区里 P1.4 的改动（23 文件 + 新协议契约测试）需先提交，否则 parity 调试的
diff 会和 P1.4 混在一起（提交由用户执行）。

## T1. 最小复现

用 smoke 同款配置复现 broken 数字，确认问题仍在、且测量抓手可用：

```text
config:  experiment/diffusion/cosmos_predict2/online_grpo_kling_video_reward
关键项:  lr=0（或 algorithm 等效项）, rollout.sde.type=cps,
         cosmos.reference_mode=per_sample, trainer.debug.first_step=true
读数:    run/training_debug.jsonl 的 abs_diff / fresh_log_prob / old_log_prob / ratio
预期:    abs_diff mean ≈ 13.9（复现成功）
```

现成证据（不必先跑也能分析）：
`outputs/model_smoke_20260609/cosmos2_v2w/run/{training_debug.jsonl,metrics.csv}`，
debug 行含 `rollout_context` / `runtime_debug` / 双侧 logprob，可离线对比。

## T2. 根因（2026-06-10 静态分析 + 离线日志取证，已定位）

**根因：predict2 的 CPS SDE/logprob 数学跑在错误的 sigma 域。**

`flow_matching.py` 的 cps 分支（`vrl/math/diffusion/flow_matching.py:74-99`）假设
rectified-flow 的归一化域 σ∈[0,1]（公式含 `sample - sigma*model_output`、
`1 - sigma_prev`、`sqrt(sigma_prev² - std²)`）。但 predict2 的
FlowMatchEulerDiscreteScheduler 配置是 **EDM 量级 sigma：sigma_max=80.0 /
sigma_min=0.002**（HF cache `Cosmos-Predict2-2B-Video2World/scheduler/
scheduler_config.json`）。executor 与 evaluator 都把原始 `scheduler.sigmas`
喂给这套 [0,1] 公式 —— 数学整体失效。模型自己的 runner 是知道要换域的：
`vrl/models/diffusion/cosmos/predict2/runner.py:28` `current_t = σ/(σ+1)`，
但 SDE 步没有做这个变换。

**数字闭环**（smoke debug 记录 `training_debug.jsonl`，event=first_step_logprob_parity）：

```text
old_log_prob  mean = -4635.1（std 0.57）
fresh_log_prob mean = -4750.6（std 32.0）
abs_diff      mean 115.5 / max 138.5；ratio ≈ e^-115 ≈ 3.5e-41
解释：noise_level=1.0 时 std_dev_t = σ_prev·sin(π/2) = σ_prev ≈ 68，
cps logprob 是未归一化平方距离（无 /2σ²、无 log 项，flow_matching.py:98-99）
⇒ old_lp ≈ -std² ≈ -4624 ≈ 实测 -4635 ✓
⇒ prev_sample_mean 的系数是 σ 量级（~68），replay 侧 noise_pred 的微小差
  （bf16 非确定性/条件还原）被 ~σ² ≈ 4600 放大 ⇒ 偏差 115 只需 Δ rms ≈ 0.02
对照组自洽：sd3_5/wan/anima 全部是 [0,1] 域 + 归一化 sde 分支 ⇒ 1e-8 ~ 2.6e-3；
唯一 EDM sigma + cps 的家族就是 predict2 ⇒ broken ✓
（smoke 文档记的 mean=13.9 与实测 115.5 不符，max=138.5 一致 —— 文档数字待勘误）
```

**附带后果（比 parity 更严重）**：采样本身也在执行结构性错误的 SDE 步 ——
窗口内步骤按 `mean + 68·noise` 注入 EDM 量级噪声，且 cps 均值公式在 σ=68 处
完全偏离流形（`(1-σ_prev)` = -67）。窗口外的确定性欧拉步
（`prev = sample + dt·model_output`，flow_matching.py:96）在 EDM 域恰好是合法的，
所以视频结果看起来还行 —— 但窗口内的训练信号从采样起就是错的，不只是 replay 不一致。

**已排除（均有代码证据）**：
- ~~sde_type 没传到 replay 侧~~：配置链全通 —— `rollout.sde.type` →
  `collector/config.py:94-106`（别名映射）→ `factory.py:183-185`（evaluator 构造）；
  生成侧经 `request_sampling()`（不在排除名单，config.py:164-171）→
  `layout.py:131` → `executor.py:495`。两侧都拿到 cps。
- ~~训练侧 scheduler 没 set_timesteps~~：`predict2/runtime.py:92-94` 构建 bundle 时
  调 `model.set_num_steps(spec.num_steps)`；evaluator 用的就是 bundle scheduler。

**待 GPU 确认的次要项**（修复时一并验证，不改变根因结论）：
- fresh 侧样本间 std=32（old 仅 0.57）—— replay noise_pred 的逐样本波动来源
  （bf16 kernel 非确定性 vs V2W 条件还原差异），修完换域后量级会缩回去，届时再看残差。
- 窗口外确定性步的 logprob 在 cps 下是 `-(euler_prev - cps_mean)²`（结构性大值，
  无意义）；buffer 存全部 35 步（executor.py:122-138 无窗口裁剪），trainer 按
  `timestep_fraction` 均匀抽样（trainer.py:534-540）会抽到这些步 —— 修复时要么
  裁剪到窗口、要么让确定性步不进训练。

## T3. 修复 + 回归测试

修复方向（实现时定夺，原则：SDE 数学只在归一化流域里做）：

```text
方案 A（倾向）：在进入 sde_step_with_logprob 前把 EDM sigma 归一化为
  σ_flow = σ/(1+σ)（即 runner.current_t 的同一变换），model_output 做配套换域；
  采样侧（executor.py:486）与 replay 侧（sde_logprob.py:86）必须用同一变换。
方案 B：给 flow_matching 增加 EDM 域的 cps/sde 推导版本，按 scheduler 域分发。
两案共同要求：变换处加注释说明域假设；wan/sd3_5/anima 路径逐位不变。
```

- 加 pin 测试：构造 sigma_max>1 的 scheduler，断言 sde_step_with_logprob 的
  std_dev_t/log_prob 量级在归一化域（防止再次把 EDM sigma 直喂 [0,1] 公式）。
- 处理窗口外确定性步（见 T2 末尾）：裁剪或排除出训练信号，并钉测试。
- 重跑 T1 的 smoke：G1 三个数字达标；同时人工看一眼窗口内步骤修复后的视频质量
  （采样噪声量级从 ~68 回到 ~1，生成行为会变化 —— 这是修复的预期效果，不是回归）。

## T4. wan parity warn（顺带，不阻塞）

wan_2_1 的 0.0026/0.045 是同一不变式的轻度版本。T2 的逐 step dump 工具做好后顺手
跑一次 wan，确认是 bf16 数值噪声（可接受、调 guard 阈值）还是真有轻度路径差异
（另立任务）。本 sprint 不强求修掉。

## 非目标

```text
- predict2 VAE tiling / 704p（cross-model smoke wiring gap §4.3）：先修正确性，
  否则 704p 跑出来的也是不可信信号
- cosmos perf P0（profiler trace）/ P1（reward 独立 GPU）：性能工作，且 P1 等多卡
- NextStep 14B / Wan 14B GRPO / AR prefill-decode：全部卡多卡硬件
- slime overlap T1-T3 加固项：不堵人，并入以后的 continuous sprint
```

## 证据索引

```text
docs/sprints/SPRINT_cross_model_smoke.md §2 §5        问题发现与优先级
docs/sprints/SPRINT_cosmos_performance.md §P1.4       live gate 标准
outputs/model_smoke_20260609/cosmos2_v2w/run/         training_debug.jsonl / metrics.csv
                                                      / resolved_config.yaml（num_steps=35,
                                                      sde window [0,10), noise_level=1.0）
HF cache .../Cosmos-Predict2-2B-Video2World/scheduler/scheduler_config.json
                                                      sigma_max=80.0 / sigma_min=0.002（EDM 域）
vrl/math/diffusion/flow_matching.py:74-99             CPS 分支（[0,1] 域假设 + 未归一化 logprob）
vrl/models/diffusion/cosmos/predict2/runner.py:28     current_t = σ/(σ+1)（模型自己的换域）
vrl/generation/diffusion/executor.py:482-497          采样侧 SDE 调用 + 窗口逻辑
vrl/rollouts/evaluators/diffusion/sde_logprob.py:86   replay 侧 SDE 调用
vrl/models/diffusion/cosmos/predict2/runtime.py:92-94 训练侧 set_num_steps（已排除项证据）
vrl/rollouts/collector/config.py:94-106,164-171       sde_type 配置链（已排除项证据）
```
