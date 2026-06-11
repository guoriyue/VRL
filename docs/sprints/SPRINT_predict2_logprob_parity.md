# SPRINT: predict2 GRPO logprob parity 修复（2026-06-10）

状态：**done（2026-06-10，G1 GPU live gate 通过）**。换域修复落在
`vrl/math/diffusion/flow_matching.py`（单点，采样/重放两侧自动一致）。

G1 实测（复跑 6/9 同一份 resolved config，lr=0 / cps / 512p93f / 35 步，
唯一变量是修复；产物为一次性验证已删除，关键数字如下表）：

| 指标 | broken 6/9 | fixed 6/10 | gate |
| --- | ---: | ---: | --- |
| old_log_prob mean | -4635 | **-0.9707** | O(1) ✓ |
| parity abs_diff mean | 115.5 | **5.3e-6** | <1e-3 ✓（优于 wan 的 2.6e-3，接近 sd3_5）|
| ratio | ~3.5e-41 | **0.999995** | ≈1 ✓ |
| approx_kl @ lr=0 | 539.5 | **0.0** | ≈0 ✓ |

验证体系四层全绿：域同构 pin 测试（6 项）→ cosmos-rl 参考实现 bit 级对照
（flow/EDM 两域 diff=0.0）→ 真 scheduler 离线复算 → 真 checkpoint GPU 复跑。
另落地一条防线：trainer first_step parity 偏差 >0.01 时 logger.warning 大声报警
（vrl/trainers/online/trainer.py），不再只写 jsonl。

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

## T0. P1.4 live gate —— done（2026-06-10，见 SPRINT_cosmos_performance.md 进度：
motion run 全程每 epoch 恰好 1 次 `reward worker built model`，epoch 13.4 → 12.35 min）

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

**predict2.5 勘误（2026-06-10 实证）**：predict2.5 不受影响。它的 UniPC config 虽写
`sigma_max=200`，但 `use_flow_sigmas: true` 使运行时 sigma 表已归一化（实测
set_timesteps(35) 后 sigmas = 0.995 → 0.0099 ∈ [0,1]）。这同时否决了"按
config.sigma_max 判域"的方案 —— 判域必须看运行时 sigma 表。

**窗口勘误（2026-06-10）**：T2 初稿"窗口外确定性步混入训练"的担忧不成立。
`layout.select_sde_window`（layout.py:211-222）在 `window_size <= 0` 时返回 None，
executor 把 None 解释为全部步随机 —— smoke 的 `window_size: 0` ⇒ 35 步全是随机
SDE 步。全仓库 configs 仅一处 window_size 且为 0，windowed SDE 当前无人使用。
留档的潜在 gap：若未来启用 window_size>0，随机窗口（per-request randint）不会
被记录进 trajectory，窗口外确定性步的 logprob 无意义且会进训练 —— 启用前需先
把窗口写入 batch context 并在 trainer 侧裁剪 train_indices。

**待 GPU 确认的次要项**：fresh 侧样本间 std=32（old 仅 0.57）—— replay noise_pred
的逐样本波动来源（bf16 kernel 非确定性 vs V2W 条件还原差异）。换域后放大系数
从 ~σ²≈4600 回到 O(1)，残差应缩到其它家族水平（≤1e-3 量级），G1 跑完看数字。

## T3. 修复 + 回归测试（已实现，2026-06-10）

实现取了方案 A 的变体：换域不在两个调用点做，而是**收进
`sde_step_with_logprob` 内部单点处理**（`vrl/math/diffusion/flow_matching.py`），
采样侧与 replay 侧自动一致，杜绝两侧变换不同步的回归面：

```text
- 判域：运行时 sigma 表 max > 1 ⇒ EDM 域（config 不可靠，见 T2 勘误）；
  判定结果缓存在 scheduler 上（_vrl_edm_sigma_domain），避免逐步 GPU→host 同步
- 换域：t = σ/(1+σ)；x̃ = x/(1+σ)；ṽ = n̂·(1+σ) − x
  （EDM 侧 model_output 是噪声估计 n̂=(x−x0)/σ，见 predict2 runner.finalize_noise_pred）
- 域约定：prev_sample 转回 scheduler 自己的域（轨迹/buffer 连续性）；
  log_prob / prev_sample_mean / std_dev_t / sqrt_neg_dt 一律 flow 域
  （Jacobian 常数在 ratio 与 KL 中相消，量级与 flow 原生家族可比）
- wan / sd3_5 / anima / predict2.5 路径逐位不变（flow 域判定为 False 走原代码）
```

验证（CPU，无 GPU）：

```text
- tests/math/test_diffusion_flow_matching.py 新增 6 项全过：
  EDM↔flow 域同构等价（sde+cps）/ EDM 采样→重放 parity（bit 级一致）/
  flow 域走原路径 / 域判定缓存
- 离线复算（真 predict2 scheduler, set_timesteps(35), cps, noise_level=1.0）：
  log_prob -0.976（修前 -4635）；replay parity abs diff 0.0（修前 ~115）
- 全量回归：tests/math+algorithms+rollouts+generation+trainers 310 passed，
  tests/models 96 passed，e2e 正常 collect
```

剩余：重跑 T1 的 GPU smoke，G1 三个数字达标；同时人工看一眼修复后的视频质量
（采样噪声量级从 ~68 回到正确尺度，生成行为会变化 —— 这是修复的预期效果，不是回归）。

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
                                                      —— 一次性产物已删（2026-06-10），
                                                      关键数字均已抄录入本文档
HF cache .../Cosmos-Predict2-2B-Video2World/scheduler/scheduler_config.json
                                                      sigma_max=80.0 / sigma_min=0.002（EDM 域）
vrl/math/diffusion/flow_matching.py:74-99             CPS 分支（[0,1] 域假设 + 未归一化 logprob）
vrl/models/diffusion/cosmos/predict2/runner.py:28     current_t = σ/(σ+1)（模型自己的换域）
vrl/generation/diffusion/executor.py:482-497          采样侧 SDE 调用 + 窗口逻辑
vrl/rollouts/evaluators/diffusion/sde_logprob.py:86   replay 侧 SDE 调用
vrl/models/diffusion/cosmos/predict2/runtime.py:92-94 训练侧 set_num_steps（已排除项证据）
vrl/rollouts/collector/config.py:94-106,164-171       sde_type 配置链（已排除项证据）
```
