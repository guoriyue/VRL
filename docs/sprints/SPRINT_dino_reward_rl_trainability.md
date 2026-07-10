# SPRINT: dino reward RL-trainability —— 用因果探针证明 reward 能被正确方向推动

状态：**in-progress（2026-07-09）**。核心因果问题已用 gradient-honest 探针答出（POSITIVE，near-significant）；
剩余项 = 正式管线 `vrl-train` 端到端复验 + Ray-kill 隔离 + 提高统计功效。性质：**correctness/causal
validation**（不是新功能），沿 [[project_first_trustworthy_curve]] 的"能不能学"这条主线。

> 来由：`online_grpo_droid_full_target_480p` 的 24h 长跑判"reward 往下掉 -4.5σ"（[[project_droid_overfit_validation]]），
> 追问"这是 reward 坏，还是操作点/梯度坏"。逐层拆解后发现前两次下跌各有 confound（采样脆 + 梯度 bug），
> 用修好的单-prompt 因果探针拿到第一条**干净**曲线。

---

## 0. 一句话

**RL 能把 dino reward 往对的方向推——方向稳、可复现、梯度逐位验证干净；但效果小（~+2–4%），
统计上正好卡在显著边缘（t≈1.6–1.96 随端点抖动），还没决定性越过 t>2。** 之前两次"往下掉"都是
artifact（采样 confound + 探针梯度 bug），不是 reward 本身坏。

## 1. 结果：干净因果曲线（单 prompt overfit，每点 16 样本，paired 同种子）

```
R0=0.4664(基线) R1=0.4720 R2=0.4692 R3=0.4706 R4=0.4698
R5=0.4772 R6=0.4831 R7=0.4836 R8=0.4878 R9=0.4907 R10=0.4856  (R11/R12 in progress)
slope/update ≈ +0.0024 (>+0.002 阈值 → 自动 VERDICT POSITIVE)
```

两个视角，都要看：
| 视角 | 读数 | 结论 |
|---|---|---|
| **趋势（over rounds）** | 斜率一直为正、R5–R9 连续 5 个新高、所有近期点远高于基线与早期 ~0.470 平台 | 方向不容置疑 |
| **配对显著性（R0 vs 端点）** | R6 t=1.20 → R8 t=1.41 → R9 t=1.96 → R10 t=1.62（回撤）；抗噪估计"末 3 点均值 vs R0" = +2.2%, t≈1.74 | near-significant，在阈值附近**抖动**，未决定性越过 |

**诚实措辞**：不是"已显著、板上钉钉",是"能推、稳定为正、效果小、刚到显著边缘"。不要拿 R9 的
t=1.96 当"crossed"——R10 回撤就掉回 1.62,单端点会两边跳。

- 每轮验证 `parity |Δlogp|_mean = 0.0000`、`clip_hits` 从 0 随策略移动上升（0→6→12→8→9→15）=
  梯度诚实 + 策略确实在动,不是随机。
- 产物：`outputs/_level0_curve/`（`state.json`、`scores_round*.json`、`lora_round_*.pt`、`verdict.json`）。
  探针脚本 `scratchpad/level0_curve.py`（Ray-free、断点续、抗杀）。

## 2. 关键 debug：为什么"往下掉"是假的

**第一次下跌（run10, 真 vrl-train, -4.5σ）= 采样 confound**：10 步生成太脆 + `same_latent` 共享噪声,
组内方差 = "哪条噪声抽签坏得少"（最差的是彩色 blob 怪）,GRPO 爬的是运气不是内容。修复 = 15 步（SDE
window [0,10) + 5 步确定性尾）+ 去共享种子（真内容多样性）。

**第二次下跌（level0 探针初版）= 梯度 bug（在探针,不在代码库）**：探针手写 replay 用了 `forward(es, 0)`,
但 cosmos 的 `forward_step` 按 `scheduler.sigmas[step_idx]` 取 sigma → step 2/4/6/8 全取到 step-0 的大
sigma → replay log-prob 差 0.6–1.5 → ratio 全错 → 梯度污染。**征兆 = ppo-epoch-1 的 `clip_hits` 恒
64/80（80=16样本×5步,只有 step0 对得上）。** 修复 = `forward_step(es, rec["step_idx"])`,parity 立刻归零：
```
step   forward(es,0) BUG   forward_step(es,k) FIX
 0        0.00000              0.00000
 2/4/6/8  0.6–1.5        →     0.00000
```
证明脚本 `scratchpad/parity_test.py`。

## 3. 你的代码库没有这个 bug（16 家族全审）

见 [[project_replay_parity_audit]]。两个正确 pattern,全家族一致：
- **sigma-indexer**（cosmos predict2/2.5/anima/cosmos3,forward_step 读 `sigmas[step_idx]`）→ 都用
  `CosmosReplayForward` mixin 的 `forward_step(state, timestep_idx)`（真索引）。`vrl/models/diffusion/cosmos/__init__.py:38`。
- **timestep-only**（sd3/flux/qwen/wan/sana/cogvideox/hunyuan×2/pixart/mochi/lumina2）→ 都用
  `pack_eval_timestep` 把第 k 步 timestep 打包到位置 0,base `forward(state,0)`。`vrl/models/diffusion/base.py:146`。
- 非标准 sigma 域都**显式处理**：cosmos EDM（`sde_step_with_logprob` 自动检测 `sigmas.max()>1` 转域）、
  mochi/lumina2 反转域（重建 descending scheduler）、pixart epsilon-DDPM（走独立 `sde_type=ddim` log-prob 路径）、
  echo 从 timestep 值直接推 sigma（免疫）。
- **生产双保险**：evaluator 走 `model.replay_forward`（按家族正确分派）+ trainer `debug.first_step` parity
  检查。探针绕过了这两道,才中招。

## 4. 基建（抗环境）

- `~/.local/bin/run-until-success`：通用进程守护,任何命令套一下即得"跑到成功为止"——SIGTERM/HUP 免疫、
  `setsid` 独立会话、等 GPU 空、断点续、连续同类报错就停不空烧 GPU。见 [[feedback_unattended_run_survival]]。
- 93f 单卡显存地板：frozen offload + CPU-paged replay tensors + grad-ckpt + samples_per_chunk=1。
  见 [[project_single_gpu_93f_probe_oom]]。

## 5. 探针 ≠ 正式管线（诚实边界）

**现在跑的是探针,不是 `vrl-train`。** 探针复刻了 trainer 的 GRPO 数学(同 `group_relative_advantages`、
同 `sde_step_with_logprob`、同常数)且 parity=0.0000（对同一 rollout,梯度和 trainer 逐位一致）,所以因果
结论可迁移。但**正式管线在正确性上更 solid**（久经考验、用正确 mixin、自带 parity 守卫、run9/11 实测
parity=0.0）。我用探针只因为**它在你这台"专杀 Ray + 抢卡"的机器上更能 survive**——正式管线依赖 Ray,
你测试进程的 `ray stop` 会整机扫杀它的 worker;探针 Ray-free 躲过了。这是**环境**问题,不是正式管线不 solid。

## 6. 剩余项（按优先级）

1. **正式管线 continuation（IN PROGRESS 2026-07-09）** ⭐：探针停在 R10（dino ~0.486, +4.6%）后,
   **warm-start 从探针策略接着用 `vrl-train` 训**(不是从零)。落地细节:
   - 探针 `lora_round_10.pt`(PEFT 格式,448 keys)→ `scratchpad/convert_probe_lora.py` 经 model.apply_lora
     路径转成 PEFT adapter 目录 `outputs/_level0_curve/probe_lora_r10_adapter`(load 时 unexpected=0,接干净)。
   - 配方 `online_grpo_droid_overfit_validation` + 显存修复(spc=1 / traj cpu / reward gpu_pool)+ `run-until-success`。
   - **断点续正确性坑**:`model.lora.path`(warm-start)与 `trainer.resume_from`(续 ckpt)**互斥**。条件启动器
     `scratchpad/cosmos_continue.sh`:首跑 warm-start 探针 adapter;之后每次被杀改从 vrl-train 最新 checkpoint 续
     (`save_freq=2`),不退回探针、不丢生产进度。
   - **⚠️ Ray-stop 风险回归**:换回整框架 = 重新接上 Ray 总闸(§6.2)。探针阶段 ray-stop 是可选防护;
     continuation 阶段变刚需——邻居频繁 `ray stop` 会让进度很碎,`run-until-success`+`save_freq=2` 只能自动续。
   - **warm-start confound(诚实)**:从探针 +4.6% 接着训 = 不浪费,但这条生产曲线**不能再独立宣称"生产也把
     dino 推上去了"**(大头探针推完了)。要独立复验须从零跑。且探针只 overfit 1 个 prompt,本配方 4 个 prompt →
     其余 3 个冷启动,是"热启动的真训练"非纯单-prompt 续。
   - **判读**:先看 `first-step log-prob` parity ≈0(证明 warm-start 权重接干净)→ 再看 eval dino 从 ~0.486 往哪走。
2. **Ray-kill 隔离**：Ray 是整机全局单例,`ray stop` 按进程名全杀。解法(按干净度)：① 训练放独立机器/容器;
   ② 同机则 `run-until-success` + 勤 checkpoint(`trainer.resume_from`)自动续;③ 若能改测试进程,让它用独立
   Ray 实例(自己的 `_temp_dir`+端口+namespace),teardown 只停自己。**待你确认测试进程是否 `ray stop`。**

   > **为什么会有 ray-stop 问题(心智模型,别再说"Ray 不 solid"):** Ray 本身很 solid,问题不在它可靠性,
   > 在于它是**整机全局单例**——`ray.init()` 起/连的是机器上**唯一一套** Ray,`ray stop` 是**按进程名整机扫杀**
   > 的管理命令(设计如此)。两个互不协调的负载同机、都用 Ray,谁跑 `ray stop` 谁把对方也端掉。这是**多租户/
   > 共享**问题,不是 Ray 脆。探针"活下来"只因它**不用 Ray = 没有被 `ray stop` 攻击的面**,不是它软件更好
   > (它反而是有 bug 的那个)。
   >
   > 打个比方:Ray 像整栋楼共用的电力总闸,`ray stop` = 有人去拉总闸。你的训练和你的测试是同一条总闸下的两户;
   > 测试为重置自己拉一下总闸,你的训练那户也跟着黑了——不是你家线路差,是你俩共用一个闸。探针 = 自带电池、
   > 不接总闸的设备,拉闸拉不到它。跟"谁的电器更 solid"无关。
   >
   > **真正的解 = 进程隔离,让邻居的 `ray stop` 够不到你:**
   > - **最干净:容器 / 独立机器。** 容器有 PID namespace 隔离——邻居在它那边 `ray stop` 看不到也杀不到你容器里的
   >   Ray 进程。根治。
   > - **只加端口/namespace/temp-dir 不够:** 那只解决"两套 Ray 启动抢端口";`ray stop` 按进程名整机扫,照样杀你隔离的那套。
   > - **同机又不隔离 = 本质脆弱:** 只要邻居能跑全局 `ray stop` 就防不住;`run-until-success` + 勤 checkpoint 是
   >   "被杀自动续"的创可贴,不是根治。
   > - **诚实前提:** "是 `ray stop`" 是从 SIGTERM 签名 + "有进程一直 testing" **推断**的,当时的杀是**混合**的
   >   (一部分外部 SIGTERM,一部分 OOM——OOM 与 Ray 无关,已被显存修复解决)。需你确认测试进程是否 `ray stop`。
3. **提高统计功效**：把每点样本 16 → 32/64,压掉点间抖动,让 t 稳定越过 2(比多跑更新更干净)。
4. **first_step 守卫改硬失败**：`vrl/trainers/online/trainer.py:1127` 现在 parity>0.01 只 **warn**;改成
   **hard-fail**（或首跑默认开）,这样未来新家族若写错 sigma 模式,不会默默用坏梯度训几小时(正是探针的教训)。

## 7. 非目标

- 不证明 dino 是"好"reward(能训 ≠ 训出好模型;泛化是多-prompt 长跑的事)。
- 不改 §3 已审过的家族 replay(它们都对)。
- 不在探针上追求生产级(它是替身;正式落地走 §6.1)。

## 8. 引用

- 探针/证据：`scratchpad/level0_curve.py`、`scratchpad/parity_test.py`、`outputs/_level0_curve/`
- 正确 pattern：`vrl/models/diffusion/cosmos/__init__.py:38`（`forward_step(state, timestep_idx)`）、
  `vrl/models/diffusion/cosmos/predict2/model.py:378`（`sigmas[step_idx]`）、
  `vrl/models/diffusion/base.py:146`（`forward(state, 0)`）、
  `vrl/math/diffusion/flow_matching.py:74`（`sigma = scheduler.sigmas[step_index]`）
- 守卫：`vrl/trainers/online/trainer.py:1127`（first_step parity warn→建议 hard-fail）
- 配方：`configs/experiment/diffusion/cosmos_predict2/online_grpo_droid_overfit_validation.yaml`
- 记忆：[[project_droid_overfit_validation]]、[[project_replay_parity_audit]]、
  [[project_single_gpu_93f_probe_oom]]、[[feedback_unattended_run_survival]]、[[project_first_trustworthy_curve]]
