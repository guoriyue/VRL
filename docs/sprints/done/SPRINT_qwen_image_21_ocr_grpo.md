# SPRINT: Qwen-Image-2.1 OCR GRPO — first long post-training run on one 5090

状态：**DONE**（2026-09-21）。60 updates 一次跑完（7h00m，无重启），held-out OCR 明显上升，见 §6/§7。

## 0. 一句话

在单张 32 GB 5090 上，用 r=256 全投影 LoRA（fp32 master）+ Flow-GRPO SDE（noise 0.7）
+ OCR reward 对 Qwen-Image-2.1 做 60 个 update 的 post-training；结论看 §6 的
held-out ODE OCR delta（bootstrap CI），不是训练 reward。

## 1. Launch

```bash
cd ~/Desktop/VRL
source ~/Desktop/wm-infra/.venvs/cosmos3/bin/activate    # project .venv lacks vLLM CuMem
export PYTHONPATH=$HOME/Desktop/VRL TORCHINDUCTOR_COMPILE_THREADS=1
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE   # see §3 (a)
python -m vrl.scripts.supervise --config experiment/qwen_image_21/online_grpo_ocr \
  --max-attempts 6 --health-metrics --health-max-grad-norm 0.5 \
  model.lora.parameter_dtype=float32 precision.float32_precision=ieee \
  trainer.output_dir=outputs/qwen_image_21_ocr_grpo_run1 \
  trainer.total_epochs=60 trainer.save_freq=20 trainer.debug.first_step=true
```

Dry run（3 updates）用同样的 overrides，`trainer.total_epochs=3 trainer.save_freq=1000`，
输出 `outputs/qwen_image_21_ocr_grpo_dryrun/`。

Recipe（`vrl/config/presets/experiment/qwen_image_21/online_grpo_ocr.yaml` +
`model/qwen_image_21/lora_wide.yaml`，本 sprint 改动见 §3）：

| 项 | 值 |
|---|---|
| base | Qwen/Qwen-Image-2.1 @ b3179ad3, transformer bf16 (14 GB), Qwen3-VL encoder cpu_resident |
| trainable | LoRA r=256 / α=512 on to_q,to_k,to_v,to_out.0,img_mlp.{proj,gate_layer,out}, 671M params, **fp32** |
| precision | bf16 autocast compute both sides, fp32 SDE/log-prob math, ieee fp32 matmul |
| sampling | 512px, 10 steps, no CFG (true_cfg_scale 1.0), SDE noise_level 0.7 |
| update | 6 prompts × 16 samples = 96/update, timestep_fraction 0.5, microbatch 1, lr 3e-4, kl 0.04, clip 1e-4 |
| memory | compile OFF, gradient_checkpointing full, AdamW8bit moments |

## 2. Dry-run gates（3 updates，`outputs/qwen_image_21_ocr_grpo_dryrun/`）

Dry run 跑的是 §3 之前的几何（8 prompts × 16 = 128 samples，timestep_fraction 0.99）。

| gate | 目标 | 实测 | 判定 |
|---|---|---|---|
| replay_parity_gate max_abs_diff | == 0.0 | **2.77e-4**（limit 0.01） | 见下 |
| trainable digest before ≠ after | 不同 | dry run 时 streaming 路径不写 after 摘要（§3 e）；用 checkpoint-final 验证：LoRA B 从 0 → ‖B‖=4.45, fp32 | PASS |
| grad_norm | 有限、非零 | 2.08e-4 / 6.23e-4 / … | PASS |
| peak GPU memory | < 30 GB | **28.05 GB**（nvidia-smi 5 s 采样） | PASS（余量 3 GB） |
| one update | < 8 min | **~12–15 min**（rollout 3.5–4 min + replay 8.5–10.5 min） | FAIL → §3 (d) |

**Parity 不是 0 的原因不是编译**：两边都是 eager（compile 关，§3 c）。剩下的差异来自
batch 形状：rollout 一次 8 个样本过 transformer，replay microbatch=1，cuBLAS/attention
对不同 batch 选不同 kernel，bf16 下 log-prob 差 2.8e-4。这个仓库早就把它写进 gate 的报错文案
（"align rollout/replay precision and batch shape"）；SD3.5 PickScore 那次也是
`samples_per_generation_batch=1` 才拿到 0.0。这里没有用 sbs=1（rollout 慢 3–4 倍），
2.8e-4 比阈值低 36 倍，记录为 kernel noise，不动阈值。

## 3. 路上修的东西（都已 commit）

(a) **HF offline 会挂 replay loader**：`load_diffusers_transformer` 走
`from_pretrained(revision=<sha>)`，`HF_HUB_OFFLINE=1` 下 hub 的 revision 解析抛
`OfflineModeIsEnabled`（Wan A/B 时同样的坑）。launch 不设 offline 变量。

(b) **Qwen-Image 两个 family 的 replay scheduler 没有 timesteps**（`ca67909d`）：
dynamic-shifting 的 mu 依赖分辨率，通用 replay loader 不知道，FLUX 有
`prepare_replay` 而 qwen_image / qwen_image_21 没有；SDE log-prob 第一步就在
`index_for_timestep` 上 cuda/cpu 设备不一致。补 `prepare_replay`（2.1：
`2*(H//32) * 2*(W//32)` 个 token；v1：`(H//16)*(W//16)`），v1 顺带把
`self.pipeline.scheduler` 改成 `self.scheduler`（replay 没 pipeline）。两个 family 各加一个
"replay grid == rollout grid" 的测试。

(c) **r=256 fp32 LoRA + compile 塞不进 32 GB**（`6f5e5281`）：第一次 replay forward 就 OOM
（29.15 GiB，还没 backward）。compile 禁止 activation checkpointing
（`vrl/config/validation.py`），而没有 checkpointing 一次 512px forward 的激活 >12 GB。
改成：compile 关（参考 pipeline 上只有 1.13×）、`gradient_checkpointing: full`、
`optim_8bit: true`（Adam m/v 5.4 GB → 1.35 GB，权重仍 fp32）。

(d) **一个 update 12–15 min → ~6.5 min**（同一 commit）：eager + 全量 recompute 的 replay
每个 sample-step ≈ 4 个 forward 当量。`timestep_fraction 0.99 → 0.5`（仓库其它
compute-bound recipe 的常规值），`prompts_per_batch 8 → 6`（96 samples/update）。
长跑实测 update 1 = 6.3 min（rollout ~3 min + replay ~4 min 含 first-step probe）。

(e) **streaming 路径没有 after-step 摘要**（`2c92a489`）：`replay_parity_gate` 只记
before 摘要，长跑没有"权重真的动了"的证据。两条 update 路径在第一次通过 parity 的 step 后
各写一条 `first_update_weights`（before/after sha256 + `moved`）。长跑 update 1：
`moved: true`（6979af2b… → 1e681942…）。

(f) **eval 分辨率**（`f0a252cb`）：`EvalSection` 只有 `num_steps`，1024px 评测加
`width/height`（`resolve_eval_sampling` 本来就按 key 读 eval 段）。

(g) **同卡有别的 agent 在跑 2.1 编辑探针**（~15 GB，每次几分钟）：dry run 被撞死两次
（`RayGenerationWorker.load_policy` / `wake` OOM）。launcher 先等卡上 <1 GB 持续 45 s
再起，supervisor `--max-attempts 6` 从最近 checkpoint 续。

## 4. 长跑 gates（update 1，`outputs/qwen_image_21_ocr_grpo_run1/training_debug.jsonl`）

| gate | 实测 |
|---|---|
| replay_parity_gate | 2.77e-4, passed |
| first_update_weights | moved=true |
| grad_norm | 2.70e-4 |
| reward_mean / std | 0.324 / 0.416 |
| clip_fraction | 0.108（clip_ratio 1e-4 < kernel noise 2.8e-4，所以裁剪率天然不低；stop 线 0.2） |

## 5. 训练曲线（`metrics.csv`，60 updates，23:58 → 06:59，**7.0 min/update**，peak **29.35 GB**）

| updates | reward_mean（±sd over 10） | clip_fraction | grad_norm |
|---|---|---|---|
| 1–10 | 0.342 ± 0.129 | 0.102 | 3.5e-4 |
| 11–20 | 0.404 ± 0.132 | 0.091 | 9.4e-4 |
| 21–30 | 0.546 ± 0.159 | 0.083 | 5.2e-4 |
| 31–40 | 0.685 ± 0.089 | 0.084 | 5.3e-4 |
| 41–50 | 0.528 ± 0.172 | 0.085 | 3.4e-4 |
| 51–60 | 0.666 ± 0.123 | 0.080 | 4.6e-4 |

clip_fraction 全程 0.08–0.125（stop 线 0.2 没碰），grad_norm 最大 3.3e-3（health 线 0.5），
logprob_abs_diff_max 最大 6.3e-4，approx_kl 全程 0（clip 1e-4 的 PPO 目标在这个尺度上
ratio≈1）。supervisor 0 次重启，同卡别的 agent 没再撞上来。

不做的中途调参、事后会改的：(1) 6 prompts/update 的 reward_mean 噪声大（sd ≈ 0.13），
看趋势要按 10-update 平均；(2) clip_ratio 1e-4 低于 bf16 kernel noise（2.8e-4），裁剪的是
噪声不是策略变化，下次值得把 clip 抬到 ≥1e-3 或把 sbs 降到和 replay microbatch 一致再比。

## 6. Held-out 评测（ODE，`vrl.scripts.eval.image_checkpoint_eval`，reward/ocr）

`manifests/ocr/test.txt` 前 64 条 prompt，seed-paired vs base，bootstrap 2000 次，统计单位 = prompt。

**512px / 10 steps，2 samples/prompt（`eval_512_10/report/summary.json`）**

| arm | OCR mean [95% CI] | paired Δ vs base [95% CI] | win / tie |
|---|---|---|---|
| base | 0.472 [0.381, 0.567] | — | — |
| ck20 | 0.561 [0.467, 0.658] | **+0.089** [+0.049, +0.130] | 0.53 / 0.34 |
| ck40 | 0.632 [0.547, 0.712] | **+0.160** [+0.087, +0.237] | 0.62 / 0.17 |
| ck60 | 0.713 [0.632, 0.786] | **+0.241** [+0.175, +0.311] | 0.70 / 0.25 |

**1024px / 40 steps（模型原生分辨率），1 sample/prompt（`eval_1024_40/report/summary.json`）**

| arm | OCR mean [95% CI] | paired Δ vs base [95% CI] | win / tie |
|---|---|---|---|
| base | 0.640 [0.538, 0.739] | — | — |
| ck60 | 0.852 [0.768, 0.927] | **+0.211** [+0.119, +0.305] | 0.39 / 0.55 |

单调、CI 不过零、在没训过的 1024/40 上也成立。

**眼看 16 对（512）+ 6 对（1024）base vs ck60，变化是什么：**

- 文字变**大、变正、变居中、变成无衬线的"招牌体"**——广告牌、路牌、石碑、marquee 上的字
  占画面比例明显上升（p9 "Gas Next Exit 2 Miles" 的路牌从远处小牌变成占满上半幅的大广告牌）。
- base 常见的**字形错乱/漏字**被修掉（"Try Our 별차차 Burger" → "Try Our New Burger"；
  墓碑上原本没字 → "Loved And Remembered"；"Lost City Crar" → "Lost City Near"）。
- 代价：构图向"文字为主"漂移，场景元素变少变平（p5 marquee 场景从写实照片变成偏插画的
  平面风），个别样本仍出错（1024 的 p5 "Toight Binary StandUp"）。这是 OCR reward 的
  典型 hacking 方向，下一步要配一个美学/prompt-alignment reward 或 KL 抬高再跑。

**2048 px / 40 steps（官方推荐设置，README 示例），前 16 条，1 sample/prompt（`eval_2048_40/report/summary.json`）**

| arm | OCR mean [95% CI] | paired Δ vs base [95% CI] | exact 1.00 |
|---|---|---|---|
| base | 0.718 [0.509, 0.924] | — | 10 / 16 |
| ck60 | 0.925 [0.799, 0.996] | **+0.206** [+0.040, +0.394] | 12 / 16 |

逐条：11 条两边都对，4 条 base 错 ck60 对（"Tonight Binary StandUp"、"Elevation 8000 Feet"、
"Abandon All Hope"、"Trespassers Will Be Jousted"），1 条两边 0（"Fearless" 花体，两边都画对，
PaddleOCR 读不出），没有 ck60 退步的例子。**base 在官方设置下 62% 完全正确，不差**；
512px 是 32×32 latent 格子的残废工作点，对比页里 base 的"差"主要是分辨率和读取器造成的。

## 7. Verdict

**LEARNED（在训练分辨率上显著；在官方全分辨率上方向一致、样本不足）。**
held-out OCR 512/10：+0.241（CI [+0.175, +0.311]）；1024/40：+0.211（CI [+0.119, +0.305]）；
2048/40（官方设置，16 条）：+0.206（CI [+0.040, +0.394]）。512px SDE 10 步训练的策略迁移到
2048px ODE 40 步：策略学的是速度场不是图，字符身份/顺序的决策与分辨率无关，dynamic shifting
把时间步按 token 数重归一化。一个 2048/40 样本是 512/10 的 64 倍算力，单卡 RL 只能这么训。

## 8. 路径

- metrics: `outputs/qwen_image_21_ocr_grpo_run1/metrics.csv`
- debug: `outputs/qwen_image_21_ocr_grpo_run1/training_debug.jsonl`
- supervisor log: `outputs/qwen_image_21_ocr_grpo_run1/supervise.log`
- dry run: `outputs/qwen_image_21_ocr_grpo_dryrun/`
- eval 512/10: `outputs/qwen_image_21_ocr_grpo_run1/eval_512_10/report/{summary.json,curve.csv,contact_sheets/}`（log `eval_512_10.log`）
- eval 2048/40（16 条）: `outputs/qwen_image_21_ocr_grpo_run1/eval_2048_40/report/`（log `eval_2048_40.log`；run dir `outputs/qwen_image_21_ocr_grpo_run1_eval2048/`；中途停掉的 64 条版本留在 `eval_2048_40_partial64/`，只有 33 张 base 图）
- eval 1024/40: `outputs/qwen_image_21_ocr_grpo_run1/eval_1024_40/report/`（log `eval_1024_40.log`；run dir 复制在 `outputs/qwen_image_21_ocr_grpo_run1_eval1024/resolved_config.yaml`，只加了 `eval: {width: 1024, height: 1024, num_steps: 40}`）
- checkpoints: `outputs/qwen_image_21_ocr_grpo_run1/checkpoint-{20,40,60,final}/lora_weights/`
