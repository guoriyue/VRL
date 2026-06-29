# SPRINT: Cosmos Predict2 2B — 第一条可信的正向 RL 学习曲线（full-param, 单卡）

状态：**blocked / dataset-blocked（2026-06-27）**。目标:**在 cosmos predict2 2B 上拿到第一条可信的正向学习曲线(eval reward 上升 >2σ)**。enabler(8-bit Adam full-param)已建好,但本地没有 source-backed Video2World reference-image dataset；`perf_smoke/ref.png` 是随机噪声并已由 `SPRINT_remove_smoke_datasets.md` 删除,不能作为 RL 目标。

> 证据:记忆 `project_first_trustworthy_curve`(2026-06-13 cosmos GRPO 持平=未学)、`project_flux_algo_validation`(ppo_epochs=1 让机制恒 0)、`project_fullparam_8bit_adam`(本轮建的 enabler)。配置:`configs/experiment/diffusion/cosmos_predict2/online_grpo_fullparam_8bit_240p.yaml`。

> **更新（2026-06-28）—— 数据阻塞已清、reward 真数据复验,阻塞换成 shape/checkpoint:**
> - **数据不再是阻塞**:936 行真实 LeRobot DROID target V2W manifest 已落盘(`data/external/video_world/manifests/droid_full_targets_{train,eval}.jsonl` = 936+64 行,1013 真首帧 + 1013 真 target clip);配置 `online_grpo_droid_full_target_240p.yaml` 已是 full-param(use_lora:false + optim_8bit + lr 2e-5 + ppo4 + per_sample)并指向它。§5 P0 的"数据前置门"达成。
> - **reward 真数据复验 PASS**:退役 pixel-L1 后的 DINOv2 + RAFT 默认,用真 DROID eval clip 跑 S4 判别探针(`future_reward_discrimination_probe`):dino gap_ratio=**0.298**(exact 0.826 ≫ blur 0.182)、motion gap_ratio=**0.969**(static 塌到 0.027 地板),均 PASS——不只合成 clip。
> - **真正的阻塞换成 shape/checkpoint**:240p_33f 在默认 720p 权重上生成垃圾(严重 OOD,见 `SPRINT_cosmos_robotic_data_factory_domain_rl` §5.5 point 6)。破局 = 模型**原生 480P** 权重(~10x 快于 704p,单卡 full-param 唯一可负担路径)。坑:HF repo `nvidia/Cosmos-Predict2-2B-Video2World` 里 480P 只是原始格式单文件 `model-480p-16fps.pt`(720P 才转好放在 `transformer/`),diffusers `from_pretrained` 只加载 `transformer/`=720P → **480P 必须先转成 diffusers transformer 再用,不是改 config 字段能解决**。
> - **新配置 `online_grpo_droid_full_target_480p.yaml`** = 240p 那份只改 shape→`/sampling/video/480p_33f` + `model.path`→转换后的本地 480P diffusers 目录(fail-fast:路径不存在即 `from_pretrained` 报错,绝不静默拿 720P 权重跑 480P 出垃圾)。§4 的 240p_33f 保留作机制/显存 smoke,但**真曲线走 480P**。

## 0. 一句话

之前 cosmos+Kling GRPO 端到端跑通但 **reward 全平**(-4.726→-4.69,噪声内)。根因诊断:① `ppo_epochs=1` 让 trust-region clip 恒 0(flux 验证实测,clip_fraction 恒 0);② LoRA 梯度太小(~2e-4),reward 推不动。三个训练侧修复仍成立——ppo_epochs=4 + full-parameter + 合适 lr——但当前 sprint 不能执行长曲线：**数据目标不成立**。Predict2 2B 是 Video2World，必须有真实 reference image；随机 smoke reference 或无关 global reference 会污染 RL 目标。

## 1. 三个修复(都已诊断,本 sprint 一起上)

| 修复 | 为什么 | 配置 |
|---|---|---|
| **ppo_epochs 1 → 4** | ppo_epochs=1 时 ratio 恒 1、clip 恒 0,trust-region 机制不存在(flux 验证证实);=4 才咬合(flux smoke 实测 clip_fraction=0.42) | `actor.ppo_epochs: 4` |
| **LoRA → full-parameter** | LoRA 梯度 ~2e-4 太小推不动 reward;full-param 给更大梯度(诊断点名) | `model.use_lora: false`(→ `apply_full_finetune`) |
| **lr → ~2e-5** | full-param 要比 LoRA 的 1e-4 低(paper §4.2.2 ~1e-5) | `actor.optim.lr: 2.0e-5` |

## 2. enabler:8-bit Adam 让 full-param 2B fit 单卡（已建+验证）

full-param 的障碍是**优化器显存**,不是权重:cosmos 2B 权重 bf16 才 4GB,但 **fp32 Adam 状态 ~16GB → OOM 32GB**。

- **已建**:`actor.optim.optim_8bit: true` → `bitsandbytes.AdamW8bit`(`vrl/trainers/online/trainer.py:_create_optimizer`),int8 Adam 状态 16GB→4GB。
- **Blackwell sm_120 实测跑通**;end-to-end 验证(fp32→AdamW,8bit→AdamW8bit)。
- **对 RL 安全**:量化的是优化器状态,**不是 forward** → 不碰 old_log_prob / 训推一致(和 fp8 forward 那种有损本质不同)。
- 显存:`权重 4 + grad 4 + 8bit 状态 4 + 激活(grad-ckpt) ≈ 15-20GB`(配 240p_33f,见 §3)。
- 待办:bnb 加进 pyproject deps(CI clean-install)。完全无损备选:DeepSpeed ZeRO-Offload(已装,fp32 状态放 94GB RAM,但集成量大)。

## 3. 第二约束:激活值 → 必须小 video

8-bit Adam 解了优化器状态,但 **video 激活值是另一大头**——现有 cosmos 配置注释:512p_93f LoRA rollout 就峰值 ~31.8GB。所以 full-param video 必须用**小 video shape**:

- `/sampling/video/240p_33f`(416×240, 33 帧,约 512p_93f 的 ~13% token)+ `gradient_checkpointing: true`。
- cosmos predict2 2B 是 **Video2World**,需 reference image 条件:`model.reference_image=/path/ref.png`(global)或 `cosmos.reference_mode=per_sample`(JSONL 带 reference_image)。

## 3.5 reference image:两种模式服务两个目的,别混（澄清 2026-06-27）

本 sprint 的"可信曲线"和"用参考图生成视频"是**两条不同的线**,reference 模式正好对应:

| 模式 | 数据 | reference image 的角色 | 适用 |
|---|---|---|---|
| **`global`** | `drawbench_train_192`(纯文本 prompt) | **占位条件**:一张固定参考帧配 192 条**无关** prompt(参考帧与 prompt 互相矛盾) | **只**用于 P0 显存 smoke + 验证三修复机制;**不是**"按参考图生成"的真演示(§7 已指出信号弱) |
| **`per_sample`** | `video_world_v2w`(droid 首帧 + 对应任务描述,真实配对) | **真条件**:每条 prompt 自带匹配首帧 | **这才是"rl train cosmos with reference image"的真身** |

**真身路径(per_sample)当前两个阻塞,各自一个 MR:**

1. **数据是 smoke**:`data/external/video_world/manifests/robot_train.jsonl` 只有 **3 train + 1 eval** 行、4 张参考图(`--limit` 小)。3 个 prompt 出不了可信曲线(固定 prompt 集太小,eval 也只有 1 条)。**MR-data**:用现成 importer 扩量 `python -m vrl.scripts.data video-world-bridge --repo-id lerobot/droid_100 --limit N`(首帧+caption 真实配对,参考图存 git 外)。
2. **没有 full-param 的 per_sample 配置**:现有 `online_grpo_v2w_reference.yaml` 是 **LoRA + 704p_93f**,与本 sprint 的 full-param 240p 论点冲突,且 704p 激活在 32GB 装不下 full-param。**MR-config**:新建 `video_world_v2w(per_sample) + 240p_33f + use_lora:false + optim_8bit + ppo_epochs:4`(=把 §4 的修复栈套到真 reference 数据上)。

**决策**:P0/P1 用 `global`+drawbench 先验机制与显存(reference 占位即可);真正的 reference-image 曲线走 per_sample,但需先落地上面两个 MR。

## 4. 配置（已写好）

`configs/experiment/diffusion/cosmos_predict2/online_grpo_fullparam_8bit_240p.yaml`:
```yaml
model: { use_lora: false }                       # full-param
actor:
  optim: { lr: 2.0e-5, optim_8bit: true }        # 修复 + enabler
  gradient_checkpointing: true
  ppo_epochs: 4                                   # 修复
algorithm: { clip_ratio: 1.0e-3, kl_coef: 0.0 }
# /sampling/video/240p_33f + /reward/kling_video_reward(本地) + total_epochs 300 + eval freq 25
```

**真曲线配方(2026-06-28 起,优先用这份)**:`online_grpo_droid_full_target_480p.yaml`
= 真 DROID per_sample(936 行)+ full-param + optim_8bit + ppo4 + **480p_33f**(原生,不 OOD)
+ dino+motion(真数据已 PASS)。唯一前置 = 转换 480P 权重(见顶部更新块)。上面的
240p_33f 配方降级为机制/显存 smoke(其生成质量不可用于读曲线)。

## 5. Phase plan

- **P0 — 数据前置门**:必须先有 source-backed V2W manifest（如 `video_world_v2w` 的 per-sample `reference_image`）。没有这个门,不跑显存 smoke、不读曲线。
- **P1 — 显存 smoke(决定可行性)**:`total_epochs=2 eval.enabled=false`,使用 source-backed per-sample reference。看两件:① 不 OOM(full-param 240p_33f 在 32GB fit)② `clip_fraction>0`(机制活)。**OOM 则降:33→若干帧 / 关 CFG / 退而 DeepSpeed offload。**
- **P2 — 短曲线(~50 更新)**:确认 reward 在动、drift guard / TIS-RS 指标健康、grad_norm 比 LoRA 的 ~2e-4 大(full-param 应明显更大)。
- **P3 — 满曲线(300 epoch)**:resumable(单卡争用,见记忆 `project_cosmos_reward_run_setup` 的 auto-resume wrapper)。
- **P4 — 判定**:见 §6。

## 6. 验收（可信判据,别自欺）

- **主判据:eval reward(固定 prompt,freq 25)涨 >2σ**——**不是**训练 `reward_mean`(它在轮换 prompt 上采样,反映 prompt 难度不是学习,flux 验证已证)。
- `clip_fraction > 0` 跨 epoch(机制活;若恒 0 说明信号没接上,别把曲线读成学习)。
- first-step log-prob diff ≈ 0(管线自洽,训推一致)。
- precision drift guard / TIS-RS 触发率可解释,不靠 mask 掉大量样本假装稳。
- **负结果也算交付**:若 full-param + ppo4 + 240p_33f 仍 <2σ,则证明问题不在这三个修复,记录并转向(reward 信号 / diffusion-loss 正则 / 数据)。

## 7. 已知坑(记忆)

- Kling reward 是**本地 HF model**(无 API key);global reference 模式"一张固定参考图 + 无关 prompt"信号弱——只拿来跑 P0 显存/机制 smoke,**真曲线走 per_sample**(见 §3.5,先补数据 + full-param 配置,别直接用 LoRA 704p 的 `online_grpo_v2w_reference`,它与本 sprint full-param 240p 论点冲突)。
- **Kling reward 不读 reference image**(只用 prompt+video 打分):架构上首帧被 `init_latents`/`cond_indicator` 钳死,但后续帧偏离参考图 reward **不惩罚**。若要奖励"与参考图一致",是单独一件事(§8 非目标),不在本 sprint。
- 别用 wan+OCR(结构性 absorbing-zero 死路)。
- `sampling.num_steps` 必须 ≥ `rollout.sde.window_range` 上限。
- 训练 reward_mean 会动但不可信(§6)。

## 8. 非目标

- 不在单卡上追 512p/704p/93帧 full-param(激活 OOM)——那要多卡 FSDP。
- 不上 fp8 forward / feature cache / 量化到 policy path(有损,污染 old_log_prob)。
- 不把 8-bit Adam 当"无损"——它是优化器状态近似(但不碰 forward,对 RL 安全);要完全无损用 DeepSpeed offload。
- 不在本 sprint 调 reward 基建(若信号弱是单独的事)。

## 9. 关键文件

- 配置:`configs/experiment/diffusion/cosmos_predict2/online_grpo_fullparam_8bit_240p.yaml`
- enabler:`vrl/trainers/online/trainer.py:_create_optimizer`、`vrl/trainers/core/types.py:OptimConfig.optim_8bit`
- full/LoRA gate:`model.use_lora`(`apply_lora` / `apply_full_finetune` 配对,本轮改名)、`vrl/models/diffusion/cosmos/predict2/runtime.py:77-89`
- 入口:`vrl/scripts/diffusion/cosmos/train.py:train_cosmos_predict2_grpo`(`_predict2_collector_kwargs` 是 global/per_sample 分叉 + global 缺图即 raise)
- reference 真身路径(§3.5):数据 importer `vrl/scripts/data/video_world.py`(`video-world-bridge`)、manifest `data/external/video_world/manifests/robot_{train,eval}.jsonl`(现 3+1 行,smoke)、dataset 配置 `configs/dataset/video_world_v2w.yaml`、LoRA 704p 配置 `online_grpo_v2w_reference.yaml`(待补 full-param 240p 变体)
- 证据:记忆 `project_first_trustworthy_curve`、`project_flux_algo_validation`、`project_fullparam_8bit_adam`、`project_cosmos_reward_run_setup`
