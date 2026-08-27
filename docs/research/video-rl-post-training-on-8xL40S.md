# 视频模型 RL 后训练：8×L40S 48GB 可复现方案调研

> 调研日期：2026-08-14
> 硬件前提：8×NVIDIA L40S 48GB（共 384GB 显存，PCIe，**无 NVLink**，~362 TFLOPS bf16 dense/卡，Ada Lovelace，支持 FP8）
>
> **后续修订（2026-08-16）**：实际硬件定为 AWS `g6e.12xlarge` = **4 × L40S 48GB**。
> 单卡规格不变，故本文的显存分析与选型结论**全部适用**；仅总算力与 wall-clock 减半。
> 执行计划、4 卡配置改动、以及本文标为"最大未解缺口"的显存问题的实测结果，
> 见 `docs/sprints/done/SPRINT_flash_grpo_wan_8gpu_repro.md`。
> 目标：(a) 至少一篇能端到端复现的论文（训练+评测，复现出论文报告的趋势）；(b) 一个活跃维护的开源 RL 代码底座，方便在上面做自己的算法改动。**可复现性优先。**
> 方法：6 个检索角度 → 23 个来源抓取 → 114 条断言 → 25 条进入三票对抗式验证 → 18 条确认、7 条否决。

---

## 目录

- [结论摘要](#结论摘要)
- [#1 复现目标：Flash-GRPO on Wan2.1-T2V-1.3B](#1-复现目标flash-grpo-on-wan21-t2v-13b)
- [#1b 低风险备选：Video-R1（视频理解 RLVR）](#1b-低风险备选video-r1视频理解-rlvr)
- [#2 可改造代码底座：EasyR1 / EasyVideoR1](#2-可改造代码底座easyr1--easyvideor1)
- [Reward Model：VideoReward (KwaiVGI/VideoAlign)](#reward-modelvideoreward-kwaivgivideoalign)
- [相邻但非 RL：GigaVideo-1](#相邻但非-rlgigavideo-1)
- [谨慎对待：VideoDPO](#谨慎对待videodpo)
- [明确"别碰"清单](#明确别碰清单)
- [建议执行顺序](#建议执行顺序)
- [必须坦白的局限](#必须坦白的局限)
- [附录 A：对抗验证中被否决的 7 条断言](#附录-a对抗验证中被否决的-7-条断言)
- [附录 B：开放问题](#附录-b开放问题)
- [附录 C：全部来源清单](#附录-c全部来源清单)
- [参考链接](#参考链接)

---

## 结论摘要

**"能复现的论文"和"能改的底座"是两件事，落在两个不同方向上。**

| 角色 | 推荐 | 一句话理由 |
|---|---|---|
| **#1 复现（视频扩散 RL）** | **Flash-GRPO** on Wan2.1-T2V-1.3B | 唯一一篇作者亲自提供 8 卡单机 ZeRO-2 脚本（~40h）并附训练曲线的视频扩散 RL 论文 |
| **#1b 复现（低风险、低显存）** | **Video-R1** | 复现包发全了——含中间 SFT checkpoint，可跳过最贵的阶段；纯自回归 rollout，比去噪便宜几个数量级 |
| **#2 可改底座** | **EasyR1**（成熟）/ **EasyVideoR1**（视频专用但年轻） | EasyR1 官方硬件表 7B 全参 GRPO = 8×40GB，卡数正好对上 |
| **Reward Model** | **VideoReward** (Qwen2-VL-2B) | ZeRO-0 训练、~72 A800 GPU-hours，单卡 L40S 就能跑甚至重训 |

**别碰**：Wan 14B 类视频 RL、32B/72B VLM 全参 GRPO、VideoAlign 的 Flow-DPO *视频*结果（放出来的 DPO 代码是纯文生图）、把 ZeRO-3/FSDP full-shard 当默认策略。

---

## #1 复现目标：Flash-GRPO on Wan2.1-T2V-1.3B

**论文**：ICML 2026 · [arXiv:2605.15980](https://arxiv.org/abs/2605.15980) · [ICML poster](https://icml.cc/virtual/2026/poster/63629)
**代码**：https://github.com/Shredded-Pork/Flash-GRPO
**置信度**：high（3-0 通过对抗验证）

### 为什么是它

这是本次调研中**唯一一篇作者亲自提供 8 卡单机脚本的视频扩散 RL 论文**——不是社区推断，不是二手移植。

README 里同一个 1.3B backbone 给了两套配置：

```
Flash-GRPO 96GPUs -> scripts/multi_node/train_wan2_1_flash.sh
Flash-GRPO 8GPUs  -> scripts/multi_node/train_wan2_1_flash_1node.sh   ~40hours
```

脚本内容确认了拓扑：

```bash
accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
  --num_machines 1 --num_processes 8 ... --config config/dgx.py:wan2_1_flash_1node
```

**关键点是 ZeRO-2 而不是 ZeRO-3** —— 这恰好是 PCIe 无 NVLink 机器最需要的选择（ZeRO-3 的参数 all-gather 会把互联打爆）。

News log 显示这是有意的作者交付，不是顺手：
- `[2026-05-11]` 代码发布，承诺后续给 8 卡版本
- `[2026-05-28]` 8 卡版本发布，**并附带训练曲线**

### 仓库健康度

| 指标 | 值 |
|---|---|
| 创建 | 2026-05-11 |
| 最后 push | 2026-06-11 |
| 最后更新 | 2026-08-03 |
| Star | 64 |
| 开放 issue | 5 |

### Rollout 成本（论文写死了）

> "We tailor the sampling schedule during training: we utilize **20 sampling steps for the 1.3B model** and an accelerated 12 sampling steps for the 14B model"
> "The classifier-free guidance (CFG) scale is fixed at **4.5**"
> "For the held-out evaluation set, we perform inference using **50 sampling steps**."

即：训练用 20 步 + CFG 4.5，评测才用 50 步。训练/评测的切分是论文自己的限定词，所以 20 步确实是真实 rollout 成本，不是取巧。

### 计算预算

> Section 5.2 原文："With **350 GPU hours** of training on Wan2.1-T2V-1.3B, Flash-GRPO achieves the highest Aesthetic Quality (66.43) and Subject Consistency (98.70), outperforming both Flow-GRPO-Fast1 and Flow-GRPO."
> Abstract/Fig.1："Flash-GRPO achieves **6x acceleration** in training cost while attaining higher evaluation performance."

单机脚本的 ~40h × 8 GPU = ~320 GPU-hours，与 350 这个数字内部自洽。

### ⚠️ 风险（必须说清楚）

1. **论文从头到尾没写用的什么卡。** 三次独立抓取确认 Implementation Details (5.1) 写了采样步数、CFG scale、模型变体，但**完全省略硬件**。alphaXiv 独立指出该论文 "does not specify which GPU hardware was used ... including the 350 GPU-hour budget figure"。配置文件叫 `dgx.py`——暗示可能是 80GB 的 A100/H100 节点。

2. **论文没写视频分辨率、帧数、GRPO group size G。** 三次抓取确认全部缺失。这三个恰恰是算 rollout 显存的乘数——步数和 CFG 只给了每段视频的去噪倍数，没有分辨率/帧数/G 就算不出总显存和总时间。

3. **wall-clock 外推**：按算力线性外推（H100 ~990 vs L40S ~362 TFLOPS bf16），你这边可能是 **80–120 小时**而不是 40 小时。一次完整复现大概三到五天。

**好消息**：第 1、2 点**一小时就能自己回答**——把脚本跑起来，`nvidia-smi` 看一步。如果 OOM，杠杆很明确：
```
开 gradient checkpointing → 减小 G → 减帧数 → 最后才考虑 ZeRO-3/offload
```

---

## #1b 低风险备选：Video-R1（视频理解 RLVR）

**代码**：https://github.com/tulerfeng/Video-R1
**置信度**：high（3-0）

### 为什么选它建立信心

这是本次调研中**唯一把复现包发全的**项目：

| 产物 | 链接 | 说明 |
|---|---|---|
| SFT 冷启动数据 | `Video-R1-COT-165k.json` | COT rationales 由 Qwen2.5-VL-72B 生成 |
| RL 数据 | `Video-R1-260k.json` | 同在 [Video-R1-data](https://huggingface.co/datasets/Video-R1/Video-R1-data) |
| **中间 SFT checkpoint** | [Qwen2.5-VL-7B-COT-SFT](https://huggingface.co/Video-R1/Qwen2.5-VL-7B-COT-SFT) | 真实 BF16 safetensors，8B params，Apache-2.0，**月下载 ~674 次**（有真实第三方使用） |
| 最终模型 | [Video-R1-7B](https://huggingface.co/Video-R1/Video-R1-7B) | 另有 ModelScope 镜像 |

README 明说可以跳过 SFT：

> "If you want to skip the SFT process, we also provide one of our SFT models at [Qwen2.5-VL-SFT]"

SFT checkpoint 的 model card 自述为 "cold start model for further RL training"。**这直接省掉最贵、也是 issue 里唯一报 OOM 的那个阶段**（issue #124 就是 SFT 阶段 OOM）。

两个阶段脚本都在：`run_sft_video.sh`、`run_grpo_video.sh`、`run_grpo_vllm_qwen25vl.sh`。

对抗检索没有找到任何"发布的 checkpoint 是坏的"的报告：issue #108 是用户**自己复现的** SFT 输出有问题（已关闭），#124 是 SFT 阶段 OOM——正是跳过 SFT 能避免的那个失败。

### Rollout 参数（在提交的脚本里查证过，不是 README 吹的）

```bash
--max_prompt_length 16384
--max_completion_length 768
--num_generations 8          # 脚本内联注释："the number of outputs G in grpo"
--max_pixels 401408
CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4
```

脚本内联注释明确写了降低 `num_generations` 可以降训练时间和显存——**这是有文档的显存杠杆**。

README 补充：训练时限制最多 16 帧，每帧最大分辨率 128×28×28，`per_device_train_batch_size=1`。

> **数字对不上的疑问已解决**：`401408` vs `100352` 的差异是因为 Qwen2.5-VL 的 ViT 做 2×2 空间 patch merging，`401408 = 4 × 100352`——同一个设置在流水线两个阶段的两种表示。

**没有去噪循环**——纯自回归采样，成本比视频扩散 RL 低几个数量级。另有 vLLM 加速版脚本（README 注明 vLLM 0.7.2）。

> ⚠️ **注意**：16 帧上限是数据侧约定，委托给 `qwen_vl_utils.process_vision_info()`，**并未在 `grpo.py` 或 `grpo_trainer.py` 中强制执行**（两个文件都抓取过，都不含帧数逻辑）；`grpo.py` 默认 `max_pixels=12845056`，靠脚本 flag 覆盖。

### ⚠️ 风险：单卡容量，不是总显存

README 声称 Video-R1-7B "can be easily trained using 4 H20 (96GB) GPUs, or 5 A100 (80G) GPUs"。

- 总显存 320–400GB **与 8×L40S 的 384GB 吻合**
- 但 48GB/卡大约是人家假设的**单卡预算的一半**
- 长视觉 token 序列恰恰是**激活显存**（而非参数显存）主导的场景——所以它是重新分片，不是按算术就能装下

Issue tracker 里的真实摩擦：
- [#124](https://github.com/tulerfeng/Video-R1/issues/124) "OUT OF MEMory on 4 a100 gpus"（SFT 阶段）
- [#125](https://github.com/tulerfeng/Video-R1/issues/125) NCCL timeout / ECC

> 🔴 **重要**：关于 Video-R1 **确切的 ZeRO stage** 和**确切的硬件需求**的两条更强断言在验证中被 **0-3 否决**。因此"4×H20 或 5×A100"是唯一有来源的硬件陈述，**不要假设特定的 DeepSpeed 配置**。

### 🔴 隐藏成本：数据

数据集是**纯标注 JSON**，指向外部语料：CLEVRER、LLaVA-Video-178K、NeXT-QA、PerceptionTest、STAR。

**总共约 309GB，且有 YouTube 链接失效风险。** "publicly released" ≠ "download-and-go"。先确认存储和下载能力，并预期会缺一些片段。

---

## #2 可改造代码底座：EasyR1 / EasyVideoR1

### EasyR1（成熟，推荐）

**代码**：https://github.com/hiyouga/EasyR1
**置信度**：high（2-1）

官方硬件表（两次独立抓取逐格验证——raw README on main + 渲染页）：

| 策略 | 1.5B | 3B | 7B | 32B | 72B |
|---|---|---|---|---|---|
| GRPO Full FT (AMP) | 2×24GB | 4×40GB | **8×40GB** | 16×80GB | 32×80GB |
| GRPO Full FT (BF16) | 1×24 | 1×40 | 4×40 | 8×80 | 16×80 |
| GRPO LoRA (AMP) | 1×12 | 1×24 | **2×32** | 2×80 | 4×80 |

BF16 行绑定 `worker.actor.fsdp.torch_dtype=bf16` + `worker.actor.optim.strategy=adamw_bf16`（无 fp32 master weights），所以 "pure BF16" 的说法是准确的。

**7B 全参 GRPO 要 8×40GB —— 卡数正好对上，48 > 40。**

仓库健康度：214 commits，5.1k stars，383 forks，未归档，无弃用声明。

> ⚠️ **两个重要限定**：
> 1. 表格自带 `* estimated` 脚注，且**未写序列长度 / batch size / 帧数**；48GB 对 40GB 只有 ~20% 余量，对视频 VLM 要当**地板**而非保证。
> 2. **"EasyR1 开箱支持视频"这个断言在对抗验证中被否决（1-2）**。它的 dataset pipeline 可能接受 video key，但没有建立证据。**不要假设视频数据路径不 fork 就能用。**

### EasyVideoR1（视频专用，但年轻）

**代码**：https://github.com/cyuQ1n/EasyVideoR1 · **论文**：[arXiv:2604.16893](https://arxiv.org/abs/2604.16893)（2026-04-18 提交）
**置信度**：medium（2-1）

论文摘要原文：

> "extending RLVR to video understanding becomes increasingly important yet remains largely unexplored... Existing open-source RL training frameworks provide solid infrastructure for text and image scenarios but lack systematic optimizations tailored for video modality. In this work, we present EasyVideoR1, a complete and efficient reinforcement learning framework specifically designed for training large vision-language models on video understanding tasks."

**支持**（列为已支持，非 coming-soon）：
- 算法：GRPO、DAPO、GSPO、CISPO、Reinforce++、ReMax、RLOO、GDPO（"and more"）
- 模型：Qwen2-VL / Qwen2.5-VL / Qwen3-VL / Qwen3.5-VL 系列

算法列表反映真实代码（2026-04-21 commit "Add GDPO support to trainer and update README"）。血统独立确认：致谢里 credit EasyR1 和 veRL，构建在 veRL 的 HybridEngine 上，同为 Apache-2.0。

#### 🔴 成熟度警告

| 指标 | 值 |
|---|---|
| commits | ~30 |
| stars | 153 |
| forks | 3 |
| 创建 | 2026-03-09 |
| 最后 push | 2026-04-27（截至 2026-08-14 已停 ~3.5 个月） |
| 发布的 checkpoint | **0** |
| README 中的显存/GPU 数字 | **0** |

**它适合当 (b) 可改底座，不适合当 (a) 复现目标。**

#### 默认配置几乎肯定在 48GB 上 OOM（但逃生口有文档）

`examples/video_rl/video_rl.yaml` 逐行验证：

```yaml
trainer:
  nnodes: 1
  n_gpus_per_node: 8
model_path: Qwen/Qwen3-VL-8B-Instruct
worker:
  actor:
    fsdp:
      enable_full_shard: true      # 兄弟配置注释为 "# Full sharding (ZeRO-3)"
      enable_cpu_offload: false
    offload:
      offload_params: false        # ← 默认全关
      offload_optimizer: false
    ulysses_size: 1                # "# Ulysses sequence parallelism degree"
    global_batch_size: 256
  rollout:
    tensor_parallel_size: 2
    gpu_memory_utilization: 0.8
    n: 8                           # "# Total samples: 7 online + 1 offline when offline data exists, otherwise 8 online"
  ref:
    offload:
      offload_params: false
data:
  video_fps: 2.0
  video_max_frames: 128
  max_pixels: 262144
  video_max_pixels: 262144
  rollout_batch_size: 256
algorithm:
  adv_estimator: grpo
```

8B VLM + FSDP full-shard + offload 全关 + `max_token_len_per_gpu=24096` + 128 帧 @2FPS + rollout n=8 + vLLM 占 0.8 显存 —— 很可能超 48GB。这正是为什么逃生口被写进 FAQ：

> README line 198, "Q: CUDA out of memory"：
> "Decrease `worker.rollout.gpu_memory_utilization` and enable `worker.actor.offload.offload_params`."

#### ✅ 对 L40S 友好的一点

`run_video_rl.sh` 里 `export NCCL_NVLS_ENABLE=0` —— **配方不假设 NVLink switch 特性**。

#### 视频理解 RL 的 rollout 为什么结构上更便宜

没有去噪循环。rollout 参数全是自回归 vLLM 旋钮（temperature、top_p、`max_num_batched_tokens=25000`、`tensor_parallel_size=2`、`max_response_length=4096`），**任何地方都没有 diffusion step count**。

`examples/unified_rl/unified_rl.yaml` 逐值佐证（L43 fps、L44 frames、L50 pixels、L92 batch、L131 n）。`gh api contents/examples/video_rl` 确认 `video_rl.yaml` 是该目录唯一配置，无竞争默认值。

> ⚠️ **规模警告**：256 × 8 = **2048 个生成响应/step** 是很大的 step 预算。要塞进 48GB，**先降 `rollout_batch_size`，再降 `n`**。

---

## Reward Model：VideoReward (KwaiVGI/VideoAlign)

**代码**：https://github.com/KlingAIResearch/VideoAlign
**权重**：https://huggingface.co/KwaiVGI/VideoReward · https://huggingface.co/KlingTeam/VideoReward
**论文**：[arXiv:2501.13918](https://arxiv.org/abs/2501.13918)（NeurIPS 2025）
**置信度**：high（3-0）

### 整条链路里最便宜的一环

Backbone 由三个独立一手来源确认：README "Our reward model is based on QWen2-VL-2B-Instruct"；HF card "Finetuned from Qwen/Qwen2-VL-2B-Instruct"；arXiv 2501.13918 implementation section backbone "Qwen2-VL-2B"（~675M ViT + Qwen2-1.5B LLM）。

`train.sh` 实际用的是：

```bash
deepspeed ds_config/zero0.json
```

**ZeRO stage 0 = 完全不做 optimizer/gradient 分片，每卡持有完整拷贝** —— 这既证明多卡分片没有必要，也让 L40S 的无 NVLink 弱点在这里完全无关。

其他参数：`per_device_train_batch_size=1`、`gradient_accumulation_steps=4`、`lora_enable=True`（rank 64, alpha 128, vision LoRA disabled）。

论文：**~72 A800 GPU-hours** 总计，batch 32，lr 2e-6，2 epochs。6 个开放 issue 中**零 OOM/显存报告**。

### ⚠️ 两个坑

1. **`train.sh` 设 `gradient_checkpointing=False` + `max_frame_pixels=200704`（~448×448）@ fps 2** —— 这是给 80GB A800 调的，是 48GB 上**最可能 OOM 的单点**，但改一个 flag 就行。

2. **论文写**："LoRA is applied to update all linear layers in the language model, while the **vision encoder's parameters are fully optimized**" —— 那 ~675M 的 ViT 是**全参微调带优化器状态**，显存比 "LoRA" 这个词暗示的高。

其他：[issue #26](https://github.com/KlingAIResearch/VideoAlign/issues) 显示**多卡推理**在 H20 上有真实的 RoPE 维度不匹配 bug，进一步支持单卡使用。issue #20 "Running without Flash Attention"（已关闭）确认存在非 FA 回退路径。

### 发布内容与两处常见错误更正

Repo contents API 确认存在：`train.sh`、`ds_config/`、`inference.py`、`eval_videogen_rewardbench.py`、`LICENSE`、`trainer.py`、`data.py`、`calc_accuracy.py`、`environment.yaml`、`vision_process.py`、`prompt_template.py`、`checkpoints/`、`datasets/`。

> 🔴 **更正 1：训练入口是 `train_reward.py`，不是 `train.py`**（repo 中不存在 `train.py`）。
>
> 🔴 **更正 2："全部 MIT" 是错的**。`LICENSE` 原文 "MIT License / Copyright (c) 2025 Kling Team, Kuaishou Technology"（SPDX MIT）**只覆盖 GitHub 代码**；两个 HF checkpoint repo 都标 **apache-2.0**，且叠加在 Qwen2-VL 自己的上游许可之上。

HF API 对 `KwaiVGI/VideoReward` 列出 `checkpoint-11352/model.pth`、`model_config.json`、`checkpoint-11352/tokenizer/*`，`gated=false`。

### 🔴 当冻结打分器用，别指望重训到论文精度

README："Prepare your own data as the instruction stated" —— **论文背后的人类偏好训练数据从未发布**，只放了 [VideoGen-RewardBench](https://huggingface.co/datasets/KwaiVGI/VideoGen-RewardBench) 评测集。

Tracker 里的复现困难信号：
- #29 "Cannot reach accuracy from Table 2"（2025-08-29 关闭）
- #22 精度差异，训练指标 ~0.8 vs 手工验证 ~0.5（2025-07-31 关闭）
- #31 "rm_head.weight not initialized from checkpoint"
- #15 "Loading state_dict error"

### 维护状态：比表面更旧，但不影响它的实际角色

GitHub commits API：
- 最近 = 2025-09-24 "add LICENSE"（+21/-0，只加 LICENSE）
- "add eval prompts" 2025-08-14（只改 README.md，+1/-0）
- "add flow-dpo" 2025-07-17（**只改 README.md**，+18/-5）

**三个 2025 年的 commit 都没改代码——最后一次真实代码 commit 是 2025-02-09 的 "fix bug"。** 比"约 11 个月没提交"的说法还要旧。

元数据：`pushed_at` 2025-09-24，**单分支**（main @ 219ab9db，无隐藏 dev 分支），`archived=false`，490 stars，7 开放 issue，共 18 commits。PR #9 自 2025-04-14 开放未合并。镜像 KwaiVGI/VideoAlign 与 KlingTeam/VideoAlign 均无开发迁移迹象。

> ✅ **反向权重**：issues API 显示 2026 年仍有互动——issue #1 更新至 2026-03-02（11 条评论），一个 2025-07-22 的 issue 更新至 2026-01-19（7 条评论），有 issue 关闭晚至 2025-12-20，还有一个 2026-01-20 的开放 issue。
>
> **"没有 commit" ≠ "没有维护者响应"**，而且一个带已发布 checkpoint 的冻结 reward model 不需要活跃提交就能用。**仓库陈旧度对 (b) 槽位远比对 (a) 槽位重要。**

---

## 相邻但非 RL：GigaVideo-1

**论文**：[arXiv:2506.10639](https://arxiv.org/html/2506.10639v1) · [OpenReview](https://openreview.net/forum?id=y7wAuwErpL)
**置信度**：high（3-0 / 2-1）

### 它不是 RL 论文

对 Wan2.1-T2V-1.3B 做**全参微调**，约 4 GPU-hours。但它的 loss 是 **reward 加权的监督回归 + 离线打分**——**没有 policy gradient，没有 rollout 循环，没有 GRPO/DPO**。

方法："As the baseline T2V generation model, we choose Wan2.1-T2V-1.3B"，作者 "perform full-parameter fine-tuning on its transformer to improve its expressiveness."

Eq. 3 原文：

```
L_Ps(θ) = E[ -r_φ(x,p) * ||u(z_t, p, t; θ) - v_t||² ]
```

即一个被标量 reward 缩放的 flow-matching 回归项，reward 来自**冻结的** LLaVA-Video-7B-Qwen2（外加 YOLO-World 和 CoTracker2 负责实例保持与相机运动维度）。

"离线"是论文自己采纳的选择，不是消融后被否的变体：论文定义在线重加权为 "predicted during training based on intermediate denoised frames"，离线为 "computed in advance from synthetic videos prior to training"，并结论 "our offline reweighting achieves the best high accuracy with minimal training overhead"。~9.5k 合成视频在微调前预生成，所以训练内部**确实没有 per-prompt 采样循环**。

Tables 1–7 **不含任何 DPO 或 GRPO baseline**；Table 3 只消融 SFT / filtered SFT / online / offline reweighting。

训练参数：832×480，81 帧，16 FPS，batch 4，lr 1e-6，1 epoch。

### 🔴 "4 GPU-hours" 无法移植

1. **论文从不写 GPU 型号，也不报任何显存数字。** 三次独立抓取（arXiv HTML 全文含附录/致谢、abs 页、项目页）确认**无任何设备名**——没有 A100/H100/H800/A800——且**零内存数字**。没有设备的 GPU-hour 在移植时是量纲无意义的；未指明的 A100/H100 与 L40S 有效吞吐差 2–3 倍。

2. **算术对不上自己的消融行。** Table 3：`Wan+SFT: 5.75h/epoch`、`Wan+SFT(filtered): 0.90h/epoch`、`Wan+Reweight_offline: 0.90h/epoch`；正文佐证 "reducing training time by more than 6x"（5.75/0.90 = 6.4x）。**0.90 h/epoch × 1 epoch = 0.90 GPU-hours，不是 4** —— 标题数字必然聚合了论文未逐项说明的额外阶段。

3. **~9.5k 合成视频的一次性生成开销**是真实推理算力，被排除在"仅微调"的标题数字之外。

4. **发布状态未验证**：论文称 "Code, model, and data will be publicly available"，但截至 2026-08 **找不到官方 GitHub 仓库**——只有 awesome-list 引用它。复现很可能需要重新实现。

### ✅ 一条有用的旁证

显存笔记：在 832×480×81 帧（~30k 激活 token）下，主导显存的是**序列长度而非参数量**；余量取决于 gradient checkpointing 远多于 ZeRO stage，而 PCIe 无 NVLink 上的 ZeRO-3 会在 L40S 上通信受限。

佐证可行性：[VideoX-Fun 官方 Wan2.1 全参微调指南](https://github.com/aigc-apps/VideoX-Fun/blob/main/scripts/wan2.1/README_TRAIN.md) 规定用 DeepSpeed-Zero-2 或 FSDP 配 `--gradient_checkpointing --mixed_precision=bf16`，81 帧，**并有一份 Wan2.1-1.3B 全参微调跑在恰好一个 8×L40S 48GB 节点上、ZeRO-2 的报告**。

---

## 谨慎对待：VideoDPO

**代码**：https://github.com/CIntellifusion/VideoDPO（CVPR 2025）
**置信度**：high（3-0）

### 显存侧：轻松装下

`configs/vc2_dpo/config.yaml`（注意是 `vc2_dpo` 不是 `vc_dpo`）逐行验证：

```yaml
model_channels: 320
context_dim: 1024
temporal_length: 16
use_checkpoint: true
resolution: [320, 512]
video_length: 16
batch_size: 1
accumulate_grad_batches: 2
# strategy: deepspeed        # ← 被注释掉了
```

UNet ~1.4B（据 CVPR2025 论文）。DPO 参考模型是真实常驻的——`lvdm/models/ddpm3d.py:141-144` 构建了第二个完整 UNet：

```python
self.ref_model = DiffusionWrapper(unet_config, conditioning_key)  # requires_grad=False
```

内部一致性：`get_batch_input` 做 `x = torch.cat(x.chunk(2, dim=1))`（win/lose 配对），`run.sh` 用 `--nproc_per_node=4`，global batch = 1×2×4 = 8，**精确匹配论文的 "4 Nvidia A100 GPUs, global batch size 8"**。

由于 `run.sh` 是纯 DDP **无任何分片**，每张 A100 已经同时持有可训 UNet + AdamW + 冻结 ref —— 约 17GB 优化器 + 3–6GB ref + checkpointed 激活，**舒适地低于 48GB**。

### 🔴 复现侧：这才是要小心的地方

[Issue #12](https://github.com/CIntellifusion/VideoDPO/issues/12) "Can not reproduce the results in paper"，**自 2025-08-15 开放至今，comments=0，`updated_at == created_at` 证明从未有任何回复。**

报告者原文：

> "I use VC2 following exact steps described in README, and bash script_sh/inference_t2v.sh ... but the generated video quality is not good."

上下文：仓库 `pushed_at` 2025-06-01（**在这个 issue 之前就已停更**），169 stars，12 个开放 issue 中 9 个评论数 ≤2，issues #13/#9/#8/#3 也是 0 评论。

> ⚖️ **解释上的界限**：报告者自己的附件显示 **baseline VC2 和 VC2+DPO 两者质量都差**，暗示是报告者侧的环境问题。**把这当作"复现无人支援 + 维护者不响应"的证据，而不是"论文复现不了"的证明。**

> 🔴 **三条更严厉的 VideoDPO 断言在验证中被 0-3 全部否决，不要重复它们**：OmniScore 代码未发布、4-GPU DDP 能直接映射到 8×L40S、"实质上无人维护"。

---

## 明确"别碰"清单

**置信度**：medium（2-1）

| 项目 | 原因 |
|---|---|
| **Wan 14B 类视频 RL** | 14B 光 FP8 **推理**就要 40–48GB，优化器状态和激活没地方放。Flash-GRPO 自己的 14B 路径把去噪砍到 12 步作为加速手段，且**跑在 96 卡规模上** |
| **32B / 72B VLM 全参 GRPO** | EasyR1 验证过的表格：32B full FT AMP = 16×80GB = 1280GB，72B = 32×80GB —— 对比你的 8 卡 384GB。**超 3 倍显存、2 倍卡数** |
| **VideoAlign 的 Flow-DPO *视频*结果** | 见下方专节——**放出来的 DPO 代码是纯文生图的** |
| **把 ZeRO-3 / FSDP full-shard 当默认策略** | PCIe 无 NVLink 会通信受限。见下方旁证 |

### 为什么 VideoAlign 的 Flow-DPO 视频结果不可复现（3-0）

README 原文：

> "For Flow-DPO, we provide an implementation for **text-to-image tasks** [here]"

链接指向 [`flow_grpo/scripts/single_node/dpo.sh`](https://github.com/yifan123/flow_grpo/blob/main/scripts/single_node/dpo.sh)。顺着链接走（而不是停在引文）：该脚本只有一行——

```bash
accelerate launch ... scripts/train_sd3_dpo.py --config config/dpo.py:geneval_sd3
```

`config/dpo.py` **只有两个入口**：`geneval_sd3()`（L30）和 `pickscore_sd3()`（L68），**两个都设** `pretrained.model='stabilityai/stable-diffusion-3.5-medium'`，输出到 `logs/geneval/sd3.5-M-dpo` 和 `logs/pickscore/sd3.5-M-dpo`。**零视频模型引用。**

对抗检查：对 `train_sd3_dpo.py` grep `video|wan` 得到 12 处命中，**全部是 `wandb` / `wandb.Image` / `wandb.log`** —— 那个 DPO trainer 记录的是**图像**。

VideoAlign 仓库根目录只有 reward model 资产，**根本没有任何生成模型 DPO 代码**。开放 issue "Code for DPO-finetuning?"（2025-03-15，**至今仍开放**）加上已关闭的 "About DPO training"（2025-05-20）佐证用户要过视频 DPO 代码但没拿到。

> ✅ **范围说明**：VideoReward 仍可以作为 reward 信号，通过活跃维护的 [flow_grpo](https://github.com/yifan123/flow_grpo)（pushed 2026-05-07，2468 stars）用于 Wan2.1-T2V-1.3B 的 Flow-GRPO 视频训练。**那才是 8×L40S 的实际路径。**

### 为什么避开 ZeRO-3（架构推理，非实测）

旁证高度一致：

| 项目 | 分片策略 |
|---|---|
| Flash-GRPO 8 卡脚本 | `deepspeed_zero2.yaml`（刻意不用 ZeRO-3） |
| VideoAlign `train.sh` | ZeRO stage **0** |
| VideoDPO `run.sh` | 纯 DDP，无分片 |
| EasyVideoR1 `run_video_rl.sh` | `export NCCL_NVLS_ENABLE=0` |
| VideoX-Fun Wan2.1 指南 | 先 ZeRO-2，**仅在"显存不足时"**才升级到 FSDP |

> ⚠️ **注意这是从配置选择得出的架构推断，不是来自已发表的 L40S all-reduce 基准。本次调研中没有任何来源实测过这些负载的 PCIe all-reduce 吞吐。**

---

## 建议执行顺序

```
1. 先花一小时：跑 train_wan2_1_flash_1node.sh，nvidia-smi 看一步
   ↓ 这是整份调研中性价比最高的实验——一次性回答唯一悬而未决的关键问题
   │
   ├─ 装得下  → 就是它。Flash-GRPO 复现 + 改造一套搞定
   │
   └─ OOM 且降 G/帧数救不回来
        → 转 Video-R1（跳过 SFT，直接从发布的 checkpoint 开 RL）
        → 底座用 EasyR1 / EasyVideoR1

2. reward model 无论走哪条路都用 VideoReward 冻结打分
```

**OOM 杠杆（按优先级）**：gradient checkpointing → 减小 group size G → 减帧数 → ZeRO-2 升 ZeRO-3 → param offload。

---

## 必须坦白的局限

### 🔴 最大的未解缺口

**本次调研中没有任何一个来源给出过任何硬件上任何这些负载的实测显存数字。** 所有"能装进 48GB"的结论都是从配置文件（ZeRO stage、batch size、gradient checkpointing flag、参数量）推出来的——推得有据，**但仍是推理**。

Flash-GRPO 尤其如此：它从不说明其 8 卡节点的 GPU 型号或显存（配置文件仅命名为 `dgx`），所以决定性的问题——**作者的 8 卡配方是装 48GB 卡还是为 80GB 卡搭的**——真正是开放的，必须靠跑来解决。**为第一次 OOM 做好预算，并提前知道你的杠杆。**

> **2026-08-16 部分解决**：本文写作时缺失的三个数字（分辨率、帧数、GRPO group size）
> 已从 `config/dgx.py` 读出——480×832、81 帧、G=4，且**是 LoRA r=16 而非全参微调**，
> gradient checkpointing 默认已开。基座推理在单卡实测 **21.98GB**（训练几何），
> 开 `enable_model_cpu_offload()` 后降至 **10.81GB**（仅慢 8%）。
> 仍未解决的是**训练侧**峰值（反传激活 + rollout buffer），那是 sprint 的 Gate 0。
> 细节见 `docs/sprints/done/SPRINT_flash_grpo_wan_8gpu_repro.md` §3.2–3.3。

### 🔴 方向 #4 完全未覆盖

**世界模型 / 具身视频策略（Cosmos、GR00T 一类）没有任何断言通过验证。** 本报告完全没有涉及该方向。

调研确实抓到了以下来源，但相关断言没能撑过对抗验证：
- https://github.com/nvidia-cosmos/cosmos-rl/blob/main/examples/ddrl.md
- https://github.com/RLinf/RLinf
- https://github.com/thuml/RLVR-World
- https://huggingface.co/blog/nvidia/cosmos-fine-tuning-for-robot-video-generation

**如果这个方向对你重要，需要单独再跑一轮调研。**

### ⚠️ 两条断言只有单一来源，无独立佐证

1. **EasyR1 的硬件表** —— 它自己的 `* estimated` 脚注是唯一限定词，且验证会话在检查 GitHub issues 寻找矛盾的 OOM 报告之前就耗尽了检索预算。
2. **GigaVideo-1 的整个画像** —— 一篇 arXiv 预印本，无代码。

**两者都要相应打折。**

### ⏱️ 时效性（评估于 2026-08-14）

| 项目 | 状态 |
|---|---|
| Flash-GRPO | 🟢 最新（pushed 2026-06-11，updated 2026-08-03） |
| flow_grpo | 🟢 最新（pushed 2026-05-07，2468 stars） |
| EasyVideoR1 | 🟡 中度陈旧（~3.5 个月） |
| VideoAlign | 🔴 陈旧但作为冻结资产可用（最后**代码** commit 2025-02-09——比其 2025-09-24 的 LICENSE commit 暗示的更旧） |
| VideoDPO | 🔴 自 2025-06-01 起停更 |

---

## 附录 A：对抗验证中被否决的 7 条断言

**这些不得再次出现。** 每条都经过 3 票对抗验证并被判定为不成立。

| # | 被否决的断言 | 票数 | 来源 |
|---|---|---|---|
| 1 | Flash-GRPO 是全参微调而非 LoRA，且论文与 README 都不报 GPU 型号或单卡显存，因此作者 8 卡脚本的显存足迹对 48GB L40S 未经验证、可能需要额外 offload/分片 | 0-3 | Flash-GRPO |
| 2 | VideoDPO 官方启动脚本只用单机 4 卡纯 PyTorch DDP（无模型/优化器分片），因此不依赖 NVLink 级互联，可直接映射到 8×L40S PCIe 硬件 | 0-3 | VideoDPO |
| 3 | 实现 OmniScore（论文核心偏好打分贡献）的代码以及整体训练流水线从未发布，使得仅凭官方仓库无法忠实端到端复现 | 0-3 | VideoDPO |
| 4 | VideoDPO 仓库截至 2026 实质上无人维护——最后实质代码 commit 是 2025-01-12，最后任何形式的 commit 是 2025-04-24 的 README 编辑，承诺的 CogVideoX 代码仍未发布 | 0-3 | VideoDPO |
| 5 | Video-R1-7B（Qwen2.5-VL-7B base）全参 T-GRPO 训练官方报告需 4×H20 96GB 或 5×A100 80GB，即 ~384-400GB 总显存——数值上匹配 8×L40S 的 384GB，但需要 ZeRO-3 分片来补偿更小的 48GB 单卡容量 | 0-3 | Video-R1 |
| 6 | Video-R1 官方 GRPO 训练脚本使用 DeepSpeed ZeRO-3（`local_scripts/zero3.json`）、`per_device_train_batch_size=1`、`gradient_accumulation_steps=1`、`gradient_checkpointing=true`、bf16、Flash Attention 2——即省显存选项已全开、LoRA 无额外余量（训练是全参、无 LoRA 路径），所以 48GB 卡除了降 `num_generations` 或帧数外没有剩余杠杆 | 0-3 | Video-R1 |
| 7 | EasyR1 的 dataset pipeline 原生支持视频输入（`video_key="videos"`、可配 `video_fps`、min/max pixels），因此视频理解 VLM 的 RLVR 开箱即用无需 fork——尽管 README 未宣传视频 | 1-2 | EasyR1 |

**由此推出的行动约束**：
- Video-R1 的确切硬件需求和确切 ZeRO 配置**都被否决**——把 README 的 "4×H20 或 5×A100" 当作唯一有来源的硬件陈述，**不要假设特定 ZeRO stage**。
- 三条对 VideoDPO 不利的断言全部被否——不要重复。
- **EasyR1 是否开箱支持视频仍是开放问题**，这正是 EasyVideoR1 存在要填的空白。

---

## 附录 B：开放问题

1. **Flash-GRPO 的 `train_wan2_1_flash_1node.sh` 到底装不装得进 48GB？** 作者从不说明其 8 卡节点的 GPU 型号或显存（配置只叫 `dgx`），论文省略了视频分辨率、帧数和 GRPO group size G——这恰是算 rollout 显存所需的乘数。**这是最该先跑的实验，而且回答成本极低：启动脚本，`nvidia-smi` 看一步。**

2. **PCIe 无 NVLink 在这些负载上的真实 wall-clock 代价是多少？** 调研的每个配方都避开 ZeRO-3（Flash-GRPO 用 ZeRO-2、VideoAlign 用 ZeRO-0、VideoDPO 用纯 DDP、EasyVideoR1 关 NVLS），这有暗示性但是**架构推断而非实测**。没有来源发布过 all-reduce 基准。**如果显存上确实需要 ZeRO-3，吞吐代价未知。**

3. **EasyR1 的 dataset pipeline 不 fork 能否真正处理视频输入？** 该断言在验证中被 1-2 否决，问题真正开放——它决定了推荐的可改底座是 EasyR1（成熟、5.1k stars）还是 EasyVideoR1（视频原生但 ~30 commits、3 forks）。**这是"建在久经考验的基座上"还是"建在年轻研究发布上"的区别。**

4. **方向 #4（世界模型、具身视频策略、Cosmos/GR00T 一类）完全未覆盖** —— 无断言通过验证。**若该方向重要，需要单独的调研轮次。**

5. **VideoReward 能否在 8×L40S 上与 Flash-GRPO 组合作为 reward 信号？** 两者单独都装得下（2B 打分器 + 1.3B 策略），flow_grpo 是有文档的桥梁，但**没有来源报告过同时常驻冻结 2B VLM 打分器 + 1.3B DiT 策略 + 其优化器状态 + rollout buffer 的组合显存足迹**。

---

## 附录 C：全部来源清单

23 个抓取来源，按检索角度分组。

### broad/primary — flow-matching & diffusion video RL landscape

| 来源 | 类型 | 断言数 |
|---|---|---|
| https://github.com/Shredded-Pork/Flash-GRPO | primary | 5 |
| https://rocm.blogs.amd.com/artificial-intelligence/wan-flow-grpo/README.html | blog | 5 |

### academic/technical — T2V preference alignment and reward models

| 来源 | 类型 | 断言数 |
|---|---|---|
| https://github.com/CIntellifusion/VideoDPO | primary | 5 |
| https://github.com/KlingAIResearch/VideoAlign | primary | 5 |
| https://arxiv.org/html/2506.10639v1 | primary | 5 |

### low-memory alternative — R1-style RLVR for video understanding VLMs

| 来源 | 类型 | 断言数 |
|---|---|---|
| https://github.com/tulerfeng/Video-R1 | primary | 5 |
| https://github.com/cyuQ1n/EasyVideoR1 | primary | 5 |
| https://github.com/hiyouga/EasyR1 | primary | 5 |
| https://github.com/www-Ye/Time-R1 | primary | 5 |
| https://arxiv.org/html/2504.06958v1 | primary | 5 |
| https://github.com/ZhangXJ199/TinyLLaVA-Video-R1 | primary | 5 |

### practitioner/implementation — framework support and hardware gotchas

| 来源 | 类型 | 断言数 |
|---|---|---|
| https://github.com/verl-project/verl-omni | primary | 5 |
| https://github.com/yifan123/flow_grpo | primary | 5 |
| https://github.com/Dao-AILab/flash-attention/issues/1978 | forum | 4 |
| https://github.com/hao-ai-lab/FastVideo | primary | 5 |
| https://github.com/XueZeyue/DanceGRPO | primary | 5 |

### contrarian/skeptical — failed reproductions, OOM reports, honest costs

| 来源 | 类型 | 断言数 |
|---|---|---|
| https://github.com/yifan123/flow_grpo/issues/204 | forum | 5 |
| https://github.com/mihirp1998/VADER | primary | 5 |
| https://github.com/tulerfeng/Video-R1/issues/124 | forum | 5 |

### world models / embodied video policies（无断言通过验证）

| 来源 | 类型 | 断言数 |
|---|---|---|
| https://github.com/nvidia-cosmos/cosmos-rl/blob/main/examples/ddrl.md | primary | 5 |
| https://github.com/RLinf/RLinf | primary | 5 |
| https://github.com/thuml/RLVR-World | primary | 5 |
| https://huggingface.co/blog/nvidia/cosmos-fine-tuning-for-robot-video-generation | primary | 5 |

### 调研统计

| 指标 | 值 |
|---|---|
| 检索角度 | 6 |
| 抓取来源 | 23 |
| 提取断言 | 114 |
| 进入验证 | 25 |
| 确认 | 18 |
| 否决 | 7 |
| 未决 | 0 |
| Agent 总数 | 106 |

---

## 参考链接

### 主推荐

- **Flash-GRPO**：https://github.com/Shredded-Pork/Flash-GRPO · [arXiv:2605.15980](https://arxiv.org/abs/2605.15980) · [ICML 2026 poster](https://icml.cc/virtual/2026/poster/63629)
- **Video-R1**：https://github.com/tulerfeng/Video-R1 · [SFT checkpoint](https://huggingface.co/Video-R1/Qwen2.5-VL-7B-COT-SFT) · [最终模型](https://huggingface.co/Video-R1/Video-R1-7B) · [数据集](https://huggingface.co/datasets/Video-R1/Video-R1-data)
- **EasyR1**：https://github.com/hiyouga/EasyR1
- **EasyVideoR1**：https://github.com/cyuQ1n/EasyVideoR1 · [arXiv:2604.16893](https://arxiv.org/abs/2604.16893)
- **VideoAlign / VideoReward**：https://github.com/KlingAIResearch/VideoAlign · [权重](https://huggingface.co/KwaiVGI/VideoReward) · [arXiv:2501.13918](https://arxiv.org/abs/2501.13918) · [VideoGen-RewardBench](https://huggingface.co/datasets/KwaiVGI/VideoGen-RewardBench)

### 上游 / 旁证

- **flow_grpo**（Flash-GRPO 的上游）：https://github.com/yifan123/flow_grpo
- **VideoX-Fun Wan2.1 训练指南**（8×L40S ZeRO-2 全参微调旁证）：https://github.com/aigc-apps/VideoX-Fun/blob/main/scripts/wan2.1/README_TRAIN.md
- **AMD ROCm Wan Flow-GRPO 博客**：https://rocm.blogs.amd.com/artificial-intelligence/wan-flow-grpo/README.html

### 谨慎 / 参考

- **VideoDPO**：https://github.com/CIntellifusion/VideoDPO · [复现失败 issue #12](https://github.com/CIntellifusion/VideoDPO/issues/12)
- **GigaVideo-1**（相邻但非 RL，是 reward 加权监督）：https://arxiv.org/html/2506.10639v1 · [OpenReview](https://openreview.net/forum?id=y7wAuwErpL)

### 其他抓取但未进入主结论的框架 / 项目

- verl-omni：https://github.com/verl-project/verl-omni
- FastVideo：https://github.com/hao-ai-lab/FastVideo
- DanceGRPO：https://github.com/XueZeyue/DanceGRPO
- VADER：https://github.com/mihirp1998/VADER
- Time-R1：https://github.com/www-Ye/Time-R1
- TinyLLaVA-Video-R1：https://github.com/ZhangXJ199/TinyLLaVA-Video-R1
- flash-attention issue #1978（Ada 兼容性）：https://github.com/Dao-AILab/flash-attention/issues/1978

### 方向 #4 未覆盖（供后续单独调研）

- cosmos-rl DDRL 示例：https://github.com/nvidia-cosmos/cosmos-rl/blob/main/examples/ddrl.md
- RLinf：https://github.com/RLinf/RLinf
- RLVR-World：https://github.com/thuml/RLVR-World
- NVIDIA Cosmos 机器人视频生成微调博客：https://huggingface.co/blog/nvidia/cosmos-fine-tuning-for-robot-video-generation

---

*本报告由 deep-research 工作流生成（106 agents，6 检索角度，23 来源，114 断言，25 条经三票对抗验证）。原始输出：`/tmp/claude-1000/-home-mingfeiguo-Desktop-VRL/eb4c76d0-4466-450a-a66d-cd8ba6be67b6/tasks/wfk1usa9v.output`*
