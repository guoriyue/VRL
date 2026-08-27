# SPRINT: Flash-GRPO on Wan2.1-T2V-1.3B — 8 卡单机复现

状态：**已执行完毕（2026-08-27）**。执行路径经 vrl 自身（2026-08-16 的路线
变更见 §0.-1），两轮 50-update 训练与配对评测的结论见 **§7.1**。本文档余下部分
保留为配方与算力分析的参考。

## 0.-1 路线变更（2026-08-16）

原计划跑外部 Flash-GRPO 仓库的唯一硬理由是 HPSv3 reward 不在 vrl 里，以及其
钉版 transformers 4.40 与 HPSv3 需要的 ≥4.45 的版本死结。vrl 环境是
transformers 4.57，死结不存在；HPSv3 已按 Kling VideoReward 的先例收窄移植为
vrl 原生 reward（`vrl/rewards/models/hpsv3.py`，MIT 归属注明），§8 的
"不修改 `vrl/`" 约束随之作废。

- **入口**：`vrl-train --config experiment/wan_2_1/online_grpo_hpsv3_4x_l40s`
  （拓扑：GPU0 trainer / GPU1-2 rollout / GPU3 reward；96 样本每 update、
  G=4、global_std、480×832×81f、20 步 CFG4.5，与论文配方对齐；
  prompt 集就是上游的 19,700 条，`datasets/flash_grpo_video/`）。
- **移植逐点对分（2026-08-16，g6e.12xlarge 实测）**：与上游
  `HPSv3RewardInferencer`（同权重、同 transformers 4.57、同 SDPA）在 8 张
  图上对比，max |Δ| = 0.27（量表 ~22 宽），8/8 排序一致。关键坑：上游
  forward 手工合并 embedding 后以 `input_ids=None` 调内层模型，**从不经过
  Qwen2-VL 的 M-RoPE 位置计算**（训练时也如此）；若走 4.57 的规范合并路径，
  分数偏移可达 0.9 且近分档排序翻转。移植忠实复刻了这一训练时行为。
- **单卡实测**：模型加载 + 3 图打分峰值 16.0GiB；81 帧 480×832 视频全帧打分
  8.4s（103ms/帧，8 帧/前向），一次 96 样本 update 约 13 分钟 —— 小于
  rollout（2 卡 ~30-40 分钟），单独占 GPU3 时串行即可盖住。
- 本文档其余部分（硬件分析 §1-2、显存实测 §3、上游修复 §4、HPSv3 的坑
  §6.1）对 vrl 路线仍然有效；§5 的 Gate 顺序照用，只是启动命令换成 vrl-train。

本文档原为外部仓库（`Shredded-Pork/Flash-GRPO`）的复现计划。
选型依据见 `docs/research/video-rl-post-training-on-8xL40S.md`。

## 0. 结论先行

在 **AWS `g6e.12xlarge` = 4 × L40S 48GB**（PCIe，无 NVLink）上复现
Flash-GRPO（ICML 2026）：Wan2.1-T2V-1.3B 基座 + LoRA r=16 + HPSv3 reward
+ DeepSpeed ZeRO-2。预计 **7–11 天 / $1,784–2,728**（§1.1）。

选它的唯一理由：**它是本次调研中唯一由作者亲自提供单机 8 卡脚本的视频扩散 RL 论文**
（`scripts/multi_node/train_wan2_1_flash_1node.sh`，README 标注 ~40h，附训练曲线），
而不是从多节点配方推断出来的。

整个计划的成败挂在**一个尚未有任何公开来源能回答的问题**上：

> 32,760 token 的激活 + G=4 的 rollout buffer，在 48GB 单卡上放不放得下？

论文未写分辨率/帧数/group size，作者也未说明其 8 卡节点的显存规格（配置文件仅命名为
`dgx`）。这个问题**用一小时就能自己回答**（§5 Gate 0），且必须先回答再投入其余算力。

## 1. 硬件前提与 L40 / L40S 的等价性

> **已定（2026-08-16）：目标硬件是 AWS `g6e.12xlarge` = 4 × L40S 48GB。**
> 执行时按 **§1.1 的 4 卡变体**（改两行配置）走，本节的 8 卡数字作为参照保留。

本节原按 8×L40 / 8×L40S 撰写。两种卡对本计划**几乎等价**：

| | L40 | L40S | 对本计划的影响 |
|---|---|---|---|
| 显存 | 48GB GDDR6 ECC | 48GB GDDR6 ECC | 相同 → 所有装载分析不变 |
| 带宽 | 864 GB/s | 864 GB/s | 相同 |
| SM / Tensor core | 142 / 568 | 142 / 568 | 相同 |
| NVLink | 无 | 无 | 相同 → ZeRO-2 的选择对两者都正确 |
| FP32 | 90.5 TFLOPS | 91.6 TFLOPS | 1.2% |
| TDP | 300W | 350W | L40S 长时满载能维持更久的 boost |

两份 datasheet 的 bf16 dense 标称是 181.05 vs 362.05（看似 2×），但这是**文档口径差异，
不是硬件差异**。NVIDIA 自己的 Ada 专业卡白皮书把 AD102 列为 364.2 TFLOPS dense；
本地验算复现了两个标称值的来源：

```text
142 SM × 512  FLOP/clk × 2.490 GHz = 181.0   ← L40  datasheet 的 181.05
142 SM × 1024 FLOP/clk × 2.520 GHz = 366.4   ← L40S datasheet 的 362.05
```

同一颗 AD102、同样 SM 数与 tensor core 数；FP32 仅差 1.2% 佐证着色器阵列一致。
（部分厂商博客称 "L40S 解锁了 Transformer Engine"——该说法不成立：TE 是软件库，
FP8 是全部 Ada 四代 tensor core 的硬件能力，50W 功耗差不可能在同硅片上变出 2× 算力。）

**结论：bf16 训练预期 1.0–1.1×（差距来自功耗预算而非算力），显存/带宽/互联完全一致。
配置无需为 L40 做任何修改，仅 wall-clock 估算 +5~10%。**

FP8 是 L40S 唯一可能真正领先之处，但本配方是 `mixed_precision="bf16"`，用不到。

### 1.1 变体：4 卡（AWS g6e.12xlarge）也可行，且更便宜

`g6e.12xlarge` = **4 × L40S 48GB**（AWS 官方 spec 表原文 `4 x NVIDIA L40S GPU`），
48 vCPU / 384 GiB RAM / 2×1900GB NVMe / 100 Gbps / **无 NVLink**（L40S 芯片本身就没有
NVLink 连接器）。

⚠️ **不要与 `g6` 系列混淆**：`g6` 用的是 **L4 24GB**，`g6.12xlarge` 是 4×L4 = 96GB，
$4.60/hr。同样的 size 命名、同样的 EPYC 7R13、同样的 vCPU 数，**只有 GPU 和显存不同**。
按 g6 报价规划 L40S 负载会让预算差 2 倍以上、单卡显存差一半。

**算法上 4 卡与 8 卡完全等价**，改两处即可保持每次 update 96 样本：

```python
# config/dgx.py:wan2_1_flash_1node
config.sample.num_batches_per_epoch = 24 → 48   # GA 会随之从 12 变 24
```
```bash
# scripts/multi_node/train_wan2_1_flash_1node.sh
--num_processes 8 → 4
```

验算：`8 卡 GA=12 → virtual=96 → 24 prompt × 4`；`4 卡 GA=24 → virtual=96 → 24 prompt × 4`。
每 epoch 同为 2 次 optimizer step。每卡采样量从 12 翻到 24 —— 这是时间翻倍的来源，
**不是算法损失**。

**时间与成本**（us-east-1 on-demand，AWS Pricing API 实价）：

| 方案 | GPU | $/hr | $/GPU-hr | 预计时长 | 预计成本 |
|---|---|---|---|---|---|
| **g6e.12xlarge** | 4 | 10.4926 | **2.62** | 170–260 h（7–11 天） | **$1,784–2,728** |
| g6e.48xlarge | 8 | 30.1312 | 3.77 | 85–130 h（3.5–5.4 天） | $2,561–3,917 |
| g6e.24xlarge | 4 | 15.0656 | 3.77 | 同 12xlarge | 更贵，仅多 CPU/RAM |

**4 卡比 8 卡便宜约 30%**（12xlarge 是全家族最便宜的每-GPU-小时选项），
代价是墙钟时间翻倍。⚠️ **`g6e.16xlarge` 只有 1 个 GPU**（$7.58/hr）——
GPU 数在这个家族里不随 size 单调递增，从 12xlarge "升级" 到 16xlarge 是从 4 卡降到 1 卡。

**🔴 显存要按实际可见容量规划，不是 48GB**：AWS 技术文档写 `178 GiB (4 x 44 GiB)`，
实测约 46,068 MB/卡。ECC 开销 + 十进制/二进制单位换算，**每卡实际少约 4GB**。
Gate 0 的判据应据此收紧：**单卡峰值 < ~40GB**（而非 44GB）。

**🔴 4 卡下 reward 共卡更危险**：

```text
训练 21.98 GB + HPSv3 16.5 GB = 38.5 GB   vs 实际可用 ~44 GiB
余量仅 5.5 GB —— 还要装 rollout buffer + 反传激活
```

8 卡时可以牺牲 1/8 算力把 reward 独占一卡；4 卡时那是 **25% 的算力**
（时间从 7–11 天变 9–14 天，成本 $2,371–3,630，反而超过 8 卡方案）。
**所以 4 卡下不要给 reward 独占一张卡** —— 但也**不需要开第二个实例**：
reward server 就跑在同一台机器上，只是那 16.5GB 该放哪张卡（或放 CPU）需要选。
四个方案与决策顺序见 **§6.4**（优先试 CPU 常驻，它零显存占用，而本机有 384 GiB
系统内存完全闲置）。

**PCIe 拓扑注意**：g6e.12xlarge 有 **2 个 NUMA node**，4 张卡很可能 2 卡/socket，
跨 socket 的卡对通信更慢。上机后先跑 `nvidia-smi topo -m` 看是 `PHB`/`PIX`（同 root complex，快）
还是 `SYS`（跨 socket，慢）。另：AWS 在 G7e 发布公告中把 GPUDirect P2P 作为**新**能力宣传，
并称 "up to four times the inter-GPU bandwidth compared to L40s GPUs featured in G6e" ——
**这是 G6e 不支持 P2P 的强证据**，GPU 间流量很可能经由主机内存中转。

好在本配方对此不敏感（§6.5）：每次 update 只 all-gather 几 KB，
ZeRO-2 只同步 LoRA 的 11.8M 参数（~24MB）。

## 2. 真正的瓶颈是 PCIe，不是算力

```text
L40/L40S PCIe Gen4 x16 : ~32 GB/s 单向
H100 NVLink            : 900 GB/s
```

相差约 28 倍。这解释了作者为何用 **ZeRO-2 而非 ZeRO-3**——ZeRO-3 每步 all-gather 参数，
在 PCIe 上会被通信拖死。手上这份配置已做对该选择。

**执行约束：若 Gate 0 OOM，优先降 G / 帧数 / 分辨率，不要先升 ZeRO-3。**
升 ZeRO-3 省显存但在无 NVLink 机器上的吞吐代价未知（调研中明确列为无实测数据的开放问题）。

同样的架构推断由多个独立配方佐证：Flash-GRPO 用 ZeRO-2、VideoAlign 用 ZeRO-0、
VideoDPO 用纯 DDP、EasyVideoR1 显式 `export NCCL_NVLS_ENABLE=0`、
VideoX-Fun 的 Wan2.1 指南也是先 ZeRO-2 后 FSDP。

## 3. 已验证的运行规模

配置在修复 `imp` 后可加载，以下数值均为 `config/dgx.py:wan2_1_flash_1node` 实读，
**不是论文数字**（论文未写分辨率/帧数/G）：

```text
model                Wan2.1-T2V-1.3B-Diffusers
use_lora             True                      ← 是 LoRA，不是全参微调
resolution           480 × 832,  81 frames
GRPO group size G    4
train / eval steps   20 / 50      CFG 4.5
sample batch/gpu     1
batches per epoch    24           grad accum 12
timestep_fraction    0.51
mixed_precision      bf16
reward_fn            {"videohpsv3": 1.0}
```

由缓存中的真实 `transformer/config.json` 推出的显存驱动量：

```text
latent 网格          21 × 60 × 104     （VAE 时间 4× / 空间 8× 下采样）
patch 后序列长度      32,760 tokens     ← 显存的真正驱动因素
hidden dim           1536  (12 heads × 128)
transformer          ~1.39B params, 30 layers

冻结基座 bf16         2.78 GB
LoRA r=16            11.8M 可训参数 → AdamW 状态 ~0.12 GB
gradient checkpoint  代码中默认已开（train_wan2_1_flash_1node.py:520）
```

**参数侧合计仅约 3GB。** 32,760 token 的激活与 G=4 的 rollout 轨迹缓冲才是大头——
这正是无法纸面推算、必须实测的部分。

### 3.0 8 卡怎么分工：`virtual_num_replicas`

并行方式是**纯数据并行**（8 张卡各持一份完整的 1.39B 模型，不切模型），
但 GRPO 要求同一 prompt 的 G 个样本必须在同一次 update 内才能算组内 advantage，
而每卡每次只放得下 1 个视频。作者的解法是把梯度累积也算进"并行度"：

```python
# scripts/train_wan2_1_flash_1node.py:578-586
train_sampler = DistributedKRepeatSampler(
    batch_size=1, k=4, num_replicas=8,
    virtual_num_replicas=8 * 12,   # N × gradient_accum —— 注释：“一次性采样整个update的prompt”
)
```

```text
virtual_num_replicas = 8 × 12 = 96      个虚拟 rank
total_samples        = 96 × 1  = 96     一次 update 的总样本
96 % k(=4) == 0                          ✓（sampler 内有 assert）
distinct prompts     = 96 / 4  = 24     每个重复 4 次
accum_chunks / rank  = 96 / 8  = 12     = gradient_accum ✓
num_batches_per_epoch = 24              → 每 epoch 2 次 optimizer step
```

分派规则 `virtual_rank = chunk * num_replicas + rank`（`:138-146`），
所以 rank0 拿虚拟 rank 0,8,16,…,88。一个 prompt 的 4 个样本落在哪张卡无所谓——
advantage 是跨卡算的：

```python
# :879, :896
gathered_rewards = {k: accelerator.gather(v) for k, v in samples["rewards"].items()}
prompt_ids       = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
```

配合 `config.sample.global_std = True`，用全局 96 个样本的标准差而非每卡本地的。

**为什么适合无 NVLink 的 8×L40**：每次 update 只 all-gather 96 个标量 reward +
96 条短 token 序列（**几 KB 量级**）；ZeRO-2 同步的梯度也只有 LoRA 的 11.8M 参数
（~24MB bf16，因为基座冻结）。最贵的 rollout（每样本 20 步去噪）在 8 卡上
**完全独立、零通信**。真正会打爆 PCIe 的是 ZeRO-3 每步 2.78GB 的参数 all-gather——
这正是 §2 说不要先升 ZeRO-3 的原因。

### 3.1 训练数据与 reward 的实际定义

**数据集**：仓库自带 `dataset/video/train.txt`（19,700 行）/ `test.txt`（300 行），
每行一条纯文本 prompt，**无配对视频、无标注**（`TextPromptDataset` 只做 `line.strip()`，
metadata 为空 dict）。这是 RL 而非 SFT 的直接体现：不需要目标视频，模型自采样、
reward model 打分、GRPO 用组内相对分数更新；数据集只提供探索方向的 prompt 分布。
内容是真实用户 T2V 查询风格（含拼写错误、西班牙语、2D 游戏素材需求）。

**reward 是 HTTP 服务，不是 in-process**。训练进程把视频逐帧编码成 JPEG、pickle 打包
POST 给 gunicorn（`flow_grpo/rewards.py:93` `video_hpsv3_remote`）；服务端才持有模型
（`flow_grpo/reward-server/reward_server/hpsv3.py`）。这就是 `gunicorn.conf.py` 写死
`NUM_DEVICES = 8` 的由来——它与训练争抢同样 8 张卡（§6 的待定取舍）。

**打分粒度：逐帧打分，取最高 30% 的均值**：

```python
# flow_grpo/rewards.py:145-147
response_data["outputs"].sort(reverse=True)
all_scores += [sum(response_data["outputs"][:int(l*0.3)])/int(l*0.3)]
```

HPSv3 是**图像**模型，不是视频模型。81 帧被当成 81 张独立图片各打一分，降序取前 30%
求平均。含义有三：优化的是"最好的那些帧有多好"而非平均质量；取 top-30% 抗运动模糊/
过渡帧的噪声；**完全不看时序**——帧间连贯性与运动合理性不在 reward 内。

因此本配方优化的是**逐帧美学 + 图文对齐**，与论文在 VBench 上主要报告 Aesthetic Quality
和 Subject Consistency 提升是自洽的。若要优化运动质量，需换 reward（如 VideoReward /
VideoAlign，见 §9 调研），那是另一个 sprint。

**已知的 reward hacking 面**（上游 issue，训练时按此盯指标）：

- #32 某些配对下**忽略 prompt** —— 它对 prompt 的条件化偏弱
- #31 **偏好正面情绪**，与 prompt 无关
- #24 名画被打很低
- #19 杂乱图像可以拿到 >15

策略会去找这些缝隙。这也是为什么 §5 的 Gate 1 不能只看 reward 曲线上升就宣布成功——
**必须同时看生成样本**。reward 涨而画面变差是本配方最可能的失败模式：
时序完全不在 reward 内（见上），所以"每帧都很美但动起来是糊的/闪烁的"不会被惩罚。

### 3.2 已实测：基座推理在单卡上正确且只要 22GB（2026-08-15）

在本地开发机（1×RTX 5090 32GB，sm_120，torch 2.11 / diffusers 0.39）跑了基座推理冒烟，
用**训练配置的真实第一条 prompt** 与**训练时的采样参数**（20 步、CFG 4.5）：

| | 33 帧 | **81 帧（训练几何）** |
|---|---|---|
| 峰值显存 | 21.73 GB | **21.98 GB** |
| 耗时 | 21.8 s | 73.0 s |
| 每步 | 1.09 s | 3.65 s |
| 空间 std（0-255） | 77.1 | 53.9 |
| 帧间平均绝对差 | 11.62 | 13.53 |

画面正确：白色背景、蓝色士兵剪影排成一列自右向左行进，逐帧位置变化。
prompt 的四个要素（`infinite white background` / `walking in a row` / `right to left` /
`one behind the other`）全部命中。**基座权重、VAE、调度器、文本编码器均正常，
缓存的 27GB 权重完整可用。**

产物（`outputs/` 已被 .gitignore，故留在本地不入库）：

```text
outputs/wan_base_smoke_20260815/
├── wan_base_rollout_smoke.py     # 可重跑的脚本（一次性验证件，非长期资产）
├── wan_offload_probe.py          # §3.3 的 CPU-offload 对比探针
├── wan_smoke/                    # 33 帧
│   ├── sample.mp4
│   ├── contact_sheet.png         # 第 0/8/16/24/32 帧拼图
│   └── stats.json
└── wan_smoke_81f/                # 81 帧，训练几何
    ├── sample.mp4
    └── stats.json
```

重跑：`python outputs/wan_base_smoke_20260815/wan_base_rollout_smoke.py --frames 81 --out <dir>`

**对 Gate 0 的意义**：22GB 是推理下界，48GB 卡尚余 ~26GB 给训练侧的
反传激活（已开 checkpointing）、LoRA 优化器（~0.12GB）和 rollout buffer
（G=4 条轨迹 × 20 步的 latent + logprob）。**Gate 0 实测的就是最后一项**，
风险比纸面推算时估计的低。

**限定**：本机是 sm_120 + torch 2.11，8 卡机是 sm_89 + torch 2.6.0（setup.py 钉版），
attention 后端可能不同（FlashAttention vs SDPA），显存会有出入。
**此测试证明的是"模型与权重没问题"，不是"L40 上就是 22GB"。**

### 3.3 已实测：CPU offload 把推理显存从 22GB 降到 10.8GB，只慢 8%

`enable_model_cpu_offload()` 让每个组件只在自己运行时上 GPU。同一 prompt、
同一训练几何（81 帧 480×832、20 步、CFG 4.5）：

| | 峰值显存 | 耗时 |
|---|---|---|
| `pipe.to("cuda")`（§3.2 基线） | 21.98 GB | 73.0 s |
| **`enable_model_cpu_offload()`** | **10.81 GB** | 78.7 s |

**省 11.2 GB，只慢 8%。** 原因看 §3 的权重分解：T5 文本编码器（bf16 约 9.4GB）
和 VAE 在整个去噪循环里根本用不上——编码完就闲置，却一直占着显存。

**对 reward 取舍的影响**（§6.4）：原本的困境是

```text
训练 21.98 + HPSv3 16.5 = 38.5 GB  vs 可用 ~44 GiB  ->  余量仅 5.5 GB
```

若 offload 在训练侧同样成立，则变成 `~10.8+ + 16.5 = ~27+ GB`，
**方案 B（reward 与训练共卡）从"很紧"变为"大概率可行"**，
不必再走 CPU 或按需搬运。

**🔴 但这是推理路径上测的，不能照抄。** Flash-GRPO 的训练脚本自行管理 pipeline 组件
（LoRA 挂在 transformer 上、要反传、走 accelerate + DeepSpeed），直接加这一行
**可能与其集成冲突**。这是有希望的方向，不是已验证的结论——Gate 0 要对比测量。

重跑：`python outputs/wan_base_smoke_20260815/wan_offload_probe.py`

### 3.4 已实测：CPU 跑 reward 会成为新瓶颈（方案 C 的判据）

`Qwen2-VL-7B` 每图前向约 **3.66 TFLOP**（28 层、hidden 3584、视觉 token 被硬限制在
`256*28*28` → 约 256 token/图）。本机 16 线程实测 CPU GEMM 约 **1.93 TFLOPS**，
按 AWS 48 vCPU 保守估两倍、取 35% 达成率：

```text
约 2.7 s/图  ->  81 帧/样本 = 219 s  ->  12 样本/卡 = 44 分钟
对比 rollout：每卡 12 样本 × 20 步 = 18–24 分钟
```

**CPU reward 约为 rollout 的 2 倍，异步掩盖不住，会成为新瓶颈。**
所以 §6.4 的方案 C 只在方案 B 被 Gate 0 否决时才考虑，
而有了 §3.3 的 offload，方案 B 大概率不会被否决。

## 4. 上游代码的五个阻塞（已修复，本地提交 `8f7552e`）

`Shredded-Pork/Flash-GRPO` 的 1-node 路径开箱即坏，五处各自独立阻断启动：

1. **`config/dgx.py` 使用 Python 3.12 已删除的 `imp`**（3.4 起弃用），而 `setup.py`
   声明 `python_requires>=3.10` → 任何 3.12 解释器上配置无法加载。改用 `importlib.util`。
   修好后配置才第一次成功加载，§3 的数值即由此得来。
2. **启动脚本指向不存在的文件**：脚本写 `train_wanx2_1_flash_1node.py`，实际文件名为
   `train_wan2_1_flash_1node.py`（无 `x`）。
3. **`PROJECT_ROOT=""`** → `cd ""` 落到 `$HOME`，所有相对路径失效。改为从脚本自身位置推导。
4. **reward server 硬编码作者集群 IP** `172.20.66.144:8080` → 改读 `REWARD_SERVER_URL`。
   （`unifiedreward_score_remote` 中另一个硬编码 IP 未动：该 reward 路径本配置不选用。）
5. **端口不一致**：`app_hpsv3.py` 声明 8080，`gunicorn.conf.py` 绑定 8081 → 统一为 8081。

验证程度：配置可加载、脚本 `bash -n` 通过、改动的 Python 文件 `py_compile` 通过。
**尚未实跑训练**（本地开发机仅单卡 RTX 5090 32GB，且 HPSv3 未安装）。

上游仓库无 push 权限（403），改动仅在本地。若要回馈需 fork 后提 PR——这五个问题
对任何复现者都是硬阻塞，尤其 `imp` 在 3.12 上直接使配置无法加载。

## 5. 执行计划与 Gate

顺序：**Gate 0.5 → Gate 0 → Gate 1 → Gate 2**。0.5 编号在前是因为它先于 0 执行
（reward server 必须先起来），但 Gate 0 才是决定计划成败的那一个。

### Gate 0（决定性，~1 小时）：单卡峰值显存

**这是整个计划里性价比最高的一小时**，它一次性回答 §0 的核心未知。

前置：两个独立环境 + HPSv3 checkpoint + reward server 起来且通过 Gate 0.5（见 §6）。

```bash
bash scripts/multi_node/train_wan2_1_flash_1node.sh   # 跑 1–2 步即可
# 另一个终端
nvidia-smi --query-gpu=index,memory.used --format=csv -l 5
```

判据：

**测两个数，不是一个**：

1. 训练峰值（原样，不加 offload）—— 基准，决定 reward 放哪
2. 训练峰值（加 `enable_model_cpu_offload()`）—— 看 §3.3 的 11GB 节省在训练侧是否成立

判据：

- **单卡峰值 < ~40GB** → 通过，进入 Gate 1。阈值按**实际可见显存**定：
  48GB 卡 `nvidia-smi` 只报 ~44–46 GiB（ECC + 单位换算，见 §1.1），再留 4GB 给碎片。
  云上按小时计费，**这一小时必须先跑**——在按天计费的机器上撞 OOM 是最贵的错误。
- **OOM** → 按下列顺序降，每降一档重测：
  1. `config.frames` 81 → 49 或 33（序列长度近似线性下降；33 帧实测 21.73GB vs 81 帧 21.98GB）
  2. `config.height/width` 480×832 → 320×576
  3. reward 从 GPU 挪到 CPU 或改按需加载（§6.4 方案 C / D，零显存占用）
  4. `config.sample.num_image_per_prompt` 4 → 2
  5. 最后才考虑 ZeRO-3 / param offload（见 §2 的警告）

**顺序理由**：降帧数/分辨率只减少单样本的计算量，**不改变 GRPO 的统计结构**；
降 G 会直接削弱 advantage 的信噪比——组内只剩 2 个样本时，组内均值/标准差的估计方差
显著变大，影响 RL 收敛质量。因此把 G 排在帧数与分辨率之后。
（注意降 G 不减少 `total_samples`：sampler 的 `virtual_num_replicas = N × GA = 96` 固定，
G 减半只是把 24 个 prompt × 4 变成 48 个 prompt × 2。）

降档会偏离论文配方，**必须在 §7 记录实际使用值**，否则复现结论不可比。

### Gate 0.5（~10 分钟）：reward server 独立连通性

在开训练之前单独验证 reward 通路，比在训练里 debug 省事得多。
拿 §3.2 已生成的 `sample.mp4` 抽几帧，直接 POST 到跑起来的 server：

判据：返回**有限浮点数**（无界，典型 -1~+12，越高越好，见 §6.3），不是报错/NaN。
同一段视频重复打分应当稳定。

### Gate 1：短程收敛信号

跑到 `eval_freq=20` 的第一个评测点。判据是三条**同时**成立：

1. reward 曲线上升、无 NaN、无发散（作者 README 附了
   `asset/train.jpg` / `asset/eval.jpg` 可作对照）
2. **人眼看生成样本没有变差** —— 不能只看曲线
3. 帧间连贯性没有退化

第 2、3 条不是形式主义：reward 完全不含时序项，且有已知的 prompt-条件化偏弱与
情绪偏置（§3.1），所以"reward 涨但视频变糊/闪烁/千篇一律"是本配方**最可能的失败模式**，
而它在曲线上看不出来。

### Gate 2：完整复现

跑到与作者可比的步数。`num_epochs=300` 是刻意设大的，论文写明"训练手动停止"。

wall-clock 估算：作者 8 卡 ~40h，GPU 型号未公开（配置名 `dgx` 暗示 80GB A100/H100 级）。
按算力线性外推 L40S 约为其 2–3 倍，L40 再 +5~10% → **约 85–130 小时，即 4–6 天**。
该估算的不确定性主要来自作者硬件未知，而非 L40/L40S 之别。

## 6. 依赖就绪状态

| 项 | 状态 | 备注 |
|---|---|---|
| Flash-GRPO 代码 | ✅ | clone 干净，五处阻塞已修（`8f7552e`） |
| 配置可加载 | ✅ | 修 `imp` 后验证 |
| 基座模型 | ✅ | `Wan2.1-T2V-1.3B-Diffusers` 已在 HF 缓存（27GB） |
| 训练 prompt | ✅ | 仓库自带 `dataset/video/train.txt` 19,700 条 / `test.txt` 300 条 |
| 8 卡启动脚本 | ✅ | 作者提供，ZeRO-2 |
| **HPSv3 包 + checkpoint** | ❌ | 见 §6.1 —— 比想象中麻烦，且**必须独立环境** |
| **训练环境（独立 conda）** | ❌ | 必须新建，见下 |

训练环境必须独立：`setup.py` 钉 `torch==2.6.0 / transformers==4.40.0 / diffusers==0.33.1 /
deepspeed==0.16.4 / peft==0.10.0`。L40/L40S 是 sm_89，该套钉版可正常安装
（对比：本地开发机的 RTX 5090 是 sm_120，torch 2.6.0+cu124 不支持，这也是不能在本机跑的原因之一）。

### 6.1 HPSv3：三个必须提前知道的坑

**坑 1（硬阻塞）：HPSv3 与训练环境的 transformers 版本不可调和。**

HPSv3 的 backbone 是 **Qwen2-VL-7B-Instruct**（不是 CLIP 级小模型），而 Qwen2-VL 支持
在 transformers 4.45 才落地。训练环境钉的是 **4.40.0**，其中根本不存在
`Qwen2VLForConditionalGeneration` —— import 即失败。这不是"尽量隔离"的建议，
**是必须分两个环境**。

好在 reward server 本来就是独立进程（HTTP），隔离零成本。另有两条独立理由支持隔离：
上游 issue #6 / #2 报告在同进程内初始化 HPSv3 会污染已初始化的 `Accelerator`
（`accelerator.config` 变成 `None`，根因是其 `TrainingConfig` 继承
`transformers.TrainingArguments` 并在构造时触碰 accelerate 全局状态）——对 RL trainer 是地雷。

**坑 2：必须用 `upgrade_transformers_version` 分支，不能用 PyPI 或 main。**

PyPI 只发过一次 `1.0.0`（2025-08-06，再未更新）。在 transformers > 4.45 上，
HF 重命名了 Qwen2-VL 的子模块树（`model.embed_tokens` → `model.language_model.embed_tokens`，
`visual.*` → `model.visual.*`），导致 16.6GB checkpoint 的 `load_state_dict(strict=True)` 失败：

```text
RuntimeError: Error(s) in loading state_dict for Qwen2VLRewardModelBT
```

这是上游 issue #15 / #29，**#29 的报告者用的正是 flow_grpo 的调用方式**
（`HPSv3RewardInferencer(device='cuda', checkpoint_path=".../HPSv3.safetensors")`），
main 分支至今未修。`upgrade_transformers_version` 分支加了 key 重映射 shim。

**坑 3：磁盘要 ~33GB，不是 16.6GB。**

`HPSv3RewardInferencer.__init__` 会先 `from_pretrained("Qwen/Qwen2-VL-7B-Instruct")`
建骨架再用 checkpoint 覆盖，所以**除了 16.6GB 的 HPSv3.safetensors 还会拉整个
~16GB 的 Qwen2-VL 基座**（上游 issue #3，未修）。可预下载后改
`hpsv3/config/HPSv3_7B.yaml` 的 `model_name_or_path` 指向本地目录。

另：`checkpoint_path` 被直接传给 `safetensors.torch.load_file()`，**不做任何路径解析**。
`flow_grpo/reward-server/reward_server/hpsv3.py` 里的相对路径 `'HPSv3/HPSv3.safetensors'`
只有在 gunicorn 的 CWD 恰好包含该目录时才成立 —— **改成绝对路径**。

### 6.2 安装与启动

```bash
# reward server 的独立环境
git clone -b upgrade_transformers_version https://github.com/MizzenAI/HPSv3.git
cd HPSv3 && python -m venv .venv-hpsv3 && source .venv-hpsv3/bin/activate
pip install torch==2.6.0 torchvision==0.21.0
pip install -e .            # 该分支会拉 transformers>=4.45.2
# flash-attn 可跳过：输入被硬限制在 256*28*28 像素，SDPA 回退代价很小，
# 而 flash-attn==2.7.4.post1 是常见的编译失败源

huggingface-cli download --repo-type model MizzenAI/HPSv3 --local-dir /abs/path/HPSv3
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct --local-dir /abs/path/Qwen2-VL-7B
# 然后把 hpsv3/config/HPSv3_7B.yaml 的 model_name_or_path 指向本地 Qwen 目录
```

启动（**必须带 `-c`**，README 的命令未带，不会读到 gunicorn 配置，端口与 GPU 分配都不对）：

```bash
cd flow_grpo/reward-server
gunicorn -c gunicorn.conf.py "app_hpsv3:create_app()"
```

### 6.3 分数语义：无界，越高越好

作者在 issue #9 明确："HPSv3 doesn't have a clear upper limit... Bradley-Terry loss
only maximizes the margin, there's no upper or lower bound."

README 的 benchmark 表里各模型均值大致落在 **-1 ~ +12**（Kolors 10.55、Flux-dev 10.43、
SDXL 8.20、SD3-Medium 5.31、SD2 **-0.24**），单图有报告 >15。

**对 GRPO 无妨**（组内相对归一化本就不依赖绝对尺度），但**不要在任何 reward 管线里
假设 0–1 或 0–10 的钳位**。

`reward()` 返回形状 `[batch, 2]` 的张量，两列是 (mu, sigma)（`output_dim: 2`,
`loss_type: "uncertainty"`）。取 mu 用 `out[:, 0]`。

### 6.4 为什么是 HTTP，以及为什么从 1 个 worker 起步

**HTTP 不是性能选择，是上游代码唯一提供的路径 + 版本冲突的解法。**
`flow_grpo/rewards.py:415` 的注册表里 `ocr` / `aesthetic` / `pickscore` / `imagereward`
都是 in-process，但 `videohpsv3` 只有 `video_hpsv3_remote` 一个实现。
即使自己补一个 in-process 版本也走不通：HPSv3 需要 transformers ≥4.45（Qwen2-VL 支持
在 4.45 落地），训练环境钉 4.40.0，同进程即同环境，版本只能二选一（§6.1 坑 1）。

**HTTP 只决定进程间通信方式，不决定进程住哪张卡。** `gunicorn.conf.py` 的
`post_fork` 给每个 worker 设 `CUDA_VISIBLE_DEVICES`，所以把 `NUM_DEVICES` 改小即可让
reward 只占少数几张卡，其余卡纯训练——**这才是"复用同样 8 张卡"的正确做法**，
而不是取消 HTTP。

**性能上 HTTP 几乎无代价，因为 reward 已经是异步的**：

```python
# scripts/train_wan2_1_flash_1node.py:797
rewards = executor.submit(reward_fn, [videos, path], prompts, ...)   # 立即返回
time.sleep(0)          # yield，让 reward 线程真正启动
...                    # 继续 rollout 下一个样本
# :829 —— 隔了整整一轮采样之后才取结果
rewards, reward_metadata = sample["rewards"].result()
```

`ThreadPoolExecutor(max_workers=8)`（`:658`）让 reward 计算与后续 rollout **重叠**。
量级对比（基于 §3.2 实测的 73s/81帧）：

| 每卡每次 update | 时间 |
|---|---|
| rollout（12 样本 × 20 步去噪） | ~18–24 分钟 |
| reward（12 样本 × 81 帧 HPSv3 前向） | ~1–1.6 分钟 |

**reward 仅占 rollout 的 5–8%，且被异步掩盖。** localhost 上 JPEG+pickle 的搬运是
毫秒级，相对 7B 模型 81 次前向可忽略。

**显存才是 HTTP 的真实代价**：每个 worker 一份完整 7B 权重，**约 16.5GB**。
所以 **worker 数从 1 起步**（早先写的 "先试 8 worker" 已被算术推翻，"2 worker" 也偏保守）：
reward 只占 5–8% 的时间且异步执行，单 worker 串行排队不会成为瓶颈，而每多一个 worker
就多吃一张卡 16.5GB。

#### ⚠️ 两个必须澄清的误解

**误解 1：HTTP ≠ 需要第二台机器。** 本文档早期版本建议"单独开一个 g6e.xlarge 跑 reward"，
**那是过度设计，已撤销**。server 就跑在训练同一台机器上：

```python
# gunicorn.conf.py:27
bind = f"0.0.0.0:{port}"                        # 监听所有网卡，localhost 自然可达
# rewards.py 默认值
REWARD_SERVER_URL = "http://127.0.0.1:8081"     # 同机 localhost，走内存回环不过网卡
```

"挪到另一台机器"只是 HTTP 架构**顺带提供的退路**，不是推荐做法。真正的问题从来只有一个：
**那 16.5GB 放在哪张卡（或不放在卡上）。**

**误解 2：改成 in-process 并不能省显存。**

```text
独立进程 :  训练 21.98 GB + HPSv3 进程 16.5 GB = 38.5 GB
in-process:  训练 21.98 GB + HPSv3 张量 16.5 GB = 38.5 GB
```

权重就是权重，放在哪个进程里都占同样的显存。in-process 唯一省下的是 CUDA context
（每进程 ~300–500MB），杯水车薪。而代价有三条：**transformers 版本冲突使其根本无法
import**（§6.1 坑 1，这条是硬的）、污染 `Accelerator`（上游 #6/#2）、
以及**失去 `executor.submit` 的异步重叠**（同步调用会把 5–8% 变成串行阻塞）。

**结论：in-process 是"付出版本冲突的代价，换来 0.5GB 和更慢"。不做。**

#### 那 16.5GB 到底放哪：四个方案

全部在**同一台机器**上，不需要第二个实例。

| 方案 | 训练可用卡 | 显存代价 | 备注 |
|---|---|---|---|
| **B. 与训练共卡** | 4 | 那张卡 38.5GB / ~44GiB | 最简单；余量 5.5GB，需 Gate 0 验证 |
| **C. reward 常驻 CPU** | 4 | **0** | 唯一零显存方案；需实测 CPU 是否够快 |
| **D. 按需加载/卸载** | 4 | 峰值不重叠 | 每 update 搬 16.5GB 过 PCIe（~1–2s，可忽略）；要改 server 代码 |
| A. reward 独占一卡 | 3 | 0（对训练卡） | **最后手段**：4 卡时这是 25% 算力，成本反超 8 卡方案 |

**方案 C 已被实测降级**（§3.4）：CPU 跑 7B reward 约需 44 分钟/卡/update，
是 rollout（18–24 分钟）的两倍，**异步掩盖不住，会变成新瓶颈**。
它只在 B 和 D 都被否决时才考虑。

**方案 B 因 §3.3 的 offload 实测而升为首选**：推理侧 offload 省下 11.2GB
（22 → 10.8GB）只慢 8%。若训练侧同样成立，共卡变成 `~27GB vs ~44GiB`，余量宽松。

#### 决策顺序

```text
1. Gate 0 测两个训练峰值：原样 X1，加 offload X2（§3.3 的节省是否在训练侧成立）
2. min(X1, X2) + 16.5 < 40GB  -> 方案 B，共卡，改动最小【首选】
3. 否则 -> 方案 D（按需加载/卸载），峰值不重叠
4. 再否则 -> 方案 C（CPU），接受它成为瓶颈（§3.4 实测约 2 倍 rollout 时长）
5. 都不行 -> 方案 A（独占一卡），4 卡下损失 25% 算力
```

**第 1 步是前提**：21.98 / 10.81GB 都是**推理**数字，训练侧的反传激活与 rollout buffer
仍是未知数。在知道 X 之前，讨论 reward 放哪都是空谈。

### 6.5 能不能让 rollout 和 training 重叠？

**在本 sprint 的范围内：不能，而且不应该试。** 但这是个真问题，值得写清为什么。

主循环是严格的两段式（`:702` `#### SAMPLING ####` → `:984` `#### TRAINING ####`）：
先 `transformer.eval()` 采完整个 update 的 96 个样本，再 `transformer.train()` 做 12 次
梯度累积。两段之间没有重叠。

**阻挡重叠的是 on-policy 约束，不是工程惰性**：

```python
# :1062
ratio = torch.exp(log_prob - sample["log_probs"][:, 0, 0])
```

`sample["log_probs"]` 是**采样时那份权重**算出的 log-prob，`log_prob` 是**当前正在训练的
权重**算出的。这个比值就是重要性采样权重，PPO/GRPO 的裁剪正是作用在它上面。

如果一边训练一边采样，采样用的权重在同一个 update 内不断漂移，`ratio` 的分母就不再对应
一个确定的行为策略——**这不是精度问题，是算法定义被破坏**。同理，GRPO 的组内 advantage
要求同 prompt 的 G 个样本来自同一策略，否则组内比较失去意义。

还有两条现实约束：

1. **显存**：rollout（推理，实测 22GB）与 training（反传激活 + 优化器）如果同时驻留，
   峰值是两者之和。48GB 卡在 §6.4 的算术下已经很紧，重叠会直接 OOM。
2. **同一份权重**：两阶段共用同一个 `pipeline.transformer`，靠 `eval()`/`train()` 切换。
   要真重叠得持有两份权重副本（+2.78GB）并做显式同步——这是 slime / veRL 那类
   异步 RL 框架的做法，不是给这个脚本打补丁能得到的。

**已经存在的重叠**（免费拿到的那部分）：reward 计算与 rollout 通过
`executor.submit` 重叠（§6.4），而这恰好是最容易重叠、收益最明确的一段——
因为 reward 用的是**已经生成完的视频**，不依赖任何还在变的权重。

**真想做异步 RL 的正确姿势**是换框架（rollout engine 与 trainer 分离、
带明确的权重同步与 staleness 策略），而不是改这个脚本。本仓 `vrl/` 里有相关工作
（`StalenessPolicy`、continuous three-stage pipeline），但那是独立方向，
**前置条件仍是本 sprint 的 Gate 1 先跑通**——先证明能复现，再谈提速。

## 7. 待记录（执行时回填）

- [x] 基座推理正确性 + 单卡推理显存 → §3.2（22GB @ 81 帧训练几何，画面正确）
- [x] CPU offload 的推理侧收益 → §3.3（10.81GB，−11.2GB，仅慢 8%）
- [x] CPU 跑 reward 的可行性 → §3.4（约 2 倍 rollout 时长，会成瓶颈）
- [ ] **Gate 0 的两个数**：训练峰值 X1（原样）与 X2（加 offload）
- [x] 实际 GPU 型号、可见显存 → 4×L40S，46,068 MiB/卡（~45GiB；2026-08-16 实测）
- [x] PCIe 拓扑 → 4 卡全 `NODE`（单 NUMA node，无跨 socket；2026-08-16 实测）
- [x] 4 卡配比 → vrl 路线：`prompts_per_batch=24` × `n_samples_per_prompt=4` = 96/update（§0.-1）
- [x] HPSv3 单图/视频吞吐与显存 → §0.-1（16.0GiB 峰值；103ms/帧；96 样本 ~13 分钟）
- [x] Gate 0 单卡峰值显存（**训练**）→ 19.4GB（全量 gradient checkpointing +
  逐样本流式 replay），生成相位 n=4 峰值 26.4GB，均在 44GB 预算内
- [x] 是否降档、降了哪些、最终实际配置值 → 见 §7.1「最终配置」
- [x] 单 update wall-clock → 坏配方 134 分钟 → 修正后 **44 分钟**（编译 rollout
  −27% 生成，单步 SDE 把反传从 96 分钟压到几分钟）
- [x] reward server worker 数的最终取舍 → 不用 HTTP server：相位轮转下 reward
  与 rollout/trainer 分时共用同一张卡，rank-local 进程内加载
- [x] Gate 1 首个评测点 → §7.1：peak checkpoint **+0.82** vs base（95% CI
  [+0.35, +1.37]，48 配对样本胜率 69%）

## 7.1 执行结论（2026-08-20 → 08-27，4×L40S）

两轮各 50 update 的训练 + 三次固定 prompt 配对评测。评测协议：24 条留出 prompt
× 2 样本 × 各臂共用种子、确定性采样、HPSv3 打分、48 个配对单元、bootstrap 95% CI。

### 结果

| 权重 | Δ `top_frame_mean` vs base | 95% CI | 胜率 |
|---|---|---|---|
| 坏配方 ckpt-42 | **−1.25** | [−1.85, −0.63] | 27% |
| 修正 ckpt-18（峰值） | **+0.82** | [+0.35, +1.37] | 69% |
| 修正 ckpt-50（终点） | +0.67 | [+0.01, +1.35] | 60% |

`frame_mean` 与 `frame_min` 同向变化（peak 分别 +0.84 / +0.53），排除
"最好的帧变好、最差的帧烂掉"的刷分模式 —— 与 §6.1 记录的 reward hacking 面对应。

### 第一轮为什么退化 —— 四个缺陷，全部只有本配方会踩

1. **rollout 是确定性的，却按随机分布算 log-prob。** `rollout.denoise_mode`
   继承 wan 家族默认的 `native`：它把调度器的确定性一步存为 action，再用 SDE
   提议密度给它打分。被求导的量是模型输出的固定二次型，不是任何抽样的对数密度，
   `E[∇log π]=0` 不成立，负优势分支没有平衡点 —— 这是"在自己的目标上单调下坡"
   的根因。已验证的旁证：`online_flash_grpo_kling_video_reward.yaml` 显式覆盖回
   `sde` 并在注释里说明 native 会绕过 window 机制。
2. **KL 项实际权重被放大约 7e4 倍。** `sde.type: cps` 的 log-prob 缺少
   `1/(2σ²)` 归一化因子而 `compute_kl_divergence` 仍然除以它；且 cps 的
   `std_dev_t` 取自 `sigma_prev`，在倒数第二步塌缩到 2.7e-3。`strided` 选步
   恰好每个 update 都训练那一步。`compute_kl_divergence` 当时零测试覆盖。
3. **信赖域量的是数值噪声。** `clip_ratio` 解析为 1e-4，低于本栈的
   rollout↔replay 漂移：`pre_update_clip_fraction=0.78`（**任何权重更新之前**），
   约 40% 梯度被随机清零。
4. **论文的三个机制一个都没开。** `algorithm.kind` 是普通 `grpo`。

**教训**：仓库里早已存在逐项对照论文、带 parity 测试的
`online_flash_grpo_kling_video_reward.yaml`，其头注释写明"参考实现优化 HPSv3，
但 vrl 无 HPSv3 集成，故改用 Kling"。HPSv3 集成落地后，正确动作是**在那份配方上
换 reward**，而不是另起一份新配置。

### 最终配置（与参考配方逐项一致，两处有意偏离）

`denoise_mode: sde` · `sde.type: flow_grpo` · `window_size: 1` ·
`window_range: [0,10]` · `noise_level: 1.0` · `timestep_selection: sde_window` ·
`timestep_fraction: 1.0` · `kind: flash_grpo` · `lr: 1e-4` · `kl_coef: 0.0` ·
EMA 0.9/8 · `samples_per_generation_batch: 4`（同温组要求：window 每次**生成请求**
抽一次，一个 prompt 组拆到多次请求会让样本时间步不同，trainer 直接拒绝该 batch）。

两处偏离及其依据：
- `clip_ratio: 1e-2`（参考解析值 1e-4，其注释声称的 1e-3 被
  `recipe/online/denoise_grpo.yaml` 遮蔽）。`ppo_epochs=1` 时策略在 update 内
  不动，该值是**漂移护栏而非信赖域**，必须高于实测漂移：本栈
  `ratio_abs_dev_max=1.95e-3`（编译 rollout vs eager replay 比参考实现漂移更大）。
  1e-3 仍留下 0.51 的裁剪率，1e-2 才归零。
- `torch_compile.scope: rollout`（FSDP2 + gradient checkpointing 禁止编译 replay）。

### 评测本身的一个坑

首次最终评测给出 base 绝对分 **−7.3**，而同一模型一周前测得 **+3.6**。原因是
评测继承了训练配置的采样器：本配方 20 步中只有 1 步随机（`window_size=1`），但
window 参数不进评测的 sampling dict，于是 20 步全部注入满噪声 —— 一个训练与部署
都不存在的工况，权重差异被采样噪声淹没。

> **评测必须自己钉住推理采样器。**复用训练配置看起来严谨，实则相反：训练配置
> 里携带的是探索机制，不属于测量。

`wan_hpsv3_checkpoint_eval.py` 现有显式 `--denoise-mode`，默认 `native`。重测的
base 臂得分 +3.562，与一周前独立测量三位小数一致 —— 这既证实了链路确定性，也
定位了 −7.3 纯属采样器。

### 我们实际改善了什么

base 分数与提升幅度的相关系数 **r = −0.479（t = −3.70，n=48）**：

| base 表现 | n | 平均提升 | 胜率 |
|---|---|---|---|
| 低于均值 | 19 | **+2.02** | 16/19 |
| 高于均值 | 29 | +0.04 | 17/29 |

分数标准差 3.44 → 3.02，最高分 9.09 → 9.18。**收益几乎全部来自"救回失败案例"，
对本来就好的 prompt 无改善** —— 训练得到的是"更少翻车"，不是"上限更高"。

### 下一轮

- **按留出集分数早停**：peak（ckpt-18）在三项指标上全面优于 final（ckpt-50），
  一半预算花在吐回收益上。训练曲线的 5-update 滚动均值准确定位了峰值，可作早停
  信号（但不能作结论）。
- **lr 1e-4 → 3e-5**：18 个 update 即达峰并越过。
- **加大 prompt batch**：24 条/update 的单点标准差 1.3，对效应量 0.8 而言噪声大于信号。
- **目标本身**：HPSv3 逐帧打分且只取最好 30%，运动质量/时间一致性不可见、最差帧
  免责。若要推高上限或改善运动，最低成本的一步是 `score_key: frame_mean`，其次
  是与带时间维度的 reward（VideoScore2 / Kling VideoReward）加权组合。

## 8. 明确不做

- **不碰 Wan 14B 类**：14B 光 FP8 推理就要 40–48GB，优化器与激活无处安放；
  Flash-GRPO 自己的 14B 配方砍到 12 步去噪且跑在 96 卡规模。
- **不碰 VideoAlign 的 Flow-DPO 视频结果**：其发布的 DPO 代码是纯文生图（SD3.5-medium），
  `config/dpo.py` 只有 `geneval_sd3()` / `pickscore_sd3()` 两个入口，无任何视频模型引用。
- **不把 ZeRO-3 当默认**：见 §2。
- ~~**不修改 `vrl/`**~~ **（2026-08-16 撤销）**：方向已改为**把 Flash-GRPO 移植进 `vrl/`**
  而不是在外部仓库复现——本仓的 runtime（Ray / FSDP / checkpoint / drift guard /
  reward 栈）远比那个研究脚本完整。移植已落地，三个机制各就其位：
  1. **temporal gradient rectification** → `FlashGRPO(GRPO)`
     （`vrl/algorithms/grpo/continuous.py`，`algorithm.kind: flash_grpo`）。
     系数按 `1/c(t)`、批均值归一（跨 rank all-reduce）解析计算，
     **不硬编码时间步表**；`tests/algorithms/test_flash_grpo.py` 中的
     parity 测试证明它经由本仓真实的 `sde_step_with_logprob` 复现参考实现的
     硬编码表（`value_dict {999: 7.4770 ...}`）到 0.2%，另有 autograd
     交叉验证 `c(t)` 确为 log-prob 对 velocity 的梯度幅度。
  2. **单随机步 rollout** → 既有 `rollout.sde.window_size=1` + `window_range=[0,10]`；
     窗口改为**每请求解析一次**（seed 派生）并记录进轨迹
     （replay tensor `sde_window`）。
  3. **iso-temporal 分组** → 每请求一次的窗口抽取天然覆盖组内全部 G 个样本
     （chunk 间共享 request seed）。
  4. **只训练随机步** → `actor.timestep_selection: "sde_window"`
     （逐 microbatch 读取记录的窗口；ODE 步永不吃 surrogate loss）。

  配方：`experiment/wan_2_1/online_flash_grpo_kling_video_reward`
  （480×832×81f、20 步 CFG 4.5、G=4、24 prompts/update、lr 1e-4、clip 1e-3、
  EMA 0.9/8、global_std=true —— 全部对齐参考超参）。
  **唯一有意分歧**：reward 用 Kling VideoReward 而非 HPSv3（本仓无 HPSv3 集成，
  三机制与 reward 无关；HPSv3 集成是后续独立工作）。
- **不在 L4 / g6 系列上跑**：L4 是 24GB、72W 的推理卡（AD104，58 SM，
  **dense bf16 仅 121 TFLOPS** = L40S 的 1/3；官网标的 242 是稀疏数字）。
  实测推理基线 21.98GB 已占满 24GB，且**降不下去**——把序列长度砍到约 1/6
  （33帧 320×576）仍要 17.44GB，因为大头是 T5 + VAE + transformer 的固定权重而非激活。
  训练还要叠反传激活、优化器与 rollout buffer，**几乎确定 OOM 且无可降空间**；
  HPSv3 的 16.5GB 更无处安放。时间上 4×L4 外推为 **21–32 天**。
  若只有 L4，应换方向（如调研中的 Video-R1，纯自回归无去噪循环），而非缩小本配方。
- **不试图让 rollout 与 training 重叠**：见 §6.5——被 on-policy 的 `ratio` 定义挡住，
  且显存装不下。异步 RL 是换框架的事（`vrl/rollouts/orchestration/continuous/`
  的 `StalenessPolicy` 与 `SPRINT_continuous_three_stage_pipeline_program.md` 是本仓的
  相关方向），不是给这个脚本打补丁。

## 9. 参考

- Flash-GRPO：<https://github.com/Shredded-Pork/Flash-GRPO> · [arXiv:2605.15980](https://arxiv.org/abs/2605.15980) · [ICML 2026 poster](https://icml.cc/virtual/2026/poster/63629)
- flow_grpo（上游）：<https://github.com/yifan123/flow_grpo>
- HPSv3：<https://github.com/MizzenAI/HPSv3> · [arXiv:2508.03789](https://arxiv.org/abs/2508.03789)
  · **要用的分支** <https://github.com/MizzenAI/HPSv3/tree/upgrade_transformers_version>
  · checkpoint <https://huggingface.co/MizzenAI/HPSv3>
- HPSv3 关键 issue：[#15](https://github.com/MizzenAI/HPSv3/issues/15) /
  [#29](https://github.com/MizzenAI/HPSv3/issues/29)（state_dict 加载失败，#29 正是 flow_grpo 用法）·
  [#3](https://github.com/MizzenAI/HPSv3/issues/3)（额外拉 Qwen2-VL 基座）·
  [#6](https://github.com/MizzenAI/HPSv3/issues/6)（污染 Accelerator）·
  [#9](https://github.com/MizzenAI/HPSv3/issues/9)（分数无界）·
  [#31](https://github.com/MizzenAI/HPSv3/issues/31) / [#32](https://github.com/MizzenAI/HPSv3/issues/32)（reward hacking 面）
- Wan2.1-T2V-1.3B：<https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers>
- 选型调研全文：`docs/research/video-rl-post-training-on-8xL40S.md`
- NVIDIA Ada 专业卡白皮书（解决 bf16 口径冲突）：<https://images.nvidia.com/aem-dam/en-zz/Solutions/technologies/NVIDIA-ADA-GPU-PROVIZ-Architecture-Whitepaper_1.1.pdf>
- VideoX-Fun Wan2.1 训练指南（8×L40S ZeRO-2 全参微调旁证）：<https://github.com/aigc-apps/VideoX-Fun/blob/main/scripts/wan2.1/README_TRAIN.md>
