# SPRINT: Flash-GRPO on Wan2.1-T2V-1.3B — 8 卡单机复现

状态：**planned / 依赖未就绪（2026-08-15）**。代码侧阻塞已全部清除并本地提交；
唯一未完成的准备工作是 HPSv3 reward model 的安装与 checkpoint 下载。
第一个可执行动作是 §5 Gate 0 的显存冒烟测试。

本文档是外部仓库（`Shredded-Pork/Flash-GRPO`）的复现计划，不改动 `vrl/`。
选型依据见 `docs/research/video-rl-post-training-on-8xL40S.md`。

## 0. 结论先行

在 8×L40 / L40S 48GB（PCIe，无 NVLink）上复现 Flash-GRPO（ICML 2026）：
Wan2.1-T2V-1.3B 基座 + LoRA r=16 + HPSv3 reward + DeepSpeed ZeRO-2。

选它的唯一理由：**它是本次调研中唯一由作者亲自提供单机 8 卡脚本的视频扩散 RL 论文**
（`scripts/multi_node/train_wan2_1_flash_1node.sh`，README 标注 ~40h，附训练曲线），
而不是从多节点配方推断出来的。

整个计划的成败挂在**一个尚未有任何公开来源能回答的问题**上：

> 32,760 token 的激活 + G=4 的 rollout buffer，在 48GB 单卡上放不放得下？

论文未写分辨率/帧数/group size，作者也未说明其 8 卡节点的显存规格（配置文件仅命名为
`dgx`）。这个问题**用一小时就能自己回答**（§5 Gate 0），且必须先回答再投入其余算力。

## 1. 硬件前提与 L40 / L40S 的等价性

用户手上是 8×L40 或 8×L40S 48GB。两者对本计划**几乎等价**：

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

### Gate 0（最高优先级，~1 小时）：单卡峰值显存

**这是整个计划里性价比最高的一小时**，它一次性回答 §0 的核心未知。

前置：独立环境 + HPSv3 + reward server 起来（见 §6）。

```bash
bash scripts/multi_node/train_wan2_1_flash_1node.sh   # 跑 1–2 步即可
# 另一个终端
nvidia-smi --query-gpu=index,memory.used --format=csv -l 5
```

判据：

- **单卡峰值 < ~44GB** → 通过，进入 Gate 1（留 4GB 余量给碎片与 reward server）
- **OOM** → 按下列顺序降，每降一档重测：
  1. `config.sample.num_image_per_prompt` 4 → 2（G 减半，rollout buffer 直接减半）
  2. `config.frames` 81 → 49 或 33
  3. `config.height/width` 480×832 → 320×576
  4. 最后才考虑 ZeRO-3 / param offload（见 §2 的警告）

降档会偏离论文配方，**必须在 §7 记录实际使用值**，否则复现结论不可比。

### Gate 1：短程收敛信号

跑到 `eval_freq=20` 的第一个评测点，确认 reward 曲线上升、无 NaN、无发散。
作者在 README 附了 train/eval 曲线（`asset/train.jpg` / `asset/eval.jpg`）可作对照。

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
| **HPSv3 包 + checkpoint** | ❌ | `pip install git+https://github.com/MizzenAI/HPSv3` + `HPSv3/HPSv3.safetensors` |
| **独立 conda 环境** | ❌ | 必须新建，见下 |

环境必须独立：`setup.py` 钉 `torch==2.6.0 / transformers==4.40.0 / diffusers==0.33.1 /
deepspeed==0.16.4 / peft==0.10.0`。L40/L40S 是 sm_89，该套钉版可正常安装
（对比：本地开发机的 RTX 5090 是 sm_120，torch 2.6.0+cu124 不支持，这也是不能在本机跑的原因之一）。

reward server 启动（注意用 `-c`，README 的命令未带，不会读到 gunicorn 配置）：

```bash
cd flow_grpo/reward-server
gunicorn -c gunicorn.conf.py "app_hpsv3:create_app()"
```

**待定的取舍**：`gunicorn.conf.py` 写死 `NUM_DEVICES = 8`，8 个 reward worker 各占一卡，
与训练争抢显存。建议先按 8 worker 试（reward 打分是 rollout 之后的间歇负载，非持续占用），
Gate 0 若因此 OOM 再降到 2–4 worker。

## 7. 待记录（执行时回填）

- [ ] 实际 GPU 型号（`nvidia-smi -L`）与驱动版本
- [ ] Gate 0 单卡峰值显存
- [ ] 是否降档、降了哪些、最终实际配置值
- [ ] 单 epoch wall-clock，据此外推总时长
- [ ] reward server worker 数的最终取舍
- [ ] Gate 1 首个评测点的 reward 值 vs 作者曲线

## 8. 明确不做

- **不碰 Wan 14B 类**：14B 光 FP8 推理就要 40–48GB，优化器与激活无处安放；
  Flash-GRPO 自己的 14B 配方砍到 12 步去噪且跑在 96 卡规模。
- **不碰 VideoAlign 的 Flow-DPO 视频结果**：其发布的 DPO 代码是纯文生图（SD3.5-medium），
  `config/dpo.py` 只有 `geneval_sd3()` / `pickscore_sd3()` 两个入口，无任何视频模型引用。
- **不把 ZeRO-3 当默认**：见 §2。
- **不修改 `vrl/`**：本 sprint 是外部仓库复现，与本仓代码解耦。若后续要把 Flash-GRPO
  的算法移植进 `vrl/`，那是独立的 sprint，前置条件是本 sprint 的 Gate 1 通过。

## 9. 参考

- Flash-GRPO：<https://github.com/Shredded-Pork/Flash-GRPO> · [arXiv:2605.15980](https://arxiv.org/abs/2605.15980) · [ICML 2026 poster](https://icml.cc/virtual/2026/poster/63629)
- flow_grpo（上游）：<https://github.com/yifan123/flow_grpo>
- HPSv3：<https://github.com/MizzenAI/HPSv3>
- Wan2.1-T2V-1.3B：<https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers>
- 选型调研全文：`docs/research/video-rl-post-training-on-8xL40S.md`
- NVIDIA Ada 专业卡白皮书（解决 bf16 口径冲突）：<https://images.nvidia.com/aem-dam/en-zz/Solutions/technologies/NVIDIA-ADA-GPU-PROVIZ-Architecture-Whitepaper_1.1.pdf>
- VideoX-Fun Wan2.1 训练指南（8×L40S ZeRO-2 全参微调旁证）：<https://github.com/aigc-apps/VideoX-Fun/blob/main/scripts/wan2.1/README_TRAIN.md>
