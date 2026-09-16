# SPRINT: Wan 2.2 A14B GRPO — 学习实验结论与 PCIe 搬权重瓶颈

状态：**planned**。学习实验部分已完成（结论见 §1-§3），吞吐部分（§4-§6）是本
sprint 的可执行项：把每次更新 30 分钟压到 10 分钟量级，同时保持 replay parity = 0。

## 0. 一句话结论

在 4×L40S 上，Wan 2.2 T2V-A14B 双专家 LoRA GRPO 链路机械正确（16 次更新 replay
差异全为 0、严格 resume 逐位确定），并且在训练集 / 验证集 / 独立测试集上给出
同量级、置信区间不含 0 的 Kling visual_quality 提升（+0.16 ~ +0.17，8 次更新）。
但当前配置的 GPU 只用了 2.5 GB/46 GB，每一步都在通过 PCIe 搬 2×28.6 GB 专家
权重；每次更新 30 分钟里 ≥ 90% 的时间是权重搬运，不是计算。

完整运行记录：`docs/runs/wan22_t2v_grpo_20260914/README.md`（分支
`exp/wan22-t2v-grpo`）。运行目录 `/mnt/nvme/outputs/wan22_t2v_grpo/`。

## 1. 学习实验结论（已完成）

配置：`wan22_rebased_gpu_checkpoint_four.yaml` 的已验收数值配置（双专家可训练、
FSDP `precision_policy=none` + `cpu_offload=true`、FP32 LoRA r32、生成 batch ==
replay microbatch == 2、CPS SDE、Kling `memory_parking_mode=reload`、parity gate
1e-8），改动三处并各有探针依据：20 步（10 步首帧马赛克）、`score_key=visual_quality`
（MQ 反向）、lr 1e-4。4 rank × 1 prompt × 8 样本 = 32 样本/更新。

| checkpoint | train_small (16) Δ VQ | val (24) Δ VQ | test (35) Δ VQ |
| --- | ---: | ---: | ---: |
| 2 | +0.009 ± 0.030 | +0.043 ± 0.024 | — |
| 4 | −0.049 ± 0.032 | +0.026 ± 0.033 | — |
| 6 | +0.097 ± 0.053 | +0.138 ± 0.037 | — |
| 8 | **+0.163 ± 0.056**（13/16） | **+0.170 ± 0.034**（20/24） | **+0.174 ± 0.044**（27/35） |
| 12 / 16 | 评估中（stage 2 已于 2026-09-15 17:39 完成 16 次更新） | | |

配对统计以 prompt 为单位（同 prompt 两个 seed 先平均），基线由同一模型禁用
adapter 生成，固定 prompt/seed，确定性 20 步采样。未优化分项 MQ / TA / overall
同向变化。学到的内容主要是修复基线约 90% 视频的坏首帧（ck8 后降到约 55%），
中间帧保持连贯；VQ 绝对值 −1.09 → −0.92，仍低于 Kling 的归一化零点。

## 2. 前置发现（比训练本身更值得记住）

- **10 步 = 首帧垃圾。** 320p/17f 下 10 步 ODE 或 CPS-SDE 的第 0 帧是马赛克、第
  1-3 帧鬼影；无损 PNG 与 mp4 一致，是模型/VAE 输出而非编码。关 VAE tiling、
  480p、官方负面提示词都无效；20 步基本干净，40 步完全干净。仓库此前所有 Wan 2.2
  验收跑都是 10 步，它们的 reward 数值不能当学习信号解读。探针：
  `vrl/scripts/perf/wan22_generation_quality_probe.py`。
- **Kling MQ 在 17 帧下反向**（加噪只在 9-25% 的视频上降低 MQ），overall_reward
  不能作训练目标；VQ、TA 方向正确。与 Cosmos 480p/33f 的发现一致，是 Kling 在短
  片上的通病。探针：`wan22_kling_checkpoint_eval.py probe`。
- **4 rank 主机内存峰值 351-354 GB / 372 GB。** 4 个 rollout worker 各 56-62 GB
  （两个 bf16 专家全驻主机）+ 4 个 trainer 各 28 GB（FSDP CPU 分片）。Ray 默认 95%
  阈值（354 GB）杀过一次 resume；改 `RAY_memory_usage_threshold=0.98` 后稳定，最
  低可用 20-24 GB。这是权宜之计，根因和 §4 同源。
- 评估脚本在 sequential offload 的 rollout 模型上不能用通用 checkpoint restore
  （权重在 meta 设备，strict 回读失败，非 strict 则静默不写入）；必须走家族的
  `load_trainable_state`（挂起 hook、拷贝、逐字节回读）。已修（80175a2d）。

## 3. 机械正确性证据（已完成）

- 16 次更新 `pre_update_logprob_abs_diff_max = 0.000000`（双专家、跨 875 边界）。
- 梯度范数 0.005-0.043，有限非零；`adv_zero_rate = 0`。
- 两个专家各 640 个 FP32 LoRA 张量全部更新，lora_B 范数持续增长；Adam 一阶矩全非零。
- 从 checkpoint-8 严格 resume 后，第 9 次更新的 reward_mean / grad_norm 与被中断的
  那次逐位相同（−0.9885 / 0.020726）。

## 4. 瓶颈：每一步都在 PCIe 上搬权重

stage 1 每次更新（每 rank 8 样本，20 步）的相位耗时：

| 相位 | 秒 | 占比 | 在做什么 |
| --- | ---: | ---: | --- |
| collect.generation | 634 | 36% | rollout 前向：`model.offload_mode=sequential` |
| collect.reward | 43 | 2% | Kling 评分（reload parking） |
| evaluate（replay 前向） | 470 | 27% | FSDP `cpu_offload=true` |
| backward | 615 | 35% | 同上 |
| optim_step + weight sync + park/restore | < 3 | 0% | |

nvidia-smi 表现：每卡 2.5 GB 显存、util 93-96%（util 只表示"有 kernel 在跑"，
包括 H2D 拷贝）。功率 185 W / 350 W。

三处搬运，全部继承自 24 GB L4 / 181 GB 主机的验收配置，在 46 GB L40S / 372 GB
主机上没有重新评估：

1. **Rollout：`offload_mode: sequential`**（Accelerate 逐层 hook）。两个 14B 专家
   都驻主机内存，每个 denoise step 把活跃专家的 40 个 block 逐个搬到 GPU 再搬回。
   每个视频 20 步 × 28.6 GB = 572 GB H2D；8 样本 / 2 per batch = 4 个 batch，
   每 rank 每次更新约 2.3 TB。PCIe 4.0 x16 实测约 20-25 GB/s → 约 100-115 s 纯
   拷贝下限，加上逐层同步开销，与实测 634 s 同量级（真正的 DiT 计算按 L40S 算力
   估算 < 60 s）。
2. **Replay 前向 + 反向：FSDP2 `cpu_offload: true`**。参数分片和梯度都在主机，
   每个 block 前向前 all-gather（需先 H2D 本 rank 分片，再 NCCL 聚合），反向再来
   一次，梯度 reduce-scatter 后 D2H。19 个 replay 步 × 2 专家 × 4 个 batch，同
   样是 TB 级搬运。`full` 激活检查点又让前向再算一遍。
3. **主机内存被同一份权重占 4 遍。** 4 个 rollout worker 各自 mmap 加载两个专家
   (56-62 GB RSS)，加上 trainer 侧 FSDP 分片。这就是 §2 的 354 GB 峰值。

显存实际只需要：活跃专家 bf16 28.6 GB + LoRA + 320p/17f 激活（< 3 GB）+ VAE，
单卡 46 GB 放得下一个专家；FSDP 4 卡分片每卡 2×7.1 GB 参数 + 梯度。

## 5. 可执行项（按预期收益排序，每项独立验证 parity）

验证协议统一：用 `/mnt/nvme/outputs/wan22_i2v_cache` 里已有的固定轨迹夹具
（`wan22_four_sample_fixture`）或 stage1/checkpoint-8 起的 2 次更新短跑，要求
`pre_update_logprob_abs_diff_max == 0`、梯度相对 L2 差 < 1e-4（对比当前配置）、
主机峰值 < 300 GB，并记录每相位耗时。

- [ ] **A. Rollout 改 `offload_mode: model`。** 整个活跃专家驻 GPU，只在 timestep
  875 边界换一次专家（每视频 2 次 × 28.6 GB，而不是 20 × 28.6 GB）。仓库
  `wan_i2v_base_sample.py` 记录过 model 模式峰值约 28 GB。预期 generation
  634 s → 100 s 以内。风险：与同卡 FSDP 训练态的 park/wake 时序、Kling reload 的
  显存争用；需要测 rollout 阶段的显存峰值（应 < 40 GB）。
- [ ] **B. 训练 `fsdp.cpu_offload: false`。** 4 卡分片后每卡约 14 GB 参数（两个专
  家 bf16 冻结 + FP32 LoRA），显存放得下；去掉 H2D/D2H，只剩 NCCL all-gather
  （NVLink 缺席的 L40S 走 PCIe，但 4 卡 all-gather 的总量比 CPU 往返小一半以上）。
  同时评估 `gradient_checkpointing: full` → 只对 attention 做 checkpoint。预期
  evaluate + backward 1085 s → 300 s 以内。风险：训练态 park 到主机时（rollout
  阶段）要搬 14 GB/卡 回主机，`_park_training_state_locally` 的路径要验证；
  `precision_policy=none` 下的梯度 reduce dtype 不变。
- [ ] **C. 4 个 rollout worker 共享冻结专家权重。** 目前每个 worker 独立 mmap +
  materialize，主机内存 4 份。选项：(i) 只有 rank 0 的 worker 加载后通过
  `/dev/shm` 或 CUDA IPC 共享；(ii) 每卡只驻一个专家，高噪/低噪专家按 rank 分工
  （需要改 dispatcher，不推荐）；(iii) 做完 A 之后活跃专家在 GPU、非活跃专家在
  主机，主机份数减半。目标：4 rank 峰值 < 250 GB，撤掉 0.98 的 Ray 阈值。
- [ ] **D. 生成 batch 2 → 4 或 8。** 一旦权重驻 GPU，batch 变大几乎不增加搬运量。
  但 replay batch 必须同步改（此前测过 batch 形状不同 → parity 0.0003-0.0008，
  clip 1e-4 下 40% 样本被裁），并重新过 parity gate。
- [ ] **E. 去掉冗余：** 文本编码器（11.4 GB bf16）在 encode 后可以从 rollout
  worker 释放或只在 rank 0 保留；VAE 解码 tiling 在 320p 下不必要（探针证明关掉
  不影响输出）。
- [ ] **F. 记录**：每一项改完在 `docs/runs/wan22_t2v_grpo_20260914/` 追加相位表，
  把最终推荐配置回写到 `vrl/config/presets/experiment/wan_2_2/` 的公开 preset
  （当前公开 preset 仍是 1 rank / `precision_policy: actor` / 10 步 / 无
  `lora_parameter_dtype`，与所有通过验收的跑都不一致）。

## 6. 非目标 / 已知限制

- 不改算法（GRPO、clip 1e-4、group-local advantage）和 reward 选择；本 sprint
  只动权重放置与 batch 形状。
- 不追求 40 步（首帧完全干净）作为训练几何，成本翻倍；20 步是当前折中。
- Kling VQ 在 320p 下主要衡量首帧质量；更全面的质量结论需要 480p+/33f+ 几何和
  一个 MQ 不反向的 reward，这是另一个 sprint。
- 学习曲线只到 16 次更新、每次 32 样本；是否饱和、lr 是否合适未做扫描。

## 7. 复现入口

```bash
# 训练（4 卡）和评估的完整命令见 docs/runs/wan22_t2v_grpo_20260914/README.md §6
/mnt/nvme/outputs/wan22_t2v_grpo/tools/launch_train.sh <yaml> <name>   # 设 RAY_memory_usage_threshold=0.98
# cron 队列（用户要求）：每 30 分钟检查 4 卡空闲后跑 /mnt/nvme/outputs/gpu_queue/jobs/pending/ 的下一个脚本
```
