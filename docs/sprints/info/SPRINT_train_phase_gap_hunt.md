# INFO：训练段 36% 空转的排查 —— 五个嫌疑人全部排除，外加一个过期数字

状态：**测量存档（2026-08-17）**，不是执行项。
硬件 RTX 5090（32 GiB，测时有其他负载，全部交替采样取中位数）。
基线 main @ `abb8e4da`。

起因：`SPRINT_cross_model_performance.md` §0 的
「cosmos **训练段** GPU 实际只有 ~64% 在做事（1.02M kernel/step，elementwise 47%）」
长期没有 sprint 接手。本文是排查记录。

## 0. 结论先行

**那个数字是过期的。**

| | 日期 |
|---|---|
| 「64% busy / 1.02M kernel/step」测量 | **2026-06-11** |
| cosmos 翻 `torch_compile.enable: true` 落地 | **2026-06-15** |

compile 那次实测的正是这条：**launch 数砍 2.6–2.9×，训练段 1.25×**
（`SPRINT_gemm_utilization.md` §P2）。也就是说 36% 空转**四天后就被直接针对性地
处理了**，只是没有人回头把 §0 的数字标成 pre-compile。

在此之上，本次把五个候选根因逐个实测排除，全部不成立（§1）。
另外测出一个与现有杠杆表**结论相反**的事实（§2），那条值得单独看。

> **给下一个读到 §0 那句话的人**：先确认你手上的数字是不是 2026-06-15 之后
> 重测的。不是的话，先重测，再立 sprint。

## 1. 五个嫌疑人，全部排除

| 嫌疑人 | 实测 | 结论 |
|---|---|---|
| 训练步 metric host sync（16 次/迭代） | 75 µs / 138 ms = **0.05%** | 排除，见 `done/SPRINT_train_step_sync_audit.md` |
| prompt encode 跨 chunk 重复 | **0.4%**（仓库 b8/b16 对照早已测过） | 排除，见 `done/SPRINT_prompt_encode_cache.md` |
| replay tensor H2D（非 pinned） | **~0.2%** | 排除，见 §1.1 |
| replay forward+backward 本身 | **93–96% busy** | 排除，见 §1.2 |
| optimizer + grad clip + EMA | 已 fused/foreach，**在带宽极限** | 排除，见 §1.3 |

### 1.1 replay tensor H2D

trainer 侧 `_move_tensor_tree` 用的是朴素阻塞 `.to(device)`
（`vrl/rollouts/batch/ops.py:83-88`），**非 pinned**；而 rollout worker 的 D2H
侧已经用了 pinned + `non_blocking`（`vrl/generation/execution/worker.py:815-817`）。
这个不对称看起来可疑，量了一下：

本机 H2D 带宽（pinned vs pageable，稳定跨尺寸）：

| 尺寸 | pageable | pinned | 加速 |
|---|---|---|---|
| 64 MB | 25.4 GB/s | 49.5 GB/s | 1.95× |
| 1 GB | 25.6 GB/s | 51.0 GB/s | 1.99× |
| 2 GB | 25.7 GB/s | 50.5 GB/s | 1.97× |

**但字节量太小。** 用真实配置
`experiment/cosmos_predict2/online_grpo_droid_target_480p`
（480×832、33 帧、35 步、8 samples/prompt × 1 prompt）算：
latent ≈ 5×60×104×16 ≈ 0.5M 元素 = 1 MB(bf16)；每步存 observation+action
= 2 MB；35 步 = 70 MB/sample；8 samples ≈ **560 MB/训练步**。

pageable 下 560 MB ≈ **22 ms**，pinned 省 ~11 ms。而 cosmos 训练步是**秒级**。
→ **0.2%**，不值得改。（若将来 batch 或分辨率大一个数量级再回看。）

### 1.2 replay forward+backward 是 GPU-bound 的

合成 wan 形状（12 blocks / d=1536 / seq 2048 / bf16 / rank 32 LoRA），
torch profiler 取 **leaf kernel self-time**（`device_time_total` 会重复计父子，
第一次量出 440% 的假 busy）：

| arm | kernels/iter | GPU 时间 | wall | busy |
|---|---|---|---|---|
| LoRA | 909 | 33.90 ms | 35.19 ms | **96.3%** |
| full-param | 440 | 36.17 ms | 39.02 ms | **92.7%** |

**93–96%，不是 64%。** 而且真实 cosmos 的 seq 更长（480p×33f ≈ 31k token，
比本测量的 2048 大 15×）→ 每个 kernel 的活更多 → busy 只会更高。
**空转不在 replay 里。**

### 1.3 optimizer / clip / EMA 已经在带宽极限

| arm | trainable | replay | clip | opt | ema | 非-replay 占比 |
|---|---:|---:|---:|---:|---:|---:|
| LoRA | 4.72 M | 33.89 ms | 0.16 | 0.12 | 0.02 | **0.9%** |
| full-param | 339.81 M | 36.99 ms | 1.43 | 3.23 | 1.36 | **14.0%** |

关键：这是用**仓库真实配置**测的 —— fused AdamW
（`vrl/trainers/optimizer.py:58-65`）+ `_foreach_lerp_` EMA
（`vrl/trainers/online/ema.py:61-81`）。用朴素 loop 版重测是 **28.1%**，
也就是说**这两处已经做过的优化正好省掉了一半**（两处注释都记了「loop 版本
~1.7k kernel/step」，与本测量一致）。

剩下的 14% 没有余量：AdamW 对 340M 参数要动 param+grad+exp_avg+exp_avg_sq
= 4 × 340M × 4B ≈ 5.4 GB 显存流量，5090 的带宽下理论 ~3.6 ms，实测 3.23 ms
（fp32 state 下更快是因为部分张量是 bf16）。EMA 同理 ~1 ms 理论 / 1.36 ms 实测。
**已经贴着硬件走了。**

## 2. 与现有杠杆表结论相反的一条：全参在**训练段**更慢

`SPRINT_gemm_utilization.md` 的杠杆表把
「**full-param 替 LoRA**（P1.5）」列为非-FP8 路径剩下最大的一条，理由是
「干掉 ~47% elementwise + lora_A/lora_B 瘦 GEMM」。

kernel 数上完全正确 —— 本次实测 LoRA 909 vs 全参 440，**LoRA 是两倍**。
但 **wall time 方向相反**：

| arm | replay | 非-replay | **整步** |
|---|---:|---:|---:|
| LoRA | 33.89 ms | 0.30 ms | **34.19 ms** |
| full-param | 36.99 ms | 6.02 ms | **43.01 ms** |

**全参训练步比 LoRA 慢 26%。** 两个原因叠加：
1. 反向要为 340M 参数算梯度，而 LoRA 只算 4.7M（replay 慢 9%）；
2. optimizer/clip/EMA 要扫 340M 而不是 4.7M（非-replay 从 0.9% 涨到 14%）。

**所以 P1.5 的收益是 rollout-only 的，训练段是净亏。** 是否划算取决于该 run 的
rollout/训练时间配比 —— 决定前应该先量那个比值，而不是当作无条件的杠杆。

> 这也正是 `done/SPRINT_rollout_lora_merge.md` 存在的理由：
> **rollout 侧折叠拿到全参的 GEMM 形状（实测 5–12%），训练侧继续用 LoRA**，
> 两边都要好的那一半，不付全参的训练税和显存。

## 3. 方法学（三条都是这次踩出来的）

1. **交替采样 + 中位数是必需的。** 同一份代码单次顺序测量给出过
   「deferred 慢 7%」和「快 40%」两个相反结论，都是噪声。
   与 2026-08-16 CUDA graph 复测的教训一致。
2. **`key_averages()` 的 `device_time_total` 会重复计父子事件**，直接拿它
   算 GPU busy 会得到 440% 这种数字。要用 `self_device_time_total`。
3. **量"某优化值不值得做"之前，先确认仓库有没有做过。** §1.3 第一版
   用朴素 AdamW/EMA 量出 28%，看起来是个大机会；换成仓库真实用的
   fused/foreach 后是 14%，而且贴着带宽极限 —— 机会是我自己造的。

## 4. 顺带做的 parked 复核（2026-08-17）

既然本次的教训是「一个数字过期了两个月没人标」，就把 `parked/` 的 46 篇按
同一个问题扫了一遍：**有没有别的也已经被后续工作悄悄解决了？**

**结论：没有。归档状态是健康的，不需要动。** 具体：

- **8 篇已带 supersession banner**（`cosmos3_full_support`、
  `cross_request_step_scheduler`、`diffusion_nft_validation`、
  `diffusion_rollout_stage_pipeline`、`generation_scheduler`、
  `physical_stage_runtime`、`video_rollout_stage_overlap`、
  `wan_2_1_i2v_proof_run`）。2026-07-12/13 有过一次维护 pass，就地加了
  「Current-state correction」而不是移走 —— 这是有意的选择，不是遗漏。
- **5 篇是「实测证伪」而非「等事件」**（`diffusion_stepwise_batching_probe`、
  `paged_trajectory_store`、`checkpoint_async_write`、
  `media_artifact_async_write`、`straggler_oversample_control`）。
  初看像是该进 `done/`，但逐篇读下来**每篇都写了真实的重启条件**
  （小 batch/低分辨率的 launch-bound 区间、全参 2B 的 24GB/次 checkpoint 规模、
  慢尾尚未 instrument…），所以按 README 的判据「在等一个 EVENT」是**成立的**，
  留在 `parked/` 正确。
- **只有 1 篇自报触发条件已满足**：`video_rollout_stage_overlap`
  （2026-07-19，4×L4 到位、probe 完成、campaign 未跑）。但**同一篇的顶部
  banner 又说它的架构前提已被删除**（Ray reward actor pool 已整体移除，
  reward 改为进程内 + sleep/wake）。两个事实在同一篇里互相矛盾 ——
  真要重启它，第一步是先判定 campaign 在新架构下还成不成立，而不是直接跑。

- 顺带核过的两个触发条件，**都未满足**：`attention_kernel_medium` 等
  「第三个 attention backend」（今天只有 `vllm_paged.py` 一个）；
  `weight_sync_transport_seam` 等「全参大模型多卡负载」（本机单卡）。

## 5. 相关

- 起点（**注意日期**）：`docs/sprints/info/SPRINT_cross_model_performance.md` §0（2026-06-11）
- 直接处理了它的工作：`docs/sprints/done/SPRINT_gemm_utilization.md` §P2（2026-06-15）
- 本次排除的三项：`docs/sprints/done/SPRINT_train_step_sync_audit.md`、
  `docs/sprints/done/SPRINT_prompt_encode_cache.md`、
  `docs/sprints/done/SPRINT_train_phase_efficiency_program.md`
- 落地的那项：`docs/sprints/done/SPRINT_rollout_lora_merge.md`
