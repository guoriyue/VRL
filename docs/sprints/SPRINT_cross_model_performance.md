# SPRINT: Cross-Model Performance（5-track 并行探索综合）

状态：wave 2 GPM scale bump 已落地，待 live gate（2026-06-10）。输入：5 个并行只读探索 agent 的结构化结果
（parity / transport / compute / util / vae-gaps），叠加 P0 trace
（`outputs/cosmos25_perf_profile_bs6/`）与 cross-model smoke
（`docs/sprints/SPRINT_cross_model_smoke.md`）。

## 0. Core Decision

三个跨家族结论：

1. **"busy 95-99%" 是第一层假象**：cosmos 训练段 GPU 实际只有 ~64% 在做事
   （1.02M kernel/step，elementwise 47%），生成段一组视频纯计算 2.8s、墙钟 25s。
   挤性能的对象是 launch 开销和传输/序列化，不是 kernel 本身。
2. **elementwise 风暴 ~70% 来自 LoRA plumbing**（280 个 PEFT 层的双 mm + fp32/bf16
   cast + scaling mul）。cosmos 转全参后自然消失；sd3_5/wan 仍在 LoRA 路径上，
   compile 是它们的主杠杆。
3. **最便宜的全家族杠杆是批量度配置**：wan 25% 利用率 = `sample_batch_size=1`；
   anima 56% = `rollout_batch_size=1` 摊不掉 ~14s 固定周期成本；cosmos 生成段
   25s/组里大半是 12 个单样本 chunk 串行的重复固定成本。

## 1. 正确性优先：predict2 GRPO parity（已修，待 G1）

根因（track 1，已在工作区修复 + 6 项单测 + 310 回归全过）：predict2 的
FlowMatch scheduler 用 EDM 量级 sigma 表（sigma_max=80），而
`sde_step_with_logprob` 的 CPS 公式假设 [0,1] 流域——logprob 是 ~-sigma² 的
垃圾值（-4635），bf16 重算噪声被 ~4600× 放大成 mean 13.9 的偏差；**采样本身
也在注入 EDM 量级噪声，修复前 predict2 上一切 GRPO 信号从采样起即不可信**。
修复为单点换域（`vrl/math/diffusion/flow_matching.py:82-114`，按运行时 sigma
表判域），离线复算 logprob -4635 → -0.98、parity → 0。

- G1 gate：lr=0 短 smoke，`training_debug.jsonl` 的 abs_diff mean < 1e-3、
  approx_kl≈0（运行中：`outputs/predict2_parity_g1/`）。
- wan 的 0.0026 warn 不共享根因（wan sigma 表本就是流域），是独立的 bf16
  重算噪声；G1 残差给出基线后再决定是否调 guard 阈值。
- 留档：windowed SDE（window_size>0）启用前必须把窗口写进 batch context 并在
  trainer 过滤 train_indices，否则确定性步的无意义 logprob 会进 loss。

## 2. Wave 1 — config-only（已应用 2026-06-10，config sweep 105 passed）

| 改动 | 文件 | 依据 / 预期 |
| --- | --- | --- |
| sd3_5 trainer compile on | sd3_5 三个实验 yaml `model.torch_compile.enable: true` | trainer 侧 compile 已接线只是被关；rollout 侧同模型实测 launch -90%；预计步时 -15~25% |
| wan `sample_batch_size` 1→4 | wan 三个 T2V/I2V 实验 yaml | 25%→预计 ~2-3× denoise 吞吐；显存估算 ~21-24GB（b1 实测 18.3GB），安全 |
| wan rollout `denoise_compile` on | 同上三个 yaml | wan trainer compile 本就默认开且实战过；rollout 侧缺失 |
| anima `rollout_batch_size` 1→4 | anima 两个 yaml（nsfw 的 sbs 也 1→4） | 摊销 ~14s 固定周期成本，56%→预计 75-85%。注意：每步 prompt 数 ×4 是实验语义变化（与 cosmos 1→6 同方向） |
| kling `artifact_format` tensor→mp4 | configs/reward/kling_video_reward.yaml | tensor (.pt) 在本环境必坏（decord 读不了，torchvision fallback 缺失）；所有正常 run 已用 mp4 |

验证结果（2026-06-10 实测，lr=0 单 epoch smoke + 500ms GPU 采样）：

| family | 基线 active util | 新配置 active util | 峰值显存 |
| --- | ---: | ---: | --- |
| wan sbs=4 (+denoise_compile) | 25% | **87%** | 18.3 → 18.6 GB（几乎不变 ⇒ 未来可试 sbs=8/n=8） |
| anima rbs=4 | 56% | **90%** | 10.5 → 9.8 GB |

附带修复（验证过程中发现的回归）：`e5af51c` 清空 YAML 占位符后
`kwargs: <name>:` 解析为 null，`MultiReward.from_dict` 的 `**None` 炸掉全部
OCR 实验启动——消费端已改为 `reward_kwargs.get(name) or {}`
（`vrl/rewards/functions/registry.py` + `vrl/scripts/common/factory.py`）。

sd3_5/wan 的 trainer compile 在 steady-state 的 recompile 行为待长 run 时用
`TORCH_LOGS=recompiles` 观察（短 smoke 只覆盖 warmup）。

## 3. Wave 2 — GPM scale bump（已应用 2026-06-10）

GPM 采样目录：

```text
outputs/sm_profile_sd3_5/gpm.csv
outputs/sm_profile_wan/gpm.csv
outputs/sm_profile_anima/gpm.csv
outputs/sm_profile_cosmos2/gpm.csv
```

active 段（`sm_util > 50%`）统计：

| family | profiled shape | sm_util | sm_occupancy | tensor core | DRAM bandwidth | read |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| sd3_5 | n/sbs=8, rbs=8 | 83.7% | 19.9% | 27.8% | 30.9% | busy but under-filled |
| wan 1.3B T2V | n/sbs=4, rbs=1 | 89.2% | 18.4% | 31.4% | 30.2% | sbs=4 仍不是上限 |
| anima | n/sbs=4, rbs=4 | 72.2% | 20.4% | 21.9% | 27.5% | fixed-cycle amortization improved, chunk still small |
| cosmos-predict2 | n=2, sbs=1, 512p93f preview | 97.0% | 26.4% | 30.5% | 30.3% | active compute not saturated |

结论：`nvidia-smi` / `sm_util` 的 70-97% busy 不是 scale 上限；SM 内部 occupancy
只有约 18-26%，tensor core 与 DRAM 都约 20-31%。这是小 shape / fragmented kernel
的指纹。下一档先加 denoise chunk，再用 OOM retry / VAE tiling 兜住容量边界。

| 改动 | 文件 | 依据 / 边界 |
| --- | --- | --- |
| sd3_5 `n/sample_batch_size` 8→16 | sd3_5 三个实验 yaml | 旧 b16 风险来自 VAE decode；`model.memory.vae_decode.tiling=true` 已接上，chunk OOM 仍会自动 split |
| wan 1.3B T2V `n/sample_batch_size` 4→8 | `online_grpo_ocr`, `online_grpo_kling_video_reward`, `online_grpo_physics` | GPM sbs=4 occupancy 18%；保留 video-reward 实验 `rollout_batch_size=1`，不把 prompt batching 和 reward 请求混在一起 |
| anima `n/sample_batch_size` →8 | anima 两个实验 yaml | `rollout_batch_size=4` 已摊掉固定周期；下一瓶颈是每 prompt denoise chunk 小 |
| predict2 `n` 4→8 / per-sample `sample_batch_size` 1→8 | predict2 两个实验 yaml | GPM preview sbs=1 occupancy 26%；同 prompt 内样本共用 reference，sample batching 不改变 V2W conditioning |

明确未改：Wan 14B I2V 没套用这次 1.3B T2V 的 GPM 结论；它的 480p81f + 14B
内存边界不同，仍需要单独 live gate。

### Live gate 结果（2026-06-10，lr=0 单 epoch + GPM 采样）

| family | gate | occupancy（上一档→这一档） | 峰值显存 | 判定 |
| --- | --- | --- | --- | --- |
| wan sbs=8 | **pass** | 18% → 19%（持平） | 17.0 GB | 吞吐另算（occupancy 上限非 batch 决定，是 kernel 形状）；sbs=8 保留 |
| anima sbs=8 | **pass** | 20% → **27%**，DRAM 27→38% | **6.3 GB** | 仍有大余量，下一档可试 16 |
| sd3_5 b16 | **pass（加 slicing 后）** | 20% → 20%，tensor 28→32% | 17.6 GB | 根因：sd3_5 yaml 此前只有 tiling 没有 **slicing**（b16 batch decode 需逐图）；`memory.vae_decode.slicing=true` 后通过。occupancy 与 wan 同样被 kernel 形状钉住 |
| predict2 sbs=8 | **fail，已回退 sbs=1** | — | OOM @ transformer RMSNorm | 512p93f 在 sbs=1 时峰值已 31.8GB，CFG 又把 batch ×2 —— **该分辨率在 32GB 上没有 sample-batch 余量**；26% occupancy 的余量只能靠 compile/融合吃 |

两次 OOM 共同证伪了表格里的安全网假设：**chunk OOM 自动 split 在 ray
executor 路径上不存在**（`vrl/generation/ray/executor.py:93` 直接 raise）。
要么补 ray 路径的 OOM 降级，要么把"自动 split"从假设里删掉——已列入 Wave 4。

## 4. Wave 3 — hours 级（cosmos squeeze 主体，按收益排序）

1. **predict2 VAE tiling 接线**（track 5a）：predict2 的 VAE 就是 wan/anima 已在用
   tiling 的同款 diffusers AutoencoderKLWan，缺的只是
   `vrl/models/diffusion/common/vae_decode_memory.py` 的 ~20 行镜像接线 + 4 行
   config。**解锁单卡原生 1280×704×93f**（encode 路径就是 704p OOM 点）。
2. **解码视频 uint8 过线**（track 2）：717MB/组 wire 预算中 474MB（66%）是 fp32
   解码视频，唯一消费者是 uint8 mp4 编码器。打包为 uint8 = -356MB/组。
   风险点：`vrl/rollouts/batch/ops.py:22` 等假设 float 视频的消费者要兼容。
3. **worker `_to_cpu` pinned+异步 + 去掉逐步 `sigma.item()` 同步**（track 2）：
   376ms/chunk 的 cudaStreamSynchronize，可回收 1-2s/组。
4. **EMA `_foreach` + AdamW `fused=True`**（track 3）：launch 1.7k→~3/step；
   收益小（1-3%）但零风险顺手。
5. cosmos 全参首跑（多卡）直接开 compile：full-param 下 elementwise 基数已剔除
   LoRA 项，预计 forward+backward -20~30%；**不要在单卡 32GB 试**（autotune
   workspace 可能压垮）。

## 5. Wave 4 — day 级（按需）

- trajectory dtype 策略移到 worker 侧 + 按算法裁剪导出（NFT 不消费
  observations/actions 的 216MB/组——现 dtype knob 在
  `batch_builder.py:54`，过线之后才生效，等于没省）。
- fp32 latents → bf16（205GB cast 流量来源）：需过 precision_drift_guard。
- predict2 global reference 传路径字符串而非 PIL（track 5b，launch contract
  primitive-only 校验的正确解法；预编码 latents 更差）。
- LoRA adapter dtype（ToCopyBackward0 ×80k 的来源）：bf16 adapter 需 A/B 稳定性。
- grad-ckpt 对小模型探测性关闭（sd3_5/wan1.3B）：slice 计算 -20~30%，先单步
  显存探测。
- wan-OCR 的 ~74s GPU 空窗（CPU PaddleOCR reward）：先做 3-epoch 时间戳归因，
  比 sbs 更值钱的可能性存在。

## 6. Non-Goals

- AR 家族（janus prefill/decode、nextstep 多卡装载）独立立项，不混入本 sprint。
- 不动 timestep_fraction / lr / reward weights；Wave 2 只把 `n` 当作让
  `sample_batch_size` 生效的容量 knob 处理，后续实验解读必须标记 group size 变化。
- reward worker 内部 mini-batch 打分维持 parked（SPRINT_reward_batched_inference.md）。
- Physical stage runtime / SGLang-Omni-style stage topology 不属于本 sprint；
  另见 `docs/sprints/SPRINT_physical_stage_runtime.md`。

## 7. References

探索原始输出：`/tmp/.../tasks/w1rwdfjj9.output`（5 track 结构化 findings，
514k subagent tokens）。关键代码锚点见各 track 的 evidence 字段；本文档只保留
行动项与依据摘要。
