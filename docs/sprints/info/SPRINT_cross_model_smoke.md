# SPRINT: Cross-model rollout smoke (2026-06-09)

状态：done（2026-06-09 一次性跨家族 smoke 测量归档；原始产物 outputs/model_smoke_20260609/ 与 data/external/model_smoke_20260609/ 已删除并经核实不存在）。本档发现的 P0 隐患 predict2 GRPO logprob parity 已修复（c66bf11，EDM→flow sigma 域转换，parity 13.9→5.3e-6，回归测试 tests/models/diffusion/test_scheduler_logprob_parity.py）；§4 gap#3 predict2 接 VAE tiling 也已修（3bb1a33，predict2/runtime.py 引入 vae_decode_memory）。注意：旧 SPRINT_predict2_logprob_parity.md 已在 3f417e0 删除，原 parity 取证现归档于本档 §2 + c66bf11 commit message。
2026-06-10 删除 —— predict2 parity 取证数字保存在 SPRINT_predict2_logprob_parity.md）。

## 0. 问题与口径

SD3.5 的利用率结论（见 `SPRINT_rollout_performance.md` U0/U1/D1）是否对其它模型家族成立？
对每个家族跑最小完整训练步：

```text
1 epoch, lr=0（权重不动）, bf16, 单卡 RTX 5090 32GB
trainer.debug.first_step=true（拿 per-chunk stage_durations_s / peak_memory_mb / logprob parity）
nvidia-smi 500ms 采样（利用率/功耗/显存全程）
rollout 组尽量小（n=2~4），不开 compile/profiler，eval off
```

GPU 利用率读法沿用 SD3.5 sprint 的分层：`nvidia-smi util` 只回答"GPU 忙不忙"，
不回答"kernel 吃没吃满 SM"。本 sprint 只做第一层；第二层（NCU）只对异常家族再做。

## 1. 结果总表

| family | config | result | active util mean | pwr max | peak mem | key timing |
| --- | --- | --- | ---: | ---: | ---: | --- |
| sd3_5 (baseline) | online_grpo_ocr | pass | ~100% busy | 575W | 23.5GB (b8) | denoise 0.44s/sample (10 step) |
| cosmos-predict2 2B V2W | online_grpo_kling_video_reward @512p93f | pass | **99%** (>90% for 98% of time) | 583W | 31.8GB | denoise 116s/sample (35 step × CFG) |
| cosmos-predict2.5 2B | online_nft_kling_video_reward, n=2 | pass | **95%** | 582W | 30.7GB | full step 240s |
| cosmos-predict2-anima | online_grpo_aesthetic | pass (after fix §3.1) | 56% | 502W | 10.5GB | denoise 0.69s/sample (10 step, chunk=4) |
| wan_2_1 1.3B | online_grpo_ocr @240p33f | pass (after fix §3.2) | **25%** | 581W | 18.3GB | denoise 2.87s/sample steady (20 step × CFG) |
| janus_pro 1B (AR) | online_grpo_ocr | pass | 57%, never >90% | **300W** | 12.7GB | 16 samples / 56s wall |
| nextstep_1 14B (AR) | online_grpo_ocr | **OOM at weight load** | — | >31GB | 14B bf16 colocated does not fit 32GB |

读法：

```text
93f 视频模型（predict2 / predict2.5）形状够大，busy 层面与 SD3.5 同级 —— "as good as SD3.5"。
wan 25% 不是模型问题：该实验 sample_batch_size=1（anima 用 4）。最便宜的杠杆是提 wan 的 sbs。
janus AR 是逐 token decode loop，1B 模型 300W —— 与 rollout-perf sprint 的判断一致：
  AR 家族需要独立的 prefill/decode 策略，diffusion denoise_compile 覆盖不到。
nextstep：权重本地已有（~/Desktop/NextStep-1/checkpoints/NextStep-1.1, 57GB），不要从 HF 下载；
  运行需 PYTHONPATH=~/Desktop/NextStep-1/inference（nextstep_model 模块）。单卡跑不了，等多卡。
```

## 2. GRPO 正确性（lr=0 ⇒ replay/rollout logprob 应一致，ratio≈1）

| family | logprob abs diff (mean / max) | verdict |
| --- | --- | --- |
| sd3_5 | ~1e-8 | pass |
| anima / janus_pro | exactly 0 | pass |
| wan_2_1 | 0.0026 / 0.045 | warn（超过 1e-3 guard 阈值，值得单独看） |
| cosmos-predict2 GRPO | **13.9 / 138.5**, approx_kl=539, clip_fraction=0.53 | **broken** |

**predict2 是本次最重要的发现**：权重一动未动，replay 重算 logprob 与采集时偏差 mean=13.9。
ratio≈1 不变式被打破 ⇒ predict2 GRPO 的训练信号当前不可信。怀疑方向：CPS SDE 重放路径、
或 per-sample reference 条件在 replay 侧未忠实还原。**在跑任何 predict2 GRPO 实验前先查这个。**
证据：`outputs/model_smoke_20260609/cosmos2_v2w/run/{training_debug.jsonl,metrics.csv}`。

## 3. 修掉的 bug（代码已改，见工作区）

### 3.1 anima adapter self-attn rotary 长度错配

`vrl/models/diffusion/cosmos/anima/adapter.py`：TransformerBlock 的 self-attention 把
qwen 侧长度的 rotary 表（`position_embeddings_context`）用在了自己的 key 上（key 就是 x，
T5 侧长度）。Qwen/T5 对同一 prompt 切出不同 token 数即崩（drawbench 第 1 条 prompt 触发，
11 vs 10）。长度相同时数值恰好正确，所以此前未暴露。修复：self-attn 两侧都用
`position_embeddings`；等长路径逐位不变，错长路径 CPU 验证通过。
现有测试只用 `llm_adapter=torch.nn.Identity()`（`tests/models/diffusion/cosmos/anima/test_forward_step.py:44`），
没有覆盖真实 adapter —— 后续值得补一个错长回归测试。

### 3.2 weight sync 泄漏 `_orig_mod.` 前缀

`vrl/trainers/weight_sync.py` `flatten_trainable_module_state`：trainer 模型开 compile 时
（wan model config 默认 `torch_compile.enable: true`），payload key 带 `_orig_mod.` 前缀，
接收端 strict 校验拒绝。设计约定（rollout-perf sprint D1）是 payload 永远用未编译命名空间，
接收端已 unwrap、sender 漏了。修复：sender 侧 `getattr(module, "_orig_mod", module)`。
SD3.5 不踩是因为它默认不 compile trainer。`tests/trainers/test_weight_sync.py` 6 passed。

## 4. 记录在案的 wiring gap（未修，属设计工作）

> 2026-07-11 对账：第 1 项是旧 Ray 双 actor 架构的历史结果。当前 in-process reward
> contract 只允许一个共享 GPU/CuMem owner；active motion-physics 配方保留 Kling 在 GPU，
> 并把 VideoCon-Physics 显式放到 CPU，静态 reward parking preflight 已覆盖该边界。

```text
1. 双 reward 单卡：motion_physics 的第二个 RewardModelWorker 分不到 GPU
   （资源计划只有 1 个 reward bundle，两个 ray-runtime reward 各起一个 actor）。
2. predict2 global reference 模式把 PIL Image 塞进 executor_kwargs，被 Ray launch contract
   的 primitive-only 校验拒绝（vrl/generation/launch_contract.py）。当前只有
   cosmos.reference_mode=per_sample 可用，且 manifest 里的路径必须在 data/external/ 根下。
3. predict2 没接 VAE tiling（wan/anima 走 vrl/models/diffusion/common/vae_decode_memory.py，
   predict2 缺席）⇒ 原生 704p93f 在 VAE encode 即 OOM，单卡只能 512p。
4. kling reward artifact_format: tensor 在本环境损坏：decord 读不了 .pt，fallback 的
   torchvision.io.read_video 在当前 torchvision 版本不存在。mp4 路径正常（predict2.5 已用）。
5. cosmos2.5 NFT 默认 n=12/rbs=6 单卡一步 >20min：release_after_score 让每个
   rollout→reward 周期拆掉重建 rollout worker（含模型重载，日志每 ~5min 一次 GPU plan resolve）。
6. NFT trainer 不写 first-step debug 记录（uses_evaluator=False），跨家族对比时只能用
   日志 + GPU 采样兜底。
```

## 5. 建议的后续（按优先级）

```text
P0  查 predict2 GRPO logprob parity（§2）—— 正确性问题，先于一切性能工作
P1  wan 实验把 rollout.sample_batch_size 从 1 提到 4~8（已在 cross-model performance wave 1/2 落地）
P2  predict2 接 vae_decode_memory（已有共享模块，wan/anima 是现成模式）解锁原生 704p
P3  AR 家族（janus）单独立项 prefill/decode 优化 —— 与 diffusion denoise 路线分开
```

## 6. 原始产物

```text
outputs/model_smoke_20260609/<model>/
  train.log            完整日志
  gpu_samples.csv      500ms nvidia-smi 采样（util/power/mem）
  result.txt           退出码 + wall time
  run/training_debug.jsonl   per-chunk stage_durations_s / peak_memory_mb / parity（diffusion GRPO 家族）
  run/metrics.csv      训练指标
  run/resolved_config.yaml

一次性产物（结论已落档，确认后可删）：
  outputs/model_smoke_20260609/_run_smoke.sh
  outputs/model_smoke_20260609/reference_704p.png + v2w_smoke.jsonl
  data/external/model_smoke_20260609/reference_704p.png
```
