# SPRINT: Training-side MFU — turn on torch.compile for the replay forward

状态：**planned / measured（2026-06-26）**。性质：**不是新机器,是把已有的 `model.torch_compile.enable` 开关在训练 replay 上逐 recipe 打开,再按实测显存重新评估 `microbatch_size`。** 这是本轮 probe 后仍站得住的训练侧 MFU 优化方向；注意这里的数字只描述 **SD3.5-medium 1024²、no-ckpt replay microbatch**，不能直接外推到 video / grad-checkpoint / FSDP recipe。

> 证据：记忆 `project_rollout_bound_class_probe`、`vrl/scripts/perf/{backward_mfu,dit_mfu,rollout_bottleneck}_probe.py`、`vrl/scripts/perf/compile_benchmark.py`。
> 相关：[[project_cosmos_compile_p2]]（rollout compile 已落地 1.25-1.37x）、[[project_torch_compile_wan]]（mode=default,避开 CUDA-graph/PEFT 冲突）、被本轮证伪的 [[SPRINT_paged_trajectory_store]] / `SPRINT_diffusion_stepwise_batching_probe`。

## 0. 一句话 + 实测数

同一个 transformer 既做 rollout 又做 replay(`base.py` `replay_forward → forward_step`),compile 它 = 训练侧直接吃 fusion。SD3.5-medium 1024² / RTX 5090 实测（两次 probe：一次 eager、一次 `--compile`；**都只看 no-ckpt**）：

```
                  eager          compiled        变化
batch 1 no-ckpt:  60% / 15.7GB → 74% / 12.4GB   +14pp MFU, -21% 显存, 1.22x
batch 2 no-ckpt:  65% / 26.8GB → 79% / 20.1GB   +14pp MFU, -25% 显存, 1.21x
```

两个信号:

1. compile 给 **no-ckpt replay** +14pp MFU、约 1.2x，并在这个 probe 里降低峰值显存。这个显存结论是经验结果，不是架构保证；不同 family / LoRA / checkpointing 下必须重测。
2. compiled no-ckpt replay 的 MFU 从 batch1 74% 到 batch2 79%，说明 batch1 还没完全喂满；但这只支持“继续测 batch3/4”，**不等价于 batch 一定能上调**。batch4 no-ckpt eager 已 OOM，compiled batch4 是否可行必须由 probe 直接回答。

不要把这组训练数字和 rollout 的 `ms/sample` / `dit_mfu` 数字合并判读：rollout probe 是 forward-only + CFG，training probe 是 replay forward+backward + no CFG，瓶颈和显存形态不同。

## 1. 证据:flag 同时覆盖 rollout 和 replay

每个家族有两个 runtime builder,**gate 在同一个 `spec.torch_compile.enable`**:

```text
vrl/models/diffusion/cosmos/predict2_5/runtime.py
  build_cosmos_predict25_runtime_bundle        (rollout)  :80  torch_compile_transformer
  build_cosmos_predict25_replay_runtime_bundle (train)    :140 torch_compile_transformer
```

wan/anima/predict2 同构。所以 `model.torch_compile.enable=true` 一次开两条路,训练 replay 不需要单独旋钮。已有 `compile_benchmark.py` 同时测 rollout + train(forward / forward+backward under grad-ckpt)两条 path 的 latency、launches/step **和数值 parity**——本 sprint 直接复用它做验收,不重造。

## 2. 现状盘点(online 训练 recipe 的 compile 态)

| 态 | recipe | 处置 |
|---|---|---|
| **enable:true**(已开,带 launch-bound profile) | sd3_5 ocr / pickscore / geneval（注释:elementwise 47%, 1M kernels/step）、cosmos_predict2_5 nft_kling_video_reward | 不动,已是目标态 |
| **enable:false（FSDP 硬门）** | sd3_5 ocr_fsdp_2x1_fullparam、cosmos_predict2_5 fsdp_2x1 | **保持 off**：compile+FSDP2 unsound（`strategy.py:480` 硬 raise） |
| **enable:false（validation/debug）** | flux 四个 validation、sd3_5 async_debug、*/smoke | validation 故意 eager 求 parity；**先跑 compile_benchmark parity,绿了再逐个开** |
| **无 block（默认 off）** | anima、wan_2_1/2_2 physics、cosmos_predict2_5 cross_node/ddp/motion、cosmos_predict2 v2w/kling、echo、wan ocr/kling、sd3_5 crossnode_debug | **主战场**：逐个判约束后加 `torch_compile: {enable: true, mode: default}` |

## 3. 三个约束（决定一个 recipe 能不能开）

1. **FSDP2**：`distributed.training.strategy=fsdp` + compile = unsound（`strategy.py:480` 已 raise）。FSDP recipe 一律保持 off,直到 inductor+fully_shard 对齐。
2. **grad-checkpointing**：compile + ckpt 会重编译/冲突([[project_torch_compile_wan]]);实测 `--compile` 时只测了 ckpt=False。**若 compile + no-ckpt 在目标 recipe 上实测能放下，就摘 ckpt**；否则不要用 no-ckpt 的 1.2x 数字承诺真实 recipe 收益。
3. **LoRA / mode**：mode 必须 `default`;`reduce-overhead`/CUDA-graph 撞 PEFT LoRA + grad-ckpt([[project_torch_compile_wan]])。

## 4. 落地清单（逐 recipe,按性价比）

对"无 block"和"非 FSDP 的 enable:false"组:

```text
对每个候选 recipe:
  1. 跑 compile_benchmark --family <f>:  确认 rollout+train 两条 path parity 绿(数值忠实)
  2. 若不走 FSDP 且能关 grad-ckpt:
       加 torch_compile: {enable: true, mode: default}
       同时评估 microbatch_size 上调一档(见 §5)
  3. 跑 backward_mfu_probe --compile 同分辨率:  记录 before/after MFU + peak GB
  4. 短 RL dry-run:  reward / drift guard / TIS-RS 指标不动(parity 已绿,这步兜 e2e)
```

非 FSDP、纯 full-finetune、不强制 ckpt 的(如部分 cosmos_predict2 / anima / wan physics)优先;FSDP 与 validation 组留到对应门解锁。

## 5. 第二杠杆:microbatch_size

训练 MFU 随 batch 还在涨(74→79%)= 小 microbatch 没完全喂饱 GEMM。`model.microbatch_size`（`schema.py:251`）是旋钮，但只能在同 family / 同分辨率 / 同 ckpt 策略下实测决定。compile 在 SD3.5 no-ckpt probe 里省了 ~20% 显存，所以它给了继续测 batch3/4 的空间，不是直接结论：

```text
compile 前 batch 2 = 26.8GB(逼近 32GB)
compile 后 batch 2 = 20.1GB  → 有头寸尝试 batch 3/4；是否 OOM/是否更快必须实测
```

注意:这是 grad-accum 的内层,改它不改有效 batch / 数值语义(memory fix,见 [[project_two_level_async]])。

## 6. 验收

- `compile_benchmark --family <f>`:rollout + train 两条 path 的 `max_abs_grad` / `max_rel_out` 在阈内(compile 数值忠实)——**这是开关前的安全门**。
- `backward_mfu_probe --compile` before/after:目标 recipe 分辨率下 train fwd+bwd MFU 上升、peak GB 下降(复现 §0 量级)。
- 短 RL dry-run:reward 曲线、`ratio_abs_dev`、TIS/RS 触发率与 eager 基线一致(parity 绿的 e2e 兜底)。
- 不开 compile 的 recipe(FSDP/validation)明确标注原因,不留"为什么这条没开"的悬念。

## 7. 非目标

- 不重做 rollout 侧 compile 规划；rollout kernel/MFU 轴已有单独 sprint 和 probe 记录（[[project_cosmos_compile_p2]]）。
- 不动 FSDP2+compile(`strategy.py:480` 硬门没解之前)。
- 不做 compiled-backward 深水区优化(74-79% 之上的 slack 在梯度累加 bandwidth-bound 算子,要自定义 fused backward,收益递减)。
- 不重写 `compile_benchmark` / probe;复用现有 parity + MFU 工具。
- 不把 paged store / stepwise batching 拉回来(本轮实测证伪)。

## 8. 关键文件

- `vrl/models/diffusion/*/runtime.py` —— 两个 builder（rollout + replay）共用 `torch_compile.enable`
- `vrl/config/schema.py:324`（`torch_compile`）、`:251`（`microbatch_size`）、`:514`（`gradient_checkpointing`）
- `vrl/trainers/strategy.py:480` —— compile+FSDP2 硬门
- `vrl/scripts/perf/compile_benchmark.py` —— rollout+train parity + launch-bound profile（验收复用）
- `vrl/scripts/perf/backward_mfu_probe.py`（`--compile`）—— train fwd+bwd MFU before/after
- 证据记忆：`project_rollout_bound_class_probe`、`project_torch_compile_wan`、`project_cosmos_compile_p2`
