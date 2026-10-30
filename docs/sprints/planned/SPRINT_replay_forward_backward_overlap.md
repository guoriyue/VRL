# SPRINT：Replay forward/backward 重叠与延后 dW 的独立试点

状态：**planned / CPU 与 tiny-real CUDA 正确性已通过；真实 checkpoint 性能和并行后端验收待执行。**
解锁事件：取得本机 5090 的独占 GPU 窗口；DualPipe/DeepEP 的集成还需要明确的多卡模型分段或专家并行任务，DeepGEMM 需要匹配的 GPU、dtype/layout 与梯度实现。

2026-10-29 对照 `25b1e40b4` 实现。开始下一步前重新核对定义、调用者、配置、测试与硬件。父计划：[执行总表](SPRINT_pytorch2026_execution_program.md)。本文件不表示 CUDA 试点已通过或生产已启用。

## 1. 已落地的执行入口

- `vrl/scripts/generation/full_sequence_probe.py::prepare_inputs`：抽出原有 recipe 加载、家族构建、编码与采样准备，原 forward 探针和新训练探针共用。原有逐步 logprob/模块定位行为保持。
- `vrl/scripts/perf/replay_overlap_probe.py`：使用正式 denoise loop、typed trajectory、replay bundle、GRPO evaluator/objective 和 `OnlineTrainer.train_on_rollout_batch`，比较一个完整 optimizer update。
- `vrl/scripts/perf/replay_overlap.py`：实验作用域与完整更新比较器；不增加 trainer 配置或默认执行分支。

三个比较臂：

| 臂 | 实际行为 | 尚不能据此声称 |
|---|---|---|
| `serial` | 当前 trainer 的顺序执行 | 是 GPU 性能上限 |
| `streams` | 不同 replay forward 轮流放到两个 stream；backward 沿对应 forward stream 运行，事件串行保护梯度写入，optimizer 等待两侧完成 | 两个 stream 一定并行，或一定更快 |
| `deferred_wgrad` | 普通 trainable Linear 即时算 dX，保存输入和输出梯度，在实际 clip/optimizer 边界前补齐 dW；冻结 base 不额外算 dW | 等于 DualPipe、支持任意 custom op 或分布式 hook |

延后 dW 保留原始 Parameter 引用。activation checkpointing 的 saved-tensor 解包可能返回不同 Tensor，不能把梯度写到那个临时对象上。排队张量会增加显存，报告保存队列字节估算与 GPU 峰值；队列估算不是精确物理显存账本。

## 2. 可执行命令

在锁同步环境执行。当前环境的检查命令：

```bash
uv sync --dry-run --frozen --inexact --group test --group lint --extra cosmos
```

先检查硬件和依赖，不加载模型、不装额外包：

```bash
.venv/bin/python -m vrl.scripts.perf.replay_overlap_probe \
  --config experiment/sd3_5/online_grpo_pickscore \
  --preflight-only --artifact-dir /tmp/vrl-overlap-preflight-new
```

以下是明确改变后的独立 SD3.5 小试配方，不冒充原生产配方的性能。保持文本编码器与 VAE 在 CPU，减小一次性模型加载显存；这些阶段不在 update 计时中。模型必须已缓存，输出目录必须不存在：

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m vrl.scripts.perf.replay_overlap_probe \
  --config experiment/sd3_5/online_grpo_pickscore \
  model.torch_compile.enable=false \
  'model.memory.cpu_resident=[text_encoder,text_encoder_2,text_encoder_3,vae]' \
  actor.ema.enable=false actor.ppo_epochs=1 actor.prompts_per_collection=1 \
  algorithm.kl_coef=0.0 actor.training_microbatch_size=1 \
  rollout.n_samples_per_prompt=4 rollout.prompts_per_batch=1 \
  rollout.samples_per_generation_batch=4 \
  sampling.width=128 sampling.height=128 sampling.num_steps=4 \
  'rollout.sde.window_range=[0,4]' precision.rollout.dtype=bf16 \
  trainer.debug.first_step=false \
  --lora-b-std 0.01 --seed 0 --warmup 1 --repeats 5 --trace \
  --artifact-dir /tmp/vrl-overlap-sd35-new
```

输出 `resolved.yaml`、`preflight.json`、`result.json` 与各臂 Chrome trace。报告含 revision、工作树状态和探针源码 SHA256。失败写 `passed=false`；某候选失败时保留已完成的 baseline 证据。命令非零退出不能解释成完成。

## 3. 比较边界与验收

- [x] 同一完整固定 trajectory 使用明确的 sample-index 分数，并通过正式算法计算组内优势；不加载 reward 模型。
- [x] kernel warmup 后恢复原始策略；从同一初始参数、buffer、optimizer、trainer/algorithm 状态和 RNG 开始各臂；排除 reset、正确性 clone、trace 与实验作用域安装时间。
- [x] 在实际 clip 前比较所有可见 trainable 梯度，比较更新后参数、optimizer state、loss 与 grad norm；每次计时重复结束后在计时区间外再次核对参数与 optimizer state。拒绝非有限值、零梯度/无变化的空跑和超过预设容差的候选。
- [x] 为更新计时记录每次 wall time、GPU allocated/reserved 起点与峰值；reserved 有 allocator 历史，不能直接当每臂的新增占用。
- [x] 在正常退出和异常路径 join 已发 CUDA 工作、丢弃未完成 dW 队列并恢复原方法；异常时不代替 trainer 做 optimizer step。
- [x] 拒绝不支持的 distributed/offload/resume 配方、GradScaler、EMA/SFT、带 hook/custom-forward 的替换目标；不静默改执行后端或绕过参数 hook。
- [x] 在测量前后检查当前 GPU 的其他计算进程；共享计算负载下拒绝性能结论。`--correctness-only` 只比较更新，所有 timing samples 为空、`timing_measured=false`；小模型数值检查允许共享，不允许把结果当性能报告。
- [x] 在受限 workspace 下通过 tiny-real CUDA 三臂更新一致性；不输出性能数据。
- [ ] 在独占 GPU、默认 workspace 下重跑 CUDA 回归，再运行上述真实 checkpoint 命令。tiny 模型不替代真实 checkpoint 验收。
- [ ] 在 trace 中核对 forward/backward 的实际 GPU 重叠、copy/collective 等待和 critical path；host range 叠在一起不等于 kernel 重叠。
- [ ] 在代表性生产 shape 重复测量，固定数值门槛，报告波动与完整 update 收益。生产原有的 compile/AC 组合也必须成为独立基线，不能把本试点的 eager 当作原生产上限。若两个计算阶段争抢同一资源而没有净收益，停止该臂。
- [ ] 有 update 收益后，单独做原完整配方的 collection/reward/weight-sync 吞吐与 held-out/multi-seed 学习验收，再决定是否集成。

固定分数只用于执行成本与数值比较，**不证明 reward 质量、学习收益或整步 RL 加速**。当前试点关闭 KL/EMA 等是显式的实验范围；生产集成前须恢复原任务条件重新验证。

## 4. DeepSeek 代码逐项处置

| 项目 | 当前证据与处置 | 下一步触发事件 |
|---|---|---|
| DualPipe / DualPipeV | 官方要求按模型实现 `overlapped_forward_backward`；本机仅一张 GPU，当前训练没有 PP 分段。借鉴调度思想，未宣称已移植或跑通多卡 | 至少两张合适 GPU，明确 stage 划分、microbatch、通信依赖、状态/激活预算；与普通 1F1B 比完整更新 |
| dX/dW 拆分 | 已有独立 deferred arm，CPU 覆盖真实 GRPO、非零 LoRA 和 activation checkpointing | CUDA 正确性、保留激活成本和实际关键路径收益通过后，才考虑与 pipeline 调度组合 |
| DeepEP | 本机未安装；此次 probe 没有 token-routed expert-parallel policy。新版实验性 bucket collective 不等于已经接入 FSDP | 明确 EP 或可复现的 DP/CP 通信热点、合适互联和匹配 CUDA/NCCL；先做独立 collective/梯度原型 |
| DeepGEMM | 本机 SM120；当前官方 NVIDIA 文档列出 SM90/SM100。未安装或强行替换 GEMM | 匹配硬件后固定实际 shape、布局、scale、dtype；含量化/转换成本和 dX/dW，再比较完整 update |
| 权重预取 | `vrl/nn/offload/prefetch.py` 已有独立 copy stream、event 和 record_stream；不重复实现 | 对 offload 配方另采 trace，证明传输等待仍暴露，再改预取距离/生命周期；不与本试点共享 mutable prefetch 状态 |

来源：[DualPipe](https://github.com/deepseek-ai/DualPipe)、[带延后 dW 的官方示例](https://github.com/deepseek-ai/DualPipe/blob/main/examples/example_dualpipe.py)、[DeepEP](https://github.com/deepseek-ai/DeepEP)、[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)。兼容性随上游版本变化，执行前须重新锁定 revision 和依赖。

## 5. 当前证据与剩余资源

2026-10-29，本机 PyTorch 2.13.0+cu130、单张 RTX 5090（SM120）。最初检查空闲；随后另一计算进程占用约 30 GB、GPU 利用率约 99%，真实 SD3.5 试跑加载阶段 OOM。没有终止其他任务，也没有取得 GPU 性能结果。最新预检位于 `/tmp/vrl-replay-overlap-preflight-20261029-final/preflight.json`；失败试跑在 `/tmp/vrl-replay-overlap-sd35-20261029-v2/result.json`。后续以实时查询为准。

CPU 回归覆盖真实 trainer 的梯度/更新等价、非零 LoRA 状态安装、checkpointed dW、错误候选被拒绝、失败恢复、unsupported arm 在 warmup 前拒绝、命令行失败 verdict，以及原 forward probe/现有 offload 相关行为。另有 `gpu` 标记的 tiny-real 三臂测试：使用 `CUBLAS_WORKSPACE_CONFIG=:16:8` 降低 workspace 占用，在共享 GPU 上以 `--correctness-only` 通过一次；包含真实 CUDA forward、backward 与 optimizer。serial、streams、deferred_wgrad 的更新均满足数值门槛，但没有采集耗时结论。CPU 还覆盖了 BF16 autocast 下的 FP32 master 更新和 dX，防止 delayed dW 混用 dtype。

PyTorch 报告了 `AccumulateGrad` stream mismatch：跨 microbatch 共享累积节点可能额外同步，也可能影响 CUDA graph capture。保留警告；独占 trace 必须检查这一点。未尝试关闭警告或宣称重叠已经带来收益。默认 workspace 下的完整验收仍未完成。最终命令：

```bash
.venv/bin/python -m pytest -q -m 'not gpu' \
  tests/scripts/test_replay_overlap_probe.py \
  tests/trainers/online/test_replay_overlap.py \
  tests/scripts/test_full_sequence_denoise_probe.py \
  tests/nn/offload/test_prefetch.py tests/nn/offload/test_block_streaming.py

.venv/bin/python -m pytest -q -m gpu tests/scripts/test_replay_overlap_probe.py
```

## 6. 应改、保持与非目标

应改：仅在完整更新与 trace 证明净收益后，把对应执行策略移入正确 owner，并补原配方、状态同步和故障验收；在此之前保持为独立试点。

保持：trainer/evaluator/strategy 的现有职责与统一方法形状；typed trajectory、Parameter/state key、原优化器和更新边界；schema、环境变量、文件名、维度与协议常量。这些边界比少几行代码更重要。

新的 `ProbeInputs` 是两个探针共享的准备结果边界；autograd Function 是 dX/dW 的框架接口；CLI 文件负责 recipe/报告，调度模块负责执行实验，均不为了减少 LOC 展平。没有新增全局业务词表或散落 ALL_CAPS prompt 数据；将来若需要大型兼容性/领域表，应放在明确命名的配置资产中。

非目标：不直接移植完整 DeepSeek 训练框架，不把全参 MoE 的收益外推到 LoRA DiT，不增加全局配置开关，不更改生产默认值，不安装不匹配的 CUDA 扩展，不启动 AWS，不后台无限轮询 GPU。
