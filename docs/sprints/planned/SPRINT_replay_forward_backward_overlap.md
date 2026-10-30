# SPRINT：Replay forward/backward 重叠与延后 dW 的独立试点

状态：**planned / CPU 与 tiny-real CUDA 正确性已通过；真实 checkpoint 性能和并行后端验收待执行。**
解锁事件：取得本机 5090 的独占 GPU 窗口；DualPipe/DeepEP 的集成还需要明确的多卡模型分段或专家并行任务，DeepGEMM 需要匹配的 GPU、dtype/layout 与梯度实现。

2026-10-29 对照 `25b1e40b4` 实现。开始下一步前重新核对定义、调用者、配置、测试与硬件。父计划：[执行总表](SPRINT_pytorch2026_execution_program.md)。CUDA 数值试点已通过；独占性能测量未证明可采用的净收益，生产保持原执行路径。

## 1. 已落地的执行入口

- `vrl/scripts/generation/full_sequence_probe.py::prepare_inputs`：抽出原有 recipe 加载、家族构建、编码与采样准备，原 forward 探针和新训练探针共用。原有逐步 logprob/模块定位行为保持。
- `vrl/scripts/perf/replay_overlap_probe.py`：使用正式 denoise loop、typed trajectory、replay bundle、GRPO evaluator/objective 和 `OnlineTrainer.train_on_rollout_batch`，比较一个完整 optimizer update。
- `vrl/scripts/perf/replay_overlap.py`：实验作用域与完整更新比较器；不增加 trainer 配置或默认执行分支。

三个比较臂：

| 臂 | 实际行为 | 尚不能据此声称 |
|---|---|---|
| `serial` | 当前 trainer 的顺序执行 | 是 GPU 性能上限 |
| `streams` | 两个 stream 各自拥有可训练叶节点和 checkpoint 模块状态，共享冻结参数；loss 缩放留在原 stream，显式收集梯度并按原顺序累加到 optimizer 的原始参数，限制每个作用域一次更新 | 两个 stream 一定并行，或一定更快 |
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

以下是明确改变后的独立 SD3.5 小试配方，不冒充原生产配方的性能。使用原生 GPU 编码器路径；编码与加载阶段不在 update 计时中。模型必须已缓存，输出目录必须不存在：

```bash
HF_HUB_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
.venv/bin/python -m vrl.scripts.perf.replay_overlap_probe \
  --config experiment/sd3_5/online_grpo_pickscore \
  trainer.deterministic=true model.torch_compile.enable=false \
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
- [x] 独占 GPU 下通过默认 workspace 的 tiny CUDA 回归，真实 SD3.5 checkpoint 在 128px 和 512px 的三臂完整更新均通过。确定性 recipe 使用明确的 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；不把非确定性 serial 自身漂移归因于候选。
- [x] 按 host `user_annotation` → CPU op External id → GPU kernel 映射，合并区间后求 F/B 交集。两种 shape 均未观察到重叠；没有 distributed collective。GPU annotation 与 host annotation 同名，必须过滤类别；已加入回归测试。profiler 会影响调度，此结果只描述所采 trace，不能证明无 profiler 时绝无重叠。
- [x] 增加 512px 的较大 shape 比较，固定数值门槛并报告波动。当前 eager、单次更新试点未证明稳定净收益，停止生产集成。完整生产分辨率/步数、compile/AC 组合仍是未来独立门槛，本试点不是生产上限。
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

## 5. 独占测量结果与采用决定

2026-10-30，单张 RTX 5090 32GB（SM120）、PyTorch 2.13.0+cu130。按用户要求等待其他 GPU 作业退出，性能测量前后均无其他计算进程；没有结束其他人的作业。所有数值门槛保持 `atol=1e-6, rtol=1e-4`，检查 clip 前梯度、更新参数、optimizer 状态、loss、grad norm，并逐次检查计时更新。两种 shape 均有非零梯度和约 0.0003 的最大参数变化。

| 工作负载 | serial 中位耗时 | streams 中位耗时 | deferred dW 中位耗时 | 结论 |
|---|---:|---:|---:|---|
| 128px，4 samples，4 sampling steps | 1.171s | 1.262s | 1.307s | streams 约慢 8%，dW 约慢 12% |
| 512px，2 samples，2 sampling steps | 0.396s | 0.438s | 0.389s | streams 约慢 11%；dW 约 1.7% 差异小于当前波动，不能认定收益 |

每臂 warmup 1 次；128px 计时 5 次，512px 复测计时 10 次；sampling steps 不等于实际 replay 调用数，以报告中的 `forward_calls` 为准。128px 与 512px 的 batch/步数不同，不能跨行比较分辨率吞吐。较大 shape 不等于已经验收原生产完整配方。

128px 的 allocated 峰值为 5.904 / 7.640 / 10.483 GiB（依表中三臂顺序），dW 队列估算 4.058 GiB。512px 复测为 9.948 / 12.279 / 13.261 GiB，dW 队列约 3.996 GiB。reserved 有 allocator 历史，不作为候选独立新增显存。512px 日志出现 allocator OOM 重试后恢复，完整比较通过，但说明显存压力和耗时离群值需要纳入判断。安装 replica 和状态恢复不在 update 计时内；这不是整个进程的启动或退出耗时。

两种 shape 的 instrumented trace 均为 F/B GPU kernel 区间交集 0ms。已依次验证原共享叶节点、显式 autograd.grad、loss stream 归属和独立叶节点/模块状态方案；最后一个方案消除了 AccumulateGrad stream mismatch，但仍没有测得重叠。同步依赖和 eager host launch 开销都是调查对象，不能把尚未隔离的因素说成唯一根因。官方 [PyTorch backward stream semantics](https://docs.pytorch.org/docs/main/notes/cuda.html#stream-semantics-of-backward-passes) 解释了 backward 与 forward stream、根梯度和消费梯度之间的同步要求。

**决定：不把本试点接入生产、不默认开启 streams 或 delayed dW。** deferred dW 单独只增加保留激活，尚无能利用它的 pipeline 调度。未来仅在 CUDA graph/compiled F-B 或明确多卡 PP 工作负载中重新立项；需跨独立窗口、相同配方稳定改善完整 update，且改善幅度超过实测波动，才进入完整 RL 配方验收。

本地证据（实验产物按生命周期保留在仓库外）：

- `/tmp/vrl-overlap-sd35-128-private-leaves-20261030/`：result、resolved recipe、三臂 trace；`trace_summary.json` 是过滤 host annotation 类别后的重算结果。
- `/tmp/vrl-overlap-sd35-512-final-20261030/`：512px 的 10 次复测和修正后的 trace attribution；报告直接包含 CUDA overlap 汇总。serial 0.387–0.403s、streams 0.398–0.499s、dW 0.386–0.398s。早期 5 次结果另在 `/tmp/vrl-overlap-sd35-512-private-leaves-20261030/`，其 attribution 使用修正的 `trace_summary.json`，不覆盖旧报告。
- `/tmp/vrl-overlap-sd35-512-no-trace-20261030/`：最终 buffer 门槛补充后再测 5 次，不采 trace；中位 serial 0.390s、streams 0.488s、dW 0.391s，dW 没有收益。进程同样在 Python atexit 之后清理等待，不算正常 CLI 退出。
- `/tmp/vrl-overlap-sd35-128-all-20261030/` 与 `...explicit-grad.../`：早期不同实现的证据，不混入最终比较。

所有 profiling 都移到全部 timing arms 完成后，避免 trace 分配/析构干扰后续候选。512px 的带 profiler 与不带 profiler 进程在报告写完、main 返回之后仍占用 GPU；诊断确认已经到达最后注册的 Python atexit，尚未定位更晚的清理等待，因此不能归因于 profiler。只中断这些已完成测量的自有进程（常规中断 exit -2），不将它们声明为 CLI 正常退出或端到端耗时验收。报告中的 `passed=true` 仅指更新数值与测量检查。

最小 serial 清理隔离另在 `/tmp/vrl-overlap-sd35-cleanup-20261030.log`：显式 `gc.collect()` 将 allocated 从约 5.14GB 降到 67MB，`empty_cache()` 成功，最后 atexit 后仍等待；说明清除 Python 模型引用不足以解决更晚的原生退出问题。此时 faulthandler 已卸载，诊断信号结束自有进程（exit -10），未取得栈；原生调试器 attach 也受当前 ptrace 限制。未把 `os._exit` 当作修复，未留下自有 GPU 进程。CLI 正常退出仍是未来采用前的阻塞项。

新的 CLI 遵循 `trainer.seed`、`trainer.deterministic` 和正式 `OnlineRunConfig.initialize_process_rng()`；显式 `--seed` 才覆盖 recipe，比较重置包含 Python、NumPy、CPU/CUDA torch RNG。首个非确定性运行连 serial 自身 optimizer state 都未达到容差，因此后续采用正式确定性配置，而不是放宽门槛。

CPU 回归 27 项通过；tiny-real CUDA 三臂完整更新通过，最终独立叶节点方案没有 AccumulateGrad 警告。覆盖 BF16 autocast、FP32 master、checkpointed dW、非零 LoRA、失败恢复、错误候选拒绝、RNG recipe、buffer 损坏拒绝和原 forward/offload 行为。复核记录在 `/tmp/vrl-overlap-self-review-20261030.json`，代码复核未发现其他问题；正常退出和性能采用门槛尚未满足。验证命令：

```bash
.venv/bin/python -m pytest -q --tb=short -m 'not gpu' \
  tests/scripts/test_replay_overlap_probe.py \
  tests/trainers/online/test_replay_overlap.py \
  tests/scripts/test_full_sequence_denoise_probe.py \
  tests/nn/offload/test_prefetch.py tests/nn/offload/test_block_streaming.py \
  tests/scripts/test_online_run_config.py

.venv/bin/python -m pytest -q --tb=short -m gpu tests/scripts/test_replay_overlap_probe.py
```

512px 命令使用第 2 节相同配置，改 `sampling.width=512 sampling.height=512 sampling.num_steps=2 rollout.n_samples_per_prompt=2 rollout.samples_per_generation_batch=2 'rollout.sde.window_range=[0,2]'`，并使用新输出目录；10 次复测使用 `--repeats 10`。已有 CPU-resident SD3 文本编码器路径在 CUDA token IDs / CPU embedding 上失败，实验保持原 GPU 编码器路径；此问题不算重叠失败。

## 6. 应改、保持与非目标

应改：仅在完整更新与 trace 证明净收益后，把对应执行策略移入正确 owner，并补原配方、状态同步和故障验收；在此之前保持为独立试点。

保持：trainer/evaluator/strategy 的现有职责与统一方法形状；typed trajectory、Parameter/state key、原优化器和更新边界；schema、环境变量、文件名、维度与协议常量。这些边界比少几行代码更重要。

新的 `_StreamLoss` 是 trainer 标量缩放与 CUDA stream 的接口边界，保留它以避免根节点回到默认 stream；`cuda_overlap_summary` 复用已有 `TimeIntervals` 做 trace 证据计算，不能用 host range 叠加替代。`ProbeInputs` 是两个探针共享的准备结果边界；autograd Function 是 dX/dW 的框架接口；CLI 文件负责 recipe/报告，调度模块负责执行实验，均不为了减少 LOC 展平。没有新增全局业务词表或散落 ALL_CAPS prompt 数据；将来若需要大型兼容性/领域表，应放在明确命名的配置资产中。

非目标：不直接移植完整 DeepSeek 训练框架，不把全参 MoE 的收益外推到 LoRA DiT，不增加全局配置开关，不更改生产默认值，不安装不匹配的 CUDA 扩展，不启动 AWS，不后台无限轮询 GPU。
