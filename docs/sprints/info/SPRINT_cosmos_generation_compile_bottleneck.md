# SPRINT: Cosmos generation is launch/bandwidth-bound — torch.compile = 1.68×

状态：info / measured（2026-06-18，单 L40S 46GB）。结论可直接落地：**给 Cosmos
Predict2.5 rollout 打开 `model.torch_compile.enable=true`**。

> 起因：paper-parity run（512p/93f/20-step）单卡生成 ~58s/video，问"吞吐到顶了吗、
> 怎么判断、能不能提"。下面是用 profiler 实测出的答案。

## 0. TL;DR

- 生成 forward **不是 compute-bound**，是 **kernel-launch + 显存带宽 bound**：每个 denoise
  step 启动 **5,122 个 kernel**，**57% 设备时间花在小 elementwise kernel** 上，`dmon` MEM%
  持续 **~82%**。
- **`torch.compile`(default) 把它融合掉 → 1.68× 提速**（57.5→34.3 s/video）。kernel 启动
  **5,122→2,832/step（-45%）**，elementwise 时间 **57%→13%**，MEM% **82%→48%**。
- 单卡纯生成全程（65,536 rollout）：~46d → **~27d**。compile 是免费的；多卡数据并行仍是
  数量级杠杆，叠加在 compile 之上。

## 1. 怎么判断"吞吐到没到顶"（方法论）

`nvidia-smi` 的 **GPU-util% 会骗人**：一个带宽受限的小 kernel 也能让 util 长期 100%。可靠信号
按强度排序：

| 信号 | 看什么 | 可靠性 |
|---|---|---|
| util% | 100% | 弱（必要不充分；≠ 峰值 FLOPS） |
| 功耗 / SM 时钟 | 近功耗墙 + 满频 = 密集算力 | 中 |
| **`nvidia-smi dmon -s u`** | **SM% vs MEM%**；MEM% 接近 SM% = 带宽-bound | 好（硬件计数器） |
| **`torch.profiler` kernel breakdown** | 哪类 kernel 吃时间 + **每 step kernel 启动数** | 好 |
| **实测 `torch.compile`** | 融合后提速多少 = 开销可不可修 | 最硬 |

**反面教材（本 sprint 自己踩的坑）**：只用 batch-scaling（sb=1/2/4 吞吐持平）就断言"compute
到顶"是**错的**——elementwise/launch 开销也随 batch 线性增长，把问题掩盖成平线。必须 profiler
+ compile 实测才看出真相。

## 2. 实测瓶颈（512p/93f/20-step，6 步 profile）

工具：`vrl/scripts/perf/generation_bottleneck_profile.py`（torch.profiler + dmon，复用
仓库 `gemm_projection_breakdown.py` 的 profiler 模式）。

| 指标 | eager | compiled(default) |
|---|---|---|
| wall / step | 2.86s | **1.44s** |
| **kernel 启动 / step** | **5,122** | **2,832**（-45%） |
| 设备时间：compute(gemm+flash-attn) | 40.9% | 40.9% |
| 设备时间：访存(elementwise/copy) | **59.1%** | **13.2%** |
| `dmon` MEM%（显存控制器） | **~82%** | **~48%** |
| 稳态 s/video（含 decode） | 57.5s | **34.3s（1.68×）** |

读法：compute（GEMM + flash attention）只占 ~41%，**~59% 耗在带宽受限的小 elementwise
kernel**；MEM% 82% = HBM 带宽吃紧。compile 把这些 elementwise 融进一个 inductor
`CompiledFxGraph` 大 kernel → 启动数 -45%、HBM 来回搬运减半（MEM% 82→48）→ 1.68× 提速。
flash attention（torch SDPA 自带，无需 `flash_attn` 包）本就在用，**不是**瓶颈。

## 3. 落地

1. **rollout 打开 compile**：`model.torch_compile.enable=true`（config 默认 false）。
   `mode=default` 即可（融合 elementwise）；`max-autotune` 编译极慢且只多调 GEMM（非瓶颈），
   不值。注意首个 rollout 含一次性编译耗时（~1-2min）；分辨率/帧数固定 → 只编一次，不反复重编。
   实现已存在：`CosmosPredict25Model.torch_compile_transformer()`（runtime 在
   `cfg.model.torch_compile.enable` 时调用）。
2. **吞吐量级**仍需多卡数据并行 rollout（仓库 cross-node 基础设施），compile 提速叠加其上。

## 4. 工具（本 sprint 新增的长期资产）

`vrl/scripts/perf/generation_bottleneck_profile.py`：
```bash
python -m vrl.scripts.perf.generation_bottleneck_profile \
  --config experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward \
  --steps 6 [--compile]      # --compile 量化融合效果
```
输出：全 kernel self-time 分桶（gemm/attention/elementwise/copy）+ **每 step 启动数** +
`dmon` SM%/MEM% + chrome trace。

**已知坑（已修）**：torch.profiler 的 `key_averages()` 同时含 CPU 侧 `aten::` 派发包装和
底层 device kernel，两者带相同 device 时间——**两个都累加会双计数**（早期版本 total CUDA
时间 36s > wall 17s 就是这个 bug）。修法：`_is_device_kernel()` 只统计原始 kernel，跳过
`aten::`/`cuda*`/profiler 标记。compiled 路径下 inductor 的 `## Call CompiledFxGraph`
包装事件同样会和其内部 kernel 双计数，分桶里它落进 "other"——比较时以 **kernel 启动数、
elementwise 时间、dmon MEM%、稳态 s/video** 为准，别看 compiled 的设备时间总和。

## 5. 参考

- 工具：`vrl/scripts/perf/generation_bottleneck_profile.py`、复用核心
  `vrl/scripts/perf/gemm_projection_breakdown.py`
- compile 入口：`vrl/models/diffusion/cosmos/predict2_5/model.py:245` `torch_compile_transformer`
- forward 路径：`forward_step`（用 `self.transformer`）→ `vrl/generation/diffusion/executor.py:680` `run_denoise_steps`
- trace：`outputs/perf/gen_eager.json`、`outputs/perf/gen_compiled.json`（perfetto.dev 打开）
- 相关：`docs/sprints/info/SPRINT_cosmos_performance.md`、`SPRINT_rollout_performance.md`、
  `docs/sprints/planned/SPRINT_cosmos_predict25_rl_paper_parity.md`
