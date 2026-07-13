# SPRINT: GPU 饱和度判读口径 + MPS 同卡共置

> 归档类型：长期测量结果；实验已关闭，但判读口径继续作为性能分析依据。
> 起因：外部文章《10 行代码把小模型吞吐提升 200%》（CUDA MPS + 多副本）。
> 本 sprint 学它的**判据**（不是那 10 行脚本），并在我们自己的机器 + 真实模型上验证。
> 状态：**CLOSED / 杠杆判死（2026-07-12，真机实测）**。MPS 同卡共置在真实 rollout 上
> 只值 1.03x，不值得做。**但这次测量推翻了仓库里一条流传已久的错误读数**（见 §0.2），
> 那个纠正比 MPS 本身重要得多。

## 0. 结论先行

### 0.1 MPS 共置 = 死杠杆（真模型实测，同质与异质都测了）

- **同质**（两个真实 SD3.5 rollout DiT 前向，512²、CFG、真实 chunk=8 → DiT batch 16）：
  合计吞吐 **1.03x**（小 chunk=2 时 1.11x）。合成小 GEMM 的 1.45x **不迁移到真实模型**。
- **异质**（DiT 前向 ⊕ VAE decode，即"用带宽负载填 tensor 负载的空隙"这个更强的假设）：
  合计有效工作量 **1.14x** —— 余量真实存在但很小；**而且拿不到**：MPS 无优先级控制，
  实测它把关键路径 DiT 打慢 **4.18x** 去喂 VAE。rollout 墙钟由 denoise 决定，这是净亏。

**不做。** 这 ~14% 的重叠余量若要拿，也该用**进程内 CUDA streams**，不是跨进程 MPS。

### 0.2 真正的收获：低 SM occupancy **不等于** GPU 没填满（纠正一条错误读数）

仓库里 `info/SPRINT_cross_model_performance.md` §3 写着"sm_util 70-97% 但 occupancy 只有
18-26% ⇒ 小 shape / fragmented kernel 的指纹 ⇒ 还没到 scale 上限"，我上一轮也继承了这个读法。
**这是错的。**

实测反证：MPS 双副本时，occupancy 从 29.2% **升到** 33.1%、DRAM 从 32.7% 升到 38.5%
（说明 MPS 确实把第二个进程的 kernel 塞进了 SM），**而吞吐只涨 3%**。
occupancy 上去了、吞吐没上去 → **occupancy 根本不是这里的限制量**。

根因：20-30% 的 occupancy 是 **tensor-core GEMM 的正常指纹**——它们每线程占用大量寄存器和
shared memory，按设计每个 SM 只驻留很少的 warp。低 occupancy 是"这是个 GEMM"的证据，
不是"机器空着"的证据。

**那什么才是有效判据？**（这一条最初写错了，2026-07-12 修正）

- ❌ **occupancy 读数**：低 occupancy 是 GEMM 指纹，不是空闲证据（上面刚证）。
- ⚠️ **吞吐随 batch 的标度**（ms/fwd 69.1@b4 → 264.7@b16 ≈ 线性）：**是弱证据，不能定案**。
  一个 kernel 完全可以只有 30% 峰值算力、却仍随 batch 线性变慢（激活值带宽受限时就是如此），
  线性 ≠ 到顶。而且改 batch 会同时改 kernel 形状与激活占用，混淆项太多。
- ✅ **共置 A/B**：直接测"再塞一路并发进来能否榨出更多吞吐"——这是"还有没有余量"这个问题的
  **定义式测量**，不是推断（§3.2、§3.3）。
- ✅ **NCU tensor SOL 对照同机方阵 GEMM 上限**：这是"kernel 有没有跑到硬件峰值"的唯一判据
  （cosmos 主 GEMM 45.3% vs 方阵 47.5% = 到顶）。注意它和上一条问的**不是同一个问题**。

counters 只能解释，永远不能定案。

### 0.3 profiling 的口径问题（原始诉求）

三个"利用率"可以在同一个 run 上同时读出 100% / 64% / 29%，而且都对（§2）。
任何饱和度结论必须同时给出三列，并注明是**单 kernel 值**还是**时间平均值**——
但**判死/判活只能靠 batch-标度或共置 A/B 这种吞吐实验**，counters 只能解释、不能定案。

### 0.4 剩下真正的靶子

rollout 在 run 级别是 **64% busy / 36% idle**，其中 ~33% 是无归属的编排间隙
（Ray/Python/传输，见 `project_real_run_profiling`）。**那才是唯一还剩的大头**，
而且它是**时间轴上的洞**，不是 SM 里的洞——解法是进程内 pipelining / resident actor，
不是 MPS（时间片本来就能填空洞，实测双进程无 MPS 在 compute 饱和时是 0.92x，
说明没有空洞可填的地方，多进程只会互相挤）。

## 1. 文章拆解

- 现象：H100 上 Higgs TTS 单副本吞吐见顶、队列满，但 SM Active 只有 ~29%。
- 处置：MPS + 2-3 副本 → 1.4-2.1x。
- **反例（最重要的一半）**：MOSS-TTS 单副本已 ~81% → 加副本零收益。
- 坑：客户端没连上 pipe directory 会**静默退回时间片**（我们原样踩到，§3.2）。
- **不可照搬**：他们是 serving——请求到达驱动、batch 不由自己控制，所以单请求真的很小。
  我们是 RL rollout，**batch 是自己定的**（samples_per_chunk=8 + CFG = DiT batch 16），
  早就把 kernel 喂大了。文章的病我们没有。

## 2. 三个"利用率"不是一回事

| 数字 | 它真正回答的问题 | 工具 | 它**不能**回答 |
| --- | --- | --- | --- |
| `nvidia-smi utilization.gpu` | 采样窗口里"有没有 kernel 驻留" | `nvidia-smi` | 用掉了多少算力 |
| kernel-union busy / wall | GPU 有没有活干、**空洞在哪** | `vrl/utils/nsys_report.py` | kernel 跑起来后填了多少机器 |
| SM occupancy / tensor SOL | 每个 SM 驻留了多少 warp / tensor 管线打满没有 | `vrl/scripts/perf/gpm_sampler.py`、NCU | **有没有可回收的吞吐**（见 §0.2） |

**第三列是最容易被误读的一列。** occupancy 低不代表有头空间；tensor SOL 才接近"打满没有"，
而且必须对照**同机方阵 GEMM 上限**（cosmos 主 GEMM 45.3% vs 方阵 47.5% = 到顶），
不能对照 100%。

顺带解掉一个历史矛盾：`done/SPRINT_cosmos_video_mfu_kernels.md` 说 cosmos GEMM 已到 bf16 上限
（**单 kernel** NCU），`info/SPRINT_cross_model_performance.md` 说 rollout occupancy 18-26%
（**整段时间平均**）。两者测的不是同一个东西，都对；但**后者被解读成"还有头空间"是错的**（§0.2）。

## 3. 实测

### 3.1 合成基线：判据本身成立（`vrl/scripts/perf/mps_colocation_probe.py`）

bf16 GEMM，合计 = 两个客户端之和：

| 场景 | 合计 TFLOPS | 相对单进程 |
| --- | ---: | ---: |
| 未填满 kernel（1024³），有 MPS | 190.0 | **1.45x** |
| 未填满 kernel，无 MPS | 126.8 | 0.97x（时间片） |
| 已饱和 kernel（8192³），有 MPS | 220.2 | 1.01x |

结论：MPS 在 GeForce RTX 5090 上**可用**；判据双向成立；无 MPS 的同卡多进程是**负收益**。

### 3.2 真实模型：杠杆死在这里（决定性）

SD3.5-medium transformer，真实 rollout 形状（512×512、CFG、10-step SDE 的 DiT 前向），
指标 = 合计 sample-steps/s（= 每秒完成多少"样本×去噪步"）：

| chunk | 1 进程 | 2 进程无 MPS | 2 进程 **有 MPS** | MPS 相对单进程 |
| --- | ---: | ---: | ---: | ---: |
| **8（真实值）** | 30.2 | 27.8 (0.92x) | 31.1 | **1.03x** |
| 2（小 chunk） | 29.0 | 27.0 (0.93x) | 32.2 | **1.11x** |

同一窗口的 GPM 三列（chunk=8）：

| 场景 | sm_util | sm_occupancy | tensor | DRAM | 合计吞吐 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 进程 | 92.4% | 29.2% | 26.6% | 32.7% | 30.2 |
| 2 进程 + MPS | **98.3%** | **33.1%** | 27.6% | **38.5%** | 31.1（**+3%**） |

**这张表就是 §0.2 的证据**：MPS 确实把第二个进程塞进了 SM（occupancy ↑、DRAM ↑、
sm_util ↑），**吞吐却只涨 3%**。机器早就没有可回收的吞吐了，counters 上的"余量"是假的。

（batch-标度只作旁证，不作判据——理由见 §0.2。）

### 3.3 异质共置：唯一还剩的假设，也死了

§3.2 测的是**两个同质负载**（都吃 tensor 管线），抢不出收益几乎是必然的——它**没有**证伪
真正有意思的那个假设：denoise 时 DRAM 只有 33%、tensor 只有 27%，**两条管线都没满**，
那把吃带宽的 VAE decode 叠到吃 tensor 的 denoise 上呢？

实测（SD3.5，DiT 前向 vs VAE decode，各自独立进程）：

| | 单独 | 无 MPS 共置 | **有 MPS 共置** |
| --- | ---: | ---: | ---: |
| DiT 前向 | 3.765 /s | 1.733（慢 2.17x） | 0.902（**慢 4.18x**） |
| VAE decode | 5.289 /s | 2.549（慢 2.07x） | 4.199（只慢 1.26x） |
| **合计有效工作量** | — | **1.04x**（≈ 无重叠） | **1.14x** |

（合计有效工作量 = 两边完成的工作按各自单独速率折算成串行耗时 ÷ 实际墙钟窗口。
1.0 = 完全没重叠，2.0 = 完美重叠。）

两条结论：

1. **确实存在重叠余量，但只有 ~14%**——远不是合成小 GEMM 那个 45%。机器在"做功"意义上
   已经用掉了约 85-90%。
2. **而且这 14% 拿不到**：MPS 没有优先级控制，实测它偏袒 VAE（多而小的 kernel 抢先），
   把 **DiT 打慢了 4.18x**。而 rollout 的墙钟就是 denoise 决定的——用关键路径 4 倍的代价去
   换 VAE 早点算完，是净亏。
3. 顺带一提：**这 14% 就算要拿，也不该用 MPS 拿**——进程内 CUDA streams 就能重叠阶段，
   不需要跨进程。我们今天不重叠 decode 与 denoise 是**程序结构**是顺序的，不是硬件不允许。

### 3.4 踩到的坑（都是真的）

- **静默退回时间片**：自定义 `CUDA_MPS_PIPE_DIRECTORY`（指到 `/tmp/claude-*/…`）会让
  control daemon 启动后**立刻退出**，客户端**静默**用普通时间片跑——第一轮我因此拿到一张
  自洽的"MPS 无效"假表。默认 pipe dir 正常。探针现在强制校验：窗口内必须在 `nvidia-smi`
  里看到 `nvidia-cuda-mps-server`，否则不许标成 "MPS"。
- **daemon 是 per-user 全局的**：开着时该用户新起的任何 CUDA 进程都会挂上去（实测把用户
  当时在跑的 pytest 一并接管了）。**不能接进训练启动脚本**，必须显式手动开关。
- **`gpm_sampler` 一直是跑不起来的**：它 import `pynvml`，而 `nvidia-ml-py` 从未在
  `pyproject.toml` 里声明——这就是这个仪器建好之后从没进过任何 runbook 的原因。
  本 sprint 补上 `[perf]` extra。

## 4. 各家族判决

| family | 判决 | 依据 |
| --- | --- | --- |
| sd3_5（image） | **共置无用**（1.03x @真实 chunk） | §3.2 真机 A/B |
| cosmos / wan（video） | **共置无用**，双重死 | 显存无余量（512p93f sbs=1 已 31.8GB）+ 主 GEMM 已到 bf16 上限 |
| anima 等小模型 | 不再单独试 | occupancy 低是 GEMM 指纹（§0.2），不是空闲证据；它的 kernel 形状与 sd3_5 同类 |

`vrl/ray/resources.py:190`（拒绝分数 GPU）与 `placement.py:180`（拒绝同卡双 worker）
**保持不动**——实测证明放开它们没有收益，无 MPS 时反而是 0.92x。

## 5. 下一步（还剩的靶子，按大小排序）

1. **时间轴上的空洞（大头）**：rollout run 级 64% busy / 36% idle，其中 ~33% 是无归属的
   编排间隙（Ray/Python/传输）。这是**时间上的洞，不是 SM 里的洞**——解法是
   **进程内 pipelining / resident actor**（已有 sprint：
   `parked/SPRINT_diffusion_rollout_stage_pipeline.md`、
   `planned/SPRINT_miles_phase_lease_and_one_continuous.md`），不是同卡多副本。
2. **阶段重叠（小头，已量化上界 ~14%）**：§3.3 测出 denoise ⊕ decode 的重叠余量约 14%。
   若要拿，走**进程内 CUDA streams**（`parked/SPRINT_video_rollout_stage_overlap.md`），
   MPS 路线因无优先级控制而净亏。注意这 14% 是**上界**，实际 decode 只占 rollout 一小段时间。

## 6. 非目标

- 不再试 MPS/MIG/同卡多副本（本 sprint 已判死）。
- 不把 MPS daemon 接进任何启动脚本。
- 不再用 occupancy 读数论证"还有头空间"——只用吞吐-batch 标度或共置 A/B 定案。

## 参考

- 外部文章：<https://www.linkedin.com/pulse/10-行代码把小模型吞吐提升200-jiaxin-deng-jnowc/>
- 合成判据探针：`vrl/scripts/perf/mps_colocation_probe.py`（+ `tests/scripts/perf/test_mps_colocation_probe.py`）
- SM 级采样：`vrl/scripts/perf/gpm_sampler.py`（现由 `pyproject.toml` 的 `[perf]` extra 提供 `nvidia-ml-py`）
- MFU 分母标定：`vrl/scripts/perf/gemm_peak_probe.py`、`vrl/scripts/perf/gpu_preflight.py`
- 被本 sprint 纠正的读数：`docs/sprints/info/SPRINT_cross_model_performance.md` §3
- cosmos 单 kernel 饱和结论：`docs/sprints/done/SPRINT_cosmos_video_mfu_kernels.md` §0
- 编排空洞（真正的靶子）：`docs/sprints/info/SPRINT_rollout_performance.md` §U0
