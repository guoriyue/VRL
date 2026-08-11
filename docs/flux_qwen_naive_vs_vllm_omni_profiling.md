# FLUX / Qwen-Image：naive 扩散路径 vs vLLM-Omni 引擎（端到端 profiling）

性质：工程 profiling + 决策记录（长期资产）。本文是 FLUX/Qwen-Image 家族落地后，把**本仓库 naive
扩散路径**与**真实 vLLM-Omni 引擎**在同口径下对照的结论沉淀。家族落地本身见
[`docs/sprints/done/SPRINT_flux_t2i.md`](sprints/done/SPRINT_flux_t2i.md) /
[`SPRINT_qwen_image_t2i.md`](sprints/parked/SPRINT_qwen_image_t2i.md)。

环境：单卡 RTX 5090（32GB，sm_120/Blackwell）。两侧各用各自 env——naive 跑主环境；vLLM-Omni 跑项目自带的
`.venvs/vllm-omni`（vllm 0.22.0 + vllm_omni 0.22.0，torch 2.11+cu130）。weights 走本地缓存（offline）。

工具（都在 `vrl/scripts/perf/`）：
- naive：`generation_bottleneck_profile.py`（`--e2e` 端到端 / 默认单步 kernel 画像）。
- vLLM-Omni：`vllm_omni_diffusion_profile.py`（在 `.venvs/vllm-omni` 里跑）。

原始数：`outputs/perf/vllm_omni_flux_results.json`（vLLM）、`outputs/perf/flux_*_naive_trace.json`（naive chrome trace）。

---

## 0. 一句话

- **公平对照（端到端一张图，FLUX bf16，256²/4 步）**：naive **538 ms/img** vs vLLM-Omni **~1800 ms/img**。
  **计算时显存两边都 ~24GB**（naive 23.4 / vLLM 实测 24.3）——**不是谁省显存**。
- **速度差 ~3.3× 的真因**：vLLM 的 model-level offload 每张图把 ~24GB transformer 在 CPU↔GPU 换入换出一次；
  naive 把 encoder 永久放 CPU、transformer 常驻，每张图 0 换入。是 **offload 规制差**，不是引擎效率差。
- **fp8 方向相反（与口径无关、公平）**：naive 端到端 fp8 略差（562 vs 538ms），vLLM fp8 更快（1517 vs 1812ms）。
- **Qwen-Image**：两侧都没跑成（权重没下全 + 20B 对单 32GB 卡偏大）。
- ⚠️ **§1 那个 3.3× 是 32GB 卡逼出来的 offload 假象，不是引擎真相**。换到 **48GB L40S、两边都 full-loaded（vLLM 关 offload）**
  后差距塌到 **~1.2×**，且 native 反而略快——但那也大半是**量的口径**偏向 native，不是引擎更强。完整结论见
  **§7（leveled-field 重跑 + 编译口径）**。

---

## 1. 公平对照：端到端一张图（FLUX，256²，4 步）

两侧都量 `generate(one image)` = encode + 4 步去噪 + VAE decode。两侧都跑各自必须的 offload（FLUX 全
pipeline transformer 24GB + T5 9.4GB > 32GB，谁都得 offload 点东西）。

| engine | dtype | **e2e ms/img** | **计算峰值显存**(实测) | 空闲/请求间 | offload 规制 |
|---|---|---|---|---|---|
| naive | bf16 | **538** | **23.4 GB** | 23.4 GB（常驻不释放）| encoder→CPU，transformer 常驻 |
| naive | fp8 | 562 | 29.9 GB | 29.9 GB | 同上 + fp8 |
| vLLM-Omni | bf16 | **~1800** | **24.3 GB**（实测 24882 MiB）| **9.85 GB** | model-level（transformer↔encoder **互斥换出**）|
| vLLM-Omni | fp8 | 1517 | <24.3 GB | <9.85 GB | 同上 + fp8 |

**怎么读（关键，曾踩坑）**：

1. **计算时显存基本相等**：naive 23.4 ≈ vLLM 24.3。vLLM 的 **9.85GB 是请求之间把 transformer 卸回 CPU 后的
   空闲占用，不是计算峰值**——去噪时 transformer 照样回 GPU，峰值 ~24GB（用 `nvidia-smi` 采样实测 24882 MiB）。
   ⚠️ 早期一版误把 9.85GB 当成"vLLM 省 2.4× 显存"，是错的，已实测改正。

2. **速度差的真因 = 每张图一次 ~24GB transformer 的 CPU↔GPU 换入换出**。vLLM 的 model-level offload 是
   **互斥**（transformer XOR encoders），FLUX 二者不能共存（24+9.4>32），所以每张图都要把 transformer 换出
   再换回（PCIe ~16-25GB/s → ~1-1.5s）。naive 把 encoder 永久丢 CPU、encode 在 CPU 上算，transformer 永不
   离开 GPU → 每张图 0 换入，所以快。

3. **为什么 vLLM 不"常驻 24GB 不换"省掉这开销**：vLLM-Omni 只暴露三档 offload——**不开**（33GB>32GB 直接
   OOM，实测过）、**model-level 互斥**（会换 transformer，就是现在这档）、**layerwise**（更碎）。**没有
   "transformer 常驻 + encoder 永久 CPU"那一档**（那正是 naive 的规制）。所以单 32GB 卡上没法把 vLLM 调成
   naive 的省换入规制 → 这条速度差有一部分是"被迫给 vLLM 开了偏激进的 offload"，不全是引擎本身慢。

4. **vLLM 真正赢的是请求间显存**：空闲掉到 9.85GB（transformer 在 CPU），naive 一直占 23.4GB。这对
   **多模型/多 stage 共卡服务**有用；对**单模型背靠背连续生成**反而每张图白付一次换入——所以这个 number
   不能当 vLLM 的最优表现。

> **公平性的边界**：这两个 offload 规制在单 32GB 卡上**无法对齐**（理由见第 3 点），所以"纯引擎效率"没法从
> "offload 开销"里干净剥离。诚实地把口径对齐 + 把各自 offload 摊明，就到此为止。

## 2. fp8 在两引擎里方向相反（与口径无关，公平）

这条是 each engine 内部 bf16-vs-fp8 自比，不跨引擎，所以公平：

| | bf16 | fp8 | 方向 |
|---|---|---|---|
| naive（端到端）| 538ms / 23.4GB | 562ms / 29.9GB | fp8 **略差**（更慢 + 更吃显存）|
| naive（单步前向）| 240ms | 380ms | fp8 **明显更慢** |
| vLLM-Omni（端到端）| 1812ms | 1517ms | fp8 **更快** |

**机理**：本仓库的 rowwise fp8 在小 shape 下，每个 GEMM 的 per-row 动态量化/反量化是净开销（GEMM 没吃满），
而且 quantized + bf16 master 共存让峰值还更高（29.9 vs 23.4GB）——naive 这边 fp8 不划算。vLLM-Omni 那边
fp8 让 transformer 权重减半 → **每张图 offload 换入带宽减半** + 引擎级 fp8 kernel 更优 → 净赚。

**结论**：fp8 该不该开取决于 (1) 你量哪一段、(2) 有没有大权重搬运（offload）。**别用单测小前向给 fp8 下结论。**

## 3. blocked / 不可比的 cell（如实记录）

| engine | dtype | 结果 |
|---|---|---|
| naive / vLLM | fp32 | **OOM**：FLUX fp32 transformer ~48GB > 32GB（两侧都 OOM）|
| vLLM-Omni | fp16(Half) | **跑不起**：`mat1/mat2 dtype BFloat16 vs Half`——FLUX 是 **bf16-native**，vLLM 按 bf16 载权重，喂 Half 激活就撞 dtype。用 bf16。|
| naive | fp16 | 能跑（权重被 cast 到 fp16），但非原生；公平对照故用 bf16 |
| Qwen-Image（两引擎，全精度）| — | **blocked**：权重没下全（offline 载不起）+ 20B>32GB（见 §4）|

## 4. Qwen-Image 的两层 blocker

1. **权重没下完**：Qwen-Image ~40GB，本机网络抖动下反复中断没下全，offline 载不起来。
2. **就算下全，单卡尺寸也悬**：transformer ~20B，bf16 权重 ~40GB > 32GB。naive 直接 OOM；vLLM-Omni 的
   **model-level** offload 在前向时 transformer 仍整份驻 GPU(~40GB) → 还是 OOM。**唯一可能跑通的是
   `enable_layerwise_offload`（逐层换入换出）**，被下载 gate 挡住没测。

补法：先把 40GB 下全，再用 vLLM-Omni `enable_layerwise_offload` 试单卡，或直接上 ≥2 卡；naive 侧 Qwen 即便
下全也 OOM，除非原生 fp8 载入/权重分片。

## 5. 附：import-gate 的教训

主 conda 环境是 **vllm 0.21.0 + vllm_omni 0.18.0**（错配），`import vllm_omni` 报
`ModuleNotFoundError: vllm.inputs.data`；钉到 vllm 0.16.0 又前进到 `get_engine_zmq_addresses` 缺失——
说明 vllm-omni 0.18.0 贴的是 vLLM main 某 commit，**任何已发布 vLLM 都救不了它**。但项目本来就备了对版的
`.venvs/vllm-omni`（0.22.0 对 0.22.0），里面直接能 import 能跑。**教训：动手装版本前先翻 `.venvs/` 有没有
现成对版 env。**

## 6. 复现

```bash
# naive（主环境）—— 端到端 / 单步 kernel 画像：
HF_HUB_OFFLINE=1 python vrl/scripts/perf/generation_bottleneck_profile.py \
  --config experiment/flux/online_grpo_smoke_single_gpu --precision bf16 --e2e
HF_HUB_OFFLINE=1 python vrl/scripts/perf/generation_bottleneck_profile.py \
  --config experiment/flux/online_grpo_smoke_single_gpu --precision bf16 \
  --trace outputs/perf/flux_bf16_naive_trace.json   # 单步 kernel 画像

# vLLM-Omni（.venvs/vllm-omni）：
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
.venvs/vllm-omni/bin/python vrl/scripts/perf/vllm_omni_diffusion_profile.py \
  --model black-forest-labs/FLUX.1-dev --precision bf16 \
  --height 256 --width 256 --steps 4 --out outputs/perf/vllm_omni_flux_results.json
# 量 vLLM 计算峰值显存：跑上面的同时后台 `nvidia-smi --query-gpu=memory.used --format=csv -l 1`
```

---

## 7. leveled-field 重跑（48GB L40S，两边 full-loaded）+ 编译口径（2026-06-22）

§1 的 3.3× 是"native full-loaded vs vLLM 被逼 cpu-offload"的产物。这一轮把混淆变量拿掉：换 **L40S 48GB**，
vLLM 侧 **关掉 offload**（`--no-offload`，`flux_omni_l40s.json` 里 `"cpu_offload": false`），transformer 两边都常驻；
再用新写的 `vrl/scripts/perf/native_denoise_probe.py` 与 omni profiler 做配对诊断。两者共享
prompt/shape/step 协议，但**并非同一计时范围**：native 只量稳态 denoise loop，omni 量完整
`Omni.generate`（包含 engine/worker boundary）。因此下列数字可定位额外开销，不能作为严格的
cross-engine latency ratio。原始数：`/mnt/nvme/perf/{flux,qwen,sd3}_*_l40s.json`。

### 7.1 一句话

- **平整场地后，3.3× 没了**：纯去噪每步 native **63.5 ms** vs omni **76 ms** ≈ **1.2×**，native 略快。
- **native 这点快是被口径喂出来的，不是引擎更强**，按权重排三个原因：(1) 量的范围不同——probe 只量去噪内循环，
  omni 那个数把 encode/decode/IPC 摊进去了；(2) 编译口径不同——native 整图静态 vs omni 逐块 dynamic；
  (3) vLLM 的服务化机制（独立 worker / zmq IPC / scheduler）在单流背靠背时是纯税。
- **vLLM-Omni 默认就编译了**，不存在"一个编译一个没编译"。

### 7.2 别被 omni 那个 `denoise_step_latency_ms` 骗了——它不是纯去噪

omni 的 `denoise_step_latency_ms = stage_gen_time_ms / num_steps`，**把 encode + VAE decode + worker IPC 全摊进每步**。
用两个步数点能干净剥开（`flux_omni_l40s.json`，256²，bf16，full-loaded）：

| steps | `stage_gen_time_ms` |
|---|---|
| 4 | 364.03 |
| 10 | 821.66 |

- **边际每步（真去噪）** = (821.66 − 364.03) / (10 − 4) = **76.3 ms/step**
- **每图固定开销** = 364.03 − 4×76.3 ≈ **59 ms**（encode + decode + 跨进程编排）

`native_denoise_probe.py` 只量去噪闭包（encoder/VAE park 在 CPU、不在计时窗口），**结构上看不到那 59ms**。

### 7.3 同口径对照（纯去噪 s/step，两边 full-loaded + 编译）

| model | shape | native(compiled) | vLLM-Omni | 比值 | 备注 |
|---|---|---|---|---|---|
| FLUX.1-dev | 256² | **63.5 ms/step** | 76.3 ms/step（边际）| ~1.2× | omni 另含 ~59 ms/img 固定税，probe 没量 |
| SD3.5-medium | 512² | **39.2 ms/step** | 65.1 ms/step | ~1.66× | `sd3_*_l40s.json` |
| Qwen-Image (20B) | 256² | 106.8 ms/step（peak 38.3GB）| — | — | omni 侧仍缺数（见 §7.5）|

> native 数：`flux_native_l40s.json` 1.270s/20步=63.5ms、peak 22.3GB；`qwen_native_l40s.json` 2.136s/20步=106.8ms。

### 7.4 编译口径：omni 默认已编译，但和 native 编得不一样

去 `.venvs/vllm-omni` 真包里查证，**编译默认开**，82ms 那批就是编译后的数：

- 开关默认（`diffusion/data.py:499`）：`enforce_eager: bool = False`。
- 门控（`diffusion/worker/diffusion_model_runner.py:170-172`）：`if not self.od_config.enforce_eager:` →
  `_compile_transformer("transformer")`；CUDA `supports_torch_inductor()`（`platforms/cuda/platform.py:185`）= `True`。
- FLUX 可编译（`diffusion/models/flux/flux_transformer.py:530`）：`_repeated_blocks = ["FluxTransformerBlock"]`。
- omni profiler `_build_omni()` 没传 `enforce_eager` → 走默认编译路径。

**但口径不同**（这是 native 略快的一部分原因，不是引擎差）：

| | native probe | vLLM-Omni |
|---|---|---|
| 编译方式 | `model.torch_compile_transformer("default")`（`native_denoise_probe.py:55`）——整模块、默认模式、对固定 256² 静态特化 | `regionally_compile(model, dynamic=True)`（`diffusion_model_runner.py:92`）——**逐块** + **dynamic=True**（形状符号化）|

`dynamic=True` 的逐块编译为"少重编、能换 shape"牺牲了特化；native 单一固定 shape 上整图静态编译本就该更快一点。

### 7.5 这轮里真 break / 缺的（如实记录）

1. **omni 的显存数废了** —— `flux_omni_l40s.json` 里 `"peak_mem_mib": 1.125`。不是真 1MiB，是 vLLM-Omni 把模型跑在
   **独立 worker 进程**，本脚本主进程的 `torch.cuda.max_memory_allocated()` 看不到那块显存（profiler docstring 已标）。
   **速度可比、显存这半边口径是坏的**，要真数得跑时后台 `nvidia-smi` 采样。
2. **qwen-omni 仍无数** —— 只有 `qwen_native_l40s.json`（106.8 ms/step、peak 38.3GB）。Qwen-Image 20B bf16 ~40GB，
   关 offload 在单卡 48GB 上贴边/装不下，§4 那两层 blocker 仍在。

### 7.6 复现

```bash
# native（主环境）—— 纯去噪 s/step，同口径：
HF_HOME=/mnt/nvme/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  python -m vrl.scripts.perf.native_denoise_probe \
    --config experiment/flux/online_grpo_smoke_single_gpu \
    --steps 20 --compile --out /mnt/nvme/perf/flux_native_l40s.json

# vLLM-Omni（.venvs/vllm-omni）—— 关 offload，两边 full-loaded；跑两个步数点以剥出边际每步：
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 .venvs/vllm-omni/bin/python \
  vrl/scripts/perf/vllm_omni_diffusion_profile.py \
    --model black-forest-labs/FLUX.1-dev --precision bf16 --height 256 --width 256 \
    --steps 4  --no-offload --out /mnt/nvme/perf/flux_omni_l40s.json
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 .venvs/vllm-omni/bin/python \
  vrl/scripts/perf/vllm_omni_diffusion_profile.py \
    --model black-forest-labs/FLUX.1-dev --precision bf16 --height 256 --width 256 \
    --steps 10 --no-offload --out /mnt/nvme/perf/flux_omni_l40s.json
# 验证编译确在起作用：上面再加 enforce_eager 跑一版 A/B（omni 侧通过 stage_kwargs 传 enforce_eager=True）
```
