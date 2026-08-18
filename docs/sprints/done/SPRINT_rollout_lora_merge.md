# SPRINT：rollout 侧 merge LoRA → dense —— 实测后撤销

状态：**done / 已实施并撤销（2026-08-17）**。功能完整落地过（机制 + pass +
worker 重折 + trainer 护栏 + 16 个测试，全量 gate 绿），随后**按实测收益撤销**。
生产代码已回到 `origin/main` 状态；本文保留全部测量，是这次工作的真正产出。

基线 main @ `abb8e4da`。撤销提交见文末。

## 0. 结论先行

`SPRINT_gemm_utilization.md` 杠杆表里唯一「列了但没写」的一条，写了，测了，撤了。

**撤销理由 = 收益不抵代价**：

| | |
|---|---|
| 真实 Wan2.1-1.3B 生产分辨率收益 | **0.6–4%**（§1） |
| 代价 | 3 个 hazard 各需一个护栏 + 278 行生产代码（§3） |

**关键教训:第一版报的 5–12% 是错的。** 那是合成 stack（12 blocks / 48 个
LoRA 层 / seq 1024–4096）的数字,而真实模型是 30 blocks / 240 层、生产 latent
32k–106k token。**收益随序列长度衰减,而我引用了自己曲线上最有利的那一端。**
在真模型上重测才得到 0.6–4%。

## 1. 真实模型实测（这才是决策该用的数字）

真实 **Wan2.1-T2V-1.3B**（30 blocks，仓库自己的 LoRA 配置 → **240 个 LoRA 层**），
每次去噪前向，交替 A/B、13 对、取中位数：

| latent | token 数 | adapter 活跃 | 折叠 | 省 | p25/p50/p75 |
|---|---:|---:|---:|---:|---|
| 240p `9×30×52` | 14k | 89.31 ms | 82.44 ms | **7.7%** | — |
| 360p `9×45×80` | 32k | 235.11 ms | 233.78 ms | **0.6%** | 0.2/0.6/0.6% |
| **480p `9×60×104`（生产）** | **56k** | **478.61 ms** | **465.82 ms** | **2.7%** | 2.5/2.6/3.1% |
| 480p 17 帧 | 106k | 1232.14 ms | 1182.84 ms | **4.0%** | — |

两点必须记住：

1. **收益随 latent 变大而衰减**，因为 LoRA 的开销是每层固定的（rank-32 的瘦
   GEMM），而主 GEMM 随序列长度增长。合成 stack 的 seq 1024 落在曲线最左端，
   所以给出 12%；生产视频落在右端。
2. **不单调**（0.6% → 2.7% → 4.0%），四分位很窄说明这不是噪声，是 shape 相关的
   kernel 选择效应。**不要从一个 shape 外推到另一个。**

**决策数字：视频家族生产分辨率下 0.6–4%。** 低分辨率（240p）或 latent 更小的
图像家族才接近 8%。

## 2. 为什么有效

rollout 每个 chunk 把 policy 重放 `steps × CFG` 次。adapter 活跃时每个被 target 的
投影是 3 个 GEMM（base、`lora_A`、`lora_B`）+ 一次 scaling 乘；折叠后是 1 个。
这正是 `SPRINT_cross_model_performance.md` §0 记的「~70% elementwise 来自 LoRA
plumbing（280 个 PEFT 层）」。

相对 P1.5（`use_lora:false` 转全参）的好处：**拿到全参的 GEMM 形状，不付全参的
显存**——训练侧仍然只存 LoRA 的优化器状态。

## 3. 当时的落地形态（已撤销，留作重开时的蓝图）

- `vrl/nn/optimization/lora_merge.py` —— 机制。按 `peft.tuners.lora.LoraLayer`
  **走树**而不是问模型，因为本仓两个 adapter 表面接法不同（`get_peft_model`
  包整棵树；diffusers `PeftAdapterMixin.add_adapter`（Wan / cosmos-predict2）不包），
  层类型是唯一同时覆盖两者的 seam —— 和 `swap_linears` 走 `nn.Linear` 同理。
- `LoraMergePass`（`vrl/nn/optimization/passes.py`）—— 排在**最前**，
  在 quantization 之前，让低精度替换看到**有效权重**。
- `model.lora.merge_for_rollout`（默认 `null`/off），identity 标 `exclude`：
  折叠是同一个 policy 的改写，不是另一个模型。**一处声明、一处透传、一处读取** ——
  schema 一个字段，`ModelBuild.lora` 的现成 extras 列表加一个键，
  `LoraMergePass.enabled` 一次读取。第一版曾在 `RolloutBuildOptions` 上另立字段、
  经 registry 接线、再在 `run.py` 裁 wire，中途还把名字从 `merge_for_rollout`
  改成 `merge_lora` —— **全是多余的**：pass 只在 `build_denoise_runtime_bundle`
  里跑（`assemble_replay_bundle` 从不调 `apply_rollout_optimizations`），
  rollout-only 是结构保证，不需要再用一个字段去表达。
- `update_weights` 每次同步后重折（`vrl/generation/execution/worker.py`）。
- `require_every_core_merged` 查**层**不查自报，照抄
  `require_every_core_quantized` 的理由（Wan 双专家半覆盖 bug）。
- **`OnlineTrainer` 拒绝折叠过的策略**（`vrl/trainers/online/trainer.py`）。
  PEFT 把已 merge 的层直接路由到冻结的 base layer，adapter **离开 autograd 图**
  —— 训练要么当场 `does not require grad`，要么梯度缺失，总之策略不再更新。
  生产走不到（rollout 在自己的 Ray actor 里），但把**同一个 bundle** 同时交给
  executor 和 trainer 的 harness 会（`tests/e2e/test_real_checkpoint_rl.py`
  就是这么搭的），所以在 trainer 入口拒绝，而不是等它以两种症状之一暴露。

## 4. 撤销前的验收结果（全部通过）

- ✅ 数值：端到端前向 max rel diff **0.0**（bf16，本例恰好逐位相同；
  `introduces_replay_drift=True` 仍按保守声明，因为累加顺序确实变了）。
- ✅ 独立复算：折叠权重 == `pristine + (alpha/r)·B@A`，由 adapter 张量独立算出对拍。
- ✅ 不漂移：200 轮随机 adapter 重折后，折入零 adapter 得到与建库快照**逐位相同**的基权重。
- ✅ 覆盖完整：多核半折叠被 `require_every_core_merged` 拒绝。
- ✅ 权重同步：重折后前向 == 持同一 adapter 的未折叠模型；并有反向对照测试证明
  「不重折就会服务旧权重」是可观测的（防止上一条空过）。
- ✅ 全量 CPU gate：**4126 passed**。剩余 2 个
  `tests/architecture/test_generation_rollout_boundaries.py` 失败**在 `origin/main`
  上预存在**（stash 验证过），与本 sprint 无关。
- ✅ **已做（2026-08-17）**：真实 Wan2.1-T2V-1.3B、真实 35 步去噪链上的
  rollout-vs-replay logprob parity。**mean 过、max 不过**，且 max 的不过是
  σ→0 的固有性质、不是折叠特有 —— 完整数据与对照见 §5。

## 5. 计划被推翻的三处

**① 「colocated 必须排除」——错。**
计划里写 colocated 共用同一个 model 对象，所以 merge 会破坏训练梯度和
`disable_adapter()`。读代码后发现不成立：rollout 走
`build_denoise_runtime_bundle`，replay 走 `assemble_replay_bundle`，
**永远是两个实例**——权重同步存在本身就是证据（共享对象就不需要同步）。
colocated 共享的是 GPU（靠 parking 交接），不是模型对象。
所以**不需要 colocated 特例**，`enabled()` 只看 `rollout is not None`。

**② 「用 PEFT 的 merge/unmerge 往返」——不可行。**
计划假设 merge/unmerge 可以反复做。实测 bf16 下相对误差：1 轮 4.8e-3，
1000 轮 **2.4e-1**——基权重在一次正常 run 里就被毁掉。
（fp32 无此问题：1000 轮 2.6e-6。）
改为：建库时快照一份 pristine 基权重（CPU 常驻），每次折叠都是
`pristine + delta`，**永不累加**。CPU 常驻是因为 rollout worker 的显存才是稀缺项，
而这份快照每次权重同步读一次，不是每个 denoise step 读一次。

**③ 发现了计划里没有的冲突：versioned trainable-state slots。**
非 draining 同步会同时保留多个 adapter 版本，让 in-flight 请求各自激活自己那版。
折叠后**没有 adapter 可切**——版本就是基权重本身，而基权重只有一份。
`conflicts()` 看不到这个（`versioned_weight_sync` 在 launch contract 上，不在
`ModelBuild` 上），所以拒绝落在 worker 里两者都可见的地方。
`conflicts()` 保持返回 `()` 并在 docstring 说明原因 ——
**不能触发的 guard 比没有 guard 更糟**（`CompilePass` 同款理由）。

## 6. Parity 红线：实测结果（2026-08-17）

在**真实 Wan2.1-T2V-1.3B**、真实 35 步去噪链、仓库自己的
`sde_step_with_logprob` + `compute_logprob_mismatch_stats` 上跑的。
rollout 腿用折叠策略走完整条链并存下每步 `(latent, prev_sample, logprob)`，
replay 腿用未折叠策略对同一批 `prev_sample` 重新打分 —— 就是 trainer 做的事。

### 5.1 结果

| | steps 0–33 平均 | 末步(34) | 总 mean | 总 max |
|---|---:|---:|---:|---:|
| **LoRA 折叠** | **2.2e-04** | 1.57e-01 | **4.7e-03 ✅** | 1.57e-01 ❌ |
| fp8 rowwise（**已发货**，作对照） | 1.17e-03 | 8.59e-01 | 2.57e-02 ❌ | 8.59e-01 ❌ |
| 无 drift 源（对照） | 0.0 | 0.0 | **0.0** | **0.0** |

两个门不是同一个：

- **mean ≤ 1e-2** —— 折叠 **过**（4.7e-3，约 2× 余量）。
- **max ≤ `trainer.debug.max_abs_logprob_diff`（默认 1e-2）** —— 折叠 **不过**
  （1.57e-01）。这是 `_validate_first_update_parity` 的硬 gate，会
  `raise RuntimeError`，但**只在 `trainer.debug.first_step=true` 时生效**
  （默认 false；44 个 preset 开着它）。

### 5.2 末步为什么炸：不是折叠的锅

逐步曲线（末 6 步）：`1.5e-4 → 2.9e-4 → 5.0e-4 → 1.2e-3 → 4.7e-3 → 1.57e-1`。

σ→0 时 logprob 是 `-(x-mean)²/(2σ²)`，**任何**对 `noise_pred` 的扰动都被 1/σ²
放大。三条证据说明这不是折叠的算术问题：

1. **fp32 折叠救不了**：把 delta 在 fp32 里算完再 cast，末步 1.58e-01
   —— 与 bf16 折叠的 1.57e-01 无差别。
2. **无 drift 对照精确为 0.0**：同一条链用同一个折叠策略重放，逐位相同。
   所以扰动确实来自折叠，但放大器来自 SDE。
3. **已发货的 fp8 量化更差 5 倍**：末步 8.59e-01、bulk 1.17e-03，
   而且**连 mean 门都不过**（2.57e-02）。

也就是说：**这个 max gate 与任何 drift 源都不兼容**，折叠只是恰好比仓库
已经支持的那个干净 5 倍。顺带一个观察：**没有任何 checked-in preset 打开
量化**（只走显式 override），而 44 个 preset 开着 parity gate —— 这两件事
从来没有在同一个 run 里碰过面。

### 5.3 因此，开启条件

`model.lora.merge_for_rollout: true` **保持默认关**。可以开的条件：

- **该 run 没开 `trainer.debug.first_step`**（默认就是关的），或者
- 开了 first_step 但训练不重放末步。

不能开的条件：**`debug.first_step=true` 且重放末步** —— 会在 step 0 硬失败。
遇到这个，不要调高 `max_abs_logprob_diff` 把门放宽，那等于把这个 gate
对所有 drift 源一起废掉。

另外仍然成立的前置：确认该 run 没开 versioned weight sync（worker 会直接
报错，这是设计行为）。收益随序列长度下降（seq 1024 约 12%，seq 4096 约 5%）。

### 5.4 复现

三个探针脚本是一次性验证产物，答案已记录在本节，故未入库。复现要点：
真实 checkpoint + 35 步链 + 存 `(x_i, prev_sample, logprob)` + 用未折叠策略
对同一 `prev_sample` 重打分 + `compute_logprob_mismatch_stats`。
**必须跑真实链** —— 用固定初始噪声 latent 喂所有 step 会在末步给出
非物理的 1.6e-01（σ 已趋零而 latent 还是纯噪声），第一次就是这么测错的。

## 7. 撤销记录

**决定（2026-08-17）：撤销。** 生产分辨率 0.6–4% 不抵三个 hazard + 278 行代码。

撤掉的东西：

- `vrl/nn/optimization/lora_merge.py`（机制，150 行）
- `LoraMergePass` 与 `ROLLOUT_PASSES` 里的注册（`passes.py`）
- `model.lora.merge_for_rollout` 配置键（`model_schema.py`）与
  `ModelBuild.lora` 的 extras 透传（`runtime.py`）
- `update_weights` 的同步后重折与 versioned-sync 拒绝（`worker.py`）
- `OnlineTrainer` 的折叠策略护栏（`trainer.py`）
- `tests/nn/optimization/test_lora_merge.py`（16 个测试）

**留下的东西**：本文的全部测量（§1 真实模型收益曲线、§6 parity 红线与
fp8 对照）。三个数字未来还会有人需要：

1. **rollout 折叠的真实收益是 0.6–4%**，不要再从合成基准外推。
2. **末步 σ→0 会把任何 drift 源放大 ~1000×**，而 `debug.first_step` 的 max 门
   与任何 drift 源都不兼容 —— 已发货的 fp8 量化在同一条链上比折叠差 5 倍，
   且连 mean 门都不过。
3. **全参替 LoRA 在训练段贵 26%**（见 `info/SPRINT_train_phase_gap_hunt.md` §2），
   所以杠杆表里的 P1.5 是 rollout-only 的收益。

**若将来重开**：门槛是低分辨率 / 小 latent 工作负载（240p 实测 7.7%）。
先在目标 shape 上量收益，超过 5% 再动工 —— 本文 §3 的落地形态可以照抄。

## 8. 相关

- 杠杆出处：`docs/sprints/done/SPRINT_gemm_utilization.md:195`
- elementwise 归因：`docs/sprints/info/SPRINT_cross_model_performance.md` §0
- 挂载 seam：`docs/sprints/SPRINT_plug_and_play_optimization_layer.md`
- 半覆盖反模式：`vrl/nn/optimization/passes.py` 的 `require_every_core_quantized`
- 父 program：`docs/sprints/done/SPRINT_train_phase_efficiency_program.md`
