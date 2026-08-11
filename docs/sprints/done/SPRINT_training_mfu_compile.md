# SPRINT: Training-side MFU — turn on torch.compile for the replay forward

状态：**done（2026-07-02）**。本 sprint 在 1×RTX 5090 可验证的范围已经完成：flag 翻转、
config fail-fast 和 Anima compiled/eager dry-run 均落地。Predict2.5 大形状的后续 compile 验收已移交
`docs/sprints/parked/SPRINT_video_context_parallel.md`；FSDP2+compile 在当前实现中是显式 non-goal，并由
preflight hard error 保护，不是本 sprint 的未闭 gate。本文数字只描述 SD3.5-medium 1024²、
no-checkpoint replay microbatch，不能外推到 video、gradient checkpointing 或 FSDP recipe。

> **当前路径修正（2026-07-18）**：family-first 重构后，rollout 与 replay 的 compile 顺序由
> `vrl/models/steps/denoise/build.py` 中的 `build_denoise_runtime_bundle` 和
> `assemble_replay_bundle` 统一拥有；family model 只提供 `torch_compile_transformer` 能力。
> 下文旧的 family-specific runtime 路径仅是当时实现记录，不再是 source of truth。

> **2026-07-02 复核要点**：文档 06-26 定稿后仓库变了三件事——① model 层 defaults 已把 sd3_5/predict2_2b/wan14b/flux/qwen 翻成 `enable:true`，§2 原表"无 block=默认 off"前提失效（**absent ≠ off，判断一律用 loader resolve，别看单文件**）；② compile×grad-ckpt 从"软建议"变**硬互斥**（fab16aca，`activation_checkpointing.py`），并顺带撞出 4 个 240p recipe 启动即炸（已修，§2.5）；③ SAC（`gradient_checkpointing: selective`）落地，成为显存紧 recipe 上 compile 的**竞争方案**（§3.2）。

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

共享 denoise builder 为 rollout 与 replay 分别构建 runtime，**两条路径都读取同一个
`build.torch_compile.enable`**：

```text
vrl/models/steps/denoise/build.py
  build_denoise_runtime_bundle  (rollout) -> model.torch_compile_transformer(...)
  assemble_replay_bundle        (train)   -> model.torch_compile_transformer(...)
```

registry descriptor 让 Wan、Anima、Predict2/2.5 等 denoise family 共用这条顺序。因此
`model.torch_compile.enable=true` 一次打开两条路，训练 replay 不需要第二个旋钮。已有
`compile_benchmark.py` 同时测 rollout 与 train path 的 latency、launches/step 和数值 parity；
本 sprint 复用它验收，不重造工具。

## 2. 现状盘点(online 训练 recipe 的 compile 态；2026-07-02 按 loader resolve 重列)

**判态规则（这次学到的）**：experiment 层"无 block"不等于 off——model 层 default 会穿透。唯一可靠的判态是 `load_config(...)` 后看 `model.torch_compile` 的 resolve 结果（配合 `actor.gradient_checkpointing`）。

| 态（resolve 后） | recipe | 处置 |
|---|---|---|
| **compile=True 已生效** | sd3_5 ocr / pickscore / geneval / crossnode_debug、wan_2_1 ocr / kling / physics / physics_i2v、wan_2_2 physics_i2v、cosmos_predict2 kling / v2w_reference（含 480p）、**anima ×2（07-02 翻，e2e dry-run PASS，§4.5）** | 已是目标态 |
| **compile=True（声明面向 ≥2 卡；单卡 OOM，live gate OPEN）** | cosmos_predict2_5 nft_kling / cross_node / ddp_2x1 / motion_physics（07-02 补齐对齐家族） | replay no-ckpt 单 5090 放不下（§4.5 实测）；这些 true 的验证等 2 卡 DDP 环境 |
| **off（FSDP 硬门）** | sd3_5 ocr_fsdp_2x1_fullparam、cosmos_predict2_5 fsdp_2x1 | **保持 off**：compile+FSDP2 unsound（`strategy.py:463` 硬 raise） |
| **off（ckpt-locked：显存必须 grad-ckpt）** | cosmos_predict2 四个 240p fullparam（07-02 修雷，见 §2.5）、droid 480p ×3、async_reward、flux validation ×4、qwen/wan smoke | 硬门强制 compile 与 ckpt 二选一；这组的对照方案是 **eager+selective**（§3.2），逐 recipe 实测再定 |
| **off（真剩余候选）** | sd3_5 async_debug（debug 故意 eager）、wan offline DPO ×2（另一 trainer 入口） | 少量个案,逐个判 |
| **不适用** | echo（`echo/runtime.py:6`："torch.compile intentionally not wired in Stage 1"）、AR 家族（janus/nextstep,不在本 sprint 范围） | 从清单移除 |

### 2.5 2026-07-02 落地记录

- **修雷 ×4**：`fullparam_8bit_240p` / `droid_target_240p` / `droid_full_target_240p` / `v2w_reference_fullparam_240p` 曾同时 resolve 出 compile=True（继承 model 层 `predict2_2b.yaml` 的 true）+ ckpt=True → 自 fab16aca 起启动即 ValueError。已各自显式 `torch_compile.enable=false` + 注释（与 droid 480p 同款处置：240p 32GB fullparam 摘不掉 ckpt，走 eager+ckpt，放弃 compile 的 ~1.25x）。
- **anima model 层翻 true**：`anima_preview3.yaml` 原注释自设的条件"等 anima experiment 落地"已满足（两个 aesthetic recipe 存在），parity 借 predict2.5 家族已绿（1.31x/1.33x、fp32 drift ~2e-6）。
- **motion_physics 补 compile-both 块**：它自称"nft_kling 的同骨架换 reward"，对齐 nft_kling 的 compile-both。
- **不变量下沉**：`require_compile_checkpointing_compatible`（`vrl/trainers/activation_checkpointing.py`）现在同时被 trainer 启动路径和 `validate_training_config`（config load）调用 → `test_load_all_experiments` 对每个 experiment 兜底，这类雷以后在测试里就炸，不用等 launch。附带单测（compile × true/full/selective 三态拒绝、off 放行）。

## 3. 三个约束（决定一个 recipe 能不能开）

1. **FSDP2**：`distributed.training.strategy=fsdp` + compile = unsound（`strategy.py:463` 已 raise）。FSDP recipe 一律保持 off,直到 inductor+fully_shard 对齐。
2. **grad-checkpointing（2026-07-02 起是硬互斥,不再是软建议）**：`activation_checkpointing.py` 的守卫在 trainer 启动和 config load 两层都 raise（compile traces `torch.utils.checkpoint` → InternalTorchDynamoError，full 和 selective/SAC 都中招）。所以每个显存紧 recipe 是实打实的**二选一**：
   - **compile + no-ckpt**：SD3.5 数据 74-79% MFU、省 ~20-25% 显存（§0）；
   - **eager + selective（SAC）**：SD3.5 数据 @batch4 恢复 full 的 ~2/3 recompute 税、no-ckpt OOM 的 batch 也能上（[[SPRINT_training_mfu_selective_checkpointing]]）。
   哪边赢取决于该 recipe 摘掉 ckpt 后 batch 能到多少——**必须 per-recipe 用 `backward_mfu_probe` 两条腿实测**，不要用 no-ckpt 的 1.2x 承诺 ckpt-locked recipe。注意一个待对齐的矛盾：probe `--compile` 仍会测 compiled×selective（探索 AOTAutograd 能否 compose），而守卫按"已测坏"直接禁——若未来 probe 证明 compose 可行，先解守卫再谈开关。
3. **LoRA / mode**：mode 必须 `default`;`reduce-overhead`/CUDA-graph 撞 PEFT LoRA + grad-ckpt([[project_torch_compile_wan]])。
4. **FP8 rollout recipe (added 2026-07-02)**: when compile is enabled with
   `precision.rollout.quantization.format=fp8`,
   `precision.rollout.quantization.recipe` must be `rowwise`. `blockwise` is
   hard-incompatible with compile because it graph-breaks into a 10x slowdown;
   the loader raises before execution (see [[SPRINT_rollout_optimization_layer]]
   §3.2). All live FP8 presets currently use the rowwise default.

## 4. 落地清单（最终状态 2026-07-02）

```text
✅ 1. flag 翻转主体（§2.5）+ anima e2e dry-run（§4.5,compiled vs eager 两臂,drift PASS）
✅ 2. ckpt-locked 组处置定案:单 32GB 视频 recipe(cosmos_predict2 240p fullparam、
     predict2_5 video 单卡)结构性必须 ckpt → compile 排除;它们的 full-vs-selective
     选择归 SPRINT_training_mfu_selective_checkpointing,与 compile 无关
↪  3. predict2_5 video compile-both：单卡 replay 放不下，已移交
     docs/sprints/parked/SPRINT_video_context_parallel.md 的 compile live gate
—  4. FSDP2+compile：当前明确不支持并 fail-fast；依赖栈未来改变时另立 sprint，不阻塞本文
□  5. 小尾巴(不 gate 本 sprint): wan offline DPO 入口单独判;videocon_physics
     sleep_offload 适配(§4.5 发现的 config 缺口,单列 follow-up)
```

## 4.5 e2e dry-run 结果（2026-07-02，1×5090，conda base env）

### anima aesthetic：compiled vs eager 两臂 A/B —— **PASS（带一个真发现）**

两臂同 seed=42、同 batch 几何（n=4/ppb=8/spc=4/gas=8，microbatch=4 样本；缩批原因见下），各 2 epoch：

| 臂 | ratio_abs_dev mean / max | mismatch_kl | reward (ep0→ep1) | reward_std | clip_fraction |
|---|---|---|---|---|---|
| **compiled**（新默认） | **5.7e-4 / 6.3e-3** | 5.7e-4 | 4.30→4.03 | 0.28-0.41 | **0.455** |
| eager + full-ckpt | 恰好 0 / 0 | 0 | 4.21→4.11 | 0.40-0.41 | 0 |

- **drift gate PASS**：compile-both 的 rollout↔replay 漂移 ~0.06%（max 0.6%）≪ 0.01 门限；reward 量级/方差/grad_norm 两臂一致。
- **真发现——tight-clip 交互**：anima `clip_ratio=1e-4` 比 compile 数值漂移（~5.7e-4）还小 → compiled 臂 45% 的 importance ratio 纯因编译数值出 clip 带（eager 臂 ratio 恒 1、clip=0）。训练 2 epoch 未见异常，但 **clip_ratio ≤ ~1e-3 的 recipe 在 compile-both 下 clip_fraction 会被编译噪声抬高**（PPO clip 对越带样本零梯度 = 变相降有效 batch）。anima 是否放宽 clip_ratio 是 recipe 调参问题，本 sprint 不动、只记录。
- **显存三角（32GB 上 anima 的真实处境）**：同 batch 下 eager no-ckpt replay 要 30.5GB → OOM；compiled 放得下（复现 §0 "compile 省 20-25% 显存"）；eager 想跑必须 full-ckpt。即 **anima 在 1×32GB 上 compile 不是锦上添花而是可跑性前提**（或退回 eager+ckpt）。
- **stock batch 几何（n=8/ppb=4/gas=4，microbatch=8）在干净 GPU 上复跑仍 OOM** → anima recipe 的默认几何本身超 1×32GB 预算（compiled 都放不下，eager 更不用说）。单 5090 跑 anima 的可行组合：`n=4/ppb=8/gas=8` + compile（本次验证 PASS），或 eager+full-ckpt。recipe 默认值面向更大显存，是否调整归 recipe 调参，不在本 sprint。
- **已知 flake（与 compile 无关）**：两臂训练全部完成后，teardown 阶段 `collector shutdown failed`（RayGenerationWorker 在清理时暴死，SIGSEGV/connection error）。eager 臂同样触发 → 非 compile 问题；epoch 指标与 checkpoint 完整。单列 infra 问题跟进。

### predict2_5 video（motion_physics / nft_kling）：单 5090 上结构性不可用，已移交 video-CP

按 480p_33f（历史真实 shape）与 512p_93f（recipe 默认）各试：**compiled replay（batch=1，14k/24.6k token，no-ckpt）驱动进程要 >31.3GB → 全部 OOM**（rollout + kling/videocon 打分全部跑通，mp4 正常生成，死在 replay 前向）。追查历史证据 `docs/runs/cosmos_predict25_nft_kling_480p33f_rbs16_20260620/launch.cmd`：那次是 **torchrun ddp_2x1 双卡**、且 `torch_compile.enable=false`——**这个 recipe 家族的目标环境从来是 ≥2 卡**，单卡从未跑通过 replay。结论：

- 单 5090 上 predict2_5 视频训练必须 grad-ckpt → 硬门排除 compile → 属于 §2 的 ckpt-locked 组（同 cosmos_predict2 240p fullparam）。nft 家族 config 里的 `enable:true` 是**面向多卡的声明**；compile-both 的 live drift gate（≤0.01）由 `docs/sprints/parked/SPRINT_video_context_parallel.md` 在目标形状跑通后重开，不再由本文持有。
- **顺带发现的真 config 缺口**：`videocon_physics`（mPLUG-Owl 级大模型）没有 sleep_offload wiring——加 `sleep_offload=true` 会在 cumem `pool.sleep()` 崩 `invalid argument`（需要 kling 当年同款的 `.to()` 适配，见 reward runtime 记忆）。单卡跑 motion_physics 在 compile 之外还被这一条挡住。单列 follow-up。
- anima 的 image 结论（drift 5.7e-4 ≪ 0.01、tight-clip 交互）覆盖的是同一条 compile-both 代码路径（同 cosmos DiT 家族、同 builder、同 trainer），视频侧剩下的主要是显存几何问题而非数值问题。

## 5. 第二杠杆:microbatch_size

训练 MFU 随 batch 还在涨(74→79%)= 小 microbatch 没完全喂饱 GEMM。`model.microbatch_size`（`schema.py:251`）是旋钮，但只能在同 family / 同分辨率 / 同 ckpt 策略下实测决定。compile 在 SD3.5 no-ckpt probe 里省了 ~20% 显存，所以它给了继续测 batch3/4 的空间，不是直接结论：

```text
compile 前 batch 2 = 26.8GB(逼近 32GB)
compile 后 batch 2 = 20.1GB  → 有头寸尝试 batch 3/4；是否 OOM/是否更快必须实测
```

注意:这是 grad-accum 的内层,改它不改有效 batch / 数值语义(memory fix,见 [[project_two_level_async]])。

## 6. 验收

- `compile_benchmark --family <f>`:rollout + train 两条 path 的 `max_abs_grad` / `max_rel_out` 在阈内(compile 数值忠实)——**这是开关前的安全门**。**工具边界**：`--family` 只有 sd3_5 / cosmos-predict2 / cosmos-predict2.5 / wan_2_1 四家；anima 蹭 predict2.5（同 Cosmos DiT，model yaml 注释已声明）；flux/qwen 若要开需先扩 family。
- `backward_mfu_probe --compile` before/after:目标 recipe 分辨率下 train fwd+bwd MFU 上升、peak GB 下降(复现 §0 量级)。**工具边界**：硬编码 `StableDiffusion3Pipeline`（:57），video recipe 不能"同分辨率跑"；旧 `video_dit_mfu_probe` 已退役，video 侧应使用短 real-run 计时。
- 短 RL dry-run:reward 曲线、`ratio_abs_dev`、TIS/RS 触发率与 eager 基线一致(parity 绿的 e2e 兜底)。
- 不开 compile 的 recipe(FSDP/ckpt-locked/validation)明确标注原因,不留"为什么这条没开"的悬念——ckpt-locked 组的原因注释已随 07-02 修雷写进各 yaml。
- ✅ compile×ckpt 不变量已下沉 config load 层(`validate_training_config`),`test_load_all_experiments` 全量兜底。

## 7. 非目标

- 不重做 rollout 侧 compile 规划；rollout kernel/MFU 轴已有单独 sprint 和 probe 记录（[[project_cosmos_compile_p2]]）。
- 不动 FSDP2+compile；`vrl/trainers/strategy.py` 的硬门是当前正确行为。依赖栈未来支持该组合时另立聚焦 sprint。
- 不做 compiled-backward 深水区优化(74-79% 之上的 slack 在梯度累加 bandwidth-bound 算子,要自定义 fused backward,收益递减)。
- 不重写 `compile_benchmark` / probe;复用现有 parity + MFU 工具。
- 不把 paged store / stepwise batching 拉回来(本轮实测证伪)。

## 8. 关键文件

- `vrl/models/steps/denoise/build.py` —— rollout/replay 共享 `torch_compile.enable` 与构建顺序
- `vrl/models/steps/denoise/base.py` —— family model 的 `torch_compile_transformer` 能力
- `vrl/config/schema.py`（`torch_compile`、`microbatch_size`、`gradient_checkpointing`）
- `vrl/trainers/strategy.py` —— compile+FSDP2 硬门
- `vrl/trainers/activation_checkpointing.py` —— compile×ckpt 硬门（`require_compile_checkpointing_compatible`，trainer 启动 + config load 双层）
- `vrl/scripts/perf/compile_benchmark.py` —— rollout+train parity + launch-bound profile（验收复用；仅 4 家 family）
- `vrl/scripts/perf/backward_mfu_probe.py`（`--compile`）—— train fwd+bwd MFU before/after（SD3.5 专用）
- 证据记忆：`project_rollout_bound_class_probe`、`project_torch_compile_wan`、`project_cosmos_compile_p2`
