# SPRINT: SANA 美学 GRPO 可信曲线

> 2026-07-13 full-parameter native-FP16 evaluation contract (v3): online fixed
> eval remains removed. After the registered full-parameter curve is complete, run
> `vrl.scripts.eval.sana_aesthetic_checkpoint_eval` over the training run. It
> reads that run's resolved config, held-out manifest, and complete numbered
> checkpoints without CLI overrides, then writes the provenance-bound canonical
> report at `sana_aesthetic_fullparam_native_fp16_eval/report.json`. Legacy BF16
> and LoRA evaluator protocols cannot be normalized into this protocol.

Status: **TRAINING COMPLETE; HELD-OUT QUALITY PENDING**. The replacement run in
`outputs/sana_aesthetic_fullparam_long/` completed all 300 updates and published
`checkpoint-final` with `global_step=300`, `uses_lora=false`, and
`run_verdict=success`. Its 300 metric rows cover epochs `0..299`, are finite,
and have no zero-gradient update. No training process or tmux session remains.
This completion proves the registered training job finished; it does not replace
the standalone held-out report required for the quality verdict. The invalid
first attempt is recorded in §3.5 and is not spliced into this run.

The previous BF16 v1 run was stopped and invalidated on 2026-07-12 after metric
row 231; `checkpoint-225` is its latest complete checkpoint. Do not resume that
curve. Its training-time 512px Flow-Euler eval
rose from `4.2352` to `5.7513`, but the corrected paired evaluation using the
official SANA 1024px DPM path measured aesthetic `5.7520 -> 5.4308` and
PickScore `0.8669 -> 0.7039`. The run therefore optimized a sampler/shape-bound
proxy rather than a transferable aesthetic improvement. It is also incomplete
under the preregistered 300-update contract. A replacement experiment must
start from a fresh baseline after its training/evaluation sampling contract is
made equivalent; metrics from this run must not be spliced into it.

The bounded five-update full-parameter pilot in
`online_grpo_aesthetic_fullparam.yaml` completed successfully on 2026-07-13.
It passed the capacity, native-precision, optimizer-state, strict-resume, and
rollout/replay parity gates recorded in §3.4. It remains a systems gate, not a
substitute for the 300-update held-out learning claim in this document.

## 0. 结论先行

这不是“跑到看起来上升为止”的探索，而是一次固定预算实验：SANA 1.6B 全参数、DrawBench 192 条训练 prompt、
与训练集精确去重后的 64 条 fixed-eval prompt、300 次 rollout update（每次 1 个 optimizer step）、
每 5 次 rollout update 保存 recovery checkpoint、每 25 次选择一个 checkpoint 做 held-out eval、
每 prompt 固定 2 个 standalone eval 样本。主结论只读
`sana_aesthetic_fullparam_native_fp16_eval/report.json` 中由逐样本 score 重算并校验的 summary；训练批次的 `reward_mean`
不参与 PASS/FAIL。

主跑 PASS 后才启动 50-update LoRA+fp8 rollout smoke。它验证 master-free fp8 adapter 同步进入真实训练循环；
**它不能单独证明“32GB 训练 17B”**。17B fp16/bf16 replay 权重自身约 34GB，训练侧仍装不下 32GB。
本 sprint 能解锁的是“17B master-free fp8 rollout 的构建与同步前置条件”；真正的 17B 单卡训练还需要
训练侧量化、参数 offload 或分片，必须另行真机验收。

## 1. 为什么选这条曲线

- SANA rollout 已完成真权重生成与 replay parity 验证，但没有跑过短 GRPO 曲线；旧 landing sprint 明确保留
  了这个空白。
- aesthetic reward 是 CLIP ViT-L/14 image embedding 加本地 MLP 头，没有外部 judge 服务，适合复现。
- SANA base transformer 与 denoiser forward 都必须走 pinned checkpoint 的
  native FP16/no-outer-autocast 路径；Gemma prompt encoder 独立使用 BF16，VAE、
  timestep、CFG combine、scheduler/log-prob math 使用 FP32。旧 run 的 BF16
  outer autocast 会改变 linear attention 并产生 color-block artifact，不能
  作为兼容表示被 evaluator 改写成 FP16。
- 仓库既有算法测量显示，8–12 次 rollout update 只能证明管线工作，不能证明 learning；可信 learning 需要约
  200–300 次 rollout update。这里预注册 300，不在中途因曲线形状延长或缩短。

## 2. 固定资产与运行配置

- 训练配置：`vrl/config/presets/experiment/diffusion/sana/online_grpo_aesthetic_fullparam_long.yaml`
- 训练 prompt：`datasets/drawbench/train_192.txt`
- fixed eval：`datasets/drawbench/eval_64.txt`
  - 从 `datasets/drawbench/test.txt` 保序去重；
  - 排除与 `train_192.txt` 精确重复项；
  - 64/64 与训练集不重叠；开跑后不得替换。
- 目标：`aesthetic: 1.0`。
- 只观测不优化：`pickscore: 0.0`. The archived active-run config predates the
  later PickScore CPU placement; its startup log shows both rewards ran on CUDA.
  The evaluator's default `--device=auto` reproduces that CUDA placement on this
  machine, while any explicit execution-device choice is recorded in the report.
  PickScore never enters the advantage. The current preset's CPU placement is
  execution topology for future runs, not a retroactive rewrite of this run.
- rollout update：300；`ppo_epochs: 1`，因此完整运行是 300 次 optimizer step；全部 396 个
  transformer 参数通过 checkpointed FP32 master 更新；训练 rollout 是 512px、10-step Flow-SDE、CFG 4.5。
- 旧 BF16 run 的 rollout/replay sample chunk 固定为 8/8；这是历史测量，不是
  native FP16 full-parameter 路径的要求。修正 TF32 backend 后，native 路径的
  1/1 replay shape 已实测 bit-exact；full-parameter pilot 使用 1 prompt × 8
  samples、rollout/replay chunk 都为 1 以控制显存。
- standalone fixed eval：在读取任何训练 checkpoint 之前，用 pinned SANA base snapshot 生成
  `epoch=-1`；随后按顺序 strict-load `checkpoint-25` 到 `checkpoint-300`。每个 prompt group 使用新建的
  official DPM-Solver++ scheduler、1024px、20 steps、CFG 4.5。训练 SDE 只负责探索，不能成为质量代理。
  64 prompts × 2 samples。每个 prompt 的两个样本保持在同一个 batched generator stream，group seed 为
  `20260710 + prompt_index * 2`, implementing the registered batched seed
  protocol. The evaluator rejects checkpoint gaps and records hashes for the
  config, manifests, checkpoints, and scored artifacts, plus explicit seed and
  reward identities.
- 输出：`outputs/sana_aesthetic_fullparam_long/`；recovery checkpoint 每 5 update，
  held-out curve 仍固定读取 `checkpoint-25,50,...,300`。

### Standalone report contract

- Full-parameter protocol schema version is `3`; the canonical path is
  `sana_aesthetic_fullparam_native_fp16_eval/report.json`. Older reports retain
  their raw provenance but are not accepted by the v3 reader.
- Provenance binds the training metrics, resolved config and canonical protocol
  digest, train/eval manifests, supervisor log, the four revisions pinned in config,
  packaged aesthetic MLP hash, sampling, batched seed grid, effective reward
  identities, checkpoint SHA-256 values, and per-sample JSONL/image SHA-256
  values.
- Every curve point contains `epoch`, `sample_count`, `eval_reward_stderr`,
  `r_aesthetic`, and `r_pickscore`. The reader recomputes these summaries from
  per-sample scores before the verdict consumes them.
- New runs do not accept an external report or CSV path. The verdict reads a
  historical `eval_metrics.csv` only when the archived `resolved_config.yaml`
  explicitly contains `trainer.eval.enabled: true`.
- Model, aesthetic CLIP, PickScore processor, and PickScore model revisions are
  explicit config fields consumed by their respective `from_pretrained` calls.
  Cache-hit logs are therefore not parsed to infer dependency state; the resolved
  config is the source of truth and its semantic digest is frozen before launch.

以下是已失效 BF16 v1 run 的历史启动命令，**不得执行或恢复**：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
vrl-train --config experiment/diffusion/sana/online_grpo_aesthetic
```

该输出目录的 checkpoint 只保留为历史证据，不能作为 full-parameter v3 的 resume
source。v3 必须从 fresh base 和 fresh baseline 开始。

v3 full-parameter 主跑只使用以下 canonical preset；supervisor 在输出目录为空时
从 pinned base 启动，后续中断只恢复该目录最新的完整 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u -m vrl.scripts.supervise \
  --config experiment/diffusion/sana/online_grpo_aesthetic_fullparam_long
```

## 3. 主跑 PASS/FAIL（baseline 之前冻结）

自动判定命令：

```bash
python -m vrl.scripts.eval.sana_aesthetic_checkpoint_eval \
  --run-dir outputs/sana_aesthetic_trustworthy_curve

python -m vrl.scripts.eval.sana_aesthetic_curve_verdict \
  --run-dir outputs/sana_aesthetic_trustworthy_curve \
  --qualitative-audit pass \
  --out outputs/sana_aesthetic_trustworthy_curve/verdict.json
```

`--qualitative-audit pass` 只能在 §4 的盲审完成并通过后填写；默认 `pending` 必须返回 FAIL。

### 3.1 全部满足才 PASS

1. **完整性**：恰好 300 条训练 metric；baseline 恰好一条；至少 12 条 post-training fixed-eval 点；
   所有读取数值 finite。
2. **主统计量**：终点定义为最后 3 个 fixed-eval 点的 `r_aesthetic` 均值。相对 baseline 同时满足：
   - 绝对增益 `>= 0.10`；
   - `gain / sqrt(stderr_baseline² + stderr_endpoint_mean²) > 2.0`。
   新 standalone report 同时保留逐样本分数，但本 run 的冻结判据仍沿用预注册的 pooled standard error；
   不在看到结果后改成 paired test 或更换显著性标准。
3. **方向**：所有 post-training fixed-eval 点对 epoch 的 OLS slope `> 0`。
4. **prompt-aware 防线**：终点 3 点平均 PickScore 不低于 baseline 的 98%。
5. **梯度/策略活性**：所有 update 都实际训练 prompt；至少一个 `grad_norm > 0`。本协议只有一次 PPO pass，
   optimizer 前 ratio 应为 1，因此所有 `active_clip_fraction`、`pre_update_clip_fraction` 和
   `pre_update_active_clip_fraction` 必须为 0；非零表示 backend drift，不是健康的策略移动。
6. **多样性**：末 32 update 的 `reward_std` 中位数 `> 1e-4`，且不低于最初 32 update 中位数的 25%。
7. **训推一致**：旧 v1 冻结门限 `0.01` 比 Flow-GRPO 的 `clip_ratio=1e-4`
   宽两个数量级，因此只保留为“旧 run 为什么曾被错误放行”的历史判据。
   Full-parameter v3 在首次 optimizer step 前要求
   `pre_update_logprob_abs_diff_max <= 1e-6`，并且
   `pre_update_clip_fraction == 0`、`pre_update_active_clip_fraction == 0`。
8. **定性审计**：§4 PASS。

### 3.2 任一项触发即 FAIL

- 到 300 update 时主统计量平、负或不显著；不得加跑找显著。
- 非有限 loss/reward/gradient、零 prompt update、长期零梯度、多样性坍缩、parity 越界。
- aesthetic 上升但 PickScore 超过允许回退，或盲审发现系统性 reward hacking。
- 运行中改变判据、seed、eval prompt、LR、精度或预算。若因实现 bug 修代码，旧 run 作废；从新 baseline
  全量重跑并在本文件记录原因，不能拼接修复前后的曲线。

### 3.3 2026-07-11 旧 BF16 run 的真机记录（非 native 启动证据）

同一 RTX 5090、同一 fresh base、同一 prompt/seed 上完成三次因果 smoke：

- rollout=16 / replay=1：VAE decode 触发一次 OOM 后拆成 8+8；首样本 t0 diff 仅 `7.3e-5`，但全体
  sample/timestep 的 pre-update max 为 `0.014056`，pre-update ratio 越界率 `0.7986`。该结果作废。
- rollout=1 / replay=1：全体 pre-update diff、ratio 越界率、active-clip 率均为精确 0，且 0 OOM，证明
  根因是 batch-shape 数值差异，不是 KL 或 reward 梯度。
- rollout=8 / replay=8、完整 4 PPO：0 OOM；18 个首 pass replay evaluation 的全量 max diff 为 0；
  后续 PPO pass 才产生 `clip_fraction=0.6181` / `active_clip_fraction=0.3264`；所有指标 finite，
  `grad_norm=0.423678`，LoRA SHA 改变，单 prompt fixed eval `4.2066 → 4.2536`。短 eval 只证明管线，
  不计入 §3.1 的 learning 结论。

当时据此把旧 BF16 run 门槛固定为对称 8/8，并在 optimizer 前对首 pass
全量 hard-fail；该 batch-shape 结论随 BF16 protocol 一并失效。全量 parity
而非单点 t0 probe 的原则仍然保留。

2026-07-13 native-FP16 因果复核取代了上述结论：rollout 与 replay 都关闭
TF32 时，独立模型的 10/10 步 noise/log-prob 逐位一致；只给 replay 开 TF32
即可重现 `1e-4` clip band 越界。首个 full-param one-step run 虽能 backward，
但因 TF32 分叉与缺少 FP32 master 而无效。replacement 必须同时通过显式 IEEE
role policy、FP32 master/GradScaler、strict resume 与 `1e-6` parity gate。

### 3.4 2026-07-13 native-FP16 full-parameter pilot

The canonical five-update run is `outputs/sana_aesthetic_fullparam/`. It uses
the checkpoint-native FP16 transformer without outer autocast, BF16 Gemma,
FP32 VAE/CFG/timestep/scheduler/log-probability math, and
`precision.float32_precision=ieee` on
both rollout and replay. `model.use_lora=false` makes all 396 transformer
parameters trainable. AdamW8bit compresses optimizer moments only; the trainer
owns checkpointed FP32 master parameters and a GradScaler.

The capacity/numerics gate passed on one RTX 5090:

- all five updates completed and published `checkpoint-1` through
  `checkpoint-5` plus `checkpoint-final`; the final checkpoint is 12,887,934,871
  bytes and records `global_step=5`, `uses_lora=false`;
- every metric was finite, all five gradients were non-zero
  (`0.124712 <= grad_norm <= 0.265342`), and the mean sampled training reward was
  `4.86664` across different prompts;
- every update reported exact-zero pre-update log-probability, ratio, clip, and
  active-clip mismatch; the first-step full check observed
  `max_abs_diff=0.0` against the `1e-6` hard limit;
- generation peaked at 9,907 MiB per one-sample chunk and no OOM occurred;
- a separate real checkpoint continuation restored model weights, AdamW8bit
  moments, FP32 residuals, GradScaler, RNG, and progress from step 1, then
  completed step 2 with exact-zero pre-update mismatch;
- after five updates, all 396 FP32 master tensors changed and 96.63490% of their
  elements differed from the FP16 initialization. Only 13.53824% had crossed an
  FP16 representable boundary, which is why the master copy is required.
  Rounding every master back to FP16 matched the published model state exactly.
- `vrl.scripts.eval.sana_checkpoint_compare` then generated the base before
  reading the training checkpoint and the current image after strict restore,
  using fresh official DPM-Solver++ schedulers and reset seed `20260712`. Both
  1024px images are coherent studio photographs rather than color blocks. On
  this single inference-integrity probe, aesthetic moved `5.90665 -> 5.82033`
  and PickScore moved `0.90663 -> 0.89834`; this is not a quality PASS and must
  not be generalized from one prompt.

These facts prove that full-parameter native-precision optimization is real and
resumable. They do not prove aesthetic improvement: training rewards came from
different sampled prompts, and five updates are not a held-out comparison. The
official DPM++ 1024 base/current image check is a separate inference-integrity
gate, and a longer curve still requires the preregistered held-out evaluation.

### 3.5 2026-07-13 first long-run launch invalidation

The first 300-update launch completed 11 finite updates but failed before its
first checkpoint. Ray killed `RayGenerationWorker` when node memory reached
88.39/91.88 GB; the worker alone held 43.23 GB. This was not a trajectory,
reward, model-reload, object-store, or GPU leak. Strict on-policy weight sync
incorrectly selected `TrainableStateSlots` solely because the diffusion model
advertised the capability. Eight retained full-parameter FP16 snapshots cost
23.91 GiB, and the colocated rollout's CuMem parking backup added 9.29 GiB.

The generic runtime now separates capability from selected mode:

- strict on-policy always direct-loads the latest state after its draining
  barrier and retains no historical payload;
- continuous LoRA keeps versioned slots for real in-flight ownership;
- continuous full-parameter sync fails closed to the draining path until a
  byte-budget gate proves retained snapshots fit host RAM.

The invalid run had no complete checkpoint and its supervisor retry correctly
started fresh, but that retry overwrote the partial metrics CSV. Its answer is
recorded here; the incomplete output is deleted as a one-shot failure artifact.
The replacement run starts from the pinned base. Recovery checkpoints now
publish every five updates, while the scientific evaluator remains fixed to
`checkpoint-25,50,...,300`; recovery IO cadence is not evaluation cadence.

### 3.6 Repository-wide effect of the hardened gate (2026-07-11)

- `debug.first_step` changed from warning when the first chunk's mean exceeded
  `0.01` to failing before the optimizer when the first pass's full maximum
  exceeds `debug.max_abs_logprob_diff` (default `0.01`). Roughly 25 non-SANA
  presets enable this gate. Bit-exact families are unaffected; a family with a
  measured benign maximum above the threshold must justify a family-specific
  threshold instead of disabling the gate.
- Older SD3 flow-GRPO FP32 recipes combined FP32 training with FP16 frozen
  components but accidentally inherited FP16 autocast from `prompt_embeds`.
  Current `precision.training.dtype=fp32` disables autocast and runs a real FP32
  forward. Parity improves, but old and new FP32 curves are not directly
  comparable.

### 3.7 Native FP32 rejection gate (2026-07-13)

The pinned `Efficient-Large-Model/Sana_1600M_1024px_diffusers` transformer does
not have a valid FP32 execution path. Its two default shards genuinely store
F32 tensors, and Diffusers selects their index before either monolithic FP16
file; the failure is therefore not an FP16-file-widening loader bug. The
published F32 tensors are distribution weights that must be downcast for this
checkpoint's recommended FP16 transformer execution.

The pre-launch probes isolated the failure:

- strict F32 loading used all 396 F32 tensors with zero missing, unexpected, or
  mismatched keys;
- on the same prompt, seed, embeddings, initial latent, and official DPM++
  schedule, block statistics stayed close through block 18, then the final F32
  block's hidden-state standard deviation jumped from `75.58` to `685.61`;
- the F32 noise prediction had standard deviation `0.2451` versus `1.1216` for
  native FP16, and the official 1024px decode produced psychedelic color-block
  artifacts;
- retaining F32 parameters while wrapping the pipeline in FP16 autocast was not
  a valid compromise: the denoise path became non-finite and decoded to a black
  image.

Therefore the canonical replacement uses FP16 transformer parameters. BF16 and
FP32 were rejected as trustworthy-curve candidates by the measured probe, while
the public config remains explicitly overridable for new experiments.
Full-parameter optimization continues to use the already validated design:
visible FP16 trainable parameters, checkpointed FP32 master parameters,
GradScaler, and AdamW8bit moments. This is full-parameter training with
full-precision update residuals, but it is not and must not be described as an
FP32 transformer forward. No FP32 long run was
launched. The failed canary and inference outputs are one-shot evidence; this
decision record and the regression tests are their retained result.

## 4. Reward-hacking 预案

根因是 aesthetic scorer 完全不读取 prompt；它只能证明图像落在审美头偏好的 embedding 区域，不能证明
prompt adherence。防线在运行前固定为：

1. **PickScore 零权重观测**：同一 fixed prompt/seed 网格，终点最多相对回退 2%。
2. **固定样本盲审**：baseline 与终点使用同一 64 prompt/seed grid；打乱 A/B 标签后检查：
   - prompt/object/attribute/relationship mismatch；
   - 高频纹理、过锐化、过饱和、边缘 halo；
   - 水印/伪文字/签名式捷径；
   - 单一构图、单一色板或近重复图导致的模式坍缩；
   - 显著 NSFW 或安全退化。
3. **裁决**：任一类别在终点出现系统性退化即 FAIL。不能在同一 run 上临时加 reward 权重“补救”；治疗
   是下一个带新 baseline 的实验，不回写本次结论。

## 5. 主跑后的 LoRA+fp8 master-free 50-update smoke

只有 §3 主跑 PASS 后才启动。固定覆盖：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
vrl-train --config experiment/diffusion/sana/online_grpo_aesthetic \
  precision.rollout.quantization.format=fp8 \
  precision.rollout.quantization.recipe=rowwise \
  precision.rollout.prompt_encoders.dtype=bf16 \
  trainer.total_epochs=50 trainer.save_freq=10 \
  trainer.output_dir=outputs/sana_aesthetic_fp8_master_free_smoke
```

smoke PASS 必须同时满足：

- 日志显示 fp8 swap 命中大 linear，且 base-precision master 释放量大于 0；
- 初始同步及之后每次 LoRA adapter 同步成功，不出现 base-weight load 到 master-free linear；
- 恰好完成 50 update，写出可读取的 `checkpoint-final`；若进程中断，只允许从最新完整 checkpoint 恢复，
  恢复后的 policy version 与 metrics 必须连续；
- 所有 mismatch 指标 finite；`ratio_abs_dev_mean < 0.01`，`ratio_abs_dev_max < 0.10`；
- `tis_clip_fraction < 0.05` 且 `rs_seq_masked_fraction < 0.05`；
- reward/样本不发生 catastrophic collapse。50 update 不要求再次达到主跑的 `>2σ` learning gate。

smoke 之前已补两个根因修复：

1. master-free `Fp8Linear` 在 adapter-only partial load 时不再从不存在的 master 重量化；真正含 base
   `weight` 的 payload 仍 hard-fail。
2. quantized LoRA rollout 改为 CPU 上 attach PEFT → swap fp8 → drop master → compact policy 搬 GPU，避免
   17B bf16 rollout 在量化前先撞 32GB 峰值。

## 6. 17B 边界（不可偷换）

smoke 通过后允许声称：

> master-free fp8 LoRA rollout 的真实 50-update adapter 同步路径已验证；17B rollout 的 CPU 量化后搬运
> 路径具备进入容量实测的条件。

不允许声称：

> 32GB 已能训练 17B。

原因不是措辞保守，而是训练 replay 与 rollout 是两份不同精度职责：fp8 只作用于 rollout GEMM；trainer
replay 仍需 fp16/bf16 可导权重。17B 权重本身已超过 32GB。后者只有在 17B 真模型完成训练侧 offload/
quantized-backward/FSDP 方案并跑过至少一个 backward 后才解锁。

## 7. 架构卫生与非目标

应改变且已改变：

- SANA preset 为 training/rollout 都显式选择 native FP16、IEEE FP32 backend
  和 `outer_autocast: false`；Gemma 使用 BF16，VAE/CFG/timestep 与受保护的
  scheduler/log-prob math 使用 FP32。
- 每个进程只接收对应的 `RolePrecision(dtype="fp16",
  float32_precision="ieee", outer_autocast=False)`。registry 不再复制 checkpoint
  precision 限制；显式 YAML override 被视为实验选择，可信曲线仍由 canonical
  config/protocol gate 保持严格。
- Low-precision trainables 自动派生 checkpointed FP32 master；AdamW8bit 只
  压缩 optimizer moments，GradScaler 保护 FP16 backward，成功 step 后把
  visible policy 发布回 FP16。SANA full-param pilot 保持 EMA disabled，因为
  low-precision EMA shadow 尚不提供 master 精度。
- `clip_fraction` 与真正选中 clipped surrogate 的 `active_clip_fraction` 分开；raw KL 与
  `weighted_kl_loss` 分开，避免再次误读。
- policy update 与 rollout/replay mismatch 在内部按职责分组；pass-zero snapshot 由 trainer 在 evaluator
  边界统一计算，不再依赖某个算法自行填 parity 字段。`metrics.csv` 继续保持既有扁平列，历史 verdict
  与曲线脚本无需迁移。
- The public precision policy separates ordinary role dtypes from selective
  rollout quantization. `PrecisionPolicy` derives both, and runtime projection
  does not maintain a second dtype decision tree.
- fixed eval 已移出 training orchestration；训练只产出完整 checkpoint，held-out prompt/seed 协议由独立
  checkpoint evaluation 作业执行。该作业只有 run-dir/device execution inputs，不允许用 CLI 替换
  config、manifest、checkpoint、sampling、seed 或 reward；verdict 会重新校验 report provenance 与 run-dir。
- canonical DrawBench preset 接入唯一 fixed-eval manifest；旧 40/64 训练重叠的 eval 列表改为真正 disjoint。
- master-free partial-load 和 LoRA+fp8 CPU 构建顺序增加行为回归测试。
- 主跑使用机械 verdict，不允许口头重解释阈值。

保持不变：

- `data.eval_manifest` 与 standalone checkpoint evaluator 保持独立边界；不在 rollout schedule 中恢复
  training-rank fixed-eval 旁路。
- `drop_quantized_masters` 保持共享抽象：它按 `QuantizedLinear` 协议释放所有量化方案的 master，避免
  为 fp8/fp4 分别维护会漂移的薄别名。
- Keep the evaluator's module-level ALL_CAPS values. They are persisted
  scientific-protocol identities: schema/file names, seed/sample grid, canonical
  config and manifest digests, model revisions, and asset hashes. They are not
  tunable defaults or a duplicated prompt vocabulary; prompt text remains in the
  manifests. The frozen canonical digest must not be derived from the live config,
  because that would make its drift gate tautological.
- Keep `_resolve_probe_model_build` as the direct-probe CLI-to-runtime adapter.
  Its single caller is intentional: routing through the selected registry
  entry's `resolve_model_build` method prevents the probe from owning another
  dtype policy.
- Keep the evaluator's small report serializers and validators in the same
  SANA-specific module. Writer and reader share them as one protocol boundary;
  splitting them into thin facade files would add navigation without removing
  complexity.
- `torch_compile` 继续关闭；SANA 首条可信曲线不同时引入 compile 变量。

Non-goals: chasing SOTA aesthetic scores, changing GRPO mathematics, adding
PickScore to the objective, generalizing a cross-family evaluation framework,
adding evaluator state to the trainer/Ray contract, or replacing a learning
conclusion with a short smoke run.

Remaining convergence items (non-blocking):

- `precision.rollout.prompt_encoders.dtype` is still a silent no-op for
  non-diffusers families. Wan, Cosmos Predict2.5, and AR loaders keep family-local
  FP32 VAE/T5 choices. Genuine family invariants should move into registry
  descriptors; other families should consume `build.rollout.prompt_encoder_dtype`.

## 参考

- `docs/sprints/done/SPRINT_sana_t2i.md`
- `docs/sprints/info/SPRINT_flux_algo_validation_curves.md`
- `docs/sprints/done/SPRINT_cosmos_kling_fixed_eval_signal.md`
- `docs/sprints/done/SPRINT_fp8_rollout_gemm_kernel.md`
- `vrl/rewards/models/aesthetic.py`
- `vrl/rewards/functions/registry.py`
- `vrl/models/diffusion/build.py`
- `vrl/models/diffusion/common/lora.py`
- `vrl/models/loader.py`
- `vrl/nn/quantization/fp8.py`
- `vrl/scripts/eval/sana_aesthetic_curve_verdict.py`
- `vrl/scripts/eval/sana_aesthetic_checkpoint_eval.py`
- https://huggingface.co/docs/diffusers/v0.32.2/en/api/pipelines/sana
- https://github.com/huggingface/diffusers/issues/10241
