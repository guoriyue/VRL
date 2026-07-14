# SPRINT: SANA 美学 GRPO 可信曲线

> 2026-07-12 evaluation contract: online fixed eval remains removed. After the
> registered checkpoint curve is complete, run
> `vrl.scripts.eval.sana_aesthetic_checkpoint_eval` over the training run. It
> reads that run's resolved config, held-out manifest, and complete numbered
> checkpoints without CLI overrides, then writes the provenance-bound canonical
> report at `sana_aesthetic_eval/report.json`. Historical `eval_metrics.csv`
> files remain readable but are not produced for new runs.

Status: **STOPPED AND INVALIDATED (2026-07-12 13:34 PDT)**. The operator
stopped the run after metric row 231; `checkpoint-225` is the latest complete
checkpoint. Do not resume this curve. The training-time 512px Flow-Euler eval
rose from `4.2352` to `5.7513`, but the corrected paired evaluation using the
official SANA 1024px DPM path measured aesthetic `5.7520 -> 5.4308` and
PickScore `0.8669 -> 0.7039`. The run therefore optimized a sampler/shape-bound
proxy rather than a transferable aesthetic improvement. It is also incomplete
under the preregistered 300-update contract. A replacement experiment must
start from a fresh baseline after its training/evaluation sampling contract is
made equivalent; metrics from this run must not be spliced into it.

## 0. 结论先行

这不是“跑到看起来上升为止”的探索，而是一次固定预算实验：SANA 1.6B、DrawBench 192 条训练 prompt、
与训练集精确去重后的 64 条 fixed-eval prompt、300 次 rollout update（每次 4 个 PPO optimizer step）、
每 25 次 rollout update 保存 checkpoint、每 prompt 固定 2 个 standalone eval 样本。主结论只读
`sana_aesthetic_eval/report.json` 中由逐样本 score 重算并校验的 summary；训练批次的 `reward_mean`
不参与 PASS/FAIL。

主跑 PASS 后才启动 50-update LoRA+fp8 rollout smoke。它验证 master-free fp8 adapter 同步进入真实训练循环；
**它不能单独证明“32GB 训练 17B”**。17B fp16/bf16 replay 权重自身约 34GB，训练侧仍装不下 32GB。
本 sprint 能解锁的是“17B master-free fp8 rollout 的构建与同步前置条件”；真正的 17B 单卡训练还需要
训练侧量化、参数 offload 或分片，必须另行真机验收。

## 1. 为什么选这条曲线

- SANA rollout 已完成真权重生成与 replay parity 验证，但没有跑过短 GRPO 曲线；旧 landing sprint 明确保留
  了这个空白。
- aesthetic reward 是 CLIP ViT-L/14 image embedding 加本地 MLP 头，没有外部 judge 服务，适合复现。
- SANA base transformer 参数必须保持 fp16；bf16 参数会让 linear attention 产生 confetti artifact。
  前向计算则固定使用 bf16 autocast：真实 fp16 SDE 尝试已产生非有限 log-prob，而 bf16 的指数范围可避免
  activation overflow. The family build descriptor fixes the former; public
  `precision.training.dtype` and `precision.rollout.dtype` express the latter.
  The evaluator normalizes this active run's archived scalar `bf16` only in its
  in-memory build copy, leaving the archived config and hash unchanged.
- 仓库既有算法测量显示，8–12 次 rollout update 只能证明管线工作，不能证明 learning；可信 learning 需要约
  200–300 次 rollout update。这里预注册 300，不在中途因曲线形状延长或缩短。

## 2. 固定资产与运行配置

- 训练配置：`vrl/config/presets/experiment/diffusion/sana/online_grpo_aesthetic.yaml`
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
- rollout update：300；每次 `ppo_epochs: 4`，因此完整运行是 1200 次 optimizer step；LoRA r16/alpha32；
  10-step denoise；CFG 4.5。
- rollout 与 replay 的 sample chunk 都固定为 8。它不是吞吐偏好：SANA bf16 kernel 对 batch shape 敏感，
  两侧不一致会在 optimizer 前制造虚假的 PPO ratio 漂移。
- standalone fixed eval：先 strict-load `checkpoint-25` 的完整 transformer state，再用同一 state 的
  adapter-off base 作为 `epoch=-1`；随后启用该 adapter 得到 checkpoint-25 点，再按顺序加载后续 checkpoint。
  Baseline 因而由本 run 的首 checkpoint hash 绑定，不依赖浮动 Hub base。
  64 prompts × 2 samples。每个 prompt 的两个样本保持在同一个 batched generator stream，group seed 为
  `20260710 + prompt_index * 2`, implementing the registered batched seed
  protocol. The evaluator rejects checkpoint gaps and records hashes for the
  config, manifests, checkpoints, and scored artifacts, plus explicit seed and
  reward identities.
- 输出：`outputs/sana_aesthetic_trustworthy_curve/`；checkpoint 每 25 update。

### Standalone report contract

- Schema version is `1`; the canonical path is
  `sana_aesthetic_eval/report.json`.
- Provenance binds the training metrics, resolved config and canonical protocol
  digest, train/eval manifests, supervisor log and four observed Hub revisions,
  packaged aesthetic MLP hash, sampling, batched seed grid, effective reward
  identities, checkpoint SHA-256 values, and per-sample JSONL/image SHA-256
  values.
- Every curve point contains `epoch`, `sample_count`, `eval_reward_stderr`,
  `r_aesthetic`, and `r_pickscore`. The reader recomputes these summaries from
  per-sample scores before the verdict consumes them.
- New runs do not accept an external report or CSV path. The verdict reads a
  historical `eval_metrics.csv` only when the archived `resolved_config.yaml`
  explicitly contains `trainer.eval.enabled: true`.
- Legacy evidence has a hard limit: the evaluator can prove that today's
  manifests match their preregistered hashes and that `supervisor.log` observed
  the four registered Hub revisions at startup. It cannot prove that a manifest
  was never temporarily replaced and restored during training. A future run
  without machine-readable revision evidence is rejected; the evaluator does
  not infer revisions by walking the HF cache layout.

启动命令：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
vrl-train --config experiment/diffusion/sana/online_grpo_aesthetic
```

中断后只能从该输出目录最新的完整 `checkpoint-*` 恢复。恢复不得修改数据、seed、精度、LR、
update 总数、eval 频率或下面阈值。

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
5. **梯度/策略活性**：所有 update 都实际训练 prompt；至少一个 `grad_norm > 0`；warm-up 后至少 25% update
   的 `active_clip_fraction > 0`。`clip_fraction` 只表示 ratio 越界，不等于 surrogate 真被截断。
6. **多样性**：末 32 update 的 `reward_std` 中位数 `> 1e-4`，且不低于最初 32 update 中位数的 25%。
7. **训推一致**：只看 optimizer step 之前的 `pre_update_logprob_abs_diff_max <= 0.01`；聚合的
   `logprob_abs_diff_max` 含后续 PPO pass 的正常策略漂移，不能拿来判 backend parity。first-step 守卫必须无报错。
8. **定性审计**：§4 PASS。

### 3.2 任一项触发即 FAIL

- 到 300 update 时主统计量平、负或不显著；不得加跑找显著。
- 非有限 loss/reward/gradient、零 prompt update、长期零梯度、多样性坍缩、parity 越界。
- aesthetic 上升但 PickScore 超过允许回退，或盲审发现系统性 reward hacking。
- 运行中改变判据、seed、eval prompt、LR、精度或预算。若因实现 bug 修代码，旧 run 作废；从新 baseline
  全量重跑并在本文件记录原因，不能拼接修复前后的曲线。

### 3.3 2026-07-11 启动前真机门槛

同一 RTX 5090、同一 fresh base、同一 prompt/seed 上完成三次因果 smoke：

- rollout=16 / replay=1：VAE decode 触发一次 OOM 后拆成 8+8；首样本 t0 diff 仅 `7.3e-5`，但全体
  sample/timestep 的 pre-update max 为 `0.014056`，pre-update ratio 越界率 `0.7986`。该结果作废。
- rollout=1 / replay=1：全体 pre-update diff、ratio 越界率、active-clip 率均为精确 0，且 0 OOM，证明
  根因是 batch-shape 数值差异，不是 KL 或 reward 梯度。
- rollout=8 / replay=8、完整 4 PPO：0 OOM；18 个首 pass replay evaluation 的全量 max diff 为 0；
  后续 PPO pass 才产生 `clip_fraction=0.6181` / `active_clip_fraction=0.3264`；所有指标 finite，
  `grad_norm=0.423678`，LoRA SHA 改变，单 prompt fixed eval `4.2066 → 4.2536`。短 eval 只证明管线，
  不计入 §3.1 的 learning 结论。

因此启动门槛固定为对称 8/8（三个 SANA preset 一并对齐，`_long`/`pickscore_validation` 原 16/1 形状
按本节实测会触发 0.014 漂移直接撞 hard-fail），并在 optimizer 前对首 pass 的所有 sample/timestep 做
hard-fail；单点 t0 probe 只保留为便于定位的详细诊断，不能再替代全量门槛。

### 3.4 Repository-wide effect of the hardened gate (2026-07-11)

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
  precision.training.dtype=bf16 \
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

- SANA family build descriptor 固定 fp16 base transformer 参数；在线配置使用 bf16 forward autocast，避免
  fp16 SDE overflow。
- SANA rollout/replay chunk 固定为实测 bit-exact 且能 backward 的 8/8；首 PPO pass 全量 parity 在
  optimizer 前 hard-fail。
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
