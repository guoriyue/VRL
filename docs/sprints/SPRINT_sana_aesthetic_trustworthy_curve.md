# SPRINT: SANA 美学 GRPO 可信曲线

> 2026-07-11 migration note: online fixed eval has been removed. The training
> configuration now only saves checkpoints; run the held-out prompt/seed
> evaluation as a separate checkpoint job. References below to
> `eval_metrics.csv`, baseline epoch `-1`, and inline eval cadence describe the
> superseded execution plan and must not be used as launch instructions.

状态：**READY / WAITING FOR GPU（2026-07-10）**。运行协议、预算和 PASS/FAIL 已在 baseline
生成前冻结；当前 GPU 正由既有 Cosmos 曲线占用，SANA 不抢占、不并跑。GPU 与 `vrl-train` 进程均空闲后启动。

## 0. 结论先行

这不是“跑到看起来上升为止”的探索，而是一次固定预算实验：SANA 1.6B、DrawBench 192 条训练 prompt、
与训练集精确去重后的 64 条 fixed-eval prompt、300 次 optimizer update、每 25 次 fixed eval、每 prompt
固定 2 个样本。主结论只读 `eval_metrics.csv`，训练批次的 `reward_mean` 不参与 PASS/FAIL。

主跑 PASS 后才启动 50-update LoRA+fp8 rollout smoke。它验证 master-free fp8 adapter 同步进入真实训练循环；
**它不能单独证明“32GB 训练 17B”**。17B fp16/bf16 replay 权重自身约 34GB，训练侧仍装不下 32GB。
本 sprint 能解锁的是“17B master-free fp8 rollout 的构建与同步前置条件”；真正的 17B 单卡训练还需要
训练侧量化、参数 offload 或分片，必须另行真机验收。

## 1. 为什么选这条曲线

- SANA rollout 已完成真权重生成与 replay parity 验证，但没有跑过短 GRPO 曲线；旧 landing sprint 明确保留
  了这个空白。
- aesthetic reward 是 CLIP ViT-L/14 image embedding 加本地 MLP 头，没有外部 judge 服务，适合复现。
- SANA 必须使用 fp16。该 checkpoint 的 linear attention 在 bf16 下已实测产生 confetti artifact；因此主跑
  配置固定 `precision: fp16`，不接受临时改回 bf16。
- 仓库既有算法测量显示，8–12 update 只能证明管线工作，不能证明 learning；可信 learning 需要约
  200–300 update。这里预注册 300，不在中途因曲线形状延长或缩短。

## 2. 固定资产与运行配置

- 训练配置：`vrl/config/presets/experiment/diffusion/sana/online_grpo_aesthetic.yaml`
- 训练 prompt：`datasets/drawbench/train_192.txt`
- fixed eval：`datasets/drawbench/eval_64.txt`
  - 从 `datasets/drawbench/test.txt` 保序去重；
  - 排除与 `train_192.txt` 精确重复项；
  - 64/64 与训练集不重叠；开跑后不得替换。
- 目标：`aesthetic: 1.0`。
- 只观测不优化：`pickscore: 0.0`。`MultiReward` 仍计算并记录 PickScore，但它不进入 advantage。
- update：300；`ppo_epochs: 4`；LoRA r16/alpha32；10-step denoise；CFG 4.5。
- fixed eval：baseline `epoch=-1`，之后每 25 update 一次，64 prompts × 2 samples，seed `20260710`。
- 输出：`outputs/sana_aesthetic_trustworthy_curve/`；checkpoint 每 25 update。

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
   当前 fixed eval 只落聚合统计量，因此这里诚实使用 pooled standard error，不伪称 paired t-test。
3. **方向**：所有 post-training fixed-eval 点对 epoch 的 OLS slope `> 0`。
4. **prompt-aware 防线**：终点 3 点平均 PickScore 不低于 baseline 的 98%。
5. **梯度/策略活性**：所有 update 都实际训练 prompt；至少一个 `grad_norm > 0`；warm-up 后至少 25% update
   的 `clip_fraction > 0`。
6. **多样性**：末 32 update 的 `reward_std` 中位数 `> 1e-4`，且不低于最初 32 update 中位数的 25%。
7. **训推一致**：`logprob_abs_diff_max <= 0.01`，且 first-step 守卫无报错。
8. **定性审计**：§4 PASS。

### 3.2 任一项触发即 FAIL

- 到 300 update 时主统计量平、负或不显著；不得加跑找显著。
- 非有限 loss/reward/gradient、零 prompt update、长期零梯度、多样性坍缩、parity 越界。
- aesthetic 上升但 PickScore 超过允许回退，或盲审发现系统性 reward hacking。
- 运行中改变判据、seed、eval prompt、LR、精度或预算。若因实现 bug 修代码，旧 run 作废；从新 baseline
  全量重跑并在本文件记录原因，不能拼接修复前后的曲线。

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
  precision.train=fp16 precision.rollout=fp8 precision.rollout_recipe=rowwise \
  trainer.total_epochs=50 trainer.save_freq=10 trainer.eval.freq=10 \
  trainer.output_dir=outputs/sana_aesthetic_fp8_master_free_smoke
```

smoke PASS 必须同时满足：

- 日志显示 fp8 swap 命中大 linear，且 `dropped bf16 masters` 释放量大于 0；
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

- SANA 两个在线配置从会破坏 linear attention 的 bf16 改为 fp16。
- canonical DrawBench preset 接入唯一 fixed-eval manifest；旧 40/64 训练重叠的 eval 列表改为真正 disjoint。
- master-free partial-load 和 LoRA+fp8 CPU 构建顺序增加行为回归测试。
- 主跑使用机械 verdict，不允许口头重解释阈值。

保持不变：

- `fixed_eval.py` 保持独立薄模块：它是多 recipe 共用的 fixed-eval 协议边界。
- `drop_fp8_masters` 保持薄函数：它是量化与 weight-sync 的安全边界。
- fp8 的协议/架构常量继续保留；不新增业务词表或重复 prompt 常量。
- `torch_compile` 继续关闭；SANA 首条可信曲线不同时引入 compile 变量。

非目标：刷 SOTA aesthetic、改 GRPO 数学、把 PickScore 加入优化目标、扩展第二个 family、用短 smoke 替代
learning 结论。

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
