# SPRINT：Janus-Pro-R1 1B Real Run 剩余工作

这份文件只保留还需要额外完成的事项。

## 1. 补齐 Janus-Pro-R1 固定评估

`configs/experiment/janus_pro_1b_r1_ocr_grpo.yaml` 已经声明 `eval.fixed`，但训练路径还需要真正产出固定评估 artifact。

需要完成：

- 在 `vrl/scripts/janus_pro/train.py` 中读取 `eval.fixed.enabled`、`eval.fixed.interval_epochs`、`eval.fixed.prompts`、`eval.fixed.output_dir`。
- 默认 80 epoch run 需要在 epoch `0`、`40`、`80` 执行固定评估。
- 固定评估使用确定性 seed，保证不同 epoch 的样本可比较。
- 固定评估只给最终图片打分，不使用训练 rollout reward。
- 在训练输出目录写入 `eval_metrics.csv`。
- 为每个固定 prompt 保存可检查的 initial image 和 final image artifact。
- 每个评估 epoch 保存两张 contact sheet：一张 initial image，一张 final image。

期望输出结构：

```text
outputs/janus_pro_1b_r1_ocr_grpo_real_run/
  eval_metrics.csv
  fixed_eval/
    epoch_0000/
      initial_contact_sheet.png
      final_contact_sheet.png
    epoch_0040/
      initial_contact_sheet.png
      final_contact_sheet.png
    epoch_0080/
      initial_contact_sheet.png
      final_contact_sheet.png
```

补一个轻量测试：不加载真实 checkpoint 的情况下，用 stub collector/output 验证 fixed eval 启用后会写 metric rows，并生成 initial/final contact sheet 路径。

## 2. 跑真实 Janus-Pro-1B 训练

必须跑真实 checkpoint。1 epoch smoke run 不算完成。

命令：

```bash
python -m vrl.scripts.train \
  --config experiment/janus_pro_1b_r1_ocr_grpo \
  trainer.total_epochs=80 \
  rollout.n_samples_per_prompt=2 \
  rollout.rollout_batch_size=1 \
  trainer.output_dir=outputs/janus_pro_1b_r1_ocr_grpo_real_run
```

通过标准：

- 加载 `deepseek-community/Janus-Pro-1B`。
- 每条 rollout 生成 initial image、self-check text、final image。
- 计算 OCR reward。
- 至少计算一次 image segment 的 TokenGRPO loss。
- backward 可以正常执行。
- 保存 `checkpoint-10`、`checkpoint-20`、`checkpoint-80`、`checkpoint-final`。
- `metrics.csv` 至少有 80 行训练记录加 header。
- `loss`、`reward_mean`、`grad_norm` 都是 finite。
- 整个 run 不能静默产出全零 loss 和全零 grad norm。
- 第 1 节的 fixed eval artifact 必须存在。

如果 advantages 全为 0，把它当作训练质量问题处理，不能标记 sprint 完成。

## 3. 验证从 checkpoint 80 resume 到 100

80 epoch run 完成后，从 `checkpoint-80` 继续训练到 100。

命令：

```bash
python -m vrl.scripts.train \
  --config experiment/janus_pro_1b_r1_ocr_grpo \
  trainer.total_epochs=100 \
  trainer.resume_from=outputs/janus_pro_1b_r1_ocr_grpo_real_run/checkpoint-80 \
  trainer.output_dir=outputs/janus_pro_1b_r1_ocr_grpo_real_run
```

通过标准：

- resume 从 epoch 80 开始，继续跑到 epoch 99。
- LoRA weights 正确恢复。
- optimizer state 正确恢复。
- RNG state 正确恢复。
- 继续 append `metrics.csv`，不能重写 header。
- resume 后保存新的 checkpoint state。
- 保存 `resume_config_*.yaml`。

## 4. 写 run gap 文档

真实训练和 resume 验证通过后，写：

```text
outputs/janus_pro_1b_r1_ocr_grpo_real_run/REFERENCE_GAP.md
```

必须说明：

- 本 run 使用 Janus-Pro-1B，不是 Janus-Pro-7B。
- 本 run 是 single-GPU / small-batch 的机制验证，不是官方 8-GPU ZeRO-2 recipe。
- reward 是 OCR / local baseline reward，不是官方 bi-level QA reward serving stack。
- trainer 使用本 repo 的 TokenGRPO / MultiSegmentTokenGRPO 路径，不是官方 TRL trainer。
- 结果不能描述成 paper-result reproduction。

## 5. 更新项目对外状态

第 1-4 节全部通过后，更新相关 README 或 experiment note，避免后续读者把它误认为 paper-result reproduction。

需要写清楚：

- `janus_pro_1b_r1_ocr_grpo` 是 Janus-Pro-R1 风格的 1B 机制级 real-run wiring。
- real-run artifacts 位于 `outputs/janus_pro_1b_r1_ocr_grpo_real_run`。
- 距离官方 Janus-Pro-R1 的差距仍包括 7B scale、official reward、official distributed recipe、Stage 1 SFT/data。

## 6. 最终验证

跑 sprint 相关结构性测试：

```bash
python -m pytest -q \
  tests/models/test_janus_r1_policy.py \
  tests/rollouts/test_ar_r1_packer.py \
  tests/rollouts/test_janus_pro_r1_wiring.py \
  tests/rollouts/test_multisegment_token_logprob.py \
  tests/algorithms/test_multisegment_token_grpo.py \
  tests/config/test_janus_pro_r1_config.py \
  tests/rollouts/test_family_registry.py
```

跑 config coverage：

```bash
python -m pytest -q tests/config
```

标记完成前跑完整默认 suite：

```bash
python -m pytest -q
```

只有下列 artifact 全部存在时，这个 sprint 才能标记完成：

```text
outputs/janus_pro_1b_r1_ocr_grpo_real_run/checkpoint-80
outputs/janus_pro_1b_r1_ocr_grpo_real_run/checkpoint-final
outputs/janus_pro_1b_r1_ocr_grpo_real_run/metrics.csv
outputs/janus_pro_1b_r1_ocr_grpo_real_run/eval_metrics.csv
outputs/janus_pro_1b_r1_ocr_grpo_real_run/REFERENCE_GAP.md
outputs/janus_pro_1b_r1_ocr_grpo_real_run/fixed_eval/epoch_0000/initial_contact_sheet.png
outputs/janus_pro_1b_r1_ocr_grpo_real_run/fixed_eval/epoch_0000/final_contact_sheet.png
outputs/janus_pro_1b_r1_ocr_grpo_real_run/fixed_eval/epoch_0040/initial_contact_sheet.png
outputs/janus_pro_1b_r1_ocr_grpo_real_run/fixed_eval/epoch_0040/final_contact_sheet.png
outputs/janus_pro_1b_r1_ocr_grpo_real_run/fixed_eval/epoch_0080/initial_contact_sheet.png
outputs/janus_pro_1b_r1_ocr_grpo_real_run/fixed_eval/epoch_0080/final_contact_sheet.png
```
