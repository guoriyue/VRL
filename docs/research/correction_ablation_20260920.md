# TIS / RS / recompute：真实训练短跑与受控 loss 检查

日期：2026-09-20。训练代码起点：`1a5ceeeb06f4e135e06d006f2d80d71bac05296c`。

## 结论与真实训练结果

四组均完成。**在本次已对齐的 SANA 设置中，三种修正都没有改变训练结果。**
不是只看曲线接近：每组最终 320 个 LoRA 张量与 off 逐位相同，最大权重差为 0；
每一行 CSV 指标也完全相同。四组共 12 次更新、96 张训练生成图。

| 模式 | 三轮最大原始 parity 差异 | TIS 截断比例 | RS 拒绝比例 | 最终不同权重张量 |
|---|---:|---:|---:|---:|
| off | 0 | 0 | 0 | 基线 |
| TIS truncate | 0 | 0 | 0 | 0 / 320 |
| RS seq_mean_k1 | 0 | 0 | 0 | 0 / 320 |
| recompute on | 0 | 0 | 0 | 0 / 320 |

共同的三轮 aesthetic reward：`4.3392, 4.6412, 4.4173`。
共同的梯度范数：`0.216281, 2.683062, 0.664936`。
初始可训练权重 SHA256：`40cc9acb1b5743e0b9be282a59bb491cc4c8379b68ef5cebcee36279ea3f6d34`。
最终 160 个 LoRA B 张量中 145 个非零，最大绝对值 `0.0009014656534418464`；
不是因为没有执行训练而四组相同。

建议：这套设置保持 correction 默认 off，保留 parity 观察即可。没有实测依据把
TIS/RS/recompute 宣传成普遍提高 reward 的开关。当前只验证一个种子和三次更新，
没有 held-out 评测；没有验证真实 FP8、compile split、异步陈旧策略、或标准配方的
ppo_epochs=4。长程收益仍未证明。若实际部署存在漂移，应在该部署上做配对实验，
不应从本次“无漂移时无变化”外推“有漂移时也没用”。

可提交的原始数值见同目录 `correction_ablation_20260920.json`。
大文件保存在 `outputs/correction_ablation_20260920/`：成功组分别是 `off/`、
`matched/seed42_tis/`、`local/seed42_rs/`、`recompute/seed42_recompute/`。
其中 `comparison.json` 也列出未完成的尝试，须以 `checkpoint_exists` 区分。

## 实验问题与边界

检查现有三个开关在真实生成、奖励、反向传播、更新和权重同步中是否改变训练，
以及它们在哪种 log-prob 差异下才触发。不是长程 reward 提升的统计实验。
不能把合成 loss 检查当作真实 FP8/compile 精度差异，也不能把不同训练提示词上的
reward 上升当作 held-out improvement。

## 固定协议

- RTX 5090 32 GB；PyTorch 2.11.0+cu130；使用 `~/Desktop/vrl2/VRL/.venv/bin/python`
  加 `PYTHONPATH=~/Desktop/VRL`，代码来自本仓库。
- SANA 1.6B，revision `d1b54936033cd7d45410ecadd692c5c502a19a38`，FP16，compile 关闭。
- 基于 `experiment/sana/online_grpo_aesthetic`，512px、10 步、CFG 4.5、LoRA rank 16。
- seed=42；每轮 2 个 prompt × 4 个样本；生成 batch=4，训练 microbatch=4。
- 每组 3 次 optimizer update；`ppo_epochs=1`（recompute 的适用约束），
  timestep_fraction=0.5、EMA 关闭、lr=3e-4、clip_ratio=1e-4、KL=0.04。
- aesthetic 优化权重为 1；原配置的 PickScore 仍为权重 0 的 CPU 观察指标。
  配置字典 overlay 不删除已有 PickScore 项，所有组保持一致。
- 每次更新检查 parity，保留默认 0.01 阈值；额外 drift guard 为 warn。
  没有为了让 baseline 通过而放宽 parity 阈值。
- 四组只切换 correction：off、TIS truncate（默认 cap=2）、RS seq_mean_k1
  （默认 log-ratio 范围 ln(0.5)..ln(2)）、recompute on。
- 初始 LoRA 指纹和最终权重直接核对；不是只比较 reward 曲线。

## 受控 loss 检查（合成输入，非真实模型误差）

直接调用生产 `GRPO.compute_loss`，clip_ratio=1e-4、KL=0。
两个样本 advantage 分别为 +1、-1。两者注入同样的 log-prob 差异，
报告 loss 对每个新 log-prob 的导数；这不是模型参数梯度，也不是 reward 实验。

| 注入差异 | off 导数 (+A, -A) | TIS truncate | RS | recompute |
|---|---|---|---|---|
| 0 | (-0.5, 0.5) | 相同 | 相同 | 相同 |
| +0.001 | (0, 0.5005003) | 相同、不触发 | 相同、不触发 | (-0.5, 0.5) |
| +0.8 | (0, 1.1127704) | (0, 0)，全部截断 | (0, 0)，全部拒绝 | (-0.5, 0.5) |
| -0.8 | (-0.2246645, 0) | 相同，单边 cap 不触发 | (0, 0)，全部拒绝 | (-0.5, 0.5) |

含义：默认 TIS/RS 的门槛远宽于 PPO 的 1e-4 clip，并不处理所有足以触发 PPO
clipping 的小漂移。当前 TIS 是截断 ratio 的梯度；不等同于 detached 权重乘 loss。
recompute 恢复了这里的无漂移导数，但真实生成分布若已改变，这也可能隐藏应有修正。
不能据此认定 recompute 总是正确或 reward 一定更高。

## 复现

在本仓库，使用有 Ray、diffusers、vLLM 的 Python 环境：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$PWD" \
  /home/mingfeiguo/Desktop/vrl2/VRL/.venv/bin/python \
  -m vrl.scripts.perf.correction_ablation \
  --output outputs/correction_ablation_new \
  --model-path /home/mingfeiguo/.cache/huggingface/hub/models--Efficient-Large-Model--Sana_1600M_1024px_diffusers/snapshots/d1b54936033cd7d45410ecadd692c5c502a19a38

.venv/bin/python -m vrl.scripts.perf.correction_ablation \
  --output outputs/correction_ablation_new \
  --compare-to outputs/correction_ablation_new/seed42_off
```

`--seeds 42 43 44` 可扩充种子；`--replay-batch 1` 可检查 batch shape 差异。
`--probe-only` 只跑合成 loss 检查，不启动模型训练。
脚本拒绝覆盖已有 run；原始命令、日志、metrics、checkpoint 均保存在 output。

## 环境排查记录

- 本仓库 `.venv` 缺少 vLLM，初次在 worker 初始化失败；改用已经安装它的 vrl2 环境。
- 完成 off/TIS 后，RS 在线启动卡在 Hugging Face model_info；终止该次尝试并保留日志。
- 仅设 HF_HUB_OFFLINE 仍触发 diffusers 的 model_info 异常；随后明确使用相同 revision
  的本地 snapshot 路径。没有替换权重或修改算法来绕过错误。
- 其他会话期间有 Qwen Image 2.1 家族文件改动；不纳入本次提交。主要 loss/trainer 文件
  的源码哈希记录在原始产物中，实验不修改这些生产实现。
- recompute 的 CLI dotlist 必须传 `recompute_old_logprob="on"`，裸 `on` 被 YAML
  解释成 bool。失败尝试在模型加载前退出；修正后脚本先做 schema 预检查。

## 验证与不变范围

104 个相关算法、diagnostics、precision guard 测试通过；实验脚本通过 Ruff。
本次新增可复现实验与报告，生产 loss、阈值、默认配置和模型实现不改。
三个脚本函数分别负责受控 loss 检查、真实 checkpoint 比较、实际训练调度；不额外拆文件，
不增加业务常量表。原有跨家族接口保持不变。
