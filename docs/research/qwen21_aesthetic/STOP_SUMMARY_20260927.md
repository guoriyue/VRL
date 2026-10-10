# Qwen-Image-2.1 美学 RL：停止与交接总结

**用户已要求停止。训练与评测等待任务均已停止；不得依据早先“持续占用 GPU”的要求自动恢复。美学改善目标尚未实现，不能把停止当成成功。**

## 停止状态

- 停止日期：2026-09-27（太平洋时间）；精确 UTC 时间见下方机器记录。
- 最后完整保存：低学习率分支第 **23** 轮，global_step **46**。
- 停止时正在第 24 轮；该轮未完成，不保证保留其未保存计算。
- 已停止 `qwen21-aesthetic-lr3e5-from15.service` 与 `qwen21-aesthetic-lr3e5-independent-eval.service`。先发送正常停止，随后清理仍未退出的训练子进程；最终两者 inactive、MainPID=0。
- 四卡释放，停止核对时没有 GPU 计算进程。近期内存约 218–224 GiB，未观测到本次训练 cgroup 的 OOM、high 或 max 事件。
- 没有删除 checkpoint、原图、数据、日志或旧实验。没有提交、重置或覆盖仓库里其他人的修改。

## 核心结论

**尚无可靠证据证明 Qwen 的总体美学或可见画质有明显提升。** 训练奖励波动、少数好看的例子或 loss 下降均不等于已改善。

旧方案第 15 轮在独立 256 条测试上：美学 +0.0067，95% CI [-0.0215, +0.0348]，不显著；独立 PickScore 约 +0.00411，CI [+0.00251, +0.00576]，151/256 图片分数高于 base。后者是偏好预测器信号，不能替代人类审美判断。该测试已查看，不再是未使用的新测试。

旧方案固定 64 条原生采样评测：

| checkpoint | 美学均分 | 相比 base 的美学差及 95% CI |
|---|---:|---|
| base | 5.869667 | — |
| 15 | 5.907712 | +0.0380 [-0.0367, +0.1302] |
| 43 | 5.933551 | +0.0639 [-0.0014, +0.1492] |
| 47 | 5.909857 | +0.0402 [-0.0269, +0.1237] |
| 60 | 5.937442 | +0.0678 [-0.0137, +0.1641] |

这些美学差的区间均跨 0。第 60 轮 PickScore 相比第 15 轮为 **-0.0044975，CI [-0.0069191, -0.0021732]**；相比 43、47 轮也下降。不能把第 60 轮的美学均分最高解释成“最好模型”。旧训练线后来完整保存到 61 轮后停止。前 60 轮训练 reward 最高是 43 轮 6.2503，但每轮提示词不同。

## 为什么降低学习率，以及它是否有效

降低学习率是诊断假设，不是已经定位到的根因。旧方案后续独立偏好回退，使“缩小更新以保留早期收益”值得验证；这不能排除奖励不符合人类审美、采样差异或其他优化问题。

本次从旧方案第 15 轮恢复完整 checkpoint（含优化器、RNG），将学习率从 1e-4 改为 3e-5。没有从 base 重新训练。原优化器加载会覆盖配置里的 LR，因此仅在派生 checkpoint 中修改 param_groups[*].lr，并验证其余内容一致。原 checkpoint 未修改。

第 16 轮是首次新更新前的采样控制：新旧 reward 完全相同，随后新方案权重更新 L2 约为旧方案 30.1%。证明干预生效，不证明美学改善。

新旧第 16–21 轮训练奖励平均差仅 +0.00102，几乎相同。相同轮号的汇总比较不等于已逐样本验证相同输入，更不是 held-out 结论。

低学习率已完成的训练奖励：

| 轮次 | 平均训练 reward |
|---|---:|
| 16 | 5.970694 |
| 17 | 6.085755 |
| 18 | 5.888514 |
| 19 | 6.035050 |
| 20 | 5.801523 |
| 21 | 5.734112 |
| 22 | 6.024601 |
| 23 | 5.875514 |

低学习率分支还没有完成任何新的固定原生采样质量评测。原定训练到 30 轮再统一评测，已按用户指示取消等待与运行。**目前没有证据支持低学习率比原学习率更好。**

## 配置与数据

- 模型：Qwen/Qwen-Image-2.1，revision `b3179ad355be050328e483a9dfdd9e60cd62adfa`。
- 4×L40S 48 GB；4-rank FSDP，LoRA rank32、alpha64，q/k/v/out.0，共 33,554,432 个可训练参数。完整参数微调此前遇到主机内存问题，本次使用 LoRA。
- 奖励：SigLIP Aesthetic Predictor V2.5，图像美学预测器，权重 1；PickScore 仅用于独立评测。
- 训练提示词：仓库 PickScore SFW，15,486 条。训练采样 1024×1024、40 步、guidance1、noise0.2，连续 2 个 SDE 步，窗口范围 [4,36]；只训练这 2 个时间步。
- 每轮四卡共 16 条提示词，每条 8 个样本；生成 batch2，训练 microbatch2，PPO epochs2，每轮 2 次优化器更新；KL0.04、clip ratio0.0001。
- 当前 LR3e-5；原两步加噪方案 LR1e-4。新旧配置实质变更记录见 `resolved_config_diff.json`。
- 生成、奖励、FSDP 更新按阶段共用四卡，并非四卡每一瞬间都满载。阶段切换允许短暂空闲。
- loss 用于诊断数值和优化行为；组内中心化优势使 policy loss 接近零可能正常，不能用最低 loss 选最好画质。

## 人像、Alice 与网页

Alice 是《爱丽丝梦游仙境》的角色，不是奖励算法。用户给的 GRPO-Guard 图主要展示避免 FlowGRPO 训练退化，不是 Qwen base 到美学增强的直接证据：https://jingw193.github.io/GRPO-Guard/ 。本项目尚未实现/验证 Guard 改法。

人像探针包含 12 个提示词（两种主体×六种风格）、每条 4 种子，base/43/60 共 144 张原生图。按提示词聚类统计：43-base +0.0384，CI [-0.0264,+0.1129]；60-base -0.0344，CI [-0.0994,+0.0386]；60-43 -0.0728，CI [-0.1258,-0.0253]。这是小型、有意选择的人像探针，不能外推总体。没有完成正式人类盲评。

- 总体比较网页：https://qwen-aesthetic-checkpoints.rigue.chatgpt.site
- 人像网页：https://qwen-aesthetic-checkpoints.rigue.chatgpt.site/portraits.html
- 历史 256 条测试：https://qwen21-aesthetic-review.rigue.chatgpt.site/fresh256.html
- 这些是之前已部署的网页；本次停止训练未关闭网页。访问权限保持原设置。展示 WebP95，评分来自原始 PNG。

## 保存位置

所有实验根目录：`/mnt/nvme/outputs/qwen21_aesthetic_lora/`

- 旧两步方案：`window2_trial/`，含 eval_step15/43/47/60、PickScore 和配对报告。
- 本次低学习率：`window2_lr3e5_from15/`。
- 最新可恢复 checkpoint：`window2_lr3e5_from15/train_to_30/checkpoint-23/`，含 checkpoint.pt、checkpoint_meta.json、lora_weights。
- 低学习率训练日志与指标：`train_to_30/service.log`、`metrics.full_precision.csv`、`reward_components.jsonl`、`resolved_config.yaml`、`run_evidence/`。
- 完整性记录：`checkpoint16_acceptance.json` 至 `checkpoint23_acceptance.json`；停止记录 `stopped_by_user.json`。
- 学习率对照记录：`first_update_comparison.json`、`training_reward_comparison_16_21.json`、`protocol.json`、`resolved_config_diff.json`。
- 人像原图与统计：`alice_portrait_probe/`，含 `scores.verified.jsonl`、`paired_summary.json`。
- 网页源目录：`/mnt/nvme/work/qwen-checkpoint-comparison/`。
- 分析与执行脚本：`/home/ubuntu/VRL/docs/research/qwen21_aesthetic/`。

## 尚未执行的工作与恢复条件

**只有用户明确要求恢复后才重新启动任务。**

1. 如继续本次实验，从 checkpoint-23 恢复完整优化器/RNG，LR3e-5；使用新的输出目录保留旧日志。total_epochs 是累计目标轮次 30，不是再追加 30 轮。不要直接重跑 lower_lr_continuation.py --run：其输出目录已存在，且协议起点仍是派生的 round15。
2. 原计划评测低学习率30与base/15，固定64条、原生40步、相同种子，加独立PickScore。两者都尚未执行。64条已反复用于验证，应称固定验证集；有积极结果后还需要预先确定的新测试集和盲评。
3. 为直接评估 LR 干预，还应补原学习率 round30 与低学习率 round30 对照。旧 checkpoint 位于 `window2_trial/train_to_60/checkpoint-30/`。
4. `evaluate_lr_control30.py` 刚写好，仅通过语法解析；**未 prepare、未启动、未做运行验证**。用户叫停时尚未完成这项工作。脚本中等待的旧 systemd PID/InvocationID 已失效，恢复前必须重设计/更新等待条件；不能直接启动。
5. `evaluate_lower_lr.py` 同样绑定已停止的旧训练进程，不能直接重新启动。应关联新的 run、进程身份、成功退出结果和 checkpoint，再核对 PNG 哈希、提示词、种子和 base 一致性。
6. 如果原生对照没有改善，应检查奖励与人类审美的一致性、采样匹配和优化诊断；不能仅继续降 LR 或无限延长训练。增加 Guard 等方法前应先确认其机制是否适用。

停止时间（UTC）：`2026-09-28T03:00:51.797981+00:00`

停止后重新核对的 checkpoint-23 SHA256：`5f810b7087bb09b7d4f72ea7e930ddbec3552ea62ef44f6d98cf5e00933464ce`
