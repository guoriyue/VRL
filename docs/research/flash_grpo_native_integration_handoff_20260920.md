# Flash-GRPO：VRL 原生集成交接

## 目标和当前状态

在 VRL 自己的 trainer / rollout / reward / checkpoint 链路中实现并复现
Flash-GRPO，使用 VRL 的标准训练入口。不要调用官方训练脚本作为后端，也不要
把外部环境 `flash-grpo-env` 作为 VRL 运行要求。

外部官方脚本实验已停止，四卡显存已释放；此前未验证完成任何 optimizer update。
外部启动辅助脚本已移出 VRL，诊断记录保留在 outputs/flash_grpo_upstream_20260920。
用户会在另一台机器准备实现，再提供 branch/commit 供此机 fetch、验收和运行。

## 一手参考资料

- 论文：https://arxiv.org/abs/2605.15980v3
- 官方源码钉版：`Shredded-Pork/Flash-GRPO@bd6051f68e1ab444e5ec7c6ffe0a1f7eaf559a0d`
  https://github.com/Shredded-Pork/Flash-GRPO/tree/bd6051f68e1ab444e5ec7c6ffe0a1f7eaf559a0d
- 按此顺序读官方文件：
  1. `config/dgx.py:wan2_1_flash_1node`、`config/base.py`：配置和默认值。
  2. `scripts/train_wan2_1_flash_1node.py`：实际采样数、组统计、更新边界、EMA 调用。
  3. `flow_grpo/diffusers_patch/wan2_1_pipeline_with_logprob_sample.py`：采样 SDE/ODE。
  4. `flow_grpo/diffusers_patch/wan2_1_pipeline_with_logprob2.py`：replay 密度和评测。
  5. `flow_grpo/stat_tracking.py`：advantage 和 global std 的精确语义。
  6. `flow_grpo/rewards.py:videohpsv3`、`flow_grpo/ema.py`。
- HPSv3：https://github.com/MizzenAI/HPSv3/tree/upgrade_transformers_version
  此机核对版本 `ad33a29d5896fe46ae74f9865444d6adbadbc750`；权重 MizzenAI/HPSv3。

## VRL 已有实现，不需要从零重写

- `vrl/algorithms/grpo/continuous.py:FlashGRPO`：已有梯度校正公式。
- `vrl/math/denoise/flow_matching.py`：SDE transition / log-prob。
- `vrl/algorithms/advantages.py`：组优势和跨卡统计。
- `vrl/generation/steps/denoise/`、`vrl/generation/types.py`：单步窗口及随机性传递。
- `vrl/trainers/online/trainer.py`：收集、replay、microbatch、optimizer 边界。
- `vrl/trainers/online/ema.py`：已有 EMA；还需核对调用次数/step 编号。
- `vrl/rewards/models/hpsv3.py`、`vrl/rewards/functions/hpsv3.py`：已有 HPSv3。
- `tests/algorithms/test_flash_grpo.py`：已有公式与参考系数表测试。
- `vrl/scripts/eval/wan_hpsv3_checkpoint_eval.py`：固定提示词配对评测。

现有 `online_grpo_hpsv3_fsdp_4x_l40s` 和
`online_flash_grpo_kling_video_reward` 只能作为结构参考：前者有历史数值/批量偏离，
后者改用了 Kling 奖励。不要凭配置名或旧 sprint 的“已完成”状态认定忠实复现。

## 必须先处理的语义差异

1. **三个机制一起成立**：每条轨迹只有一个随机训练步骤；同一 prompt 组共享
   该步骤；loss 使用时间相关梯度校正。其余采样步骤走确定性 ODE，不参与 surrogate。
   分块生成、跨 rank 分组、改变生成 batch 大小时，不能重新抽组内 timestep。
2. **校正分母的范围**：VRL 当前 `_loss_weight` 按当前 microbatch 跨 rank 求均值。
   官方梯度累积路径按整个 optimizer update 的窗口求均值。不能把二者当作完全等价；
   特别是小 microbatch 会消掉或改变校正作用。应提前计算完整 update 的归一化量，
   在各 microbatch 使用同一分母，并验证对 microbatch 划分不敏感。
3. **实际批量不能只读 YAML**：官方脚本硬编码 `num_videos_per_prompt=2`，并对
   prompt/replay 张量 repeat_interleave(2)。八卡 24 requests/rank/epoch、累积 12，
   在零优势过滤前实际是 384 videos/epoch、192 videos/update、两次更新；配置中
   写 G=4，但实际重复 prompt 通常有 8 条视频。四卡对应 48 requests/rank、累积 24。
   这些是源码静态推导，需用一次真实 shape/计数日志核实，过滤后实际数量另记。
4. **收集/标准化/更新次序**：官方先收集一整个 epoch，计算该集合的 global std，
   再做两次 optimizer update。每 96 条重新采样和标准化，不是同一训练语义。
   还需核对 shuffle、零优势过滤、advantage clipping、group key、std 的 epsilon。
5. **EMA 不只是 decay=.9 / interval=8**：核对 warmup decay、global_step 自增位置、
   每个 microbatch 是否调用，以及评测/导出使用 raw 还是 EMA。不要直接照搬配置数字。
6. **奖励输入也属于配方**：原实现先将帧量化为 uint8 并 JPEG 编码，再用 HPSv3
   逐帧打分，取最高 floor(81*.3)=24 帧的均值。核对像素预处理、文本模板、special
   token、M-RoPE 行为、快/慢 processor 和帧 batch；固定真实帧做排序/分数比较。

## 对齐的基础设置

Wan2.1-T2V-1.3B-Diffusers 基座，revision
`0fad780a534b6463e45facd96134c9f345acfa5b`，从零初始化 adapter。
LoRA r=16、alpha=32、Gaussian init；target modules 以官方脚本列表为准。
480×832、81 帧；训练 20 steps、评测 50 steps；CFG 4.5；随机步骤从前 10 步中选。
AdamW lr=1e-4、betas=.9/.999、weight_decay=1e-4、eps=1e-8；clip=.001；KL=0。
参数、autocast、SDE 和 trajectory 存储精度要分别记录，不能仅写“BF16”。

官方数据 `dataset/video/train.txt` / `test.txt`；VRL 对应
`manifests/flash_grpo_video/train.txt` / `test.txt`，此机已逐字节匹配：

- train：19,700 条，SHA256 `610e16e6b0e8a30bbb1e75cf87413bf592fe491faa74d918d57d914d865e0016`
- test：300 条，SHA256 `96536b01ab2d091158118c4fd9f548950abce4fa5fa9a7f9ac2ebaae47776da5`

## 验收与交付

先做便宜的固定输入参考对照：SDE mean/std/logprob、rectification 权重、advantages、
loss、adapter gradient 和一次更新。增加改变 microbatch/rank 划分而结果不变的测试；
明确浮点比较容差和精度，不把公式表测试通过当成完整训练等价。

四卡验收检查实际视频数、每组样本数、组内 timestep、标准化集合、optimizer 次数；
要求有限且非零梯度、有效 reward variation、checkpoint 恢复一致。
保留用户要求的零误差 unchanged-policy parity：应在任何参数更新之前检查对应
采样策略；官方第二次更新已改变策略，不能再对旧 rollout 要求差值为零，也不能
重写 old_log_probs 来“通过”检查。存储应无损、rollout/replay 精度和 batch 对齐。

评测必须固定提示词和 seed、使用确定性采样，并明确 raw/EMA。初期 24 条留出提示词
用于肉眼和配对统计，完整结果再跑 300 条；同时记录 frame_mean/frame_min，避免只
报告 top-frame reward。报告每阶段耗时、videos/s、每更新样本数、显存和主机峰值。

完成后给本机：branch + commit SHA、VRL 标准入口的 preset 名、依赖变更、已通过
测试、仍有的明确偏离。届时再 fetch，先短验收，再长跑；不要启动外部官方脚本。
