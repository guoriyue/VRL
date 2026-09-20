# SPRINT: Flash-GRPO 忠实复现（VRL 原生链路）

状态：本机实施完成，CPU 测试通过（2026-09-20）。4 卡验收未做，见 §6。
交接来源：4 卡机上的 "Flash-GRPO：VRL 原生集成交接"。

官方源码钉版：`Shredded-Pork/Flash-GRPO@bd6051f68e1ab444e5ec7c6ffe0a1f7eaf559a0d`
（本机 `~/Desktop/Flash-GRPO`，pinned commit 存在）。下文行号均指
`scripts/train_wan2_1_flash_1node.py`，除非另注。

## 1. 从源码读出的语义（不是配置名）

### 1.1 批量与更新顺序（sampler + 训练循环）

- `config/dgx.py:wan2_1_flash_1node`：`num_batches_per_epoch=24`，
  `gradient_accumulation_steps=24//2=12`，`num_image_per_prompt=4`（注释自认
  "实际是 2"），`train_batch_size=1`，`global_std=True`，`ema=True`。
- `DistributedKRepeatSampler`（100-160 行）：`virtual_num_replicas = 8 rank × 12`
  = 96 个槽位，`m = 96 / k(=4) = 24` 个唯一 prompt，每个重复 4 次后打乱分到
  96 个虚拟 rank。`set_epoch(virtual_step)`（712 行）每个累积窗口重抽一次，
  所以**每个 update 24 个唯一 prompt，每个 prompt 4 槽 × `num_videos_per_prompt=2`
  = 8 条视频，192 条/update；每 epoch 两个窗口，384 条**。
- 采样完整个 epoch 后（836 行起）先 `stat_tracker.update` 算一次优势，再
  `randperm` 打乱样本、按 `micoe_batch=2` 切成 24 个 microbatch，累积 12 个
  做一次 `optimizer.step`：**一次标准化，两次更新，样本级打乱**。
- 零优势过滤：`mask = advantages.abs().sum(dim=1) != 0`，再随机补回若干零优势
  样本凑齐整除。

### 1.2 优势（`flow_grpo/stat_tracking.py`）

`(r - mean_prompt) / (std_all + 1e-4)`，std 是全部 gathered rewards 的总体标准差，
**epsilon 是加法**。之后 `clamp(±adv_clip_max=5)`。

### 1.3 梯度校正与归一化（1002-1014、1046-1060 行）

`value_dict` 表 = `1/c(t)`。`gradient_accumulation_steps != 1` 时：
`value_norm_list[k] = mean over 12 microbatches of allreduce_mean(value.mean())`，
`w = 1 / value_norm_list[i // 12]`，loss = `mean(max(-w·value·A·ratio, clipped))`。
**分母是整个累积窗口的均值，窗口内所有 microbatch 共用。**

### 1.4 梯度尺度与裁剪

`accelerator.backward(loss)` 不除以累积步数：每 rank 梯度 = 12 个 microbatch
均值之和 = 12 × 全 update 均值；`clip_grad_norm_(max_grad_norm=1.0)` 作用在这个
和上。AdamW（解耦 weight decay）对常数尺度不变，只有裁剪阈值需要换算。

### 1.5 EMA（1118-1126 行 + `flow_grpo/ema.py`）

`ema.step(params, global_step)` 在 **每个 microbatch 之后** 调用，与
`sync_gradients` 无关；`global_step` 在 optimizer.step 之后自增，所以一个 update
的最后一个 microbatch 用的是自增后的计数。门控 `(step+1) % 8 == 0`，decay
`min((1+s)/(10+s), 0.9)`。效果：命中门控的 update 上做 23 次同 decay 的 lerp
加 1 次下一计数的 lerp，0.9^23 ≈ 0.09，**实际接近每 8 个 update 把原始权重
拷进 EMA**。评测和保存用 EMA 权重（`copy_ema_to`）。

### 1.6 SDE 与时间步

- `wan2_1_pipeline_with_logprob2.py:51`：`std_dev_t = sigma_min + (sigma_max -
  sigma_min) * sigma`，`sigma_max = sigmas[1]`，`sigma_min = sigmas[-1]`。
  VRL `flow_grpo` 在 `noise_level=1.0` 下是同一公式。
- 随机步：每个 batch 迭代由 rank 0 对当轮所有 prompt 各抽一个 `index ∈ U{0..9}`
  并广播（737-748 行）。同一 prompt 的 4 个槽位落在不同迭代时会拿到不同的
  index；论文的 iso-temporal grouping 在官方代码里只在同一迭代内成立。
- `if epoch < 2: continue`（731 行）：前两个 epoch 只评测不采样不训练。

### 1.7 HPSv3 输入（`rewards.py:125-190`、`reward-server/app_hpsv3.py`）

`(video*255).round().clamp().uint8` → 每帧 PIL → `save(format="JPEG")`（PIL
默认质量）→ 服务端 `Image.open(formats=["jpeg"])` 打分 → 降序取前
`int(81*0.3)=24` 帧均值。输入来自采样张量本身，不是导出的 mp4。

## 2. VRL 侧的差距与改动

| 项 | 改动前 | 改动后 |
| --- | --- | --- |
| 优势 epsilon | `clamp(std, min=eps)` | `std + eps`（`advantages.py`），`GRPOConfig.eps` 本来就是 1e-4 |
| 校正归一化范围 | 每 microbatch 跨卡均值 | `FlashGRPO.prepare_update(timesteps, scheduler=...)` 在 replay 前用整个 update 的记录时间步算 `mean(1/c)`（`flow_sde_scale_terms` 与 SDE step 共用同一套 sigma/dt/std 读取），所有 microbatch 共用 |
| 一次采集两次更新 | 无 | `actor.optimizer_steps_per_batch`：一次优势计算后，每组样本打乱后分发到 N 个 update，各自 `prepare_update` + optimizer step + EMA（`trainer._optimizer_step_chunks`） |
| EMA 调用时机 | 每 optimizer step 一次 | `actor.ema.step_per_microbatch`：每个 microbatch 后一次（最后一个除外），optimizer step 后计数 +1 再一次 |
| HPSv3 输入 | mp4 解码（H.264 有损） | `input_artifact_format="tensor"` 直接读采样帧；`worker_config.jpeg_roundtrip` 复现 JPEG 往返 |
| 梯度裁剪等价 | — | 配方里 `max_norm = 1/24`（推导见 §1.4） |

`flow_matching.py` 顺带把 sigma 域转换抽成 `_flow_domain_schedule`，
`sde_step_with_logprob` 和新 helper 共用；SDE parity 测试不变。

`FlashGRPO` 的 `prepare_update` 由 `OnlineTrainer._run_replay_pass` 在每个
update 开头调用（`_update_timesteps` 按 `sde_window` 读每组的窗口）。流式
累积（`prompts_per_collection>0` 且多个 collection）下只能看到当前 collection，
分母就是该 collection；忠实复现走全批路径。原有的
`online_flash_grpo_kling_video_reward` 预设不受影响（它仍是流式），但归一化
范围变成"每个 collection"而不是"每个 microbatch"。

## 3. 配方 `experiment/wan_2_1/online_flash_grpo_hpsv3_reference_4x_l40s`

4 卡换算：每 rank 12 组 × 8 样本 × 4 rank = 384 条/批，`optimizer_steps_per_batch=2`
→ 192 条/update；`prompts_per_collection=0`（全批路径）；`global_std=true`；
`clip_ratio=1e-3`；`max_norm=1/24`（4 卡下官方 `gradient_accumulation_steps=24`）；
LoRA r=16/alpha=32/gaussian/8 个投影；`trajectory_storage.dtype=float32`
（无损）；`torch_compile.enable=false`（官方 eager，且 inductor 会让 parity
偏离零误差）；EMA 0.9/8/`step_per_microbatch`；HPSv3 `jpeg_roundtrip`。

与官方的已知差异（有意）：
- 组内 timestep：VRL 每组一个窗口（严格 iso-temporal）；官方只在同一迭代内一致。
- 零优势样本：VRL 直接丢弃；官方丢弃后随机补回若干以凑整除。
- 前两个 epoch 不训练：不复现。
- 4 卡而不是 8 卡：每 update 样本数相同（192），每 rank 的 microbatch 数翻倍，
  只影响 `max_norm` 换算和 EMA 的 lerp 次数（0.9^47 而不是 0.9^23，都≈拷贝）。

## 4. 测试

- `tests/algorithms/test_flash_grpo.py`：参考表复现、update 级分母、microbatch
  切分不变性、未 prepare 即报错。
- `tests/algorithms/test_diffusion_nft.py`：优势 oracle 改为加法 epsilon 的闭式值。
- `tests/trainers/online/test_update_scoped_loss_weight.py`：trainer 把整个
  update 的 (样本, 训练步) 时间步交给算法，每个 ppo epoch 一次。
- `tests/trainers/online/test_optimizer_steps_per_batch.py`：一次优势计算、两次
  optimizer step、样本互斥且完整、每组都分到两个 update、重跑确定性。
- `tests/trainers/online/test_ema_reference_stepping.py`：计数序列 `[0,0,1, 1,1,2]`。
- `tests/rewards/models/test_input_artifact_format.py`：HPSv3 声明 tensor 输入。
- 全部实验预设可加载（`test_load_all_experiments` 等）。

## 5. 4 卡验收清单（交给 4 卡机）

1. `trainer.total_epochs=2` smoke：日志核实每 update 192 条、每组 8 条、
   `optimizer_steps_per_batch` 产生 2 个 `global_step`、`first_step` parity 为 0。
2. 固定输入对照：同一批 rollout 张量上，官方 `sde_step_with_logprob` 与 VRL
   的 log-prob / `prev_sample_mean` / `coe` 逐元素对比；`PerPromptStatTracker`
   与 `group_relative_advantages` 逐元素对比。
3. HPSv3：固定 8 条真实视频，官方客户端+服务端 vs VRL `jpeg_roundtrip=true`
   的 `top_frame_mean`，排序一致、|Δ| 记录。
4. 记录 videos/s、每阶段耗时、显存与主机峰值；评测固定 24 条留出 prompt、
   固定 seed、确定性采样，raw 与 EMA 分别报。

## 6. 未做

- 任何 GPU 运行（本机单卡 5090）。
- 官方"补回零优势样本凑整除"的行为。
- 让流式多 collection 也做 update 级归一化（需要先采完再训，与流式目的相悖）。
