# SPRINT 总纲：把 VRL 的执行层提到 miles_diffusion 的水平

状态：**active（2026-09-14）**。用户目标：基础设施层追平 miles_diffusion；算法、reward、
配置层保持 VRL 的领先（对照见 `reading/SPRINT_miles_diffusion_comparison.md`）。
本文是总纲：七条工作流各自的现状证据、目标形态、KILL-RISK 门、首个可执行步骤；
每条落地后把执行记录写回对应 sprint 文件。

## 0. 已知事实（本仓库 + 本机实测）

- SD3.5 512px replay 发射绑定：eager 下 evaluate 阶段 GPU SM 44%；compile 后 63–67%，
  epoch 1.89–2.48×，但 rollout/replay 漂移 0.014–0.030 使 `clip_ratio=1e-4` 下 54–60% 样本被
  clip（`SPRINT_four_l40s_execution.md`）。
- 进程内 reward 与训练争 GIL：+61 s/epoch；已用托管服务 + park/wake 租约解决
  （`planned/SPRINT_reward_service_isolation.md` P1–P4）。
- 序列并行只在 rollout 侧手写了 SD3 Ulysses，512px 图像上无收益；训练侧无 CP。
  diffusers 0.38 自带 `ContextParallelConfig`（ulysses_degree/ring_degree）和 Wan 的
  `_cp_plan`，SD3/Cosmos 没有 plan。
- FSDP mesh 只允许 `["dp_shard"]`（`vrl/trainers/fsdp.py:46`）。
- 权重同步：Ray 快照 / bucket（`RayGenerationWeightSync`）；LoRA 不走 IPC。
- sglang-diffusion 的 RL API（`/rollout/generate`、rollout log-prob、`[T+1]` latent
  trajectory）在 2026-07 已核对（`parked/SPRINT_sglang_diffusion_execution_provider.md`）；
  miles_diffusion 用它跑通 SD3.5 / Qwen-Image / Wan2.2 / LTX-2.3 / Cosmos3，引擎从
  `sgl-project/sglang` 的 `sglang-miles-h3` 分支构建，并用 monkey patch 把引擎算子对齐到训练侧
  前向（`miles/backends/sglang_diffusion_utils/monkey_patches/`）。

## 1. 七条工作流与顺序

顺序由依赖和收益决定，不是由表格顺序决定：

| # | 工作流 | 解决表格哪一行 | 依赖 | 首个门 |
|---|---|---|---|---|
| A | 反序列化与 artifact 出 driver | 反序列化 | 无 | Wan HPSv3 continuous 下 py-spy 无 driver 侧 pickle/mp4 热点 |
| B | sglang-diffusion 作为 rollout provider | rollout 引擎、train/rollout 一致性、LoRA IPC 的前提 | 无（spike 独立） | 引擎在 L40S 上能装能跑 SD3.5 `/rollout/generate`，trajectory 可回放 |
| C | 训练侧 USP（diffusers CP） | 训练并行 | 无 | Wan 1.3B 2-GPU CP 与 1-GPU 梯度逐位/容差一致 |
| D | driver 拆成控制 + 训练 actor | driver | A（把 driver 侧重活先搬走，剩下的才值得拆） | 单卡 recipe 指标不变、epoch 墙钟不劣化 |
| E | LoRA 经 IPC 同步到 colocated 引擎 | 权重同步 | B | 同步时长与 bucket 传输对比 |
| F | 引擎侧一致性（deterministic 模式 + patch） | train/rollout 一致性 | B | compile/batch>1 下 parity 门通过或按实测重定 |
| G | 多节点验证 | 多节点 | 第二台机器 | 2 节点 Wan 配方 2 update + resume |

A、B、C 三条互不依赖，可以并行推进；D、E、F 在其后；G 等硬件。

## 2. 各工作流

### A. 反序列化与 artifact 出 driver（对应 miles 的 parser actor 池）

现状：Ray worker → driver 的 GenerationOutput 走 pickle 进 driver；reward artifact
（`DiskRewardArtifactStore._write_one`，视频是 libx264 编码）在 driver 线程里落盘。
目标：
1. rollout worker 直接产出 reward artifact 文件（它已持有张量），driver 只传路径 + sha256；
   `RewardSample.output` 对磁盘型 reward 变成引用。
2. trajectory 以 Ray object ref 交给 trainer 侧的 batch builder，在 collector 线程外的
   CPU actor 里做切 batch 与校验（parser actor 池），driver 拿到即用的 batch。
门：Wan 1.3B + HPSv3 continuous 的 py-spy 采样里 driver 侧 pickle/mp4 帧 < 2%，且
epoch 墙钟不劣化。
非目标：跨节点对象传输策略。

### B. sglang-diffusion rollout provider（KILL-RISK spike 先行）

Spike（本机 GPU 3，一天内）：
1. 在 `/mnt/nvme/venvs/sglang-diff` 从 `sglang-miles-h3` 分支安装（miles_diffusion 的
   Dockerfile 配方：`pip install -e python[diffusion]` + `sglang-kernel 0.4.5+cu129` +
   `torch_memory_saver`），验证 `sglang.multimodal_gen` 可导入。
2. 起 SD3.5-medium 服务，`POST /rollout/generate` 拿 `[T+1]` latent trajectory 与
   rollout log-prob；用 VRL 的 replay evaluator 对同一 trajectory 算 log-prob，量漂移。
   漂移可接受（或用 `precision_correction.recompute_old_logprob=on` 由训练侧重算）即过门。
3. 记录 L40S 上每 16 样本组的生成墙钟，对比我们 eager loop 的 13.1 s（batch 16）。
落地形状沿用 2026-07 的设计：`GenerationWorkerCore` 内的 chunk executor 持有一个子服务进程，
response 转 native trajectory；native runtime 仍拥有 admission、policy version、lease。
KILL 条件：引擎装不上（CUDA/驱动/内核 wheel 不兼容）、SD3.5 rollout mixin 缺失、或
trajectory 不能被 VRL replay 回放。

### C. 训练侧 USP

现状：`build_fsdp_mesh` 拒绝非 1D mesh。diffusers 0.38 提供 `ContextParallelConfig` 与
Wan 的 `_cp_plan`（`transformer_wan.py:552`）。
目标：`distributed.training.fsdp.mesh: ["dp_shard", "cp"]` + `context_parallel:
{ulysses_degree, ring_degree}`；trainer 对 transformer 调 diffusers 的
`enable_parallelism(ContextParallelConfig)`，replay 的 logprob 归约按 CP 切分做 all-reduce。
门：Wan 1.3B、2 GPU、cp=2 与单 GPU 的 loss / grad-norm 在容差内一致；峰值激活显存下降。
没有 `_cp_plan` 的家族（SD3、Cosmos）先不做，需要时自写 plan。

### D. driver 控制平面化

在 A 之后评估：把训练步移进 `TrainActor`（torchrun rank 进程只留控制、producer、Ray 通信）。
这是最大的结构改动，只有 A 完成后 py-spy 仍显示 driver 侧残留争用时才启动。

### E. LoRA IPC 权重同步（依赖 B）

miles_diffusion 的 `--lora-ipc-weight-sync` 只传 `lora_A/lora_B` 并在引擎侧合并。VRL 的
`RayGenerationWeightSync` 已按可训练状态扁平化传输；引擎落地后，colocated 情形改走 CUDA IPC。

### F. 引擎侧一致性（依赖 B）

复用 miles 的 patch 组（Wan norm/residual fp32 站点、Qwen-Image 逐位对齐）；引擎的
deterministic 模式用于 E2E 标准。与此同时，VRL 的 parity 门和 `clip_ratio` 按实测漂移重定
（batch 16、compile 同一个决定，见 `SPRINT_four_l40s_execution.md`）。

### G. 多节点

仓库已有 `ray_rollout_cross_node` 与双节点 FSDP 记录（2026-06-21）。需要第二台机器时再排。

## 3. 本轮执行记录

- 2026-09-14：程序立项；B 的 spike 安装启动（`/mnt/nvme/venvs/sglang-diff`）；
  A 的设计阅读开始。
- 2026-09-14：A 第 1 步落地（cb92c573）：`RewardArtifactSpec` 随 GenerationRequest 下发，
  rollout worker 在前向后直接写 `.pt`/mp4（uuid 名 + sha256 + size），driver 只收
  `MaterializedArtifact` 引用，`GenerationOutput.video` 对磁盘型 reward 置空；driver 侧
  store `_adopt` 接管文件并保留 release 归属。A 的门（py-spy < 2% driver 侧 pickle/mp4）
  等 GPU 空出后测。托管服务的文件名 bug（hub id 含 `/`）修于 986b5c25。
- 2026-09-14：C 实现完成（未提交，等主机空闲跑测试）：`fsdp.mesh: [dp_shard, cp]` +
  `fsdp.context_parallel.{ulysses_degree, ring_degree}`；trainer 建 3D mesh
  `("dp_shard","ring","ulysses")`（diffusers `ContextParallelConfig.setup` 只认这两个名字），
  `fully_shard` 仍用 1D world mesh；`vrl/trainers/context_parallel.py` 在 `fully_shard`
  之前对解包后的 diffusers 模型调 `enable_parallelism`（无 `_cp_plan` 的家族按类名报错）。
  输出在 `proj_out` 已 all-gather，log-prob / loss 数学在完整序列上不变。参数按 miles_diffusion
  的布局在整个 world 上分片（CP 同组在 FSDP 分片轴内），每个 CP rank 的梯度是同一 loss 的分片
  贡献，`FSDPStrategy.backward` 把 loss 乘以 cp 抵消 FSDP 的 1/(dp·cp) 平均，无需额外集合通信。
  Rollout：`ContextParallelRolloutSchedule` 让 CP leader 采样、组内 gloo 广播 batches，follower
  以空 prompt 走同一 lifecycle（FSDP 权重导出/同步是全 rank 集合通信）；prompt sampler 身份改为
  `dp_rank/dp_size`。预设 `base/distributed/training_fsdp_cp2.yaml`。
  待办：2 GPU Wan 1.3B cp=2 vs 1 GPU 的 loss/grad-norm 门；follower 的 rollout 引擎目前空转，
  与 rollout SP（`gpus_per_engine == cp`）配对是后续项。
- 2026-09-14：F 的第一步（miles 做法）实现：`precision_correction.recompute_old_logprob=on` 从
  "构造即 NotImplementedError" 变为真实模式——old log-prob 取训练侧 replay 前向的 detach 值，
  ppo_epochs=1 下 ratio 恒 1、零额外前向；trainer 拒绝 ppo_epochs>1 与 continuous staleness>0
  的组合；parity 指标仍量 rollout 记录值，漂移可见但不再进梯度。待 GPU：SD3.5 batch-16 +
  compile 配方开此开关，验证 pre-update clip 归零、epoch 1.89–2.48× 提速兑现。
- 2026-09-14：B 第 2 步的脚本就绪（scratchpad `sglang_diff_serve_sd35.sh` 起服务、
  `sglang_rollout_probe.py` 做对比）：POST `/rollout/generate`（sde、noise_level 0.7、debug 张量、
  denoising_env、dit_trajectory），解 msgpack+safetensors，用 VRL 的 SD3.5 transformer（同 revision，
  bf16）在同一 `[T+1]` 轨迹上重算 CFG 噪声预测和 flow_grpo log-prob，分别报告"前向漂移"和
  "公式漂移"（后者用引擎自己的 model_output 算，隔离 SDE 公式差异）。注意 VRL 在 noise_level=1.0
  时 std_dev_t 用 `sigma_min + (sigma_max-sigma_min)*sigma`，引擎恒用 `sqrt(sigma/(1-sigma))*level`，
  对比时必须用 level≠1（这也是引擎 provider 落地时要对齐的一个公式点）。等 GPU。
- 2026-09-14：E（权重同步）按实测重排优先级：SD3.5 LoRA 专用 3x1 配方的
  `rollout.weight_sync_s` = 0.68 s/epoch（epoch ≈ 500 s，<0.2%），bucket 传输对 LoRA 配方不是瓶颈；
  CUDA IPC 只对全参同步（SD3.5 2B bf16 ≈ 4 GB、Wan 14B）有意义，且依赖 B 的 colocated 引擎。
  E 排在 B 之后不变，但门改为"全参配方的 sync 时长"，LoRA 配方不作为目标。
- 2026-09-14 22:55：**B spike 过门**（GPU 3，SD3.5-medium 512px、10 步、CFG 4.5、bf16 DiT）。
  引擎装上并服务 `/rollout/generate`；`[T+1]` 轨迹（fp32）+ 每步 log-prob 可被 VRL 的 SD3.5
  transformer（同 revision）重放：用 VRL 前向重算的 log-prob 与引擎值之差，步 0–7 ≤ 4e-3、步 8
  0.010、终端步（sigma 0.009）0.056；用引擎自己的 model_output 套 VRL 公式，差 ≤ 2e-4（终端步 0.053，
  来源是引擎循环内 bf16 latents 与 fp32 log-prob 的舍入，VRL bf16 trajectory 存储同样会有）。
  模型输出相对差 3.5–6.8%（两侧都是 bf16 核，属预期 kernel 级差异）。吞吐（每 16 样本，单卡）：
  引擎 eager 12.3–13.0 s ≈ VRL eager 13.1 s；引擎 `--enable-torch-compile` 7.8 s（wall 9.3 s 含
  96 MB msgpack 轨迹）。结论：引擎的收益来自 compile/CUDA graph 路径而非 eager 循环本身；
  parity 量级与我们现有的 batch-shape 漂移同级，`recompute_old_logprob=on` 下不进梯度。
  下一步：按 `parked/SPRINT_sglang_diffusion_execution_provider.md` 的设计落地 provider
  （chunk executor 持有引擎子进程，响应转 native trajectory），先做 SD3.5，再对 Wan 做 norm/RoPE
  舍入位置对齐（F）。
- 2026-09-14 23:05：**范围校正（用户）**：VRL 本身已经成熟；本程序的目标是学习这些系统在
  架构层面的解法（进程/GIL 隔离、driver 作为控制平面、事件循环外反序列化、生命周期握手、
  权重更新事务），并用 VRL 自己的设计和词汇落地，**不是**把引擎或框架换成它们的。
  据此调整：B 的引擎只作为参照与测量工具（已给出 compile 路径 1.7× 的证据），不再做
  "provider 替换"；B 的后续改为把同样的收益在 VRL 自己的 rollout 循环里实现
  （rollout 侧 compile/CUDA-graph 路径、batch 不变 kernel、轨迹序列化出事件循环），归入
  rollout 性能那条 cron 与 A/D 两条工作流。E（IPC 权重同步）改为"VRL 自己的 colocated
  worker 间 IPC"，仍以全参配方的 sync 时长为门。
- 2026-09-14 23:25：**A 门第一次测量（严格模式，Wan 1.3B + HPSv3 服务，单 rank GPU 3）**：
  py-spy 20 Hz 采 25 分钟（覆盖 6 次生成），driver 的 Python 线程只活跃 69 个采样 ≈ 3.5 s（0.2%），
  其中 81% 是 Ray pickle 反序列化 + `torch.load`（GenerationBatchResult 里的轨迹张量），其余是
  `catenate_sample_values` 和 Ray 日志转发。结论：严格模式下 driver 侧反序列化在墙钟上可忽略，
  不构成做 parser actor 池的理由；它只在 continuous 模式（反序列化与 backward 同进程争 GIL）
  才可能成为问题，下一次测量在 continuous Wan（trainer + rollout 分卡）上做。同一 run 的
  reward+训练阶段另录 20 分钟（`pyspy_train.raw`）看训练期 driver 的非 GPU 帧。
- 2026-09-14 23:40：C 门进行中。单卡基线（`cp2_baseline_1gpu`，同种子）epoch 0：loss −9.6e-5、
  reward −4.5297±5.2703、parity 0.001973、clip 0、grad_norm 7.07e-4，与 P2 smoke 逐位一致（同种子
  可复现）。cp=2 第一次启动因 CP follower 在广播里等 leader 采样超过 gloo 30 分钟默认超时而失败
  （fe3fc019 改为 12 小时），23:28 重启，epoch 0 约 00:10。
- 2026-09-15 00:40：**A 门第二次测量（严格模式，reward+训练阶段 20 分钟）**：driver 活跃 361 s
  （30%），其中 pickle/torch.load 只有 5.9 s（活跃的 1.6%）；活跃时间的 49% 是 FSDP
  `wait_for_unshard`（等 GPU），**17.5% 是 `move_training_batch_to_device` + trajectory 逐张量
  `.to(device)`（约 63 s/20 min）**——这是一个真实的 driver 侧成本，且与 GIL 无关（主线程同步
  H2D 拷贝）。结论：严格模式下 A 第 2 步（parser actor 池）无收益；更值得做的是训练 batch 上
  设备的拷贝（pinned + non_blocking、或只搬当前 timestep 的切片），归入 D 之前的"driver 瘦身"
  小项。单 rank 两个 update 的 metrics 与基线 epoch 0 逐位一致。
- 2026-09-15 00:40：**C 门**：cp=2 第二次启动通过了采样、reward、组内广播与指纹校验、前向，
  在 backward 失败：Ulysses attention 的 backward 重算 SDPA 时 q/k 为 fp32、v 为 bf16
  （`attention_dispatch.py:869`）。单卡基线两个 update 正常（epoch 1：loss 1.41e-4、reward
  −5.69、parity 0.00204）。假设：激活检查点重算路径与 CP hook 的组合让 norm_q/norm_k 在重算
  时以 fp32 权重执行；正用 `actor.gradient_checkpointing=off` 复跑 cp=2 一个 update 验证
  （cp=2 每卡一半 token，无检查点可能已放得下，本身也是 CP 的内存收益证据）。
- 2026-09-15 01:50：**C 门定位**：2 卡最小复现（Wan 1.3B transformer，随机小 latent）四种组合：
  `gc=on×mp=actor×cp=on` 失败（CheckpointError：重算张量元数据不同）、`gc=off×actor×cp=on` 通过、
  `gc=on×mp=none×cp=on` 通过、`gc=on×actor×cp=off` **也失败**。即问题是 FSDP2 `MixedPrecisionPolicy
  (param_dtype=bf16)` × 非重入激活检查点在多 rank 下的已知冲突（记忆项：双节点 FSDP 第 4 条，解法
  `fsdp.precision_policy=none`），CP 只是把它暴露成 attention backward 的 dtype 不一致。
  无检查点的 cp=2 在 81 帧上 OOM（44 GB），所以检查点必须开。
  处置：cp=2 与单卡基线都改用 `precision_policy=none`（trainable 组决定 dtype，即原生 FP32 LoRA
  路径）重跑，两者同配置可比；`actor` 策略在多 rank + 检查点下的修复另立项。
- 2026-09-15 02:40：**C 门根因（第二层）**：`precision_policy=none` 下同样在 attention backward
  报 q/k fp32、v bf16。原因：该策略的 replay 前向跑在外层 autocast(bf16) 里，norm_q/norm_k 在
  autocast 下升 fp32，v 投影是 bf16；前向的 SDPA 受 autocast 转型所以通过，而 CP 的注意力是自定义
  autograd.Function，backward 在 autocast 之外用**保存的原始张量**重算 SDPA，拿到混合 dtype。
  2 卡最小复现没有外层 autocast 所以没复现。修法：VRL 在启用 CP 时给 native attention 的
  forward op 加转型垫片（保存前把 q/k/v 统一到 autocast dtype），前向数值不变、backward 看到与
  前向一致的 dtype（`vrl/trainers/context_parallel.py`）。单卡基线（none）epoch 0：loss −9.5e-5、
  reward −4.5297、parity 0.002158、grad_norm 7.08e-4。
- 2026-09-15 03:40：**C 门通过（epoch 0）**：cp=2（GPU 2-3，`precision_policy=none`，attention dtype
  垫片）与单卡基线（同配置、同种子）逐指标对比（full precision）：loss −9.5033e-5 vs −9.50328e-5
  （rel 2.7e-6）、reward_mean/std 完全相同、parity 0.00215822 相同、grad_norm 7.07975e-4 vs
  7.08056e-4（rel 1.1e-4，bf16 量级）、clip 0。即 token 分片 + loss×cp 的梯度与单卡等价。
  时间：每次 24 视频生成 325 s（leader 单独生成，与基线相同——follower 引擎空转是已知待办）。
  epoch 1 进行中；基线 epoch 1：loss 1.50e-4、reward −5.6918、parity 0.001893、grad_norm 4.87e-3。
- 2026-09-15 04:35：**C 门通过（两个 update）**：cp=2（`none`）epoch 1：loss 1.389e-4、reward −5.670、
  parity 0.001887、grad_norm 4.874e-3；基线 epoch 1：1.50e-4 / −5.692 / 0.001893 / 4.866e-3——epoch 1 的
  差异来自 update 0 后权重的 bf16 级非确定性（两次单卡 run 之间也有同量级差异），epoch 0 逐位一致。
  两项决策按用户指示照参照系统的形状定：(1) CP 预设默认 `precision_policy=none`
  （可训练参数 fp32 master、bf16 计算走 autocast）；(2) CP 组内每个 rank 各生成自己的 prompt 切片、
  组内 all_gather 并集（8f63044d），不再有空转引擎，组生成墙钟应减半（每 rank 3 组）。
  验证 run `cp2_allpeers` 已在 GPU 2-3 启动。附：2 卡复现里 actor 策略的 CheckpointError 来自复现脚本
  对 block 用了默认 `cast_forward_inputs=True`（RoPE 张量前向 bf16、重算 fp32），VRL 的
  `apply_fsdp` 对 block 显式关闭该转型，故 actor 策略在 VRL 内不受此影响。
- 2026-09-15 05:00：`cp2_allpeers`（两 rank 各生成 3 组）epoch 0：采集阶段 3 次生成/rank
  （约 16.5 分钟）而非 6 次（33 分钟），组生成墙钟如预期减半；parity 0.0020、clip 0、reward
  −1.877±5.23、grad_norm 9.8e-3、loss −3.0e-4。指标与单卡基线不再逐位可比：rank 1 的引擎用自己的
  driver RNG 抽采样种子，并集的样本集合与单 rank 生成的不同（合法的随机样本，只是不同的一组）。
  等价性已由 leader 布局的 C 门证明；本 run 验证的是全员生成布局能跑通且指标量级合理。
  若日后需要跨布局逐位可比，采样种子应由全局 prompt 序号派生而非各 rank 的 RNG 流。
