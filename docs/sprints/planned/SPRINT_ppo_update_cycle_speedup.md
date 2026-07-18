# SPRINT: PPO 训练相位提速（GRPO update cycle speedup, cosmos V2W 93f 单卡）

状态：proposed / planned（2026-07-10）。

> 来源：本篇的每个论断都核对了三处——①本仓实测（2026-07-10 EMA-off 480p_93f LoRA 曲线 run
> 的日志与 resolved config）；②NVIDIA cosmos-rl（`~/Desktop/cosmos-rl` @ 5b9fdba,
> `cosmos_rl/policy/{trainer,model,config}/wfm/`，同一个 Predict2 模型家族的官方 RL 栈）；
> ③THUDM slime（`~/Desktop/slime`，Megatron+SGLang 的 LLM RL 框架）。
> 外部引用均带 file:line，读时若行号漂移以符号名为准。

## 0. 一句话

一次 GRPO 更新 114 分钟里 **75% 是 PPO 反向相位**（~85 min），由三个乘数堆出：
`2 ppo_epochs × 16 samples × 5 timesteps = 160 次梯度单元`，每单元又是
**CFG 双前向 × 全量重算**（≈8 个前向当量）。最大的单个杠杆是 cosmos-rl 给的：
**NVIDIA 自己的 Predict2 RL 全部 shipped config 都用 `guidance = 0.0`**——RL 阶段
根本不开 CFG。若质量门通过，单此一项 rollout 和 PPO 同时砍半（114 → ~57 min）；
全部杠杆叠加单卡可到 ~35 min/更新，结构解仍是多卡（cosmos-rl 同工作负载的
最低配置是 **8 GPU FSDP2**）。

## 1. 实测账本（2026-07-10 run，日志核对）

运行：cosmos predict2 2B V2W，480p × 93f，LoRA r32，1×RTX 5090 32GB。
config：`outputs/cosmos_pred2_droid_lora_480p_curve/resolved_config.yaml`（wm-infra 侧）。

| 相位 | 实测 | 构成 |
|---|---|---|
| rollout | 1719.6 s（日志 `generation wall chunks=16`） | 16 samples × 15 steps × 2 (CFG) = 480 次前向 → **3.58 s/前向** |
| reward | ~31 s | DINO + RAFT，16 帧抽帧，忽略不计 |
| PPO | ~85 min（10:00:24 → 11:25 metrics 落盘） | 见下 |
| 合计 | **~114 min/更新** | eval 每 2 epoch 另加 ~15 min |

PPO 相位的乘法：

```
2 (actor.ppo_epochs) × 16 (samples) × 5 (timestep_fraction=0.5 × 10 步 SDE 窗口)
  = 160 次梯度单元
每单元 = 2 前向 (CFG cond+uncond, guidance_scale=7)
       + 全量重算 (actor.gradient_checkpointing=true → full)
       + 反向 (~2× 前向 FLOPs)
       ≈ 8 前向当量 × 3.58 s ≈ 28.6 s
160 × 28.6 s ≈ 76 min 预测 vs 85 min 实测 —— 账目吻合（余量是 optimizer/搬运/SFT 分支）
```

关键代码锚点（本仓）：
- timestep 子集与逐 timestep 反向循环：`vrl/trainers/online/trainer.py:759-934`
- CFG 双前向（cond+uncond 都在梯度内）：`vrl/models/diffusion/common/backbone.py:127-145`
- `do_cfg = guidance_scale > 1.0` 的现成开关：`vrl/models/diffusion/cosmos/predict2/model.py:241-242,284-285`
- 重算三档 off|full|selective：`vrl/config/schema.py:549`，applier
  `vrl/trainers/activation_checkpointing.py:128-169`（selective 不被模块支持时显式告警回退 full）
- 改造前生成/训练共用一个 chunk 旋钮：`vrl/trainers/core/types.py` 的
  `rollout.samples_per_chunk`；现已拆为 generation declaration 与独立 replay runtime verdict。

## 2. 外部对照：cosmos-rl 与 slime 怎么做

### 2.1 cosmos-rl（同模型家族，最强对照）

| 维度 | cosmos-rl 的做法 | 证据 | 对我们的含义 |
|---|---|---|---|
| RL 阶段 CFG | **guidance = 0.0，所有 shipped config**（默认值 + 14b/2b-720 显式写 0；2b-480 不写=默认 0）。他们代码里 uncond 前向仍在跑但数学上乘零，是他们没守卫的死算力 | `cosmos_rl/policy/config/wfm/__init__.py:162`；`t2v_model.py:971` x0_fn | **P0 杠杆**。我们 `do_cfg = guidance_scale > 1.0` 现成守卫，设 `sampling.guidance_scale=0` 即可两相位同时砍掉 uncond 前向 |
| timestep 子集 | 显式索引表 `rl.train_on`：2b-480 练 8/10（连续前缀），14b/2b-720 练 **10/20 隔一取一（strided）** | `config/wfm/__init__.py:152-153`；`configs/.../14b-720-...ddrl.toml:74,76` | 我们 `timestep_fraction=0.5`（5/10 strided）已在他们旗舰配比上；再降是超出 prior art 的实验 |
| 梯度 epoch 数 | **mu=1，不存在 ppo_epochs**；trajectory 只消费一遍，`len(train_on)` 个 timestep 折进一次 grad-accum；clip（1e-4）在 on-policy 下近乎不动作 | `wfm_trainer.py:662-677,627-629`；`config/wfm/__init__.py:154,172` | 与我们 `ppo_epochs=2` 的分歧点，见 §4 讨论 |
| 重算 | **SAC 是默认**（block_wise 每块全重算），且备有本家族的 op-selective 策略（`predict2_2b_720`：保 flash-attn 输出+部分 matmul，重算其余）显存富余时放松 | `minimal_v4_dit.py:147-154,1775-1801`；`config/wfm/__init__.py:52` | **P1 杠杆的同家族 prior art**：显存允许时从 full 放松到 selective 是他们的既定玩法 |
| batch 几何 | 每卡 batch=1 个 93 帧视频、每前向 1 个 timestep；GRPO 组靠 **num_rollout=8 个 DP 实例**跨卡凑 | `wfm_trainer.py:452`；`t2v_model.py:417,460`；`2b-480-grpo.toml:178` | 单卡内不 batch 样本——支持我们训练回放 chunk=1 不动，加速靠 DP 而不是 intra-GPU batch |
| 并行 | FSDP2 (`fully_shard`) + CP；**2b-480（93f）最低 8 GPU**（fsdp_shard_size=8）；14b-720 ≈ 1024 GPU | `utils/wfm/distributed.py:474-496`；`minimal_v4_dit.py:1803-1813` | 我们单张 5090 在他们地板的 1/8——**P4 的定量依据** |
| torch.compile | DiT 默认关（`use_torch_compile=False`），没有任何视频 config 打开 | `config/wfm/__init__.py:227`；`t2v_model.py:387-409` | 与本仓 compile×grad-ckpt 互斥守卫一致：此路不通不是我们独有 |
| 其他 | FusedAdam 默认；旗舰 kl_beta=0（免 ref 前向）；reward 用独立异步服务（文档明说 reward 是他们要藏的开销）；无 fp8、无 RL 降帧降分辨率 | `config/wfm/__init__.py:251,173`；`docs/wfm/overview.rst:333-341` | 我们 kl_coef=0、reward inline+sleep_offload 已同构 |

### 2.2 slime（LLM RL，反向的世界观）

slime 明说自己是 **rollout-bound**（"inference latency cannot be reduced by adding more
GPUs"，`docs/en/blogs/release_v0.1.0.md:35`），训练相位是固定的 on-policy 单遍——和我们
backward-bound(75%) 的处境**正好相反**，所以它的调优优先级不可照搬，可搬的是机制：

- **没有 ppo_epochs**：`--num-steps-per-rollout` 只是把一批 rollout 切成 N 个 optimizer
  step，不重复消费数据（`slime/utils/arguments.py:588-600`）。off-policy 靠权重滞后 +
  TIS 截断重要性采样（`loss.py:744-764`），不靠多 epoch。
- **动态 microbatch**：按 token 预算 first-fit 装箱 + DP 组内 all-reduce MAX 对齐
  grad-accum 数 + Karmarkar-Karp 均衡（`data.py:347-376`）。视频定形无变长序列，
  几何不可搬；"按算力预算而非样本数定 microbatch"的思想对应我们的 chunk 解耦（P2）。
- **零优势组过滤**（DAPO 式，整组 reward 同值就不进训练 batch，
  `filter_hub/dynamic_sampling_filters.py:9-15`）——本仓 `actor.drop_zero_advantage=true`
  已等价在做。
- **重算全开**（所有 shipped config `--recompute-granularity full`）——因为他们不在乎
  backward FLOPs；我们 backward-bound 应反着做（→P1）。
- 训练侧无 torch.compile；fp8 仅 LLM/TE 路径。

**两个外部栈都没有多 epoch 重放、都不在训练相位开 CFG**——我们的 PPO 相位贵，一半是
自己的两个选择（ppo_epochs=2、guidance=7）叠出来的。这不自动等于选错（见 §4），但
说明这两个选择必须各自用实验挣回门票。

## 3. 工作项（按性价比排序）

### P0 — RL 阶段 guidance=0（预期 ~2×，config-only，带质量 KILL-RISK 门）

改 `sampling.guidance_scale: 7.0 → 0.0`（`do_cfg` 随之为 false，uncond 前向在
`backbone.py:146-149` 被跳过；rollout 与训练回放同一条路，old_log_prob 契约自动一致）。
预期:rollout 480→240 前向（28.7→~14 min），PPO 每单元 8→4 前向当量（85→~43 min），
**114 → ~57 min/更新**。

**KILL-RISK 门（先跑再信）**：cosmos-rl 的 guidance=0 是在他们自己的训练栈+权重上成立
的；我们的 diffusers V2W 权重在 480p93f 无 CFG 下可能画质崩（参考 240p garbage 教训:
形状/引导变化必须眼看 mp4）。门：固定 4-prompt 网格各出 2 条 mp4，肉眼过 + reward 均值
不塌（对比 baseline 0.68 量级），过了才允许起长跑。**注意这是换策略**：与 guidance=7 的
旧曲线不可比，新长跑要重打 baseline。

### P1 — ~~`gradient_checkpointing: selective`~~ **已测,93f 单卡死刑**（2026-07-10 探针）

在真 93f 形状（潜变量 24×60×104,37,440 token,LoRA r32 冻结基座,单分支,
空卡 31.8GB 可用）上实测三档:

```
     ckpt |   fwd ms |  step ms |   bwd ms | bwd/fwd | peak GB
      off | OOM
     full |     2824 |    10993 |     8169 |    2.89 |   14.20
selective | OOM
```

- **selective 和 off 都 OOM**——即使整卡空闲、模型只占 ~4GB。93f 的激活体量让 SAC
  想保的 matmul/attn 输出（37k token × 28 块）单独就超过 27GB。SD3.5-1024² 的
  "拿回 2/3 重算税"结论**不迁移**到 93f;重算税（≈1.0 个前向当量,占单元 26%）
  在 32GB 卡上被显存**结构性锁死**。
- **compile 连带死刑**:compile 需要 ckpt off(互斥守卫),off 都 OOM → 93f 单卡
  上训练侧 compile 无入场资格,实测确认(此前只是推断)。
- bwd/fwd = 2.89 ≈ 理论值(重算 1.0 + 反向 ~1.9)→ backward 是算力 bound 不是
  launch-bound,**加大 batch 也救不了 backward**(与 cosmos-rl 单卡 batch=1 一致)。
- 本项对 image / 短视频形状仍然有效(SD3.5 实测在案);93f 视频上从叠加账里移除。
- 探针:scratchpad `cosmos_backward_probe.py`(一次性,数字已录此处,不进仓)。

**93f 单卡上 backward 内核仅存的杠杆:attention backward。** 37k token 下
attention ≈ 2/3 FLOPs(28 层 × 4×seq²×d 对 2×2e9×seq),SDPA → flash-attn
专用 backward 是唯一与 ckpt 正交的内核升级,预期个位数到十几个点,装包后用
本探针 A/B 即可定量(flash-attn 未安装,sm_120 支持需先验证)。

### P2 — 生成 chunk 与训练回放 chunk 解耦（rollout 再 -30%，小代码改动）

**实现状态（2026-07-10）**：代码与 CPU 合约测试已落地；真实 Cosmos Predict2 2B
`480p × 93f` 三次 update GPU 门仍为 **PENDING**，因此 P2 目前不是 DONE。

改造前 `rollout.samples_per_chunk` 一个旋钮同时喂生成端拼 batch 和训练回放拼 batch
（`TrainerConfig.samples_per_chunk` → `OnlineTrainer` 两条 backward loop）。生成端 1→4
在本机本模型实测 **1.40×**
（139.6→99.5 s/chunk,峰值 25.3GB 放得下,old==fresh logprob 位精确——2026-06-29 P1 记录）,
但训练回放 chunk=4 在 93f 全量重算下必 OOM,所以现在整体只能卡在 1。
拆成 `samples_per_chunk`（生成）+ `actor.replay_samples_per_chunk`（训练）。generation 保留
Ray 启动时 forward-only `auto`；replay 只接受显式非负整数，默认安全值为 1，且不再继承
generation。`0` 明确表示不切 sample axis、一次回放完整 prompt group。

canonical 93f recipe 固定为 generation 4 / replay 1：前者吃到实测 batching 收益，后者使用
2026-07-10 full-checkpointed backward 的显存地板。训练循环不再内嵌 replay OOM probe、optimizer
ready 状态、headroom policy 或 family capability。若未来确实需要 replay 自动校准，应做成训练
外部的一次性工具：每个 candidate 启动全新进程；DDP/FSDP 下任一 rank OOM 就销毁整个 process
group，最后把全局安全值写回生产配置的显式整数。生产 `OnlineTrainer` 不承担 OOM recovery。

prior art:cosmos-rl 单卡内也不 batch 训练样本（§2.1 batch 几何行）,加速靠生成端与 DP。
注意:1.40× 是 guidance=7 时量的,P0 落地后前向变短,batching 增益需复测（预期仍为正,
launch/调度开销占比反而更大）。

### P3 —（可选实验）`actor.timestep_fraction: 0.5 → 0.3` + `timestep_selection: random`

PPO 线性再 -40%。**超出 prior art**（cosmos-rl 旗舰是 0.5 strided,2b-480 甚至 0.8）,
所以定位是实验不是默认:random 选择（DanceGRPO,`trainer.py:770-782`）让各更新覆盖不同
timestep,门是"曲线仍在动"（eval 网格趋势不塌）。若 P0 过门,优先吃 P0 的 2×,
本项仅在还嫌慢时叠加。

### P4 — 结构解:多卡 + 稳定代码树（唯一能把"天级"变"小时级"的）

- cosmos-rl 对**同一工作负载**（2b, 480p, 93 帧, batch=1/卡）的最低 shipped 配置是
  **8×GPU FSDP2**（fsdp_shard_size=8, `2b-480-grpo.toml`）;我们在 1/8 地板下单卡硬扛。
  GRPO 组按样本切天然 DP:2 卡近似砍半（P0-P2 后 ~57→~30 min/更新量级）。
  本仓 online DDP/FSDP 已走 per-rank-local symmetric-colocated 路径；显式 replay chunk 通过
  `_balanced_training_sample_chunks` 对齐各 rank collective slot。P2 不把 distributed OOM retry
  混进生产循环，边界见上。
- **运行必须从稳定树起**（`~/Desktop/VRL`,或容器）:2026-07-10 的 run 死因不是缺数据,
  是 wm-infra 树被并行重构——Ray worker 按周期 kill+重启并**从磁盘重新 import**,
  epoch-1 eval 的新 worker 载入了半成品代码（driver 侧 `_resolve_reference_artifacts`
  尚未存在于 driver 的 09:16 内存态,新 worker 却按新契约期望绝对路径）→
  `ENOENT: video_world/references/droid_001182_first.png`（文件其实存在）。
  supervisor 重启修不了这个,换稳定树才能修。

### 预期叠加账（单卡,P3 不计）

| 配置 | rollout | PPO | 更新周期 |
|---|---|---|---|
| 现状 | 28.7 min | ~85 min | ~114 min |
| +P0 guidance=0 | ~14.3 | ~43 | **~57** |
| +P2 chunk 解耦 | ~10 | ~43 | **~53**（≈2.2×） |
| +flash-attn bwd（待量） | ~9-10 | ~38-41 | ~48-51 |
| 再 +P4 两卡 | — | — | **~27 min 量级** |

（P1 selective 已被 93f 实测 OOM 移出叠加账,见上;大卡/多卡上 SAC 与 compile
双双复活,是 P4 的隐藏收益之一。）

## 4. 保持不动的（及为什么,免得下次再议）

- **`ppo_epochs=2` 不降**。prior art 确实两家都是单遍（cosmos-rl mu=1、slime 无此旋钮）,
  但:①cosmos-rl 的 mu=1 靠 8 个 DP 实例凑组、单步有效样本量远大于我们;②本仓 flux
  四算法验证的定论是 ppo_epochs=1 时 clip/trust-region/guard 全部恒等于 0（平曲线根因）,
  cosmos+kling 无学习那次也归因于单遍小步;③trainer 有守卫直接拒绝
  strict_on_policy+ppo_epochs≤1（`trainer.py:552-558`）。开放问题留档:若多卡后单步样本
  量上来,可以重开"mu=1+大组"对"mu=2+小组"的 A/B;单卡阶段不动。
- **torch.compile 不进本 sprint**。compile×grad-ckpt 互斥（守卫
  `vrl/config/validation.py:344-350`）,93f 不开重算必 OOM;cosmos-rl 的 DiT 默认也不
  compile,slime 训练侧无 compile。
- **fp8 不进本 sprint**。cosmos-rl 的 WFM 路径无 fp8;本仓 blockwise-fp8×compile 有既往
  阴性记录,且量化会污染 old_log_prob(verl 规则)。
- **训练回放去 CFG 而 rollout 保 CFG**——数学上直接破坏 ratio,禁止;P0 是两侧一致地关。
- **回放侧 uncond 守卫**:cosmos-rl 在 guidance=0 时仍空跑 uncond 前向(他们的死算力,
  §2.1),本仓 `do_cfg = guidance_scale > 1.0` 已天然守卫,无事可做,记一笔防止有人"对齐"回去。
- **零优势过滤 / KL-free / 异步 reward**:`drop_zero_advantage=true`、`kl_coef=0`、
  reward inline+sleep_offload 均已与两家 prior art 同构,不重复建设。

## 5. 验证顺序

1. ~~P1 显存门~~ 已跑(2026-07-10 backward 探针):selective/off 双 OOM,93f 单卡
   锁定 full,compile 连带出局。无需再验。
2. P0 质量门:guidance=0 固定网格 8 条 mp4 肉眼 + reward 均值(半天,不占长跑)。
3. P2 CPU 合约（已过）：chunk 解耦不碰 old_log_prob；replay 默认 1、允许显式 0/正整数、拒绝
   `auto`/负数；generation auto 不传播到 replay；DDP/FSDP 继续消费显式 replay 整数；generation
   auto 通过公共 launch-input 路径到达 Ray runtime，不再被 executor 的 `int()` 提前截断。
4. P2 真实 GPU PASS gate（仍待跑）：使用 canonical Cosmos Predict2 2B 480p recipe、
   `832×480×93f`、LoRA r32、generation chunk 4、replay chunk 1、trajectory CPU，在空闲 RTX 5090
   32GB 的 fresh process 中至少完成 3 个 optimizer updates：
   - 三次 update 都固定使用 replay chunk 1，并完整消费两个 PPO epochs、全部合法样本和配置
     选中的 timesteps，完成 optimizer step、weight sync 与 metrics 落盘；
   - 三次 update 均无 CUDA OOM/nonfinite，first-step parity ≤ 0.01，`grad_norm > 0`，
     global step/metrics 连续且 checkpoint 可读；
   - P2 的预注册性能收益只来自 generation 1→4；replay chunk 固定 1，不宣称 replay batching
     speedup。

   空闲 32GB 卡上的 canonical gate 命令（不 resume；关闭 fixed eval，只验证三次真实训练循环）：

   ```bash
   CUDA_VISIBLE_DEVICES=0 \
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
     vrl-train \
       --config experiment/cosmos_predict2/online_grpo_droid_lora_480p_curve \
       trainer.total_epochs=3 trainer.save_freq=3 trainer.eval.enabled=false \
       trainer.output_dir=outputs/cosmos_pred2_replay1_480p93f_gate
   ```

5. (可选)flash-attn 安装 + sm_120 验证 + backward 探针 A/B 定量 attention bwd 收益。
6. 全叠加后重打 baseline,起新长跑;旧 guidance=7 曲线归档为对照,不续。

## 引用

- 本仓账本:`outputs/cosmos_pred2_droid_lora_480p_curve/{resolved_config.yaml,metrics.csv}`(wm-infra 侧)、
  run 日志 `/tmp/run-until-success.cosmos_fullcurve.1000.log`
- 本仓代码:`vrl/trainers/online/trainer.py`、`vrl/models/diffusion/common/backbone.py`、
  `vrl/trainers/activation_checkpointing.py`、`vrl/trainers/core/types.py`、`vrl/config/validation.py`
- cosmos-rl @5b9fdba:`cosmos_rl/policy/config/wfm/__init__.py`、
  `cosmos_rl/policy/trainer/wfm_trainer.py`、`cosmos_rl/policy/model/wfm/models/t2v_model.py`、
  `.../minimal_v4_dit.py`、`configs/cosmos-predict2-5/*.toml`、`docs/wfm/overview.rst`
- slime:`slime/utils/arguments.py`、`slime/backends/megatron_utils/{data,model,actor}.py`、
  `slime/utils/seqlen_balancing.py`、`slime/rollout/filter_hub/dynamic_sampling_filters.py`、
  `docs/en/blogs/release_v0.1.0.md`、`examples/fully_async/`
