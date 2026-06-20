# SPRINT: DDP 2x1 首次真机验证 — bug 修复、wrap/unwrap 边界设计、perf 画像

状态：**双卡 2x1 DDP 已端到端跑通**（首次真机验证 SPRINT_symmetric_colocated_ddp.md
预告的"用户的 2x1 跑就是首次真实验证"）。两个首跑 bug 已修并验证；一个 principled
重构待定（见 §3）。分支 `fix/ddp-2x1-lora-nft-first-run`，修复 commit `ab7cab1`。

配方：`configs/experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward_ddp_2x1.yaml`
首验覆盖：`/sampling/video=480p_33f sampling.guidance_scale=7.0 rollout.rollout_batch_size=4
distributed.training.ddp.find_unused_parameters=true trainer.eval.enabled=false`。
torchrun rank0=nodeA(master 172.31.36.21) / rank1=nodeB，`NCCL_SOCKET_IFNAME=enp39s0`，
`--max-restarts=0`。**不起共享 Ray**——每 rank 本地 `ray.init()` 独占本机卡。

## 1. 验证结论（端到端正确）

- epoch 0：`loss=8.96 reward=-2.83 grad_norm=0.147`；epoch 1：`reward=-3.32 grad_norm=0.330`。
  grad_norm 非零且在变 → LoRA 真在更新，optimizer 真迈步。
- **跨节点 all-reduce 正确**：DDP backward 的 all-reduce 是 NCCL collective（屏障）。它顺利
  完成、无 hang → 两 rank 在每个 backward 汇合、梯度被跨节点平均（若没接好会 hang/报错）。
- **rank0-only IO 正确**：rank1 不写 `metrics.csv`（设计如此），无重复写。
- 两 rank 全程 GPU 100%、零错误，稳定推进多个 epoch。

## 2. 修复的两个首跑 bug（ws=1 CPU 单测测不到，真机 ws=2 才暴露）

### Bug A — LoRA state_dict gather（`vrl/trainers/strategy.py` DDPStrategy）
`_unwrapped_full_state` 原走 FSDP 的 `gather_full_state_dict`（DCP
`get_model_state_dict(full_state_dict=True)`）。在 **world_size>1** 时该 API 走分布式
all-gather，对**全量复制（非分片）**的 DDP 模块会**丢掉 PEFT LoRA key** →
`select_trainable_state()` 报每个 `lora_A/lora_B` "missing" → 首次 weight sync 崩。
ws=1 时 gather 是 no-op 所以 CPU 单测过了。
**修复**：DDP 每 rank 全量复制，unwrap 后直接 `inner.state_dict()`（与
`inner.named_parameters()` 同 key 空间），cpu-offload 保持 rollout payload 契约。

### Bug B — DDP wrapper 不透传 PEFT 方法（`vrl/algorithms/diffusion_nft.py`）
NFT 一步内对 transformer 做 3 次 forward：previous-policy（adapter='previous', no_grad）、
current（adapter='default', **grad**）、reference（adapters disabled, no_grad）。DDP 把
transformer 包成 `DDP(PeftModel)`，而 `getattr(DDP(...), 'set_adapter')` → None →
`RuntimeError: requires set_adapter('previous')`。
**修复**：previous/reference 分支的 PEFT 方法从 `unwrap_compile_and_ddp(transformer)`
取（DDP 包的是同一对象，切 adapter 对 wrapped 视图也可见）；grad forward 仍走 wrapper。
另：`find_unused_parameters=true`——'previous' adapter 参数 requires_grad 但走 no_grad
不拿梯度，reducer 默认会报缺梯度。

## 3. wrap/unwrap 边界设计（principled，待实现）

**核心矛盾**：同一个 transformer handle 既要"包着"做 grad forward（DDP hook 必须触发），
又要"剥开"做控制操作（set_adapter / state_dict）。**不是"包更多"——包更多会坏 grad
forward。** 真问题是边界要**集中、一致**。

代码库已有两个正确的集中边界：
- **Strategy = state 操作边界**：FSDP/DDP 各自拥有 gather/plain-state 逻辑。Bug A 的修复
  落在这里，**位置正确、是根因解法、保留**。
- **Model = adapter 控制边界**：`base.py:169 disable_adapter()`（context manager）、
  cosmos `sync_previous_policy_adapter()` 已是模型方法。

漏的是：**NFT 算法绕过模型**，直接 `getattr(model.transformer, 'set_adapter')`。Bug B 的修复
在算法层加 unwrap（**能跑、且因为在 `vrl/algorithms/` 所以已对所有 NFT 模型通用**），但与
已有封装不一致。

**三层划分（该放哪）**：
- 通用机制（unwrap + 激活任意名字的 adapter）→ **base diffusion model**，与 `disable_adapter()`
  并列，一处覆盖所有模型 + 所有 wrapper（DDP/FSDP/compile）。
  ```python
  # base.py，泛化、NFT-无关：
  def activate_adapter(self, name: str) -> AbstractContextManager[None]: ...
  ```
- adapter 名字 'previous' / 何时切 → **算法**（NFT 语义）：
  ```python
  with model.activate_adapter("previous"), torch.no_grad():
      prev = model.transformer(**inputs)[0].detach()
  with model.disable_adapter(), torch.no_grad():
      ref  = model.transformer(**inputs)[0].detach()
  ```
- adapter 的**创建**（`add_adapter('previous')` + `sync_previous_policy_adapter`）→ **各模型**
  专属配置，保持现状。

**关键：放 base 不放 Cosmos**——机制对所有 LoRA 扩散模型一致；放 Cosmos-only 会强制 Wan
等重复，且把已通用的算法层修复退化成专属。`'previous'` 这个名字是 NFT 概念，不该污染 base，
所以 base 给的是"激活任意名字 adapter"的通用动作，名字由算法传。

落地：base 加 `activate_adapter` + 算法改用它 + 删 `diffusion_nft.py` 里临时的 `_adapter_host`
函数；跑 `tests/trainers/test_ddp.py` + NFT 相关 CPU 单测，下一轮 DDP 起跑回归验证。

## 4. Perf 画像（2x1, 480p/33f, rbs=4）

稳态 **~9:50/组**，4 组/epoch → **~40 min/epoch**，从头一致**无恶化**（早期更短的间隔是启动期
模型加载重叠造成的误读）。每组 ~600s 构成：
- 模型 on_demand 重载（cosmos + reward 各 1 次/组，NVMe）：~20-30s，可忽略
- **生成 8 视频（480p/33f, 20 步, CFG 2×）：~4-5 min** ← 大头
- reward 打分（Qwen2-VL ×8）：~30-60s
- **DDP 训练步（NFT 每样本 3 次 forward + backward + cross-node all-reduce + torch.compile）：~3-5 min** ← 大头

**慢是 compute 主导，不是 IO**：
- rollout 训练轨迹（latents/logprobs）走 Ray **内存**对象库，**无 disk spill**（已查 `/tmp/ray`
  无 spill 文件）。
- 唯一落盘是 mp4 artifacts（共 **4.1M**，可忽略）+ checkpoint（每 2 epoch）。
- ⚠️ `outputs/` 在慢的 EBS root 盘（非 NVMe）——当前数据小不影响；若 checkpoint 频繁可考虑
  指到 `/mnt/nvme`。

提速杠杆（按收益）：降分辨率/步数/CFG（生成 + 训练 forward 都是 transformer 调用，最大头）>
减少 NFT 每步 forward 次数 > resident rollout worker（`rollout.gpu_pool: trainer` +
`memory_fraction` 省每组 ~20-30s 重载，但拆分显存）。

## 5. 待定

- [ ] §3 的 `activate_adapter` 三层重构是否实现（当前算法层修复已能跑，重构是"通用解一致性"升级）
- [ ] 双节点自愈 wrapper（torchrun 跨节点 + resume-from-checkpoint），用于长时无人值守
- [ ] 是否放大到论文 batch（rbs↑）——受 §4 的 compute 成本约束
