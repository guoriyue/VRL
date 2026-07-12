# SPRINT: Multi-GPU Training

状态：parked / blocked-on-event（等待真实 multi-GPU 硬件 → Phase 6 的 torchrun 2-GPU FSDP2 SD3 OCR 真实运行）。前置 readiness sprint 已落地（schema TrainingSection.strategy Literal["single_process","fsdp"]、DistributedTrainingContext/resolve_training_context、Strategy/SingleProcessStrategy 接缝、collect/train step split，commits d19faa2/b0d7b57/0d1b046/f979cce/fea4ba9/e6facbd/001ab41/1e6dc24）。FSDP2 **strategy 层**（FSDPStrategy / fully_shard / DTensor export）也已落地（见下方 Implementation status），但多卡编排（Phase 4 rank-split collect/train、§6.5 torchrun↔Ray、Phase 6 真实 2-GPU 运行）尚未实现，strategy=fsdp 当前由 run_online_recipe 的 _require_supported_online_strategy 主动 fail-fast。

> **复核更新（2026-06-20）**：本 doc「无真实 2-GPU run」一句**已 stale**——一条 **DDP** 多卡路径其后独立落地并归档
> `done/`（`vrl/trainers/strategy.py:464 DDPStrategy`、`SPRINT_symmetric_colocated_ddp` + `SPRINT_ddp_2x1_first_run_findings`,
> 配方 `online_nft_kling_video_reward_ddp_2x1.yaml` 已端到端跑通真实 2×L40S）。`_require_supported_online_strategy`
> 现放行 `{single_process, ddp}`、仍 fail-fast `fsdp`（`vrl/scripts/common/online.py:75`）。**本 sprint 的 FSDP2 rank-split
> 多卡编排（Phase 4/6.5/6）仍未实现、仍卡硬件**，故保持 parked；只是「2-GPU 从未跑过」不再成立。
>
> **复核更新（2026-06-21）：Phase 4 + §6.5 编排已落地，gate 放行 `fsdp`。** 关键架构决定：
> **复用 DDP 那条对称 colocated 编排，不实现 §6 的非对称 rank0-collect/broadcast 设计。** 依据——
> `run_online_recipe` 的多 rank 脚手架（per-rank disjoint prompt 分片、`is_primary` gate 全部 IO、
> per-rank 本地 `ray.init()` + colocated rollout、跨 rank reward stats all-reduce）DDP 已做完并跑通真实 2×1；
> FSDP 与 DDP 仅差 `prepare_model`（wrap）与 state gather，两者皆已实现+测试；collective 匹配粒度
> （每次 `backward()`）两者相同。落地清单：
> 1. `FSDPStrategy.prepare_model` 现显式 `init_training_process_group`（建 PG + `set_device`，与 DDPStrategy 对称）——
>    `init_device_mesh` 虽会兜底建 PG 但不 set device。
> 2. `_require_supported_online_strategy` 放行 `{single_process, ddp, fsdp}`（`vrl/scripts/common/online.py`）。
> 3. **跨 rank skip-backward 一致性**（`vrl/trainers/online/trainer.py:_all_ranks_have_work`）：某 rank microbatch
>    全过滤跳 backward、另一 rank 不跳 → FSDP per-layer all-gather/reduce-scatter 错配 → NCCL 死锁。新增
>    `all_reduce(MIN)` 使「跳过」全 rank 一致（DDP 同样受益）。
> 4. 配方 `online_nft_kling_video_reward_fsdp_2x1.yaml` + `docs/training_examples/.../fsdp_2x1_launch.sh`
>    （`num_nodes=2/gpus_per_node=1`、EMA off、torch_compile off、无 resume——皆 §10 gate 所限）。
> 验证：CPU/gloo 单测 + gloo 2-rank skip 一致性单测 + **真实 L40S world_size=1 GPU 冒烟**（nccl PG / cuda DTensor /
> bf16 fwd-bwd / DTensor clip / cpu_offload gather 干净 keys / export-load round-trip）全过；config→FSDPStrategy
> 解析通过。**Phase 6 真实跨机 2-node run 已于 2026-06-21 跑通**（A=172.31.36.21 / B=172.31.32.107，2 epochs，
> reward 上升、3 个非空 checkpoint），现场又抓出并修掉 5 个仅在真机 NCCL 暴露的 bug——详见本文件顶部「Done
> (2026-06-21)」清单。注：本模型 LoRA 放得下单卡，cross-node FSDP 每层 all-gather 走网络比 DDP 慢；FSDP 的价值
> 在「放不下单卡 / full-param」场景。

The FSDP2 **strategy layer** (Phase 2 + the FSDP2 core of Phases 3/5) is now
implemented and unit-tested on a single CPU rank (gloo, `world_size=1`), where
`fully_shard` really shards, forward/backward runs, and
`get_model_state_dict(full_state_dict=True)` materializes plain full tensors:

- `vrl/trainers/fsdp.py` — process-group init/destroy, 1D `dp_shard` mesh,
  `MixedPrecisionPolicy` mapping, `unwrap_module` (compile + PEFT), `iter_blocks`
  (`_no_split_modules`), `apply_fsdp` (per-block + root `fully_shard`),
  `gather_full_state_dict` / `load_full_state_dict` (DTensor ↔ rank0 full).
- `vrl/trainers/strategy.py` — `FSDPStrategy` (prepare_model wrap / backward /
  DTensor-aware clip / export_trainable_state / export_rollout_state /
  load_trainable_state / barrier) + `build_strategy(cfg, context)` factory + the
  §10 config gates (fsdp + EMA, fsdp + optimizer resume → fail-fast).
- `vrl/trainers/online/trainer.py` — routes its model through
  `strategy.prepare_model` once (identity for single_process).
- `vrl/config/schema.py` + `configs/base/distributed/training_fsdp.yaml` — the
  `distributed.training.fsdp` knobs the strategy reads (mesh / precision_policy /
  reshard_after_forward).
- `tests/trainers/test_fsdp.py` — real CPU `fully_shard` round-trips, the §9
  rollout-key-space invariant (sharded export == single-process export), pure
  helpers, and the gates.

**Done (2026-06-21):** Phase 4 (online GRPO rank split — reused the symmetric
colocated path), §6.5 (torchrun↔Ray coordination via the per-rank-local Ray
model), AND **Phase 6 real cross-node 2×1 run** — `online_nft_kling_video_reward_fsdp_2x1`
ran 2 epochs on two L40S nodes (A=172.31.36.21 / B=172.31.32.107), reward improved
-3.97→-3.11 (r_kling -4.05→-3.46), grad_norm non-zero/changing (0.13→1.31),
rank0-only metrics, 3 non-empty 4.4G checkpoints (gathered full state incl. 1120
LoRA keys), clean finish. Five real-multi-rank bugs surfaced ONLY on the live
2-node run (every one invisible to the world_size=1 CPU tests — the §"validate on
real NCCL" lesson), each now fixed with a regression test:

1. **resources.py treated fsdp as asymmetric** (trainer must own all `world_size`
   GPUs, disjoint rollout) → added the symmetric-colocated branch (rollout.gpu_pool
   =trainer ⇒ per-rank-local 1-GPU resolution like ddp). `test_fsdp_colocate_resolves_per_rank_local_single_gpu`.
2. **`gather_full_state_dict` dropped LoRA on non-rank0**: `cpu_offload=True` makes
   `get_model_state_dict(full_state_dict=True)` rank0-only (empty elsewhere) → every
   rank's colocated rollout sync raised "missing trainable". Fixed to `cpu_offload=False`
   (full on every rank). `test_fsdp_gather_distributed.py`.
3. **skip-backward divergence** would deadlock FSDP collectives → `_all_ranks_have_work`
   all-reduce(MIN). `test_skip_backward_agreement_distributed.py`.
4. **FSDP MixedPrecisionPolicy(bf16) × activation checkpointing**: the GC recompute
   bypasses FSDP's pre-forward param-cast hook → forward bf16 vs recompute fp32
   CheckpointError. Config uses `precision_policy=none` (fp32 sharded master + bf16
   via the trainer's autocast, consistent across forward/recompute). GC is required
   (no-GC OOMs a 44GB L40S at 480p/33f).
5. **checkpoint gather deadlock**: the trainable-state gather is a collective but the
   recipe gated the whole save to rank0 → rank0 hung at the all-gather. Fixed: the
   gather runs on every rank, only rank0 writes; sharded LoRA `save_pretrained` is
   skipped under fsdp (payload carries the gathered state). `test_save_training_checkpoint_non_primary_gathers_but_writes_nothing`.

**Done (2026-06-22) — FULL-PARAM FSDP validated:** the case FSDP actually earns its
keep. `online_grpo_ocr_fsdp_2x1_fullparam` (SD3.5 GRPO OCR, `model.use_lora=false`)
ran 2 epochs on the same 2×1 rig: the WHOLE ~2.47B transformer is sharded (ZeRO-3),
grad_norm non-zero both epochs (0.11/0.13), r_ocr rose 0.45→0.56/0.75, and the
gathered full-param checkpoint (909 tensors, 2.47B params, 0 LoRA keys, optimizer
state present) loads. GRPO is required, not NFT: NFT's previous-policy snapshot is a
PEFT adapter with no full-param path. Two recipe-level (not FSDP) deltas vs the LoRA
config: `kl_coef=0` (the LoRA KL reference is "policy with adapter disabled"; no
adapter ⇒ would need a separate frozen copy) and `gradient_accumulation_steps=1`
(rbs=1 divisibility). Two findings:

6. Full-param checkpoints are ~8.8 GB each (full params + fp32 Adam). The first
   `checkpoint-final` truncated — root cause was DISK FULL (root fs at 95%), not a
   code bug; the in-loop `checkpoint-2` is the same complete epoch-2 artifact and
   loads fine. Mitigation: enough disk, fewer saves, or save model-only (no
   optimizer) under fsdp since resume is gated anyway.
7. Added a `strategy.barrier()` after every checkpoint save (online.py): the gather
   is collective but the rank0 write is solo and slow for an 8.8 GB payload, so a
   non-primary rank could otherwise return, hit shutdown, and let torchrun tear down
   rank0 mid-write. Defensive against that race (independent of the disk issue).

**Still NOT done (deferred, not blocking):** Phase 7 (offline DPO distributed);
DTensor-aware optimizer/EMA state and `resume_from` (still §10 fail-fast for fsdp);
keeping bf16 FSDP params *with* GC (would need the MP cast re-applied during
recompute — see bug 4); HF-format LoRA `save_pretrained` artifact under fsdp (the
torch.save payload already holds the full gathered state — see bug 5); bf16 NCCL
weight-sync perf (§10.5). For a model that fits on one card, DDP remains faster
(cross-node FSDP all-gathers params over the network each forward); FSDP's value is
the model-doesn't-fit / full-param case.

## 0. Core Decision

本 sprint 的目标是让训练侧支持真正的 multi-GPU training，而不是只把 rollout worker 分到多张卡。

**主路径是 FSDP2（torch-native `fully_shard` + DTensor），不是 Megatron。** 依据（详见 §10.5）：

- 你的 OOM 是"优化器/梯度/激活显存"问题、模型**前向放得下一张卡** —— 这正是 FSDP(ZeRO-3) 的本命，不是 TP 的场景。
- 模型是 diffusers `SD3Transformer2DModel` / Wan / Cosmos + PEFT LoRA。Megatron 不吃 diffusers/PEFT，要按它的并行层重写模型 —— 是模型移植，不是加 flag。
- 连 cosmos-rl 都是 **torch-native**（`reading/cosmos-rl.md:699`："no Megatron-LM engine ... torchtitan-style: DTensor TP plans + FSDP2 `fully_shard`"），diffusion 更是纯 FSDP2；它从整个 Megatron 只借了**一个 MoE kernel**（dense DiT 用不上）。

策略集合收敛到最小开关，**不预留其他 backend 空槽**（避免投机抽象）：

```text
single_process    # 当前单卡路径，行为不变
fsdp              # 新增：FSDP2，本 sprint 主体
```

- `megatron` **不进 schema**。只有"大 MoE / 最大规模 dense LLM"才值得，diffusion+LoRA 永远走 FSDP（§10.5）。

本 sprint 基于已经落地的 role-level resource resolver，但职责不同：

- `vrl/ray/resources.py` 和 `vrl/generation/ray/*`：决定 trainer / rollout 拿多少资源并创建 Ray placement。
- 本 sprint：决定 trainer 如何在这些资源上创建 rank、`fully_shard` model、shard batch、reduce-scatter 梯度、保存 DTensor checkpoint、向 rollout workers 推送 **unwrapped** 权重。

## 1. Current Code Reality

当前训练入口是 single-process：

```python
trainer = OnlineTrainer(
    model=sd3_5_model,
    ...
)
```

`OnlineTrainer` 直接从 `self.model.parameters()` 建 optimizer：

```python
trainable = [p for p in self.model.parameters() if p.requires_grad]
self._optimizer = _create_optimizer(trainable, self.config)
```

rollout weight sync **不是**直接推 `self.model.state_dict()`——`OnlineTrainer` 已经走一个显式的 trainable-state getter，并在缺失它时 fail-fast（`vrl/trainers/online/trainer.py:171-175`，注释明确："syncing model.state_dict() would send frozen modules"）：

```python
# TrainableStateGetter = Callable[[], dict[str, Any]]  (vrl/trainers/weight_sync.py:11)
sync_state_getter=build_trainable_state_sync_getter(bundle)  # online.py:155
# rollout schedule 内部用 weight_syncer + sync_state_getter 推 flatten 后的 trainable state
```

所以多 GPU 的诉求**不是**"干掉 `self.model.state_dict()`"（那行不存在），而是：让 `sync_state_getter` 返回的 state 在 FSDP wrap 之后仍然是 unwrapped、policy-facing 的 key space。

checkpoint 依赖 `RuntimeBundle.trainable_modules`：

```python
"trainable_modules": export_trainable_state(bundle)
```

所以 multi-GPU training 不能只是把 policy 外面套一层 FSDP。必须同时解决：

- trainer rank / local rank / world size。
- model wrapper 放在哪一层。
- rollout batch 如何广播 / 分片。
- checkpoint 如何 rank0-only 保存，FSDP 如何导出 full state。
- rollout workers 需要收到普通 `state_dict`，不能收到 FSDP shard。
- EMA、LoRA export、resume 必须继续正确。

## 2. Scope

本 sprint 覆盖：

- `torchrun` 启动的 single-node FSDP。
- 为 multi-node FSDP 保留 schema 和环境变量路径。
- rank0 负责 rollout collection、reward、eval、metrics、checkpoint。
- all ranks 负责 replay loss、backward、optimizer step。
- rank0 从 wrapped model 导出 rollout-compatible trainable state，并同步给 Ray rollout workers。

本 sprint 不覆盖：

- Megatron Tensor Parallel / Pipeline Parallel 真实实现。
- RayTrainGroup 主训练入口。
- ZeRO / DeepSpeed。
- 多节点 hostname/rack-aware placement。
- 让 rollout worker 也持有 FSDP/Megatron shard。
- 非 LoRA full-model 大 checkpoint 的高性能 shard 存储优化。

## 3. Target Config

新增 base config：

```text
configs/base/distributed/training_single_process.yaml
configs/base/distributed/training_fsdp.yaml
```

目标 schema：

```yaml
distributed:
  training:
    strategy: single_process   # single_process | fsdp
    launcher: none             # none | torchrun
    num_nodes: 1
    gpus_per_node: 1
    backend: nccl
    init_method: env

    fsdp:                          # FSDP2 (fully_shard / DTensor)
      mesh: ["dp_shard"]           # 1D 起步；多节点 HSDP ["dp_replicate","dp_shard"]
      precision_policy: actor      # → MixedPrecisionPolicy(param=bf16, reduce=fp32)，接 precision config
      reshard_after_forward: true  # = ZeRO-3；false 省通信、费显存
      activation_checkpointing: actor
      cpu_offload: false
      state_dict: full_rank0       # DTensor → rank0 full（dcp.get_model_state_dict, full_state_dict=True）
```

规则：

- `strategy=single_process` 时行为必须和当前 repo 一致。
- `strategy=fsdp` 时必须通过 `torchrun` 启动；`WORLD_SIZE` 缺失时 fail-fast。
- `strategy=megatron` **不接受**（schema 里没有这个值），传入即 fail-fast 指向 §10.5。
- `distributed.resources.trainer.num_gpus` 必须等于 `training.num_nodes * training.gpus_per_node`，否则启动时报错。

## 4. Training Strategy Abstraction

新增：

```text
vrl/trainers/distributed.py        # DistributedTrainingContext + torchrun env resolver
vrl/trainers/strategy.py
vrl/trainers/fsdp.py
tests/trainers/test_distributed_training.py
tests/trainers/test_strategy.py
```

> **更正（2026-06-14）**：`vrl/trainers/fsdp.py` 已被删除（commit `a34b815`），且它是 **FSDP1**（`FullyShardedDataParallel` / `transformer_auto_wrap_policy` / `FullStateDictConfig`）——**不要复活它**。`FSDPStrategy` 写 **FSDP2**（`torch.distributed.fsdp.fully_shard` + DTensor，逐 block）新实现。可借的"老 multi-GPU 逻辑"只有两处：
> 1. 老 fsdp.py 里 `init_device_mesh` 建 mesh 的形状（mesh 概念可借，wrapping 全改 FSDP2）。
> 2. **`vrl/trainers/data/samplers.py:DistributedKRepeatSampler`**（已存在但当前是 dead code）——它正是"K 个 sample/prompt、按 rank 切、`k` 必须整除 `num_replicas*batch_size`、**不破坏 GRPO group**"的分布式 sampler（`samplers.py:14-40`）。DPO 路径直接复用；GRPO 的 rank-split（§6）也参照它的 group-aware 切法。
>
> 布局上不新开 `vrl/distributed/` 顶层包。torchrun/FSDP 训练身份和 strategy 归属
> `vrl/trainers/`；Ray placement、Ray actor lifecycle 继续归属 `vrl/ray/`。FSDP2 applier
>（reach handle → 穿 PEFT → `fully_shard` blocks）可放 `vrl/trainers/fsdp.py` 重建。

目标接口：

```python
class TrainingStrategy(Protocol):
    context: DistributedTrainingContext

    def prepare_bundle(self, bundle: RuntimeBundle) -> RuntimeBundle:
        ...

    def prepare_optimizer(self, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
        ...

    def backward(self, loss: torch.Tensor) -> None:
        ...

    def clip_grad_norm(self, parameters: Iterable[torch.nn.Parameter], max_norm: float) -> float:
        ...

    def export_trainable_state(self, bundle: RuntimeBundle) -> dict[str, dict[str, Any]]:
        ...

    def load_trainable_state(self, bundle: RuntimeBundle, state: dict[str, Any]) -> None:
        ...

    def barrier(self) -> None:
        ...
```

`DistributedTrainingContext`：

```python
@dataclass(frozen=True, slots=True)
class DistributedTrainingContext:
    strategy: str
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    is_primary: bool
    device: torch.device
```

原则：

- train script 不直接读 `RANK` / `LOCAL_RANK`。
- trainer 不直接知道 FSDP 细节。
- checkpoint 和 rollout sync 通过 strategy 导出 unwrapped/full trainable state。

## 5. Model Wrapping Contract

不要 wrap 整个 policy。优先 wrap `RuntimeBundle.trainable_modules` 里的 trainable root。

**已核对的 seam（2026-06-14，path:line）—— applier 基本通用，只分两种 shape：**

```text
diffusion（sd3_5/wan/cosmos 统一）   handle = model.transformer
  vrl/models/diffusion/common/backbone.py:88   self.transformer = transformer
  vrl/models/diffusion/base.py:207             trainable_modules 属性
  vrl/models/loader.py:1-32                     _set_transformer / apply_lora_to_transformer
AR（janus_pro / nextstep_1）          handle = model.language_model
  vrl/models/ar/janus_pro/model.py:1100        self.language_model = LlamaForCausalLM(...)
  runtime.py                                   trainable_modules={"model": model}
```

要点：
- **block 不在 VRL 代码里**，在被包的库模型里（diffusers DiT 的 `transformer_blocks` / `_no_split_modules`，Llama 的 `model.layers`）。applier 走 `trainable_modules` 契约拿 handle，再 shard 它的 blocks（多数能从 `_no_split_modules` 自动派生）。
- **LoRA 之后 transformer 是 `PeftModel`**（`loader.py` `get_peft_model`），applier 必须**穿过 PEFT 包装**找到 base transformer 的 blocks 再 `fully_shard`；LoRA-before-shard 天然满足（build 时注入）。
- 所以**不是每家族手写 hook**，是一个 applier + 两种 handle（diffusion/AR），换 mesh 即从 8 卡扩到上千卡，模型代码不变。

> **直接照抄起点（cosmos-rl 源码，已核对）**：`~/Desktop/cosmos-rl/cosmos_rl/policy/model/diffusers/parallelize.py` —— 它和 VRL 的设计**一模一样**：
> - `:21-24` `from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy`（确认是 FSDP2）
> - `:43` `assert pp_size == 1`（diffusion 不要 PP/TP）
> - `:52` `apply_fsdp(model.transformer, ...)`（shard 的就是 `.transformer` handle —— VRL 同款）
> - `:88` `getattr(model, "_no_split_modules", None)`（block 列表从 `_no_split_modules` 派生 —— 确认我的设计）
> - `:94-110` per-block `fully_shard(module, reshard_after_forward=True)` → 最后 `fully_shard(model)`（root），外加 `_skip_layerwise_casting_patterns` 让敏感层走 high-precision MixedPrecisionPolicy
>
> VRL 的 `apply_fsdp` 就是这段 + "穿过 PeftModel 找 base transformer" 一步。

Diffusion families 当前已经适合：

```python
trainable_modules={"transformer": transformer}
```

需要新增统一 helper：

```text
vrl/models/trainable.py
```

目标能力：

- `iter_trainable_modules(bundle)`
- `replace_trainable_module(bundle, name, wrapped_module)`
- `unwrap_trainable_module(module)`
- `export_unwrapped_state_dict(module)`

Diffusion policy 需要显式支持替换：

- SD3: `policy._set_transformer(wrapped)`
- Wan diffusers: `policy._set_transformer(wrapped)`
- Cosmos: `policy._set_transformer(wrapped)`

AR family 需要先收窄 trainable contract：

- Janus-Pro 当前是 `trainable_modules={"model": model}`，后续还可以改成更明确的 language model LoRA root。
- NextStep-1 当前是 `trainable_modules={"model": model}`，后续还可以改成 language model / image head 的明确 roots。

第一阶段只要求 SD3 / Wan diffusers / Cosmos。Janus-Pro / NextStep-1 可以 fail-fast：

```text
distributed.training.strategy=fsdp is not supported for family=janus_pro until trainable_modules exposes explicit trainable roots.
```

## 6. Online GRPO Loop Changes

当前 `OnlineTrainer.step()` 既 collect 又 train。multi-GPU 下要拆成两层：

```text
rank0:
  collect rollout batches
  compute rewards
  compute global advantages
  broadcast training batch payload

all ranks:
  shard training payload
  replay forward/backward
  reduce-scatter through FSDP
  optimizer step

rank0:
  export trainable state
  push rollout weights
  log metrics/checkpoint/eval
```

新增方法：

```python
async def collect_training_batch(...)
async def train_on_rollout_batch(...)
```

`OnlineTrainer.step()` 在 single-process 下继续调用两者，保持旧行为。

`train_on_rollout_batch()` 必须保持 async，因为当前 per-timestep 训练内循环会
`await asyncio.sleep(0)`，让 continuous rollout producer 在同一个 asyncio loop 上继续
推进。拆分 collect/train 时不能把这条交织行为变成同步阻塞。

FSDP 训练 shard 规则：

- GRPO group 必须完整保留，不能把同一个 prompt 的 `n` 个 samples 分散后再各自算 advantage。
- advantage 在 rank0 统一算好后再分片。
- 分片单位是 rollout sample 或 microbatch，但不能破坏 prompt group。
- 如果 batch 不能被 world size 整除，先支持 padding + mask，不 silent drop。

## 6.5 torchrun training world ↔ Ray rollout cluster 协同（最深的未知）

这是本 sprint 唯一"未知中的未知"，必须在 Phase 4 之前定清楚：现有 rollout 是 Ray-based（`RayGenerationRuntime`，`vrl/generation/ray/runtime.py`），而 FSDP 训练侧是 `torchrun` 起的 N 个进程。两套并发模型要拉通：

- **谁持有 Ray client**：约定只有 rank0 连 Ray、提交 generate / 收 reward；非 rank0 在 collection 阶段进入 `strategy.barrier()` 等待 rank0 broadcast training payload。避免 N 个 rank 各自连 Ray 重复采样。
- **资源不打架**：`vrl/ray/resources.py` 现在按 role 分卡，但它不知道 `torchrun` 又 fork 了 `gpus_per_node` 个训练进程。必须明确：训练 ranks 占用的物理卡（`LOCAL_RANK` → device）与 Ray rollout worker 的 placement **不重叠**，或在 colocated 模式下显式 release。新增校验：`training.num_nodes * training.gpus_per_node` 的训练卡集合 ∩ rollout worker 卡集合 = ∅（除非 colocated 且声明 overlap）。
- **weight sync 时序**：rank0 train 完导出 state → push 给 Ray rollout worker → 其余 rank 在 `barrier()` 处等 rank0 完成 sync 再进入下一轮 collect，保证 policy version 单调且全 rank 看到同一版本。
- **退化保证**：`strategy=single_process` 时这一层完全短路，行为与当前 repo 一致。

DoD：新增一个 torchrun + Ray 的集成约定测试（mock Ray runtime），断言 (1) 只有 rank0 调 `runtime.generate` (2) 非 rank0 在 collect 阶段不触发采样 (3) 训练卡与 rollout 卡集合不重叠时启动通过、重叠且未声明 overlap 时 fail-fast。

## 7. Offline DPO Loop Changes

Wan DPO 是 offline trainer，应该单独接 distributed dataloader：

- `DistributedSampler`
- rank-local batch
- FSDP backward/step
- rank0-only metrics/checkpoint

这条路径比 online GRPO 简单，可以作为 FSDP 的第二个验证目标。

## 8. Checkpoint and Resume

修改：

```text
vrl/trainers/checkpointing.py
vrl/trainers/online/trainer.py
vrl/trainers/offline/dpo.py
```

要求：

- single-process checkpoint 格式保持兼容。
- **FSDP2 checkpoint 第一版用 DTensor → rank0 full**：`torch.distributed.checkpoint.state_dict.get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))`（或逐 param `DTensor.full_tensor()` gather 到 rank0），不做 sharded checkpoint。
- resume 时所有 ranks 必须加载同一份 trainable state（rank0 读 → broadcast，或各 rank `dcp` load）。
- optimizer state：`get_optimizer_state_dict(..., full_state_dict=True)` rank0 full；如实现复杂，FSDP optimizer resume 可先 fail-fast，但写入 DoD。
- EMA：FSDP2 + EMA 第一版 fail-fast，除非实现完整 DTensor gather/update。

> **这是本 sprint 最大的触点**：三处现在都假设完整 tensor，FSDP2 下它们拿到的是 DTensor 分片，必须先 materialize：
> `export_trainable_state`（`checkpointing.py:250`）、`OnlineTrainer.state_dict`（`trainer.py:938`）、`flatten_trainable_module_state`（`weight_sync.py:101`）。统一走 `get_model_state_dict(full_state_dict=True)` / `full_tensor()` 收成 full、再 `to_cpu`/导出。

rollout sync 的现状是 `build_trainable_state_sync_getter(bundle)`（`vrl/trainers/weight_sync.py:86`）返回一个 `Callable[[], dict]`，由 rollout schedule 内部推送。多 GPU 下要把这个 getter 换成 strategy-aware 版本：

```python
# 现状（single-process）：
sync_state_getter = build_trainable_state_sync_getter(bundle)
# 目标：getter 内部走 strategy，导出 unwrapped/full state（去掉 wrapper / FSDP internal key）
sync_state_getter = lambda: strategy.export_trainable_state(bundle)
```

关键不变量：getter 返回的 key space 必须是 rollout policy 能直接 `load_trainable_state()` 的（diffusion 要求 `transformer.*` prefix），**不能**泄漏 wrapper 的 `module.` 或 FSDP 的 shard key。

## 9. Rollout Weight Sync Contract

当前 Ray rollout worker 调用：

```python
policy.load_trainable_state(state_ref)
```

这个 contract 要保留，但 trainer 侧必须保证传过去的是 rollout policy 能加载的普通 unwrapped state。

Diffusion families 当前要求 `transformer.*` prefix：

```python
load_trainable_state only accepts trainable keys prefixed with "transformer."
```

因此 strategy 导出后要统一成 policy-facing key space，而不是 wrapper / FSDP internal key space。

新增测试：

- wrapped transformer 导出的 key 不包含 `module.` 泄漏。
- FSDP full state 导出的 key 能被 fresh rollout policy `load_trainable_state()` 加载。
- LoRA-only sync 仍然只推 trainable adapter state，不推 frozen checkpoint。

## 10. Megatron Boundary

`megatron` **不进 strategy schema**（§3）。传入 `strategy=megatron` 直接 fail-fast 指向 §10.5：

```text
distributed.training.strategy=megatron is not a supported value.
diffusion + LoRA uses FSDP2 (torch-native); see §10.5 for why Megatron is not the tool here.
```

下面记录"如果有一天真要 Megatron"（训大 MoE / 最大规模 dense LLM）才需要独立解决什么：

- model family 是否有 Megatron-compatible module。
- tensor/pipeline parallel size。
- Megatron optimizer and scheduler。
- Megatron checkpoint import/export。
- rollout worker 如何接收 TP/PP shard 或 rank0 gathered LoRA state。
- diffusion transformer 是否值得 Megatron 化，还是只支持 AR LLM-like trunk。

## 10.5 并行 / kernel 来源（torch-native，不是 Megatron）

这一节是"为什么 FSDP2、不要 Megatron"的存档，免得以后反复 re-litigate。

**领域已经把 Megatron 的精华 unbundle 进了 torch core + TransformerEngine：**

```text
FSDP/ZeRO-3        → torch.distributed.fsdp.fully_shard
DTensor 切分       → torch DTensor
1F1B 流水          → torch.distributed.pipelining（cosmos patch 的就是 torch 的）
dist-checkpoint    → torch.distributed.checkpoint
fused attention    → flash-attn / TransformerEngine
FP8 / fused norm / CP → TransformerEngine
MoE token dispatch → Megatron（唯一真正独立可搬，dense DiT 用不上）
```

所以 cosmos-rl（上千卡）才是 torch-native、从整个 Megatron 只 import 一个 `pad_routing_map`（MoE kernel，`reading/cosmos-rl.md:699,1232`）。

**kernel 来源表（你的 dense DiT 该去哪拿）：**

| 需求 | 来源（不是 Megatron） |
|---|---|
| attention | flash-attn（drop-in）|
| fused norm / FP8 | TransformerEngine（有 DiT 用法）|
| 长视频序列并行 | xDiT / Ulysses / TE-CP |
| 自研 kernel | moemoekit 的 Triton（native executor 线）|

**FSDP2 的搭档：NCCL weight sync（follow-on，不在 Phase 1）。**
LoRA 同步走现有 Ray object-store 推送已够（`weight_sync.py` 已用单次 `ray.put` 广播）；一旦上 **full-param FSDP2**，"Ray 推 CPU state_dict" 会变成每步多 GB 瓶颈、且 O(#rollout-actor) 放大 → 换 GPU↔GPU NCCL（cosmos `pynccl.py` 可整段 lift）。这是 §9 weight-sync contract 的性能升级，不改 contract。

**视频 DiT 的 scaling（FSDP 之后的下一层，按"分辨率×帧数"触发）：**

```text
2.5B–14B 图像/视频，参数显存       → FSDP2（层放得下；cosmos 14B 已验证纯 FSDP）
高清 / 长视频（激活/attention 爆）  → FSDP2 + CP/SP（序列并行），必要时 + TP
```

DiT 深而不宽，所以先碰到的是 **CP/SP（序列），不是经典 TP（权重）**；且 CP/SP 叠在 FSDP 之上，不替代。这一档单独排（见 `parked/` 视频 scaling，触发条件写成"激活占比超阈值"），本 sprint 不做。

## 11. Implementation Phases

### Phase 1: Config and Context

新增 distributed training config 和 context resolver。

完成条件：

- `single_process` 默认行为不变。
- `fsdp` 能从 `torchrun` 环境解析 rank/local_rank/world_size/device。
- `megatron` fail-fast。
- 配置校验覆盖 resources/training GPU 数一致性。

### Phase 2: Strategy Abstraction

新增 `SingleProcessStrategy`、`FSDPStrategy` skeleton。

完成条件：

- trainer 通过 strategy backward/clip/export/load。
- single-process tests 全部不变。
- fake wrapper unit test 能证明 wrapper key export 正确。

### Phase 3: Diffusion Family Wrapping

先支持 SD3 / Wan diffusers / Cosmos。

完成条件：

- `prepare_bundle()` 可以 wrap transformer 并写回 policy/backend。
- `policy.forward_step()` 仍然走 wrapped transformer。
- `load_trainable_state()` 能加载 strategy 导出的 state。

### Phase 4: Online GRPO Rank Split

拆 `OnlineTrainer.step()`。

完成条件：

- rank0 collect/reward/advantage。
- all ranks train。
- rank0 log/checkpoint/sync。
- single-process 行为和当前 metrics header 兼容。

### Phase 5: Checkpoint and Rollout Sync

接入 strategy-aware checkpoint。

完成条件：

- FSDP rank0 full checkpoint 正常。
- rollout workers 收到的 state 是 unwrapped policy-facing state。
- FSDP + EMA 如未完整实现必须 fail-fast。

### Phase 6: Real Runs

必须完成（FSDP2 是本 sprint 的真实交付目标）：

```bash
torchrun --nproc-per-node=2 -m vrl.scripts.train \
  --config experiment/diffusion/sd3_5/online_grpo_ocr \
  distributed.training.strategy=fsdp \
  distributed.resources.trainer.num_gpus=2
```

真实 checkpoint DoD：

- **FSDP2 2-GPU SD3 OCR 能跑至少 2 epoch**（reward 曲线不发散），或在明确不支持的配置上 fail-fast。
- rank0 metrics 不重复写。
- checkpoint 可以 resume（DTensor → rank0 full）。
- rollout worker 收到 **unwrapped** state、policy version 正常递增。
- fixed eval 使用 rank0 model state，输出不重复。
- 前置：先有一条**单卡** SD3 OCR 会涨的 reward 曲线作 baseline（多卡只是"同样的东西更快"，不该同时引入新变量）。

## 12. Acceptance Criteria

代码层：

- 所有 train scripts 不再直接决定 distributed rank/device。
- `sync_state_getter` 升级为 strategy-aware：FSDP wrap 后导出的仍是 unwrapped、policy-facing state（保持 `OnlineTrainer` 现有的"必须显式 getter、不推全量 state_dict"不变量）。
- checkpoint export/load 走 strategy。
- FSDP 不污染 rollout-facing state dict keys。
- unsupported family + unsupported strategy 有明确错误。

测试层：

```bash
python -m pytest -q tests/trainers/test_distributed_training.py
python -m pytest -q tests/trainers/test_strategy.py
python -m pytest -q tests/trainers/test_online.py
python -m pytest -q tests/trainers/test_checkpointing.py
python -m pytest -q tests/ray
python -m pytest -q tests/config/test_load_all_experiments.py
```

手动真实运行：

- single-process SD3 OCR 仍然能跑（baseline，reward 会涨）。
- 2-GPU **FSDP2** SD3 OCR 能跑或在明确未支持项 fail-fast。

## 13. References

当前代码切点：

- `/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/train.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/common/online.py`（各家族共享的 `run_online_recipe`，rank split 的真实落点）
- `/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/common/factory.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/diffusion/sd3_5/train.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/diffusion/wan_2_1/train.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/diffusion/cosmos/train.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/ar/janus_pro/train.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/ar/nextstep_1/train.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/diffusion/wan_2_1/train_dpo.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/online/trainer.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/offline/dpo.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/checkpointing.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/weight_sync.py`（`TrainableStateGetter` + `build_trainable_state_sync_getter`）
- `vrl/trainers/fsdp.py`（**已删除**，commit `a34b815`，原为 FSDP1；本 sprint 在此重建 **FSDP2** applier）
- `vrl/trainers/data/samplers.py`（`DistributedKRepeatSampler`，已存在但 dead，GRPO-group-aware 分布式 sampler，可复用）
- `docs/sprints/reading/cosmos-rl.md`（torch-native FSDP2 模板 + "no Megatron" 证据，§699/§1232）
- `docs/sprints/reading/SPRINT_cosmos_rl_scaling_learnings.md`（FSDP2/NCCL 的 Tier 1 路线）
- `/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/data/prompts.py`（prompt loader / sampler，原 `vrl/trainers/data.py` 已拆成包）
- `/home/mingfeiguo/Desktop/wm-infra/vrl/models/interfaces/runtime.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/models/diffusion/base.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/models/diffusion/sd3_5/model.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/models/diffusion/wan_2_1/model.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/models/diffusion/cosmos/predict2/model.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/models/ar/janus_pro/model.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/models/ar/nextstep_1/model.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/generation/ray/runtime.py`（`RayGenerationRuntime`，rollout 侧）
- `/home/mingfeiguo/Desktop/wm-infra/vrl/generation/ray/worker.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/generation/ray/weight_sync.py`
- 注：`vrl/distributed/ray/train/group.py`（RayTrainGroup）在第 2 节是 non-goal，当前不存在，本 sprint 不创建。

cosmos-rl 源码（`~/Desktop/cosmos-rl`，实现时直接对照 —— 已逐条核对 path:line）：

```text
# FSDP2 模板（diffusion —— 你的主路径，几乎照抄）
cosmos_rl/policy/model/diffusers/parallelize.py
  :21-24   from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy  (FSDP2)
  :33-66   parallelize(): mesh dims (dp_shard_cp / dp_replicate)，apply_fsdp(model.transformer)
  :43      assert pp_size == 1   (diffusion 不要 PP/TP)
  :69-110  apply_fsdp(): MixedPrecisionPolicy + _no_split_modules → per-block fully_shard → root fully_shard
  :88-89   _no_split_modules / _skip_layerwise_casting_patterns（block 列表 + 敏感层高精度）

# FSDP2 模板（AR / LLM trunk —— 给 janus/nextstep 用）
cosmos_rl/policy/model/gpt/parallelize.py
  :22      fully_shard / MixedPrecisionPolicy / CPUOffloadPolicy
  :82-95   mesh dims + apply_fsdp 入口
  :366+    apply_fsdp() 实现

# mesh / 并行维度
cosmos_rl/utils/parallelism.py
  :29      init_device_mesh
  :85-126  class ParallelDims（dp_shard / dp_replicate / dp_shard_with_ep；1D→HSDP 怎么建）

# meta-init → parallelize → 物化（大模型加载，别 from_pretrained().cuda()）
cosmos_rl/policy/trainer/llm_trainer/llm_trainer.py
  :92      with torch.device("meta")
  :116-119 model.parallelize_fn(...)
  :146     post_to_empty_hook  (物化分片)
  :315+    sync_all_states     (rank→rank NCCL 广播全套 state)

# DTensor 优化器 + checkpoint（本 sprint 最大触点的参照）
cosmos_rl/policy/trainer/optm/__init__.py
  :28      get_optimizer_state_dict
  :135-153 fused 模式按 p.device_mesh 分组（DTensor 不能跨 mesh fused）
  :214     get_optimizer_state_dict(...) 做 per-rank checkpoint

# NCCL 权重同步（FSDP2 全参的搭档，§10.5）
cosmos_rl/utils/pynccl.py            # 自包含 ctypes NCCL wrapper，可整段 lift
cosmos_rl/utils/parallelism_map.py   # 跨布局分片指令（DTensor metadata → per-rank slice）

# 唯一从 Megatron 借的（MoE only，dense DiT 用不上）
cosmos_rl/policy/kernel/megatron_moe/token_dispatcher.py:284  pad_routing_map
```

相关设计：

- `/home/mingfeiguo/Desktop/wm-infra/vrl/ray/resources.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/generation/ray/placement.py`
- `/home/mingfeiguo/Desktop/slime/slime/utils/arguments.py`
- `/home/mingfeiguo/Desktop/slime/slime/ray/placement_group.py`
- `/home/mingfeiguo/Desktop/miles/miles/utils/arguments.py`
