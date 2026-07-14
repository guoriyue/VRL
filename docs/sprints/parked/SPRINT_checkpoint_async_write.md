# SPRINT: Checkpoint async write —— 把 checkpoint 落盘移出训练关键路径 + pinned non_blocking gather

状态：**PARKED（2026-06-29）。** PROFILED（含 REAL save，见下）：门未过——默认 `save_freq=50` 下单次 ~6-7s GPU-idle 摊成 **≤0.16% 训练 wall**。**Un-park 触发：改用频繁 checkpoint（`save_freq ≲ 4-5`，长跑 crash recovery）→ 那时一个 `asyncio.to_thread` 把写盘丢后台才值得（~1h 活）。** 否则不做。**真正大杠杆在别处**：reward-stage 14% bubble（[[SPRINT_video_rollout_stage_overlap]]）、sbs=4（已落，~33% boundary idle）——都比本 sprint 大 1-2 个数量级。 性质：**EXACT（无损）throughput 杠杆。** 现在每 `save_freq` 个 epoch，训练在 `trainer.step` 返回后**同步**做「GPU→CPU gather + `torch.save`（含整个优化器 state）」，期间默认 strict-on-policy 单卡的整条流水线（rollout+train 共用一张卡）完全空转。本 sprint 把它拆成「pinned non_blocking gather（缩短 GPU 空转）+ 后台 writer 队列（藏掉写盘）」。

> ## ⚠️ PROFILE 实测结论（2026-06-29，RTX 5090，VRL/.venv）
> **第一轮是合成 primitive 估算（composed），不是真 save；第二轮跑了 REAL save 修正。** 两个探针：`scratchpad/io_stall_profile.py`（primitive）+ `scratchpad/real_ckpt_save.py`（**真 2B 模型 + 真 `torch.optim.AdamW` + 真 `opt.step()` 填 `exp_avg/exp_avg_sq` + 真 payload dict（mirror checkpointing.py:133-144）+ 真 `torch.save`**）。
> - **REAL 单次 save（真 model.state_dict() + 真 optimizer.state_dict() → torch.save+fsync→NVMe）**：
>   - bf16 params + AdamW（bf16 state，12.0GB payload）→ **6.7s**（1.78 GB/s）
>   - fp32 params + AdamW（fp32 state，24.0GB payload）→ **7.8s**（3.07 GB/s）
>   - → 真 `full-param 2B + AdamW8bit`（4GB 参数 + 4GB int8 state = 8GB payload）插值 ≈ **~5s 写盘 + ~1-2s GPU gather ≈ 6-7s GPU-idle/次**。
> - **修正前一轮 composed 高估**：composed 用了保守 1.2 GB/s、把 fp32 估成 24.8s；**真 NVMe 跑到 1.8-3.1 GB/s，真 fp32 24GB 只 7.8s。**（注：盘吞吐随冷/热缓存 + 与 live run 抢 I/O 有方差,真值落 single-digit 秒,不是 tens of seconds——那句源码注释偏保守。）
> - **摊到训练 wall**：`save_freq=50`、`total_epochs=300` → ~7 次 save。epoch wall ≥ 89s（240p_33f sbs=4 rollout,含训练更大）→ 7×~6s / (300×89s) = stall **≤ 0.16%** 训练 wall。**远低于 2-3% 门（比 composed 还低）。**
> - **breakeven**：单次 ~6s，要占 2% 训练 wall 需 `save_freq ≲ 4-5`。save_freq=50 时 ~0.16%。
> - **P1 pinned-gather 几乎无意义**：gather（~1-2s）远小于写盘主导项；且 gather 探针被 host-pinned-alloc（`cudaHostAlloc`）开销污染、pinned-vs-naive 没测干净——但写盘主导,不影响结论。
> - **仍存的 caveat（诚实）**：① REAL save 跑在 CPU（torch.save 成本在 CPU/磁盘,这步真实；但真跑里 model 在 GPU,多一段 D2H gather ~1-2s,本测未含,已单列）；② 未用真 cosmos predict2 + bnb AdamW8bit 在 GPU 端跑（本 venv 无 bnb）;③ epoch wall 用 89s rollout 下界,真值更大 → 占比只会更小。
> - **净结论**：**单次 save 是真 GPU-idle（8bit 真配置 ~6-7s），但默认 save_freq=50 下摊成 ≤0.16%,throughput ROI 不够 → 默认不做。** 只在 ① 频繁 checkpoint（每 ≤5 epoch）② 在意那个每 50ep 一次的 ~6-7s 顿挫本身（长跑/利用率平滑）才值。真要做 P2 后台写盘把关键路径 ~5s→~0（一个 `asyncio.to_thread`),trivial 但低优先。

> 证据：`vrl/scripts/common/online.py:989-996`（epoch 循环内 `run.save_checkpoint(...)` 内联、**无 await**；每 `save_freq` 一次 + 末尾无条件一次）、`vrl/scripts/common/online.py:533`（`def save_checkpoint(...) -> None` 是**非 async** 方法）、`vrl/scripts/common/online.py:567-568`（注释自承 rank0 写盘 "can take tens of seconds for a full-param payload"）、`vrl/trainers/checkpointing.py:126,144`（先 `export_trainable` D2H gather,后 `torch.save(payload, ...)`）、`vrl/trainers/checkpointing.py:274-284`（single_process gather = `to_cpu(state_dict())` per module）、`vrl/trainers/weight_sync.py:214-219`（`to_cpu` = 裸 `leaf.detach().cpu()`,**无 pinned/non_blocking**）、`vrl/trainers/online/trainer.py:1505-1506`（payload 内嵌**整个** `optimizer.state_dict()`,且这些 CUDA 张量**没被预先 to_cpu**,pickling 时才拷下来）、`vrl/trainers/online/trainer.py:143,150`（优化器是 AdamW / AdamW8bit,每参 2 个 state 张量）、`vrl/trainers/core/types.py:335`（`save_freq=50` 默认）、`vrl/trainers/strategy.py:112-113`（`SingleProcessStrategy.barrier()` 是 no-op,post-save barrier 不加成本）、`vrl/trainers/online/trainer.py:662-668`（step 内 rollout collect/offload/sync 时分共用单卡 → save 在 step 后跑时 GPU 全空）。
> Current primitive correction (2026-07-13): pinned asynchronous D2H remains in `vrl/generation/execution/worker.py` and `vrl/generation/diffusion/pipeline.py`; bounded background-task ownership remains in the continuous orchestration package. The former physical-stage `pipeline_runner.py` example was deleted with its test-only seam. Current blocking-call offload examples live in `vrl/generation/ray/launcher.py`, `vrl/generation/ray/runtime.py`, and `vrl/rollouts/orchestration/continuous/owner.py`; none is a shared physical-stage runtime contract.
> 相关：[[feedback_mfu_bound_north_star]]（probe 先行）、[[project_real_run_profiling.md]]（kernel-union 量 idle）、[[project_fullparam_8bit_adam]]（full-param 2B 优化器 state 是写盘大头;AdamW8bit 把它缩 ~4x）、[[project_two_level_async]]（默认 strict+colocated 单卡 = 串行 GPU 时分,save 期间无重叠）、[[SPRINT_media_artifact_async_write]]（姊妹 sprint,同源 pinned-copy + to_thread 队列,但 drain/失败语义不同）。

## 0. 一句话 + 为什么这是真 stall

默认模式（strict-on-policy + colocated 单卡）下 rollout 和 train 本就时分共用一张卡，而 checkpoint save 在 `await trainer.step(...)` 返回**之后**内联同步执行（没有 await、没有线程、没有 Ray offload）。所以序列化 + 写盘的整段时间里 **GPU 100% 空转**——不是「写盘在后台慢慢来」，而是「整条训练流水线停下来等磁盘」。代码注释自己都写了 full-param payload 写盘 "tens of seconds"。

**为什么异步化在串行单卡也是真收益**：下一个 epoch 的 GPU 工作**不依赖** `checkpoint.pt` 落盘完成——参数还在 GPU 上，下一步 forward 不读这个文件。所以把「gather 到 host → 后台慢慢写」之后让主循环立刻开跑下一个 epoch，GPU 空转就被填上。这与「serial 模式异步化只是挪 stall」的陷阱不同：陷阱发生在**写完立刻 join**；这里关键就是**不 join**（下游无依赖）。

## 1. 现状（坐实，全同步内联）

```python
# vrl/scripts/common/online.py:989-996  —— epoch 循环内,无 await,无 offload
if trainer_config.save_freq > 0 and (epoch + 1) % trainer_config.save_freq == 0:
    run.save_checkpoint(output_dir / f"checkpoint-{epoch + 1}", epoch=epoch + 1)
...
run.save_checkpoint(output_dir / "checkpoint-final", epoch=trainer_config.total_epochs)
```

`save_checkpoint`（online.py:533）是**非 async** 方法 → `save_training_checkpoint`：先 gather 再写盘，两段都在关键路径：

```python
# vrl/trainers/checkpointing.py:126-144
trainable_modules = export_trainable(bundle)         # ① GPU→CPU gather（卡 GPU）
trainer_state = trainer.state_dict()                 # 内嵌整个 optimizer.state_dict()
payload = {..., "trainer": trainer_state, "model": {"trainable_modules": trainable_modules}, ...}
torch.save(payload, path / TRAINING_CHECKPOINT_NAME) # ② 同步写盘（卡磁盘）
```

gather 用的是裸 `.cpu()`，无 pinned/non_blocking：

```python
# vrl/trainers/weight_sync.py:214-219
def to_cpu(value):
    return map_tensor_tree(value, lambda leaf: leaf.detach().cpu(), ...)
```

而 payload 里**优化器 state 没被预先 to_cpu**——只有 `trainable_modules` 走了 to_cpu，trainer state 是原样塞进去，AdamW 的 `exp_avg`/`exp_avg_sq` 还在 GPU 上，`torch.save` pickling 时才拷下来：

```python
# vrl/trainers/online/trainer.py:1505-1506
if self._optimizer is not None:
    d["optimizer"] = self._optimizer.state_dict()   # full-param 时这是写盘大头
```

## 2. 把成本拆三段（决定异步化能省什么）

| 成本段 | 真正卡什么 | 怎么治 |
|---|---|---|
| (a) trainable params D2H gather（checkpointing.py:283 裸 `.cpu()`） | 卡 GPU | pinned `non_blocking`（worker._to_cpu）→ **无条件缩短 GPU 空转** |
| (b) optimizer state D2H（pickling 时隐式,trainer.py:1505） | 卡 GPU（且可能在 host RAM 全量物化,见 §7） | 显式先 pinned gather 到 host,再交后台 |
| (c) `torch.save` 写盘 + `save_pretrained` adapter（checkpointing.py:144,176） | 卡磁盘 | **后台 writer 队列藏掉**（下游无依赖,但退出/weight-sync 前必须 drain） |

**规模**（[[project_fullparam_8bit_adam]]，估算非实测）：full-param 2B fp32 ≈ 8GB 参数 + 16GB 优化器 ≈ 24GB/次；AdamW8bit 优化器缩 ~4x；bf16 参数再减半。LoRA 则只有几十～几百 MB，stall 亚秒～个位秒。**所以这条 sprint 的 ROI 强烈依赖 full-param + 高 `save_freq`；LoRA run 基本不值得做。**

## 3. 北极星纪律：KILL-RISK profile 门（必须先过）

按 [[feedback_mfu_bound_north_star]]：先证明「save stall × 频率」吃掉训练 wall 的可观比例，再动手。

**门 1 —— 量单次 save 的两段 wall（代码现在没有计时,profile 第一步是加上）。**
- 在 `save_training_checkpoint` 外层加 `time.perf_counter()`，分别量 ① `export_trainable` gather、② `torch.save` 写盘（+ optimizer pickling）两段耗时。
- 跑一个 full-param run（如 `online_grpo_fullparam_8bit_240p.yaml`，PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True）实测，而不是信 "tens of seconds" 这句注释。

**门 2 —— 量 stall 占训练 wall 的比例。**
- `单次 save wall × (total_epochs / save_freq) / 总训练 wall`。
- **过门判据**：`≳ 2-3%` 才值得建队列；若 `save_freq` 设得很稀（如 9999 禁用 / 100）且单次只几秒 → 占比微乎其微,关掉本 sprint。
- 用 [[project_real_run_profiling.md]] 的 kernel-union 法确认 save 窗口内 GPU 确实空转（坐实它真是 idle，不是被别的东西填着）。

**门 3 —— 拆 gather vs write 的占比（决定先做哪段）。**
- 若门 1 显示 **gather 段（a+b）占大头** → P1 的 pinned non_blocking 改进就是主菜，可能单独就够（不必建队列）。
- 若 **write 段（c）占大头** → 后台队列是主菜。
- 两段都大 → 两个 phase 都做。

## 4. 设计（复用已有原语，别另造）

**① pinned non_blocking gather（P1，无条件赢，所有模式成立）。** 把 checkpoint gather 从裸 `.cpu()` 换成已有的 pinned 版本，并**把优化器 state 也显式 pinned gather 到 host**（别留给 torch.save 隐式拷）：

```python
# 现成可抄：vrl/generation/execution/worker.py:486-521
host = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True)
host.copy_(tensor, non_blocking=True)   # 全部 leaf 排队
torch.cuda.synchronize()                # 末尾一次,不是 per-tensor
# 进阶（专用 stream + event,可与后续 compute 重叠）：pipeline.py:156-169 _move_tree_to_cpu_async
```
落点：`vrl/trainers/weight_sync.py:214-219` 的 `to_cpu`（gather 和 weight-sync 共用它，一处改两处受益）+ 让 `trainer.state_dict()` 在进 payload 前对 optimizer state 做同样的 pinned gather。

**② 后台 writer 队列（P2）。** mirror `ContinuousRolloutProducer` 的后台 `asyncio.Task`（producer.py:77-90），实际写用 `asyncio.to_thread`：

```python
# save 路径改成：pinned gather → host payload 入队 → 主循环立刻返回开跑下个 epoch
await asyncio.to_thread(torch.save, host_payload, path / TRAINING_CHECKPOINT_NAME)
# asyncio.to_thread 现成样板：vrl/generation/ray/weight_sync.py:59
```
**入队的是已经在 host 的 payload**（pinned gather 之后），绝不入队 CUDA 张量。`save_pretrained` adapter（checkpointing.py:176，非 fsdp 才走）同样丢后台。

**③ 有界 + 单飞**：checkpoint payload 巨大（GB 级），队列深度设 1-2 即可（mirror queue.py 的 byte 预算思路）；若上一个 checkpoint 还没写完下一个就来了，**同步等上一个**（背压），别让 host RAM 堆两份 24GB。

## 5. 阶段

- **P0 — KILL-RISK profile 门（§3）**：加两段计时，full-param run 实测单次 save wall + 占比 + gather/write 拆分。**不过门就停。**
- **P1 — pinned non_blocking gather（无条件赢，独立可落）**：`weight_sync.to_cpu` + optimizer state 改 pinned `non_blocking`。这步即使不建队列也成立，且同时让 weight-sync 路径受益（它现在也是裸 `.cpu()`）。先落、单独验。
- **P2 — 后台 writer 队列**：`save_training_checkpoint` 改成「pinned gather host payload → enqueue → 返回」；后台 task 用 `asyncio.to_thread(torch.save / save_pretrained)` drain。
- **P3 — drain/失败/回归**：接上 §6 的硬 drain 点 + 丢写 raise；用 P0 同一套计时确认主路径 save wall 掉到 ≈ gather 时间、训练 wall 下降、save 窗口 GPU busy% 回升。

## 6. Drain 与失败处理（比 media sprint 严格）

- **两个硬 drain 点必须 join**：
  1. **进程退出 / 训练结束前**——否则丢 checkpoint。末尾那个无条件 `checkpoint-final`（online.py:993-996）尤其要确保后台写完才退。
  2. **weight-sync 之前**——若后台还在读旧 state dict 做 D2H，而新一轮要改写参数，会读到不一致数据。weight-sync barrier（rollout pause→drain→sync→resume）处必须先 `drain_inflight()`（producer.py:134-143 语义）排空写队列。
- **丢写必须 raise，不能吞**：后台 `asyncio.gather(..., return_exceptions=True)` 收集后**检查异常并在主循环抛出**——丢掉的 checkpoint = 丢训练状态，必须让训练显式失败/告警。这与 [[SPRINT_media_artifact_async_write]] 的「可降级 + 告警」相反，**不要共用一个队列实例或一套失败策略**。
- **resume 正确性**：异步写期间若进程被杀，可能留下半写的 `checkpoint.pt`。写「临时文件 + 原子 rename」保证 resume 只看到完整 checkpoint（`trainer.resume_from` 路径不能读到 torn write）。

## 7. Non-goals / 不确定

- **LoRA run 不做**：payload 小，stall 亚秒，ROI 不够（门 2 会直接刷掉）。本 sprint 面向 full-param。
- **不改 checkpoint 格式 / 不引入 sharded / async-checkpoint 库**（如 torch DCP async save）：先用仓库已有的 to_thread + pinned 原语把同步 stall 削掉；引第三方 async-checkpoint 是更大的另一个决定，超本 sprint 范围。
- **FSDP / 多卡路径**：本 sprint 主要坐实 single_process 单卡。FSDP 的 gather 是 collective all-gather（strategy.py），其输出张量 gather 时是否仍在 GPU（决定能否吃 pinned 收益）需读 `get_model_state_dict` 内部——**未打开,标注待确认**。
- **不确定（profile 才能定）**：单次 save 真实秒数（"tens of seconds" 是注释非实测）；`torch.save` pickling 优化器 CUDA 张量时是否在 host RAM 先全量物化（可能瞬时双倍 host-RAM 峰值，影响队列深度设计）；后台 writer drain task 在 trainer 的 `await asyncio.sleep(0)` 让步点能否稳定被调度而不饿死（未完整 trace trainer 循环）。
