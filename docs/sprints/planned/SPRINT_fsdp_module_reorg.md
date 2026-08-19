# SPRINT: fsdp.py 归位与去重 — process-group 生命周期归 distributed.py，DTensor 物化三份合一

**日期**: 2026-08-16  **状态**: PLANNED
**触发**: 用户问 "fsdp.py 能不能更有条理？_COORDINATION_GROUP 是干什么的？能否参考 deepspeed？"
**证据来源**: 全文件逐函数普查（15 个顶层函数 × 全仓调用方 grep，.venv 除外）+
DeepSpeed 0.x 本地源码对照（`~/miniconda3/.../deepspeed/runtime/zero/`）。

---

## 0. 先回答触发问题（结论，不是计划）

**`_COORDINATION_GROUP` 是必要的，且刚修完一个驱动级崩溃。** 它来自
`abb8e4da`（2026-08-16 Xid 79 postmortem）：phase cycling 的 park/wake 窗口
在 unmap 多 GB cumem 池，各 rank 不同步；窗口内的协调消息若走 NCCL，会在
"本 rank 刚 unmap 完、慢 rank 还在 unmap" 的卡上发 GPU kernel → Xid 79/154
全卡楔死。所以协调消息必须零 GPU kernel，即专用 gloo 子组（CPU/TCP）：

```python
if backend == "nccl":
    # Collective creation: every rank reaches this line inside the same
    # init call, so the subgroup handshake cannot mismatch.
    _COORDINATION_GROUP = dist.new_group(backend="gloo")   # fsdp.py:89-92
```

它必须是模块级全局而非参数：`dist.new_group` 本身是 collective，每个 rank
必须在同一点、同一顺序调用——惰性创建会死锁，因此在 `init_process_group`
里创建一次、到处读。两个消费者都在危险窗口内：pre-park quiesce barrier
（`strategy.py:861`）和 park 成败旗（`strategy.py:1131`，必须 collective——
一个 rank 单独回滚会让各 rank shard 驻留不一致，下一个 all-gather 直接挂死
而不是报真实错误）。

**DeepSpeed 对照的结论与直觉相反，逐条：**

| 问题 | DeepSpeed 的答案 |
|---|---|
| 725 行是不是太大 | 不大。`stage3.py` 3150 行、`stage_1_and_2.py` 2523、`partition_parameters.py` 2257 |
| 模块级可变全局是不是反模式 | 不是。`FWD_MODULE_STACK`（parameter_offload.py:249）、`zero_init_context`/`top_level_context`（partition_parameters.py:324-342）、`reuse_buffers` 全是同类分布式生命周期全局 |
| 值得学的是什么 | **按关切分层**：sharding 在 `runtime/zero/`，checkpoint 格式/转换在独立顶层 `checkpoint/` 包（`zero_checkpoint.py`、`universal_checkpoint.py`）。我们的 fsdp.py 两半都装 |

**前审判例必须遵守**：`SPRINT_grab_bag_file_audit.md`（done/）已裁定
(a) fsdp.py 的"一袋函数"风格合法（vLLM weight_utils 先例，函数间无共享状态
穿线），**不要为了"看起来像类"而类化**；(b) 该审计曾误判
`shutdown_training_process_group` 零调用方并被撤销——本文件上的
调用计数分析已经失手过一次，本计划的每条删除/移动都附带调用方证列。

---

## 1. 普查事实（计划的证据底座）

- 全部 13 个公开函数**都有真实调用方**，无死代码。9 个恰好单一生产调用方，
  全部集中在 `strategy.py`（经函数内延迟 import）。**没有任何其他生产模块
  import `vrl.trainers.fsdp`。**
- 57% 的行数（260-671，共 411/725 行）是 state-dict gather/load，不是 FSDP
  包装本身。
- `distributed.py` 的 docstring **明文声明**自己不创建 process group、并指向
  fsdp.py——而 `DDPStrategy` 也在调 `init_training_process_group`
  （strategy.py:944）和 `shutdown_training_process_group`（:1035）。DDP 与
  sharding 无关：**生命周期住在一个以 sharding 算法命名的文件里，服务着一个
  不 shard 的策略**，`_COORDINATION_GROUP` 服务的 phase cycling 同样不是
  FSDP 专属。这是全文件唯一真正的归位错误。
- 真重复三处：
  1. 两个模型 gather 的 missing-key 块**逐字符一致仅差一个词**
     （:308-312 vs :362-366），DCP import 块字节一致（:288-292 vs :339-343）；
  2. DTensor→CPU 物化存在**三份**：:321-326、:372-376、以及泛型
     `_materialize_full_cpu:463`（前两处是它 tensor 分支的手抄平铺版）；
  3. 跨文件：`"checkpoint module roots mismatch"` 错误串出现 **5 次**
     （strategy.py:767,793；checkpointing.py:1184,1410,1449），其中
     strategy.py 的 `load_checkpoint_state`(:751) 与
     `load_full_checkpoint_state`(:777) **逐行一致仅差被调函数名**。

---

## 2. Sprint A — Cluster A 归位 distributed.py（纯移动，零行为变化）

### 移什么

`fsdp.py:36-103` 整块 → `vrl/trainers/distributed.py`：

- `_COORDINATION_GROUP` 全局，**顺手改名 `_CPU_COORDINATION_GROUP`**——
  "CPU" 是整个安全性质（零 GPU kernel），现在定义处看不见，而访问器已叫
  `cpu_coordination_group()`。改名只动这一个词，与移动同一个 commit。
- `cpu_coordination_group()`、`init_training_process_group()`、
  `shutdown_training_process_group()` 三个函数原样移动，docstring 不改写
  （Xid 79 的 WHY 注释逐字保留）。

### 为什么是 distributed.py 而不是新文件

`distributed.py` 已经拥有该关切的**描述半边**（`DistributedTrainingContext`、
`resolve_training_context`），docstring 自己声明"行为半边在别处"。把行为
半边搬回来是补完一次刻意的拆分，不是发明新边界。新建
`process_group.py` 会把一个 157 行的模块和一个 65 行的兄弟并排放——
制造 lean file，AGENTS.md 明令不做。

无循环 import：`distributed.py` 现在只 import torch/`vrl.utils.config`；
移入的函数用的 `torch.distributed` 全是函数内延迟 import；
`init_training_process_group(context: DistributedTrainingContext, ...)` 的
参数类型搬入后变成同模块引用。fsdp.py 对 distributed.py 的既有 import
（`fsdp.py:31`）方向不变。

### 必须同步更新的调用方（全列，防止半途迁移）

**禁止在 fsdp.py 留 re-export**——grab-bag 审计的核心教训就是
"一个符号两个 import 家园"（artifacts.py 半途迁移，8+ 消费方走 pass-through）。
一次移干净：

| 文件 | 行 | 改动 |
|---|---|---|
| `vrl/trainers/strategy.py` | 894, 938, 1033 | `from vrl.trainers.fsdp import` → `from vrl.trainers.distributed import`（init/shutdown） |
| `vrl/trainers/strategy.py` | 1124, 1144 | 同上（`cpu_coordination_group`） |
| `tests/trainers/test_ddp.py` | 198, 225 | patch 字符串 `"vrl.trainers.fsdp.…"` → `"vrl.trainers.distributed.…"` |
| `tests/trainers/test_fsdp.py` | 32, 871, 877, 902, 1061 | import 源与 `monkeypatch.setattr(fsdp_mod, …)` 的模块对象换成 distributed |
| `vrl/trainers/distributed.py` | 模块 docstring | 删除"本模块不创建 process group、见 fsdp.py"的免责段，改为声明拥有身份+生命周期两半 |

patch 目标为什么必须跟着走：strategy.py 全部是函数内延迟 import，
`unittest.mock.patch("vrl.trainers.fsdp.X")` 之所以生效是因为补丁打在
**源模块**上、延迟 import 在调用时才解析。移动后源模块变了，patch 字符串
不改会静默 patch 一个不存在的属性（`create=False` 下直接报错，算是幸运）。

搬完后 fsdp.py 剩余部分**再无模块级可变状态**（普查确认全局仅
`logger` + `_COORDINATION_GROUP` 两个），文件名与内容第一次对齐：
纯 FSDP2 包装 + sharded state-dict。

### 验收

- `grep -rn 'init_training_process_group\|shutdown_training_process_group\|cpu_coordination_group\|_COORDINATION_GROUP' vrl/ tests/` 中 `vrl.trainers.fsdp` 零命中；
- `pytest tests/trainers/ -q` 全绿（重点 test_ddp.py 的两个 patch 用例、test_fsdp.py:863-905 的 spy 用例）；
- ruff check/format 仅触碰文件。

---

## 3. Sprint B — DTensor→CPU 物化三份合一（行为等价收敛）

两个模型 gather 的平铺循环（:319-326、:370-376）改为调用共享私有 helper。
**不合并两个 gather 函数本身**——见 §5 非目标。抽取物：

```python
def _gather_named_full_cpu(
    sharded_state: Mapping[str, Any],
    names: Iterable[str],
    *,
    keep: bool,
    what: str,   # error domain: "trainable parameters" / "checkpoint-owned state"
) -> dict[str, Any]:
```

- 排序遍历（"All ranks must enter DTensor collectives in the same order"
  注释随体移动）、`full_tensor()`、非 tensor 即 `TypeError`、
  `keep` 才 `detach().cpu().clone()`——即现在两份平铺代码的并集。
- `what` 是**错误域参数**，不是 caller 身份——AGENTS.md 放置规则 1 的
  counter-test 明确允许（config key / manifest field / error domain 留自由函数）。
- missing-key 检查块（仅差一个词的那五行）一并进同一 helper 或紧邻的
  `_require_names_present(sharded_state, names, what)`，二选一以最终 diff
  小者为准。
- `_materialize_full_cpu`（:463）**保持独立**：它是递归树版本，服务 optimizer
  state 的嵌套 dict/list/tuple；把平铺路径强行塞进递归版会让两个 gather 的
  错误信息失去参数名。共享的只是"3 行 tensor 分支"，抽成 helper 后
  `_materialize_full_cpu` 的 tensor 分支也改调它，三份变一份。

**红线**：`gather_trainable_state_dict` 的 `cpu_offload=False` 大注释
（:278-285，2×1 NCCL 真机复现、world_size=1 测不出）与
`gather_checkpoint_state_dict` 的 `ignore_frozen_params` 翻转逻辑是**不同的
定理**，各自的 DCP 调用和注释一个字不动。合并的只有定理下游的机械搬运段。

### 验收

- `pytest tests/trainers/test_fsdp.py tests/trainers/test_fsdp_gather_distributed.py tests/trainers/test_fsdp_fp32_master.py -q` 全绿；
- 有 GPU 环境时补跑一次 `test_fsdp_gather_distributed.py` 的 NCCL lane
  （该文件存在的意义就是抓 CPU 测试抓不到的 gather 分歧）。

---

## 4. Sprint C — strategy.py 孪生 wrapper 合并（跨文件重复的最小可做子集）

`load_checkpoint_state`(:751-775) 与 `load_full_checkpoint_state`(:777-801)
逐行一致仅差被调函数。合并为一个私有实现，公开签名不变：

```python
def _load_module_states(self, bundle, state, *, strict, load_one) -> None:
    ...  # 现有 root-key 校验 + 逐模块循环，load_one 为被调的 fsdp 函数
```

两个公开方法各自变成 3 行委托。root-key 校验块（missing/extra 五行）随之
从 2 份变 1 份。

**checkpointing.py 里那三份同串校验（:1184,:1410,:1449）本 sprint 不动**：
它们操作的是非 sharded 的 `state_dict()`/`load_state_dict()` 路径，与
strategy.py 的 bundle 层各有 owner；跨两个 67KB/47KB 文件抽公共校验函数
需要先回答"归谁"（放置规则 1），证据不足时宁可留重复——记入 §5。

### 验收

`pytest tests/trainers/ -q` 全绿；两个公开方法的异常类型与消息逐字不变
（grab-bag 审计的异常类型漂移红线）。

---

## 5. 非目标（每条附理由，防止下次审计重提）

1. **不类化、不引入 Manager/Handler**。前审判例 + vLLM `weight_utils` 先例；
   搬走 Cluster A 后文件真正零共享状态，一袋纯函数是正确形态。
2. **不建 `fsdp_state_dict.py`**。checkpoint 关切已经被按"sharded /
   非 sharded"切在 fsdp.py 与 checkpointing.py 两处（同名函数、同错误串）；
   再加第三个文件是把错误的切分维度做深。正确的合并方向是把两半按关切
   收拢（DeepSpeed 的 `checkpoint/` 包模式），但那是动 67KB checkpointing.py
   的独立 sprint，需要自己的证据普查。
3. **不合并两个模型 gather 函数**。`cpu_offload`/`ignore_frozen_params`/
   `rank0_only` 的差异各自背着真机事故注释——同文本不同定理，合并删掉的是
   论证不是重复（AGENTS.md 放置规则 3 counter-test）。
4. **`normalize_fsdp_parameter_dtype` 不搬**。它确实不碰 FSDP API，但唯一
   生产调用方是 `FSDPStrategy.prepare_model`（strategy.py:635），语义上是
   "shard 前置准备"；搬去 `vrl/models/` 会把单消费方工具跨包提升为共享 API
   （放置规则的 god-object/过度提升警告）。
5. **`_distribute_fp32_master_optimizer_state` 不搬去 optimizer.py**。
   它是 `FP32MasterWeightOptimizer` 序列化格式的 **DTensor 感知半边**；
   optimizer.py 目前策略无关，塞入 DTensor/mesh 知识是反向污染。格式定义
   （optimizer.py:247-330）与分发实现的这道缝是刻意的，加一行注释指明即可。
6. **`unwrap_module`/`iter_blocks` 不改私有**。有内部生产调用方
   （`apply_fsdp:241-242`）+ 直接单测，非死代码；public→private 纯属改名churn。
7. **checkpointing.py 内部的三份 roots-mismatch 校验不动**（§4 已述）。

---

## 6. 执行顺序与门槛

A → B → C，各自独立 commit，彼此可单独回滚。每个 commit 前：

1. `ruff check --fix <touched> && ruff format <touched>`，之后复查 diff 无无关 churn；
2. `pytest tests/trainers/ -q` 全绿（当前基线含 2 个既有 architecture 失败在
   `tests/architecture/`，与本计划无关，已由 stash 复跑证明）;
3. Sprint A 额外跑一次全仓 grep 验收（§2）；
4. 不 push；三个 commit 攒齐后一起给用户过目。

**硬前提**：动手前 `git status` 检查——本仓有后台进程在写 anima 相关文件
（2026-08-16 两次观测到工作树自行增长），凡 stash/rebase 先清点。

## 7. 参考

- `vrl/trainers/fsdp.py:36-103`（Cluster A）、`:260-671`（state-dict 块）
- `vrl/trainers/distributed.py`（描述半边 + 免责 docstring）
- `vrl/trainers/strategy.py:751-801,861,894-1035,1119-1158`
- `vrl/trainers/checkpointing.py:1372-1499`（非 sharded 孪生）
- `docs/sprints/done/SPRINT_grab_bag_file_audit.md`（前审判例：风格合法 + 误判撤销记录）
- `abb8e4da` — Xid 79 修复，`_COORDINATION_GROUP` 的出生 commit
- DeepSpeed 本地源：`runtime/zero/`（大文件与全局先例）、`checkpoint/`（关切切分先例）
